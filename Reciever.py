#!/usr/bin/env python3
"""
Phase 2 Receiver - phase2_receiver.py

Listens on TCP, receives FILE_INIT, METADATA, and PART packets from Phase 1,
acknowledges each packet, reassembles Base64 chunks into original JPEG (lossless),
verifies file checksum, saves to Phase 2's input directories.
"""

import argparse
import sqlite3
import socket
import threading
import time
import hashlib
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from datetime import datetime, timezone
import fcntl

# Constants
MAX_PACKET_SIZE = 2048
DELIMITER = "---DATA---"
END_DELIMITER = "---END---"

class FileTransfer(NamedTuple):
    file_id: str
    filename: str
    file_size: int
    file_md5: str
    total_parts: int
    status: str

class PacketPart(NamedTuple):
    file_id: str
    part_index: int
    chunk_md5: str
    status: str

class Phase2Receiver:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.db_lock = threading.Lock()
        self.json_lock = threading.Lock()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize directories
        self.setup_directories()
        
        # Initialize database
        self.init_database()
        
        # Load processed images tracking
        self.load_processed_images()
        
        self.logger.info(f"Phase 2 Receiver initialized")
        self.logger.info(f"  Listen address: {self.args.listen_host}:{self.args.listen_port}")
        self.logger.info(f"  Input directory: {self.args.input_dir}")
        self.logger.info(f"  Receive queue: {self.args.receive_queue}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.args.log_level.upper())
        
        handlers = [logging.StreamHandler()]
        if self.args.log_file:
            handlers.append(logging.FileHandler(self.args.log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup_directories(self):
        """Ensure required directories exist."""
        Path(self.args.input_dir).mkdir(parents=True, exist_ok=True)
        Path(self.args.input_dir, 'images').mkdir(parents=True, exist_ok=True)
        Path(self.args.input_dir, 'metadata').mkdir(parents=True, exist_ok=True)
        Path(self.args.receive_queue).mkdir(parents=True, exist_ok=True)
        
        # Ensure processed JSON directory exists
        Path(self.args.processed_json).parent.mkdir(parents=True, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database for receive state."""
        self.db_path = Path(self.args.db_path)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_md5 TEXT NOT NULL,
                    total_parts INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'receiving',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    saved_path TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS parts (
                    file_id TEXT NOT NULL,
                    part_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'missing',
                    chunk_md5 TEXT NOT NULL,
                    received_at TIMESTAMP,
                    PRIMARY KEY (file_id, part_index),
                    FOREIGN KEY (file_id) REFERENCES files (file_id)
                )
            ''')
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_files_status ON files (status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_parts_status ON parts (status)')
    
    def load_processed_images(self):
        """Load existing processed images JSON."""
        self.processed_images = {'processed_images': []}
        
        try:
            if Path(self.args.processed_json).exists():
                with open(self.args.processed_json, 'r') as f:
                    self.processed_images = json.load(f)
                    
                # Ensure proper structure
                if not isinstance(self.processed_images, dict):
                    self.processed_images = {'processed_images': []}
                if 'processed_images' not in self.processed_images:
                    self.processed_images['processed_images'] = []
                    
                self.logger.info(f"Loaded {len(self.processed_images['processed_images'])} processed image records")
        except Exception as e:
            self.logger.error(f"Error loading processed images JSON: {e}")
            self.processed_images = {'processed_images': []}
    
    def update_processed_images(self, file_id: str, filename: str, file_md5: str, saved_path: str, metadata_path: str):
        """Update processed images JSON with atomic write."""
        try:
            with self.json_lock:
                # Add new entry
                new_entry = {
                    'file_id': file_id,
                    'filename': filename,
                    'file_md5': file_md5,
                    'received_at': datetime.now(timezone.utc).isoformat(),
                    'saved_path': saved_path,
                    'metadata_path': metadata_path
                }
                
                self.processed_images['processed_images'].append(new_entry)
                self.processed_images['last_updated'] = datetime.now(timezone.utc).isoformat()
                
                # Write to temporary file then rename (atomic)
                temp_path = Path(self.args.processed_json).with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.processed_images, f, indent=2)
                
                temp_path.rename(self.args.processed_json)
                
                self.logger.info(f"Updated processed images JSON with {filename}")
                
        except Exception as e:
            self.logger.error(f"Error updating processed images JSON: {e}")
    
    def parse_packet(self, packet_data: str) -> Dict:
        """Parse incoming packet and extract headers and data."""
        try:
            # Split packet into lines
            lines = packet_data.strip().split('\n')
            
            headers = {}
            data_started = False
            data_lines = []
            
            for line in lines:
                if line.strip() == DELIMITER:
                    data_started = True
                    continue
                elif line.strip() == END_DELIMITER:
                    break
                elif data_started:
                    data_lines.append(line)
                elif ':' in line and not data_started:
                    key, value = line.split(':', 1)
                    headers[key.strip().upper()] = value.strip()
            
            return {
                'headers': headers,
                'data': '\n'.join(data_lines) if data_lines else None
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing packet: {e}")
            return {'headers': {}, 'data': None}
    
    def create_ack_packet(self, packet_type: str, file_id: str, status: str = 'OK', **kwargs) -> str:
        """Create ACK packet response."""
        lines = [
            f"TYPE: ACK",
            f"FILE_ID: {file_id}",
            f"STATUS: {status}",
            f"RECEIVED_AT: {datetime.now(timezone.utc).isoformat()}"
        ]
        
        for key, value in kwargs.items():
            lines.append(f"{key.upper()}: {value}")
        
        return '\n'.join(lines) + '\n'
    
    def create_file_complete_packet(self, file_id: str, file_md5: str, saved_path: str, status: str = 'OK') -> str:
        """Create FILE_COMPLETE packet."""
        lines = [
            f"TYPE: FILE_COMPLETE",
            f"FILE_ID: {file_id}",
            f"FILE_MD5: {file_md5}",
            f"STATUS: {status}",
            f"SAVED_PATH: {saved_path}",
            f"COMPLETED_AT: {datetime.now(timezone.utc).isoformat()}"
        ]
        
        return '\n'.join(lines) + '\n'
    
    def send_response(self, conn: socket.socket, response: str):
        """Send response packet to client."""
        try:
            response_bytes = response.encode('utf-8')
            conn.sendall(len(response_bytes).to_bytes(4, 'big'))
            conn.sendall(response_bytes)
        except Exception as e:
            self.logger.error(f"Error sending response: {e}")
    
    def handle_file_init(self, conn: socket.socket, parsed: Dict) -> bool:
        """Handle FILE_INIT packet."""
        try:
            headers = parsed['headers']
            
            file_id = headers.get('FILE_ID')
            filename = headers.get('FILE')
            file_size = int(headers.get('FILE_SIZE', 0))
            file_md5 = headers.get('FILE_MD5')
            total_parts = int(headers.get('TOTAL_PARTS', 0))
            
            if not all([file_id, filename, file_md5]):
                self.send_response(conn, self.create_ack_packet('FILE_INIT', file_id or 'unknown', 'ERROR', error='missing_required_fields'))
                return False
            
            # Create working directory
            work_dir = Path(self.args.receive_queue) / file_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Store file info in database
            with self.db_lock:
                with sqlite3.connect(self.db_path) as db_conn:
                    # Insert or update file record
                    db_conn.execute('''
                        INSERT OR REPLACE INTO files 
                        (file_id, filename, file_size, file_md5, total_parts, status, last_update)
                        VALUES (?, ?, ?, ?, ?, 'receiving', CURRENT_TIMESTAMP)
                    ''', (file_id, filename, file_size, file_md5, total_parts))
                    
                    # Pre-create part records
                    for part_idx in range(total_parts):
                        db_conn.execute('''
                            INSERT OR IGNORE INTO parts (file_id, part_index, chunk_md5)
                            VALUES (?, ?, '')
                        ''', (file_id, part_idx))
            
            self.logger.info(f"Initialized file transfer: {filename} ({file_size} bytes, {total_parts} parts)")
            
            # Send ACK
            ack = self.create_ack_packet('FILE_INIT', file_id, 'OK')
            self.send_response(conn, ack)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling FILE_INIT: {e}")
            file_id = parsed['headers'].get('FILE_ID', 'unknown')
            self.send_response(conn, self.create_ack_packet('FILE_INIT', file_id, 'ERROR', error=str(e)))
            return False
    
    def handle_metadata(self, conn: socket.socket, parsed: Dict) -> bool:
        """Handle METADATA packet."""
        try:
            headers = parsed['headers']
            
            file_id = headers.get('FILE_ID')
            filename = headers.get('FILENAME')
            metadata_content = parsed.get('data', '')
            
            if not file_id:
                self.send_response(conn, self.create_ack_packet('METADATA', 'unknown', 'ERROR', error='missing_file_id'))
                return False
            
            # Save metadata to working directory
            work_dir = Path(self.args.receive_queue) / file_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = work_dir / 'metadata.txt'
            
            # Handle multi-part metadata
            if headers.get('TYPE') == 'METADATA_PART':
                part_info = headers.get('PART', '1/1')
                part_num, total_parts = map(int, part_info.split('/'))
                
                # Save part
                part_file = work_dir / f'metadata_part_{part_num:03d}.txt'
                with open(part_file, 'w', encoding='utf-8') as f:
                    f.write(metadata_content)
                
                # Check if all parts received
                parts_received = len(list(work_dir.glob('metadata_part_*.txt')))
                if parts_received == total_parts:
                    # Combine parts
                    combined_content = []
                    for i in range(1, total_parts + 1):
                        part_file = work_dir / f'metadata_part_{i:03d}.txt'
                        if part_file.exists():
                            with open(part_file, 'r', encoding='utf-8') as f:
                                combined_content.append(f.read())
                    
                    # Write combined metadata
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        f.write(''.join(combined_content))
                    
                    # Clean up part files
                    for i in range(1, total_parts + 1):
                        part_file = work_dir / f'metadata_part_{i:03d}.txt'
                        if part_file.exists():
                            part_file.unlink()
            else:
                # Single metadata packet
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    f.write(metadata_content)
            
            self.logger.debug(f"Saved metadata for {file_id}")
            
            # Send ACK
            ack = self.create_ack_packet('METADATA', file_id, 'OK')
            self.send_response(conn, ack)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling METADATA: {e}")
            file_id = parsed['headers'].get('FILE_ID', 'unknown')
            self.send_response(conn, self.create_ack_packet('METADATA', file_id, 'ERROR', error=str(e)))
            return False
    
    def handle_part(self, conn: socket.socket, parsed: Dict) -> bool:
        """Handle PART packet."""
        try:
            headers = parsed['headers']
            
            file_id = headers.get('FILE_ID')
            part_info = headers.get('PART', '1/1')
            chunk_len = int(headers.get('CHUNK_LEN', 0))
            chunk_md5 = headers.get('CHUNK_MD5')
            base64_data = parsed.get('data', '')
            
            if not all([file_id, part_info, chunk_md5, base64_data]):
                self.send_response(conn, self.create_ack_packet('PART', file_id or 'unknown', 'ERROR', error='missing_required_fields'))
                return False
            
            # Parse part index
            part_num, total_parts = map(int, part_info.split('/'))
            part_index = part_num - 1  # Convert to 0-based index
            
            # Decode base64 data
            try:
                chunk_data = base64.b64decode(base64_data.replace('\n', ''))
            except Exception as e:
                self.logger.error(f"Error decoding base64 for part {part_num}: {e}")
                self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error='base64_decode_error'))
                return False
            
            # Verify chunk length
            if len(chunk_data) != chunk_len:
                self.logger.error(f"Chunk length mismatch for part {part_num}: expected {chunk_len}, got {len(chunk_data)}")
                self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error='chunk_length_mismatch'))
                return False
            
            # Verify chunk MD5
            actual_md5 = hashlib.md5(chunk_data).hexdigest()
            if actual_md5 != chunk_md5:
                self.logger.error(f"Chunk MD5 mismatch for part {part_num}: expected {chunk_md5}, got {actual_md5}")
                self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error='checksum_mismatch'))
                return False
            
            # Save chunk to working directory
            work_dir = Path(self.args.receive_queue) / file_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_file = work_dir / f'part_{part_index:05d}.bin'
            with open(chunk_file, 'wb') as f:
                f.write(chunk_data)
            
            # Update database
            with self.db_lock:
                with sqlite3.connect(self.db_path) as db_conn:
                    db_conn.execute('''
                        UPDATE parts 
                        SET status = 'received', chunk_md5 = ?, received_at = CURRENT_TIMESTAMP
                        WHERE file_id = ? AND part_index = ?
                    ''', (chunk_md5, file_id, part_index))
            
            self.logger.debug(f"Received part {part_num}/{total_parts} for {file_id}")
            
            # Send ACK
            ack = self.create_ack_packet('PART', file_id, 'OK', part=part_info)
            self.send_response(conn, ack)
            
            # Check if all parts received
            with sqlite3.connect(self.db_path) as db_conn:
                cursor = db_conn.execute('''
                    SELECT COUNT(*) FROM parts 
                    WHERE file_id = ? AND status = 'received'
                ''', (file_id,))
                received_count = cursor.fetchone()[0]
                
                cursor = db_conn.execute('''
                    SELECT total_parts FROM files WHERE file_id = ?
                ''', (file_id,))
                total_parts_db = cursor.fetchone()[0]
            
            if received_count == total_parts_db:
                # All parts received, assemble file
                self.assemble_and_save_file(conn, file_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling PART: {e}")
            file_id = parsed['headers'].get('FILE_ID', 'unknown')
            part_info = parsed['headers'].get('PART', 'unknown')
            self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error=str(e)))
            return False
    
    def assemble_and_save_file(self, conn: socket.socket, file_id: str):
        """Assemble all parts and save final file."""
        try:
            # Get file info from database
            with sqlite3.connect(self.db_path) as db_conn:
                cursor = db_conn.execute('''
                    SELECT filename, file_size, file_md5, total_parts
                    FROM files WHERE file_id = ?
                ''', (file_id,))
                file_info = cursor.fetchone()
                
                if not file_info:
                    raise ValueError(f"File info not found for {file_id}")
                
                filename, file_size, expected_md5, total_parts = file_info
            
            work_dir = Path(self.args.receive_queue) / file_id
            
            # Assemble file from parts
            assembled_data = bytearray()
            
            for part_index in range(total_parts):
                part_file = work_dir / f'part_{part_index:05d}.bin'
                
                if not part_file.exists():
                    raise FileNotFoundError(f"Missing part file: {part_file}")
                
                with open(part_file, 'rb') as f:
                    part_data = f.read()
                    assembled_data.extend(part_data)
            
            # Verify assembled file size
            if len(assembled_data) != file_size:
                raise ValueError(f"File size mismatch: expected {file_size}, got {len(assembled_data)}")
            
            # Verify assembled file MD5
            actual_md5 = hashlib.md5(assembled_data).hexdigest()
            if actual_md5 != expected_md5:
                raise ValueError(f"File MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            
            # Save to final location
            images_dir = Path(self.args.input_dir) / 'images'
            metadata_dir = Path(self.args.input_dir) / 'metadata'
            
            final_image_path = images_dir / filename
            final_metadata_path = metadata_dir / f"{Path(filename).stem}.txt"
            
            # Write image file
            with open(final_image_path, 'wb') as f:
                f.write(assembled_data)
            
            # Copy metadata file
            metadata_source = work_dir / 'metadata.txt'
            if metadata_source.exists():
                with open(metadata_source, 'r', encoding='utf-8') as src:
                    with open(final_metadata_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            
            # Update processed images JSON
            self.update_processed_images(
                file_id=file_id,
                filename=filename,
                file_md5=actual_md5,
                saved_path=str(final_image_path),
                metadata_path=str(final_metadata_path)
            )
            
            # Update database
            with self.db_lock:
                with sqlite3.connect(self.db_path) as db_conn:
                    db_conn.execute('''
                        UPDATE files 
                        SET status = 'complete', saved_path = ?, last_update = CURRENT_TIMESTAMP
                        WHERE file_id = ?
                    ''', (str(final_image_path), file_id))
            
            self.logger.info(f"Successfully assembled and saved: {filename}")
            
            # Clean up working directory
            if self.args.cleanup_temp:
                self.cleanup_working_directory(work_dir)
            
            # Send FILE_COMPLETE response
            complete_response = self.create_file_complete_packet(
                file_id=file_id,
                file_md5=actual_md5,
                saved_path=str(final_image_path),
                status='OK'
            )
            self.send_response(conn, complete_response)
            
        except Exception as e:
            self.logger.error(f"Error assembling file {file_id}: {e}")
            
            # Update database to failed status
            with self.db_lock:
                with sqlite3.connect(self.db_path) as db_conn:
                    db_conn.execute('''
                        UPDATE files 
                        SET status = 'failed', last_update = CURRENT_TIMESTAMP
                        WHERE file_id = ?
                    ''', (file_id,))
            
            # Send error response
            error_response = self.create_file_complete_packet(
                file_id=file_id,
                file_md5='',
                saved_path='',
                status='ERROR'
            )
            self.send_response(conn, error_response)
    
    def cleanup_working_directory(self, work_dir: Path):
        """Clean up temporary working directory."""
        try:
            if work_dir.exists():
                # Remove all files in directory
                for file_path in work_dir.iterdir():
                    if file_path.is_file():
                        file_path.unlink()
                
                # Remove directory
                work_dir.rmdir()
                
                self.logger.debug(f"Cleaned up working directory: {work_dir}")
                
        except Exception as e:
            self.logger.warning(f"Error cleaning up working directory {work_dir}: {e}")
    
    def handle_client_connection(self, conn: socket.socket, addr):
        """Handle individual client connection."""
        self.logger.info(f"New connection from {addr}")
        
        try:
            while self.running:
                # Receive packet length
                length_bytes = conn.recv(4)
                if len(length_bytes) != 4:
                    break
                
                packet_length = int.from_bytes(length_bytes, 'big')
                
                # Receive packet data
                packet_data = b''
                while len(packet_data) < packet_length:
                    chunk = conn.recv(min(4096, packet_length - len(packet_data)))
                    if not chunk:
                        break
                    packet_data += chunk
                
                if len(packet_data) != packet_length:
                    self.logger.warning(f"Incomplete packet received from {addr}")
                    break
                
                # Decode and parse packet
                try:
                    packet_text = packet_data.decode('utf-8')
                    parsed = self.parse_packet(packet_text)
                    
                    packet_type = parsed['headers'].get('TYPE')
                    
                    if packet_type == 'FILE_INIT':
                        self.handle_file_init(conn, parsed)
                    elif packet_type in ['METADATA', 'METADATA_PART']:
                        self.handle_metadata(conn, parsed)
                    elif packet_type == 'PART':
                        self.handle_part(conn, parsed)
                    else:
                        self.logger.warning(f"Unknown packet type: {packet_type}")
                        
                except UnicodeDecodeError as e:
                    self.logger.error(f"Error decoding packet from {addr}: {e}")
                    break
                except Exception as e:
                    self.logger.error(f"Error processing packet from {addr}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error handling connection from {addr}: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
            self.logger.info(f"Connection closed from {addr}")
    
    def run_server(self):
        """Run the TCP server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((self.args.listen_host, self.args.listen_port))
            server_socket.listen(self.args.max_connections)
            
            self.logger.info(f"Server listening on {self.args.listen_host}:{self.args.listen_port}")
            
            while self.running:
                try:
                    conn, addr = server_socket.accept()
                    
                    # Handle connection in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client_connection,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Socket error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            server_socket.close()
    
    def run(self):
        """Run the receiver daemon."""
        self.logger.info("Starting Phase 2 Receiver daemon")
        
        try:
            self.run_server()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.logger.info("Phase 2 Receiver daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Receiver - Receive anomaly detection data from Phase 1")
    
    # Network options
    parser.add_argument('--listen-host', default='0.0.0.0',
                       help='Listen host address (default: 0.0.0.0)')
    parser.add_argument('--listen-port', type=int, default=5001,
                       help='Listen port (default: 5001)')
    parser.add_argument('--max-connections', type=int, default=2,
                       help='Maximum concurrent connections (default: 2)')
    
    # Directory options
    parser.add_argument('--input-dir', default='captures',
                       help='Input directory for Phase 2 (default: captures)')
    parser.add_argument('--receive-queue', default='receive_queue',
                       help='Temporary receive queue directory (default: receive_queue)')
    
    # Storage options
    parser.add_argument('--db-path', default='receiver.db',
                       help='SQLite database path (default: receiver.db)')
    parser.add_argument('--processed-json', default='output/processed_images.json',
                       help='Processed images JSON file (default: output/processed_images.json)')
    parser.add_argument('--cleanup-temp', action='store_true',
                       help='Clean up temporary files after completion')
    
    # Logging options
    parser.add_argument('--log-file',
                       help='Log file path (default: stdout)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    # Testing options
    parser.add_argument('--test-mode', action='store_true',
                       help='Enable test mode with simulated packet loss')
    
    args = parser.parse_args()
    
    try:
        receiver = Phase2Receiver(args)
        receiver.run()
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
