import argparse
import sqlite3
import socket
import time
import hashlib
import base64
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from datetime import datetime, timezone
import signal
import math
import struct

# Constants
MAX_PACKET_SIZE = 1400  # Reduced for UDP to avoid fragmentation
HEADER_OVERHEAD_ESTIMATE = 300  # Increased for UDP headers
base64_overhead_factor = 4 / 3  # ~1.333

raw_chunk_size = int((MAX_PACKET_SIZE - HEADER_OVERHEAD_ESTIMATE) / base64_overhead_factor)

DELIMITER = "---DATA---"
END_DELIMITER = "---END---"

# UDP-specific constants
UDP_TIMEOUT = 5.0
MAX_RETRIES = 5
ACK_TIMEOUT = 2.0

class FileInfo(NamedTuple):
    file_id: str
    filename: str
    filepath: str
    metadata_path: str
    file_size: int
    file_md5: str
    total_parts: int

class PacketInfo(NamedTuple):
    file_id: str
    part_index: int
    chunk_data: bytes
    chunk_md5: str

class UDPSender:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.db_lock = threading.Lock()
        self.socket = None
        self.sequence_number = 0
        self.sequence_lock = threading.Lock()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize directories
        self.setup_directories()
        
        # Initialize database
        self.init_database()
        
        # Calculate packet sizing
        self.calculate_packet_sizing()
        
        # Initialize UDP socket
        self.init_socket()
        
        self.logger.info(f"UDP Sender initialized")
        self.logger.info(f"  Images dir: {self.args.images_dir}")
        self.logger.info(f"  Metadata dir: {self.args.metadata_dir}")
        self.logger.info(f"  Target host: {self.args.host}:{self.args.port}")
        self.logger.info(f"  Packet size: {self.args.packet_size} bytes")
        self.logger.info(f"  Raw chunk size: {self.raw_chunk_size} bytes")
    
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
        Path(self.args.images_dir).mkdir(parents=True, exist_ok=True)
        Path(self.args.metadata_dir).mkdir(parents=True, exist_ok=True)
        
        if self.args.archive_dir:
            Path(self.args.archive_dir).mkdir(parents=True, exist_ok=True)
    
    def calculate_packet_sizing(self):
        """Calculate optimal chunk sizes for packet transmission."""
        # Start with safe defaults for UDP
        self.raw_chunk_size = 800  # Conservative default for UDP
        
        # Calculate raw chunk size directly from packet size
        available_space = self.args.packet_size - self.args.header_overhead
        max_base64_chars = (available_space // 4) * 4
        self.raw_chunk_size = (max_base64_chars // 4) * 3  # Convert back to raw bytes
        
        # Safety margin (reduce by ~10% for UDP headers and sequence numbers)
        self.raw_chunk_size = int(self.raw_chunk_size * 0.90)
        
        self.logger.info(f"Calculated raw chunk size: {self.raw_chunk_size} bytes")
        self.logger.info(f"Estimated base64 size: {math.ceil(self.raw_chunk_size * 4/3)} chars")
    
    def init_socket(self):
        """Initialize UDP socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(UDP_TIMEOUT)
            self.logger.info("UDP socket initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize UDP socket: {e}")
            raise
    
    def get_next_sequence(self):
        """Get next sequence number thread-safely."""
        with self.sequence_lock:
            self.sequence_number += 1
            return self.sequence_number
    
    def init_database(self):
        """Initialize SQLite database for transfer state."""
        self.db_path = Path(self.args.db_path)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    metadata_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_md5 TEXT NOT NULL,
                    total_parts INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS packets (
                    file_id TEXT NOT NULL,
                    part_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    chunk_md5 TEXT NOT NULL,
                    retries INTEGER DEFAULT 0,
                    last_attempt TIMESTAMP,
                    PRIMARY KEY (file_id, part_index),
                    FOREIGN KEY (file_id) REFERENCES files (file_id)
                )
            ''')
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_files_status ON files (status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_packets_status ON packets (status)')
    
    def compute_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error computing hash for {filepath}: {e}")
            raise
    
    def compute_chunk_hash(self, data: bytes) -> str:
        """Compute MD5 hash of data chunk."""
        return hashlib.md5(data).hexdigest()

    def discover_new_files(self) -> List[Tuple[str, str]]:
        """Discover new image/metadata pairs, including initial_scene.jpg."""
        
        pairs = []

        # --- 1. Normal detection images ---
        image_files = []
        for ext in ['*.jpg', '*.jpeg']:
            image_files.extend(Path(self.args.images_dir).glob(ext))

        # --- 2. Add initial_scene.jpg from captures/ directory ---
        captures_dir = Path(self.args.images_dir).parent
        initial_scene_path = captures_dir / "initial_scene.jpg"
        if initial_scene_path.exists():
            # Look for metadata
            meta_path = Path(self.args.metadata_dir) / f"{initial_scene_path.stem}.txt"
            if meta_path.exists():
                pairs.append((str(initial_scene_path), str(meta_path)))
            else:
                # If no metadata, still send it (empty metadata file)
                temp_meta = Path(self.args.metadata_dir) / f"{initial_scene_path.stem}.txt"
                temp_meta.write_text("INITIAL_SCENE\n", encoding='utf-8')
                pairs.append((str(initial_scene_path), str(temp_meta)))

        # --- 3. Match other image/metadata pairs ---
        for img_path in image_files:
            metadata_path = Path(self.args.metadata_dir) / f"{img_path.stem}.txt"
            if metadata_path.exists():
                pairs.append((str(img_path), str(metadata_path)))

        return pairs

    def is_file_queued(self, filepath: str) -> bool:
        """Check if file is already in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM files WHERE filepath = ?",
                (filepath,)
            )
            return cursor.fetchone()[0] > 0
    
    def enqueue_file_for_transfer(self, filepath: str, metadata_path: str) -> bool:
        """Add file to transfer queue."""
        try:
            if self.is_file_queued(filepath):
                self.logger.debug(f"File already queued: {filepath}")
                return True
            
            # Compute file info
            file_size = os.path.getsize(filepath)
            file_md5 = self.compute_file_hash(filepath)
            filename = Path(filepath).name
            file_id = Path(filepath).stem  # Use stem as file_id
            
            # Calculate total parts needed
            total_parts = math.ceil(file_size / self.raw_chunk_size)
            
            file_info = FileInfo(
                file_id=file_id,
                filename=filename,
                filepath=filepath,
                metadata_path=metadata_path,
                file_size=file_size,
                file_md5=file_md5,
                total_parts=total_parts
            )
            
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Insert file record
                    conn.execute('''
                        INSERT INTO files 
                        (file_id, filename, filepath, metadata_path, file_size, file_md5, total_parts)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        file_info.file_id, file_info.filename, file_info.filepath,
                        file_info.metadata_path, file_info.file_size, 
                        file_info.file_md5, file_info.total_parts
                    ))
                    
                    # Pre-create packet records
                    for part_idx in range(total_parts):
                        # Read chunk to compute hash
                        chunk_data = self.read_file_chunk(filepath, part_idx)
                        chunk_md5 = self.compute_chunk_hash(chunk_data)
                        
                        conn.execute('''
                            INSERT INTO packets (file_id, part_index, chunk_md5)
                            VALUES (?, ?, ?)
                        ''', (file_info.file_id, part_idx, chunk_md5))
            
            self.logger.info(f"Queued file for transfer: {filename} ({file_size} bytes, {total_parts} parts)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error queuing file {filepath}: {e}")
            return False
    
    def read_file_chunk(self, filepath: str, part_index: int) -> bytes:
        """Read specific chunk from file."""
        offset = part_index * self.raw_chunk_size
        
        with open(filepath, 'rb') as f:
            f.seek(offset)
            return f.read(self.raw_chunk_size)
    
    def create_packet_header(self, packet_type: str, **kwargs) -> str:
        """Create packet header with given type and parameters."""
        lines = [f"TYPE: {packet_type}"]
        
        # Add sequence number for UDP reliability
        lines.append(f"SEQ: {self.get_next_sequence()}")
        
        for key, value in kwargs.items():
            lines.append(f"{key.upper()}: {value}")
        
        # Add timestamp
        lines.append(f"SENT_AT: {datetime.now(timezone.utc).isoformat()}")
        
        return "\n".join(lines) + "\n"
    
    def create_file_init_packet(self, file_info: FileInfo) -> str:
        """Create FILE_INIT packet."""
        # Add sender identification
        header = self.create_packet_header(
            "FILE_INIT",
            file=file_info.filename,
            file_id=file_info.file_id,
            file_size=file_info.file_size,
            file_md5=file_info.file_md5,
            total_parts=file_info.total_parts,
            encoding="base64",
            sender_id=self.args.sender_id,
            sender_hostname=socket.gethostname()
        )
        
        packet = header + END_DELIMITER + "\n"
        
        if len(packet.encode('utf-8')) > self.args.packet_size:
            raise ValueError(f"FILE_INIT packet too large: {len(packet)} bytes")
        
        return packet
    
    def create_metadata_packet(self, file_id: str, metadata_path: str) -> List[str]:
        """Create METADATA packet(s) for metadata file."""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = f.read()
            
            filename = Path(metadata_path).name
            
            # Check if metadata fits in single packet
            header = self.create_packet_header(
                "METADATA",
                file_id=file_id,
                filename=filename,
                length=len(metadata_content),
                sender_id=self.args.sender_id
            )
            
            single_packet = header + DELIMITER + "\n" + metadata_content
            
            if len(single_packet.encode('utf-8')) <= self.args.packet_size:
                return [single_packet]
            
            # Split into multiple parts if too large
            packets = []
            max_content_size = self.args.packet_size - len(header.encode('utf-8')) - len(DELIMITER) - 50  # Extra margin for UDP
            
            parts = []
            for i in range(0, len(metadata_content), max_content_size):
                parts.append(metadata_content[i:i + max_content_size])
            
            for i, part in enumerate(parts):
                part_header = self.create_packet_header(
                    "METADATA_PART",
                    file_id=file_id,
                    filename=filename,
                    part=f"{i+1}/{len(parts)}",
                    length=len(part),
                    sender_id=self.args.sender_id
                )
                
                packet = part_header + DELIMITER + "\n" + part
                packets.append(packet)
            
            return packets
            
        except Exception as e:
            self.logger.error(f"Error creating metadata packet: {e}")
            raise
    
    def create_part_packet(self, file_info: FileInfo, part_index: int, chunk_data: bytes, chunk_md5: str) -> str:
        """Create PART packet for image chunk."""
        # Encode chunk as base64
        base64_data = base64.b64encode(chunk_data).decode('ascii')
        
        # Create header
        header = self.create_packet_header(
            "PART",
            file_id=file_info.file_id,
            part=f"{part_index+1:03d}/{file_info.total_parts:03d}",
            chunk_len=len(chunk_data),
            chunk_md5=chunk_md5,
            sender_id=self.args.sender_id
        )
        
        # Assemble packet
        packet = header + DELIMITER + "\n" + base64_data
        
        # Verify size constraint
        packet_size = len(packet.encode('utf-8'))
        if packet_size > self.args.packet_size:
            raise ValueError(f"PART packet too large: {packet_size} bytes")
        
        return packet
    
    def send_udp_packet_with_retry(self, packet: str, expected_ack_type: str) -> bool:
        """Send UDP packet and wait for ACK with retry logic."""
        packet_bytes = packet.encode('utf-8')
        
        for attempt in range(self.args.max_retries + 1):
            try:
                if self.args.verbose:
                    self.logger.debug(f"Sending UDP packet (attempt {attempt + 1}): {packet[:100]}...")
                
                # Send packet via UDP
                self.socket.sendto(packet_bytes, (self.args.host, self.args.port))
                
                # Wait for ACK with timeout
                start_time = time.time()
                while time.time() - start_time < ACK_TIMEOUT:
                    try:
                        self.socket.settimeout(0.1)  # Short timeout for non-blocking receive
                        ack_data, addr = self.socket.recvfrom(4096)
                        ack_text = ack_data.decode('utf-8')
                        
                        if self.args.verbose:
                            self.logger.debug(f"Received UDP ACK: {ack_text[:100]}...")
                        
                        # Parse ACK
                        if self.parse_ack_response(ack_text, expected_ack_type):
                            return True
                        else:
                            self.logger.warning(f"Received NACK or error for {expected_ack_type}")
                            break
                            
                    except socket.timeout:
                        continue
                    except Exception as e:
                        self.logger.debug(f"Error receiving ACK: {e}")
                        continue
                
                self.logger.warning(f"No valid ACK received for {expected_ack_type}")
                        
            except Exception as e:
                self.logger.warning(f"UDP send attempt {attempt + 1} failed: {e}")
                
                if attempt < self.args.max_retries:
                    backoff = self.args.retry_backoff ** attempt
                    self.logger.info(f"Retrying in {backoff:.1f} seconds...")
                    time.sleep(backoff)
        
        return False
    
    def parse_ack_response(self, ack_data: str, expected_context: str) -> bool:
        """Parse ACK response and determine if successful."""
        try:
            lines = ack_data.strip().split('\n')
            ack_info = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    ack_info[key.strip()] = value.strip()
            
            # Check if this is the expected ACK
            ack_type = ack_info.get('TYPE')
            status = ack_info.get('STATUS')
            
            return ack_type == 'ACK' and status == 'OK'
            
        except Exception as e:
            self.logger.error(f"Error parsing ACK: {e}")
            return False
    
    def send_file(self, file_info: FileInfo) -> bool:
        """Send complete file to receiver."""
        try:
            self.logger.info(f"Starting UDP transfer of {file_info.filename}")
            
            # Update status to sending
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE files SET status = 'sending', last_update = CURRENT_TIMESTAMP WHERE file_id = ?",
                        (file_info.file_id,)
                    )
            
            # Send FILE_INIT
            init_packet = self.create_file_init_packet(file_info)
            if not self.send_udp_packet_with_retry(init_packet, "FILE_INIT"):
                raise RuntimeError("Failed to send FILE_INIT")
            
            # Send METADATA
            metadata_packets = self.create_metadata_packet(file_info.file_id, file_info.metadata_path)
            for i, metadata_packet in enumerate(metadata_packets):
                if not self.send_udp_packet_with_retry(metadata_packet, "METADATA"):
                    raise RuntimeError(f"Failed to send METADATA part {i+1}")
            
            # Send PART packets
            for part_index in range(file_info.total_parts):
                # Check if part already ACKed
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT status FROM packets WHERE file_id = ? AND part_index = ?",
                        (file_info.file_id, part_index)
                    )
                    row = cursor.fetchone()
                    if row and row[0] == 'acked':
                        self.logger.debug(f"Skipping already ACKed part {part_index+1}")
                        continue
                
                # Read chunk and create packet
                chunk_data = self.read_file_chunk(file_info.filepath, part_index)
                chunk_md5 = self.compute_chunk_hash(chunk_data)
                
                part_packet = self.create_part_packet(file_info, part_index, chunk_data, chunk_md5)
                
                if self.send_udp_packet_with_retry(part_packet, "PART"):
                    # Mark packet as ACKed
                    with self.db_lock:
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                "UPDATE packets SET status = 'acked', last_attempt = CURRENT_TIMESTAMP WHERE file_id = ? AND part_index = ?",
                                (file_info.file_id, part_index)
                            )
                else:
                    raise RuntimeError(f"Failed to send PART {part_index+1}")
            
            # Wait for FILE_COMPLETE confirmation with longer timeout
            self.logger.info(f"Waiting for FILE_COMPLETE confirmation...")
            
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout for FILE_COMPLETE
                try:
                    self.socket.settimeout(1.0)
                    complete_data, addr = self.socket.recvfrom(4096)
                    complete_text = complete_data.decode('utf-8')
                    
                    if self.args.verbose:
                        self.logger.debug(f"Received FILE_COMPLETE: {complete_text}")
                    
                    if self.parse_file_complete_response(complete_text, file_info):
                        # Mark file as transferred
                        with self.db_lock:
                            with sqlite3.connect(self.db_path) as conn:
                                conn.execute(
                                    "UPDATE files SET status = 'transferred', last_update = CURRENT_TIMESTAMP WHERE file_id = ?",
                                    (file_info.file_id,)
                                )
                        
                        self.logger.info(f"Successfully transferred {file_info.filename}")
                        
                        # Handle file cleanup
                        self.handle_file_cleanup(file_info)
                        
                        return True
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.debug(f"Error waiting for FILE_COMPLETE: {e}")
                    continue
            
            raise RuntimeError("FILE_COMPLETE confirmation timeout")
            
        except Exception as e:
            self.logger.error(f"Error sending file {file_info.filename}: {e}")
            
            # Mark file as failed
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE files SET status = 'failed', last_update = CURRENT_TIMESTAMP WHERE file_id = ?",
                        (file_info.file_id,)
                    )
            
            return False
    
    def parse_file_complete_response(self, response_data: str, file_info: FileInfo) -> bool:
        """Parse FILE_COMPLETE response."""
        try:
            if self.args.verbose:
                self.logger.debug(f"Parsing FILE_COMPLETE response: {response_data}")
            
            lines = response_data.strip().split('\n')
            response_info = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    response_info[key.strip()] = value.strip()
            
            # Check if this is an ACK response for FILE_COMPLETE
            is_ack = response_info.get('TYPE') == 'ACK'
            is_file_complete = response_info.get('ORIGINAL_TYPE') == 'FILE_COMPLETE'
            is_success = response_info.get('STATUS') == 'OK'
            correct_file = response_info.get('FILE_ID') == file_info.file_id
            
            self.logger.debug(f"FILE_COMPLETE analysis: ACK={is_ack}, COMPLETE={is_file_complete}, SUCCESS={is_success}, FILE_MATCH={correct_file}")
            
            # Accept either format:
            # 1. Direct FILE_COMPLETE response
            # 2. ACK response with ORIGINAL_TYPE=FILE_COMPLETE
            success = False
            if is_ack and is_file_complete and is_success and correct_file:
                success = True
                self.logger.debug("FILE_COMPLETE confirmed via ACK format")
            elif (response_info.get('TYPE') == 'FILE_COMPLETE' and 
                  is_success and correct_file):
                success = True
                self.logger.debug("FILE_COMPLETE confirmed via direct format")
            
            if not success:
                self.logger.warning(f"FILE_COMPLETE validation failed: {response_info}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error parsing FILE_COMPLETE: {e}")
            return False
    
    def handle_file_cleanup(self, file_info: FileInfo):
        """Handle file cleanup after successful transfer."""
        try:
            if self.args.delete_after_transfer:
                # Delete original files
                os.remove(file_info.filepath)
                os.remove(file_info.metadata_path)
                self.logger.info(f"Deleted transferred files: {file_info.filename}")
                
            elif self.args.archive_dir:
                # Move to archive
                archive_img = Path(self.args.archive_dir) / file_info.filename
                archive_meta = Path(self.args.archive_dir) / Path(file_info.metadata_path).name
                
                os.rename(file_info.filepath, archive_img)
                os.rename(file_info.metadata_path, archive_meta)
                self.logger.info(f"Archived transferred files: {file_info.filename}")
                
        except Exception as e:
            self.logger.error(f"Error during file cleanup: {e}")
    
    def get_pending_files(self) -> List[FileInfo]:
        """Get list of files pending transfer."""
        files = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT file_id, filename, filepath, metadata_path, file_size, file_md5, total_parts
                FROM files 
                WHERE status IN ('pending', 'sending', 'failed')
                ORDER BY created_at
            ''')
            
            for row in cursor.fetchall():
                files.append(FileInfo(*row))
        
        return files
    
    def process_transfers(self):
        """Main transfer processing loop."""
        while self.running:
            try:
                # Discover new files
                new_pairs = self.discover_new_files()
                
                # Enqueue new files
                for img_path, meta_path in new_pairs:
                    self.enqueue_file_for_transfer(img_path, meta_path)
                
                # Process pending transfers
                pending_files = self.get_pending_files()
                
                for file_info in pending_files:
                    if not self.running:
                        break
                    
                    # Verify files still exist
                    if not (Path(file_info.filepath).exists() and Path(file_info.metadata_path).exists()):
                        self.logger.warning(f"File missing, skipping: {file_info.filename}")
                        continue

                    if self.send_file(file_info):
                        time.sleep(0.5)  # Small delay between transfers
                
                if self.args.once:
                    break
                
                # Wait before next scan
                time.sleep(self.args.watch_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                break
            except Exception as e:
                self.logger.error(f"Error in transfer loop: {e}")
                time.sleep(self.args.watch_interval)
    
    def run(self):
        """Run the sender daemon."""
        self.logger.info("Starting UDP Sender daemon")
        
        try:
            self.process_transfers()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if self.socket:
                self.socket.close()
            self.logger.info("UDP Sender daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="UDP Sender - Transfer anomaly detection data to Phase 2")
    
    # Directory options
    parser.add_argument('--images-dir', default='captures/images', 
                       help='Directory containing JPEG images (default: captures/images)')
    parser.add_argument('--metadata-dir', default='captures/metadata',
                       help='Directory containing metadata txt files (default: captures/metadata)')
    parser.add_argument('--archive-dir', 
                       help='Directory to archive transferred files (optional)')
    
    # Network options
    parser.add_argument('--host', default='phase2.local',
                       help='Phase 2 receiver hostname or IP (default: phase2.local)')
    parser.add_argument('--port', type=int, default=5001,
                       help='Phase 2 receiver port (default: 5001)')
    
    # Sender identification
    parser.add_argument('--sender-id', default=f'sender_{socket.gethostname()}',
                       help='Unique sender identifier (default: sender_<hostname>)')
    
    # Transfer options
    parser.add_argument('--watch-interval', type=float, default=2.0,
                       help='File watch interval in seconds (default: 2.0)')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES,
                       help=f'Maximum retry attempts per packet (default: {MAX_RETRIES})')
    parser.add_argument('--retry-backoff', type=float, default=1.5,
                       help='Retry backoff multiplier (default: 1.5)')
    parser.add_argument('--packet-size', type=int, default=MAX_PACKET_SIZE,
                       help=f'Maximum packet size in bytes (default: {MAX_PACKET_SIZE})')
    parser.add_argument('--header-overhead', type=int, default=HEADER_OVERHEAD_ESTIMATE,
                       help=f'Header overhead estimate in bytes (default: {HEADER_OVERHEAD_ESTIMATE})')
    
    # Storage options
    parser.add_argument('--db-path', default='sender.db',
                       help='SQLite database path (default: sender.db)')
    parser.add_argument('--delete-after-transfer', action='store_true',
                       help='Delete files after successful transfer')
    
    # Operation modes
    parser.add_argument('--once', action='store_true',
                       help='Run once then exit (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - discover files but do not transfer')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # Logging options
    parser.add_argument('--log-file',
                       help='Log file path (default: stdout)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    args = parser.parse_args()
    
    try:
        sender = UDPSender(args)
        sender.run()
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
