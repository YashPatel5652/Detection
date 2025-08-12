#!/usr/bin/env python3
"""
Phase 2 Multi-Sender Receiver - phase2_receiver.py

Enhanced receiver that handles multiple senders simultaneously,
maintains sender isolation, and triggers incremental reconstruction.
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
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple, Set
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import queue

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
    sender_id: str

class SenderState:
    """Track state for individual senders."""
    
    def __init__(self, sender_id: str, base_dir: Path):
        self.sender_id = sender_id
        self.base_dir = base_dir
        self.images_dir = base_dir / 'images'
        self.metadata_dir = base_dir / 'metadata'
        self.output_dir = Path('output') / sender_id
        self.batch_dir = self.output_dir / 'batch_001'
        
        # State tracking
        self.initial_scene_received = False
        self.initial_scene_path = None
        self.pending_objects = set()  # New object files since last reconstruction
        self.reconstruction_lock = threading.Lock()
        self.last_reconstruction_time = 0
        self.reconstruction_queue = queue.Queue()
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
    
    def add_pending_object(self, filename: str):
        """Add new object to pending reconstruction queue."""
        self.pending_objects.add(filename)
    
    def clear_pending_objects(self):
        """Clear pending objects after reconstruction."""
        self.pending_objects.clear()
    
    def should_trigger_reconstruction(self) -> bool:
        """Check if reconstruction should be triggered."""
        return (self.initial_scene_received and 
                len(self.pending_objects) > 0)
    
    def get_reconstruction_args(self, debug_mode: bool = False) -> List[str]:
        """Get arguments for 2 phase final reconstruction with proper boolean formatting."""
        args = [
            sys.executable, '2 phase final.py',
            '--input-dir', str(self.base_dir),
            '--output-dir', str(self.output_dir),
            '--scene-image', 'initial_scene.jpg',
            '--batch-dir', str(self.batch_dir),
            '--sender-id', self.sender_id
        ]
        
        # Add debug mode if requested
        if debug_mode:
            args.append('--debug-mode')  # This is also a store_true flag
        
        return args

class MultiSenderPhase2Receiver:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.db_lock = threading.Lock()
        self.json_lock = threading.Lock()
        self.senders_lock = threading.Lock()
        
        # Multi-sender state management
        self.senders: Dict[str, SenderState] = {}
        self.reconstruction_executor = ThreadPoolExecutor(max_workers=args.max_reconstruction_workers)
        self.reconstruction_timers: Dict[str, threading.Timer] = {}
        
        # Base directories
        self.received_data_dir = Path(args.received_data_dir)
        self.received_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize database
        self.init_database()
        
        # Load global processed images tracking
        self.load_processed_images()
        
        self.logger.info(f"Multi-Sender Phase 2 Receiver initialized")
        self.logger.info(f"  Listen address: {self.args.listen_host}:{self.args.listen_port}")
        self.logger.info(f"  Received data directory: {self.args.received_data_dir}")
        self.logger.info(f"  Max reconstruction workers: {self.args.max_reconstruction_workers}")
        self.logger.info(f"  Reconstruction delay: {self.args.reconstruction_delay}s")
    
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
    
    def init_database(self):
        """Initialize SQLite database with sender support."""
        self.db_path = Path(self.args.db_path)
        
        with sqlite3.connect(self.db_path) as conn:
            # Files table with sender_id
            conn.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    sender_id TEXT NOT NULL,
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
            
            # Parts table
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
            
            # Senders table for tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS senders (
                    sender_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    files_received INTEGER DEFAULT 0,
                    initial_scene_received BOOLEAN DEFAULT FALSE,
                    last_reconstruction TIMESTAMP
                )
            ''')
            
            # Add indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_files_sender ON files (sender_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_files_status ON files (status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_parts_status ON parts (status)')
    
    def load_processed_images(self):
        """Load existing processed images JSON (global tracking)."""
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
    
    def get_or_create_sender(self, sender_id: str) -> SenderState:
        """Get or create sender state."""
        with self.senders_lock:
            if sender_id not in self.senders:
                sender_base_dir = self.received_data_dir / sender_id
                self.senders[sender_id] = SenderState(sender_id, sender_base_dir)
                
                # Update database
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute('''
                            INSERT OR IGNORE INTO senders (sender_id, first_seen, last_activity)
                            VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ''', (sender_id,))
                
                self.logger.info(f"Created new sender state: {sender_id}")
            
            return self.senders[sender_id]
    
    def update_sender_activity(self, sender_id: str):
        """Update sender's last activity timestamp."""
        with self.db_lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE senders 
                    SET last_activity = CURRENT_TIMESTAMP
                    WHERE sender_id = ?
                ''', (sender_id,))
    
    def parse_packet(self, packet_data: str) -> Dict:
        """Parse incoming packet and extract headers and data."""
        try:
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
            f"ORIGINAL_TYPE: {packet_type}",
            f"FILE_ID: {file_id}",
            f"STATUS: {status}",
            f"RECEIVED_AT: {datetime.now(timezone.utc).isoformat()}"
        ]
        
        for key, value in kwargs.items():
            lines.append(f"{key.upper()}: {value}")
        
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
        """Handle FILE_INIT packet with sender identification."""
        try:
            headers = parsed['headers']
            
            file_id = headers.get('FILE_ID')
            filename = headers.get('FILE')
            file_size = int(headers.get('FILE_SIZE', 0))
            file_md5 = headers.get('FILE_MD5')
            total_parts = int(headers.get('TOTAL_PARTS', 0))
            sender_id = headers.get('SENDER_ID', 'unknown_sender')
            
            if not all([file_id, filename, file_md5]):
                self.send_response(conn, self.create_ack_packet('FILE_INIT', file_id or 'unknown', 'ERROR', error='missing_required_fields'))
                return False
            
            # Get or create sender state
            sender = self.get_or_create_sender(sender_id)
            self.update_sender_activity(sender_id)
            
            # Create working directory for this transfer
            work_dir = Path(self.args.receive_queue) / file_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Store file info in database with sender_id
            with self.db_lock:
                with sqlite3.connect(self.db_path) as db_conn:
                    db_conn.execute('''
                        INSERT OR REPLACE INTO files 
                        (file_id, sender_id, filename, file_size, file_md5, total_parts, status, last_update)
                        VALUES (?, ?, ?, ?, ?, ?, 'receiving', CURRENT_TIMESTAMP)
                    ''', (file_id, sender_id, filename, file_size, file_md5, total_parts))
                    
                    # Pre-create part records
                    for part_idx in range(total_parts):
                        db_conn.execute('''
                            INSERT OR IGNORE INTO parts (file_id, part_index, chunk_md5)
                            VALUES (?, ?, '')
                        ''', (file_id, part_idx))
            
            self.logger.info(f"[{sender_id}] Initialized file transfer: {filename} ({file_size} bytes, {total_parts} parts)")
            
            # Send ACK
            ack = self.create_ack_packet('FILE_INIT', file_id, 'OK', sender_id=sender_id)
            self.send_response(conn, ack)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling FILE_INIT: {e}")
            file_id = parsed['headers'].get('FILE_ID', 'unknown')
            sender_id = parsed['headers'].get('SENDER_ID', 'unknown')
            self.send_response(conn, self.create_ack_packet('FILE_INIT', file_id, 'ERROR', error=str(e), sender_id=sender_id))
            return False
    
    def handle_metadata(self, conn: socket.socket, parsed: Dict) -> bool:
        """Handle METADATA packet with sender isolation."""
        try:
            headers = parsed['headers']
            
            file_id = headers.get('FILE_ID')
            filename = headers.get('FILENAME')
            sender_id = headers.get('SENDER_ID', 'unknown_sender')
            metadata_content = parsed.get('data', '')
            
            if not file_id:
                self.send_response(conn, self.create_ack_packet('METADATA', 'unknown', 'ERROR', error='missing_file_id'))
                return False
            
            # Get sender state
            sender = self.get_or_create_sender(sender_id)
            self.update_sender_activity(sender_id)
            
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
            
            self.logger.debug(f"[{sender_id}] Saved metadata for {file_id}")
            
            # Send ACK
            ack = self.create_ack_packet('METADATA', file_id, 'OK', sender_id=sender_id)
            self.send_response(conn, ack)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling METADATA: {e}")
            file_id = parsed['headers'].get('FILE_ID', 'unknown')
            sender_id = parsed['headers'].get('SENDER_ID', 'unknown')
            self.send_response(conn, self.create_ack_packet('METADATA', file_id, 'ERROR', error=str(e), sender_id=sender_id))
            return False
    
    def handle_part(self, conn: socket.socket, parsed: Dict) -> bool:
        """Handle PART packet with sender isolation."""
        try:
            headers = parsed['headers']
            
            file_id = headers.get('FILE_ID')
            part_info = headers.get('PART', '1/1')
            chunk_len = int(headers.get('CHUNK_LEN', 0))
            chunk_md5 = headers.get('CHUNK_MD5')
            sender_id = headers.get('SENDER_ID', 'unknown_sender')
            base64_data = parsed.get('data', '')
            
            if not all([file_id, part_info, chunk_md5, base64_data]):
                self.send_response(conn, self.create_ack_packet('PART', file_id or 'unknown', 'ERROR', error='missing_required_fields'))
                return False
            
            # Parse part index
            part_num, total_parts = map(int, part_info.split('/'))
            part_index = part_num - 1  # Convert to 0-based index
            
            # Get sender state
            sender = self.get_or_create_sender(sender_id)
            self.update_sender_activity(sender_id)
            
            # Decode base64 data
            try:
                chunk_data = base64.b64decode(base64_data.replace('\n', ''))
            except Exception as e:
                self.logger.error(f"[{sender_id}] Error decoding base64 for part {part_num}: {e}")
                self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error='base64_decode_error', sender_id=sender_id))
                return False
            
            # Verify chunk length and MD5
            if len(chunk_data) != chunk_len:
                self.logger.error(f"[{sender_id}] Chunk length mismatch for part {part_num}: expected {chunk_len}, got {len(chunk_data)}")
                self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error='chunk_length_mismatch', sender_id=sender_id))
                return False
            
            actual_md5 = hashlib.md5(chunk_data).hexdigest()
            if actual_md5 != chunk_md5:
                self.logger.error(f"[{sender_id}] Chunk MD5 mismatch for part {part_num}: expected {chunk_md5}, got {actual_md5}")
                self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error='checksum_mismatch', sender_id=sender_id))
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
            
            self.logger.debug(f"[{sender_id}] Received part {part_num}/{total_parts} for {file_id}")
            
            # Send ACK
            ack = self.create_ack_packet('PART', file_id, 'OK', part=part_info, sender_id=sender_id)
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
                result = cursor.fetchone()
                if result:
                    total_parts_db = result[0]
                else:
                    raise ValueError(f"File record not found for {file_id}")
            
            if received_count == total_parts_db:
                # All parts received, assemble file
                self.assemble_and_save_file(conn, file_id, sender_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling PART: {e}")
            file_id = parsed['headers'].get('FILE_ID', 'unknown')
            sender_id = parsed['headers'].get('SENDER_ID', 'unknown')
            part_info = parsed['headers'].get('PART', 'unknown')
            self.send_response(conn, self.create_ack_packet('PART', file_id, 'ERROR', part=part_info, error=str(e), sender_id=sender_id))
            return False
    
    def assemble_and_save_file(self, conn: socket.socket, file_id: str, sender_id: str):
        """Assemble all parts and save final file to sender-specific directory."""
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
            sender = self.senders[sender_id]
            
            # Assemble file from parts
            assembled_data = bytearray()
            
            for part_index in range(total_parts):
                part_file = work_dir / f'part_{part_index:05d}.bin'
                
                if not part_file.exists():
                    raise FileNotFoundError(f"Missing part file: {part_file}")
                
                with open(part_file, 'rb') as f:
                    part_data = f.read()
                    assembled_data.extend(part_data)
            
            # Verify assembled file
            if len(assembled_data) != file_size:
                raise ValueError(f"File size mismatch: expected {file_size}, got {len(assembled_data)}")
            
            actual_md5 = hashlib.md5(assembled_data).hexdigest()
            if actual_md5 != expected_md5:
                raise ValueError(f"File MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            
            # Determine save location based on file type
            if filename == 'initial_scene.jpg' or filename.startswith('scene_'):
                # Save initial scene in sender's root directory
                final_path = sender.base_dir / filename
                sender.initial_scene_received = True
                sender.initial_scene_path = str(final_path)
                
                # Update sender database record
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as db_conn:
                        db_conn.execute('''
                            UPDATE senders 
                            SET initial_scene_received = TRUE, last_activity = CURRENT_TIMESTAMP
                            WHERE sender_id = ?
                        ''', (sender_id,))
                
                self.logger.info(f"[{sender_id}] Initial scene received: {filename}")
                
            else:
                # Object image - save to images directory
                final_path = sender.images_dir / filename
                sender.add_pending_object(filename)
                
                # Also save corresponding metadata
                metadata_source = work_dir / 'metadata.txt'
                if metadata_source.exists():
                    metadata_filename = f"{Path(filename).stem}.txt"
                    final_metadata_path = sender.metadata_dir / metadata_filename
                    
                    with open(metadata_source, 'r', encoding='utf-8') as src:
                        with open(final_metadata_path, 'w', encoding='utf-8') as dst:
                            dst.write(src.read())
                    
                    self.logger.debug(f"[{sender_id}] Saved metadata: {metadata_filename}")
                
                self.logger.info(f"[{sender_id}] Object image received: {filename}")
            
            # Write assembled file
            with open(final_path, 'wb') as f:
                f.write(assembled_data)
            
            # Update database
            with self.db_lock:
                with sqlite3.connect(self.db_path) as db_conn:
                    db_conn.execute('''
                        UPDATE files 
                        SET status = 'complete', saved_path = ?, last_update = CURRENT_TIMESTAMP
                        WHERE file_id = ?
                    ''', (str(final_path), file_id))
                    
                    # Update sender files count
                    db_conn.execute('''
                        UPDATE senders 
                        SET files_received = files_received + 1, last_activity = CURRENT_TIMESTAMP
                        WHERE sender_id = ?
                    ''', (sender_id,))
            
            # Clean up working directory
            if self.args.cleanup_temp:
                self.cleanup_working_directory(work_dir)
            
            # Check if reconstruction should be triggered
            self.schedule_reconstruction_if_needed(sender)
            
            # Send FILE_COMPLETE response
            complete_response = self.create_file_complete_packet(
                file_id=file_id,
                file_md5=actual_md5,
                saved_path=str(final_path),
                sender_id=sender_id,
                status='OK'
            )
            self.send_response(conn, complete_response)
            
        except Exception as e:
            self.logger.error(f"[{sender_id}] Error assembling file {file_id}: {e}")
            
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
                sender_id=sender_id,
                status='ERROR'
            )
            self.send_response(conn, error_response)
    
    def create_file_complete_packet(self, file_id: str, file_md5: str, saved_path: str, 
                                  sender_id: str, status: str = 'OK') -> str:
        """Create FILE_COMPLETE packet with sender info."""
        lines = [
            f"TYPE: FILE_COMPLETE",
            f"FILE_ID: {file_id}",
            f"SENDER_ID: {sender_id}",
            f"FILE_MD5: {file_md5}",
            f"STATUS: {status}",
            f"SAVED_PATH: {saved_path}",
            f"COMPLETED_AT: {datetime.now(timezone.utc).isoformat()}"
        ]
        
        return '\n'.join(lines) + '\n'
    
    def schedule_reconstruction_if_needed(self, sender: SenderState):
        """Schedule reconstruction for sender if conditions are met."""
        if not sender.should_trigger_reconstruction():
            return
        
        # Cancel existing timer for this sender
        if sender.sender_id in self.reconstruction_timers:
            self.reconstruction_timers[sender.sender_id].cancel()
        
        # Schedule new reconstruction with delay to batch multiple files
        def trigger_reconstruction():
            self.trigger_reconstruction(sender)
        
        timer = threading.Timer(self.args.reconstruction_delay, trigger_reconstruction)
        timer.start()
        self.reconstruction_timers[sender.sender_id] = timer
        
        self.logger.info(f"[{sender.sender_id}] Scheduled reconstruction in {self.args.reconstruction_delay}s "
                        f"({len(sender.pending_objects)} pending objects)")
    
    def trigger_reconstruction(self, sender: SenderState):
        """Trigger Phase 2 reconstruction for a specific sender."""
        with sender.reconstruction_lock:
            try:
                if not sender.should_trigger_reconstruction():
                    self.logger.debug(f"[{sender.sender_id}] Reconstruction conditions not met, skipping")
                    return
                
                self.logger.info(f"[{sender.sender_id}] Triggering reconstruction for {len(sender.pending_objects)} new objects")
                
                # Prepare reconstruction arguments with proper boolean formatting
                reconstruction_args = sender.get_reconstruction_args(debug_mode=self.args.debug_reconstruction)
                
                # Submit reconstruction task to thread pool
                future = self.reconstruction_executor.submit(
                    self.run_reconstruction_process, 
                    sender.sender_id, 
                    reconstruction_args
                )
                
                # Clear pending objects
                pending_count = len(sender.pending_objects)
                sender.clear_pending_objects()
                sender.last_reconstruction_time = time.time()
                
                # Update database
                with self.db_lock:
                    with sqlite3.connect(self.db_path) as db_conn:
                        db_conn.execute('''
                            UPDATE senders 
                            SET last_reconstruction = CURRENT_TIMESTAMP
                            WHERE sender_id = ?
                        ''', (sender.sender_id,))
                
                self.logger.info(f"[{sender.sender_id}] Reconstruction task submitted ({pending_count} objects)")
                
            except Exception as e:
                self.logger.error(f"[{sender.sender_id}] Error triggering reconstruction: {e}")
    
    def run_reconstruction_process(self, sender_id: str, args: List[str]) -> bool:
        """Run the reconstruction process as a subprocess."""
        try:
            self.logger.info(f"[{sender_id}] Starting reconstruction subprocess")
            self.logger.debug(f"[{sender_id}] Reconstruction command: {' '.join(args)}")
            
            # Start subprocess
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.args.reconstruction_timeout)
                
                if process.returncode == 0:
                    self.logger.info(f"[{sender_id}] Reconstruction completed successfully")
                    if stdout.strip():
                        self.logger.debug(f"[{sender_id}] Reconstruction output: {stdout.strip()}")
                    return True
                else:
                    self.logger.error(f"[{sender_id}] Reconstruction failed with return code {process.returncode}")
                    if stderr.strip():
                        self.logger.error(f"[{sender_id}] Reconstruction error: {stderr.strip()}")
                    if stdout.strip():
                        self.logger.error(f"[{sender_id}] Reconstruction stdout: {stdout.strip()}")
                    return False
                    
            except subprocess.TimeoutExpired:
                self.logger.error(f"[{sender_id}] Reconstruction timed out after {self.args.reconstruction_timeout}s")
                process.kill()
                return False
                
        except Exception as e:
            self.logger.error(f"[{sender_id}] Error running reconstruction process: {e}")
            return False
    
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
        """Handle individual client connection with enhanced logging."""
        self.logger.info(f"New connection from {addr}")
        
        try:
            while self.running:
                # Receive packet length
                length_bytes = conn.recv(4)
                if len(length_bytes) != 4:
                    break
                
                packet_length = int.from_bytes(length_bytes, 'big')
                
                if packet_length > MAX_PACKET_SIZE * 4:  # Safety check
                    self.logger.warning(f"Packet too large from {addr}: {packet_length} bytes")
                    break
                
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
                    sender_id = parsed['headers'].get('SENDER_ID', f'unknown_{addr[0]}')
                    
                    if packet_type == 'FILE_INIT':
                        self.handle_file_init(conn, parsed)
                    elif packet_type in ['METADATA', 'METADATA_PART']:
                        self.handle_metadata(conn, parsed)
                    elif packet_type == 'PART':
                        self.handle_part(conn, parsed)
                    else:
                        self.logger.warning(f"[{sender_id}] Unknown packet type: {packet_type}")
                        
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
        """Run the TCP server with connection pooling."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Connection pool for handling multiple clients
        with ThreadPoolExecutor(max_workers=self.args.max_connections) as executor:
            try:
                server_socket.bind((self.args.listen_host, self.args.listen_port))
                server_socket.listen(self.args.max_connections)
                
                self.logger.info(f"Multi-Sender server listening on {self.args.listen_host}:{self.args.listen_port}")
                self.logger.info(f"Ready to handle {self.args.max_connections} concurrent connections")
                
                while self.running:
                    try:
                        conn, addr = server_socket.accept()
                        
                        # Submit connection to thread pool
                        executor.submit(self.handle_client_connection, conn, addr)
                        
                    except socket.error as e:
                        if self.running:
                            self.logger.error(f"Socket error: {e}")
                            
            except Exception as e:
                self.logger.error(f"Server error: {e}")
            finally:
                server_socket.close()
    
    def get_sender_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all senders."""
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT sender_id, first_seen, last_activity, files_received, 
                           initial_scene_received, last_reconstruction
                    FROM senders
                ''')
                
                for row in cursor.fetchall():
                    sender_id, first_seen, last_activity, files_received, scene_received, last_recon = row
                    
                    # Get pending objects count
                    pending_count = 0
                    if sender_id in self.senders:
                        pending_count = len(self.senders[sender_id].pending_objects)
                    
                    stats[sender_id] = {
                        'first_seen': first_seen,
                        'last_activity': last_activity,
                        'files_received': files_received,
                        'initial_scene_received': bool(scene_received),
                        'last_reconstruction': last_recon,
                        'pending_objects': pending_count
                    }
        
        except Exception as e:
            self.logger.error(f"Error getting sender statistics: {e}")
        
        return stats
    
    def print_status_report(self):
        """Print status report for all senders."""
        stats = self.get_sender_statistics()
        
        print("\n" + "="*60)
        print("MULTI-SENDER RECEIVER STATUS")
        print("="*60)
        print(f"Active Senders: {len(stats)}")
        print(f"Total Active Connections: {len([s for s in self.senders.values()])}")
        print()
        
        for sender_id, sender_stats in stats.items():
            print(f"Sender: {sender_id}")
            print(f"  Files Received: {sender_stats['files_received']}")
            print(f"  Scene Received: {sender_stats['initial_scene_received']}")
            print(f"  Pending Objects: {sender_stats['pending_objects']}")
            print(f"  Last Activity: {sender_stats['last_activity']}")
            print(f"  Last Reconstruction: {sender_stats['last_reconstruction'] or 'Never'}")
            print()
    
    def run(self):
        """Run the multi-sender receiver daemon."""
        self.logger.info("Starting Multi-Sender Phase 2 Receiver daemon")
        
        # Print initial status
        self.print_status_report()
        
        try:
            self.run_server()
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            self.running = False
            
            # Cancel all pending reconstruction timers
            for timer in self.reconstruction_timers.values():
                timer.cancel()
            
            # Shutdown reconstruction executor
            self.reconstruction_executor.shutdown(wait=True)
            
            self.logger.info("Multi-Sender Phase 2 Receiver daemon stopped")


def main():
    parser = argparse.ArgumentParser(description="Multi-Sender Phase 2 Receiver - Handle multiple anomaly detection senders")
    
    # Network options
    parser.add_argument('--listen-host', default='0.0.0.0',
                       help='Listen host address (default: 0.0.0.0)')
    parser.add_argument('--listen-port', type=int, default=5001,
                       help='Listen port (default: 5001)')
    parser.add_argument('--max-connections', type=int, default=10,
                       help='Maximum concurrent connections (default: 10)')
    
    # Directory options
    parser.add_argument('--received-data-dir', default='received_data',
                       help='Base directory for received data organized by sender (default: received_data)')
    parser.add_argument('--receive-queue', default='receive_queue',
                       help='Temporary receive queue directory (default: receive_queue)')
    
    # Reconstruction options
    parser.add_argument('--max-reconstruction-workers', type=int, default=3,
                       help='Maximum concurrent reconstruction processes (default: 3)')
    parser.add_argument('--reconstruction-delay', type=float, default=5.0,
                       help='Delay before triggering reconstruction to batch files (default: 5.0s)')
    parser.add_argument('--reconstruction-timeout', type=int, default=300,
                       help='Reconstruction process timeout in seconds (default: 300)')
    parser.add_argument('--debug-reconstruction', action='store_true',
                       help='Enable debug mode for reconstruction processes')
    
    # Storage options
    parser.add_argument('--db-path', default='multi_sender_receiver.db',
                       help='SQLite database path (default: multi_sender_receiver.db)')
    parser.add_argument('--processed-json', default='output/global_processed_images.json',
                       help='Global processed images JSON file (default: output/global_processed_images.json)')
    parser.add_argument('--cleanup-temp', action='store_true',
                       help='Clean up temporary files after completion')
    
    # Logging options
    parser.add_argument('--log-file',
                       help='Log file path (default: stdout)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    # Status reporting
    parser.add_argument('--status-interval', type=int, default=300,
                       help='Status report interval in seconds (default: 300, 0 to disable)')
    
    args = parser.parse_args()
    
    try:
        receiver = MultiSenderPhase2Receiver(args)
        
        # Optional status reporting thread
        if args.status_interval > 0:
            def status_reporter():
                while receiver.running:
                    time.sleep(args.status_interval)
                    if receiver.running:
                        receiver.print_status_report()
            
            status_thread = threading.Thread(target=status_reporter, daemon=True)
            status_thread.start()
        
        receiver.run()
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
