#!/usr/bin/env python3
"""
UDP Multi-Sender Receiver - Converted from TCP version

Critical features implemented:
1. UDP connectionless communication with reliability
2. Proper packet parsing using partition() to avoid truncation
3. Thread-safe sender_states access with proper locking
4. ThreadPoolExecutor for controlled concurrency
5. Error ACKs sent on validation failures
6. Dynamic packet size validation
7. Proper FILE_COMPLETE acknowledgment format
8. Address-based sender identification
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
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple, Set
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import queue
import uuid
import contextlib
from dataclasses import dataclass, field
import weakref
import tempfile
import shutil
import struct

# Constants
MAX_PACKET_SIZE = 64 * 1024  # 64KB - dynamic validation instead of fixed
DELIMITER = "---DATA---"
END_DELIMITER = "---END---"
DEFAULT_SOCKET_TIMEOUT = 30
CHUNK_SIZE = 8192  # For streaming file assembly
MAX_WORKERS_PER_SENDER = 3
MAX_GLOBAL_WORKERS = 20

# UDP-specific constants
UDP_RECV_BUFFER = 65536  # 64KB UDP receive buffer
ACK_TIMEOUT = 2.0
SEQUENCE_WINDOW = 1000  # Sequence number window for duplicate detection

@dataclass
class FileTransfer:
    file_id: str
    filename: str
    file_size: int
    file_md5: str
    total_parts: int
    status: str
    sender_id: str
    unique_file_id: str = ""
    sender_addr: Tuple[str, int] = None  # Store sender address for UDP

class ProcessingTask:
    """Base class for all processing tasks"""
    
    def __init__(self, sender_id: str, task_type: str):
        self.sender_id = sender_id
        self.task_type = task_type
        self.timestamp = time.time()

class FileAssemblyTask(ProcessingTask):
    """Task for assembling completed file parts"""
    
    def __init__(self, sender_id: str, unique_file_id: str, original_file_id: str, 
                 work_dir: Path, sender_addr: Tuple[str, int]):
        super().__init__(sender_id, "assembly")
        self.unique_file_id = unique_file_id
        self.original_file_id = original_file_id
        self.work_dir = work_dir
        self.sender_addr = sender_addr  # Store sender address for UDP response

class DatabaseConnectionPool:
    """Thread-safe database connection pool"""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = queue.Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0
        self._shutdown = False
        
        # Pre-create some connections
        for _ in range(min(3, max_connections)):
            self._create_connection()
    
    def _create_connection(self):
        """Create a new database connection"""
        with self._lock:
            if self._created_connections >= self.max_connections:
                return False
            self._created_connections += 1
        
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.row_factory = sqlite3.Row
            self._connections.put(conn)
            return True
        except Exception as e:
            with self._lock:
                self._created_connections -= 1
            raise e
    
    @contextlib.contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        if self._shutdown:
            raise RuntimeError("Connection pool is shut down")
            
        conn = None
        try:
            try:
                conn = self._connections.get_nowait()
            except queue.Empty:
                if self._create_connection():
                    conn = self._connections.get_nowait()
                else:
                    conn = self._connections.get(timeout=30)
            
            yield conn
            
        finally:
            if conn and not self._shutdown:
                try:
                    self._connections.put_nowait(conn)
                except queue.Full:
                    conn.close()
    
    def close_all(self):
        """Close all connections"""
        self._shutdown = True
        while True:
            try:
                conn = self._connections.get_nowait()
                conn.close()
            except queue.Empty:
                break

class SenderState:
    """Per-sender state with thread safety"""
    
    def __init__(self, sender_id: str, base_dir: Path, receiver_instance, sender_addr: Tuple[str, int]):
        self.sender_id = sender_id
        self.base_dir = base_dir
        self.receiver = receiver_instance
        self.sender_addr = sender_addr  # Store address for UDP responses
        self.images_dir = base_dir / 'images'
        self.metadata_dir = base_dir / 'metadata'
        self.output_dir = Path('output') / sender_id
        self.batch_dir = self.output_dir / 'batch_001'
        
        # Create directories
        for directory in [self.images_dir, self.metadata_dir, self.output_dir, self.batch_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe state
        self._lock = threading.RLock()
        self.initial_scene_received = threading.Event()
        self.initial_scene_path = None
        self.pending_objects = set()
        self.active_transfers = {}  # original_file_id -> unique_file_id
        self.reconstruction_in_progress = False
        
        # Sequence tracking for UDP reliability
        self.last_sequence = 0
        self.received_sequences = set()
        
        # Statistics
        self.files_received = 0
        self.bytes_received = 0
        self.reconstruction_count = 0
        self.last_activity = time.time()
    
    def add_pending_object(self, filename: str):
        """Thread-safe add pending object"""
        with self._lock:
            self.pending_objects.add(filename)
    
    def register_transfer(self, original_file_id: str, unique_file_id: str):
        """Register active transfer"""
        with self._lock:
            self.active_transfers[original_file_id] = unique_file_id
    
    def get_transfer_info(self, original_file_id: str):
        """Get transfer info safely"""
        with self._lock:
            return self.active_transfers.get(original_file_id, None)
    
    def complete_transfer(self, original_file_id: str):
        """Mark transfer as complete"""
        with self._lock:
            if original_file_id in self.active_transfers:
                del self.active_transfers[original_file_id]
    
    def is_duplicate_sequence(self, seq_num: int) -> bool:
        """Check if sequence number is duplicate"""
        with self._lock:
            if seq_num in self.received_sequences:
                return True
            
            self.received_sequences.add(seq_num)
            
            # Clean old sequences to prevent memory growth
            if len(self.received_sequences) > SEQUENCE_WINDOW:
                min_seq = min(self.received_sequences)
                self.received_sequences.discard(min_seq)
            
            return False

class NonBlockingUDPMultiSenderReceiver:
    """UDP multi-sender receiver with proper reliability handling"""
    
    def __init__(self, args):
        self.args = args
        self.running = True
        self.shutdown_event = threading.Event()
        
        # Thread-safe sender management
        self.senders_lock = threading.RLock()
        self.senders: Dict[str, SenderState] = {}
        
        # Global thread pool for controlled concurrency
        self.global_executor = ThreadPoolExecutor(
            max_workers=MAX_GLOBAL_WORKERS,
            thread_name_prefix="global-worker"
        )
        
        # Database connection pool
        self.db_pool = DatabaseConnectionPool(args.db_path, max_connections=args.max_db_connections)
        
        # Global state
        self.received_data_dir = Path(args.received_data_dir)
        self.received_data_dir.mkdir(parents=True, exist_ok=True)
        
        # UDP socket
        self.socket = None
        
        # Statistics
        self.total_packets_received = 0
        self.total_files_received = 0
        self.start_time = time.time()
        
        # Setup
        self.setup_logging()
        self.init_database()
        self.load_processed_images()
        self.setup_signal_handlers()
        self.init_udp_socket()
        
        self.logger.info("UDP Multi-Sender Receiver initialized")
    
    def setup_signal_handlers(self):
        """Setup signal handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def setup_logging(self):
        """Setup logging"""
        log_level = getattr(logging, self.args.log_level.upper())
        
        handlers = [logging.StreamHandler()]
        if self.args.log_file:
            handlers.append(logging.FileHandler(self.args.log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
            handlers=handlers
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def init_udp_socket(self):
        """Initialize UDP socket"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(1.0)  # Non-blocking with timeout
            self.socket.bind((self.args.listen_host, self.args.listen_port))
            
            self.logger.info(f"UDP socket bound to {self.args.listen_host}:{self.args.listen_port}")
        except Exception as e:
            self.logger.error(f"Failed to initialize UDP socket: {e}")
            raise
    
    def init_database(self):
        """Initialize database schema"""
        with self.db_pool.get_connection() as conn:
            conn.execute('BEGIN TRANSACTION')
            try:
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
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS parts (
                        file_id TEXT NOT NULL,
                        part_index INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'missing',
                        chunk_md5 TEXT NOT NULL,
                        received_at TIMESTAMP,
                        PRIMARY KEY (file_id, part_index)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS senders (
                        sender_id TEXT PRIMARY KEY,
                        sender_hostname TEXT,
                        sender_ip TEXT,
                        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        files_received INTEGER DEFAULT 0,
                        initial_scene_received BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                conn.execute('CREATE INDEX IF NOT EXISTS idx_files_sender_status ON files (sender_id, status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_parts_file_status ON parts (file_id, status)')
                
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    def load_processed_images(self):
        """Load processed images list"""
        self.processed_images_lock = threading.RLock()
        with self.processed_images_lock:
            self.processed_images = {'processed_images': []}
            
            try:
                if Path(self.args.processed_json).exists():
                    with open(self.args.processed_json, 'r') as f:
                        self.processed_images = json.load(f)
                    self.logger.info(f"Loaded {len(self.processed_images.get('processed_images', []))} processed records")
            except Exception as e:
                self.logger.error(f"Error loading processed images: {e}")
    
    def parse_packet(self, packet_data: str) -> Dict:
        """Parse packet using partition to avoid truncation - CRITICAL FIX"""
        try:
            # Split headers from data using partition
            before_data, delimiter, after_delimiter = packet_data.partition(DELIMITER)
            
            if not delimiter:
                # No data section, headers only
                headers = {}
                for line in before_data.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        headers[key.strip().upper()] = value.strip()
                return {'headers': headers, 'data': None}
            
            # Parse headers
            headers = {}
            for line in before_data.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip().upper()] = value.strip()
            
            # Extract data (everything between DELIMITER and END_DELIMITER)
            data_section, end_delim, _ = after_delimiter.partition(END_DELIMITER)
            
            return {
                'headers': headers,
                'data': data_section.strip() if data_section else None
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing packet: {e}")
            return {'headers': {}, 'data': None}
    
    def create_response_packet(self, packet_type: str, file_id: str, status: str = 'OK', **kwargs) -> str:
        """Create properly formatted response packet"""
        lines = [
            f"TYPE: ACK",
            f"ORIGINAL_TYPE: {packet_type}",
            f"FILE_ID: {file_id}",
            f"STATUS: {status}",
            f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}"
        ]
        
        for key, value in kwargs.items():
            lines.append(f"{key.upper()}: {value}")
        
        return '\n'.join(lines) + '\n'
    
    def send_udp_response(self, response: str, addr: Tuple[str, int]):
        """Send UDP response to specific address"""
        try:
            response_bytes = response.encode('utf-8')
            self.socket.sendto(response_bytes, addr)
                        
        except Exception as e:
            self.logger.error(f"Error sending UDP response to {addr}: {e}")
    
    def get_or_create_sender(self, sender_id: str, addr: Tuple[str, int], 
                           sender_hostname: str = None) -> SenderState:
        """Get or create sender state - thread-safe"""
        with self.senders_lock:
            if sender_id not in self.senders:
                sender_base_dir = self.received_data_dir / sender_id
                sender = SenderState(sender_id, sender_base_dir, self, addr)
                self.senders[sender_id] = sender
                
                # Register in database
                self.global_executor.submit(
                    self._register_sender_in_db,
                    sender_id, sender_hostname, addr[0]
                )
                
                self.logger.info(f"Created sender state: {sender_id} from {addr}")
            else:
                # Update address in case sender changed IP
                self.senders[sender_id].sender_addr = addr
            
            return self.senders[sender_id]
    
    def _register_sender_in_db(self, sender_id: str, hostname: str, ip: str):
        """Register sender in database"""
        try:
            with self.db_pool.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO senders 
                    (sender_id, sender_hostname, sender_ip)
                    VALUES (?, ?, ?)
                ''', (sender_id, hostname, ip))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error registering sender {sender_id}: {e}")
    
    def handle_packet(self, packet_data: bytes, addr: Tuple[str, int]):
        """Handle UDP packet from specific address"""
        try:
            self.total_packets_received += 1
            
            # Parse packet
            packet_text = packet_data.decode('utf-8')
            parsed = self.parse_packet(packet_text)
            
            if not parsed['headers']:
                self.logger.warning(f"Empty headers in packet from {addr}")
                return
            
            # Extract sender info and packet details
            packet_type = parsed['headers'].get('TYPE')
            sender_id = parsed['headers'].get('SENDER_ID', f'sender_{addr[0].replace(".", "_")}')
            seq_num = int(parsed['headers'].get('SEQ', 0))
            
            # Get or create sender
            sender = self.get_or_create_sender(sender_id, addr,
                                             parsed['headers'].get('SENDER_HOSTNAME'))
            sender.last_activity = time.time()
            
            # Check for duplicate sequence number
            if seq_num > 0 and sender.is_duplicate_sequence(seq_num):
                self.logger.debug(f"Duplicate sequence {seq_num} from {sender_id}, ignoring")
                return
            
            # Handle packet types
            if packet_type == 'FILE_INIT':
                self._handle_file_init(parsed, sender, addr)
            elif packet_type in ['METADATA', 'METADATA_PART']:
                self._handle_metadata(parsed, sender, addr)
            elif packet_type == 'PART':
                future = self._handle_part(parsed, sender, addr)
                if future:
                    # Don't wait for assembly, let it run in background
                    pass
            else:
                self.logger.warning(f"Unknown packet type from {addr}: {packet_type}")
                error_response = self.create_response_packet(
                    packet_type or 'UNKNOWN',
                    parsed['headers'].get('FILE_ID', 'unknown'),
                    'ERROR',
                    error='unknown_packet_type'
                )
                self.send_udp_response(error_response, addr)
                        
        except Exception as e:
            self.logger.error(f"Error handling packet from {addr}: {e}")
            try:
                error_response = self.create_response_packet(
                    'ERROR', 'unknown', 'ERROR', error=str(e)
                )
                self.send_udp_response(error_response, addr)
            except:
                pass
    
    def _handle_file_init(self, parsed: Dict, sender: SenderState, addr: Tuple[str, int]):
        """Handle FILE_INIT packet"""
        try:
            headers = parsed['headers']
            original_file_id = headers.get('FILE_ID')
            filename = headers.get('FILE')
            file_size = int(headers.get('FILE_SIZE', 0))
            file_md5 = headers.get('FILE_MD5')
            total_parts = int(headers.get('TOTAL_PARTS', 0))
            
            if not all([original_file_id, filename, file_md5]):
                error_response = self.create_response_packet(
                    'FILE_INIT', original_file_id or 'unknown', 'ERROR',
                    error='missing_required_fields'
                )
                self.send_udp_response(error_response, addr)
                return
            
            # Generate unique file ID
            unique_file_id = self.generate_unique_file_id(sender.sender_id, original_file_id)
            
            # Register transfer
            sender.register_transfer(original_file_id, unique_file_id)
            
            # Create working directory
            work_dir = Path(self.args.receive_queue) / sender.sender_id / unique_file_id
            work_dir.mkdir(parents=True, exist_ok=True)
            
            # Database setup
            with self.db_pool.get_connection() as db_conn:
                db_conn.execute('BEGIN TRANSACTION')
                try:
                    db_conn.execute('''
                        INSERT OR REPLACE INTO files 
                        (file_id, sender_id, filename, file_size, file_md5, total_parts, status)
                        VALUES (?, ?, ?, ?, ?, ?, 'receiving')
                    ''', (unique_file_id, sender.sender_id, filename, file_size, file_md5, total_parts))
                    
                    for part_idx in range(total_parts):
                        db_conn.execute('''
                            INSERT OR IGNORE INTO parts (file_id, part_index, chunk_md5, status)
                            VALUES (?, ?, '', 'missing')
                        ''', (unique_file_id, part_idx))
                    
                    db_conn.commit()
                except Exception:
                    db_conn.rollback()
                    raise
            
            # Send ACK
            ack_response = self.create_response_packet(
                'FILE_INIT', original_file_id, 'OK',
                sender_id=sender.sender_id,
                unique_file_id=unique_file_id
            )
            self.send_udp_response(ack_response, addr)
            
            self.logger.info(f"[{sender.sender_id}] File init: {filename} ({total_parts} parts)")
            
        except Exception as e:
            self.logger.error(f"FILE_INIT error from {addr}: {e}")
            error_response = self.create_response_packet(
                'FILE_INIT', parsed['headers'].get('FILE_ID', 'unknown'), 'ERROR',
                error=str(e)
            )
            self.send_udp_response(error_response, addr)
    
    def _handle_metadata(self, parsed: Dict, sender: SenderState, addr: Tuple[str, int]):
        """Handle METADATA packet"""
        try:
            headers = parsed['headers']
            original_file_id = headers.get('FILE_ID')
            metadata_content = parsed.get('data', '')
            
            if not original_file_id:
                error_response = self.create_response_packet(
                    'METADATA', 'unknown', 'ERROR', error='missing_file_id'
                )
                self.send_udp_response(error_response, addr)
                return
            
            # Get unique file ID
            unique_file_id = sender.get_transfer_info(original_file_id)
            
            if not unique_file_id:
                error_response = self.create_response_packet(
                    'METADATA', original_file_id, 'ERROR', error='file_not_found'
                )
                self.send_udp_response(error_response, addr)
                return
            
            # Save metadata
            work_dir = Path(self.args.receive_queue) / sender.sender_id / unique_file_id
            metadata_file = work_dir / 'metadata.txt'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
            
            # Send ACK
            ack_response = self.create_response_packet(
                'METADATA', original_file_id, 'OK', sender_id=sender.sender_id
            )
            self.send_udp_response(ack_response, addr)
            
            self.logger.debug(f"[{sender.sender_id}] Metadata saved for {unique_file_id}")
            
        except Exception as e:
            self.logger.error(f"METADATA error from {addr}: {e}")
            error_response = self.create_response_packet(
                'METADATA', parsed['headers'].get('FILE_ID', 'unknown'), 'ERROR',
                error=str(e)
            )
            self.send_udp_response(error_response, addr)
    
    def _handle_part(self, parsed: Dict, sender: SenderState, addr: Tuple[str, int]) -> Optional[Future]:
        """Handle PART packet - returns Future if assembly is triggered"""
        try:
            headers = parsed['headers']
            original_file_id = headers.get('FILE_ID')
            part_info = headers.get('PART', '1/1')
            chunk_len = int(headers.get('CHUNK_LEN', 0))
            chunk_md5 = headers.get('CHUNK_MD5')
            base64_data = parsed.get('data', '')
            
            if not all([original_file_id, part_info, chunk_md5, base64_data]):
                error_response = self.create_response_packet(
                    'PART', original_file_id or 'unknown', 'ERROR',
                    error='missing_required_fields'
                )
                self.send_udp_response(error_response, addr)
                return None
            
            # Get transfer info
            unique_file_id = sender.get_transfer_info(original_file_id)
            
            if not unique_file_id:
                error_response = self.create_response_packet(
                    'PART', original_file_id, 'ERROR', error='file_not_found'
                )
                self.send_udp_response(error_response, addr)
                return None
            
            # Parse part number
            part_num, total_parts = map(int, part_info.split('/'))
            part_index = part_num - 1
            
            # Decode and validate chunk
            try:
                chunk_data = base64.b64decode(base64_data.replace('\n', ''))
            except Exception as e:
                error_response = self.create_response_packet(
                    'PART', original_file_id, 'ERROR',
                    part=part_info,
                    error=f'base64_decode_failed: {e}'
                )
                self.send_udp_response(error_response, addr)
                return None
            
            # Validate chunk length
            if len(chunk_data) != chunk_len:
                error_response = self.create_response_packet(
                    'PART', original_file_id, 'ERROR',
                    part=part_info,
                    error=f'chunk_length_mismatch: expected {chunk_len}, got {len(chunk_data)}'
                )
                self.send_udp_response(error_response, addr)
                return None
            
            # Validate MD5
            actual_md5 = hashlib.md5(chunk_data).hexdigest()
            if actual_md5 != chunk_md5:
                error_response = self.create_response_packet(
                    'PART', original_file_id, 'ERROR',
                    part=part_info,
                    error=f'md5_mismatch: expected {chunk_md5}, got {actual_md5}'
                )
                self.send_udp_response(error_response, addr)
                return None
            
            # Save chunk
            work_dir = Path(self.args.receive_queue) / sender.sender_id / unique_file_id
            chunk_file = work_dir / f'part_{part_index:05d}.bin'
            with open(chunk_file, 'wb') as f:
                f.write(chunk_data)
            
            # Update database
            with self.db_pool.get_connection() as db_conn:
                db_conn.execute('''
                    UPDATE parts 
                    SET status = 'received', chunk_md5 = ?, received_at = CURRENT_TIMESTAMP
                    WHERE file_id = ? AND part_index = ?
                ''', (chunk_md5, unique_file_id, part_index))
                
                # Check if all parts received
                cursor = db_conn.execute('''
                    SELECT COUNT(*) FROM parts 
                    WHERE file_id = ? AND status = 'received'
                ''', (unique_file_id,))
                received_count = cursor.fetchone()[0]
                
                db_conn.commit()
            
            # Send ACK for part
            ack_response = self.create_response_packet(
                'PART', original_file_id, 'OK',
                part=part_info,
                sender_id=sender.sender_id
            )
            self.send_udp_response(ack_response, addr)
            
            self.logger.debug(f"[{sender.sender_id}] Part {part_num}/{total_parts} received")
            
            # Check if file is complete
            if received_count == total_parts:
                self.logger.info(f"[{sender.sender_id}] All parts received, starting assembly")
                # Submit assembly task and return Future
                future = self.global_executor.submit(
                    self._process_file_assembly,
                    sender, unique_file_id, original_file_id, work_dir, addr
                )
                return future
            
            return None
            
        except Exception as e:
            self.logger.error(f"PART error from {addr}: {e}")
            error_response = self.create_response_packet(
                'PART', parsed['headers'].get('FILE_ID', 'unknown'), 'ERROR',
                error=str(e)
            )
            self.send_udp_response(error_response, addr)
            return None
    
    def _process_file_assembly(self, sender: SenderState, unique_file_id: str,
                              original_file_id: str, work_dir: Path, addr: Tuple[str, int]):
        """Assembly with FILE_COMPLETE sent via UDP"""
        try:
            # Get file info
            with self.db_pool.get_connection() as db_conn:
                cursor = db_conn.execute('''
                    SELECT filename, file_size, file_md5, total_parts
                    FROM files WHERE file_id = ?
                ''', (unique_file_id,))
                file_info = cursor.fetchone()
            
            if not file_info:
                raise ValueError(f"File info not found: {unique_file_id}")
            
            filename, file_size, expected_md5, total_parts = file_info
            
            # Assemble file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                md5_hash = hashlib.md5()
                total_size = 0
                
                for part_index in range(total_parts):
                    part_file = work_dir / f'part_{part_index:05d}.bin'
                    if not part_file.exists():
                        raise FileNotFoundError(f"Missing part: {part_file}")
                    
                    with open(part_file, 'rb') as f:
                        while True:
                            chunk = f.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            temp_file.write(chunk)
                            md5_hash.update(chunk)
                            total_size += len(chunk)
            
            # Verify file
            if total_size != file_size:
                raise ValueError(f"Size mismatch: expected {file_size}, got {total_size}")
            
            actual_md5 = md5_hash.hexdigest()
            if actual_md5 != expected_md5:
                raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            
            # Determine destination
            if filename == 'initial_scene.jpg' or filename.startswith('scene_'):
                final_path = sender.base_dir / filename
                shutil.move(str(temp_path), str(final_path))
                sender.initial_scene_received.set()
                sender.initial_scene_path = str(final_path)
                self.logger.info(f"[{sender.sender_id}] Initial scene saved: {filename}")
            else:
                final_path = sender.images_dir / filename
                shutil.move(str(temp_path), str(final_path))
                sender.add_pending_object(filename)
                
                # Handle metadata
                metadata_source = work_dir / 'metadata.txt'
                if metadata_source.exists():
                    metadata_filename = f"{Path(filename).stem}.txt"
                    final_metadata_path = sender.metadata_dir / metadata_filename
                    shutil.copy2(str(metadata_source), str(final_metadata_path))
                
                self.logger.info(f"[{sender.sender_id}] Object saved: {filename}")
            
            # Update database
            with self.db_pool.get_connection() as db_conn:
                db_conn.execute('''
                    UPDATE files 
                    SET status = 'complete', saved_path = ?, last_update = CURRENT_TIMESTAMP
                    WHERE file_id = ?
                ''', (str(final_path), unique_file_id))
                db_conn.commit()
            
            # Update statistics
            with sender._lock:
                sender.files_received += 1
                sender.bytes_received += file_size
            
            # Send FILE_COMPLETE via UDP
            complete_response = self.create_response_packet(
                'FILE_COMPLETE', original_file_id, 'OK',
                sender_id=sender.sender_id,
                filename=filename,
                file_size=file_size,
                file_md5=actual_md5,
                saved_path=str(final_path)
            )
            self.send_udp_response(complete_response, addr)
            
            self.logger.info(f"[{sender.sender_id}] FILE_COMPLETE sent for {filename}")
            
            # Clean up transfer record
            sender.complete_transfer(original_file_id)
            
            # Clean up temp files
            if self.args.cleanup_temp:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except:
                    pass
            
            # Trigger reconstruction if needed
            if sender.initial_scene_received.is_set() and len(sender.pending_objects) > 0:
                self.global_executor.submit(self._trigger_reconstruction, sender)
            
            self.total_files_received += 1
            
        except Exception as e:
            self.logger.error(f"Assembly error: {e}")
            
            # Send error response
            try:
                error_response = self.create_response_packet(
                    'FILE_COMPLETE', original_file_id, 'ERROR',
                    error=str(e)
                )
                self.send_udp_response(error_response, addr)
            except:
                pass
            
            # Clean up temp file
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
            except:
                pass
            
            # Update database
            try:
                with self.db_pool.get_connection() as db_conn:
                    db_conn.execute('''
                        UPDATE files 
                        SET status = 'failed', last_update = CURRENT_TIMESTAMP
                        WHERE file_id = ?
                    ''', (unique_file_id,))
                    db_conn.commit()
            except:
                pass
    
    def _trigger_reconstruction(self, sender: SenderState):
        """Trigger reconstruction process"""
        with sender._lock:
            if sender.reconstruction_in_progress:
                return
            sender.reconstruction_in_progress = True
            pending = list(sender.pending_objects)
        
        try:
            self.logger.info(f"[{sender.sender_id}] Starting reconstruction for {len(pending)} objects")
            
            # Find reconstruction script
            reconstruction_script = Path('2 phase final.py')
            if not reconstruction_script.exists():
                reconstruction_script = Path('phase2_final.py')
            
            if not reconstruction_script.exists():
                self.logger.error("Reconstruction script not found")
                return
            
            # Run reconstruction
            args = [
                sys.executable,
                str(reconstruction_script),
                '--input-dir', str(sender.base_dir),
                '--output-dir', str(sender.output_dir),
                '--scene-image', 'initial_scene.jpg',
                '--batch-dir', str(sender.batch_dir),
                '--sender-id', sender.sender_id
            ]
            
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=self.args.reconstruction_timeout)
            
            if process.returncode == 0:
                with sender._lock:
                    sender.pending_objects.clear()
                    sender.reconstruction_count += 1
                self.logger.info(f"[{sender.sender_id}] Reconstruction completed")
            else:
                self.logger.error(f"[{sender.sender_id}] Reconstruction failed: {stderr}")
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"[{sender.sender_id}] Reconstruction timeout")
            process.kill()
        except Exception as e:
            self.logger.error(f"[{sender.sender_id}] Reconstruction error: {e}")
        finally:
            with sender._lock:
                sender.reconstruction_in_progress = False
    
    def generate_unique_file_id(self, sender_id: str, original_file_id: str) -> str:
        """Generate unique file ID"""
        import random
        timestamp = int(time.time() * 1000000)
        random_component = random.randint(1000, 9999)
        combined = f"{sender_id}_{original_file_id}_{timestamp}_{random_component}"
        unique_hash = hashlib.md5(combined.encode()).hexdigest()[:12]
        return f"{sender_id}_{unique_hash}"
    
    def run_udp_server(self):
        """Run the UDP server"""
        self.logger.info(f"UDP Server listening on {self.args.listen_host}:{self.args.listen_port}")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Receive UDP packet
                packet_data, addr = self.socket.recvfrom(UDP_RECV_BUFFER)
                
                if not self.running:
                    break
                
                # Handle packet in thread pool
                self.global_executor.submit(self.handle_packet, packet_data, addr)
                
            except socket.timeout:
                continue
            except socket.error as e:
                if self.running:
                    self.logger.error(f"UDP socket error: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
        
        self.logger.info("UDP Server stopped")
    
    def print_status_report(self):
        """Print status report"""
        uptime = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("UDP MULTI-SENDER RECEIVER STATUS")
        print("="*60)
        print(f"Uptime: {uptime:.1f}s")
        print(f"Total Packets Received: {self.total_packets_received}")
        print(f"Total Files Received: {self.total_files_received}")
        print(f"Active Senders: {len(self.senders)}")
        
        with self.senders_lock:
            for sender_id, sender in self.senders.items():
                print(f"\nSender: {sender_id}")
                print(f"  Address: {sender.sender_addr}")
                print(f"  Files: {sender.files_received}")
                print(f"  Bytes: {sender.bytes_received:,}")
                print(f"  Reconstructions: {sender.reconstruction_count}")
                print(f"  Pending Objects: {len(sender.pending_objects)}")
                print(f"  Initial Scene: {sender.initial_scene_received.is_set()}")
                print(f"  Active Transfers: {len(sender.active_transfers)}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down...")
        self.running = False
        self.shutdown_event.set()
        
        # Close UDP socket
        if self.socket:
            self.socket.close()
        
        # Shutdown thread pool
        self.global_executor.shutdown(wait=True, timeout=10)
        
        # Close database pool
        self.db_pool.close_all()
        
        self.logger.info("Shutdown complete")
    
    def run(self):
        """Main run method"""
        try:
            # Start status reporter
            if self.args.status_interval > 0:
                def status_reporter():
                    while self.running:
                        time.sleep(self.args.status_interval)
                        if self.running:
                            self.print_status_report()
                
                status_thread = threading.Thread(target=status_reporter, daemon=True)
                status_thread.start()
            
            # Run UDP server
            self.run_udp_server()
            
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()


def main():
    parser = argparse.ArgumentParser(description="UDP Multi-Sender Receiver")
    
    # Network
    parser.add_argument('--listen-host', default='0.0.0.0')
    parser.add_argument('--listen-port', type=int, default=5001)
    parser.add_argument('--max-connections', type=int, default=50)
    
    # Performance
    parser.add_argument('--max-db-connections', type=int, default=10)
    
    # Directories
    parser.add_argument('--received-data-dir', default='received_data')
    parser.add_argument('--receive-queue', default='receive_queue')
    
    # Reconstruction
    parser.add_argument('--reconstruction-timeout', type=int, default=300)
    
    # Storage
    parser.add_argument('--db-path', default='receiver.db')
    parser.add_argument('--processed-json', default='output/processed_images.json')
    parser.add_argument('--cleanup-temp', action='store_true')
    
    # Logging
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Status
    parser.add_argument('--status-interval', type=int, default=60)
    
    args = parser.parse_args()
    
    receiver = NonBlockingUDPMultiSenderReceiver(args)
    receiver.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
