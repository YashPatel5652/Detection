import cv2
import numpy as np
import os
import glob
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set
from collections import defaultdict, namedtuple
import re
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Setup argument parser for Enhanced Phase 2
parser = argparse.ArgumentParser(description="Enhanced Phase 2: Multi-Sender Incremental Anomaly Scene Reconstructor")
parser.add_argument('--input-dir', type=str, default='captures', 
                   help='Input directory containing Phase 1 outputs (same as --save_dir from Phase 1)')
parser.add_argument('--output-dir', type=str, default='reconstructed_scenes',
                   help='Output directory for reconstructed scene images')
parser.add_argument('--scene-image', type=str, default='initial_scene.jpg',
                   help='Name of the initial scene image file')
parser.add_argument('--batch-dir', type=str, default=None,
                   help='Persistent batch directory for incremental processing')
parser.add_argument('--sender-id', type=str, default='default_sender',
                   help='Sender ID for multi-sender environments')
parser.add_argument('--min-object-size', type=int, default=20,
                   help='Minimum object size (width or height) to process')
parser.add_argument('--debug-mode', action='store_false',
                   help='Add debug overlays (bounding boxes, labels) to output images')
parser.add_argument('--batch-process', action='store_false',
                   help='Process objects into batch reconstruction scenes (grouped by object ID)')
parser.add_argument('--quality-filter', type=float, default=0.0,
                   help='Minimum quality score threshold (0.0-1.0)')
parser.add_argument('--save-reason-filter', type=str, choices=['all', 'disappeared', 'periodic', 'shutdown'], 
                   default='all', help='Filter objects by save reason')
parser.add_argument('--max-workers', type=int, default=2,
                   help='Number of worker threads for processing')
parser.add_argument('--preserve-timestamp-order', type=lambda x: x.lower() in ['true', '1', 'yes'], 
                   default=True, help='Process objects in timestamp order for temporal consistency')
parser.add_argument('--create-montage', type=lambda x: x.lower() in ['true', '1', 'yes'], 
                   default=True, help='Create montage views showing all anomalies together')
parser.add_argument('--only-new', type=lambda x: x.lower() in ['true', '1', 'yes'], 
                   default=True, help='Process only newly arrived images since last run (default: True)')
parser.add_argument('--persistent-mode', action='store_false',
                   help='Enable persistent incremental mode for multi-sender environments')
args = parser.parse_args()

# Enhanced data structures for Phase 1 compatibility
DetectedObject = namedtuple('DetectedObject', 
                          ['obj_id', 'label', 'bbox', 'confidence', 'quality_score', 
                           'save_reason', 'timestamp', 'total_frames', 'image_path', 'metadata_path'])

class IncrementalBatchTracker:
    """
    Tracks processed images in persistent batch directories for incremental processing.
    Maintains state across multiple reconstruction runs.
    """
    
    def __init__(self, batch_dir: Path, sender_id: str):
        self.batch_dir = batch_dir
        self.sender_id = sender_id
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate tracking for batch processing
        self.processed_file = self.batch_dir / 'processed_files.json'
        self.batch_state_file = self.batch_dir / 'batch_state.json'
        
        self.processed_files: Set[str] = set()
        self.batch_state: Dict = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{sender_id}")
        
        # Load existing state
        self.load_batch_state()
    
    def load_batch_state(self):
        """Load batch processing state from persistent files."""
        # Load processed files list
        try:
            if self.processed_file.exists():
                with open(self.processed_file, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed', []))
                    self.logger.info(f"Loaded {len(self.processed_files)} processed files from batch state")
        except Exception as e:
            self.logger.error(f"Error loading processed files: {e}")
            self.processed_files = set()
        
        # Load batch state
        try:
            if self.batch_state_file.exists():
                with open(self.batch_state_file, 'r') as f:
                    self.batch_state = json.load(f)
                    self.logger.info(f"Loaded batch state for sender {self.sender_id}")
            else:
                self.batch_state = {
                    'sender_id': self.sender_id,
                    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_runs': 0,
                    'total_objects_processed': 0,
                    'last_run': None,
                    'montage_version': 0,
                    'cumulative_object_count': 0
                }
        except Exception as e:
            self.logger.error(f"Error loading batch state: {e}")
            self.batch_state = {}
    
    def is_processed(self, image_filename: str) -> bool:
        """Check if an image has already been processed in this batch."""
        return image_filename in self.processed_files
    
    def mark_processed(self, image_filename: str):
        """Mark an image as processed and save state."""
        if image_filename not in self.processed_files:
            self.processed_files.add(image_filename)
            self.save_batch_state()
    
    def save_batch_state(self):
        """Save batch state to persistent files."""
        try:
            # Save processed files
            processed_data = {
                'processed': sorted(list(self.processed_files)),
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'sender_id': self.sender_id
            }
            
            with open(self.processed_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            # Save batch state
            self.batch_state['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.batch_state['total_runs'] += 1
            
            with open(self.batch_state_file, 'w') as f:
                json.dump(self.batch_state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving batch state: {e}")
    
    def update_batch_statistics(self, new_objects_count: int, total_objects_in_batch: int):
        """Update batch processing statistics."""
        self.batch_state['total_objects_processed'] += new_objects_count
        self.batch_state['cumulative_object_count'] = total_objects_in_batch
        self.save_batch_state()
    
    def get_batch_summary(self) -> Dict:
        """Get current batch summary."""
        return {
            'sender_id': self.sender_id,
            'batch_directory': str(self.batch_dir),
            'total_runs': self.batch_state.get('total_runs', 0),
            'total_processed_files': len(self.processed_files),
            'cumulative_objects': self.batch_state.get('cumulative_object_count', 0),
            'last_run': self.batch_state.get('last_run'),
            'montage_version': self.batch_state.get('montage_version', 0)
        }

class EnhancedProcessedImagesTracker:
    """
    Enhanced tracker that works with incremental batch processing.
    Maintains global state while supporting batch-specific tracking.
    """
    
    def __init__(self, output_dir: Path, sender_id: str = None):
        self.output_dir = output_dir
        self.sender_id = sender_id
        self.processed_file = output_dir / 'processed_images.json'
        self.processed_images: Set[str] = set()
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{sender_id or 'global'}")
        
        # Load existing processed images
        self.load_processed_images()
    
    def load_processed_images(self):
        """Load previously processed images from JSON file."""
        try:
            if self.processed_file.exists():
                with open(self.processed_file, 'r') as f:
                    data = json.load(f)
                    
                    # Handle both old and new formats
                    if isinstance(data, dict):
                        if 'processed_images' in data:
                            # New format with metadata
                            self.processed_images = set(data['processed_images'])
                        elif self.sender_id and self.sender_id in data:
                            # Sender-specific data
                            self.processed_images = set(data[self.sender_id].get('processed_images', []))
                        else:
                            self.processed_images = set()
                    elif isinstance(data, list):
                        # Old format - just a list
                        self.processed_images = set(data)
                    
                    self.logger.info(f"Loaded {len(self.processed_images)} previously processed images")
            else:
                self.logger.info("No processed images record found - will process all images")
        except Exception as e:
            self.logger.error(f"Error loading processed images: {e}")
            self.processed_images = set()
    
    def is_processed(self, image_path: str) -> bool:
        """Check if an image has already been processed."""
        return str(Path(image_path).name) in self.processed_images
    
    def mark_processed(self, image_path: str):
        """Mark an image as processed and save to disk."""
        image_name = str(Path(image_path).name)
        if image_name not in self.processed_images:
            self.processed_images.add(image_name)
            self.save_processed_images()
    
    def save_processed_images(self):
        """Save processed images list to JSON file."""
        try:
            # Load existing data to preserve other senders' data
            existing_data = {}
            if self.processed_file.exists():
                try:
                    with open(self.processed_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = {}
            
            # Ensure proper structure
            if not isinstance(existing_data, dict):
                existing_data = {}
            
            # Update data for this sender
            if self.sender_id:
                if 'senders' not in existing_data:
                    existing_data['senders'] = {}
                
                existing_data['senders'][self.sender_id] = {
                    'processed_images': sorted(list(self.processed_images)),
                    'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                # Global processed images
                existing_data['processed_images'] = sorted(list(self.processed_images))
            
            existing_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.processed_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving processed images: {e}")

class Phase1MetadataParser:
    """
    Parser specifically designed for Phase 1 metadata format.
    Handles both the simple format and enhanced format with quality details.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def parse_metadata_file(self, metadata_path: str) -> Optional[Dict]:
        """Parse Phase 1 metadata file format."""
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the enhanced metadata format from Phase 1
            metadata = {}
            
            # Extract basic information using regex patterns
            patterns = {
                'object_id': r'Object ID:\s*(.+)',
                'label': r'Class Label:\s*(.+)',
                'confidence': r'Detection Confidence:\s*([\d.]+)',
                'frame_count': r'Frame Number:\s*(\d+)',
                'timestamp': r'Timestamp:\s*(.+)',
                'save_reason': r'Save Reason:\s*(.+)',
                'total_frames': r'Total Frames Tracked:\s*(\d+)',
                'quality_score': r'Overall Quality Score:\s*([\d.]+)',
                'sharpness': r'Sharpness Score:\s*([\d.]+)',
                'brightness': r'Brightness Score:\s*([\d.]+)',
                'size_quality': r'Size Quality Score:\s*([\d.]+)',
                'face_detected': r'Face Detected:\s*(Yes|No)',
                'bbox_area': r'Bounding Box Area:\s*(\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1).strip()
                    if key in ['confidence', 'quality_score', 'sharpness', 'brightness', 'size_quality']:
                        metadata[key] = float(value)
                    elif key in ['frame_count', 'total_frames', 'bbox_area']:
                        metadata[key] = int(value)
                    elif key == 'face_detected':
                        metadata[key] = value.lower() == 'yes'
                    else:
                        metadata[key] = value
            
            # Extract bounding box coordinates
            bbox_match = re.search(r'Top-left:\s*\((\d+),\s*(\d+)\)', content)
            bbox_match2 = re.search(r'Bottom-right:\s*\((\d+),\s*(\d+)\)', content)
            
            if bbox_match and bbox_match2:
                x1, y1 = int(bbox_match.group(1)), int(bbox_match.group(2))
                x2, y2 = int(bbox_match2.group(1)), int(bbox_match2.group(2))
                metadata['bbox'] = (x1, y1, x2, y2)
            else:
                self.logger.error(f"Could not extract bounding box from {metadata_path}")
                return None
            
            # Set defaults for missing fields
            metadata.setdefault('save_reason', 'unknown')
            metadata.setdefault('total_frames', 1)
            metadata.setdefault('quality_score', 0.5)
            metadata.setdefault('confidence', 0.6)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to parse metadata file {metadata_path}: {e}")
            return None
    
    def extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """Extract timestamp from Phase 1 filename format."""
        # Phase 1 format: class_ID##_timestamp_reason.jpg
        patterns = [
            r'_(\d{13,})(?:_\w+)?\.jpg$',  # 13+ digit timestamp with optional reason
            r'_(\d{10})(?:_\w+)?\.jpg$',   # 10 digit timestamp with optional reason
            r'ID\d+_(\d{10,})(?:_\w+)?\.jpg$'  # Alternative format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        return None

class IncrementalAnomalySceneReconstructor:
    """
    Enhanced scene reconstructor with incremental batch processing for multi-sender environments.
    Maintains persistent state and processes only new objects while preserving batch context.
    """
    
    def __init__(self, input_dir: str, output_dir: str, scene_image: str = 'initial_scene.jpg',
                 batch_dir: str = None, sender_id: str = 'default_sender',
                 min_object_size: int = 20, debug_mode: bool = False, max_workers: int = 2):
        """Initialize the Enhanced Phase 2 reconstructor with incremental capabilities."""
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sender_id = sender_id
        self.scene_image_name = scene_image
        self.min_object_size = min_object_size
        self.debug_mode = debug_mode
        self.max_workers = max_workers
        
        # Batch processing setup
        if batch_dir:
            self.batch_dir = Path(batch_dir)
        else:
            self.batch_dir = self.output_dir / f'batch_{sender_id}'
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        self.individual_scenes_dir = self.batch_dir / 'individual_scenes'
        self.batch_scenes_dir = self.batch_dir / 'batch_scenes'
        self.montages_dir = self.batch_dir / 'montages'
        self.debug_info_dir = self.batch_dir / 'debug_info'
        
        for directory in [self.individual_scenes_dir, self.batch_scenes_dir, self.montages_dir, self.debug_info_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.batch_dir / f'reconstruction_{sender_id}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{sender_id}")
        
        # Initialize components
        self.metadata_parser = Phase1MetadataParser()
        
        # Use incremental batch tracker instead of regular processed tracker
        self.batch_tracker = IncrementalBatchTracker(self.batch_dir, sender_id)
        self.processed_tracker = EnhancedProcessedImagesTracker(self.output_dir, sender_id)
        
        # Scene data
        self.scene_image = None
        self.scene_height = 0
        self.scene_width = 0
        
        # Processing statistics
        self.stats = {
            'objects_found': 0,
            'objects_processed': 0,
            'new_objects_processed': 0,
            'objects_skipped': 0,
            'objects_already_processed': 0,
            'scenes_created': 0,
            'batch_scenes_created': 0,       
            'batch_scenes_updated': 0,   
            'processing_time': 0,
            'errors': 0,
            'save_reasons': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'objects_by_class': defaultdict(int),
            'batch_run_number': self.batch_tracker.batch_state.get('total_runs', 0) + 1,
            'sender_id': sender_id
        }
        
        self.logger.info(f"Enhanced Phase 2 Reconstructor initialized for sender: {sender_id}")
        self.logger.info(f"  Input directory: {self.input_dir}")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Batch directory: {self.batch_dir}")
        self.logger.info(f"  Incremental processing: {args.only_new}")
        self.logger.info(f"  Batch run number: {self.stats['batch_run_number']}")
    
    def load_scene_image(self) -> bool:
        """Load the initial scene image from input directory."""
        scene_path = self.input_dir / self.scene_image_name
        
        try:
            if not scene_path.exists():
                self.logger.error(f"Scene image not found: {scene_path}")
                return False
                
            self.scene_image = cv2.imread(str(scene_path))
            if self.scene_image is None:
                self.logger.error(f"Failed to load scene image: {scene_path}")
                return False
                
            self.scene_height, self.scene_width = self.scene_image.shape[:2]
            self.logger.info(f"Scene loaded: {self.scene_width}x{self.scene_height} from {scene_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading scene image: {e}")
            return False
    
    def discover_phase1_objects(self) -> List[DetectedObject]:
        """Discover all objects from Phase 1, filtering for incremental processing."""
        objects = []
        
        images_dir = self.input_dir / 'images'
        metadata_dir = self.input_dir / 'metadata'
        
        if not images_dir.exists() or not metadata_dir.exists():
            self.logger.error(f"Phase 1 directories not found: {images_dir}, {metadata_dir}")
            return objects
        
        # Find all image files
        image_patterns = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(images_dir.glob(pattern))
        
        self.logger.info(f"Found {len(image_files)} image files from Phase 1")
        
        # Filter for incremental processing
        if args.only_new:
            original_count = len(image_files)
            
            # Filter using both global and batch trackers
            new_image_files = []
            for img in image_files:
                if args.persistent_mode:
                    # In persistent mode, use batch tracker
                    if not self.batch_tracker.is_processed(img.name):
                        new_image_files.append(img)
                else:
                    # In regular mode, use global tracker
                    if not self.processed_tracker.is_processed(str(img)):
                        new_image_files.append(img)
            
            image_files = new_image_files
            filtered_count = len(image_files)
            skipped_count = original_count - filtered_count
            
            self.stats['objects_already_processed'] = skipped_count
            self.logger.info(f"Incremental mode: Processing {filtered_count} new images, skipped {skipped_count} already processed")
        
        for image_path in image_files:
            try:
                # Find corresponding metadata file
                metadata_filename = image_path.stem + '.txt'
                metadata_path = metadata_dir / metadata_filename
                
                if not metadata_path.exists():
                    self.logger.warning(f"Metadata not found for {image_path.name}")
                    continue
                
                # Parse metadata
                metadata = self.metadata_parser.parse_metadata_file(str(metadata_path))
                if metadata is None:
                    self.logger.warning(f"Failed to parse metadata for {image_path.name}")
                    continue
                
                # Extract timestamp from filename
                timestamp_str = self.metadata_parser.extract_timestamp_from_filename(image_path.name)
                timestamp = int(timestamp_str) if timestamp_str else int(time.time())
                
                # Create DetectedObject
                detected_obj = DetectedObject(
                    obj_id=metadata.get('object_id', 'unknown'),
                    label=metadata.get('label', 'unknown'),
                    bbox=metadata['bbox'],
                    confidence=metadata.get('confidence', 0.6),
                    quality_score=metadata.get('quality_score', 0.5),
                    save_reason=metadata.get('save_reason', 'unknown'),
                    timestamp=timestamp,
                    total_frames=metadata.get('total_frames', 1),
                    image_path=str(image_path),
                    metadata_path=str(metadata_path)
                )
                
                objects.append(detected_obj)
                
                # Update statistics
                self.stats['objects_found'] += 1
                self.stats['save_reasons'][detected_obj.save_reason] += 1
                self.stats['objects_by_class'][detected_obj.label] += 1
                
                # Quality distribution (binned)
                quality_bin = int(detected_obj.quality_score * 10) / 10  # 0.1 bins
                self.stats['quality_distribution'][quality_bin] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path.name}: {e}")
                self.stats['errors'] += 1
        
        self.logger.info(f"Successfully discovered {len(objects)} new objects from Phase 1")
        return objects

    def create_batch_scene_for_object_id(self, obj_id: str, new_objects: List[DetectedObject], is_update: bool = False) -> bool:
        """
        Create or update a batch scene for a specific object ID.
        For incremental processing: loads existing batch scene and adds only NEW objects.
        Uses the same simple logic as the old batch creation code.
        """
        try:
            if not new_objects:
                return False
            
            # Start with the existing batch scene if this is an update, otherwise fresh scene
            if is_update:
                existing_batch_path = self.find_existing_batch_scene(obj_id)
                if existing_batch_path and existing_batch_path.exists():
                    # Load existing batch scene as starting point
                    scene_copy = cv2.imread(str(existing_batch_path))
                    if scene_copy is not None:
                        self.logger.info(f"Loaded existing batch scene for ID {obj_id}: {existing_batch_path.name}")
                    else:
                        self.logger.warning(f"Failed to load existing batch scene, starting fresh")
                        scene_copy = self.scene_image.copy()
                else:
                    scene_copy = self.scene_image.copy()
            else:
                # New batch scene - start with fresh scene
                scene_copy = self.scene_image.copy()
            
            objects_added = 0
            
            # Sort new objects by timestamp (oldest first, just like old code)
            sorted_new_objects = sorted(new_objects, key=lambda x: x.timestamp)
            
            # Add each NEW object on top of existing batch scene
            # Using exact same logic as old code: simple direct placement
            for obj in sorted_new_objects:
                # Validate bounding box
                clamped_bbox = self.validate_and_clamp_bbox(obj.bbox)
                if clamped_bbox is None:
                    continue
                
                x1, y1, x2, y2 = clamped_bbox
                target_width, target_height = x2 - x1, y2 - y1
                
                # Load and resize object image
                object_img = self.load_and_resize_object_image(obj.image_path, (target_width, target_height))
                if object_img is None:
                    continue
                
                # Paste object (later timestamps will overlay earlier ones) - EXACT OLD CODE LOGIC
                scene_copy[y1:y2, x1:x2] = object_img
                
                # Add debug overlay - EXACT OLD CODE LOGIC
                scene_copy = self.add_debug_overlay(scene_copy, obj)
                
                objects_added += 1
            
            if objects_added == 0:
                self.logger.warning(f"No objects added to batch scene for ID {obj_id}")
                return False
            
            # Generate batch scene filename for this object ID - EXACT OLD CODE LOGIC
            first_obj = sorted_new_objects[0]
            safe_label = re.sub(r'[^\w\-_]', '_', first_obj.label)
            safe_id = re.sub(r'[^\w\-_]', '_', str(obj_id))
            
            # Count total objects (existing + new)
            existing_count = self.count_total_objects_in_existing_batch(obj_id) if is_update else 0
            total_objects = existing_count + objects_added
            
            # Remove old batch scene if this is an update
            if is_update:
                old_batch_path = self.find_existing_batch_scene(obj_id)
                if old_batch_path and old_batch_path.exists():
                    try:
                        old_batch_path.unlink()
                        self.logger.debug(f"Removed old batch scene: {old_batch_path.name}")
                    except:
                        pass
            
            batch_filename = f"batch_{safe_label}_ID{safe_id}_{total_objects}objects.jpg"
            batch_output_path = self.batch_scenes_dir / batch_filename
            
            # Save batch scene - EXACT OLD CODE LOGIC
            success = cv2.imwrite(str(batch_output_path), scene_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                self.stats['scenes_created'] += 1
                self.logger.info(f"{'Updated' if is_update else 'Created'} batch scene: {batch_filename} "
                            f"(+{objects_added} new, total: {total_objects} objects)")
                return True
            else:
                self.logger.error(f"Failed to save batch scene: {batch_output_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error creating batch scene for object ID {obj_id}: {e}")
            return False

    def count_total_objects_in_existing_batch(self, obj_id: str) -> int:
        """Count how many objects are already in the existing batch scene for this ID."""
        try:
            existing_batch_path = self.find_existing_batch_scene(obj_id)
            if existing_batch_path and existing_batch_path.exists():
                # Extract object count from filename
                match = re.search(r'_(\d+)objects\.jpg$', existing_batch_path.name)
                if match:
                    return int(match.group(1))
            return 0
        except:
            return 0
    
    def find_existing_batch_scene(self, obj_id: str) -> Optional[Path]:
        """Find existing batch scene file for a given object ID."""
        try:
            # Safe ID for filename matching
            safe_id = re.sub(r'[^\w\-_]', '_', str(obj_id))
            
            # Search for existing batch scene with this object ID
            pattern = f"batch_*_ID{safe_id}_*objects.jpg"
            existing_files = list(self.batch_scenes_dir.glob(pattern))
            
            if existing_files:
                # Return the first match (should only be one)
                return existing_files[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding existing batch scene for ID {obj_id}: {e}")
            return None

    def reconstruct_incremental_batch_scenes_by_object_id(self, new_objects: List[DetectedObject]) -> bool:
        """
        Create or update batch scenes grouped by object ID with incremental processing.
        Uses the simple logic from old code but adapted for incremental updates.
        """
        try:
            if not new_objects:
                self.logger.info("No new objects to process for batch scenes")
                return True
            
            # Group NEW objects by object ID
            new_objects_by_id = defaultdict(list)
            for obj in new_objects:
                new_objects_by_id[obj.obj_id].append(obj)
            
            batch_scenes_created = 0
            batch_scenes_updated = 0
            
            # Process each object ID that has new objects
            for obj_id, new_obj_list in new_objects_by_id.items():
                # Check if this is an update (existing batch scene) or new creation
                existing_batch_path = self.find_existing_batch_scene(obj_id)
                is_update = existing_batch_path is not None
                
                # Create/update the batch scene using simple old logic
                success = self.create_batch_scene_for_object_id(obj_id, new_obj_list, is_update)
                
                if success:
                    if is_update:
                        batch_scenes_updated += 1
                        self.logger.info(f"Updated batch scene for ID {obj_id} with {len(new_obj_list)} new objects")
                    else:
                        batch_scenes_created += 1
                        self.logger.info(f"Created new batch scene for ID {obj_id} with {len(new_obj_list)} objects")
                    
                    # Mark new objects as processed
                    for obj in new_obj_list:
                        if args.persistent_mode:
                            self.batch_tracker.mark_processed(Path(obj.image_path).name)
                        else:
                            self.processed_tracker.mark_processed(obj.image_path)
            
            # Update statistics
            total_scenes_affected = batch_scenes_created + batch_scenes_updated
            self.stats['batch_scenes_created'] = batch_scenes_created
            self.stats['batch_scenes_updated'] = batch_scenes_updated
            
            self.logger.info(f"Batch scene processing complete: {batch_scenes_created} created, "
                            f"{batch_scenes_updated} updated, {total_scenes_affected} total affected")
            
            return total_scenes_affected > 0
            
        except Exception as e:
            self.logger.error(f"Error in incremental batch scene reconstruction: {e}")
            self.stats['errors'] += 1
            return False
        
    def filter_objects(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Apply filters based on command line arguments."""
        filtered = []
        
        for obj in objects:
            # Quality filter
            if obj.quality_score < args.quality_filter:
                continue
            
            # Save reason filter
            if args.save_reason_filter != 'all' and obj.save_reason != args.save_reason_filter:
                continue
            
            # Size filter
            x1, y1, x2, y2 = obj.bbox
            width, height = x2 - x1, y2 - y1
            if width < self.min_object_size or height < self.min_object_size:
                continue
            
            filtered.append(obj)
        
        self.logger.info(f"Filtered {len(objects)} -> {len(filtered)} objects")
        return filtered
        
    def validate_and_clamp_bbox(self, bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """Validate and clamp bounding box to scene dimensions."""
        x1, y1, x2, y2 = bbox
        
        # Basic validation
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Clamp to scene bounds
        x1 = max(0, min(x1, self.scene_width - 1))
        y1 = max(0, min(y1, self.scene_height - 1))
        x2 = max(x1 + 1, min(x2, self.scene_width))
        y2 = max(y1 + 1, min(y2, self.scene_height))
        
        return (x1, y1, x2, y2)
        
    def load_and_resize_object_image(self, image_path: str, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load and resize object image to target dimensions."""
        try:
            object_img = cv2.imread(image_path)
            if object_img is None:
                self.logger.error(f"Failed to load object image: {image_path}")
                return None
            
            target_width, target_height = target_size
            
            # Check aspect ratio
            orig_h, orig_w = object_img.shape[:2]
            orig_ratio = orig_w / orig_h
            target_ratio = target_width / target_height
            
            if abs(orig_ratio - target_ratio) > 0.4:  # 40% difference
                self.logger.warning(f"Aspect ratio mismatch for {image_path}: "
                                  f"{orig_ratio:.2f} vs {target_ratio:.2f}")
            
            # Resize to exact dimensions
            resized = cv2.resize(object_img, (target_width, target_height))
            return resized
            
        except Exception as e:
            self.logger.error(f"Error processing object image {image_path}: {e}")
            return None
    
    def add_debug_overlay(self, scene: np.ndarray, obj: DetectedObject) -> np.ndarray:
        """Add debug information overlay."""
        if not self.debug_mode:
            return scene
        
        x1, y1, x2, y2 = obj.bbox
        
        # Color based on save reason
        color_map = {
            'disappeared': (0, 0, 255),     # Red - anomaly detected
            'periodic': (0, 255, 255),      # Yellow - long duration
            'shutdown': (255, 0, 0),        # Blue - system shutdown
            'manual': (255, 0, 255),        # Magenta - manual save
            'unknown': (128, 128, 128)      # Gray - unknown
        }
        
        color = color_map.get(obj.save_reason, (0, 255, 0))  # Default green
        
        # Draw bounding box
        cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label with comprehensive info
        label_lines = [
            f"{obj.label} [ID:{obj.obj_id}]",
            f"Reason: {obj.save_reason.upper()}",
            f"Quality: {obj.quality_score:.2f}",
            f"Conf: {obj.confidence:.2f}",
            f"Frames: {obj.total_frames}",
            f"Sender: {self.sender_id}"
        ]
        
        # Calculate text area
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in label_lines]
        max_width = max(size[0] for size in text_sizes)
        total_height = sum(size[1] + 3 for size in text_sizes)
        
        # Background rectangle
        bg_x1 = x1
        bg_y1 = max(0, y1 - total_height - 10)
        bg_x2 = min(self.scene_width, x1 + max_width + 10)
        bg_y2 = y1
        
        # Draw background
        overlay = scene.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        scene = cv2.addWeighted(overlay, 0.7, scene, 0.3, 0)
        
        # Draw text lines
        y_offset = bg_y1 + 12
        for line in label_lines:
            cv2.putText(scene, line, (x1 + 3, y_offset), font, font_scale, (255, 255, 255), thickness)
            y_offset += 15
        
        return scene
    
    def reconstruct_individual_scene(self, obj: DetectedObject) -> bool:
        """Reconstruct scene with a single object."""
        try:
            # Validate bounding box
            clamped_bbox = self.validate_and_clamp_bbox(obj.bbox)
            if clamped_bbox is None:
                self.logger.warning(f"Invalid bounding box for object {obj.obj_id}")
                return False
            
            x1, y1, x2, y2 = clamped_bbox
            target_width, target_height = x2 - x1, y2 - y1
            
            # Load and resize object image
            object_img = self.load_and_resize_object_image(obj.image_path, (target_width, target_height))
            if object_img is None:
                return False
            
            # Create scene copy
            scene_copy = self.scene_image.copy()
            
            # Paste object
            scene_copy[y1:y2, x1:x2] = object_img
            
            # Add debug overlay
            scene_copy = self.add_debug_overlay(scene_copy, obj)
            
            # Generate output filename with sender info
            timestamp_str = str(obj.timestamp)
            safe_reason = re.sub(r'[^\w\-_]', '_', obj.save_reason)
            safe_label = re.sub(r'[^\w\-_]', '_', obj.label)
            safe_id = re.sub(r'[^\w\-_]', '_', str(obj.obj_id))
            
            filename = f"scene_{self.sender_id}_{safe_label}_ID{safe_id}_{timestamp_str}_{safe_reason}.jpg"
            output_path = self.individual_scenes_dir / filename
            
            # Save reconstructed scene
            success = cv2.imwrite(str(output_path), scene_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                self.stats['scenes_created'] += 1
                self.stats['objects_processed'] += 1
                self.stats['new_objects_processed'] += 1
                
                # Mark as processed in appropriate tracker
                if args.persistent_mode:
                    self.batch_tracker.mark_processed(Path(obj.image_path).name)
                else:
                    self.processed_tracker.mark_processed(obj.image_path)
                
                self.logger.info(f"Created scene: {filename} (Quality: {obj.quality_score:.2f}, Reason: {obj.save_reason})")
                return True
            else:
                self.logger.error(f"Failed to save scene: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error reconstructing scene for object {obj.obj_id}: {e}")
            self.stats['errors'] += 1
            return False
    
    def create_incremental_batch_montage(self, new_objects: List[DetectedObject]) -> bool:
        """Create or update incremental batch montage with ALL objects in batch directory."""
        try:
            # Get all existing individual scene images in batch directory
            existing_scenes = list(self.individual_scenes_dir.glob(f"scene_{self.sender_id}_*.jpg"))
            
            if not existing_scenes:
                self.logger.warning("No scene images found for montage creation")
                return False
            
            # Sort by timestamp for consistent ordering
            def extract_timestamp_from_scene(filepath: Path) -> int:
                match = re.search(r'_(\d{10,})_', filepath.name)
                return int(match.group(1)) if match else 0
            
            existing_scenes.sort(key=extract_timestamp_from_scene)
            
            # Calculate montage grid
            total_scenes = len(existing_scenes)
            cols = min(6, total_scenes)
            rows = (total_scenes + cols - 1) // cols
            
            # Montage parameters
            thumb_size = 200
            margin = 10
            header_height = 80
            
            # Create montage canvas
            montage_width = cols * thumb_size + (cols + 1) * margin
            montage_height = rows * thumb_size + (rows + 1) * margin + header_height
            montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
            montage.fill(32)  # Dark gray background
            
            # Add header information
            batch_summary = self.batch_tracker.get_batch_summary()
            header_lines = [
                f"Anomaly Montage - Sender: {self.sender_id}",
                f"Total Objects: {total_scenes} | Batch Run: {batch_summary['total_runs']} | New in this run: {len(new_objects)}"
            ]
            
            y_pos = 25
            for line in header_lines:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                x_pos = (montage_width - text_size[0]) // 2
                cv2.putText(montage, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_pos += 30
            
            # Place scene thumbnails
            for i, scene_path in enumerate(existing_scenes):
                row = i // cols
                col = i % cols
                
                x = margin + col * (thumb_size + margin)
                y = header_height + margin + row * (thumb_size + margin)
                
                # Load scene image
                scene_img = cv2.imread(str(scene_path))
                if scene_img is not None:
                    # Resize maintaining aspect ratio
                    h, w = scene_img.shape[:2]
                    if w > h:
                        new_w = thumb_size
                        new_h = int(h * thumb_size / w)
                    else:
                        new_h = thumb_size
                        new_w = int(w * thumb_size / h)
                    
                    resized = cv2.resize(scene_img, (new_w, new_h))
                    
                    # Center in thumbnail area
                    start_x = x + (thumb_size - new_w) // 2
                    start_y = y + (thumb_size - new_h) // 2
                    
                    montage[start_y:start_y+new_h, start_x:start_x+new_w] = resized
                    
                    # Highlight new objects with colored border
                    is_new = any(Path(obj.image_path).stem in scene_path.name for obj in new_objects)
                    if is_new:
                        border_color = (0, 255, 0)  # Green for new objects
                        cv2.rectangle(montage, (x-2, y-2), (x+thumb_size+2, y+thumb_size+2), border_color, 3)
                    
                    # Add scene info
                    # Extract info from filename
                    match = re.search(r'scene_(.+?)_(.+?)_ID(.+?)_(\d+)_(.+?)\.jpg', scene_path.name)
                    if match:
                        _, label, obj_id, timestamp, reason = match.groups()
                        info_text = f"{label[:8]} R:{reason[:3]}"
                        cv2.putText(montage, info_text, (x + 2, y + thumb_size - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            # Save montage (overwrite previous version)
            montage_filename = f"incremental_montage_{self.sender_id}.jpg"
            montage_path = self.montages_dir / montage_filename
            
            success = cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if success:
                # Update montage version
                self.batch_tracker.batch_state['montage_version'] = self.batch_tracker.batch_state.get('montage_version', 0) + 1
                self.batch_tracker.save_batch_state()
                
                self.logger.info(f"Updated incremental montage: {montage_filename} ({total_scenes} total objects, {len(new_objects)} new)")
                return True
            else:
                self.logger.error(f"Failed to save montage: {montage_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error creating incremental batch montage: {e}")
            return False
    
    def process_all_objects(self) -> bool:
        """Main processing function with incremental batch capabilities."""
        start_time = time.time()
        
        try:
            # Load scene
            if not self.load_scene_image():
                return False
            
            # Discover Phase 1 objects (with incremental filtering built in)
            all_objects = self.discover_phase1_objects()
            if not all_objects:
                if args.only_new and self.stats['objects_already_processed'] > 0:
                    self.logger.info("No new objects found - all images already processed")
                    # Still update montage with existing objects if requested
                    if args.create_montage:
                        self.create_incremental_batch_montage([])
                    return True
                else:
                    self.logger.warning("No objects found from Phase 1 system")
                    return True
            
            # Apply filters
            filtered_objects = self.filter_objects(all_objects)
            if not filtered_objects:
                self.logger.warning("No objects remaining after filtering")
                return True
            
            # Sort by timestamp if requested
            if args.preserve_timestamp_order:
                filtered_objects.sort(key=lambda x: x.timestamp)
            
            self.logger.info(f"Processing {len(filtered_objects)} new objects for sender {self.sender_id}...")
            
            # Process individual scenes
            successful_objects = []
            # Always create individual scenes first
            for obj in filtered_objects:
                if self.reconstruct_individual_scene(obj):
                    successful_objects.append(obj)
                else:
                    self.stats['objects_skipped'] += 1

            # Also create batch scenes if enabled
            if args.batch_process and successful_objects:
                self.reconstruct_incremental_batch_scenes_by_object_id(successful_objects) 
            
            # Create or update incremental montage
            if args.create_montage and successful_objects:
                self.create_incremental_batch_montage(successful_objects)
            
            # Update batch statistics
            self.batch_tracker.update_batch_statistics(
                new_objects_count=len(successful_objects),
                total_objects_in_batch=len(list(self.individual_scenes_dir.glob(f"scene_{self.sender_id}_*.jpg")))
            )
            
            # Generate processing report
            self.generate_incremental_processing_report(successful_objects)
            
            # Processing complete
            self.stats['processing_time'] = time.time() - start_time
            self.logger.info(f"Processing completed in {self.stats['processing_time']:.2f} seconds")
            self.log_final_statistics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fatal error during processing: {e}")
            return False
    
    def generate_incremental_processing_report(self, objects: List[DetectedObject]):
        """Generate incremental processing report."""
        try:
            report_path = self.batch_dir / f'processing_report_run_{self.stats["batch_run_number"]}.json'
            batch_summary = self.batch_tracker.get_batch_summary()
            
            # Comprehensive report
            detailed_stats = {
                'processing_summary': {
                    'sender_id': self.sender_id,
                    'batch_run_number': self.stats['batch_run_number'],
                    'batch_directory': str(self.batch_dir),
                    'new_objects_found': self.stats['objects_found'],
                    'new_objects_processed': self.stats['new_objects_processed'],
                    'objects_skipped': self.stats['objects_skipped'],
                    'objects_already_processed': self.stats['objects_already_processed'],
                    'scenes_created_this_run': self.stats['scenes_created'],
                    'processing_time_seconds': self.stats['processing_time'],
                    'errors_encountered': self.stats['errors'],
                    'cumulative_objects_in_batch': batch_summary['cumulative_objects'],
                    'total_batch_runs': batch_summary['total_runs']
                },
                'batch_state': batch_summary,
                'object_distribution_this_run': {
                    'by_save_reason': dict(self.stats['save_reasons']),
                    'by_class': dict(self.stats['objects_by_class']),
                    'quality_distribution': dict(self.stats['quality_distribution'])
                },
                'configuration_used': {
                    'input_directory': str(self.input_dir),
                    'output_directory': str(self.output_dir),
                    'batch_directory': str(self.batch_dir),
                    'scene_image': self.scene_image_name,
                    'sender_id': self.sender_id,
                    'incremental_mode': args.only_new,
                    'persistent_mode': args.persistent_mode,
                    'debug_mode': self.debug_mode,
                    'create_montage': args.create_montage,
                    'batch_processing': args.batch_process
                },
                'new_objects_details': []
            }
            
            # Details for new objects processed
            for obj in objects:
                obj_detail = {
                    'object_id': obj.obj_id,
                    'label': obj.label,
                    'save_reason': obj.save_reason,
                    'quality_score': obj.quality_score,
                    'confidence': obj.confidence,
                    'total_frames': obj.total_frames,
                    'timestamp': obj.timestamp,
                    'bbox_area': (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
                }
                detailed_stats['new_objects_details'].append(obj_detail)
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(detailed_stats, f, indent=2)
            
            # Also save/update cumulative report
            cumulative_report_path = self.batch_dir / 'cumulative_processing_report.json'
            self.update_cumulative_report(cumulative_report_path, detailed_stats)
            
            self.logger.info(f"Processing report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating processing report: {e}")
    
    def update_cumulative_report(self, cumulative_path: Path, current_stats: Dict):
        """Update cumulative processing report across all runs."""
        try:
            # Load existing cumulative data
            cumulative_data = {}
            if cumulative_path.exists():
                with open(cumulative_path, 'r') as f:
                    cumulative_data = json.load(f)
            
            # Initialize if empty
            if not cumulative_data:
                cumulative_data = {
                    'sender_id': self.sender_id,
                    'batch_directory': str(self.batch_dir),
                    'first_run': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_runs': 0,
                    'cumulative_objects_processed': 0,
                    'cumulative_scenes_created': 0,
                    'cumulative_processing_time': 0.0,
                    'runs_history': []
                }
            
            # Update cumulative statistics
            cumulative_data['last_run'] = time.strftime('%Y-%m-%d %H:%M:%S')
            cumulative_data['total_runs'] = current_stats['processing_summary']['total_batch_runs']
            cumulative_data['cumulative_objects_processed'] += current_stats['processing_summary']['new_objects_processed']
            cumulative_data['cumulative_scenes_created'] += current_stats['processing_summary']['scenes_created_this_run']
            cumulative_data['cumulative_processing_time'] += current_stats['processing_summary']['processing_time_seconds']
            
            # Add this run to history
            run_summary = {
                'run_number': current_stats['processing_summary']['batch_run_number'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'new_objects_processed': current_stats['processing_summary']['new_objects_processed'],
                'scenes_created': current_stats['processing_summary']['scenes_created_this_run'],
                'processing_time': current_stats['processing_summary']['processing_time_seconds'],
                'save_reasons': current_stats['object_distribution_this_run']['by_save_reason']
            }
            
            cumulative_data['runs_history'].append(run_summary)
            
            # Save cumulative report
            with open(cumulative_path, 'w') as f:
                json.dump(cumulative_data, f, indent=2)
            
            self.logger.info(f"Updated cumulative report: {cumulative_path}")
            
        except Exception as e:
            self.logger.error(f"Error updating cumulative report: {e}")
    
    def log_final_statistics(self):
        """Log comprehensive final statistics for this run."""
        batch_summary = self.batch_tracker.get_batch_summary()
        
        self.logger.info("=" * 70)
        self.logger.info(f"INCREMENTAL PHASE 2 RECONSTRUCTION COMPLETED - SENDER: {self.sender_id}")
        self.logger.info("=" * 70)
        
        self.logger.info(f"Batch Run Number: {self.stats['batch_run_number']}")
        self.logger.info(f"Processing Time: {self.stats['processing_time']:.2f} seconds")
        self.logger.info(f"New Objects Found: {self.stats['objects_found']}")
        self.logger.info(f"New Objects Processed: {self.stats['new_objects_processed']}")
        self.logger.info(f"Objects Already Processed: {self.stats['objects_already_processed']}")
        self.logger.info(f"Objects Skipped: {self.stats['objects_skipped']}")
        self.logger.info(f"New Scenes Created: {self.stats['scenes_created']}")
        self.logger.info(f"Batch Scenes Created: {self.stats.get('batch_scenes_created', 0)}")    # Add this
        self.logger.info(f"Batch Scenes Updated: {self.stats.get('batch_scenes_updated', 0)}")    # Add this
        self.logger.info(f"Errors: {self.stats['errors']}")
        
        self.logger.info("\nCUMULATIVE BATCH STATISTICS:")
        self.logger.info(f"  Total Batch Runs: {batch_summary['total_runs']}")
        self.logger.info(f"  Total Objects in Batch: {batch_summary['cumulative_objects']}")
        self.logger.info(f"  Montage Version: {batch_summary['montage_version']}")
        
        self.logger.info("\nNEW OBJECTS BREAKDOWN:")
        for reason, count in self.stats['save_reasons'].items():
            self.logger.info(f"  {reason.upper()}: {count} objects")
        
        self.logger.info("\nNEW OBJECT CLASSES:")
        for obj_class, count in self.stats['objects_by_class'].items():
            self.logger.info(f"  {obj_class}: {count} objects")
        
        if self.stats['objects_found'] > 0:
            success_rate = (self.stats['new_objects_processed'] / self.stats['objects_found']) * 100
            self.logger.info(f"\nProcessing Success Rate: {success_rate:.1f}%")
        
        self.logger.info(f"\nBatch Directory: {self.batch_dir}")
        self.logger.info(f"Individual Scenes: {self.individual_scenes_dir}")
        self.logger.info(f"Montages: {self.montages_dir}")
        self.logger.info("=" * 70)


def main():
    """Main entry point for Enhanced Phase 2 Anomaly Scene Reconstructor."""
    
    try:
        print("="*70)
        print("ENHANCED PHASE 2: INCREMENTAL MULTI-SENDER ANOMALY RECONSTRUCTOR")
        print("Integrated with Phase 1 Enhanced Object Tracking System")
        print("="*70)
        print()
        
        # Validate arguments
        if args.batch_dir is None:
            args.batch_dir = str(Path(args.output_dir) / f'batch_{args.sender_id}')
        
        # Initialize reconstructor
        reconstructor = IncrementalAnomalySceneReconstructor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            scene_image=args.scene_image,
            batch_dir=args.batch_dir,
            sender_id=args.sender_id,
            min_object_size=args.min_object_size,
            debug_mode=args.debug_mode,
            max_workers=args.max_workers
        )
        
        # Display configuration
        print("CONFIGURATION:")
        print("-" * 15)
        print(f"Sender ID: {args.sender_id}")
        print(f"Input Directory (Phase 1): {args.input_dir}")
        print(f"Output Directory: {args.output_dir}")
        print(f"Batch Directory: {args.batch_dir}")
        print(f"Scene Image: {args.scene_image}")
        print(f"Quality Filter: {args.quality_filter}")
        print(f"Save Reason Filter: {args.save_reason_filter}")
        print(f"Batch Processing: {args.batch_process}")
        print(f"Create Montages: {args.create_montage}")
        print(f"Debug Mode: {args.debug_mode}")
        print(f"Incremental Mode: {args.only_new}")
        print(f"Persistent Mode: {args.persistent_mode}")
        print(f"Preserve Timestamp Order: {args.preserve_timestamp_order}")
        print()
        
        # Show batch status
        batch_summary = reconstructor.batch_tracker.get_batch_summary()
        if batch_summary['total_runs'] > 0:
            print("BATCH STATUS:")
            print("-" * 13)
            print(f"Previous Runs: {batch_summary['total_runs']}")
            print(f"Last Run: {batch_summary['last_run']}")
            print(f"Total Objects in Batch: {batch_summary['cumulative_objects']}")
            print(f"Montage Version: {batch_summary['montage_version']}")
            print()
        
        # Process all objects
        success = reconstructor.process_all_objects()
        
        if success:
            print("\n" + "="*70)
            print("INCREMENTAL RECONSTRUCTION COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"Sender: {args.sender_id}")
            print(f"Batch directory: {args.batch_dir}")
            
            # Show output locations
            individual_scenes_dir = Path(args.batch_dir) / 'individual_scenes'
            montages_dir = Path(args.batch_dir) / 'montages'
            
            print(f"Individual scenes: {individual_scenes_dir}/")
            print(f"Incremental montage: {montages_dir}/incremental_montage_{args.sender_id}.jpg")
            print(f"Processing reports: {args.batch_dir}/processing_report_run_*.json")
            print(f"Cumulative report: {args.batch_dir}/cumulative_processing_report.json")
            print(f"Batch state: {args.batch_dir}/batch_state.json")
            
            # Show new objects processed
            if reconstructor.stats['new_objects_processed'] > 0:
                print(f"\nNew objects processed in this run: {reconstructor.stats['new_objects_processed']}")
            else:
                print(f"\nNo new objects found - batch remains at {batch_summary['cumulative_objects']} total objects")
        else:
            print("\n" + "="*70)
            print("INCREMENTAL RECONSTRUCTION FAILED")
            print("="*70)
            print("Check the log files for detailed error information.")
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
