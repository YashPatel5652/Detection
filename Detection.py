import cv2
import time
import os
import argparse
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import math
import hashlib
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import logging
from pathlib import Path
import signal
import sys
import json

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='yolov8l.pt', help='Path to YOLOv8 model file')
parser.add_argument('-s', '--source', type=str, default='0')
parser.add_argument('--save_dir', type=str, default='captures')
parser.add_argument('--confidence', type=float, default=0.60)
parser.add_argument('--iou_threshold', type=float, default=0.4)
parser.add_argument('--save-scene', action='store_false', help='Save initial scene for reference')
parser.add_argument('--no-gui', action='store_true', help='Disable GUI for production')
parser.add_argument('--detection-size', type=int, default=640, help='Detection resolution')
parser.add_argument('--max-workers', type=int, default=2, help='Number of worker threads')
parser.add_argument('--strict-tracking', action='store_true', help='Enable strict ID consistency mode')
parser.add_argument('--buffer-size', type=int, default=15, help='Max images to buffer per object')
parser.add_argument('--save-delay', type=int, default=30, help='Frames to wait after object disappears before saving')
parser.add_argument('--min-total-frames', type=int, default=5, help='Minimum total frames for disappeared object to be saved')
parser.add_argument('--periodic-save-interval', type=int, default=300, help='Frames interval for re-saving long-duration objects')
args = parser.parse_args()

# Setup logging
os.makedirs(args.save_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(args.save_dir, 'detection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImageQualityAnalyzer:
    """Lightweight image quality assessment for object crops"""
    
    def __init__(self):
        # Quality weight parameters
        self.sharpness_weight = 0.4
        self.size_weight = 0.25
        self.brightness_weight = 0.15
        self.face_bonus_weight = 0.2
        
        # Face detection for person/animal priority
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_detection_enabled = True
            logger.info("Face detection enabled for quality assessment")
        except:
            self.face_detection_enabled = False
            logger.warning("Face detection disabled - cascade not available")
    
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Laplacian variance for sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (typical values: 0-2000)
            normalized_sharpness = min(1.0, laplacian_var / 1000.0)
            
            return normalized_sharpness
            
        except Exception as e:
            logger.debug(f"Sharpness calculation failed: {e}")
            return 0.0
    
    def calculate_brightness_quality(self, image):
        """Calculate brightness quality (avoid too dark/bright images)"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            mean_brightness = np.mean(gray)
            
            # Optimal brightness range: 80-180 (0-255 scale)
            # Calculate quality based on distance from optimal range
            if 80 <= mean_brightness <= 180:
                quality = 1.0
            elif mean_brightness < 80:
                # Too dark
                quality = max(0.0, mean_brightness / 80.0)
            else:
                # Too bright
                quality = max(0.0, (255 - mean_brightness) / 75.0)
            
            return quality
            
        except Exception as e:
            logger.debug(f"Brightness calculation failed: {e}")
            return 0.5
    
    def detect_face_presence(self, image, object_label):
        """Detect if image contains a visible face (for persons/animals)"""
        if not self.face_detection_enabled:
            return False
        
        # Only check for faces in person/animal objects
        face_relevant_classes = ['person', 'human', 'man', 'woman', 'child', 'people']
        if object_label.lower() not in face_relevant_classes:
            return False
        
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
            
            return len(faces) > 0
            
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
            return False
    
    def calculate_size_quality(self, image, bbox_area):
        """Calculate quality based on bounding box size"""
        try:
            # Larger objects generally provide better detail
            # Normalize based on typical object sizes (1000-50000 pixels)
            normalized_size = min(1.0, bbox_area / 25000.0)
            
            # Bonus for very large objects
            if bbox_area > 40000:
                normalized_size = min(1.0, normalized_size * 1.1)
            
            return normalized_size
            
        except Exception as e:
            logger.debug(f"Size quality calculation failed: {e}")
            return 0.5
    
    def calculate_overall_quality(self, image, bbox_area, object_label):
        """Calculate comprehensive quality score"""
        try:
            # Individual quality metrics
            sharpness = self.calculate_sharpness(image)
            brightness = self.calculate_brightness_quality(image)
            size_quality = self.calculate_size_quality(image, bbox_area)
            
            # Face detection bonus
            face_bonus = 0.0
            if self.detect_face_presence(image, object_label):
                face_bonus = 1.0
                logger.debug(f"Face detected in {object_label} - quality bonus applied")
            
            # Weighted combination
            overall_score = (
                sharpness * self.sharpness_weight +
                size_quality * self.size_weight +
                brightness * self.brightness_weight +
                face_bonus * self.face_bonus_weight
            )
            
            # Face bonus can exceed 1.0 for face-containing images
            final_score = min(1.5 if face_bonus > 0 else 1.0, overall_score)
            
            return final_score, {
                'sharpness': sharpness,
                'brightness': brightness,
                'size_quality': size_quality,
                'face_detected': face_bonus > 0,
                'bbox_area': bbox_area
            }
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.0, {}

class IntelligentImageBuffer:
    """Enhanced buffer system with anomaly-aware saving for disappeared and long-duration objects"""
    
    def __init__(self, max_buffer_size=15, save_delay_frames=30, min_total_frames_for_disappeared=5, periodic_save_interval=300):
        self.max_buffer_size = max_buffer_size
        self.save_delay_frames = save_delay_frames
        self.min_total_frames_for_disappeared = min_total_frames_for_disappeared  # NEW: Minimum frames for disappeared objects
        self.periodic_save_interval = periodic_save_interval  # NEW: Periodic save interval for long-duration objects
        
        # Object image buffers: {obj_id: [(image, quality_score, metadata, timestamp), ...]}
        self.object_buffers = defaultdict(list)
        
        # Track when objects disappeared for delayed saving
        self.disappeared_objects = {}  # {obj_id: frame_count_when_disappeared}
        
        # NEW: Track total frames seen per object (for disappeared object evaluation)
        self.object_total_frames = defaultdict(int)
        
        # NEW: Track when objects were last saved (for periodic saving)
        self.object_last_saved_frame = defaultdict(int)
        
        # NEW: Track objects that have been saved at least once
        self.objects_ever_saved = set()
        
        # Quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer()
        
        # Statistics
        self.total_images_buffered = 0
        self.total_images_saved = 0
        self.objects_processed = 0
        self.disappeared_objects_saved = 0  # NEW: Count disappeared objects saved
        self.periodic_saves = 0  # NEW: Count periodic saves
        
        logger.info(f"Enhanced intelligent image buffer initialized:")
        logger.info(f"  - Buffer size: {max_buffer_size}")
        logger.info(f"  - Save delay: {save_delay_frames} frames")
        logger.info(f"  - Min total frames for disappeared objects: {min_total_frames_for_disappeared}")
        logger.info(f"  - Periodic save interval: {periodic_save_interval} frames")
    
    def add_image(self, obj_id, image, bbox, confidence, object_label, frame_count):
        """Add image to buffer with quality assessment and enhanced tracking"""
        try:
            # Calculate bbox area
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            # Calculate quality score
            quality_score, quality_details = self.quality_analyzer.calculate_overall_quality(
                image, bbox_area, object_label
            )
            
            # NEW: Increment total frames seen for this object
            self.object_total_frames[obj_id] += 1
            
            # Create metadata
            metadata = {
                'object_id': obj_id,
                'label': object_label,
                'bbox': bbox,
                'confidence': confidence,
                'frame_count': frame_count,
                'quality_score': quality_score,
                'quality_details': quality_details,
                'timestamp': time.time(),
                'total_frames_seen': self.object_total_frames[obj_id]  # NEW: Include total frames
            }
            
            # Add to buffer
            buffer_entry = (image.copy(), quality_score, metadata, time.time())
            self.object_buffers[obj_id].append(buffer_entry)
            self.total_images_buffered += 1
            
            # Maintain buffer size - keep only the best images
            if len(self.object_buffers[obj_id]) > self.max_buffer_size:
                # Sort by quality score and keep top images
                self.object_buffers[obj_id].sort(key=lambda x: x[1], reverse=True)
                self.object_buffers[obj_id] = self.object_buffers[obj_id][:self.max_buffer_size]
            
            # Remove from disappeared list if object reappeared
            if obj_id in self.disappeared_objects:
                del self.disappeared_objects[obj_id]
            
            logger.debug(f"Added image for object {obj_id} (quality: {quality_score:.3f}, buffer size: {len(self.object_buffers[obj_id])}, total frames: {self.object_total_frames[obj_id]})")
            
        except Exception as e:
            logger.error(f"Failed to add image to buffer for object {obj_id}: {e}")
    
    def mark_object_disappeared(self, obj_id, frame_count):
        """Mark object as disappeared for delayed saving - FIXED VERSION"""
        if obj_id in self.object_buffers and len(self.object_buffers[obj_id]) > 0:
            # Only mark if not already marked to avoid overwriting the original disappeared frame
            if obj_id not in self.disappeared_objects:
                self.disappeared_objects[obj_id] = frame_count
                logger.info(f"Object {obj_id} marked as disappeared at frame {frame_count} (total frames seen: {self.object_total_frames[obj_id]})")
            else:
                logger.debug(f"Object {obj_id} already marked as disappeared at frame {self.disappeared_objects[obj_id]}")

    def get_objects_ready_for_saving(self, current_frame_count):
        """DEBUGGING VERSION - Enhanced method with detailed logging"""
        ready_objects = []
        
        # 1. Handle disappeared objects
        for obj_id, disappeared_frame in list(self.disappeared_objects.items()):
            frames_since_disappeared = current_frame_count - disappeared_frame
            
            # DEBUG: Log the calculation
            logger.debug(f"Checking disappeared object {obj_id}: "
                        f"current_frame={current_frame_count}, "
                        f"disappeared_frame={disappeared_frame}, "
                        f"frames_since_disappeared={frames_since_disappeared}, "
                        f"save_delay_frames={self.save_delay_frames}")
            
            if frames_since_disappeared >= self.save_delay_frames:
                ready_objects.append((obj_id, 'disappeared'))
                logger.info(f"Object {obj_id} ready for disappeared save (waited {frames_since_disappeared} frames)")
        
        # 2. Handle periodic saving for long-duration objects
        for obj_id in list(self.object_buffers.keys()):
            if obj_id in self.disappeared_objects:
                continue
                
            if self._should_periodic_save(obj_id, current_frame_count):
                ready_objects.append((obj_id, 'periodic'))
                logger.debug(f"Object {obj_id} ready for periodic save")
        
        return ready_objects
    
    def _should_periodic_save(self, obj_id, current_frame_count):
        """NEW: Determine if object should be periodically saved"""
        # Must have been seen for minimum total frames
        if self.object_total_frames[obj_id] < self.min_total_frames_for_disappeared:
            return False
        
        # Check if enough frames have passed since last save
        last_saved_frame = self.object_last_saved_frame.get(obj_id, 0)
        frames_since_last_save = current_frame_count - last_saved_frame
        
        # For objects never saved, use a shorter initial interval
        if obj_id not in self.objects_ever_saved:
            initial_save_threshold = max(self.periodic_save_interval // 3, 50)  # 1/3 of periodic interval, min 50 frames
            return frames_since_last_save >= initial_save_threshold
        
        # For objects already saved, use full periodic interval
        return frames_since_last_save >= self.periodic_save_interval
    
    def get_best_image(self, obj_id):
        """Get the best quality image for an object"""
        if obj_id not in self.object_buffers or len(self.object_buffers[obj_id]) == 0:
            return None
        
        # Sort by quality score and get the best one
        buffer = self.object_buffers[obj_id]
        buffer.sort(key=lambda x: x[1], reverse=True)
        
        best_image, best_score, best_metadata, _ = buffer[0]
        
        logger.info(f"Selected best image for object {obj_id}: quality {best_score:.3f} from {len(buffer)} candidates")
        
        return best_image, best_metadata
    
    def save_and_cleanup_object(self, obj_id, save_reason, image_saver):
        """Enhanced method to save best image with reason-specific handling"""
        try:
            best_result = self.get_best_image(obj_id)
            if best_result is None:
                logger.warning(f"No images available for object {obj_id}")
                return False
            
            best_image, metadata = best_result
            
            # NEW: Enhanced logic based on save reason
            should_save = False
            cleanup_buffer = False
            
            if save_reason == 'disappeared':
                # For disappeared objects, check minimum total frames requirement
                total_frames = self.object_total_frames.get(obj_id, 0)
                if total_frames >= self.min_total_frames_for_disappeared:
                    should_save = True
                    cleanup_buffer = True  # Full cleanup for disappeared objects
                    self.disappeared_objects_saved += 1
                    logger.info(f"Saving disappeared object {obj_id} (total frames: {total_frames})")
                else:
                    logger.info(f"Skipping disappeared object {obj_id} - insufficient frames ({total_frames}/{self.min_total_frames_for_disappeared})")
                    cleanup_buffer = True  # Still cleanup buffer even if not saving
                    
            elif save_reason == 'periodic':
                should_save = True
                cleanup_buffer = False  # Keep buffer for future periodic saves, but reset it
                self.periodic_saves += 1
                logger.info(f"Periodic save for long-duration object {obj_id}")
            
            else:
                # Original logic for standard saves
                should_save = True
                cleanup_buffer = True
            
            if should_save:
                # Generate filename with timestamp and save reason
                timestamp = int(time.time() * 1000)
                save_suffix = f"_{save_reason}" if save_reason in ['disappeared', 'periodic'] else ""
                filename = f"{metadata['label']}_ID{obj_id}_{timestamp}{save_suffix}.jpg"
                
                # Add save reason to metadata
                metadata['save_reason'] = save_reason
                metadata['total_frames_when_saved'] = self.object_total_frames.get(obj_id, 0)
                
                # Save image
                success = image_saver.save_best_image(best_image, filename, metadata)
                
                if success:
                    self.total_images_saved += 1
                    if cleanup_buffer:
                        self.objects_processed += 1
                    
                    # NEW: Update tracking for periodic saves
                    if save_reason == 'periodic':
                        self.object_last_saved_frame[obj_id] = metadata['frame_count']
                        self.objects_ever_saved.add(obj_id)
                        # Reset buffer for periodic saves to capture new frames
                        self.object_buffers[obj_id] = []
                        logger.info(f"Buffer reset for object {obj_id} after periodic save")
                    
                    # Cleanup based on save reason
                    if cleanup_buffer:
                        self.cleanup_object_buffer(obj_id)
                    
                    logger.info(f"SAVED BEST IMAGE: {filename} (quality: {metadata['quality_score']:.3f}, reason: {save_reason})")
                    return True
                else:
                    logger.error(f"Failed to save best image for object {obj_id}")
                    return False
            else:
                # Cleanup even if not saving
                if cleanup_buffer:
                    self.cleanup_object_buffer(obj_id)
                return False
                
        except Exception as e:
            logger.error(f"Error saving best image for object {obj_id}: {e}")
            return False
    
    def cleanup_object_buffer(self, obj_id):
        """Remove object from all buffers and tracking"""
        if obj_id in self.object_buffers:
            del self.object_buffers[obj_id]
        
        if obj_id in self.disappeared_objects:
            del self.disappeared_objects[obj_id]
        
        # NEW: Clean up enhanced tracking data
        if obj_id in self.object_total_frames:
            del self.object_total_frames[obj_id]
        
        if obj_id in self.object_last_saved_frame:
            del self.object_last_saved_frame[obj_id]
        
        self.objects_ever_saved.discard(obj_id)
    
    def get_buffer_stats(self):
        """Get enhanced buffer statistics"""
        total_buffered_images = sum(len(buffer) for buffer in self.object_buffers.values())
        
        return {
            'active_object_buffers': len(self.object_buffers),
            'total_buffered_images': total_buffered_images,
            'disappeared_objects': len(self.disappeared_objects),
            'total_images_buffered': self.total_images_buffered,
            'total_images_saved': self.total_images_saved,
            'objects_processed': self.objects_processed,
            'disappeared_objects_saved': self.disappeared_objects_saved,  # NEW
            'periodic_saves': self.periodic_saves,  # NEW
            'objects_ever_saved': len(self.objects_ever_saved),  # NEW
            'objects_ready_for_periodic_save': sum(1 for obj_id in self.object_buffers.keys() 
                                                 if obj_id not in self.disappeared_objects and 
                                                 self._should_periodic_save_stats(obj_id))  # NEW
        }
    
    def _should_periodic_save_stats(self, obj_id):
        """Helper method for statistics - check if object is close to periodic save threshold"""
        if self.object_total_frames[obj_id] < self.min_total_frames_for_disappeared:
            return False
        
        last_saved_frame = self.object_last_saved_frame.get(obj_id, 0)
        current_frame = max(metadata['frame_count'] for _, _, metadata, _ in self.object_buffers[obj_id]) if self.object_buffers[obj_id] else 0
        frames_since_last_save = current_frame - last_saved_frame
        
        threshold = max(self.periodic_save_interval // 3, 50) if obj_id not in self.objects_ever_saved else self.periodic_save_interval
        return frames_since_last_save >= (threshold * 0.8)  # 80% of threshold
    
    def cleanup_old_buffers(self, max_age_seconds=300):
        """Cleanup very old buffers to prevent memory issues"""
        current_time = time.time()
        cleanup_objects = []
        
        for obj_id, buffer in self.object_buffers.items():
            if buffer:
                oldest_timestamp = min(entry[3] for entry in buffer)
                if current_time - oldest_timestamp > max_age_seconds:
                    cleanup_objects.append(obj_id)
        
        for obj_id in cleanup_objects:
            logger.info(f"Cleaning up old buffer for object {obj_id}")
            self.cleanup_object_buffer(obj_id)
    
    def debug_disappeared_objects(self, current_frame_count):
        """Debug method to check disappeared objects status"""
        if self.disappeared_objects:
            logger.info("=== DISAPPEARED OBJECTS DEBUG ===")
            for obj_id, disappeared_frame in self.disappeared_objects.items():
                frames_since_disappeared = current_frame_count - disappeared_frame
                total_frames = self.object_total_frames.get(obj_id, 0)
                buffer_size = len(self.object_buffers.get(obj_id, []))
                
                logger.info(f"Object {obj_id}: disappeared_frame={disappeared_frame}, "
                          f"frames_since={frames_since_disappeared}, "
                          f"total_frames={total_frames}, "
                          f"buffer_size={buffer_size}, "
                          f"ready={frames_since_disappeared >= self.save_delay_frames}")
            logger.info("===================================")

class RobustObjectTracker:
    """Enhanced tracker with strict ID consistency for static scenes"""
    
    def __init__(self, max_disappeared=60, strict_mode=False):
        # Core parameters - much more conservative for static scenes
        self.max_disappeared = max_disappeared
        self.strict_mode = strict_mode
        
        # Main tracking structures
        self.objects = {}  # Active objects: {id: (label, bbox, conf, timestamp, features)}
        self.disappeared = {}  # Track disappeared frames
        self.object_memory = {}  # Long-term memory for static objects
        self.next_id = 0
        
        # Advanced features for ID consistency
        self.object_features = {}  # Store visual features for each object
        self.object_positions = defaultdict(list)  # Position history
        self.object_sizes = defaultdict(list)  # Size history  
        self.stable_objects = set()  # Objects confirmed as static
        self.occlusion_memory = {}  # Remember objects during occlusion
        
        # Matching thresholds - tuned for static scene consistency
        self.position_threshold = 90  # Pixels - more lenient for edge cases
        self.size_tolerance = 0.50  # 40% size variation allowed
        self.iou_threshold = 0.25  # Lower IoU threshold for partial overlaps
        self.feature_similarity_threshold = 0.50  # Feature matching threshold
        
        # Static object detection
        self.static_detection_frames = 4  # Frames to confirm static
        self.position_variance_threshold = 30  # Low variance = static object
        
        # Occlusion handling
        self.occlusion_timeout = 50  # Remember occluded objects for 150 frames
        self.partial_visibility_threshold = 0.4  # 30% visible = still trackable
        
        # MODIFIED: Simplified save management - delegate to IntelligentImageBuffer
        self.object_consecutive_frames = defaultdict(int)  # Track consecutive detections
        self.min_consecutive_frames = 3  # Minimum frames before buffering (reduced since we now buffer)
        
        # Performance tracking
        self.frame_count = 0
        self.total_objects_created = 0
        self.id_switches = 0
        
        logger.info(f"Robust tracker initialized - Strict mode: {strict_mode}, Min consecutive frames for buffering: {self.min_consecutive_frames}")
    
    def extract_visual_features(self, image_crop):
        """Extract simple but effective visual features from object crop"""
        try:
            if image_crop.size == 0:
                return None
                
            # Resize to standard size for consistent feature extraction
            crop_resized = cv2.resize(image_crop, (64, 64))
            
            # Color histogram features
            hist_b = cv2.calcHist([crop_resized], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([crop_resized], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([crop_resized], [2], None, [16], [0, 256])
            
            # Edge features
            gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / (64 * 64)
            
            # Combine features
            features = np.concatenate([
                hist_b.flatten(),
                hist_g.flatten(), 
                hist_r.flatten(),
                [edge_ratio]
            ])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def calculate_feature_similarity(self, features1, features2):
        """Calculate similarity between visual features"""
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            # Cosine similarity
            dot_product = np.dot(features1, features2)
            return max(0.0, dot_product)  # Clamp to [0, 1]
        except:
            return 0.0
    
    def get_object_center(self, bbox):
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def get_bbox_area(self, bbox):
        """Calculate bounding box area"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def is_object_static(self, obj_id):
        """Determine if object is static based on position history"""
        if obj_id not in self.object_positions:
            return False
            
        positions = self.object_positions[obj_id]
        if len(positions) < self.static_detection_frames:
            return False
        
        # Calculate position variance
        recent_positions = positions[-self.static_detection_frames:]
        if len(recent_positions) < 2:
            return False
            
        x_coords = [pos[0] for pos in recent_positions]
        y_coords = [pos[1] for pos in recent_positions]
        
        x_var = np.var(x_coords)
        y_var = np.var(y_coords)
        
        # Low variance indicates static object
        is_static = (x_var + y_var) < self.position_variance_threshold
        
        if is_static and obj_id not in self.stable_objects:
            self.stable_objects.add(obj_id)
            logger.info(f"Object {obj_id} confirmed as static")
        
        return is_static
    
    def calculate_overlap_ratio(self, box1, box2):
        """Calculate how much box1 overlaps with box2"""
        intersection_area = self.calculate_iou(box1, box2) * min(
            self.get_bbox_area(box1), self.get_bbox_area(box2)
        )
        box1_area = self.get_bbox_area(box1)
        
        return intersection_area / box1_area if box1_area > 0 else 0.0
    
    def match_detection_to_object(self, detection, obj_data, frame_crop=None):
        """Enhanced matching with multiple criteria for better ID consistency"""
        det_label, det_bbox, det_conf = detection
        obj_label, obj_bbox, obj_conf, obj_timestamp = obj_data[:4]
        
        # Same class requirement
        if det_label != obj_label:
            return 0.0
        
        # Multi-criteria scoring
        scores = {}
        
        # 1. Position similarity
        det_center = self.get_object_center(det_bbox)
        obj_center = self.get_object_center(obj_bbox)
        distance = self.calculate_distance(det_center, obj_center)
        scores['position'] = max(0, 1 - (distance / self.position_threshold))
        
        # 2. Size similarity  
        det_area = self.get_bbox_area(det_bbox)
        obj_area = self.get_bbox_area(obj_bbox)
        if max(det_area, obj_area) > 0:
            size_ratio = min(det_area, obj_area) / max(det_area, obj_area)
            scores['size'] = size_ratio if size_ratio >= self.size_tolerance else 0
        else:
            scores['size'] = 0
        
        # 3. IoU similarity
        iou = self.calculate_iou(det_bbox, obj_bbox)
        scores['iou'] = iou
        
        # 4. Visual feature similarity (if available)
        if frame_crop is not None and len(obj_data) > 4:
            det_features = self.extract_visual_features(frame_crop)
            obj_features = obj_data[4] if len(obj_data) > 4 else None
            
            if det_features is not None and obj_features is not None:
                feature_sim = self.calculate_feature_similarity(det_features, obj_features)
                scores['features'] = feature_sim
            else:
                scores['features'] = 0.5  # Neutral if features unavailable
        else:
            scores['features'] = 0.5
        
        # 5. Static object bonus
        obj_id = next((k for k, v in self.objects.items() if v == obj_data), None)
        if obj_id is not None and obj_id in self.stable_objects:
            scores['static_bonus'] = 1.2  # 20% bonus for static objects
        else:
            scores['static_bonus'] = 1.0
        
        # Weighted combination - emphasize position and features for static scenes
        if self.strict_mode:
            # Strict mode: Higher weight on features and position consistency
            total_score = (
                scores['position'] * 0.3 +
                scores['size'] * 0.2 +
                scores['iou'] * 0.2 +
                scores['features'] * 0.3
            ) * scores['static_bonus']
        else:
            # Standard mode: Balanced scoring
            total_score = (
                scores['position'] * 0.25 +
                scores['size'] * 0.25 +
                scores['iou'] * 0.25 +
                scores['features'] * 0.25
            ) * scores['static_bonus']
        
        # Minimum thresholds for matching
        if scores['position'] < 0.3 and scores['iou'] < self.iou_threshold:
            return 0.0
        
        return min(1.0, total_score)  # Clamp to [0, 1]
    
    def handle_occlusion_recovery(self, unmatched_detections, frame):
        """Attempt to recover objects from occlusion memory"""
        recovered_matches = {}
        
        for det_idx, detection in enumerate(unmatched_detections):
            if det_idx in recovered_matches:
                continue
                
            det_label, det_bbox, det_conf = detection
            det_center = self.get_object_center(det_bbox)
            
            best_match_id = None
            best_score = 0.4  # Higher threshold for occlusion recovery
            
            # Check occlusion memory for potential matches
            for obj_id, (mem_label, mem_bbox, mem_features, mem_timestamp) in self.occlusion_memory.items():
                if det_label != mem_label:
                    continue
                
                # Skip if too much time has passed
                if self.frame_count - mem_timestamp > self.occlusion_timeout:
                    continue
                
                # Position-based matching for occluded objects
                mem_center = self.get_object_center(mem_bbox)
                distance = self.calculate_distance(det_center, mem_center)
                
                if distance < self.position_threshold * 1.5:  # More lenient for occlusion
                    # Try feature matching if available
                    score = 0.5  # Base score
                    
                    if mem_features is not None:
                        try:
                            x1, y1, x2, y2 = det_bbox
                            crop = frame[y1:y2, x1:x2]
                            det_features = self.extract_visual_features(crop)
                            
                            if det_features is not None:
                                feature_sim = self.calculate_feature_similarity(det_features, mem_features)
                                score = feature_sim
                        except:
                            pass
                    
                    if score > best_score:
                        best_score = score
                        best_match_id = obj_id
            
            if best_match_id is not None:
                recovered_matches[det_idx] = best_match_id
                logger.info(f"Recovered object {best_match_id} from occlusion (score: {best_score:.2f})")
        
        return recovered_matches
    
    def update(self, detections, frame=None):
        """Main tracking update with enhanced ID consistency"""
        self.frame_count += 1
        current_time = time.time()
        
        if not detections:
            self._handle_no_detections()
            return self.objects, set()
        
        # Filter valid detections
        valid_detections = []
        for det in detections:
            label, bbox, conf = det
            area = self.get_bbox_area(bbox)
            if area >= 1000 and conf >= args.confidence:  # Minimum size filter
                valid_detections.append(det)
        
        if not valid_detections:
            self._handle_no_detections()
            return self.objects, set()
        
        # Matching phase
        matched_objects = {}
        objects_for_buffering = set()  # MODIFIED: Changed from objects_to_save
        unmatched_detections = list(range(len(valid_detections)))
        
        # Match with existing objects using enhanced criteria
        for obj_id, obj_data in self.objects.items():
            best_match_idx = None
            best_score = 0.55 if self.strict_mode else 0.35  # Higher threshold in strict mode
            
            for i, detection in enumerate(valid_detections):
                if i not in unmatched_detections:
                    continue
                
                # Extract crop for feature matching
                crop = None
                if frame is not None:
                    try:
                        x1, y1, x2, y2 = detection[1]
                        crop = frame[y1:y2, x1:x2]
                    except:
                        pass
                
                score = self.match_detection_to_object(detection, obj_data, crop)
                
                if score > best_score:
                    best_score = score
                    best_match_idx = i
            
            if best_match_idx is not None:
                detection = valid_detections[best_match_idx]
                label, bbox, conf = detection
                
                # Extract and store features
                features = None
                if frame is not None:
                    try:
                        x1, y1, x2, y2 = bbox
                        crop = frame[y1:y2, x1:x2]
                        features = self.extract_visual_features(crop)
                    except:
                        pass
                
                # Update object data
                matched_objects[obj_id] = (label, bbox, conf, current_time, features)
                unmatched_detections.remove(best_match_idx)
                
                # Update consecutive frame counter for matched objects
                self.object_consecutive_frames[obj_id] += 1
                
                # Update position history
                center = self.get_object_center(bbox)
                self.object_positions[obj_id].append(center)
                if len(self.object_positions[obj_id]) > 50:  # Keep recent history
                    self.object_positions[obj_id].pop(0)
                
                # Update size history
                area = self.get_bbox_area(bbox)
                self.object_sizes[obj_id].append(area)
                if len(self.object_sizes[obj_id]) > 20:
                    self.object_sizes[obj_id].pop(0)
                
                # Check if object is static
                self.is_object_static(obj_id)
                
                # Remove from disappeared and occlusion memory
                self.disappeared.pop(obj_id, None)
                self.occlusion_memory.pop(obj_id, None)
                
                # MODIFIED: Check if should buffer image (instead of immediate save)
                if self._should_buffer_object(obj_id, detection):
                    objects_for_buffering.add(obj_id)
        
        # Attempt occlusion recovery for unmatched detections
        if frame is not None and self.occlusion_memory:
            recovery_matches = self.handle_occlusion_recovery(
                [valid_detections[i] for i in unmatched_detections], frame
            )
            
            for det_idx_in_unmatched, obj_id in recovery_matches.items():
                actual_det_idx = unmatched_detections[det_idx_in_unmatched]
                detection = valid_detections[actual_det_idx]
                label, bbox, conf = detection
                
                # Extract features
                features = None
                try:
                    x1, y1, x2, y2 = bbox
                    crop = frame[y1:y2, x1:x2]
                    features = self.extract_visual_features(crop)
                except:
                    pass
                
                # Restore object
                matched_objects[obj_id] = (label, bbox, conf, current_time, features)
                unmatched_detections.remove(actual_det_idx)
                
                # Update consecutive frame counter for recovered objects
                self.object_consecutive_frames[obj_id] += 1
                
                # Update histories
                center = self.get_object_center(bbox)
                self.object_positions[obj_id].append(center)
                
                # Remove from occlusion memory
                self.occlusion_memory.pop(obj_id, None)
                self.disappeared.pop(obj_id, None)
                
                # Check if should buffer
                if self._should_buffer_object(obj_id, detection):
                    objects_for_buffering.add(obj_id)
        
        # Create new objects for remaining unmatched detections (more conservative)
        for i in unmatched_detections:
            detection = valid_detections[i]
            label, bbox, conf = detection
            
            # More stringent criteria for new object creation in strict mode
            if self.strict_mode:
                # Check if this detection is too similar to existing objects
                is_too_similar = False
                det_center = self.get_object_center(bbox)
                
                for existing_obj_data in matched_objects.values():
                    existing_center = self.get_object_center(existing_obj_data[1])
                    distance = self.calculate_distance(det_center, existing_center)
                    
                    if distance < self.position_threshold * 0.65:  # Very close to existing
                        is_too_similar = True
                        break
                
                if is_too_similar:
                    logger.debug(f"Skipping new object creation - too similar to existing")
                    continue
            
            # Create new object
            new_obj_id = self.next_id
            self.next_id += 1
            self.total_objects_created += 1
            
            # Extract features
            features = None
            if frame is not None:
                try:
                    x1, y1, x2, y2 = bbox
                    crop = frame[y1:y2, x1:x2]
                    features = self.extract_visual_features(crop)
                except:
                    pass
            
            matched_objects[new_obj_id] = (label, bbox, conf, current_time, features)
            
            # Initialize consecutive frame counter for new objects
            self.object_consecutive_frames[new_obj_id] = 1
            
            # Initialize histories
            center = self.get_object_center(bbox)
            self.object_positions[new_obj_id] = [center]
            self.object_sizes[new_obj_id] = [self.get_bbox_area(bbox)]
            
            # Check if should buffer (new objects need min consecutive frames)
            if self._should_buffer_object(new_obj_id, detection):
                objects_for_buffering.add(new_obj_id)
            
            logger.info(f"Created new object {new_obj_id}: {label} (total created: {self.total_objects_created})")
        
        # Handle disappeared objects
        self._update_disappeared_objects(matched_objects)
        
        # Update main objects dictionary
        self.objects = matched_objects
        
        # Periodic cleanup
        if self.frame_count % 100 == 0:
            self._cleanup_old_data()
        
        return self.objects, objects_for_buffering  # MODIFIED: Return buffering set instead of save set
    
    def _should_buffer_object(self, obj_id, detection):
        """MODIFIED: Determine if object should be buffered for quality assessment"""
        
        # Define static objects to exclude from buffering/saving
        EXCLUDED_STATIC_OBJECTS = {
            # Indoor static objects
            'tv', 'television', 'refrigerator', 'fridge', 'chair', 'sofa', 'couch', 
            'bed', 'microwave', 'oven', 'dining table', 'diningtable',
            
            # Outdoor static objects  
            'traffic light', 'trafficlight', 'fire hydrant', 'firehydrant', 
            'stop sign', 'stopsign', 'parking meter', 'parkingmeter', 
            'bench', 'mailbox', 'street sign', 'streetsign'
        }

        label = detection[0].lower()  # Convert to lowercase for comparison
    
        # Check if this object type should be excluded
        if label in EXCLUDED_STATIC_OBJECTS:
            logger.debug(f"Object {obj_id} ({label}) excluded from buffering - static object type")
            return False
        
        # Check if object has been consistently tracked for minimum consecutive frames
        if self.object_consecutive_frames[obj_id] < self.min_consecutive_frames:
            logger.debug(f"Object {obj_id} not ready for buffering: {self.object_consecutive_frames[obj_id]}/{self.min_consecutive_frames} consecutive frames")
            return False
        
        # Object is ready for buffering
        return True
    
    def _handle_no_detections(self):
        """Handle frames with no detections"""
        for obj_id in list(self.objects.keys()):
            self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
            
            # Reset consecutive frame counter when object disappears
            self.object_consecutive_frames[obj_id] = 0
            
            # Move to occlusion memory if object was static
            if obj_id in self.stable_objects and obj_id not in self.occlusion_memory:
                obj_data = self.objects[obj_id]
                features = obj_data[4] if len(obj_data) > 4 else None
                self.occlusion_memory[obj_id] = (
                    obj_data[0], obj_data[1], features, self.frame_count
                )
                logger.debug(f"Moved static object {obj_id} to occlusion memory")
            
            # Remove if disappeared too long
            if self.disappeared[obj_id] > self.max_disappeared:
                self._remove_object(obj_id)
    
    def _update_disappeared_objects(self, matched_objects):
        """Update disappeared object tracking"""
        for obj_id in list(self.objects.keys()):
            if obj_id not in matched_objects:
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                
                # Reset consecutive frame counter for disappeared objects
                self.object_consecutive_frames[obj_id] = 0
                
                # Move static objects to occlusion memory
                if (obj_id in self.stable_objects and 
                    obj_id not in self.occlusion_memory and 
                    self.disappeared[obj_id] <= 10):
                    
                    obj_data = self.objects[obj_id]
                    features = obj_data[4] if len(obj_data) > 4 else None
                    self.occlusion_memory[obj_id] = (
                        obj_data[0], obj_data[1], features, self.frame_count
                    )
                
                # Remove if disappeared too long
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._remove_object(obj_id)
    
    def _remove_object(self, obj_id):
        """Clean up removed object data"""
        self.objects.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)
        self.occlusion_memory.pop(obj_id, None)
        self.object_features.pop(obj_id, None)
        self.object_positions.pop(obj_id, None)
        self.object_sizes.pop(obj_id, None)
        self.stable_objects.discard(obj_id)
        
        # Clean up consecutive frame counter
        self.object_consecutive_frames.pop(obj_id, None)
    
    def _cleanup_old_data(self):
        """Periodic cleanup of old data"""
        current_frame = self.frame_count
        
        # Clean old occlusion memory
        expired_occlusions = []
        for obj_id, (_, _, _, timestamp) in self.occlusion_memory.items():
            if current_frame - timestamp > self.occlusion_timeout:
                expired_occlusions.append(obj_id)
        
        for obj_id in expired_occlusions:
            self.occlusion_memory.pop(obj_id, None)
            logger.debug(f"Expired occlusion memory for object {obj_id}")
    
    def get_tracking_stats(self):
        """Get comprehensive tracking statistics"""
        return {
            'active_objects': len(self.objects),
            'disappeared_objects': len(self.disappeared),
            'static_objects': len(self.stable_objects),
            'occluded_objects': len(self.occlusion_memory),
            'total_created': self.total_objects_created,
            'id_switches': self.id_switches,
            'frame_count': self.frame_count,
            'objects_ready_for_buffering': len([obj_id for obj_id, count in self.object_consecutive_frames.items() 
                                             if count >= self.min_consecutive_frames])
        }

class EnhancedImageSaver:
    """MODIFIED: Enhanced image saver with quality-based selection and metadata generation"""
    
    def __init__(self, save_dir, max_workers=2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.save_queue = queue.Queue(maxsize=50)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        
        self.saver_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.saver_thread.start()
        
        # Track saved content to avoid duplicates
        self.saved_hashes = set()
        
        logger.info(f"Enhanced image saver initialized with metadata support")
    
    def _calculate_image_hash(self, image):
        """Calculate hash for duplicate detection"""
        try:
            small = cv2.resize(image, (32, 32))
            return hashlib.md5(small.tobytes()).hexdigest()[:12]
        except:
            return str(time.time())
    
    def _generate_metadata_content(self, metadata):
        """Generate human-readable metadata content"""
        try:
            content = []
            content.append(f"Object Detection Metadata")
            content.append(f"========================")
            content.append(f"")
            content.append(f"Object ID: {metadata['object_id']}")
            content.append(f"Class Label: {metadata['label']}")
            content.append(f"Detection Confidence: {metadata['confidence']:.3f}")
            content.append(f"Frame Number: {metadata['frame_count']}")
            content.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['timestamp']))}")
            
            # NEW: Enhanced metadata for save reason and total frames
            if 'save_reason' in metadata:
                content.append(f"Save Reason: {metadata['save_reason'].upper()}")
            if 'total_frames_when_saved' in metadata:
                content.append(f"Total Frames Tracked: {metadata['total_frames_when_saved']}")
            
            content.append(f"")
            content.append(f"Bounding Box Coordinates:")
            x1, y1, x2, y2 = metadata['bbox']
            content.append(f"  Top-left: ({x1}, {y1})")
            content.append(f"  Bottom-right: ({x2}, {y2})")
            content.append(f"  Width: {x2 - x1} pixels")
            content.append(f"  Height: {y2 - y1} pixels")
            content.append(f"  Area: {(x2 - x1) * (y2 - y1)} pixels")
            content.append(f"")
            content.append(f"Image Quality Assessment:")
            content.append(f"  Overall Quality Score: {metadata['quality_score']:.3f}")
            
            if 'quality_details' in metadata and metadata['quality_details']:
                details = metadata['quality_details']
                content.append(f"  Sharpness Score: {details.get('sharpness', 0):.3f}")
                content.append(f"  Brightness Score: {details.get('brightness', 0):.3f}")
                content.append(f"  Size Quality Score: {details.get('size_quality', 0):.3f}")
                content.append(f"  Face Detected: {'Yes' if details.get('face_detected', False) else 'No'}")
                content.append(f"  Bounding Box Area: {details.get('bbox_area', 0)} pixels")
            
            content.append(f"")
            content.append(f"Selection Criteria:")
            if metadata.get('save_reason') == 'disappeared':
                content.append(f"  This image was selected as the best quality representation")
                content.append(f"  of an object that disappeared from the scene after being")
                content.append(f"  tracked for {metadata.get('total_frames_when_saved', 0)} total frames.")
                content.append(f"  Selection prioritized anomaly detection significance.")
            elif metadata.get('save_reason') == 'periodic':
                content.append(f"  This image was selected during periodic saving of a")
                content.append(f"  long-duration object to capture potential changes or")
                content.append(f"  anomalies over time. Object had been tracked for")
                content.append(f"  {metadata.get('total_frames_when_saved', 0)} total frames at save time.")
            else:
                content.append(f"  This image was selected as the highest quality")
                content.append(f"  representation of this object from multiple frames.")
            
            content.append(f"  Selection based on sharpness, brightness, size,")
            content.append(f"  and face detection (for persons/animals).")
            
            return "\n".join(content)
            
        except Exception as e:
            logger.error(f"Failed to generate metadata content: {e}")
            return f"Metadata generation failed: {e}"
    
    def _save_worker(self):
        """Background thread for saving images and metadata"""
        while self.running:
            try:
                save_data = self.save_queue.get(timeout=1.0)
                if save_data is None:
                    break
                
                image, filename, metadata = save_data
                
                # Check for duplicates
                img_hash = self._calculate_image_hash(image)
                if img_hash in self.saved_hashes:
                    logger.debug(f"Skipped duplicate: {filename}")
                    continue
                
                # Save image
                file_path = self.save_dir / 'images'/filename
                params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # Higher quality for best images
                
                success = cv2.imwrite(str(file_path), image, params)
                
                if success:
                    self.saved_hashes.add(img_hash)
                    
                    # Save metadata file
                    metadata_filename = filename.rsplit('.', 1)[0] + '.txt'
                    metadata_path = self.save_dir /'metadata'/ metadata_filename
                    
                    try:
                        metadata_content = self._generate_metadata_content(metadata)
                        with open(metadata_path, 'w', encoding='utf-8') as f:
                            f.write(metadata_content)
                        
                        save_reason = metadata.get('save_reason', 'standard')
                        logger.info(f"SAVED BEST IMAGE: {filename} (quality: {metadata['quality_score']:.3f}, reason: {save_reason})")
                        logger.info(f"SAVED METADATA: {metadata_filename}")
                        
                    except Exception as e:
                        logger.error(f"Failed to save metadata for {filename}: {e}")
                        # Still consider image save successful even if metadata fails
                    
                else:
                    logger.error(f"Failed to save image: {filename}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Save worker error: {e}")
    
    def save_best_image(self, image, filename, metadata):
        """Queue best quality image for saving with metadata"""
        try:
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'
            
            save_data = (image.copy(), filename, metadata)
            self.save_queue.put_nowait(save_data)
            return True
        except queue.Full:
            logger.warning(f"Save queue full: {filename}")
            return False
    
    def save_initial_scene(self, frame):
        """Save initial scene"""
        try:
            scene_path = self.save_dir / "initial_scene.jpg"
            success = cv2.imwrite(str(scene_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                logger.info(f"Initial scene saved: {scene_path}")
            return success
        except Exception as e:
            logger.error(f"Scene save error: {e}")
            return False
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        self.save_queue.put(None)
        self.saver_thread.join(timeout=5.0)
        self.executor.shutdown(wait=True)

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    global running
    logger.info("Shutdown signal received")
    running = False

def main():
    global running
    running = True
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("Initializing Enhanced Object Tracking System with Anomaly-Aware Image Selection...")
        
        # Create directories
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'metadata'), exist_ok=True)

        # Initialize model
        logger.info(f"Loading model: {args.model}")
        model = YOLO(args.model)
        model.to('cpu')  # Force CPU to avoid CUDA issues
        
        # Initialize components
        image_saver = EnhancedImageSaver(args.save_dir, max_workers=args.max_workers)
        tracker = RobustObjectTracker(strict_mode=args.strict_tracking)
        image_buffer = IntelligentImageBuffer(
            max_buffer_size=args.buffer_size,
            save_delay_frames=args.save_delay,
            min_total_frames_for_disappeared=args.min_total_frames,  # NEW
            periodic_save_interval=args.periodic_save_interval  # NEW
        )

        previous_active_objects = set()
        # Video capture
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        
        # Optimized settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {args.source}")
        
        logger.info("Enhanced anomaly-aware system initialized successfully")
        logger.info(f"Configuration:")
        logger.info(f"  - Image buffer size: {args.buffer_size}")
        logger.info(f"  - Save delay: {args.save_delay} frames")
        logger.info(f"  - Min total frames for disappeared objects: {args.min_total_frames}")
        logger.info(f"  - Periodic save interval: {args.periodic_save_interval} frames")
        
        # Performance tracking
        frame_count = 0
        fps_times = deque(maxlen=30)
        last_stats = time.time()
        scene_saved = False
        
        while running and cap.isOpened():
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            frame_count += 1
            
            # Save initial scene
            if args.save_scene and not scene_saved:
                image_saver.save_initial_scene(frame)
                scene_saved = True
            
            try:
                # YOLO detection
                results = model(
                    frame, 
                    conf=args.confidence,
                    iou=args.iou_threshold,
                    verbose=False,
                    device='cpu',
                    imgsz=args.detection_size
                )
                
                # Extract detections
                detections = []
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, conf, cls in zip(boxes, confs, classes):
                            if conf >= args.confidence:
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Validate bbox
                                if (x2 > x1 and y2 > y1 and 
                                    x1 >= 0 and y1 >= 0 and 
                                    x2 < frame.shape[1] and y2 < frame.shape[0]):
                                    
                                    label = model.names[cls]
                                    detections.append((label, (x1, y1, x2, y2), float(conf)))
                
                # Update tracker
                tracked_objects, objects_for_buffering = tracker.update(detections, frame)
                current_active_objects = set(tracked_objects.keys())
                
                # Buffer images for active objects
                for obj_id in objects_for_buffering:
                    if obj_id in tracked_objects:
                        try:
                            label, bbox, conf, timestamp, _ = tracked_objects[obj_id]
                            x1, y1, x2, y2 = bbox
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size > 0:
                                image_buffer.add_image(obj_id, crop, bbox, conf, label, frame_count)
                        except Exception as e:
                            logger.error(f"Buffering failed for object {obj_id}: {e}")
                
                # FIXED: Improved disappeared object detection
                if frame_count > 1:  # Skip first frame as we need previous state
                    # Find objects that were active in previous frame but not current frame
                    newly_disappeared = previous_active_objects - current_active_objects
                    
                    # Mark newly disappeared objects in buffer
                    for obj_id in newly_disappeared:
                        if obj_id in image_buffer.object_buffers:
                            image_buffer.mark_object_disappeared(obj_id, frame_count)
                            logger.debug(f"Object {obj_id} newly disappeared at frame {frame_count}")
                
                # ADDITIONAL: Also check tracker's disappeared objects and mark them in buffer
                for obj_id, disappeared_count in tracker.disappeared.items():
                    # Mark object as disappeared if it has a disappeared count but isn't yet marked in buffer
                    if obj_id not in image_buffer.disappeared_objects and obj_id in image_buffer.object_buffers:
                        # Calculate when it actually disappeared (current frame - disappeared count)
                        disappeared_frame = frame_count - disappeared_count
                        image_buffer.mark_object_disappeared(obj_id, disappeared_frame)
                        logger.debug(f"Tracker disappeared object {obj_id} marked in buffer (disappeared at frame {disappeared_frame})")
                
                # Update previous objects for next frame
                previous_active_objects = current_active_objects.copy()

                # Process ready objects for saving
                ready_objects = image_buffer.get_objects_ready_for_saving(frame_count)
                for obj_id, save_reason in ready_objects:
                    success = image_buffer.save_and_cleanup_object(obj_id, save_reason, image_saver)
                    if success:
                        if save_reason == 'disappeared':
                            logger.info(f"Disappeared object {obj_id} saved successfully")
                        elif save_reason == 'periodic':
                            logger.info(f"Periodic save completed for long-duration object {obj_id}")
                
                if frame_count % 100 == 0 and image_buffer.disappeared_objects:
                    image_buffer.debug_disappeared_objects(frame_count)

                # NEW: Periodic buffer cleanup
                if frame_count % 200 == 0:  # Every 200 frames (about 20 seconds at 10fps)
                    image_buffer.cleanup_old_buffers()
                
                # Visualization (if GUI enabled)
                if not args.no_gui:
                    display_frame = frame.copy()
                    
                    # Draw tracking results
                    for obj_id, obj_data in tracked_objects.items():
                        try:
                            label, bbox, conf, timestamp, _ = obj_data
                            x1, y1, x2, y2 = bbox
                            
                            # Color coding
                            if obj_id in tracker.stable_objects:
                                color = (0, 255, 0)  # Green for static
                                status = "STATIC"
                            elif obj_id in tracker.occlusion_memory:
                                color = (255, 0, 255)  # Magenta for occluded
                                status = "OCCLUDED"
                            else:
                                color = (255, 255, 0)  # Cyan for tracking
                                status = "TRACKING"
                            
                            # Draw bounding box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Show enhanced status information
                            consecutive_frames = tracker.object_consecutive_frames.get(obj_id, 0)
                            ready_for_buffer = consecutive_frames >= tracker.min_consecutive_frames
                            
                            # Check buffer status
                            in_buffer = obj_id in image_buffer.object_buffers
                            buffer_size = len(image_buffer.object_buffers.get(obj_id, []))
                            total_frames_seen = image_buffer.object_total_frames.get(obj_id, 0)
                            
                            # Check if object is ready for periodic save
                            ready_for_periodic = (obj_id not in image_buffer.disappeared_objects and 
                                                image_buffer._should_periodic_save(obj_id, frame_count))
                            
                            if ready_for_buffer and in_buffer:
                                if ready_for_periodic:
                                    buffer_status = f"BUF:{buffer_size}|PERIODIC"
                                else:
                                    buffer_status = f"BUF:{buffer_size}|T{total_frames_seen}"
                            elif ready_for_buffer:
                                buffer_status = "READY"
                            else:
                                buffer_status = f"{consecutive_frames}/{tracker.min_consecutive_frames}"
                            
                            # Draw label with enhanced information
                            text = f"{label}[{obj_id}] {conf:.2f} - {status} ({buffer_status})"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            
                            # Background for text
                            cv2.rectangle(display_frame, 
                                        (x1, y1 - text_size[1] - 10), 
                                        (x1 + text_size[0] + 4, y1), 
                                        color, -1)
                            
                            cv2.putText(display_frame, text, (x1 + 2, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            # Draw center point
                            center = tracker.get_object_center(bbox)
                            cv2.circle(display_frame, center, 3, color, -1)
                            
                            # NEW: Visual indicator for objects ready for periodic save
                            if ready_for_periodic:
                                cv2.circle(display_frame, (x2 - 10, y1 + 10), 5, (0, 255, 255), -1)  # Yellow dot
                            
                        except Exception as e:
                            logger.error(f"Visualization error for {obj_id}: {e}")
                    
                    # Draw enhanced statistics
                    tracker_stats = tracker.get_tracking_stats()
                    buffer_stats = image_buffer.get_buffer_stats()
                    
                    stats_text = [
                        f"Frame: {frame_count}",
                        f"Active: {tracker_stats['active_objects']}",
                        f"Static: {tracker_stats['static_objects']}",
                        f"Buffered Images: {buffer_stats['total_buffered_images']}",
                        f"Images Saved: {buffer_stats['total_images_saved']}",
                        f"Disappeared Saved: {buffer_stats['disappeared_objects_saved']}",  # NEW
                        f"Periodic Saves: {buffer_stats['periodic_saves']}",  # NEW
                        f"Objects Processed: {buffer_stats['objects_processed']}"
                    ]
                    
                    y_pos = 30
                    for text in stats_text:
                        cv2.putText(display_frame, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                        y_pos += 30
                    
                    # FPS display
                    frame_time = time.time() - frame_start
                    if frame_time > 0:
                        fps_times.append(1.0 / frame_time)
                    
                    if fps_times:
                        avg_fps = sum(fps_times) / len(fps_times)
                        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", 
                                  (10, display_frame.shape[0] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow("Enhanced Anomaly-Aware Object Tracking", display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        running = False
                        break
                    elif key == ord('r'):  # Reset tracker
                        tracker = RobustObjectTracker(strict_mode=args.strict_tracking)
                        image_buffer = IntelligentImageBuffer(
                            max_buffer_size=args.buffer_size,
                            save_delay_frames=args.save_delay,
                            min_total_frames_for_disappeared=args.min_total_frames,
                            periodic_save_interval=args.periodic_save_interval
                        )
                        logger.info("Tracker and buffer reset")
                    elif key == ord('s'):  # Save current scene
                        scene_name = f"scene_{int(time.time())}.jpg"
                        cv2.imwrite(os.path.join(args.save_dir, scene_name), frame)
                        logger.info(f"Scene saved: {scene_name}")
                    elif key == ord('f'):  # Force save all buffered objects
                        logger.info("Force saving all buffered objects...")
                        for obj_id in list(image_buffer.object_buffers.keys()):
                            if len(image_buffer.object_buffers[obj_id]) > 0:
                                success = image_buffer.save_and_cleanup_object(obj_id, 'manual', image_saver)
                                if success:
                                    logger.info(f"Manually saved object {obj_id}")
                
                # Periodic statistics logging
                if time.time() - last_stats > 10:
                    tracker_stats = tracker.get_tracking_stats()
                    buffer_stats = image_buffer.get_buffer_stats()
                    avg_fps = sum(fps_times) / len(fps_times) if fps_times else 0
                    
                    logger.info(f"Tracking: {tracker_stats}")
                    logger.info(f"Enhanced Buffer: {buffer_stats} | FPS: {avg_fps:.1f}")
                    last_stats = time.time()
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        logger.info("Shutting down enhanced anomaly-aware system...")
        
        # NEW: Enhanced shutdown - save all remaining buffered images with appropriate reasons
        if 'image_buffer' in locals():
            logger.info("Saving remaining buffered images...")
            
            # Save all remaining objects as 'shutdown' saves
            saved_count = 0
            for obj_id in list(image_buffer.object_buffers.keys()):
                try:
                    # Determine save reason for remaining objects
                    if obj_id in image_buffer.disappeared_objects:
                        save_reason = 'disappeared'
                    elif image_buffer.object_total_frames.get(obj_id, 0) >= image_buffer.min_total_frames_for_disappeared:
                        save_reason = 'shutdown'
                    else:
                        save_reason = 'shutdown_minimal'  # Even save objects with few frames during shutdown
                    
                    success = image_buffer.save_and_cleanup_object(obj_id, save_reason, image_saver)
                    if success:
                        saved_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to save remaining image for object {obj_id}: {e}")
            
            logger.info(f"Saved {saved_count} remaining objects during shutdown")
        
        # Cleanup
        if 'cap' in locals():
            cap.release()
        
        if not args.no_gui:
            cv2.destroyAllWindows()
        
        if 'image_saver' in locals():
            image_saver.shutdown()
        
        # Save enhanced final statistics
        if 'tracker' in locals() and 'image_buffer' in locals():
            try:
                tracker_stats = tracker.get_tracking_stats()
                buffer_stats = image_buffer.get_buffer_stats()
                
                final_stats = {
                    'tracker_stats': tracker_stats,
                    'buffer_stats': buffer_stats,
                    'total_frames': frame_count,
                    'configuration': {
                        'strict_mode': args.strict_tracking,
                        'confidence_threshold': args.confidence,
                        'buffer_size': args.buffer_size,
                        'save_delay_frames': args.save_delay,
                        'min_total_frames_for_disappeared': args.min_total_frames,  # NEW
                        'periodic_save_interval': args.periodic_save_interval,  # NEW
                        'min_consecutive_frames_for_buffering': tracker.min_consecutive_frames
                    },
                    'session_duration': time.time() - (last_stats - 10),
                    'anomaly_detection_metrics': {  # NEW: Anomaly-specific metrics
                        'disappeared_objects_saved': buffer_stats.get('disappeared_objects_saved', 0),
                        'periodic_saves': buffer_stats.get('periodic_saves', 0),
                        'objects_ever_saved': buffer_stats.get('objects_ever_saved', 0),
                        'total_anomaly_events': buffer_stats.get('disappeared_objects_saved', 0) + buffer_stats.get('periodic_saves', 0)
                    }
                }
                
                stats_file = os.path.join(args.save_dir, f"anomaly_aware_final_stats_{int(time.time())}.json")
                
                with open(stats_file, 'w') as f:
                    json.dump(final_stats, f, indent=2)
                
                logger.info(f"Enhanced final statistics saved: {stats_file}")
                logger.info(f"Anomaly-Aware Session Summary:")
                logger.info(f"  - Total frames processed: {frame_count}")
                logger.info(f"  - Total objects created: {tracker_stats.get('total_created', 0)}")
                logger.info(f"  - Total images buffered: {buffer_stats.get('total_images_buffered', 0)}")
                logger.info(f"  - Best images saved: {buffer_stats.get('total_images_saved', 0)}")
                logger.info(f"  - Disappeared objects saved: {buffer_stats.get('disappeared_objects_saved', 0)}")
                logger.info(f"  - Periodic saves: {buffer_stats.get('periodic_saves', 0)}")
                logger.info(f"  - Objects fully processed: {buffer_stats.get('objects_processed', 0)}")
                logger.info(f"  - Total anomaly events detected: {buffer_stats.get('disappeared_objects_saved', 0) + buffer_stats.get('periodic_saves', 0)}")
                
            except Exception as e:
                logger.error(f"Failed to save enhanced final stats: {e}")

if __name__ == "__main__":
    main()
