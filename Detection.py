import cv2
import time
import os
import argparse
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import math
import hashlib
import json
from datetime import datetime
import threading
from queue import Queue

# Fixed argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
parser.add_argument('-s', '--source', type=str, default='0', help='Video source')
parser.add_argument('--save_dir', type=str, default='captures', help='Directory to save captures')
parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
parser.add_argument('--iou_threshold', type=float, default=0.4, help='IoU threshold for NMS')
parser.add_argument('--pi_mode', action='store_true', help='Enable Raspberry Pi optimizations')
parser.add_argument('--depth_camera', action='store_true', help='Use depth camera if available')
parser.add_argument('--scene_capture', action='store_true', help='Capture initial scene')
args = parser.parse_args()

class KalmanFilter:
    """Simple Kalman filter for object position prediction"""
    def __init__(self):
        self.dt = 1.0  # Time step
        self.A = np.array([[1, 0, self.dt, 0],
                          [0, 1, 0, self.dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)  # State transition matrix
        
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=np.float32)  # Observation matrix
        
        self.Q = np.eye(4, dtype=np.float32) * 0.1  # Process noise
        self.R = np.eye(2, dtype=np.float32) * 1.0  # Measurement noise
        
        self.x = np.zeros((4, 1), dtype=np.float32)  # State [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * 1000  # Error covariance
        
        self.initialized = False
    
    def predict(self):
        if not self.initialized:
            return None
        
        try:
            self.x = np.dot(self.A, self.x)
            self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
            return self.x[:2].flatten()
        except Exception as e:
            print(f"[KALMAN] Prediction error: {e}")
            return None
    
    def update(self, measurement):
        try:
            measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)
            
            if not self.initialized:
                self.x[:2] = measurement
                self.initialized = True
                return
            
            y = measurement - np.dot(self.H, self.x)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            
            self.x = self.x + np.dot(K, y)
            self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        except Exception as e:
            print(f"[KALMAN] Update error: {e}")

class EnhancedObjectTracker:
    def __init__(self, max_disappeared=30, movement_threshold=25, stationary_threshold=45, 
                 min_stable_frames=3, depth_enabled=False):
        
        # Core tracking parameters
        self.max_disappeared = max_disappeared
        self.movement_threshold = movement_threshold
        self.stationary_threshold = stationary_threshold
        self.min_stable_frames = min_stable_frames
        self.depth_enabled = depth_enabled
        
        # Object tracking data structures
        self.objects = {}  # Active tracked objects
        self.disappeared = {}  # Recently disappeared objects
        self.ghost_objects = {}  # Objects for occlusion recovery
        self.object_filters = {}  # Kalman filters for each object
        
        # Enhanced tracking features
        self.object_movement_history = defaultdict(lambda: deque(maxlen=30))
        self.object_capture_flags = {}  # Prevent redundant captures
        self.object_initial_positions = {}  # Starting positions
        self.object_stationary_counts = defaultdict(int)  # Stationary frame counts
        self.object_areas = defaultdict(lambda: deque(maxlen=10))  # Area consistency tracking
        self.object_depths = defaultdict(lambda: deque(maxlen=10)) if depth_enabled else None
        
        # Scene context and persistence buffer
        self.scene_buffer = deque(maxlen=50)  # Optimized buffer size
        self.spatial_grid = defaultdict(list)  # Spatial indexing for faster lookup
        self.grid_size = 50  # Grid cell size in pixels
        
        # Timing and frame management
        self.frame_count = 0
        self.next_id = 0
        self.last_cleanup = time.time()
        self.cleanup_interval = 10  # Cleanup every 10 seconds
        
        # Quality and capture parameters
        self.min_detection_area = 800 if args.pi_mode else 1200
        self.min_bbox_size = 25 if args.pi_mode else 35
        self.confidence_threshold = args.confidence
        self.area_change_threshold = 0.3  # 30% area change tolerance
        self.position_smoothing_factor = 0.7
        
        # Background subtractor for motion detection
        try:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=50, history=200
            )
        except Exception as e:
            print(f"[INIT] Background subtractor failed: {e}")
            self.bg_subtractor = None
        
        # Scene capture setup
        self.scene_captured = False
        self.initial_scene = None
        
        print(f"[INIT] Enhanced tracker initialized with Pi-mode: {args.pi_mode}")
    
    def capture_initial_scene(self, frame):
        """Capture the initial scene for reference"""
        if not self.scene_captured and args.scene_capture:
            try:
                self.initial_scene = frame.copy()
                scene_path = os.path.join(args.save_dir, f"initial_scene_{int(time.time())}.jpg")
                cv2.imwrite(scene_path, self.initial_scene, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.scene_captured = True
                print(f"[SCENE] Initial scene captured: {scene_path}")
            except Exception as e:
                print(f"[SCENE] Failed to capture initial scene: {e}")
    
    def get_spatial_grid_key(self, center):
        """Get spatial grid key for efficient neighbor lookup"""
        try:
            return (int(center[0]) // self.grid_size, int(center[1]) // self.grid_size)
        except:
            return (0, 0)
    
    def update_spatial_grid(self, obj_id, center, operation='add'):
        """Update spatial indexing grid"""
        try:
            grid_key = self.get_spatial_grid_key(center)
            
            if operation == 'add':
                if obj_id not in self.spatial_grid[grid_key]:
                    self.spatial_grid[grid_key].append(obj_id)
            elif operation == 'remove':
                if obj_id in self.spatial_grid[grid_key]:
                    self.spatial_grid[grid_key].remove(obj_id)
                    if not self.spatial_grid[grid_key]:
                        del self.spatial_grid[grid_key]
        except Exception as e:
            print(f"[GRID] Spatial grid update error: {e}")
    
    def find_nearby_objects(self, center, radius=100):
        """Find objects near a given center using spatial indexing"""
        nearby_objects = []
        try:
            grid_key = self.get_spatial_grid_key(center)
            
            # Check surrounding grid cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_key = (grid_key[0] + dx, grid_key[1] + dy)
                    if check_key in self.spatial_grid:
                        for obj_id in self.spatial_grid[check_key]:
                            if obj_id in self.objects:
                                obj_center = self.get_object_center(self.objects[obj_id][1])
                                distance = self.calculate_distance(center, obj_center)
                                if distance <= radius:
                                    nearby_objects.append((obj_id, distance))
        except Exception as e:
            print(f"[GRID] Nearby objects search error: {e}")
        
        return sorted(nearby_objects, key=lambda x: x[1])
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        try:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        except:
            return float('inf')
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        try:
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
        except:
            return 0.0
    
    def get_object_center(self, bbox):
        """Get center point of bounding box"""
        try:
            return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        except:
            return (0, 0)
    
    def get_bbox_area(self, bbox):
        """Calculate bounding box area"""
        try:
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        except:
            return 0
    
    def is_motion_detected(self, frame, bbox):
        """Verify motion using background subtraction"""
        if self.bg_subtractor is None:
            return True  # Default to motion detected if no background subtractor
        
        try:
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return True
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return True
            
            fg_mask = self.bg_subtractor.apply(roi)
            motion_pixels = cv2.countNonZero(fg_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
            return motion_ratio > 0.1  # 10% motion threshold
        except Exception as e:
            print(f"[MOTION] Motion detection error: {e}")
            return True  # Default to motion detected if error
    
    def calculate_total_movement(self, obj_id, current_center):
        """Calculate total movement distance for an object"""
        try:
            if obj_id not in self.object_initial_positions:
                self.object_initial_positions[obj_id] = current_center
                return 0.0
            
            # Add current position to history
            self.object_movement_history[obj_id].append(current_center)
            
            # Calculate total distance from initial position
            initial_pos = self.object_initial_positions[obj_id]
            total_distance = self.calculate_distance(initial_pos, current_center)
            
            return total_distance
        except Exception as e:
            print(f"[MOVEMENT] Movement calculation error: {e}")
            return 0.0
    
    def should_capture_object(self, obj_id, current_center, bbox, frame):
        """Determine if object should be captured based on movement and other criteria"""
        try:
            # Check if already captured
            if self.object_capture_flags.get(obj_id, False):
                total_movement = self.calculate_total_movement(obj_id, current_center)
                
                # Re-capture if significant movement from last capture position
                if total_movement > self.movement_threshold * 2:
                    self.object_capture_flags[obj_id] = False  # Reset for re-capture
                    return True
                return False
            
            # Calculate movement since initial detection
            total_movement = self.calculate_total_movement(obj_id, current_center)
            
            # Check stationary duration
            if total_movement < self.movement_threshold:
                self.object_stationary_counts[obj_id] += 1
                if self.object_stationary_counts[obj_id] > self.stationary_threshold:
                    return False  # Too stationary, don't capture
            else:
                self.object_stationary_counts[obj_id] = 0  # Reset stationary count
            
            # Verify motion using background subtraction
            if not self.is_motion_detected(frame, bbox):
                return False
            
            # Check if minimum movement threshold is met
            if total_movement >= self.movement_threshold:
                return True
            
            return False
        except Exception as e:
            print(f"[CAPTURE] Capture decision error: {e}")
            return False
    
    def area_consistency_check(self, obj_id, current_area):
        """Check if object area is consistent over time"""
        try:
            self.object_areas[obj_id].append(current_area)
            
            if len(self.object_areas[obj_id]) < 3:
                return True  # Not enough data
            
            areas = list(self.object_areas[obj_id])
            avg_area = sum(areas) / len(areas)
            
            # Check if current area is within acceptable range
            if avg_area == 0:
                return True
            
            area_change = abs(current_area - avg_area) / avg_area
            return area_change < self.area_change_threshold
        except Exception as e:
            print(f"[AREA] Area consistency check error: {e}")
            return True
    
    def match_detection_to_object(self, detection, obj_data):
        """Enhanced object matching with multiple criteria"""
        try:
            det_label, det_bbox, det_conf = detection
            
            # Fixed: Handle the object data format correctly
            if len(obj_data) == 4:
                obj_label, obj_bbox, obj_conf, obj_timestamp = obj_data
            else:
                return 0.0
            
            if det_label != obj_label:
                return 0.0
            
            # Calculate various similarity metrics
            iou_score = self.calculate_iou(det_bbox, obj_bbox)
            
            det_center = self.get_object_center(det_bbox)
            obj_center = self.get_object_center(obj_bbox)
            distance = self.calculate_distance(det_center, obj_center)
            
            det_area = self.get_bbox_area(det_bbox)
            obj_area = self.get_bbox_area(obj_bbox)
            
            if max(det_area, obj_area) == 0:
                return 0.0
            
            size_ratio = min(det_area, obj_area) / max(det_area, obj_area)
            conf_similarity = 1.0 - abs(det_conf - obj_conf)
            
            # Multi-criteria scoring
            if distance > 100 or size_ratio < 0.3 or iou_score < 0.1:
                return 0.0
            
            distance_score = max(0, 1.0 - (distance / 80.0))
            
            total_score = (
                iou_score * 0.3 +
                distance_score * 0.4 +
                size_ratio * 0.2 +
                conf_similarity * 0.1
            )
            
            return total_score
        except Exception as e:
            print(f"[MATCH] Object matching error: {e}")
            return 0.0
    
    def update(self, detections, frame=None):
        """Main tracking update with enhanced features"""
        try:
            current_time = time.time()
            self.frame_count += 1
            
            # Capture initial scene
            if frame is not None:
                self.capture_initial_scene(frame)
            
            # Handle empty detections
            if not detections:
                self._handle_no_detections()
                return self.objects, set()
            
            # Filter valid detections
            valid_detections = self._filter_detections(detections, frame)
            
            if not valid_detections:
                self._handle_no_detections()
                return self.objects, set()
            
            # Match detections with existing objects
            matched_objects = {}
            objects_to_save = set()
            unmatched_detections = list(range(len(valid_detections)))
            
            # Enhanced matching with Kalman filtering
            for obj_id, obj_data in self.objects.items():
                best_match_idx = None
                best_score = 0.3  # Minimum matching threshold
                
                for i, detection in enumerate(valid_detections):
                    if i not in unmatched_detections:
                        continue
                    
                    score = self.match_detection_to_object(detection, obj_data)
                    
                    if score > best_score:
                        best_score = score
                        best_match_idx = i
                
                if best_match_idx is not None:
                    detection = valid_detections[best_match_idx]
                    label, bbox, conf = detection
                    
                    # Update Kalman filter
                    center = self.get_object_center(bbox)
                    if obj_id not in self.object_filters:
                        self.object_filters[obj_id] = KalmanFilter()
                    self.object_filters[obj_id].update(center)
                    
                    # Update object data
                    matched_objects[obj_id] = (label, bbox, conf, current_time)
                    unmatched_detections.remove(best_match_idx)
                    
                    # Update spatial grid
                    self.update_spatial_grid(obj_id, center, 'add')
                    
                    # Check if should capture
                    if frame is not None and self.should_capture_object(obj_id, center, bbox, frame):
                        # Perform area consistency check
                        current_area = self.get_bbox_area(bbox)
                        if self.area_consistency_check(obj_id, current_area):
                            objects_to_save.add(obj_id)
                            self.object_capture_flags[obj_id] = True
                    
                    # Remove from disappeared
                    self.disappeared.pop(obj_id, None)
            
            # Create new objects from unmatched detections
            for i in unmatched_detections:
                detection = valid_detections[i]
                label, bbox, conf = detection
                center = self.get_object_center(bbox)
                
                # Check for nearby existing objects to avoid duplicates
                nearby_objects = self.find_nearby_objects(center, radius=50)
                if not nearby_objects:  # No nearby objects, create new
                    new_obj_id = self.next_id
                    self.next_id += 1
                    
                    matched_objects[new_obj_id] = (label, bbox, conf, current_time)
                    
                    # Initialize Kalman filter
                    self.object_filters[new_obj_id] = KalmanFilter()
                    self.object_filters[new_obj_id].update(center)
                    
                    # Update spatial grid
                    self.update_spatial_grid(new_obj_id, center, 'add')
                    
                    # Check initial capture criteria
                    if frame is not None:
                        current_area = self.get_bbox_area(bbox)
                        self.area_consistency_check(new_obj_id, current_area)
                        
                        # Always capture new objects that meet criteria
                        if (current_area >= self.min_detection_area and 
                            self.is_motion_detected(frame, bbox)):
                            objects_to_save.add(new_obj_id)
                            self.object_capture_flags[new_obj_id] = True
            
            # Handle disappeared objects
            self._update_disappeared_objects(matched_objects)
            
            # Periodic cleanup
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_data()
                self.last_cleanup = current_time
            
            # Update main objects dictionary
            self.objects = matched_objects
            
            return self.objects, objects_to_save
            
        except Exception as e:
            print(f"[UPDATE] Tracker update error: {e}")
            return self.objects, set()
    
    def _filter_detections(self, detections, frame):
        """Filter and validate detections"""
        valid_detections = []
        
        for detection in detections:
            try:
                label, bbox, conf = detection
                area = self.get_bbox_area(bbox)
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                
                # Size and confidence checks
                if (area >= self.min_detection_area and 
                    conf >= self.confidence_threshold and
                    bbox_w >= self.min_bbox_size and 
                    bbox_h >= self.min_bbox_size):
                    
                    # Additional motion verification if frame available
                    if frame is None or self.is_motion_detected(frame, bbox):
                        valid_detections.append(detection)
                        
            except Exception as e:
                print(f"[FILTER] Detection filtering failed: {e}")
                continue
        
        return valid_detections
    
    def _handle_no_detections(self):
        """Handle frames with no valid detections"""
        for obj_id in list(self.objects.keys()):
            self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
            
            if self.disappeared[obj_id] > self.max_disappeared:
                # Clean up object data
                self._remove_object(obj_id)
    
    def _update_disappeared_objects(self, matched_objects):
        """Update tracking for disappeared objects"""
        for obj_id in list(self.objects.keys()):
            if obj_id not in matched_objects:
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                
                # Move to ghost memory for occlusion recovery
                if self.disappeared[obj_id] <= 10:
                    self.ghost_objects[obj_id] = self.objects[obj_id]
                
                # Remove if disappeared too long
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._remove_object(obj_id)
    
    def _remove_object(self, obj_id):
        """Clean up all data for a removed object"""
        # Remove from main tracking
        self.objects.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)
        self.ghost_objects.pop(obj_id, None)
        
        # Clean up enhanced tracking data
        self.object_filters.pop(obj_id, None)
        self.object_movement_history.pop(obj_id, None)
        self.object_capture_flags.pop(obj_id, None)
        self.object_initial_positions.pop(obj_id, None)
        self.object_stationary_counts.pop(obj_id, None)
        self.object_areas.pop(obj_id, None)
        
        if self.depth_enabled and self.object_depths:
            self.object_depths.pop(obj_id, None)
    
    def _cleanup_old_data(self):
        """Periodic cleanup of old tracking data"""
        try:
            current_time = time.time()
            
            # Clean up old scene buffer entries
            while (self.scene_buffer and 
                   current_time - self.scene_buffer[0].get('timestamp', 0) > 60):
                self.scene_buffer.popleft()
            
            # Clean up empty spatial grid cells
            empty_cells = [key for key, value in self.spatial_grid.items() if not value]
            for key in empty_cells:
                del self.spatial_grid[key]
        except Exception as e:
            print(f"[CLEANUP] Cleanup error: {e}")
    
    def get_tracking_stats(self):
        """Get comprehensive tracking statistics"""
        stats = {
            'active_objects': len(self.objects),
            'disappeared_objects': len(self.disappeared),
            'ghost_objects': len(self.ghost_objects),
            'total_captured': sum(1 for flag in self.object_capture_flags.values() if flag),
            'frame_count': self.frame_count,
            'spatial_grid_cells': len(self.spatial_grid)
        }
        return stats

# Asynchronous image saving
def async_image_saver(save_queue, save_dir):
    """Background thread for saving images asynchronously"""
    while True:
        try:
            save_data = save_queue.get(timeout=1)
            if save_data is None:  # Shutdown signal
                break
            
            crop, filename, obj_id, label = save_data
            filepath = os.path.join(save_dir, filename)
            
            # Optimize image for storage
            if args.pi_mode:
                # Resize if too large
                h, w = crop.shape[:2]
                if max(h, w) > 416:  # Optimized max size
                    scale = 416 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Use lower quality to save space
                quality = [cv2.IMWRITE_JPEG_QUALITY, 85]
            else:
                quality = [cv2.IMWRITE_JPEG_QUALITY, 95]
            
            success = cv2.imwrite(filepath, crop, quality)
            if success:
                print(f"[SAVED] {filename} - ID:{obj_id} - Size: {crop.shape[1]}x{crop.shape[0]}")
            else:
                print(f"[ERROR] Failed to save {filename}")
            
            save_queue.task_done()
            
        except Exception as e:
            if "Empty" not in str(e):  # Ignore timeout exceptions
                print(f"[ERROR] Image save failed: {e}")

# Main execution
def main():
    try:
        # Initialize model with CPU-only execution
        print("[INIT] Loading YOLO model...")
        model = YOLO(args.model)
        
        # Force CPU usage to avoid CUDA errors
        model.to('cpu')
        
        if args.pi_mode:
            # Pi-specific optimizations
            try:
                model.fuse()  # Fuse layers for speed
            except:
                print("[PI-OPT] Model fusion failed, continuing without fusion")
            os.environ['OMP_NUM_THREADS'] = '4'  # Optimize for Pi's 4 cores
        
        # Setup directories
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Initialize enhanced tracker
        tracker = EnhancedObjectTracker(
            depth_enabled=args.depth_camera,
            max_disappeared=20 if args.pi_mode else 30,
            movement_threshold=20 if args.pi_mode else 25
        )
        
        # Setup video capture
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        
        if args.pi_mode:
            # Pi Camera optimized settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            raise ValueError("Cannot open video source")
        
        # Start async image saver
        save_queue = Queue(maxsize=20)
        saver_thread = threading.Thread(
            target=async_image_saver, 
            args=(save_queue, args.save_dir),
            daemon=True
        )
        saver_thread.start()
        
        print(f"[START] Enhanced tracking started - Pi mode: {args.pi_mode}")
        
        frame_count = 0
        fps_counter = deque(maxlen=30)
        last_stats_time = time.time()
        
        while cap.isOpened():
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Dynamic processing based on Pi mode
            if args.pi_mode:
                should_process = (frame_count % 2 == 0)  # Every 2nd frame
            else:
                should_process = True  # Every frame
            
            if not should_process:
                cv2.imshow("Enhanced Object Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            try:
                # YOLO detection with fixed device parameter
                results = model(
                    frame, 
                    conf=args.confidence, 
                    iou=args.iou_threshold, 
                    verbose=False,
                    device='cpu'  # Always use CPU to avoid CUDA errors
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
                                
                                # Validate bounding box
                                if (x2 > x1 and y2 > y1 and 
                                    x1 >= 0 and y1 >= 0 and 
                                    x2 < frame.shape[1] and y2 < frame.shape[0]):
                                    
                                    label = model.names[cls]
                                    detections.append((label, (x1, y1, x2, y2), float(conf)))
                
                # Update tracker
                tracked_objects, objects_to_save = tracker.update(detections, frame)
                
                # Visualization and saving
                for obj_id, obj_data in tracked_objects.items():
                    try:
                        label, bbox, conf, timestamp = obj_data
                        x1, y1, x2, y2 = bbox
                        center = tracker.get_object_center(bbox)
                        
                        # Calculate movement for display
                        total_movement = tracker.calculate_total_movement(obj_id, center)
                        is_captured = tracker.object_capture_flags.get(obj_id, False)
                        
                        # Color coding based on status
                        if is_captured:
                            color = (0, 255, 0)  # Green - captured
                            status = "CAPTURED"
                        elif total_movement > tracker.movement_threshold:
                            color = (0, 165, 255)  # Orange - moving
                            status = f"MOVING({total_movement:.1f})"
                        else:
                            color = (255, 255, 0)  # Cyan - tracking
                            status = f"TRACK({total_movement:.1f})"
                        
                        # Draw bounding box and tracking info
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Enhanced labeling with tracking info
                        text_lines = [
                            f"{label}[{obj_id}] {conf:.2f}",
                            f"{status}"
                        ]
                        
                        # Add Kalman prediction if available
                        if obj_id in tracker.object_filters:
                            predicted = tracker.object_filters[obj_id].predict()
                            if predicted is not None:
                                pred_x, pred_y = map(int, predicted)
                                cv2.circle(frame, (pred_x, pred_y), 3, (255, 0, 255), -1)  # Magenta prediction
                        
                        # Draw text with background for better visibility
                        y_offset = y1 - 10
                        for line in text_lines:
                            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(frame, (x1, y_offset - text_size[1] - 2), 
                                        (x1 + text_size[0] + 4, y_offset + 2), color, -1)
                            cv2.putText(frame, line, (x1 + 2, y_offset), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            y_offset -= (text_size[1] + 4)
                        
                        # Draw movement trail
                        if obj_id in tracker.object_movement_history:
                            trail = list(tracker.object_movement_history[obj_id])
                            if len(trail) > 1:
                                for i in range(1, len(trail)):
                                    cv2.line(frame, trail[i-1], trail[i], color, 1)
                    
                    except Exception as e:
                        print(f"[VIZ] Visualization error for object {obj_id}: {e}")
                        continue
                
                # Asynchronous image saving
                for obj_id in objects_to_save:
                    if obj_id in tracked_objects:
                        try:
                            label, bbox, conf, _ = tracked_objects[obj_id]
                            x1, y1, x2, y2 = bbox
                            
                            # Extract and prepare crop
                            crop = frame[y1:y2, x1:x2].copy()
                            if crop.size > 0:
                                # Generate unique filename with movement info
                                movement = tracker.calculate_total_movement(
                                    obj_id, tracker.get_object_center(bbox)
                                )
                                timestamp_str = str(int(time.time() * 1000))
                                filename = f"{label}_ID{obj_id}_mv{movement:.1f}_{timestamp_str}.jpg"
                                
                                # Add to save queue (non-blocking)
                                if not save_queue.full():
                                    save_queue.put((crop, filename, obj_id, label))
                                else:
                                    print(f"[WARNING] Save queue full, skipping {filename}")
                                    
                        except Exception as e:
                            print(f"[ERROR] Failed to prepare save for object {obj_id}: {e}")
                
                # Enhanced status display
                stats = tracker.get_tracking_stats()
                status_lines = [
                    f"Frame: {frame_count} | Active: {stats['active_objects']} | "
                    f"Disappeared: {stats['disappeared_objects']} | "
                    f"Captured: {stats['total_captured']}",
                    f"Grid Cells: {stats['spatial_grid_cells']} | "
                    f"Pi Mode: {args.pi_mode} | "
                    f"Scene: {'✓' if tracker.scene_captured else '✗'}"
                ]
                
                y_pos = 25
                for line in status_lines:
                    # Background for better visibility
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (10, y_pos - text_size[1] - 5), 
                                (10 + text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(frame, line, (15, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 30
                
                # FPS calculation and display
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    fps_counter.append(1.0 / frame_time)
                avg_fps = sum(fps_counter) / len(fps_counter) if fps_counter else 0
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (15, frame.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Periodic statistics logging
                if time.time() - last_stats_time > 10:  # Every 10 seconds
                    print(f"[STATS] {stats} | FPS: {avg_fps:.1f}")
                    last_stats_time = time.time()
                
            except Exception as e:
                print(f"[ERROR] Frame processing failed: {e}")
                continue
            
            # Display frame
            cv2.imshow("Enhanced Object Tracking", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Manual scene capture
                try:
                    scene_path = os.path.join(args.save_dir, f"manual_scene_{int(time.time())}.jpg")
                    cv2.imwrite(scene_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"[MANUAL] Scene saved: {scene_path}")
                except Exception as e:
                    print(f"[ERROR] Manual scene capture failed: {e}")
            elif key == ord('r'):  # Reset tracker
                try:
                    tracker = EnhancedObjectTracker(
                        depth_enabled=args.depth_camera,
                        max_disappeared=20 if args.pi_mode else 30,
                        movement_threshold=20 if args.pi_mode else 25
                    )
                    print("[RESET] Tracker reset")
                except Exception as e:
                    print(f"[ERROR] Tracker reset failed: {e}")
    
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("[CLEANUP] Shutting down...")
        
        # Stop async saver
        if 'save_queue' in locals():
            try:
                save_queue.put(None)  # Shutdown signal
            except:
                pass
        
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Save final statistics
        if 'tracker' in locals():
            try:
                final_stats = tracker.get_tracking_stats()
                stats_file = os.path.join(args.save_dir, f"tracking_stats_{int(time.time())}.json")
                
                # Prepare serializable stats
                serializable_stats = {
                    'final_stats': final_stats,
                    'total_frames_processed': frame_count,
                    'pi_mode': args.pi_mode,
                    'model_used': args.model,
                    'confidence_threshold': args.confidence,
                    'session_duration': time.time() - (last_stats_time - 10),
                    'objects_captured': final_stats.get('total_captured', 0)
                }
                
                with open(stats_file, 'w') as f:
                    json.dump(serializable_stats, f, indent=2)
                print(f"[STATS] Final statistics saved: {stats_file}")
            except Exception as e:
                print(f"[ERROR] Failed to save stats: {e}")

# Utility functions for system optimization
def optimize_for_system():
    """Apply system-specific optimizations"""
    try:
        # Set threading for optimal performance
        cv2.setNumThreads(4)  # Use 4 cores
        
        # Set environment variables for better performance
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        
        print("[OPT] System optimizations applied")
        return True
    except Exception as e:
        print(f"[OPT] Optimization failed: {e}")
        return False

# System monitoring class
class SystemMonitor:
    """Monitor system resources and performance"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=50)
        
    def log_frame_time(self, frame_time):
        self.frame_times.append(frame_time)
        
    def get_average_fps(self):
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0
        
    def check_memory_usage(self):
        # Simplified memory check without external dependencies
        return True
            
    def get_runtime_stats(self):
        runtime = time.time() - self.start_time
        avg_fps = self.get_average_fps()
        
        stats = {
            'runtime_seconds': runtime,
            'average_fps': avg_fps,
            'frames_processed': len(self.frame_times),
            'memory_usage_avg': 0  # Simplified without psutil
        }
        return stats

# Additional utility functions
def create_optimized_cap(source):
    """Create optimized video capture"""
    try:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        
        # Basic optimizations that work on most systems
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
        
        return cap
    except Exception as e:
        print(f"[CAP] Failed to create optimized capture: {e}")
        return None

def validate_model_path(model_path):
    """Validate that the model file exists and is accessible"""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model path is not a file: {model_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.pt', '.onnx', '.engine']
    if not any(model_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"[WARNING] Unusual model file extension: {model_path}")
    
    return True

if __name__ == "__main__":
    # Validate model path before starting
    if not validate_model_path(args.model):
        print("[FATAL] Cannot proceed without valid model file")
        exit(1)
    
    # Apply system optimizations
    print("[OPT] Applying system optimizations...")
    optimize_for_system()
    
    # Start main tracking system
    main()
