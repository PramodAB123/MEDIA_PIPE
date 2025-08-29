import mediapipe as mp
import cv2
import numpy as np
import time
import urllib.request
import os
from typing import Optional, List, Dict, Tuple
import math


class HandLandmarkDetector:
    """
    Complete Hand Landmark Detection System for real-time video processing
    Supports image, video file, and live webcam detection
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Hand Landmark Detector
        
        Args:
            model_path: Path to custom model file (optional, will download default if not provided)
        """
        # MediaPipe imports
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # Store results for live stream mode
        self.latest_result = None
        self.result_timestamp = 0
        self.detection_count = 0
        
        # Model setup
        self.model_path = self._setup_model(model_path)
        
        # Hand landmark names for reference
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        # Hand connections for drawing
        self.hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
    
    def _setup_model(self, model_path: Optional[str]) -> Optional[str]:
        """Download and setup the hand landmark model"""
        if model_path and os.path.exists(model_path):
            print(f"Using provided model: {model_path}")
            return model_path
        
        # Default model setup
        default_model_path = "hand_landmarker.task"
        
        if os.path.exists(default_model_path):
            print(f"Using existing model: {default_model_path}")
            return default_model_path
        
        print("Downloading hand landmark model...")
        try:
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(model_url, default_model_path)
            print(f"Model downloaded successfully: {default_model_path}")
            return default_model_path
        except Exception as e:
            print(f"Failed to download model: {e}")
            print("Please download the model manually from:")
            print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            return None
    
    def create_detector(self, mode: str, **kwargs):
        """Create detector based on mode"""
        if not self.model_path:
            raise ValueError("Model not available. Please check model download.")
        
        # Default configuration
        config = {
            'num_hands': kwargs.get('num_hands', 2),
            'min_hand_detection_confidence': kwargs.get('min_hand_detection_confidence', 0.7),
            'min_hand_presence_confidence': kwargs.get('min_hand_presence_confidence', 0.5),
            'min_tracking_confidence': kwargs.get('min_tracking_confidence', 0.5)
        }
        
        base_options = self.BaseOptions(model_asset_path=self.model_path)
        
        if mode == 'image':
            options = self.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=self.VisionRunningMode.IMAGE,
                **config
            )
        elif mode == 'video':
            options = self.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=self.VisionRunningMode.VIDEO,
                **config
            )
        elif mode == 'live':
            options = self.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=self.VisionRunningMode.LIVE_STREAM,
                result_callback=self._result_callback,
                **config
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return self.HandLandmarker.create_from_options(options)
    
    def _result_callback(self, result: mp.tasks.vision.HandLandmarkerResult, 
                        output_image: mp.Image, timestamp_ms: int):
        """Callback function for live stream mode"""
        self.latest_result = result
        self.result_timestamp = timestamp_ms
        self.detection_count += 1
    
    def detect_from_image(self, image_path: str, save_result: bool = False):
        """
        Detect hand landmarks from an image file
        
        Args:
            image_path: Path to image file
            save_result: Whether to save annotated image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with self.create_detector('image') as detector:
            # Load image
            mp_image = mp.Image.create_from_file(image_path)
            
            # Detect landmarks
            result = detector.detect(mp_image)
            
            if save_result:
                # Convert mp.Image to cv2 format for annotation
                image_cv2 = cv2.imread(image_path)
                annotated_image = self.draw_landmarks(image_cv2, result)
                
                # Save annotated image
                output_path = f"annotated_{os.path.basename(image_path)}"
                cv2.imwrite(output_path, annotated_image)
                print(f"Annotated image saved: {output_path}")
            
            return result
    
    def detect_from_array(self, image_array: np.ndarray):
        """Detect hand landmarks from numpy array"""
        with self.create_detector('image') as detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            result = detector.detect(mp_image)
            return result
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None, 
                          display: bool = True):
        """
        Process video file and detect hand landmarks
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        with self.create_detector('video') as detector:
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp_ms = int(frame_count * 1000 / fps)
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect landmarks
                result = detector.detect_for_video(mp_image, timestamp_ms)
                
                # Annotate frame
                annotated_frame = self.draw_landmarks(frame, result)
                self._add_info_overlay(annotated_frame, result, frame_count, total_frames)
                
                # Save frame
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Video Processing', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress update
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_count) * (total_frames - frame_count)
                    print(f"Progress: {progress:.1f}% - ETA: {eta:.1f}s")
        
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"Video processing completed in {time.time() - start_time:.1f}s")
    
    def start_live_detection(self, camera_id: int = 0, mirror: bool = True, 
                           show_fps: bool = True, show_info: bool = True):
        """
        Start live hand detection from webcam
        
        Args:
            camera_id: Camera device ID
            mirror: Mirror the video horizontally
            show_fps: Display FPS counter
            show_info: Display hand information
        """
        print("Starting live hand detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Take screenshot")
        print("  'i' - Toggle info display")
        print("  'f' - Toggle FPS display")
        print("  'SPACE' - Pause/Resume")
        
        cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        
        with self.create_detector('live') as detector:
            start_time = time.time()
            frame_count = 0
            fps_counter = 0
            fps_timer = time.time()
            paused = False
            screenshot_count = 0
            
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read from camera")
                        break
                    
                    # Mirror frame if requested
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    
                    # Calculate timestamp
                    current_time = time.time()
                    timestamp_ms = int((current_time - start_time) * 1000)
                    
                    # Convert and send for detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    detector.detect_async(mp_image, timestamp_ms)
                    
                    frame_count += 1
                    fps_counter += 1
                
                # Use latest detection result
                display_frame = frame.copy() if not paused else frame
                
                if self.latest_result:
                    display_frame = self.draw_landmarks(display_frame, self.latest_result)
                    
                    if show_info:
                        self._add_live_info_overlay(display_frame, self.latest_result)
                
                # Calculate and display FPS
                if show_fps:
                    current_fps_time = time.time()
                    if current_fps_time - fps_timer >= 1.0:
                        current_fps = fps_counter / (current_fps_time - fps_timer)
                        fps_counter = 0
                        fps_timer = current_fps_time
                    else:
                        current_fps = fps_counter / (current_fps_time - fps_timer + 0.001)
                    
                    cv2.putText(display_frame, f'FPS: {current_fps:.1f}', 
                               (display_frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add pause indicator
                if paused:
                    cv2.putText(display_frame, 'PAUSED - Press SPACE to resume', 
                               (50, display_frame.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Hand Landmarks - Live Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Screenshot
                    screenshot_name = f'screenshot_{screenshot_count:04d}.jpg'
                    cv2.imwrite(screenshot_name, display_frame)
                    print(f"Screenshot saved: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord('i'):  # Toggle info
                    show_info = not show_info
                    print(f"Info display: {'ON' if show_info else 'OFF'}")
                elif key == ord('f'):  # Toggle FPS
                    show_fps = not show_fps
                    print(f"FPS display: {'ON' if show_fps else 'OFF'}")
                elif key == ord(' '):  # Pause/Resume
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\nSession completed:")
        print(f"  Duration: {total_time:.1f}s")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Detections: {self.detection_count}")
    
    def draw_landmarks(self, image: np.ndarray, result) -> np.ndarray:
        """Draw hand landmarks and connections on image"""
        if not result.hand_landmarks:
            return image
        
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Colors for different hands
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
            color = colors[hand_idx % len(colors)]
            
            # Convert normalized coordinates to pixel coordinates
            landmarks_px = []
            for landmark in hand_landmarks:
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                landmarks_px.append((x_px, y_px))
            
            # Draw connections
            for connection in self.hand_connections:
                start_idx, end_idx = connection
                start_point = landmarks_px[start_idx]
                end_point = landmarks_px[end_idx]
                cv2.line(annotated_image, start_point, end_point, color, 2)
            
            # Draw landmarks
            for i, landmark_px in enumerate(landmarks_px):
                # Different sizes for different landmark types
                if i == 0:  # Wrist
                    radius = 8
                    thickness = -1
                elif i in [4, 8, 12, 16, 20]:  # Finger tips
                    radius = 6
                    thickness = -1
                else:  # Other landmarks
                    radius = 4
                    thickness = -1
                
                cv2.circle(annotated_image, landmark_px, radius, color, thickness)
                
                # Add landmark number (optional, for debugging)
                if False:  # Set to True to show landmark numbers
                    cv2.putText(annotated_image, str(i), 
                               (landmark_px[0] + 5, landmark_px[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return annotated_image
    
    def _add_info_overlay(self, image: np.ndarray, result, frame_num: int, total_frames: int):
        """Add information overlay to image"""
        # Progress bar for video processing
        if total_frames > 0:
            progress = frame_num / total_frames
            bar_width = 300
            bar_height = 20
            bar_x = image.shape[1] - bar_width - 20
            bar_y = 20
            
            # Background
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            # Progress
            cv2.rectangle(image, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress), bar_y + bar_height), 
                         (0, 255, 0), -1)
            # Text
            cv2.putText(image, f'{frame_num}/{total_frames}', 
                       (bar_x, bar_y + bar_height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_live_info_overlay(self, image: np.ndarray, result):
        """Add live information overlay"""
        if not result.hand_landmarks:
            cv2.putText(image, 'No hands detected', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        
        y_offset = 50
        for i, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            # Hand info
            hand_name = handedness[0].category_name
            confidence = handedness[0].score
            text = f'{hand_name} Hand: {confidence:.2f}'
            
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Finger positions
            finger_states = self._analyze_finger_positions(hand_landmarks)
            y_offset += 30
            
            for finger, state in finger_states.items():
                color = (0, 255, 0) if state == 'extended' else (0, 100, 255)
                cv2.putText(image, f'{finger}: {state}', (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
            
            y_offset += 20  # Space between hands
    
    def _analyze_finger_positions(self, hand_landmarks) -> Dict[str, str]:
        """Analyze finger positions (extended/closed)"""
        fingers = {
            'Thumb': {'tip': 4, 'pip': 3, 'mcp': 2},
            'Index': {'tip': 8, 'pip': 6, 'mcp': 5},
            'Middle': {'tip': 12, 'pip': 10, 'mcp': 9},
            'Ring': {'tip': 16, 'pip': 14, 'mcp': 13},
            'Pinky': {'tip': 20, 'pip': 18, 'mcp': 17}
        }
        
        finger_states = {}
        
        for finger_name, indices in fingers.items():
            tip = hand_landmarks[indices['tip']]
            pip = hand_landmarks[indices['pip']]
            mcp = hand_landmarks[indices['mcp']]
            
            if finger_name == 'Thumb':
                # Thumb uses different logic due to its orientation
                extended = abs(tip.x - mcp.x) > abs(tip.y - mcp.y) * 0.5
            else:
                # Other fingers: tip should be higher (lower y value) than pip and mcp
                extended = tip.y < pip.y and tip.y < mcp.y
            
            finger_states[finger_name] = 'extended' if extended else 'closed'
        
        return finger_states
    
    def get_landmark_info(self, result) -> List[Dict]:
        """Extract detailed landmark information"""
        if not result.hand_landmarks:
            return []
        
        hands_info = []
        
        for i, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            hand_info = {
                'hand_index': i,
                'handedness': handedness[0].category_name,
                'handedness_score': handedness[0].score,
                'landmarks': [],
                'world_landmarks': [],
                'finger_positions': {},
                'gesture': None
            }
            
            # Extract landmarks
            for j, landmark in enumerate(hand_landmarks):
                hand_info['landmarks'].append({
                    'id': j,
                    'name': self.landmark_names[j],
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z
                })
            
            # Extract world landmarks if available
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world_landmarks = result.hand_world_landmarks[i]
                for j, landmark in enumerate(world_landmarks):
                    hand_info['world_landmarks'].append({
                        'id': j,
                        'name': self.landmark_names[j],
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
            
            # Analyze finger positions
            hand_info['finger_positions'] = self._analyze_finger_positions(hand_landmarks)
            
            # Simple gesture recognition
            hand_info['gesture'] = self._recognize_gesture(hand_info['finger_positions'])
            
            hands_info.append(hand_info)
        
        return hands_info
    
    def _recognize_gesture(self, finger_positions: Dict[str, str]) -> str:
        """Simple gesture recognition based on finger positions"""
        extended_fingers = [finger for finger, state in finger_positions.items() 
                          if state == 'extended']
        extended_count = len(extended_fingers)
        
        if extended_count == 0:
            return 'Fist'
        elif extended_count == 1:
            if 'Index' in extended_fingers:
                return 'Pointing'
            elif 'Thumb' in extended_fingers:
                return 'Thumbs Up'
        elif extended_count == 2:
            if 'Index' in extended_fingers and 'Middle' in extended_fingers:
                return 'Peace Sign'
        elif extended_count == 5:
            return 'Open Hand'
        
        return f'{extended_count} Fingers'


def demo_image_detection():
    """Demo: Image detection"""
    print("=== Image Detection Demo ===")
    detector = HandLandmarkDetector()
    
    # You can replace this with your own image path
    print("Place an image file named 'test_image.jpg' in the current directory")
    print("Or modify the code to use your own image path")
    
    image_path = "test_image.jpg"
    if os.path.exists(image_path):
        try:
            result = detector.detect_from_image(image_path, save_result=True)
            hands_info = detector.get_landmark_info(result)
            
            print(f"Detected {len(hands_info)} hands:")
            for hand in hands_info:
                print(f"  {hand['handedness']}: {hand['handedness_score']:.2f}")
                print(f"  Gesture: {hand['gesture']}")
                
        except Exception as e:
            print(f"Image detection failed: {e}")
    else:
        print(f"Image not found: {image_path}")


def demo_live_detection():
    """Demo: Live webcam detection"""
    print("=== Live Detection Demo ===")
    detector = HandLandmarkDetector()
    
    try:
        detector.start_live_detection(
            camera_id=0,
            mirror=True,
            show_fps=True,
            show_info=True
        )
    except Exception as e:
        print(f"Live detection failed: {e}")
        print("Make sure your camera is connected and not being used by another application")


def demo_video_processing():
    """Demo: Video file processing"""
    print("=== Video Processing Demo ===")
    detector = HandLandmarkDetector()
    
    video_path = "test_video.mp4"
    if os.path.exists(video_path):
        try:
            detector.process_video_file(
                video_path,
                output_path="output_with_landmarks.mp4",
                display=True
            )
        except Exception as e:
            print(f"Video processing failed: {e}")
    else:
        print(f"Video not found: {video_path}")
        print("Place a video file named 'test_video.mp4' in the current directory")


def main():
    """Main function with menu"""
    print("Hand Landmarks Detection System")
    print("===============================")
    
    while True:
        print("\nSelect demo:")
        print("1. Live webcam detection (recommended)")
        print("2. Image detection")
        print("3. Video file processing")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            demo_live_detection()
        elif choice == '2':
            demo_image_detection()
        elif choice == '3':
            demo_video_processing()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()