import mediapipe as mp
import cv2
import numpy as np
import time
import urllib.request
import os
from typing import Optional, Dict
import math

class HandLandmarkDetector:
    """
    Hand Landmark Detection System for real-time webcam video processing
    """
    def __init__(self, model_path: Optional[str] = None):
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.latest_result = None
        self.result_timestamp = 0
        self.detection_count = 0

        self.model_path = self._setup_model(model_path)
        self.landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]

    def _setup_model(self, model_path: Optional[str]) -> Optional[str]:
        if model_path and os.path.exists(model_path):
            print(f"Using provided model: {model_path}")
            return model_path
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
        if not self.model_path:
            raise ValueError("Model not available. Please check model download.")
        config = {
            'num_hands': kwargs.get('num_hands', 2),
            'min_hand_detection_confidence': kwargs.get('min_hand_detection_confidence', 0.7),
            'min_hand_presence_confidence': kwargs.get('min_hand_presence_confidence', 0.5),
            'min_tracking_confidence': kwargs.get('min_tracking_confidence', 0.5)
        }
        base_options = self.BaseOptions(model_asset_path=self.model_path)
        if mode == 'live':
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
        self.latest_result = result
        self.result_timestamp = timestamp_ms
        self.detection_count += 1

    def start_live_detection(self, camera_id: int = 0, mirror: bool = True, 
                           show_fps: bool = True, show_info: bool = True):
        print("Starting live hand detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Take screenshot")
        print("  'i' - Toggle info display")
        print("  'f' - Toggle FPS display")
        print("  'SPACE' - Pause/Resume")
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
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
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    current_time = time.time()
                    timestamp_ms = int((current_time - start_time) * 1000)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    detector.detect_async(mp_image, timestamp_ms)
                    frame_count += 1
                    fps_counter += 1
                display_frame = frame.copy() if not paused else frame
                if self.latest_result:
                    display_frame = self.draw_landmarks(display_frame, self.latest_result)
                    if show_info:
                        self._add_live_info_overlay(display_frame, self.latest_result)
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
                if paused:
                    cv2.putText(display_frame, 'PAUSED - Press SPACE to resume', 
                               (50, display_frame.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Hand Landmarks - Live Detection', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_name = f'screenshot_{screenshot_count:04d}.jpg'
                    cv2.imwrite(screenshot_name, display_frame)
                    print(f"Screenshot saved: {screenshot_name}")
                    screenshot_count += 1
                elif key == ord('i'):
                    show_info = not show_info
                    print(f"Info display: {'ON' if show_info else 'OFF'}")
                elif key == ord('f'):
                    show_fps = not show_fps
                    print(f"FPS display: {'ON' if show_fps else 'OFF'}")
                elif key == ord(' '):
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
        if not result.hand_landmarks:
            return image
        annotated_image = image.copy()
        height, width = image.shape[:2]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
            color = colors[hand_idx % len(colors)]
            landmarks_px = []
            for landmark in hand_landmarks:
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                landmarks_px.append((x_px, y_px))
            for connection in self.hand_connections:
                start_idx, end_idx = connection
                start_point = landmarks_px[start_idx]
                end_point = landmarks_px[end_idx]
                cv2.line(annotated_image, start_point, end_point, color, 2)
            for i, landmark_px in enumerate(landmarks_px):
                if i == 0:
                    radius = 8
                    thickness = -1
                elif i in [4, 8, 12, 16, 20]:
                    radius = 6
                    thickness = -1
                else:
                    radius = 4
                    thickness = -1
                cv2.circle(annotated_image, landmark_px, radius, color, thickness)
                # Draw the landmark label
                cv2.putText(
                    annotated_image,
                    self.landmark_names[i],
                    (landmark_px[0] + 5, landmark_px[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        return annotated_image

    def _add_live_info_overlay(self, image: np.ndarray, result):
        if not result.hand_landmarks:
            cv2.putText(image, 'No hands detected', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
        y_offset = 50
        for i, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
            hand_name = handedness[0].category_name
            confidence = handedness[0].score
            text = f'{hand_name} Hand: {confidence:.2f}'
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            finger_states = self._analyze_finger_positions(hand_landmarks)
            y_offset += 30
            for finger, state in finger_states.items():
                color = (0, 255, 0) if state == 'extended' else (0, 100, 255)
                cv2.putText(image, f'{finger}: {state}', (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 25
            y_offset += 20

    def _analyze_finger_positions(self, hand_landmarks) -> Dict[str, str]:
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
                extended = abs(tip.x - mcp.x) > abs(tip.y - mcp.y) * 0.5
            else:
                extended = tip.y < pip.y and tip.y < mcp.y
            finger_states[finger_name] = 'extended' if extended else 'closed'
        return finger_states

def main():
    print("=== Live Detection Only ===")
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
        print("Make sure your camera is connected and not being used by another")

if __name__ == "__main__":
    main()