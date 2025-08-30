import mediapipe as mp
import numpy as np
import cv2
from typing import Optional, Dict, Tuple

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp_drawing
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if results.pose_landmarks:
            return self._extract_keypoints(results.pose_landmarks)
        return None
    
    def _extract_keypoints(self, landmarks) -> Dict[str, Tuple[float, float]]:
        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name.lower()
            keypoints[landmark_name] = (landmark.x, landmark.y)
        return keypoints
    
    def draw_pose(self, image: np.ndarray, landmarks) -> np.ndarray:
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image, 
            landmarks, 
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return annotated_image