import mediapipe as mp
import numpy as np
import cv2
from typing import Optional, Dict, Tuple

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
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