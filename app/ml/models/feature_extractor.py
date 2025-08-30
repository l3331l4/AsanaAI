import numpy as np
from typing import Dict, Tuple
import math

class FeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def extract_features(self, keypoints: Dict[str, Tuple[float, float]]) -> np.ndarray:
        features = []

        features.extend(self._calculate_angles(keypoints))
        features.extend(self._calculate_distances(keypoints))

        return np.array(features)

    def _calculate_angles(self, keypoints: Dict) -> list:
        angles = []

        # Right elbow
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles.append(self._angle_between_points(
                keypoints['right_shoulder'], keypoints['right_elbow'], keypoints['right_wrist']))
        else:
            angles.append(0.0)

        # Left elbow
        if all(k in keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles.append(self._angle_between_points(
                keypoints['left_shoulder'], keypoints['left_elbow'], keypoints['left_wrist']))
        else:
            angles.append(0.0)

        # Right shoulder
        if all(k in keypoints for k in ['right_elbow', 'right_shoulder', 'right_hip']):
            angles.append(self._angle_between_points(
                keypoints['right_elbow'], keypoints['right_shoulder'], keypoints['right_hip']))
        else:
            angles.append(0.0)

        # Left shoulder
        if all(k in keypoints for k in ['left_elbow', 'left_shoulder', 'left_hip']):
            angles.append(self._angle_between_points(
                keypoints['left_elbow'], keypoints['left_shoulder'], keypoints['left_hip']))
        else:
            angles.append(0.0)

        # Right knee
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles.append(self._angle_between_points(
                keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle']))
        else:
            angles.append(0.0)

        # Left knee
        if all(k in keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles.append(self._angle_between_points(
                keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle']))
        else:
            angles.append(0.0)

        # Right hip
        if all(k in keypoints for k in ['right_shoulder', 'right_hip', 'right_knee']):
            angles.append(self._angle_between_points(
                keypoints['right_shoulder'], keypoints['right_hip'], keypoints['right_knee']))
        else:
            angles.append(0.0)

        # Left hip
        if all(k in keypoints for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles.append(self._angle_between_points(
                keypoints['left_shoulder'], keypoints['left_hip'], keypoints['left_knee']))
        else:
            angles.append(0.0)

        return angles

    def _angle_between_points(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    def _calculate_distances(self, keypoints: Dict) -> list:
        distances = []
        
        # ---- Left side ----
        # Torso length: shoulder to hip
        if 'left_shoulder' in keypoints and 'left_hip' in keypoints:
            distances.append(self._point_distance(keypoints['left_shoulder'], keypoints['left_hip']))
        else:
            distances.append(0.0)

        # Arm length: shoulder to elbow
        if 'left_shoulder' in keypoints and 'left_elbow' in keypoints:
            distances.append(self._point_distance(keypoints['left_shoulder'], keypoints['left_elbow']))
        else:
            distances.append(0.0)

        # Arm length: elbow to wrist
        if 'left_elbow' in keypoints and 'left_wrist' in keypoints:
            distances.append(self._point_distance(keypoints['left_elbow'], keypoints['left_wrist']))
        else:
            distances.append(0.0)

        # Leg lengths: hip to knee
        if 'left_hip' in keypoints and 'left_knee' in keypoints:
            distances.append(self._point_distance(keypoints['left_hip'], keypoints['left_knee']))
        else:
            distances.append(0.0)

        # Leg lengths: knee to ankle
        if 'left_knee' in keypoints and 'left_ankle' in keypoints:
            distances.append(self._point_distance(keypoints['left_knee'], keypoints['left_ankle']))
        else:
            distances.append(0.0)

        # ---- Right side ----
        # Torso length: shoulder to hip
        if 'right_shoulder' in keypoints and 'right_hip' in keypoints:
            distances.append(self._point_distance(keypoints['right_shoulder'], keypoints['right_hip']))
        else:
            distances.append(0.0)

        # Arm: shoulder to elbow
        if 'right_shoulder' in keypoints and 'right_elbow' in keypoints:
            distances.append(self._point_distance(keypoints['right_shoulder'], keypoints['right_elbow']))
        else:
            distances.append(0.0)

        # Arm: elbow to wrist
        if 'right_elbow' in keypoints and 'right_wrist' in keypoints:
            distances.append(self._point_distance(keypoints['right_elbow'], keypoints['right_wrist']))
        else:
            distances.append(0.0)

        # Leg: hip to knee
        if 'right_hip' in keypoints and 'right_knee' in keypoints:
            distances.append(self._point_distance(keypoints['right_hip'], keypoints['right_knee']))
        else:
            distances.append(0.0)

        # Leg: knee to ankle
        if 'right_knee' in keypoints and 'right_ankle' in keypoints:
            distances.append(self._point_distance(keypoints['right_knee'], keypoints['right_ankle']))
        else:
            distances.append(0.0)

        # Shoulder width: left to right shoulder
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            distances.append(self._point_distance(keypoints['left_shoulder'], keypoints['right_shoulder']))
        else:
            distances.append(0.0)

        # Hip width: left to right hip
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            distances.append(self._point_distance(keypoints['left_hip'], keypoints['right_hip']))
        else:
            distances.append(0.0)
        
        return distances

    def _point_distance(self, p1: Tuple, p2: Tuple) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)