import av
import cv2
import torch
import numpy as np

from services.pose_detector import PoseDetector
from ml.models.feature_extractor import FeatureExtractor

class VideoProcessor:
    def __init__(self, model, pose_names):
        self.model = model
        self.pose_names = pose_names
        self.detector = PoseDetector()
        self.extractor = FeatureExtractor()
        self.last_pose = None
        self.last_conf = None

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        keypoints, annotated_image = self.detector.detect(frm, return_frame=True)

        if keypoints:
            features = self.extractor.extract_features(keypoints)
            with torch.no_grad():
                input_tensor = torch.tensor([features], dtype=torch.float32)
                prediction = self.model(input_tensor)
                pose_class = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1).max().item()

            self.last_pose = self.pose_names[pose_class]
            self.last_conf = confidence

            cv2.putText(
                annotated_image,
                f"{self.last_pose} ({self.last_conf:.0%})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")
