import os
import json
import cv2
from app.services.pose_detector import PoseDetector
from app.ml.models.feature_extractor import FeatureExtractor

def collect_pose_data(image_folder: str, pose_name: str):
    detector = PoseDetector()
    extractor = FeatureExtractor()
    
    data = []
    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path) 
            if image is None:
                continue

            keypoints = detector.detect(image)
            if keypoints:
                features = extractor.extract_features(keypoints)
                data.append({
                    "image": image_file,
                    "features": features.tolist()
                })

    with open(f'data/training/{pose_name}_features.json', 'w') as f:
        json.dump(data, f)