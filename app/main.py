import streamlit as st
import cv2
import numpy as np
import torch

from services.pose_detector import PoseDetector
from ml.models.pose_classifier import PoseClassifier
from ml.models.feature_extractor import FeatureExtractor

st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="assets/favicon.ico.",
    layout="wide"
)

st.title("Real-Time Yoga Pose Classifier")
st.write("Practice yoga with AI-powered pose detection and real-time feedback")

detector = PoseDetector()
extractor = FeatureExtractor()

@st.cache_resource
def load_model():
    model = PoseClassifier(input_dim=20, num_classes=5)  # adjust input_dim if you change features
    model.load_state_dict(torch.load("app/ml/models/pose_classifier.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

pose_names = ["downward_dog", "goddess", "plank", "tree", "warrior_2"]

col1, col2 = st.columns(2)

with col1:
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        bytes_data = camera_input.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        keypoints = detector.detect(cv2_img)
        
        if keypoints:
            st.success(f"Pose detected! Found {len(keypoints)} keypoints")
            features = extractor.extract_features(keypoints)
            
            with torch.no_grad():
                input_tensor = torch.tensor([features], dtype=torch.float32)
                prediction = model(input_tensor)
                pose_class = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1).max().item()

            st.write(f"**Detected Pose:** {pose_names[pose_class]}")
            st.write(f"**Confidence:** {confidence:.2%}")
        else:
            st.warning("No pose detected. Please try again.")