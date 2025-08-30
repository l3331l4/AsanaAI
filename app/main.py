import streamlit as st
import cv2
import numpy as np
from services.pose_detector import PoseDetector

st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="assets/favicon.ico.",
    layout="wide"
)

st.title("Real-Time Yoga Pose Classifier")
st.write("Practice yoga with AI-powered pose detection and real-time feedback")

detector = PoseDetector()

col1, col2 = st.columns(2)

with col1:
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        bytes_data = camera_input.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        keypoints = detector.detect(cv2_img)
        
        if keypoints:
            st.success(f"Pose detected! Found {len(keypoints)} keypoints")
            st.json(keypoints)
        else:
            st.warning("No pose detected. Please try again.")