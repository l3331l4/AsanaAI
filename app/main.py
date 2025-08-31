import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import torch

from ml.models.pose_classifier import PoseClassifier
from video_processor import VideoProcessor

st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="assets/favicon.ico.",
    layout="wide"
)

st.title("Real-Time Yoga Pose Classifier")
st.write("Practice yoga with AI-powered pose detection and real-time feedback")

@st.cache_resource
def load_model():
    model = PoseClassifier(input_dim=20, num_classes=5)
    model.load_state_dict(torch.load("ml/models/pose_classifier.pth", map_location='cpu'))
    model.eval()
    return model

pose_names = ["downward_dog", "goddess", "plank", "tree", "warrior_2"]
model = load_model()

columns = st.columns(2)

with columns[0]:
    webrtc_streamer(
        key="key", 
        video_processor_factory=lambda: VideoProcessor(model, pose_names),
        rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )