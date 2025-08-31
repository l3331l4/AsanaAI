import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from streamlit_autorefresh import st_autorefresh
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

st.sidebar.header("Settings")
st.sidebar.write("---")
st.sidebar.write("**Model:** PoseClassifier v1.0")

@st.cache_resource
def load_model():
    model = PoseClassifier(input_dim=20, num_classes=5)
    model.load_state_dict(torch.load("app/ml/models/pose_classifier.pth", map_location='cpu'))
    model.eval()
    return model

pose_names = ["downward_dog", "goddess", "plank", "tree", "warrior_2"]
model = load_model()

col1, col2 = st.columns([2, 2])
st_autorefresh(interval=250, key="pose_refresh") 

with col1:
    st.subheader("Live Camera Feed")
    ctx = webrtc_streamer(
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
with col2:
    st.subheader("Detected Pose")

    pose_container = st.empty()
    confidence_container = st.empty()
    instruction_container = st.empty()
    
    if ctx.video_processor:
            pose_container.markdown(f"### {ctx.video_processor.last_pose.replace('_', ' ').title()}")
            if ctx.video_processor.last_conf:
                confidence_container.write(f"Confidence: {ctx.video_processor.last_conf:.1%}")
    else:
        instruction_container.info("Press 'Start' to begin pose detection")