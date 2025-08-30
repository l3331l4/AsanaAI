import streamlit as st

st.set_page_config(
    page_title="Yoga Pose Classifier",
    page_icon="assets/favicon.ico.",
    layout="wide"
)

st.title("Real-Time Yoga Pose Classifier")
st.write("Practice yoga with AI-powered pose detection and real-time feedback")

if st.button("Start Practice"):
    st.success("Ready to begin!")