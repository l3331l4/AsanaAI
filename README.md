<img src="assets/logo.png" alt="AsanaAI Logo"/>

# AsanaAI – Real-Time Yoga Pose Classifier

## Overview

**AsanaAI** is a real-time yoga pose recognition system that detects body landmarks with **MediaPipe**, extracts features, and classifies poses using a **PyTorch model**. 
The system runs in the browser via **Streamlit** with live webcam inference via **streamlit-webrtc**.

* **Pose detection (MediaPipe):** landmark extraction from webcam frames.
* **Feature extraction:** angles and normalized distances from landmarks.
* **PyTorch classifier:** predicts yoga poses in real time.
* **Interactive demo (Streamlit):** displays annotated frames with on-frame confidence.

Validation accuracy: **80.98%** on a 5-class yoga pose dataset.

**Pipeline:** Webcam -> MediaPipe landmarks -> Feature extraction -> PyTorch classifier -> Streamlit display


## Key Features

* **Live pose recognition:** Runs in real time on webcam input directly in the browser.
* **Modular components:** Separate services handle pose detection, feature extraction, and classification.
* **Trainable model:** Includes utilities for preparing dataset, training, and evaluating the PyTorch classifier.
* **Performance insights:** Evaluation script provides confusion matrix and per-class accuracy breakdown.

## Future Improvements

* Expand dataset to add support for more yoga poses.
* Add pose correction feedback using angle analysis

## Tech Stack

* **Python 3.10+**
* **PyTorch** – model training and inference
* **MediaPipe** – real-time pose landmark detection
* **Streamlit + streamlit-webrtc** – browser-based demo with live webcam input
* **OpenCV & NumPy** – video frame handling and feature processing
* **scikit-learn** – evaluation and metrics

(See full list in [`requirements.txt`](requirements.txt))


## Setup

### 1. Create and activate virtual environment

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Training & Evaluation

### Prepare training data

* Download dataset from Kaggle with `scripts/download_data.py`
* Extract features from dataset images with `scripts/collect_data.py`
* Preprocess/merge features with `scripts/prepare_training_data.py`

### Train model

```bash
python app/ml/training/train_model.py
```

### Evaluate model

```bash
python app/ml/training/evaluate_model.py
```

## Run the Demo

Start the Streamlit app:

```bash
streamlit run app/main.py
```

Then open [http://localhost:8501](http://localhost:8501) and click **Start** in the webcam widget.
