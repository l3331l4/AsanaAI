import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json
from app.ml.models.pose_classifier import PoseClassifier

DATA_FILE = "data/training/all_features.json"
MODEL_PATH = "app/ml/models/pose_classifier.pth"
BATCH_SIZE = 16
INPUT_DIM = 20
NUM_CLASSES = 5
RANDOM_STATE = 1
TEST_SPLIT = 0.2

with open(DATA_FILE, "r") as f:
    data = json.load(f)

features = torch.tensor([item["features"] for item in data], dtype=torch.float32)
labels = torch.tensor([item["label"] for item in data], dtype=torch.long)

_, x_val, _, y_val = train_test_split(
    features, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=labels
)

val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Input Dimension: {INPUT_DIM}")
print(f"Number of Classes: {NUM_CLASSES}")
model = PoseClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for features, labels in val_loader:
        outputs = model(features)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")
