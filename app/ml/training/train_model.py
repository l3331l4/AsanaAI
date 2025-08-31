import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json
from pathlib import Path
from app.ml.models.pose_classifier import PoseClassifier
from app.ml.training.data_loader import PoseDataset

DATA_FILE = "data/training/all_features.json"
MODEL_SAVE_PATH = Path("app/ml/models/pose_classifier.pth")
BATCH_SIZE = 16
EPOCHS = 80
INPUT_DIM = 20
NUM_CLASSES = 5
LR = 0.001
RANDOM_STATE = 1
TEST_SPLIT = 0.2

def train_model():

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    features = torch.tensor([item["features"] for item in data], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in data], dtype=torch.long)

    x_train, _, y_train, _ = train_test_split(
        features, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=labels
    )

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PoseClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
