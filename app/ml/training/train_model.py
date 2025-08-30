import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from app.ml.models.pose_classifier import PoseClassifier
from app.ml.training.data_loader import PoseDataset

def train_model():

    dataset = PoseDataset("data/training/all_features.json")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    input_dim = 20
    num_classes = 5

    model = PoseClassifier(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epochs = 50
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

    model_path = Path("app/ml/models/pose_classifier.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_model()
