import torch
from torch.utils.data import DataLoader, TensorDataset
from app.ml.training.utils import load_train_val_data
from app.ml.models.pose_classifier import PoseClassifier

DATA_FILE = "data/training/all_features.json"
MODEL_PATH = "app/ml/models/pose_classifier.pth"
BATCH_SIZE = 16
INPUT_DIM = 20
NUM_CLASSES = 5

def evaluate_model():
    _, x_val, _, y_val = load_train_val_data()

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

if __name__ == "__main__":
    evaluate_model()