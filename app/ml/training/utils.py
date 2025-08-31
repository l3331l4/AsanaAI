import torch
import json
from sklearn.model_selection import train_test_split

DATA_FILE = "data/training/all_features.json"
RANDOM_STATE = 1
TEST_SPLIT = 0.2

def load_train_val_data():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    features = torch.tensor([item["features"] for item in data], dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in data], dtype=torch.long)

    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, 
        test_size=TEST_SPLIT, 
        random_state=RANDOM_STATE, 
        stratify=labels
    )

    return x_train, x_val, y_train, y_val
