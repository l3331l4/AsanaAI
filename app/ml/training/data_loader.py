import torch
from torch.utils.data import Dataset
import json
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, features_path: str):
        with open(features_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        return features, label
