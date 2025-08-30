import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from collect_data import collect_pose_data
import os
import json

poses = ["downward_dog", "goddess", "plank", "tree", "warrior_2"]

all_data = []
for pose in poses:
    folder = f"data/training/{pose}"
    pose_data = collect_pose_data(folder, pose)
    all_data.extend(pose_data)

os.makedirs("data/training", exist_ok=True)
with open("data/training/all_features.json", "w") as f:
    json.dump(all_data, f)