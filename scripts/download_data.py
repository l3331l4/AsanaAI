import os
import shutil
import kagglehub

DATA_DIR = "data/training"
KAGGLE_DATASET = "ujjwalchowdhury/yoga-pose-classification"
POSE_MAPPING = {
    "downdog": "downward_dog",
    "goddess": "goddess",
    "plank": "plank",
    "tree": "tree",
    "warrior2": "warrior_2"
}

def download_and_extract():

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)

    if os.path.isdir(path):
        shutil.copytree(path, DATA_DIR, dirs_exist_ok=True)

    yoga_poses_dir = os.path.join(DATA_DIR, "YogaPoses")
    if os.path.exists(yoga_poses_dir):
        for item in os.listdir(yoga_poses_dir):
            shutil.move(os.path.join(yoga_poses_dir, item), DATA_DIR)
        os.rmdir(yoga_poses_dir)

    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(folder_path):
            key = folder.lower()
            if key in POSE_MAPPING:
                new_path = os.path.join(DATA_DIR, POSE_MAPPING[key])
                if folder_path != new_path:
                    shutil.move(folder_path, new_path)

    print("Dataset ready!")


if __name__ == "__main__":
    download_and_extract()
