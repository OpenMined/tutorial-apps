import torch
import torch.nn as nn
from datetime import datetime, timedelta
from syftbox.lib import Client
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import os
from pathlib import Path
import torch.optim as optim
import json
import shutil

def look_for_datasets(dataset_path: Path):
    if not dataset_path.is_dir():
        os.makedirs(str(dataset_path))

    dataset_path_files = [f for f in os.listdir(str(dataset_path)) if f.endswith(".pt")]
    return dataset_path_files


def copy_folder_contents(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        if not os.path.isdir(src_path):
            shutil.copy2(src_path, dest_path)


if __name__ == "__main__":
    client = Client.load()

    dataset_path = Path(client.datasite_path / "private" / "datasets")
    public_folder = Path(client.datasite_path / "public")
    output_model_path = Path(public_folder / "model.pth")
    output_model_info = Path(public_folder / "model_training.json")
    os.makedirs(dataset_path, exist_ok=True)
    train_model(client.datasite_path, dataset_path, public_folder)
    save_training_timestamp(client.datasite_path)
