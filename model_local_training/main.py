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


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def look_for_datasets(dataset_path: Path):
    if not dataset_path.is_dir():
        os.makedirs(str(dataset_path))

    dataset_path_files = [f for f in os.listdir(str(dataset_path)) if f.endswith(".pt")]
    return dataset_path_files


def train_model(datasite_path: Path, dataset_path: Path, public_folder_path: Path):
    dataset_path_files = look_for_datasets(dataset_path=dataset_path)

    if len(dataset_path_files) == 0:
        print(f"No dataset found in {dataset_path} skipping training.")
        return

    model = SimpleNN()  # Initialize train_model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    all_datasets = []
    for dataset_file in dataset_path_files:

        # load the saved mnist subset
        images, labels = torch.load(str(dataset_path) + "/" + dataset_file)

        # create a tensordataset
        dataset = TensorDataset(images, labels)

        all_datasets.append(dataset)

    combined_dataset = ConcatDataset(all_datasets)

    # create a dataloader for the dataset
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    # Open log file in append mode
    output_logs_path = Path(public_folder_path / "training.log")
    log_file = open(str(output_logs_path), "a")

    # Log training start
    start_msg = f"[{datetime.now().isoformat()}] Starting training...\n"
    log_file.write(start_msg)
    log_file.flush()

    # training loop
    for epoch in range(1000):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        log_msg = f"[{datetime.now().isoformat()}] Epoch {epoch + 1:04d}: Loss = {avg_loss:.6f}\n"
        log_file.write(log_msg)
        log_file.flush()  # Force write to disk

    # Serialize the model
    output_model_path = Path(public_folder_path / "model.pth")
    torch.save(model.state_dict(), str(output_model_path))

    output_training_json_path = Path(public_folder_path / "model_training.json")
    with open(str(output_training_json_path), "w") as training_info_file:
        timestamp = datetime.now().isoformat()
        json.dump({"last_train": timestamp}, training_info_file)

    # Log completion
    final_msg = f"[{datetime.now().isoformat()}] Training completed. Final loss: {avg_loss:.6f}\n"
    log_file.write(final_msg)
    log_file.flush()
    log_file.close()


def time_to_train(datasite_path: Path):
    last_round_file_path: Path = (
        Path(datasite_path) / "app_pipelines" / "fl_app" / "last_round.json"
    )
    fl_pipeline_path: Path = last_round_file_path.parent

    if not fl_pipeline_path.is_dir():
        os.makedirs(str(fl_pipeline_path))
        copy_folder_contents("./mnist_samples", str(datasite_path / "private/"))
        return True

    with open(str(last_round_file_path), "r") as last_round_file:
        last_round_info = json.load(last_round_file)

        last_trained_time = datetime.fromisoformat(last_round_info["last_train"])
        time_now = datetime.now()

        if (time_now - last_trained_time) >= timedelta(seconds=10):
            return True

    return False


def save_training_timestamp(datasite_path: Path) -> None:
    last_round_file_path: Path = (
        Path(datasite_path) / "app_pipelines" / "fl_app" / "last_round.json"
    )
    with open(str(last_round_file_path), "w") as last_round_file:
        timestamp = datetime.now().isoformat()
        json.dump({"last_train": timestamp}, last_round_file)


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
    if not time_to_train(client.datasite_path):
        print("It's not time for a new training routine. skipping it for now.")
        exit()

    dataset_path = Path(client.datasite_path / "private" / "datasets")
    public_folder = Path(client.datasite_path / "public")
    output_model_path = Path(public_folder / "model.pth")
    output_model_info = Path(public_folder / "model_training.json")
    os.makedirs(dataset_path, exist_ok=True)
    train_model(client.datasite_path, dataset_path, public_folder)
    save_training_timestamp(client.datasite_path)
