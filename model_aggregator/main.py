from pathlib import Path
from syftbox.lib import Client
import os
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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


def aggregate_model(
    participants: list[str], datasite_path: Path, global_model_path: Path
):
    global_model = SimpleNN()
    global_model_state_dict = global_model.state_dict()

    aggregated_model_weights = {}

    n_peers = len(participants)
    for user_folder in participants:
        model_file: Path = Path(datasite_path) / user_folder / "public" / "model.pth"

        user_model_state = torch.load(str(model_file))
        for key in global_model_state_dict.keys():

            # If user model has a different architecture than my global model.
            # Skip it
            if user_model_state.keys() != global_model_state_dict.keys():
                continue

            if aggregated_model_weights.get(key, None) is None:
                aggregated_model_weights[key] = user_model_state[key] * (1 / n_peers)
            else:
                aggregated_model_weights[key] += user_model_state[key] * (1 / n_peers)

    if aggregated_model_weights:
        global_model.load_state_dict(aggregated_model_weights)
        torch.save(global_model.state_dict(), str(global_model_path))
        return global_model
    else:
        return None


def peer_has_new_model(dataset_peer_path: Path) -> bool:
    model_path: Path = Path(dataset_peer_path / "public" / "model.pth")
    last_model_update: Path = Path(dataset_peer_path / "public" / "model_training.json")

    if model_path.is_file() and last_model_update.is_file():
        with open(str(last_model_update), "r") as model_info:
            model_info = json.load(model_info)

        last_trained_time = datetime.fromisoformat(model_info["last_train"])
        time_now = datetime.now()

        if (time_now - last_trained_time) <= timedelta(minutes=10):
            return True

    return False


def network_participants(datasite_path: Path):
    exclude_dir = ["apps", ".syft"]

    entries = os.listdir(datasite_path)

    users = []
    for entry in entries:
        user_path = Path(datasite_path / entry)
        is_excluded_dir = entry in exclude_dir

        is_valid_peer = user_path.is_dir() and not is_excluded_dir
        if is_valid_peer and peer_has_new_model(user_path):
            users.append(entry)

    return users


def time_to_aggregate(datasite_path: Path):
    last_round_file_path: Path = (
        Path(datasite_path) / "app_pipelines" / "fl_app" / "last_round.json"
    )
    fl_pipeline_path: Path = last_round_file_path.parent

    if not fl_pipeline_path.is_dir():
        os.makedirs(str(fl_pipeline_path))
        return True

    with open(str(last_round_file_path), "r") as last_round_file:
        last_round_info = json.load(last_round_file)

        last_trained_time = datetime.fromisoformat(last_round_info["last_train"])
        time_now = datetime.now()

        if (time_now - last_trained_time) >= timedelta(seconds=10):
            return True

    return False


def save_aggregation_timestamp(datasite_path: Path) -> None:
    last_round_file_path: Path = (
        Path(datasite_path) / "app_pipelines" / "fl_app" / "last_round.json"
    )
    with open(str(last_round_file_path), "w") as last_round_file:
        timestamp = datetime.now().isoformat()
        json.dump({"last_train": timestamp}, last_round_file)


def evaluate_global_model(global_model: nn.Module, dataset_path: Path) -> float:
    global_model.eval()

    # load the saved mnist subset
    images, labels = torch.load(str(dataset_path))

    # create a tensordataset
    dataset = TensorDataset(images, labels)

    # create a dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # dataset = torch.load(str(dataset_path))
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    client = Client.load()

    if not time_to_aggregate(client.datasite_path):
        print("It's not time for a new aggregation round,  skipping it for now.")
        exit()

    participants = network_participants(client.datasite_path.parent)

    global_model = None
    if len(participants) == 0:
        print("No new model found! Skipping aggregation...")
    else:
        print("Aggregating models between ", participants)

        global_model = aggregate_model(
            participants,
            client.datasite_path.parent,
            client.datasite_path / "public" / "global_model.pth",
        )

    if global_model:
        dataset_path = "./mnist_dataset.pt"
        accuracy = evaluate_global_model(global_model, dataset_path)
        print(f"Global model accuracy: {accuracy:.2f}%")

    save_aggregation_timestamp(client.datasite_path)
