from pathlib import Path
from syftbox.lib import Client
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import re


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


def get_model_file(path: Path) -> str | None:
    model_files = []
    entries= os.listdir(path)
    pattern = r'^pretrained_label_[0-9]\.pth$'

    for entry in entries:
        if re.match(pattern):
            model_files.append(entry)
    
    return model_files[0] if model_files else None


def aggregate_model(
    participants: list[str], datasite_path: Path, global_model_path: Path
):
    global_model = SimpleNN()
    global_model_state_dict = global_model.state_dict()

    aggregated_model_weights = {}

    n_peers = len(participants)
    for user_folder in participants:
        public_folder_path: Path = Path(datasite_path) / user_folder / "public"

        model_file = get_model_file(public_folder_path) 
        if model_file is None:
            continue

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




def network_participants(datasite_path: Path):
    exclude_dir = ["apps", ".syft"]

    entries = os.listdir(datasite_path)

    users = []
    for entry in entries:
        user_path = Path(datasite_path / entry)
        is_excluded_dir = entry in exclude_dir

        is_valid_peer = user_path.is_dir() and not is_excluded_dir
        if is_valid_peer:
            users.append(entry)

    return users



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


    participants = network_participants(client.datasite_path.parent)
    
    global_model = None
    
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



