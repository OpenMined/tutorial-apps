from syftbox.lib import Client
from syftbox.lib import SyftPermission
from pathlib import Path
import json
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
from datetime import datetime
import torch.optim as optim

import shutil
import re
import importlib.util

# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass

# TODO: Currently setting the permissions with public write
# change the permission model later
#NOTE: we mainly want the aggregator to have write access to
# fl_client/request folder
# fl_client/running/project_name/agg_weights folder
def add_public_write_permission(client: Client, path: Path) -> None:
    """
    Adds public write permission to the given path
    """
    permission = SyftPermission.mine_with_public_write(client.email)
    permission.ensure(path)


def get_all_directories(path: Path) -> list:
    """
    Returns the list of directories present in the given path
    """
    return [x for x in path.iterdir() if x.is_dir()]

def look_for_datasets(path: Path) -> list[Path]:
    # We return all the files in the path
    # with a particular regex pattern like mnist_label_*.pt
    # NOTE: this is an hardcoded pattern for demonstration purposes
    pattern = r'^mnist_label_[0-9]\.pt$'
    dataset_files = [f for f in path.iterdir() if re.match(pattern, f.name)]
    return dataset_files

def init_fl_client_app(client: Client) -> None:
    """
    Creates the `fl_client` app in the `app_pipelines` folder
    with the following structure:
    ```
    app_pipelines
    └── fl_client
            └── request
            └── running
    ```
    """
    app_pipelines = Path(client.datasite_path) / "app_pipelines"
    fl_client = app_pipelines / "fl_client"


    for folder in ["request", "running"]:
        fl_client_folder = fl_client / folder
        fl_client_folder.mkdir(parents=True, exist_ok=True)
        

    add_public_write_permission(client, fl_client/"request")

    # We additionaly create a private folder for the client to place the datasets
    private_folder_path = client.datasite_path / "private"
    private_folder_path.mkdir(parents=True, exist_ok=True)

    

def init_client_dirs(proj_folder: Path) -> None:
    """
    Step 1: Ensure the project has init directories (like round_weights, agg_weights)
    """
    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    round_weights_folder.mkdir(parents=True, exist_ok=True)
    agg_weights_folder.mkdir(parents=True, exist_ok=True)

    add_public_write_permission(client, agg_weights_folder)


def load_model_class(model_path: Path) -> type:
    model_class_name = "FLModel"
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    model_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arch)
    model_class = getattr(model_arch, model_class_name)  

    return model_class


def train_model(client: Client, proj_folder: Path, round_num: int) -> None:
    """
    Trains the model for the given round number
    """
    
    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"

    fl_config_path = proj_folder / "fl_config.json"
    with open(fl_config_path, "r") as f:
        fl_config: dict = json.load(f)
    
    # Retrieve all the mnist datasets from the private folder
    dataset_path = client.datasite_path / "private"
    dataset_path_files = look_for_datasets(dataset_path)

    if len(dataset_path_files) == 0:
        raise StateNotReady(f"No dataset found in private folder skipping training.")
    
    # Load the Model from the model_arch.py file
    model_class = load_model_class(proj_folder / fl_config["model_arch"])
    model: nn.Module = model_class()
    
    # Load the aggregated weights from the previous round
    agg_weights_file = agg_weights_folder / f"agg_model_round_{round_num - 1}.pt"
    model.load_state_dict(torch.load(agg_weights_file,weights_only=True))

    criterion = nn.CrossEntropyLoss()
    #TODO: Update learning rate from the fl_config
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    all_datasets = []
    for dataset_path_file in dataset_path_files:

        # load the saved mnist subset
        images, labels = torch.load(str(dataset_path_file),weights_only=True)

        # create a tensordataset
        dataset = TensorDataset(images, labels)

        all_datasets.append(dataset)

    combined_dataset = ConcatDataset(all_datasets)

    # create a dataloader for the dataset
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    # Open log file for writing
    logs_folder_path = proj_folder / "logs"
    logs_folder_path.mkdir(parents=True, exist_ok=True)
    output_logs_path = logs_folder_path / f"training_logs_round_{round_num}.txt"
    log_file = open(str(output_logs_path), "w")
    
    # Log training start
    start_msg = f"[{datetime.now().isoformat()}] Starting training...\n"
    log_file.write(start_msg)
    log_file.flush()


    #TODO: Update epoch from the fl_config
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
    output_model_path = round_weights_folder / f"trained_model_round_{round_num}.pt"
    torch.save(model.state_dict(), str(output_model_path))

    # Log completion
    final_msg = f"[{datetime.now().isoformat()}] Training completed. Final loss: {avg_loss:.6f}\n"
    log_file.write(final_msg)
    log_file.flush()
    log_file.close()

def shift_project_to_done_folder(client: Client, proj_folder: Path, total_rounds: int) -> None:
    """
    Moves the project to the `done` folder
    a. Create a directory in the `done` folder with the same name as the project
    b. moves the agg weights and round weights to the done folder
    c. delete the project folder from the running folder
    """
    done_folder = (
        Path(client.datasite_path) / "app_pipelines" / "fl_client" / "done"
    )
    done_proj_folder = done_folder / proj_folder.name
    done_proj_folder.mkdir(parents=True, exist_ok=True)

    # Move the agg weights and round weights folder to the done project folder
    shutil.move(proj_folder / "agg_weights", done_proj_folder)
    shutil.move(proj_folder / "round_weights", done_proj_folder)

    # Delete the project folder from the running folder
    shutil.rmtree(proj_folder)
    


def advance_fl_round(client: Client, proj_folder: Path) -> None:
    """
    Step 2: Has the aggregate sent the weights for the current round x (in the agg_weights folder)
    b. The client trains the model on the given round  and places the trained model in the round_weights folder
    c. It sends the trained model to the aggregator.
    d. repeat a until all round completes
    """
    round_weights_folder = proj_folder / "round_weights"
    agg_weights_folder = proj_folder / "agg_weights"


    fl_config_path = proj_folder / "fl_config.json"
    with open(fl_config_path, "r") as f:
        fl_config: dict = json.load(f)
    
    total_rounds = fl_config["rounds"]
    round_num = len(list(round_weights_folder.iterdir())) +1
    if round_num > total_rounds:
        print(f"FL project {proj_folder.name} has completed all the rounds")
        # TODO: move the project to the `done` folder
        # Q: Do we move them when the aggregator has sent the weights for the last round?
        shift_project_to_done_folder(client, proj_folder, total_rounds)
        return


    # Check if the aggregate has sent the weights for the previous round
    # We always use the previous round weights to train the model
    # from the agg_weights folder to train for the current round
    agg_weights_file = agg_weights_folder / f"agg_model_round_{round_num - 1}.pt"
    if not agg_weights_file.is_file():
        raise StateNotReady(f"Aggregator has not sent the weights for the round {round_num}")

    # Train the model
    train_model(client, proj_folder, round_num)

    # Send the trained model to the aggregator
    aggregator_email = fl_config["aggregator"]
    trained_model_file = round_weights_folder / f"trained_model_round_{round_num}.pt"
    fl_aggregator_app_path = client.datasite_path.parent / aggregator_email / "app_pipelines" / "fl_aggregator"
    fl_aggregator_running_folder = fl_aggregator_app_path / "running" / proj_folder.name
    fl_aggregator_client_path = fl_aggregator_running_folder / "fl_clients" / client.email

    # Copy the trained model to the aggregator's client folder
    shutil.copy(trained_model_file, fl_aggregator_client_path)



def _advance_fl_project(client: Client,proj_folder: Path) -> None:
    """
    Iterate over all the project folder, it will try to advance its state.
    1. Ensure the project has init directories (like round_weights, agg_weights)
    2. Has the aggregate sent the weights for the current round x (in the agg_weights folder)
    b. The client trains the model on the given round  and places the trained model in the round_weights folder
    c. It sends the trained model to the aggregator.
    d. repeat a until all round completes
    """

    try: 

        init_client_dirs(proj_folder)

        advance_fl_round(client, proj_folder)
    
    except StateNotReady as e:
        print(e)
        return


def advance_fl_projects(client: Client) -> None:
    """
    Iterates over the `running` folder and tries to advance the FL projects
    """
    running_folder = (
        Path(client.datasite_path) / "app_pipelines" / "fl_client" / "running"
    )
    for proj_folder in running_folder.iterdir():
        if proj_folder.is_dir():
            proj_name = proj_folder.name
            print(f"Advancing FL project {proj_name}")
            _advance_fl_project(client, proj_folder)




if __name__ == "__main__":
    client = Client.load()

    # Step 1: Init the FL Aggregator App
    init_fl_client_app(client)

    # Step 2: Advance the FL Projects.
    # Iterates over the running folder and tries to advance the FL project
    advance_fl_projects(client)
