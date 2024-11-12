import importlib.util
from syftbox.lib import Client
from syftbox.lib import SyftPermission
from pathlib import Path
import json
import shutil
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset

TEST_DATASET_PATH = Path("./mnist_dataset.pt")

# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass


# TODO: Currently setting the permissions with public write
# change the permission model later to be more secure
#NOTE: we mainly want the aggregator to have write access to
# fl_aggregator/running/fl_project_name/fl_clients/*
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

def init_fl_aggregator_app(client: Client) -> None:
    """
    Creates the `fl_aggregator` app in the `app_pipelines` folder
    with the following structure:
    ```
    app_pipelines
    └── fl_aggregator
            └── launch
            └── running
            └── done
    ```
    """
    app_pipelines = Path(client.datasite_path) / "app_pipelines"
    fl_aggregator = app_pipelines / "fl_aggregator"

    for folder in ["launch", "running", "done"]:
        fl_aggregator_folder = fl_aggregator / folder
        fl_aggregator_folder.mkdir(parents=True, exist_ok=True)


def initialize_fl_project(client: Client, fl_config_json_path: Path) -> None:
    """
    Initializes the FL project by reading the `fl_config.json` file
    If the project with same name already exists in the `running` folder
    then it skips creating the project

    If the project does not exist, it creates a new project with the
    project name and creates the folders for the clients and the aggregator

    app_pipelines
    └── fl_aggregator
            └── launch
            └── running
                └── <fl_project_name>
                    ├── fl_clients
                    │   ├── ..
                    ├── agg_weights
                    ├── fl_config.json
                    ├── global_model_weights.pt
                    ├── model_arch.py
                    └── state.json
            └── done
    """
    with open(fl_config_json_path, "r") as f:
        fl_config: dict = json.load(f)
    
    
    proj_name = fl_config["project_name"]
    participants = fl_config["participants"]

    app_pipelines = Path(client.datasite_path) / "app_pipelines"
    fl_aggregator = app_pipelines / "fl_aggregator"
    running_folder = fl_aggregator / "running"
    proj_folder = running_folder / proj_name

    if proj_folder.is_dir():
        print(f"FL project {proj_name} already exists")
        return
    else:
        print(f"Creating new FL project {proj_name}")
        proj_folder.mkdir(parents=True, exist_ok=True)
        fl_clients_folder = proj_folder / "fl_clients"
        agg_weights_folder = proj_folder / "agg_weights"
        fl_clients_folder.mkdir(parents=True, exist_ok=True)
        agg_weights_folder.mkdir(parents=True, exist_ok=True)

        # create the folders for the participants
        for participant in participants:
            participant_folder = fl_clients_folder / participant
            participant_folder.mkdir(parents=True, exist_ok=True)
            # TODO: create a custom syft permission for the clients in the `fl_clients` folder
            add_public_write_permission(client, participant_folder)

        # copy the config file to the project's running folder
        shutil.move(fl_config_json_path, proj_folder)

        # move the model architecture to the project's running folder
        model_arch_src = fl_aggregator / "launch" / fl_config["model_arch"]
        shutil.move(model_arch_src, proj_folder)

        
        # copy the global model weights to the project's agg_weights folder as `agg_model_round_0.pt`
        # and move the global model weights to the project's running folder
        model_weights_src = fl_aggregator / "launch" / fl_config["model_weight"]
        shutil.copy(model_weights_src, agg_weights_folder / "agg_model_round_0.pt")
        shutil.move(model_weights_src, proj_folder)


        # TODO: create a state.json file to keep track of the project state
        # if needed while running the FL rounds

def launch_fl_project(client: Client) -> None:
    """
    - Checks if `fl_config.json` file is present in the `launch` folder
    - Check if the project exists in the `running` folder with the same `project_name`.
        If not, create a new Project
        a. creates a directory with the project name in running folder
        b. inside the project it creates the folders of clients with a custom syft permissions
        c. copies over the fl_config.json and model_arch.py and global_model_weights.pt

    Example:

    - Manually Copy the `fl_config.json`, `model_arch.py` and `global_model_weights.pt` to the `launch` folder
        app_pipelines
        └── fl_aggregator
                └── launch
                    ├── fl_config.json (dragged and dropped by the user)
                    ├── model_arch.py (dragged and dropped by the FL user)
                    ├── global_model_weights.pt (dragged and dropped by the FL user)
    """
    launch_folder = (
        Path(client.datasite_path) / "app_pipelines" / "fl_aggregator" / "launch"
    )

    fl_config_json_path = launch_folder / "fl_config.json"
    if not fl_config_json_path.is_file():
        print(f"`fl_config.json` not found in the {launch_folder} folder. Skipping...")
        return
    
    initialize_fl_project(client, fl_config_json_path)

def get_network_participants(client: Client):
    datasite_path = Path(client.datasite_path.parent)
    exclude_dir = ["apps", ".syft"]

    entries = datasite_path.iterdir()

    users = []
    for entry in entries:
        if Path(datasite_path / entry).is_dir() and entry not in exclude_dir:
            users.append(entry.name)

    return users

def check_fl_client_installed(client: Client, proj_folder: Path):
    """
    Checks if the client has installed the `fl_client` app
    """
    fl_clients = get_all_directories(proj_folder / "fl_clients")
    network_participants = get_network_participants(client)
    for fl_client in fl_clients:
        if fl_client.name not in network_participants:
            raise StateNotReady(f"Client {fl_client.name} is not part of the network")

        fl_client_app_path = client.datasite_path.parent / fl_client.name / "app_pipelines" / "fl_client"
        if not fl_client_app_path.is_dir():
            raise StateNotReady(f"Client {fl_client.name} has not installed the `fl_client` app")
    
def check_proj_requests(client: Client, proj_folder: Path):
    """
    Step 1: Checks if the project requests are sent to the clients
    Step 2: Checks if all the clients have approved the project

    Note: The clients approve the project when they move from the `request` folder to the `running` folder

    """
    fl_clients = get_all_directories(proj_folder / "fl_clients")
    project_unapproved_clients = []
    for fl_client in fl_clients:
        fl_client_app_path = client.datasite_path.parent / fl_client.name / "app_pipelines" / "fl_client"
        fl_client_request_folder = fl_client_app_path / "request" / proj_folder.name
        if not fl_client_request_folder.is_dir():
            # Create a request folder for the client
            fl_client_request_folder.mkdir(parents=True, exist_ok=True)

            # Copy the fl_config.json, model_arch.py and global_model_weights.pt to the request folder
            shutil.copy(proj_folder / "fl_config.json", fl_client_request_folder)
            shutil.copy(proj_folder / "model_arch.py", fl_client_request_folder)

            print(f"Request sent to {fl_client.name} for the project {proj_folder.name}")

        fl_client_running_folder = fl_client_app_path / "running" / proj_folder.name
        if not fl_client_running_folder.is_dir():
            project_unapproved_clients.append(fl_client.name)

    
    if project_unapproved_clients:
        raise StateNotReady(f"Project {proj_folder.name} is not approved by the clients {project_unapproved_clients}")


def load_model_class(model_path: Path) -> type:
    model_class_name = "FLModel"
    spec = importlib.util.spec_from_file_location(model_path.stem, model_path)
    model_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_arch)
    model_class = getattr(model_arch, model_class_name)  

    return model_class

def aggregate_model(
    fl_config, proj_folder, trained_model_paths,current_round
) -> Path:
    print("Aggregating the trained models")
    print(f"Trained model paths: {trained_model_paths}")
    global_model_class = load_model_class(proj_folder / fl_config["model_arch"])
    global_model: nn.Module = global_model_class()
    global_model_state_dict = global_model.state_dict()

    aggregated_model_weights = {}

    n_peers = len(trained_model_paths)
    for model_file in trained_model_paths:

        user_model_state = torch.load(str(model_file))
        for key in global_model_state_dict.keys():

            # If user model has a different architecture than my global model.
            # Skip it
            if user_model_state.keys() != global_model_state_dict.keys():
                raise ValueError("User model has a different architecture than the global model")

            if aggregated_model_weights.get(key, None) is None:
                aggregated_model_weights[key] = user_model_state[key] * (1 / n_peers)
            else:
                aggregated_model_weights[key] += user_model_state[key] * (1 / n_peers)

    
    global_model.load_state_dict(aggregated_model_weights)
    global_model_output_path = proj_folder / "agg_weights" / f"agg_model_round_{current_round}.pt"
    torch.save(global_model.state_dict(), str(global_model_output_path))

    return global_model_output_path

def shift_project_to_done_folder(client: Client, proj_folder: Path, total_rounds: int) -> None:
    """
    Moves the project to the `done` folder
    a. Create a directory in the `done` folder with the same name as the project
    b. moves the agg weights and fl_clients to the done folder
    c. delete the project folder from the running folder
    """
    done_folder = (
        Path(client.datasite_path) / "app_pipelines" / "fl_aggregator" / "done"
    )
    done_proj_folder = done_folder / proj_folder.name
    done_proj_folder.mkdir(parents=True, exist_ok=True)

    # Move the agg weights and round weights folder to the done project folder
    # Move the fl_clients folder to the done project folder
    shutil.move(proj_folder / "agg_weights", done_proj_folder)
    shutil.move(proj_folder / "fl_clients", done_proj_folder)

    # Delete the project folder from the running folder
    shutil.rmtree(proj_folder)

def evaluate_agg_model(agg_model: nn.Module, dataset_path: Path) -> float:
    agg_model.eval()

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
            outputs = agg_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    # Return accuracy as a percentage
    return accuracy /100

def save_model_accuracy_metrics(client: Client, proj_folder: Path,current_round: int, accuracy: float):
    """
    Saves the model accuracy in the public folder of the datasite under project name
    """
    metrics_folder = Path(client.datasite_path) / "public" / proj_folder.name
    # if the    metrics folder does not exist, create it and copy
    #  index.html and metrics.json from current directory
    if not metrics_folder.is_dir():
        metrics_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy("index.html", metrics_folder)
        shutil.copy("metrics.json", metrics_folder)
    
    metrics_file = metrics_folder / "metrics.json"
    # Schema of json files
    # [ {round: 1, accuracy: 0.98}, {round: 2, accuracy: 0.99} ]
    # Append the accuracy and round to the json file
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    metrics.append({"round": current_round, "accuracy": accuracy})
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

def advance_fl_round(client: Client, proj_folder: Path):
    """
    1. Wait for the trained model from the clients
    3. Aggregate the trained model and place it in the `agg_weights` folder
    4. Send the aggregated model to all the clients
    5. Repeat until all the rounds are complete
    """
    agg_weights_folder = proj_folder / "agg_weights"
    current_round = len(list(agg_weights_folder.iterdir()))

    with open(proj_folder / "fl_config.json", "r") as f:
        fl_config: dict = json.load(f)

    total_rounds = fl_config["rounds"]
    if current_round >= total_rounds + 1:
        print(f"FL project {proj_folder.name} is complete ✅")
        shift_project_to_done_folder(client, proj_folder, total_rounds)
        return


    participants = fl_config["participants"]

    if current_round == 1:
        for participant in participants:
            client_app_path = client.datasite_path.parent / participant / "app_pipelines" / "fl_client"
            client_agg_weights_folder = client_app_path / "running" / proj_folder.name / "agg_weights"
            client_round_1_model = client_agg_weights_folder / "agg_model_round_0.pt"
            if not client_round_1_model.is_file():
                shutil.copy(proj_folder / "agg_weights" / "agg_model_round_0.pt", client_agg_weights_folder)
    
    pending_clients = []
    trained_model_paths = []
    for participant in participants:
        participant_folder = proj_folder / "fl_clients" / participant
        participant_round_folder = participant_folder / f"trained_model_round_{current_round}.pt"
        trained_model_paths.append(participant_round_folder)
        if not participant_round_folder.is_file():
            pending_clients.append(participant)
    
    if pending_clients:
        raise StateNotReady(f"Waiting for trained model from the clients {pending_clients} for round {current_round}")
            
    
    # Aggregate the trained model
    agg_model_output_path = aggregate_model(fl_config, proj_folder, trained_model_paths, current_round)

    # Evaluate the aggregate model
    model_class = load_model_class(proj_folder / "model_arch.py")
    model: nn.Module = model_class()
    model.load_state_dict(torch.load(str(agg_model_output_path),weights_only=True))
    accuracy = evaluate_agg_model(model,TEST_DATASET_PATH)
    print(f"Accuracy of the aggregated model for round {current_round}: {accuracy}")
    save_model_accuracy_metrics(client, proj_folder, current_round, accuracy)

    # Send the aggregated model to all the clients
    for participant in participants:
        client_app_path = client.datasite_path.parent / participant / "app_pipelines" / "fl_client"
        client_agg_weights_folder = client_app_path / "running" / proj_folder.name / "agg_weights"
        shutil.copy(agg_model_output_path, client_agg_weights_folder)


def _advance_fl_project(client: Client,proj_folder: Path) -> None:
    """
    Iterate over all the project folder, it will try to advance its state.
    1. Has the client installed the fl_client app or not (app_pipelines/fl_client), if not throw an error message
    2. have we submitted the project request to the clients  (app_pipelines/fl_client/request)
    3. Have all the clients approved the project or not.
    4. let assume the round ix x,  place agg_model_round_x.pt inside all the clients
    5. wait for the trained model from the clients 
    6. aggregate the trained model 
    7. repeat d until all the rounds are complete
    """

    try: 
        check_fl_client_installed(client, proj_folder)

        check_proj_requests(client, proj_folder)

        advance_fl_round(client, proj_folder)
    
    except StateNotReady as e:
        print(e)
        return


def advance_fl_projects(client: Client) -> None:
    """
    Iterates over the `running` folder and tries to advance the FL projects
    """
    running_folder = (
        Path(client.datasite_path) / "app_pipelines" / "fl_aggregator" / "running"
    )
    for proj_folder in running_folder.iterdir():
        if proj_folder.is_dir():
            proj_name = proj_folder.name
            print(f"Advancing FL project {proj_name}")
            _advance_fl_project(client, proj_folder)




if __name__ == "__main__":
    client = Client.load()

    # Step 1: Init the FL Aggregator App
    init_fl_aggregator_app(client)

    # Step 2: Launch the FL Project
    # Iterates over the `launch` folder and creates a new FL project
    # if the `fl_config.json` is found in the `launch` folder
    launch_fl_project(client)

    # Step 3: Advance the FL Projects.
    # Iterates over the running folder and tries to advance the FL project
    advance_fl_projects(client)