from syftbox.lib import Client
from pathlib import Path
import json
import shutil
import os

# Exception name to indicate the state cannot advance
# as there are some pre-requisites that are not met
class StateNotReady(Exception):
    pass


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
        proj_configs: dict = json.load(f)
    
    
    proj_name = proj_configs["project_name"]
    participants = proj_configs["participants"]

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

        # copy the config file to the project's running folder
        shutil.copy(fl_config_json_path, proj_folder)

        # move the model architecture and global model weights to the project's running folder
        model_arch_src = fl_aggregator / "launch" / proj_configs["model_arch"]
        shutil.move(model_arch_src, proj_folder)

        global_model_src = fl_aggregator / "launch" / proj_configs["model_weight"]
        shutil.move(global_model_src, proj_folder)


        # TODO: create a state.json file to keep track of the project state
        # if needed while running the FL rounds

        # Remove fl_config.json, model_arch.py and global_model_weights.pt from the launch folder
        # TODO: Can we use shutil to remove the files?
        os.remove(fl_config_json_path)
        os.remove(model_arch_src)
        os.remove(global_model_src)

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
            users.append(entry)

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

        fl_client_app_path = client.datasite_path.parent / fl_client.name / "apps" / "fl_client"
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
        fl_client_app_path = client.datasite_path.parent / fl_client.name / "apps" / "fl_client"
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
    


def _advance_fl_project(client: Client,proj_folder: Path) -> None:
    """
    Iterate over all the project folder, it will try to advance its state.
    1. Has the client installed the fl_client app or not (app_pipelines/fl_client), if not throw an error message
    2. have we submitted the project request to the clients  (app_pipelines/fl_client/request)
    3. Have all the clients approved the project or not.
    4. let assume the round ix x,  place global_weights_round_x.pt inside all the clients
    5. wait for the trained model from the clients 
    6. aggregate the trained model 
    7. repeat d until all the rounds are complete
    """

    try: 
        check_fl_client_installed(client, proj_folder)

        check_proj_requests(client, proj_folder)

        # advance_fl_round(client, proj_folder)
    
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