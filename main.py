from syftbox.lib import Client
from pathlib import Path
import json
from pprint import pprint
import shutil


class ProjectWorkspace:
    """
    Hold the app configs and create the app directories
    according to the structure below:
    ```
    app_pipelines
    └── fl_aggregator
            └── launch
            └── running
                └── <fl_project_name>
                    ├── fl_clients
                    │   ├── a@openmined.org
                    │   ├── b@openmined.org
                    │   ├── c@openmined.org
                    ├── agg_weights  # to store aggregator's weights for each round
                    ├── fl_config.json  # moved from the launch folder after the app start
                    ├── global_model_weights.pt  # moved from the launch folder
                    ├── model_arch.py  # moved from the launch folder
                    └── state.json
            └── done
                └── <fl_project_name>
                    └── aggregated_model_weights.pt
    ```
    """

    def __init__(self, client: Client, project_configs: dict) -> None:
        self.client = client
        self.configs = project_configs
        self.proj_name = self.configs["project_name"]
        self.participants = self.configs["participants"]
        self.setup_paths()

    def setup_paths(self) -> None:
        self.app_pipelines_app_path = (
            Path(self.client.datasite_path) / "app_pipelines" / "fl_aggregator"
        )
        self.running_folder = self.app_pipelines_app_path / "running" / self.proj_name
        self.done_folder = self.app_pipelines_app_path / "done" / self.proj_name
        self.fl_clients_folder = self.running_folder / "fl_clients"
        self.agg_weights_folder = self.running_folder / "agg_weights"
        
        self.configs_path = self.running_folder / "fl_config.json"
        self.model_arch_path = self.running_folder / self.configs["model_arch"]
        self.global_model_weight_path = self.running_folder / self.configs["model_weight"]

    def create_dirs(self) -> None:
        self.app_pipelines_app_path.mkdir(parents=True, exist_ok=True)
        self.running_folder.mkdir(parents=True, exist_ok=True)
        self.done_folder.mkdir(parents=True, exist_ok=True)
        self.agg_weights_folder.mkdir(parents=True, exist_ok=True)

        self.fl_clients_folder.mkdir(parents=True, exist_ok=True)
        for participant in self.participants:
            participant_folder = self.fl_clients_folder / participant
            participant_folder.mkdir(parents=True, exist_ok=True)
        
        # TODO: create a custom syft permission for the clients in the `fl_clients` folder

    @property
    def participants_proj_path(self):
        return [
            Path(self.client.datasite_path.parent)
            / participant
            / "app_pipelines"
            / "fl_client"
            for participant in self.participants
        ]
    
    @property
    def online_participants(self):
        return set(self.participants).intersection(self.all_users())

    def all_users(self) -> list[str]:
        exclude_dir = ["apps", ".syft"]
        entries = self.client.datasite_path.parent.iterdir()
        users = []
        for entry in entries:
            is_excluded_dir = entry.name in exclude_dir
            is_valid_peer = entry.is_dir() and not is_excluded_dir
            if is_valid_peer:
                users.append(entry.name)
        return users


def launch_fl_project(client: Client) -> ProjectWorkspace | None:
    """
    - Create the `launch` folder for the `fl_aggregator` app
        app_pipelines
        └── fl_aggregator
                └── launch
                    ├── fl_config.json (dragged and dropped by the FL aggregator)
                    ├── model_arch.py (dragged and dropped by the FL aggregator)
                    ├── global_model_weights.py (dragged and dropped by the FL aggregator)
    - Read the `fl_config.json` file in the `launch` folder
    - Check if the project exists in the `running` folder with the same `project_name`.
        If not, create a new `ProjectWorkspace` that keeps track of in `project_name/running/`
        and `project_name/done/`
    a. creates a directory with the project name in running folder
    b. inside the project it creates the folders of clients with a custom syft permissions
    c. copies over the fl_config.json and model_arch.py and global_model_weights.pt
    """
    launch_folder = (
        Path(client.datasite_path) / "app_pipelines" / "fl_aggregator" / "launch"
    )
    launch_folder.mkdir(parents=True, exist_ok=True)

    fl_config_json = launch_folder / "fl_config.json"
    if not fl_config_json.is_file():
        print(f"`fl_config.json` not found in the {launch_folder} folder. Please put it there.")
        return

    # this should be the only place where the `fl_config.json` be read (single source of truth)
    with open(fl_config_json, "r") as f:
        proj_configs: dict = json.load(f)
        proj_workspace = ProjectWorkspace(client, proj_configs)

    if proj_workspace.running_folder.is_dir():
        print(f"FL project {proj_workspace.proj_name} already exists")
    else:
        print(f"Creating new FL project {proj_workspace.proj_name}")
        proj_workspace.create_dirs()
        # copy the config file to the project's running folder
        if not proj_workspace.configs_path.is_file():
            shutil.copy(fl_config_json, proj_workspace.configs_path)

    # move the model architecture and global model weights to the project's running folder
    model_arch_src = launch_folder / proj_workspace.configs["model_arch"]
    if not proj_workspace.model_arch_path.is_file():
        if not model_arch_src.is_file():
            print(f"Model architecture file not found in the {launch_folder} folder. Please put it there!")
        else:
            shutil.move(model_arch_src, proj_workspace.model_arch_path)  

    global_model_src = launch_folder / proj_workspace.configs["model_weight"]
    if not proj_workspace.global_model_weight_path.is_file():
        if not global_model_src.is_file():
            print(f"Initial global model weight not found in the {launch_folder} folder. Please put it there!")
            # print(
            #     f"Creating random initial weights and save to {proj_workspace.global_model_weight_path}"
            # )
            # from model_arch import FLModel
            # import torch
            # model = FLModel()
            # torch.save(model.state_dict(), proj_workspace.global_model_weight_path)
            return
        else:
            print(f"Moving initial global model weights to {proj_workspace.global_model_weight_path}")
            shutil.move(global_model_src, proj_workspace.global_model_weight_path)

    return proj_workspace


def request_fl_client(proj_workspace: ProjectWorkspace):
    """
    Creates a request to the participants to join the FL flow by copying the
    content of the `launch` folder to the participant client's
    `app_pipelines/fl_client/request` folder
    """
    print("Requesting participants to join the FL flow")
    
    # TODO: check first if the fl_client folder exists in the participant's app_pipelines
    # after the client installs the fl_client app, copy the request folder to the fl_client folder
    for participant_proj_path in proj_workspace.participants_proj_path:
        request_folder = participant_proj_path / "request"
        # request_folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            proj_workspace.launch_folder, request_folder, dirs_exist_ok=True
        )
        print(f"Request sent to {participant_proj_path}")


def run_fl_rounds():
    """
    Check if all the clients approved the project or not.
    If so we do 
        while round <= max_rounds:
            d. let assume the round ix x,  place global_weights_round_x.pt inside all the clients
            e. wait for the trained model of round x from the clients 
            f. aggregate the trained model of round x
    """
    pass


if __name__ == "__main__":
    client = Client.load()
    
    proj_workspace = launch_fl_project(client)

    # request_fl_client(proj_workspace)