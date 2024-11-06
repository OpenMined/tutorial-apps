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
    TODO: change my_cool_fl_proj to be under launch/, running/ and done/ folders 
    TODO: manually copy the fl_config.json, model_arch.py and global_model_weight.pt to the launch/ folder
    app_pipelines
    └── fl_aggregator
            └── my_cool_fl_proj
                └── launch
                        ├── fl_config.json
                        ├── model_arch.py
                        ├── global_model_weights.py
                └── running
                        ├── fl_clients 
                        │   ├── a@openmined.org
                        │   ├── b@openmined.org
                        │   ├── c@openmined.org
                        ├── agg_weights  # to store aggregator's weights for each round
                        ├── fl_config.json  # moved from the launch folder after the app start
                        ├── global_model_weights.pt
                        ├── model_arch.py  # moved from the launch folder
                        └── state.json
                └── done
                        └── aggregated_model_weights.pt
    ```
    """
    def __init__(self, config_path: str | Path, client: Client) -> None:
        self.config_path = config_path
        self.client = client
        self.load_config()
        self.setup_paths()

    def load_config(self) -> None:
        with open(Path(self.config_path), "r") as f:
            self.configs = json.load(f)
            pprint(self.configs)
            self.proj_name = self.configs["project_name"]
            self.participants = set(self.configs["participants"]).intersection(self.all_users())
            self.model_arch_path = Path(self.configs["model_arch"])
            self.global_model_weight_path = Path(self.configs["model_weight"])

    def setup_paths(self) -> None:
        self.app_pipelines_proj_path = Path(self.client.datasite_path) / "app_pipelines" / "fl_aggregator" / self.proj_name
        self.launch_folder = self.app_pipelines_proj_path / "launch"
        self.running_folder = self.app_pipelines_proj_path / "running"
        self.done_folder = self.app_pipelines_proj_path / "done"
        self.fl_clients_folder = self.running_folder / "fl_clients"
        self.agg_weights_folder = self.running_folder / "agg_weights"
    
    def create_dirs(self) -> None:
        self.app_pipelines_proj_path.mkdir(parents=True, exist_ok=True)
        self.launch_folder.mkdir(parents=True, exist_ok=True)
        self.running_folder.mkdir(parents=True, exist_ok=True)
        self.done_folder.mkdir(parents=True, exist_ok=True)
        self.fl_clients_folder.mkdir(parents=True, exist_ok=True)
        self.agg_weights_folder.mkdir(parents=True, exist_ok=True)

        for participant in self.participants:
            participant_folder = self.fl_clients_folder / participant
            participant_folder.mkdir(parents=True, exist_ok=True)

    @property
    def participants_proj_path(self):
        return [
            Path(self.client.datasite_path.parent) / participant / "app_pipelines" / "fl_client" / self.proj_name
            for participant in self.participants
        ]

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


def launch(proj_workspace: ProjectWorkspace) -> None:
    """
    Copies the `fl_config.json`, `model_arch.py` and `global_model_weight.pt`
    to the `launch` folder. If the `global_model_weight.pt` does not exist,
    creates random initial weights from `model_arch.py`
    """

    config_dst = proj_workspace.launch_folder / "fl_config.json"
    model_arch_dst = proj_workspace.launch_folder / "model_arch.py"
    global_model_dst = proj_workspace.launch_folder / "global_model_weight.pt"
    
    # Copy the config file
    if not config_dst.is_file():
        shutil.copy(proj_workspace.config_path, config_dst)

    if not model_arch_dst.is_file():
        # Copy the model architecture file
        shutil.copy(proj_workspace.model_arch_path, model_arch_dst)

    if not global_model_dst.is_file():
        if proj_workspace.global_model_weight_path.is_file():
            # Copy the global model weights
            print(f"Copying initial weights to {global_model_dst}")
            shutil.copy(proj_workspace.global_model_weight_path, global_model_dst)
        else:
            from model_arch import FLModel
            import torch
            model = FLModel()
            torch.save(model.state_dict(), global_model_dst)
            print(f"Creating random initial weights and save to {global_model_dst}")


def request_fl_client(proj_workspace: ProjectWorkspace):
    """
    Creates a request to the participants to join the FL flow by copying the
    content of the `launch` folder to the participant client's 
    `app_pipelines/fl_client/request` folder
    """
    print("Requesting participants to join the FL flow")
    for participant_proj_path in proj_workspace.participants_proj_path:
        request_folder = participant_proj_path / "request"
        # request_folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(proj_workspace.launch_folder, request_folder, dirs_exist_ok=True)
        print(f"Request sent to {participant_proj_path}")


if __name__ == "__main__":
    client = Client.load()

    fl_config_path = Path("fl_config.json")
    proj_workspace =  ProjectWorkspace(fl_config_path, client)
    proj_workspace.create_dirs()

    launch(proj_workspace)
    request_fl_client(proj_workspace)