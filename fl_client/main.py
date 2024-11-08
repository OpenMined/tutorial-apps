from syftbox.lib import Any, Client, SyftPermission
from pathlib import Path
from typing import Any
import torch
import json
import os
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


class ProjectWorkspace:
    """
    app_pipelines
    └── fl_client
        ├── request
            └── my_cool_fl_proj
                ├── fl_config.json
                ├── global_model_weight.pt
                └── model_arch.py
        └── running
            └── my_cool_fl_proj
                ├── fl_config.json
                ├── global_model_weight.pt
                ├── model_arch.py
                ├── dataset
                ├── round_weights
                │   ├── trained_model_round_1.pt
                │   └── trained_model_round_2.pt
                └── agg_weights
                    ├── agg_model_round_1.pt
                    └── agg_model_round_2.pt
    """

    def __init__(self, client: Client) -> None:
        self.client = client
        self.setup_paths()

    def setup_paths(self) -> None:
        self.app_pipelines_proj_path: Path = (
            Path(self.client.datasite_path) / "app_pipelines" / "fl_client"
        )
        self.request_folder = self.app_pipelines_proj_path / "request"

        self.running_folder = self.app_pipelines_proj_path / "running"
        # self.round_weights_folder = self.running_folder / "round_weights"
        # self.agg_weights_folder = self.running_folder / "agg_weights"
        # self.datasets_folder = self.running_folder / "dataset"

    def create_dirs(self) -> None:
        self.app_pipelines_proj_path.mkdir(parents=True, exist_ok=True)
        self.request_folder.mkdir(parents=True, exist_ok=True)
        (self.request_folder / "dummy").touch()
        self.running_folder.mkdir(parents=True, exist_ok=True)

        fl_permissions = SyftPermission.datasite_default(client.email)  # type: ignore
        fl_permissions.write.append("GLOBAL")
        fl_permissions.read.append("GLOBAL")
        fl_permissions.filepath = str(self.app_pipelines_proj_path / "_.syftperm")
        fl_permissions.save()

    def create_fl_project_dirs(self) -> None:
        fl_projects = [d.name for d in self.request_folder.iterdir() if d.is_dir()]
        self.dataset_folders = {}
        self.fl_projects_paths = []
        for fl_proj in fl_projects:
            proj_running_folder = self.running_folder / fl_proj
            proj_running_folder.mkdir(parents=True, exist_ok=True)
            self.fl_projects_paths.append(proj_running_folder)
            round_weights_folder = proj_running_folder / "round_weights"
            agg_weights_folder = proj_running_folder / "agg_weights"
            round_weights_folder.mkdir(parents=True, exist_ok=True)
            agg_weights_folder.mkdir(parents=True, exist_ok=True)

            # Create dataset folder which is used to put datasets during training task.
            datasets_folder = proj_running_folder / "dataset"
            datasets_folder.mkdir(parents=True, exist_ok=True)
            fl_permissions = SyftPermission.datasite_default(client.email)  # type: ignore
            fl_permissions.filepath = str(datasets_folder / "_.syftperm")
            fl_permissions.save()
            self.dataset_folders[proj_running_folder] = datasets_folder

    def load_fl_config(self, path: Path) -> dict[str, Any] | None:
        fl_config = None
        fl_config_path: Path = path / "fl_config.json"
        if not fl_config_path.exists():
            return None
        with open(str(path / "fl_config.json"), "r") as fl_config_file:
            fl_config = json.load(fl_config_file)

        return fl_config

    def load_datasets(self, fl_proj: Path) -> None | DataLoader:
        dataset_folder = self.dataset_folders.get(fl_proj, None)
        if dataset_folder is None:
            raise Exception("Invalid FL Project")

        dataset_path_files = [
            f for f in os.listdir(str(dataset_folder)) if f.endswith(".pt")
        ]

        if len(dataset_path_files) == 0:
            return None
        else:
            all_datasets = []
            for dataset_file in dataset_path_files:
                # load the saved mnist subset
                images, labels = torch.load(str(dataset_folder) + "/" + dataset_file)
                # create a tensordataset
                dataset = TensorDataset(images, labels)
                all_datasets.append(dataset)

            combined_dataset = ConcatDataset(all_datasets)
            # create a dataloader for the dataset
            return DataLoader(combined_dataset, batch_size=64, shuffle=True)


if __name__ == "__main__":
    client = Client.load()

    proj_workspace = ProjectWorkspace(client)
    proj_workspace.create_dirs()
    proj_workspace.create_fl_project_dirs()

    for project in proj_workspace.fl_projects_paths:
        print(f"Processing project: {project}")
        fl_config = proj_workspace.load_fl_config(project)
        if fl_config is None:
            print("Config file not found. Skipping training pipeline.")
            continue

        dataset = proj_workspace.load_datasets(project)
        if dataset is None:
            print("No datasets found. Skipping training pipeline.")
