from syftbox.lib import Client
import random
import os
from pathlib import Path
import shutil


def copy_random_pretrained_model(path: Path):
    pattern = "pretrained_mnist_label_[0-9]\.pt$"
    # check if there is any pretrained model in the path
    if len(list(path.glob(pattern))) > 0:
        return

    pretrained_model_path = Path("./") / "pretrained_models"
    entries = os.listdir(pretrained_model_path)
    # generate random number from the number of entries
    random_index = random.randint(0, len(entries) - 1)
    # get the random entry
    random_entry = entries[random_index]
    # copy the random entry to the path
    shutil.copy2(pretrained_model_path / random_entry, path)
    

if __name__ == "__main__":
    client = Client.load()

    # Create Private Directory
    private_path = Path(client.datasite_path ) / "private"
    os.makedirs(private_path, exist_ok=True)

    # Copy a Randon Pretrained Model to private
    copy_random_pretrained_model(private_path)