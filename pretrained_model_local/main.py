from syftbox.lib import Client
import random
import os
from pathlib import Path
import shutil
import re


def copy_random_pretrained_model(path: Path):
    pattern = "pretrained_mnist_label_[0-9]\.pt$"
    # check if there is any pretrained model in the path
    pretrained_entries = os.listdir(path)
    pretrained_model_exists = any(
        [re.match(pattern, entry) for entry in pretrained_entries]
    )
    if pretrained_model_exists:
        return

    pretrained_model_path = Path("./") / "pretrained_models"
    entries = os.listdir(pretrained_model_path)
    # generate random number from the number of entries
    random_index = random.randint(0, len(entries) - 1)
    # get the random entry
    random_entry = entries[random_index]

    print("Copying Pretrained Model: ", random_entry)
    # copy the random entry to the path
    shutil.copy2(pretrained_model_path / random_entry, path)


if __name__ == "__main__":
    client = Client.load()

    # Create Private Directory
    private_path = Path(client.datasite_path) / "private"
    os.makedirs(private_path, exist_ok=True)

    # Copy a Randon Pretrained Model to private
    copy_random_pretrained_model(private_path)
