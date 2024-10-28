from pathlib import Path
from syftbox.lib import Client

def aggregate(participants: list[str], datasite_path: Path):
    """
    Aggregates numerical values from a list of participants.

    Args:
        participants (list[str]): List of participants' folder names.
        datasite_path (Path): Path to the base directory containing participant folders.

    Returns:
        tuple: A tuple containing the total aggregated value and a list of participants without a value file.
    """
    total = 0  # Accumulator to store the total value
    missing = []  # List to track participants missing the value.txt file

    # Iterate over the participant folders
    for user_folder in participants:
        # Construct the path to the value.txt file for the current participant
        value_file: Path = Path(datasite_path) / user_folder / "public" / "value.txt"

        # If the value file exists, read and add its value to the total
        if value_file.exists():
            with value_file.open("r") as file:
                total += float(file.read())
        else:
            # If the value file is missing, add the participant to the missing list
            missing.append(user_folder)

    # Return the total aggregated value and the list of missing participants
    return total, missing


if __name__ == "__main__":
    # Load the Syftbox client configuration to locate datasite directories
    client = Client.load()

    # Define the list of participants (identified by their folder names)
    participants = []

    # Aggregate values from participants
    total, missing = aggregate(participants, client.datasite_path.parent)

    # Print the results
    print("\n====================\n")
    print("Total aggregation value: ", total)
    print("Missing value.txt peers: ", missing)
    print("\n====================\n")
