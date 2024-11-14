import json
from pathlib import Path
from enum import Enum

class ParticipantStateCols(Enum):
    EMAIL = 'Email'
    FL_CLIENT_INSTALLED = 'Fl Client Installed'
    PROJECT_APPROVED = 'Project Approved'
    HAS_DATA = 'Has Data'
    ROUND = 'Round (current/total)'

def read_json(data_path: Path):
    with open(data_path) as fp:
        data = json.load(fp) 
    return data


def save_json(data: dict, data_path: Path):
    with open(data_path, "w") as fp:
        json.dump(data, fp, indent=4)


def create_participant_json_file(participants: list, total_rounds: int, output_path: Path):
    data = []
    for participant in participants:
        data.append({
            ParticipantStateCols.EMAIL.value               : participant,
            ParticipantStateCols.FL_CLIENT_INSTALLED.value : False,
            ParticipantStateCols.PROJECT_APPROVED.value    : False,
            ParticipantStateCols.HAS_DATA.value            : False,
            ParticipantStateCols.ROUND.value               : f'0/{total_rounds}'
        })

    save_json(data=data, data_path=output_path)

def update_json(data_path: Path, participant_email: str, column_name: ParticipantStateCols, column_val: str):

    if column_name not in ParticipantStateCols:
        return
    participant_history = read_json(data_path=data_path)
    for participant in participant_history:
        if participant[ParticipantStateCols.EMAIL.value] == participant_email:
            participant[column_name.value] = column_val
    
    save_json(participant_history, data_path)
