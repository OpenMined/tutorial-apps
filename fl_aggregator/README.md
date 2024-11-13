# FL Aggregator

## Usage
**1. Install the app**
```
syftbox app install OpenMined/tutorial-apps --branch fl_aggregator
```

**2. Agree on who will aggregate and who will participate**
In this example, we'll have
- Aggregator: user1@openmined.org
- Participants: user2@openmined.org and user3@openmined.org

NOTE:
- Aggregator must decide and implement the model architecture, in this example it's implemented in `model_arch.py`
- Aggregator will also provide a seed model weights which will be fine-tuned by each participant.

**3. Setup the FL config**

Based on the above, we create/edit `fl_config.json` to configure our federated learning flow

```json
{
    "project_name": "MNIST_FL",
    "aggregator": "user1@openmined.org",
    "participants": ["user2@openmined.org","user3@openmined.org"],
    "rounds": 3,
    "model_arch": "model_arch.py",
    "model_weight": "global_model_weight.pt",
    "epoch": 10,
    "learning_rate": 0.1
}
```

**4. Kickstart FL**

To start the FL training, we must copy the following files in `datasites/<aggregator_email>/app_pipelines/fl_aggregator/launch` directory
- `fl_config.json`
- `global_model_weight.pt`
- `model_arch.py`

If this directory isn't available, either run the syftbox client with fl_aggregator app installed OR create it manually.


## Post-kickoff

- Aggregator will wait for all participants to apps_piplines/fl_client
  - Once available, aggreator will copy these seed files (`fl_config.json`, `model_arch.py`) to each participants apps_piplines/fl_client
- Once all clients have approved, the FL rounds kick start
  - Sending the model
  - Client training on that
  - Client saving the model at in aggregator's pipeline

## Development

1. The aggregator install the aggregator app with `syftbox app install ./fl_aggregator --config <path_to_config.json>` where `<path_to_config.json>` points to the client's `config.json` (no the app's `config.json`)
2. Once the app is installed, the `launch` folder is created in `apps_pipeline/fl_aggregator/` together with other folders (`running`, and `done`)
3. Inside `launch`, there are 3 files: `config.json`, `model_arch.py`, `global_model_weights.pt` that the aggregator have to put in the `launch` folder
4. Once the files are in the `launch` folder, the app will create a project with the name specified in the `config.json` file inside the `running` folder, e.g. `my_cool_fl_project`
5. Inside the `my_cool_fl_project` folder, the app will create a folder for each participant where participants' models for each round are collected. Each participant folder will have a `syft_perm` file that contains the permissions to who can read / write its models
5. Then the `fl_aggregator` creates a request to the participants to join the FL flow
6. 

```
app_pipelines
└── fl_server
    ├── launch
    │   ├── config.json
    │   ├── model_arch.py
    │   ├── global_model_weights.py
    └── running
        └── my_cool_fl_proj
            ├── fl_clients 
            │   ├── a@openmined.org
            │   ├── b@openmined.org
            │   ├── c@openmined.org
            ├── agg_weights  # to store aggregator's weights for each round
            ├── config.json  # moved from the launch folder after the app start
            ├── global_model_weights.pt
            ├── model_arch.py  # moved from the launch folder
            └── state.json
    └── done
        └── my_cool_fl_proj
            └── aggregated_model_weights.pt
```