1. The aggregator install the aggregator app with `syftbox app install fl_aggregator --config <path_to_config.json>` where `<path_to_config.json>` points to the client's `config.json` (no the app's `config.json`)
2. Once the app is installed, the `launch` folder is created in `apps_pipeline/fl_aggregator/` together with other folders (`running`, and `done`)
3. Inside `launch`, there are 3 files: `config.json`, `model_arch.py`, `global_model_weights.pt` that the aggregator have to put in the `launch` folder
4. Once the files are in the `launch` folder, the app will create a project with the name specified in the `config.json` file inside the `running` folder, e.g. `my_cool_fl_project`
5. Inside the `my_cool_fl_project` folder, the app will create a folder for each participant where participants' models for each round are collected. Each participant folder will have a `syft_perm` file that contains the permissions to who can read / write its models
6. Then the `fl_aggregator` creates a request to the participants to join the FL flow
7.

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
