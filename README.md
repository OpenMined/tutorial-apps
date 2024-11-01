# Pretrained Model Aggregator App

This app, together with the `pretrained_model_local` app, form an aggregation pipeline for pretrained MNIST models using SyftBox

## Running in dev mode
0. Reset everything if necessary with `just reset`. Clone the `pretrained_model_local` and `pretrained_model_aggregator` branches into the current directory, e.g. with the command `git clone --branch pretrained_model_aggregator https://github.com/OpenMined/tutorial-apps.git pretrained_model_aggregator`
1. Run upto 11 clients (`a, b, c...`) with `just run-client <client_name>` and the SyftBox cache server `just run-server`. Here, `a` will be the model aggregator
2. Install the local pretrained app on `b, c, ...`: `syftbox app install pretrained_model_local --config_path <path_to_config.json>` where `<path_to_config.json>` points to `b` or `c` or other clients' `config.json` file
3. Install the model aggregator app on `a`: `syftbox app install pretrained_model_aggregator --config_path <path_to_config.json>` where `<path_to_config.json>` points to `a`'s `config.json` file
4. In `a`'s `apps/pretrained_model_aggregator` folder, modify the participants in `participants.json`
4. Move the participants' MNIST pretrained model in their `private` folder into `public` to let it be aggregated by `a`
5. `a` will automatically look for and aggregate the models from other clients. Check `a`'s SyftBox client log to see if the global model accuracy has improved, look for key words "Aggregated models from" and "Global model accuracy"

## Running live as a model aggregator
0. Clone the `pretrained_model_aggregator` branch into the current directory, e.g. with the command `git clone --branch pretrained_model_aggregator https://github.com/OpenMined/tutorial-apps.git pretrained_model_aggregator`
1. Run your syft client with `syftbox client`
2. Install the aggregator app `syftbox app install pretrained_model_aggregator`
3. Go to your sync folder's `apps/pretrained_model_aggregator` and change the list of participants in `participants.json`
4. Wait for the participants to submit their pretrained models. Monitor your syftbox client logs, look for key words "Aggregated models from" and "Global model accuracy"