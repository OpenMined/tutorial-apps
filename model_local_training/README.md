# Model Local Training App

This app, together with the `model_aggregator` app, form a semi-FL app using SyftBox

## Running in dev mode

0. Clone the `model_local_training_app` and `model_aggregator` branches into the current directory
1. Run 3 clients (`a, b, c`) with `just run-client a` and the SyftBox cache server `just run-server`
2. Install the local training app on `b, c`: `syftbox app install model_local_training_app --config_path <path_to_config.json>` where `<path_to_config.json>` points to `b` or `c`'s `config.json` file
3. Install the model aggregator app on `a`: `syftbox app install model_aggregator --config_path <path_to_config.json>` where `<path_to_config.json>` points to `a`'s `config.json` file
4. Moving the MNIST data parts in `b` and `c`'s `private` into `private/datasets` to train
5. The training logs can be seen at `b` and `c`'s `public/training.log`, and the trained model `.pt` will also be in the `public` folder once it's trained
6. `a` will automatically aggregate the models from `b` and `c` when it detects if there are new trained model in `b` and `c`'s public folders. Check `a`'s SyftBox client log to see if the global model accuracy has improved. Look for key words "Aggregating models between" and "Global model accuracy"
