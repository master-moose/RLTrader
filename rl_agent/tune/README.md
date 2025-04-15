# Ray Tune Hyperparameter Optimization

This directory contains scripts for hyperparameter optimization of the RecurrentPPO agent using Ray Tune.

## Installation

First, install the required dependencies:

```bash
pip install "ray[tune]" hyperopt optuna
```

## Running Hyperparameter Optimization

To run a hyperparameter optimization experiment, use the `run_tune_sweep.py` script:

```bash
python rl_agent/tune/run_tune_sweep.py \
  --data_path /path/to/training_data.csv \
  --val_data_path /path/to/validation_data.csv \
  --exp_name ppo_stability_sweep \
  --num_samples 20 \
  --cpus_per_trial 2 \
  --gpus_per_trial 0.25 \
  --search_algo optuna
```

### Important Parameters

- `--data_path`: Path to the training data (required)
- `--val_data_path`: Path to the validation data (required)
- `--data_key`: Key for HDF5 data files (if applicable)
- `--exp_name`: Name for the experiment (default: "ppo_tune")
- `--num_samples`: Number of trials to run (default: 20)
- `--cpus_per_trial`: CPU resources per trial (default: 2.0)
- `--gpus_per_trial`: GPU resources per trial (default: 0.25)
- `--search_algo`: Search algorithm to use ("basic", "optuna", or "hyperopt") (default: "optuna")
- `--timesteps_per_trial`: Timesteps per trial (default: 1,000,000)
- `--base_config`: Path to a JSON file with base configuration (optional)

## Current Search Space

The current hyperparameter optimization focuses on stability parameters:

- `learning_rate`: Log-uniform from 1e-5 to 1e-3
- `gamma`: Choice of [0.9, 0.95, 0.99, 0.995, 0.999]
- `n_steps`: Choice of [512, 1024, 2048, 4096, 8192]

## Using the Results

After optimization completes, the best configuration will be saved to:

```
./ray_results/{exp_name}/best_config.json
```

You can use this configuration to train a model with:

```bash
python rl_agent/train.py --load_config ./ray_results/{exp_name}/best_config.json --data_path <path> [other options]
```

## Extending the Search Space

To modify the search space, edit the `define_search_space()` function in `run_tune_sweep.py`.

## Dashboard

Ray Tune provides a dashboard for monitoring experiments. When you run an experiment, Ray will output a URL (typically http://127.0.0.1:8265) where you can view the progress of your trials. 