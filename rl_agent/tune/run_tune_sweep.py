#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Ray Tune hyperparameter sweeps for the RecurrentPPO agent.

This script configures and launches hyperparameter optimization using Ray Tune
for the RecurrentPPO reinforcement learning agent on financial time series data.
"""

import argparse
import json
import os
import sys
import traceback
from typing import Dict, Any
import shortuuid  # Add this import

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables for Ray
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# Ray Tune imports with improved error handling
RAY_AVAILABLE = False
try:
    import ray
    print(f"Ray version: {ray.__version__}")
    try:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune import CLIReporter
        from ray.air.config import CheckpointConfig  # Import CheckpointConfig
        # Remove RunConfig import since it's not available in Ray 2.5.1
        RAY_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        print(f"ERROR importing Ray Tune modules: {e}")
        traceback.print_exc()
        print("\nRay was imported but Ray Tune components could not be imported.")
        print("This might be due to a version mismatch or an incomplete Ray installation.")
except ImportError as e:
    print(f"ERROR importing Ray base package: {e}")
    print("Ray base package could not be imported.")

if not RAY_AVAILABLE:
    print("ERROR: Ray Tune not available. Please ensure you have the correct versions installed:")
    print("pip install 'ray[tune]>=2.0.0' hyperopt>=0.2.7 optuna>=3.0.0")
    sys.exit(1)

# Import the trainable function from train.py
from rl_agent.train import train_rl_agent_tune
from rl_agent.utils import load_config, ensure_dir_exists

# Default configuration with fixed parameters
DEFAULT_CONFIG = {
    # Model type
    "model_type": "recurrentppo",
    
    # Features - most important for all timeframes, using _scaled columns
    "features": "open_scaled,high_scaled,low_scaled,close_scaled,volume_scaled,sma_7_scaled,sma_25_scaled,sma_99_scaled,ema_9_scaled,ema_21_scaled,rsi_14_scaled,open_scaled_4h,high_scaled_4h,low_scaled_4h,close_scaled_4h,volume_scaled_4h,sma_7_scaled_4h,sma_25_scaled_4h,sma_99_scaled_4h,ema_9_scaled_4h,ema_21_scaled_4h,rsi_14_scaled_4h,open_scaled_1d,high_scaled_1d,low_scaled_1d,close_scaled_1d,volume_scaled_1d,sma_7_scaled_1d,sma_25_scaled_1d,sma_99_scaled_1d,ema_9_scaled_1d,ema_21_scaled_1d,rsi_14_scaled_1d",
    
    # Reward structure - simplified as requested
    "drawdown_penalty_weight": 0.0,
    "sharpe_reward_weight": 0.0,
    "fee_penalty_weight": 0.0,
    "benchmark_reward_weight": 0.0,
    "consistency_penalty_weight": 0.0,
    "idle_penalty_weight": 0.0,
    "profit_bonus_weight": 0.0,
    "exploration_bonus_weight": 0.0,
    "trade_penalty_weight": 0.0,
    
    # Fixed PPO params as requested
    "n_epochs": 10,
    "clip_range": 0.1,
    "vf_coef": 0.5,
    "ent_coef": "0.01",
    
    # Fixed LSTM params as requested
    "lstm_hidden_size": 256,
    "n_lstm_layers": 2,
    "shared_lstm": "shared",
    
    # Fixed environment params as requested
    "sequence_length": 64,
    # num_envs is calculated dynamically based on available CPUs
    "max_steps": 20000,
    
    # Trading environment parameters
    "initial_balance": 10000.0,
    "commission": 0.0, # Updated to 0.0% Maker fee (BNB Discount)
    "reward_scaling": 1.0,
    
    # Other fixed params
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "norm_obs": "auto",
    
    # Training parameters
    "total_timesteps": 1000000,  # 1M steps total
    "eval_freq": 10000,  # Evaluate every 10k steps
    "n_eval_episodes": 3,
    "save_freq": 50000,  # Save every 50k steps
    "verbose": 1
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Ray Tune hyperparameter optimization for RecurrentPPO trading agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to training data (CSV or HDF5)"
    )
    parser.add_argument(
        "--val_data_path", type=str, required=True,
        help="Path to validation data (CSV or HDF5)"
    )
    parser.add_argument(
        "--data_key", type=str, default=None,
        help="Key for HDF5 data (if applicable)"
    )
    
    # Experiment settings
    parser.add_argument(
        "--exp_name", type=str, default="ppo_tune",
        help="Name for this experiment"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--cpus_per_trial", type=float, default=4.0,
        help="CPUs per trial"
    )
    parser.add_argument(
        "--gpus_per_trial", type=float, default=0.25,
        help="GPUs per trial (can be fractional)"
    )
    parser.add_argument(
        "--search_algo", type=str, default="optuna",
        choices=["basic", "optuna", "hyperopt"],
        help="Search algorithm to use"
    )
    
    # Ray settings
    parser.add_argument(
        "--ray_address", type=str, default=None,
        help="Address of existing Ray cluster (if any)"
    )
    parser.add_argument(
        "--local_dir", type=str, default="./ray_results",
        help="Local directory for Ray Tune results"
    )
    
    # Base config
    parser.add_argument(
        "--base_config", type=str, default=None,
        help="Path to base config file (JSON)"
    )
    
    # Training limits
    parser.add_argument(
        "--timesteps_per_trial", type=int, default=1000000,
        help="Timesteps per trial"
    )
    
    parser.add_argument("--search-algo", type=str, default="optuna", choices=["basic", "optuna", "hyperopt"], help="Search algorithm to use (basic random, Optuna, HyperOpt)")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address (e.g., 'auto' or 'ray://<head_node_ip>:10001')")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for trial logging (0=warning, 1=info, 2=debug)")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments per trial (overrides calculation from cpus_per_trial)")

    return parser.parse_args()

def define_search_space(model_type: str) -> Dict[str, Any]:
    """Define the hyperparameter search space for Ray Tune based on model type."""

    # --- Common Parameters ---
    search_space = {
        # Learning rates - Adjust range based on typical values for algorithms
        "learning_rate": tune.loguniform(1e-5, 1e-3), # Wider range for flexibility
        # Discount factors
        "gamma": tune.choice([0.95, 0.99, 0.995]),
        # Reward Component Weights (Focus on stability and profit)
        "portfolio_change_weight": tune.uniform(1.0, 7.0),
        "drawdown_penalty_weight": tune.uniform(0.01, 0.2),
        "idle_penalty_weight": tune.uniform(0.0, 0.05),
        "profit_bonus_weight": tune.uniform(0.5, 1.5),
        "trade_penalty_weight": tune.uniform(0.0, 0.1),
    }

    # --- PPO-Specific Parameters ---
    if 'ppo' in model_type:
        print(f"Defining PPO-specific search space for {model_type}")
        ppo_params = {
            # PPO n_steps (horizon length)
            "n_steps": tune.choice([512, 1024, 2048, 4096]), # Expanded choices
            # Entropy coefficient
            "ent_coef": tune.loguniform(1e-4, 0.1), # Slightly wider range lower end
            # Value function coefficient
            "vf_coef": tune.uniform(0.4, 1.0), # Slightly wider range
            # PPO clip range
            "clip_range": tune.choice([0.1, 0.2, 0.3]), # Re-added 0.3
            # GAE lambda
            "gae_lambda": tune.choice([0.9, 0.95, 0.98]),
            # PPO epochs per update
            "n_epochs": tune.choice([3, 5, 10]), # Added 10
            # Gradient clipping norm
            "max_grad_norm": tune.choice([0.5, 1.0, 2.0]), # Re-added 2.0
        }
        search_space.update(ppo_params)

    # --- SAC-Specific Parameters ---
    if 'sac' in model_type:
        print(f"Defining SAC-specific search space for {model_type}")
        sac_params = {
            # Replay buffer size
            "buffer_size": tune.choice([int(5e4), int(1e5), int(5e5)]), # Smaller options
            # Batch size for SAC updates
            "batch_size": tune.choice([256, 512, 1024]), # Typical SAC batches
            # SAC target smoothing coefficient
            "tau": tune.loguniform(0.001, 0.02),
            # Number of gradient steps per sample
            "gradient_steps": tune.choice([1, 2, 4]), # Allow more steps
            # Timesteps before learning starts
            "learning_starts": tune.choice([1000, 5000, 10000]), # Adjusted range
            # SAC entropy coefficient (numeric only for tuning)
            # Note: 'auto' is often default, consider fixing or tuning separately
            "ent_coef": tune.loguniform(1e-4, 0.1),
            # Add SDE tuning if desired
            # "use_sde": tune.choice([True, False]),
            # "sde_sample_freq": tune.choice([-1, 4, 8]), # Only if use_sde is True
        }
        search_space.update(sac_params)

    # --- TCN-Specific Parameters (If model uses TCN) ---
    if 'tcn' in model_type:
        print(f"Defining TCN-specific search space for {model_type}")
        tcn_params = {
            "tcn_num_layers": tune.choice([3, 4, 5]),
            "tcn_num_filters": tune.choice([32, 64, 96]),
            "tcn_kernel_size": tune.choice([2, 3]),
            "tcn_dropout": tune.uniform(0.05, 0.3),
        }
        search_space.update(tcn_params)

    # Remove parameters not relevant for the specific model type
    if 'ppo' not in model_type:
        for k in ["n_steps", "vf_coef", "clip_range", "gae_lambda", "n_epochs", "max_grad_norm"]:
            search_space.pop(k, None)
    if 'sac' not in model_type:
        for k in ["buffer_size", "batch_size", "tau", "gradient_steps", "learning_starts"]:
            search_space.pop(k, None)
    # Note: ent_coef is kept for both, but might need separate handling for SAC 'auto'

    return search_space

def short_trial_dirname_creator(trial):
    """Creates a shorter directory name using the trial ID and first few params."""
    # Example: trial_f8a3b1_lr0.001_ns2048
    try:
        parts = [f"trial_{shortuuid.uuid(name=trial.trial_id)[:6]}"]
        # Add a few key hyperparameter values for easier identification
        lr = trial.config.get("learning_rate")
        ns = trial.config.get("n_steps")
        if lr:
            parts.append(f"lr{lr:.1e}")
        if ns:
            parts.append(f"ns{ns}")
        return "_".join(parts)
    except Exception as e:
        # Fallback if config access fails
        return f"trial_{shortuuid.uuid(name=trial.trial_id)[:6]}"

def run_tune_experiment(args):
    """Run the Ray Tune experiment"""
    
    # Load base config (if provided) or use defaults
    if args.base_config and os.path.exists(args.base_config):
        with open(args.base_config, 'r') as f:
            base_config = json.load(f)
        print(f"Loaded base config from {args.base_config}")
    else:
        base_config = DEFAULT_CONFIG.copy()
        print("Using default base configuration")
    
    # Override base config with CLI arguments
    # Convert args namespace to dict for easier update
    cli_args_dict = {k: v for k, v in vars(args).items() if v is not None}
    base_config.update(cli_args_dict)
    print(f"Config after CLI overrides: {base_config}") # Debug print
    
    # Add command line args to config (ensure paths are absolute)
    if base_config.get("data_path"):
        data_path = os.path.abspath(os.path.expanduser(base_config["data_path"]))
        base_config["data_path"] = data_path
        print(f"Using data path: {data_path}")
    
    if base_config.get("val_data_path"):
        val_data_path = os.path.abspath(os.path.expanduser(base_config["val_data_path"]))
        base_config["val_data_path"] = val_data_path
        print(f"Using validation data path: {val_data_path}")
        
    if base_config.get("data_key"):
        base_config["data_key"] = base_config["data_key"]
    
    # Set timesteps per trial
    if args.timesteps_per_trial is not None:
        base_config["total_timesteps"] = args.timesteps_per_trial

    # Add defaults if not present
    base_config.setdefault("cpu_only", False)
    base_config.setdefault("verbose", 1) # Set default verbose from args or 1
    base_config["verbose"] = args.verbose # Ensure CLI verbose overrides

    # Determine num_envs based on CLI or CPU count
    if args.num_envs is not None:
        base_config["num_envs"] = args.num_envs
        print(f"INFO: Explicitly setting num_envs to {args.num_envs} from CLI.")
    else:
        cpus_per_trial_arg = getattr(args, 'cpus_per_trial', 1.0)
        num_envs_to_set = max(1, int(cpus_per_trial_arg)) # Use at least 1 env
        base_config["num_envs"] = num_envs_to_set
        print(f"INFO: Setting num_envs to {num_envs_to_set} based on cpus_per_trial={cpus_per_trial_arg}.")

    # Ensure Ray results directory exists
    ensure_dir_exists(args.local_dir)
    storage_path_abs = os.path.abspath(os.path.expanduser(args.local_dir))
    print(f"Using absolute storage path: {storage_path_abs}")

    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        print(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init(ignore_reinit_error=True)
        print("Initialized Ray locally")
    
    # <<<<< Get model_type from base_config to define search space >>>>>
    model_type = base_config.get("model_type", "recurrentppo").lower()
    search_space = define_search_space(model_type=model_type)
    # <<<<< End change >>>>>

    print("Search space defined with parameters:")
    for param, space in search_space.items():
        print(f"  {param}: {space}")
    
    # Merge base config and search space
    full_config = base_config.copy()
    for param in search_space:
        if param in full_config:
            # Don't delete, allow base config to provide fixed values
            # Only use search space if key isn't fixed in base_config
            # This logic is tricky, let's just overwrite base_config with search_space
            pass # Keep base config value if it exists
        # Add search space param if not in base_config
        elif param not in full_config:
             full_config[param] = search_space[param] # Add if missing

    # Construct the final tune_config by overlaying search space onto full_config
    # This ensures fixed params from base_config remain, search params are added/overwritten
    tune_config = full_config.copy()
    tune_config.update(search_space)

    print("\nFinal Tune Configuration (Fixed + Search Space):")
    for k, v in sorted(tune_config.items()):
        print(f"  {k}: {v}")
    print("-" * 30)

    # Configure search algorithm
    if args.search_algo == "optuna":
        search_alg = OptunaSearch(
            metric="eval/combined_score",
            mode="max"
        )
        print("Using Optuna search algorithm optimizing for combined metric (normalized reward and explained variance)")
    elif args.search_algo == "hyperopt":
        search_alg = HyperOptSearch(
            metric="eval/combined_score",  # Use the combined metric
            mode="max"
        )
        print("Using HyperOpt search algorithm optimizing for combined metric")
    else:  # "basic"
        search_alg = None
        print("Using basic random search algorithm")
    
    # Configure scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="timesteps",
        metric="eval/combined_score",  # Primary metric for early stopping
        mode="max",
        max_t=args.timesteps_per_trial,
        grace_period=100000,  # Min steps before stopping a trial
        reduction_factor=2
    )
    print(f"Using ASHA scheduler with {args.timesteps_per_trial} max timesteps")
    
    # --- Configure Reporter --- #
    # Define the metrics we want to see in the table and their short names
    metric_columns = {
        "training_iteration": "iter",
        "timesteps_total": "steps",
        "episode_reward_mean": "ep_rew_mean", # From Monitor
        "episode_len_mean": "ep_len_mean",   # From Monitor
        "eval/mean_reward": "eval_reward",   # From Callback
        "eval/combined_score": "score",       # From Callback
        "eval/explained_variance": "expl_var", # From Callback
        "eval/sharpe_ratio": "sharpe",
        "eval/sortino_ratio": "sortino",
        "eval/calmar_ratio": "calmar",
        "eval/mean_return_pct": "return_%", # In percent
        "eval/max_drawdown_pct": "drawdown_%", # In percent
        # "eval/total_trades": "trades",
        "time_total_s": "time(s)",
        "time/fps": "fps" # Make sure FPS is prominently displayed
    }
    
    # Define the hyperparameters to display
    # Get keys from the defined search space for consistency
    parameter_columns_config = list(search_space.keys())
    # Limit the number shown if it gets too wide
    if len(parameter_columns_config) > 6:
        parameter_columns_config = parameter_columns_config[:6] 
        print(f"Warning: Limiting displayed hyperparams to first 6: {parameter_columns_config}")

    reporter = CLIReporter(
        # Pass the configured metric columns
        metric_columns=metric_columns,
        # Pass the configured parameter columns
        parameter_columns=parameter_columns_config,
        # Sort by the primary metric we are optimizing
        sort_by_metric=True,
        metric="eval/combined_score", # Metric used for sorting
        mode="max", # Sort mode
        # Infer and report the best metric value found so far after 2 trials report
        infer_limit=2,
        # Optional: Customize max report frequency
        # max_report_frequency=60
    )
    print("Configured CLIReporter for console output")
    # ------------------------ #

    # Create and run the tuner - Updated for Ray 2.5.1
    analysis = tune.run(
        train_rl_agent_tune,
        config=tune_config,
        resources_per_trial={
            "cpu": args.cpus_per_trial,
            "gpu": args.gpus_per_trial
        },
        num_samples=args.num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=storage_path_abs,  # Use the absolute path
        name=args.exp_name,
        checkpoint_config=CheckpointConfig(num_to_keep=2), # Use new API
        progress_reporter=reporter,
        verbose=1,
        trial_dirname_creator=short_trial_dirname_creator,  # Add this line
    )
    
    print(f"Tuning completed! Analyzed {analysis.trials} trials")
    
    # Get and print best result
    best_trial = analysis.get_best_trial(metric="eval/combined_score", mode="max")
    print("\n==== Best Trial Results ====")
    print(f"Trial ID: {best_trial.trial_id}")
    
    # Show both optimization metrics in the output
    best_combined_score = best_trial.last_result.get('eval/combined_score', 'N/A')
    best_reward = best_trial.last_result.get('eval/mean_reward', 'N/A')
    best_explained_variance = best_trial.last_result.get('eval/explained_variance', 'N/A')
    best_sharpe = best_trial.last_result.get('eval/sharpe_ratio', 'N/A')
    best_sortino = best_trial.last_result.get('eval/sortino_ratio', 'N/A')
    best_calmar = best_trial.last_result.get('eval/calmar_ratio', 'N/A')
    best_return_pct = best_trial.last_result.get('eval/mean_return_pct', 'N/A')

    print(f"Best Combined Score: {best_combined_score}")
    print(f"Best Mean Eval Reward: {best_reward}")
    print(f"Best Sharpe Ratio: {best_sharpe}")
    print(f"Best Sortino Ratio: {best_sortino}")
    print(f"Best Calmar Ratio: {best_calmar}")
    print(f"Best Mean Return (%): {best_return_pct}")
    print(f"Best Explained Variance: {best_explained_variance}")
    
    print(f"Final Timesteps: {best_trial.last_result.get('timesteps_total', 'N/A')}")
    print("\nBest Hyperparameters:")
    for param_name in search_space.keys():
        print(f"  {param_name}: {best_trial.config[param_name]}")
    
    # Save best config to file
    best_config_path = os.path.join(args.local_dir, args.exp_name, "best_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_trial.config, f, indent=2)
    print(f"\nBest config saved to: {best_config_path}")
    
    print("\nTo use the best configuration, run training script with:")
    print(f"python rl_agent/train.py --load_config {best_config_path} --data_path <path> [other options]")
    
    return best_trial

if __name__ == "__main__":
    if not RAY_AVAILABLE:
        print("ERROR: Ray Tune not available. Install with 'pip install ray[tune] hyperopt optuna'")
        sys.exit(1)
    
    args = parse_args()
    best_trial = run_tune_experiment(args) 