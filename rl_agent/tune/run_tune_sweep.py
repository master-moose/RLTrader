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

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables for Ray
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
# Set default cache location for Ray Air
os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = os.path.abspath("./ray_results")

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
    
    # Features - keeping OHLCV features fixed as requested
    "features": "open_scaled,high_scaled,low_scaled,close_scaled,volume_scaled,rsi_14_scaled,macd_hist_scaled,ema_9_scaled,ema_21_scaled",
    
    # Reward structure - simplified as requested
    "portfolio_change_weight": 1.0,
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
    "n_lstm_layers": 1,
    "shared_lstm": "shared",
    
    # Fixed environment params as requested
    "sequence_length": 60,
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
    
    return parser.parse_args()

def define_search_space() -> Dict[str, Any]:
    """Define the hyperparameter search space for Ray Tune"""
    
    # Focus on stability parameters as requested in requirements
    # Refined based on previous sweep results (50/50 terminated, best reward -24k)
    # Further refined based on long run showing extreme drawdowns
    search_space = {
        # Learning rates to try (log scale)
        # Narrowed range around previous best (1.7e-4)
        "learning_rate": tune.loguniform(5e-5, 5e-4),

        # Discount factors to try - removed extremes
        "gamma": tune.choice([0.95, 0.99, 0.995]),

        # PPO n_steps (horizon length) - re-explore slightly larger
        "n_steps": tune.choice([1024, 2048, 4096]),

        # --- Additional RecurrentPPO Parameters ---
        # Entropy coefficient - Widen range slightly higher
        "ent_coef": tune.loguniform(5e-3, 0.05),
        # Value function coefficient - Give slightly more weight
        "vf_coef": tune.uniform(0.5, 1.0),
        # PPO clip range - removed 0.3 as potentially too high
        "clip_range": tune.choice([0.1, 0.2]),
        # GAE lambda - removed 0.99
        "gae_lambda": tune.choice([0.9, 0.95, 0.98]),
        # PPO epochs per update - Reduced range to try improving stability
        "n_epochs": tune.choice([3, 5, 8]),
        # Gradient clipping norm - removed 2.0
        "max_grad_norm": tune.choice([0.5, 1.0]),

        # --- Reward Component Weights (Focus on stability and profit) --- #
        # INCREASED PENALTY RANGES SIGNIFICANTLY
        "drawdown_penalty_weight": tune.uniform(0.5, 2.0), # Reduced upper bound
        "fee_penalty_weight": tune.uniform(0.5, 2.0),       # Reduced upper bound
        "idle_penalty_weight": 0.0, # Keep idle penalty at 0
        "profit_bonus_weight": tune.uniform(0.75, 2.0), # Shifted range slightly higher
        "trade_penalty_weight": tune.uniform(0.1, 1.0),  # Increased range
    }
    
    return search_space

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
    
    # Add command line args to config
    # Make sure paths are absolute for Ray workers
    if args.data_path:
        data_path = os.path.abspath(os.path.expanduser(args.data_path))
        base_config["data_path"] = data_path
        print(f"Using data path: {data_path}")
    
    if args.val_data_path:
        val_data_path = os.path.abspath(os.path.expanduser(args.val_data_path))
        base_config["val_data_path"] = val_data_path
        print(f"Using validation data path: {val_data_path}")
        
    if args.data_key:
        base_config["data_key"] = args.data_key
    
    # Set timesteps per trial
    base_config["total_timesteps"] = args.timesteps_per_trial

    # --- Add default for cpu_only if not present ---
    base_config.setdefault("cpu_only", False)
    # --------------------------------------------------

    # --- Add verbose setting from args (if available) ---
    # Default to 1 (INFO) if not provided by the calling script
    base_config["verbose"] = getattr(args, 'verbose', 1)
    print(f"Setting verbosity level for trials: {base_config['verbose']}")
    # ---------------------------------------------------

    # --- Explicitly set num_envs based on cpus_per_trial ---
    # Remove the potentially inaccurate comment from DEFAULT_CONFIG
    if "# num_envs is calculated dynamically based on available CPUs" in DEFAULT_CONFIG:
        del DEFAULT_CONFIG["# num_envs is calculated dynamically based on available CPUs"] # Or adjust comment if needed

    cpus_per_trial_arg = getattr(args, 'cpus_per_trial', 1.0) # Default to 1 if not provided
    num_envs_to_set = max(1, int(cpus_per_trial_arg))
    base_config["num_envs"] = num_envs_to_set
    print(f"INFO: Explicitly setting num_envs to {num_envs_to_set} based on --cpus_per_trial={cpus_per_trial_arg}")
    # -------------------------------------------------------

    # Ensure Ray results directory exists
    ensure_dir_exists(args.local_dir)
    
    # Initialize Ray
    if args.ray_address:
        ray.init(address=args.ray_address)
        print(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init()
        print("Initialized Ray locally")
    
    # Define the search space
    search_space = define_search_space()
    print("Search space defined with parameters:")
    for param, space in search_space.items():
        print(f"  {param}: {space}")
    
    # Merge base config and search space
    full_config = base_config.copy()
    for param in search_space:
        if param in full_config:
            del full_config[param]  # Remove from base config if in search space
    
    tune_config = {**full_config, **search_space}
    
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
    reporter = CLIReporter(
        metric_columns={
            "training_iteration": "iter",
            "timesteps_total": "steps",
            "eval/mean_reward": "reward",
            "eval/explained_variance": "variance",
            "eval/combined_score": "combined",
            "time_total_s": "time(s)"
        },
        parameter_columns=list(search_space.keys()), # Show tuned hyperparameters
        sort_by_metric=True,
        metric="eval/combined_score",
        mode="max"
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
        storage_path=args.local_dir,
        name=args.exp_name,
        keep_checkpoints_num=2,
        progress_reporter=reporter,
        verbose=1,
    )
    
    print(f"Tuning completed! Analyzed {analysis.trials} trials")
    
    # Get and print best result
    best_trial = analysis.get_best_trial(metric="eval/combined_score", mode="max")
    print("\n==== Best Trial Results ====")
    print(f"Trial ID: {best_trial.trial_id}")
    
    # Show both optimization metrics in the output
    best_reward = best_trial.last_result.get('eval/mean_reward', 'N/A')
    best_explained_variance = best_trial.last_result.get('eval/explained_variance', 'N/A')
    print(f"Best Mean Reward: {best_reward}")
    print(f"Best Explained Variance: {best_explained_variance}")
    
    print(f"Final Timesteps: {best_trial.last_result['timesteps']}")
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