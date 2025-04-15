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
    "features": "open_scaled,high_scaled,low_scaled,close_scaled,volume_scaled",
    
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
    "sequence_length": 15,
    # num_envs is calculated dynamically based on available CPUs
    "max_steps": 20000,
    
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
    search_space = {
        # Learning rates to try (log scale from 1e-5 to 1e-3)
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        
        # Discount factors to try
        "gamma": tune.choice([0.9, 0.95, 0.99, 0.995, 0.999]),
        
        # PPO n_steps (horizon length) values to try
        "n_steps": tune.choice([512, 1024, 2048, 4096, 8192]),
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
    base_config["data_path"] = args.data_path
    base_config["val_data_path"] = args.val_data_path
    if args.data_key:
        base_config["data_key"] = args.data_key
    
    # Set timesteps per trial
    base_config["total_timesteps"] = args.timesteps_per_trial
    
    # Calculate appropriate num_envs based on CPUs per trial
    # RecurrentPPO is limited by available CPU cores
    cpus_available = args.cpus_per_trial
    # Reserve 1 CPU for the main process, use the rest for environments
    # At least 1 environment, max 8 environments
    num_envs = max(1, min(8, int(cpus_available - 1)))
    base_config["num_envs"] = num_envs
    print(f"Setting num_envs={num_envs} based on cpus_per_trial={cpus_available}")
    
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
            metric="eval/mean_reward",
            mode="max"
        )
        print("Using Optuna search algorithm")
    elif args.search_algo == "hyperopt":
        search_alg = HyperOptSearch(
            metric="eval/mean_reward",
            mode="max"
        )
        print("Using HyperOpt search algorithm")
    else:  # "basic"
        search_alg = None
        print("Using basic random search algorithm")
    
    # Configure scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="timesteps",
        metric="eval/mean_reward",
        mode="max",
        max_t=args.timesteps_per_trial,
        grace_period=100000,  # Min steps before stopping a trial
        reduction_factor=2
    )
    print(f"Using ASHA scheduler with {args.timesteps_per_trial} max timesteps")
    
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
        local_dir=args.local_dir,
        name=args.exp_name,
        keep_checkpoints_num=2,
        verbose=1,
    )
    
    print(f"Tuning completed! Analyzed {analysis.trials} trials")
    
    # Get and print best result
    best_trial = analysis.best_trial
    print("\n==== Best Trial Results ====")
    print(f"Trial ID: {best_trial.trial_id}")
    print(f"Mean Reward: {best_trial.last_result['eval/mean_reward']}")
    if 'eval/explained_variance' in best_trial.last_result:
        print(f"Explained Variance: {best_trial.last_result['eval/explained_variance']}")
    else:
        print("Explained Variance: N/A")
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