#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convenience script to run a stability-focused hyperparameter sweep
for the RecurrentPPO agent using Ray Tune.
"""

import argparse
import os
import sys
import traceback

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set the environment variable to disable strict metric checking
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

# Explicitly check and import Ray packages with better error reporting
RAY_AVAILABLE = False
try:
    import ray
    print(f"Ray version: {ray.__version__}")
    try:
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.search.hyperopt import HyperOptSearch
        RAY_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        print(f"ERROR importing Ray Tune modules: {e}")
        traceback.print_exc()
        print("\nRay was imported but Ray Tune could not be imported.")
        print("This might be due to a version mismatch or an incomplete Ray installation.")
except ImportError as e:
    print(f"ERROR importing Ray base package: {e}")
    print("Ray base package could not be imported.")

# Only import if Ray is available
if RAY_AVAILABLE:
    from rl_agent.tune.run_tune_sweep import run_tune_experiment
else:
    print("ERROR: Ray Tune not available. Please ensure you have the correct versions installed:")
    print("pip install 'ray[tune]>=2.0.0' hyperopt>=0.2.7 optuna>=3.0.0")
    sys.exit(1)

def parse_args():
    """Parse command line arguments with preset values for stability sweep"""
    parser = argparse.ArgumentParser(
        description="Run a stability-focused hyperparameter sweep for RecurrentPPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
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
    
    # Preset arguments with reasonable defaults for stability sweep
    parser.add_argument(
        "--exp_name", type=str, default="ppo_stability_sweep",
        help="Name for this experiment"
    )
    parser.add_argument(
        "--num_samples", type=int, default=15,
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
    parser.add_argument(
        "--local_dir", type=str, default="./ray_results",
        help="Local directory for Ray Tune results"
    )
    parser.add_argument(
        "--timesteps_per_trial", type=int, default=500000,
        help="Timesteps per trial (reduced for faster iteration)"
    )
    
    args = parser.parse_args()
    
    # Create a class to match the expected arguments structure
    class ArgNamespace:
        pass
    
    # Convert args to the expected structure for run_tune_experiment
    arg_obj = ArgNamespace()
    for key, value in vars(args).items():
        setattr(arg_obj, key, value)
    
    # Add ray_address (None) for compatibility
    setattr(arg_obj, "ray_address", None)
    
    # Add base_config (None) for compatibility
    setattr(arg_obj, "base_config", None)
    
    return arg_obj

if __name__ == "__main__":
    args = parse_args()
    print("Starting stability-focused hyperparameter sweep...")
    print(f"Training data: {args.data_path}")
    print(f"Validation data: {args.val_data_path}")
    print(f"Running {args.num_samples} trials with {args.timesteps_per_trial} steps each")
    print("Note: TUNE_DISABLE_STRICT_METRIC_CHECKING=1 has been set to prevent metric validation errors")
    
    if args.search_algo == "optuna":
        print("Using Optuna search algorithm optimizing for mean reward")
    
    best_trial = run_tune_experiment(args)
    
    # Print a summary of the best found configuration
    print("\n=====================================================")
    print("STABILITY SWEEP RESULTS")
    print("=====================================================")
    print(f"Best Mean Reward: {best_trial.metrics['eval/mean_reward']:.4f}")
    print(f"Best Explained Variance: {best_trial.metrics.get('eval/explained_variance', 'N/A')}")
    
    # Print the stability parameter values
    print("\nBest Stability Parameters:")
    print(f"  learning_rate: {best_trial.config['learning_rate']:.6f}")
    print(f"  gamma: {best_trial.config['gamma']:.4f}")
    print(f"  n_steps: {best_trial.config['n_steps']}")
    
    # Calculate the equivalent batch_size
    n_steps = best_trial.config['n_steps']
    num_envs = best_trial.config.get('num_envs', 8)
    batch_size = best_trial.config.get('batch_size', min(n_steps * num_envs, 2048))
    
    print(f"  batch_size: {batch_size} (calculated from n_steps={n_steps} and num_envs={num_envs})")
    
    # Generate a command to use this configuration
    best_config_path = os.path.join(args.local_dir, args.exp_name, "best_config.json")
    print("\nTo train with these optimal parameters:")
    print(f"python rl_agent/train.py --load_config {best_config_path} \\\n  --data_path {args.data_path} \\\n  --val_data_path {args.val_data_path}")
    
    print("\n=====================================================") 