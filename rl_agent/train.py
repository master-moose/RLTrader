#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training script for the LSTM-DQN agent.

This script provides a command-line interface for training and evaluating 
the LSTM-DQN reinforcement learning agent on financial time series data.
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

# Add parent directory to path *before* attempting local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SB3 models
from stable_baselines3 import DQN, PPO, A2C, SAC  # Add A2C, PPO, SAC
# Import SB3 Contrib models
from sb3_contrib import QRDQN, RecurrentPPO
# Import SB3 VecEnv utils
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
# Import Buffers
from stable_baselines3.common.buffers import ReplayBuffer #, PrioritizedReplayBuffer
# Import Policies
from stable_baselines3.dqn.policies import MlpPolicy as DqnMlpPolicy #, CnnPolicy as DqnCnnPolicy
from stable_baselines3.common.policies import ActorCriticPolicy  # For PPO/A2C
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy  # For SAC
# Import Base class and Monitor
from stable_baselines3.common.base_class import BaseAlgorithm as BaseRLModel
from stable_baselines3.common.monitor import Monitor
# Import specific recurrent policy if needed (ensure sb3_contrib is installed)
# from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy # Example for RecurrentPPO
# Note: RecurrentPPO often uses strings like "MlpLstmPolicy" directly

# Import local modules (now possible after sys.path modification)
from rl_agent.callbacks import get_callback_list # Use this instead of individual imports below
# Remove unused specific callback imports
# from rl_agent.callback import TensorboardCallback, CheckpointCallback, ProgressBarManager
from rl_agent.utils import (
    setup_logger,
    setup_sb3_logger,
    check_resources,
    save_config,
    load_config,
    create_evaluation_plots,
    calculate_trading_metrics,
    ensure_dir_exists,
    set_seeds
)
from rl_agent.environment import TradingEnvironment
from rl_agent.models import LSTMFeatureExtractor
from rl_agent.data.data_loader import DataLoader

# Define technical indicators - ensure these match the environment needs
# INDICATORS = [
#     'macd', 'rsi', 'cci', 'dx', 'bb_upper', 'bb_lower', 'bb_middle', 'volume'
# ]

# Initialize logger globally (will be configured in main/train/evaluate)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments, incorporating args from dqn_old.py."""
    parser = argparse.ArgumentParser(
        description="Train RL agents (DQN, PPO, A2C, SAC) for trading"
    )

    # --- Model Selection --- #
    parser.add_argument(
        "--model_type", type=str, default="dqn",
        choices=[
            "dqn", "ppo", "a2c", "sac", "lstm_dqn", "qrdqn", "recurrentppo"
        ],
        help="RL algorithm (default: dqn). Use lstm_dqn for DQN w/ LSTM feats."
    )
    parser.add_argument(
        "--load_model", type=str, default=None,
        help="Path to saved model to continue training from"
    )
    parser.add_argument(
        "--lstm_model_path", type=str, default=None,
        help="Path to saved LSTM model state_dict for feature extraction"
    )

    # --- Data Parameters --- #
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to data file (CSV or HDF5)"
    )
    parser.add_argument(
        "--val_data_path", type=str, default=None,
        help="Path to validation data file (optional)"
    )
    parser.add_argument(
        "--test_data_path", type=str, default=None,
        help="Path to test data file (optional)"
    )
    parser.add_argument(
        "--data_key", type=str, default=None,
        help="Key for HDF5 file (e.g., '/15m')"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTC/USDT",
        help="Trading symbol (used if env needs it, default: BTC/USDT)"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=60,
        help="Length of history sequence for features/LSTM (default: 60)"
    )
    parser.add_argument(
        "--features", type=str,
        default="close,volume,open,high,low,rsi_14,macd,ema_9,ema_21",
        help="Comma-separated list of features/indicators to use"
    )

    # --- Environment Parameters --- #
    parser.add_argument(
        "--initial_balance", type=float, default=10000,
        help="Initial balance for trading environment (default: 10000)"
    )
    parser.add_argument(
        "--commission", type=float, default=0.001,
        help="Trading commission percentage (default: 0.001 = 0.1%%)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=20000,
        help="Maximum steps per episode (default: 20000)"
    )
    parser.add_argument(
        "--episode_length", type=int, default=None,
        help="Length of each episode in days (overrides max_steps if set)"
    )
    parser.add_argument(
        "--reward_scaling", type=float, default=1.0,
        help="Scaling factor for environment rewards (default: 1.0)"
    )
    parser.add_argument(
        "--max_holding_steps", type=int, default=8,
        help="Max steps to hold before potential forced action (default: 8)"
    )
    parser.add_argument(
        "--take_profit_pct", type=float, default=0.03,
        help="Take profit percentage (default: 0.03)"
    )
    parser.add_argument(
        "--target_cash_ratio", type=str, default="0.3-0.7",
        help="Target cash ratio range for reward shaping (default: '0.3-0.7')"
    )
    
    # --- Reward Component Weights --- #
    parser.add_argument(
        "--portfolio_change_weight", type=float, default=1.0,
        help="Weight for portfolio value change reward (default: 1.0)"
    )
    parser.add_argument(
        "--drawdown_penalty_weight", type=float, default=0.5,
        help="Weight for drawdown penalty (default: 0.5)"
    )
    parser.add_argument(
        "--sharpe_reward_weight", type=float, default=0.5,
        help="Weight for Sharpe ratio reward (default: 0.5)"
    )
    parser.add_argument(
        "--fee_penalty_weight", type=float, default=2.0,
        help="Weight for transaction fee penalty (default: 2.0)"
    )
    parser.add_argument(
        "--benchmark_reward_weight", type=float, default=0.5,
        help="Weight for benchmark comparison reward (default: 0.5)"
    )
    parser.add_argument(
        "--consistency_penalty_weight", type=float, default=0.2,
        help="Weight for trade consistency penalty (default: 0.2)"
    )
    parser.add_argument(
        "--idle_penalty_weight", type=float, default=0.1,
        help="Weight for idle position penalty (default: 0.1)"
    )
    parser.add_argument(
        "--profit_bonus_weight", type=float, default=0.5,
        help="Weight for profit bonus (default: 0.5)"
    )
    
    # --- Additional Reward Parameters --- #
    parser.add_argument(
        "--sharpe_window", type=int, default=20,
        help="Window size for Sharpe ratio calculation (default: 20)"
    )
    parser.add_argument(
        "--consistency_threshold", type=int, default=3,
        help="Minimum consecutive actions before flip is acceptable (default: 3)"
    )
    parser.add_argument(
        "--idle_threshold", type=int, default=5,
        help="Number of consecutive holds before applying idle penalty (default: 5)"
    )

    # --- Training Parameters --- #
    parser.add_argument(
        "--total_timesteps", type=int, default=1000000,
        help="Total number of training timesteps (default: 1,000,000)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0003,
        help="Learning rate for the optimizer (default: 0.0003)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048,
        help="Batch size (default: 2048 for PPO/A2C/SAC, smaller for DQN)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor for future rewards (default: 0.99)"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=10000,
        help="Evaluation frequency during training (default: 10000)"
    )
    parser.add_argument(
        "--n_eval_episodes", type=int, default=5,
        help="Number of episodes for evaluation (default: 5)"
    )
    parser.add_argument(
        "--save_freq", type=int, default=50000,
        help="Model saving frequency during training (default: 50000)"
    )
    parser.add_argument(
        "--keep_checkpoints", type=int, default=3,
        help="Number of model checkpoints to keep (default: 3)"
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=10,
        help="Patience for early stopping (0=disable, default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of parallel environments for training (default: 1)"
    )

    # --- PPO/A2C Specific --- #
    parser.add_argument(
        "--n_steps", type=int, default=2048,
        help="Steps per update for PPO/A2C (default: 2048)"
    )
    parser.add_argument(
        "--ent_coef", type=str, default="0.01",
        help="Entropy coef. (PPO/A2C/SAC, default: 0.01 PPO/A2C, 'auto' SAC)"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5,
        help="Value function coefficient for PPO/A2C (default: 0.5)"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10,
        help="Number of epochs per update for PPO (default: 10)"
    )
    parser.add_argument(
        "--clip_range", type=float, default=0.2,
        help="PPO clip range (default: 0.2)"
    )

    # --- DQN/SAC/QRDQN Specific --- #  (Consolidated section)
    parser.add_argument(
        "--buffer_size", type=int, default=100000,
        help="Replay buffer size (DQN/SAC/QRDQN, default: 100000)"
    )
    parser.add_argument(
        "--exploration_fraction", type=float, default=0.1,
        help="Exploration decay fraction (DQN/QRDQN, default: 0.1)"
    )
    parser.add_argument(
        "--exploration_initial_eps", type=float, default=1.0,
        help="Initial exploration rate epsilon (DQN/QRDQN, default: 1.0)"
    )
    parser.add_argument(
        "--exploration_final_eps", type=float, default=0.05,
        help="Final exploration rate epsilon (DQN/QRDQN, default: 0.05)"
    )
    parser.add_argument(
        "--target_update_interval", type=int, default=10000,
        help="Target network update freq (DQN/SAC/QRDQN, default: 10000)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Soft update coefficient tau (SAC, default: 0.005)"
    )
    parser.add_argument(
        "--gradient_steps", type=int, default=1,
        help="Gradient steps per update (DQN/SAC/QRDQN, default: 1)"
    )
    parser.add_argument(
        "--learning_starts", type=int, default=1000,
        help="Steps before learning starts (DQN/SAC/QRDQN, default: 1000)"
    )

    # --- LSTM Specific --- #
    parser.add_argument(
        "--lstm_hidden_size", type=int, default=128,
        help="Hidden size of LSTM layer (if using LSTM features, default: 128)"
    )
    parser.add_argument(
        "--fc_hidden_size", type=int, default=64,
        help="Hidden size of FC layers after LSTM/features (default: 64)"
    )

    # --- Resource Management & Logging --- #
    parser.add_argument(
        "--resource_check_freq", type=int, default=5000,
        help="Frequency of resource usage checks (default: 5000)"
    )
    parser.add_argument(
        "--metrics_log_freq", type=int, default=1000,
        help="Frequency of trading metrics logging (default: 1000)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="Directory for saving logs (default: ./logs)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Directory for saving model checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Name for model/log folder (default: auto-generated)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level (0: no output, 1: info, 2: debug)"
    )
    parser.add_argument(
        "--load_config", type=str, default=None,
        help="Path to config file (overrides defaults, overridden by CLI)"
    )
    parser.add_argument(
        "--cpu_only", action="store_true",
        help="Force using CPU even if GPU is available"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Only evaluate a trained model (--load_model required)"
    )

    # --- New DQN PER arguments --- (Commented out for SB3 v2.6.0 compatibility)
    # parser.add_argument(
    #     "--prioritized_replay", action="store_true",
    #     help="Enable Prioritized Experience Replay (PER) for DQN"
    # )
    # parser.add_argument(
    #     "--prioritized_replay_alpha", type=float, default=0.6,
    #     help="Alpha parameter for PER (default: 0.6)"
    # )
    # parser.add_argument(
    #     "--prioritized_replay_beta0", type=float, default=0.4,
    #     help="Initial beta for PER importance sampling (default: 0.4)"
    # )
    # parser.add_argument(
    #     "--prioritized_replay_eps", type=float, default=1e-6,
    #     help="Epsilon for PER to avoid zero priority (default: 1e-6)"
    # )
    # --- End New DQN PER arguments ---

    # Add new exploration bonus arguments for TradingEnvironment
    parser.add_argument(
        "--exploration_start", type=float, default=1.0,
        help="Starting value for exploration bonus in TradingEnvironment (default: 1.0)"
    )
    parser.add_argument(
        "--exploration_end", type=float, default=0.01,
        help="Final value for exploration bonus in TradingEnvironment (default: 0.01)"
    )
    parser.add_argument(
        "--exploration_decay_rate", type=float, default=0.0001,
        help="Decay rate for exploration bonus per step (default: 0.0001)"
    )
    parser.add_argument(
        "--exploration_bonus_weight", type=float, default=0.1,
        help="Weight for exploration bonus in reward calculation (default: 0.1)"
    )

    # QRDQN specific parameters
    parser.add_argument(
        "--n_quantiles", type=int, default=200,
        help="Number of quantiles for QRDQN (default: 200)"
    )

    args = parser.parse_args()

    # Auto-generate model name if not provided
    if args.model_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.model_name = f"{args.model_type}_{timestamp}"

    # Process features string into list
    if isinstance(args.features, str):
        args.features = args.features.split(",")

    return args


def args_to_config(args) -> Dict[str, Any]:
    """Convert argparse arguments to config dictionary."""
    return vars(args)


# Update create_env to use config dict and potentially handle HDF5
def create_env(
    config: Dict[str, Any],  # Use config dict
    data_override: Optional[pd.DataFrame] = None,
    is_eval: bool = False
) -> gym.Env:
    """
    Create a trading environment based on configuration.

    Args:
        config: Configuration dictionary containing env parameters.
        data_override: Optional DataFrame to use instead of loading from path.
        is_eval: Flag indicating if this is for evaluation (affects seeding).

    Returns:
        Trading environment instance.
    """
    # Load data if not provided
    if data_override is not None:
        data = data_override
    else:
        data_path = config["data_path"]
        data_key = config.get("data_key")  # Use .get for optional key
        data_loader = DataLoader(data_path=data_path, data_key=data_key)
        data = data_loader.load_data()

    # Determine seed for this environment instance
    # base_seed = config.get("seed")
    # env_seed = None
    # if base_seed is not None:
    #     # Use different seeds for train vs eval vs parallel envs
    #     if is_eval:
    #         env_seed = base_seed + 1  # Simple offset for eval
    #     else:
    #         env_seed = base_seed  # Use base seed for main training env
    # Note: Seeding is handled by make_vec_env wrapper now

    # Create environment instance
    # Start with core arguments
    env_kwargs = {
        "data": data,
        "features": config.get("features"),
        "sequence_length": config["sequence_length"],
        "initial_balance": config["initial_balance"],
        "transaction_fee": config["commission"],
        "reward_scaling": config["reward_scaling"],
        # Optional args from TradingEnvironment (using defaults or config)
        "window_size": config.get("window_size", 20),
        "max_position": config.get("max_position", 1.0),  # <-- Make sure this line exists
        "max_steps": config.get("max_steps"),
        "random_start": config.get("random_start", True), # <-- Make sure this line exists
    }

    # --- Conditionally add new reward parameters --- 
    # Define the keys for the new reward parameters
    reward_param_keys = [
        "portfolio_change_weight", "drawdown_penalty_weight",
        "sharpe_reward_weight", "fee_penalty_weight",
        "benchmark_reward_weight", "consistency_penalty_weight",
        "idle_penalty_weight", "profit_bonus_weight",
        "exploration_bonus_weight", "sharpe_window",
        "consistency_threshold", "idle_threshold",
        "exploration_start", "exploration_end",
        "exploration_decay_rate"
    ]
    
    # Add them to env_kwargs ONLY if they exist in the config dict
    for key in reward_param_keys:
        if key in config:
            env_kwargs[key] = config[key]
            logger.debug(f"Passing reward parameter '{key}' = {config[key]} to environment.")
        # else: # Optional: Log if a parameter is *not* found in config
        #     logger.debug(f"Reward parameter '{key}' not found in config, using env default.")
    # ---------------------------------------------

    # Filter out None values from core arguments if necessary
    # (Though None is valid for max_steps, features, etc.)
    # env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
    # Removing the above filter as None might be intended for some args.

    env = TradingEnvironment(**env_kwargs)

    # Apply wrappers (e.g., SafeTradingEnvWrapper) if needed
    # env = SafeTradingEnvWrapper(env, ...)
    # env = TimeLimit(env, max_episode_steps=config["max_steps"])

    return env


# Update create_model to handle more algorithms and LSTM features
def create_model(
    env: gym.Env,  # Can be a VecEnv
    config: Dict[str, Any],
) -> BaseRLModel:
    """
    Create a reinforcement learning model based on configuration.

    Args:
        env: Training environment (potentially vectorized).
        config: Configuration dictionary.

    Returns:
        Reinforcement learning model instance.
    """
    model_type = config["model_type"].lower()
    learning_rate = config["learning_rate"]
    seed = config.get("seed")

    # Determine device setting
    device = "cpu" if config["cpu_only"] else "auto"

    policy_kwargs = {}
    model_kwargs = {
        "policy": None,  # Determined below
        "env": env,
        "learning_rate": learning_rate,
        "gamma": config["gamma"],
        "seed": seed,
        "device": device,
        "verbose": 0,  # Callbacks handle progress output
        "policy_kwargs": policy_kwargs,  # Updated below
        "tensorboard_log": os.path.join(
            config["log_dir"], config["model_name"], "sb3_logs"
        )
    }

    # --- LSTM Feature Extractor Setup --- #
    use_lstm_features = (
        model_type == "lstm_dqn" or config.get("lstm_model_path")
    )
    if use_lstm_features:
        lstm_state_dict = None
        if config.get("lstm_model_path"):
            lstm_state_dict = config["lstm_model_path"]
            logger.info(f"Using LSTM model path: {lstm_state_dict}")

        policy_kwargs["features_extractor_class"] = LSTMFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "lstm_state_dict": lstm_state_dict,
            "features_dim": config.get("lstm_hidden_size", 128)
        }
        # Note: Env patching might be needed for LSTM dims
        # comprehensive_environment_patch(env, features_dim, logger)

        # Adjust network architecture if FC layers are specified
        if "fc_hidden_size" in config and config["fc_hidden_size"] > 0:
            fc_size = config["fc_hidden_size"]
            if model_type in ["ppo", "a2c"]:
                # SB3 uses net_arch=[dict(pi=[...], vf=[...])] or shared
                policy_kwargs["net_arch"] = [fc_size, fc_size]  # Shared
            # DQN/SAC net_arch handled below in specific sections

    # --- Algorithm Specific Setup --- #
    if model_type == "dqn" or model_type == "lstm_dqn":
        model_kwargs["policy"] = DqnMlpPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config["learning_starts"]
        model_kwargs["gradient_steps"] = config["gradient_steps"]
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["exploration_fraction"] = config["exploration_fraction"]
        model_kwargs["exploration_initial_eps"] = config["exploration_initial_eps"]
        model_kwargs["exploration_final_eps"] = config["exploration_final_eps"]

        # Set default net_arch for DqnMlpPolicy if not overridden
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model_kwargs["policy_kwargs"] = policy_kwargs

        # Use default ReplayBuffer (PER commented out)
        model_kwargs["replay_buffer_class"] = ReplayBuffer
        model_kwargs["replay_buffer_kwargs"] = {}

        # Remove kwargs not accepted by DQN.__init__
        model_kwargs.pop("tensorboard_log", None)  # Uses learn's tb_log_name

        model = DQN(**model_kwargs)

    elif model_type == "ppo":
        model_kwargs["policy"] = ActorCriticPolicy
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["n_epochs"] = config["n_epochs"]
        model_kwargs["ent_coef"] = float(config["ent_coef"])
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["clip_range"] = config["clip_range"]
        model_kwargs["gae_lambda"] = config.get("gae_lambda", 0.95)
        model_kwargs["max_grad_norm"] = config.get("max_grad_norm", 0.5)
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model = PPO(**model_kwargs)

    elif model_type == "a2c":
        model_kwargs["policy"] = ActorCriticPolicy
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["ent_coef"] = float(config["ent_coef"])
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["gae_lambda"] = config.get("gae_lambda", 0.95)
        model_kwargs["max_grad_norm"] = config.get("max_grad_norm", 0.5)
        model_kwargs["rms_prop_eps"] = config.get("rms_prop_eps", 1e-5)
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model = A2C(**model_kwargs)

    elif model_type == "sac":
        model_kwargs["policy"] = SacMlpPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config["learning_starts"]
        model_kwargs["gradient_steps"] = config["gradient_steps"]
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["tau"] = config["tau"]

        # Handle ent_coef ('auto' or float)
        ent_coef_value = config.get("ent_coef", "auto")
        if isinstance(ent_coef_value, str) and ent_coef_value.lower() == 'auto':
            model_kwargs["ent_coef"] = 'auto'
        else:
            try:
                model_kwargs["ent_coef"] = float(ent_coef_value)
            except ValueError:
                logger.warning(
                    f"Invalid ent_coef '{ent_coef_value}'. Defaulting 'auto'."
                )
                model_kwargs["ent_coef"] = 'auto'

        # Set default net_arch for SacMlpPolicy if not overridden
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model = SAC(**model_kwargs)

    elif model_type == "qrdqn":
        # Import QRDQNPolicy from sb3_contrib
        from sb3_contrib.qrdqn.policies import QRDQNPolicy
        model_kwargs["policy"] = QRDQNPolicy  # Use QRDQNPolicy instead of DQNMlpPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config["learning_starts"]
        model_kwargs["gradient_steps"] = config["gradient_steps"]
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["exploration_fraction"] = config["exploration_fraction"]
        model_kwargs["exploration_initial_eps"] = config["exploration_initial_eps"]
        model_kwargs["exploration_final_eps"] = config["exploration_final_eps"]

        # Set default net_arch for QRDQNPolicy
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        
        # QRDQN specific args - set n_quantiles in policy_kwargs
        policy_kwargs["n_quantiles"] = config.get("n_quantiles", 200)  # Default to 200 quantiles
        model_kwargs["policy_kwargs"] = policy_kwargs

        # Remove incompatible args
        model_kwargs.pop("tensorboard_log", None)
        model = QRDQN(**model_kwargs)

    elif model_type == "recurrentppo":
        # Use the string identifier for the policy
        model_kwargs["policy"] = "MlpLstmPolicy"
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["n_epochs"] = config["n_epochs"]
        model_kwargs["ent_coef"] = float(config["ent_coef"])
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["clip_range"] = config["clip_range"]
        model_kwargs["gae_lambda"] = config.get("gae_lambda", 0.95)
        model_kwargs["max_grad_norm"] = config.get("max_grad_norm", 0.5)

        # Configure LSTM within the policy_kwargs
        if "lstm_hidden_size" in config:
            policy_kwargs["lstm_hidden_size"] = config["lstm_hidden_size"]
        # policy_kwargs["n_lstm_layers"] = config.get("n_lstm_layers", 1)
        # policy_kwargs["enable_critic_lstm"] = config.get("enable_critic_lstm", True)

        # MlpLstmPolicy generally handles net_arch internally based on LSTM size

        model = RecurrentPPO(**model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(
        f"Created {model_type.upper()} model with policy "
        f"{model_kwargs['policy'] if isinstance(model_kwargs['policy'], str) else model_kwargs['policy'].__name__}"
    )
    return model


# Update evaluate_model to handle vectorized environments
def evaluate_model(
    model: BaseRLModel,
    env: gym.Env,  # VecEnv expected
    config: Dict[str, Any],
    n_episodes: int = 1,
    deterministic: bool = True,
) -> Tuple[float, np.ndarray, List[int], List[float]]:
    """
    Evaluate a model over n_episodes and return metrics.

    Args:
        model: The trained model
        env: The vectorized environment to evaluate on
        config: Configuration dictionary
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions

    Returns:
        Tuple containing:
        - mean reward
        - portfolio values (flattened from all episodes)
        - actions taken (flattened)
        - rewards received (flattened)
    """
    is_vectorized = hasattr(env, 'num_envs')
    if not is_vectorized:
        # This function expects a VecEnv
        logger.warning("evaluate_model expects a VecEnv, wrapping in DummyVecEnv")
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs

    # Tracking variables
    all_episode_rewards = []
    all_portfolio_values = []
    all_actions = []
    all_rewards = []

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_portfolio_values = [[] for _ in range(n_envs)]
    current_actions = [[] for _ in range(n_envs)]
    current_rewards_list = [[] for _ in range(n_envs)]

    # Reset environment
    obs = env.reset()
    episodes_completed = 0

    # Main evaluation loop
    while episodes_completed < n_episodes:
        # Get model prediction
        action, _ = model.predict(obs, deterministic=deterministic)

        # Step environment
        obs, rewards, terminated, infos = env.step(action)
        # In new gym API, 'done' is split into 'terminated' and 'truncated'
        # SB3 VecEnv 'infos' usually contains 'TimeLimit.truncated'
        truncated = np.array([info.get('TimeLimit.truncated', False)
                              for info in infos])
        dones = np.logical_or(terminated, truncated)

        # Update tracking for each environment
        for i in range(n_envs):
            current_rewards[i] += rewards[i]
            current_lengths[i] += 1

            portfolio_value = infos[i].get("portfolio_value", 0.0)
            current_portfolio_values[i].append(portfolio_value)
            current_actions[i].append(action[i])
            current_rewards_list[i].append(rewards[i])

            # Handle episode completion
            if dones[i]:
                if episodes_completed < n_episodes:
                    all_episode_rewards.append(current_rewards[i])
                    all_portfolio_values.extend(current_portfolio_values[i])
                    all_actions.extend(current_actions[i])
                    all_rewards.extend(current_rewards_list[i])
                    episodes_completed += 1

                # Reset tracking for this environment
                current_rewards[i] = 0
                current_lengths[i] = 0
                current_portfolio_values[i] = []
                current_actions[i] = []
                current_rewards_list[i] = []

                # Important: Check if the environment needs manual reset after done
                # VecEnv handles auto-reset, but Monitor might need manual if not wrapped?
                # Generally VecEnv handles this.

    # Calculate mean reward across all completed episodes
    mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0

    return (
        mean_reward,
        np.array(all_portfolio_values),
        all_actions,
        all_rewards
    )


# Implement the missing evaluate function
def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    # Setup logger
    log_path = os.path.join(config["log_dir"], config["model_name"], "evaluation")
    ensure_dir_exists(log_path)
    setup_logger(
        log_dir=log_path,
        log_level=logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO
    )

    # Check if model path exists
    model_path = config["load_model"]
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)

    num_envs = config.get("num_envs", 1)
    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv

    # Load test data
    if not config.get("test_data_path"):
        logger.error("No test data path provided")
        sys.exit(1)

    # Create test environment with same settings as training
    logger.info(f"Creating test environment from: {config['test_data_path']}")
    test_env_config = config.copy()
    test_env_config["data_path"] = config["test_data_path"]

    base_seed_from_config = config.get("seed")

    # Function to create a single env instance
    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            env_config = test_env_config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=True)
            monitor_log_path_eval = os.path.join(log_path, f'monitor_eval_{rank}.csv')
            env = Monitor(env, filename=monitor_log_path_eval)
            return env
        return _init

    # Create VecEnv for evaluation (can use DummyVecEnv for simplicity)
    test_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=base_seed_from_config),
        n_envs=1, # Evaluate on a single env instance for clearer metrics
        seed=None,  # Seeding handled in make_single_env
        vec_env_cls=DummyVecEnv, # Use DummyVecEnv for eval
        env_kwargs=None
    )

    # --- Apply VecNormalize to test_env --- #
    vec_normalize_stats_path = None
    potential_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(potential_stats_path):
        vec_normalize_stats_path = potential_stats_path

    if vec_normalize_stats_path:
        logger.info(f"Loading VecNormalize stats: {vec_normalize_stats_path}")
        test_env = VecNormalize.load(vec_normalize_stats_path, test_env)
        test_env.training = False  # Set to inference mode
        test_env.norm_reward = False  # Do not normalize rewards for eval
    else:
        logger.warning("VecNormalize stats not found. Evaluation might be less accurate.")

    # Load the model
    model_cls = {
        "dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
        "lstm_dqn": DQN, "qrdqn": QRDQN, "recurrentppo": RecurrentPPO
    }
    model = model_cls[config["model_type"]].load(model_path, env=test_env)

    # Run evaluation
    n_eval_episodes = config.get("n_eval_episodes", 5)
    logger.info(f"Starting evaluation for {n_eval_episodes} episodes")

    mean_reward, portfolio_values, actions, rewards = evaluate_model(
        model=model,
        env=test_env,
        config=config,
        n_episodes=n_eval_episodes,
        deterministic=True
    )

    # Calculate additional metrics
    if len(portfolio_values) > 0:
        final_portfolio_value = portfolio_values[-1]
        initial_portfolio_value = config.get("initial_balance", 10000)
        # Use initial balance from config as start for total return calc
        total_return = (
            (final_portfolio_value / initial_portfolio_value) - 1
            if initial_portfolio_value > 0 else 0
        )
    else:
        final_portfolio_value = 0.0
        initial_portfolio_value = config.get("initial_balance", 10000)
        total_return = 0.0
        logger.warning("No portfolio values recorded during evaluation!")

    metrics = {
        "mean_reward": mean_reward,
        "final_portfolio_value": final_portfolio_value,
        "total_return": total_return,
    }

    if len(portfolio_values) > 10:
        try:
            trading_metrics = calculate_trading_metrics(portfolio_values)
            metrics.update(trading_metrics)
            create_evaluation_plots(
                portfolio_values=portfolio_values,
                actions=actions,
                rewards=rewards,
                save_path=log_path
            )
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
    else:
        logger.warning(
            f"Only {len(portfolio_values)} values; skipping advanced metrics."
        )

    # Log metrics
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save metrics to JSON file
    metrics_file = os.path.join(log_path, "evaluation_metrics.json")
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert numpy types for JSON serialization
            serializable_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_file}")
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")

    # Close environment
    test_env.close()

    return metrics


# Update train function
def train(config: Dict[str, Any]) -> Tuple[BaseRLModel, Dict[str, Any]]:
    """
    Train a reinforcement learning agent based on config.
    """
    # Setup logger
    log_path = os.path.join(config["log_dir"], config["model_name"])
    ensure_dir_exists(log_path)
    setup_logger(
        log_dir=log_path,
        log_level=logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO
    )

    # Setup SB3 logger
    sb3_log_path = log_path
    sb3_logger_instance = setup_sb3_logger(log_dir=sb3_log_path)

    # Save configuration
    save_config(config=config, log_dir=log_path, filename="config.json")

    # Set random seeds
    if config.get("seed") is not None:
        set_seeds(config["seed"])

    # Log system information
    logger.info(f"Starting training with model: {config['model_name']}")
    device = ("cpu" if config["cpu_only"]
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training on device: {device}")
    check_resources(logger)

    # --- Environment Creation --- #
    num_envs = config.get("num_envs", 1)
    logger.info(f"Creating {num_envs} parallel environment(s)...")

    # Function to create a single env instance
    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            env_config = config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=False)
            monitor_log_path = os.path.join(log_path, f'monitor_{rank}.csv')
            os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
            env = Monitor(env, filename=monitor_log_path)
            return env
        return _init

    # Create vectorized environment
    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    base_seed_from_config = config.get("seed")
    train_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=base_seed_from_config),
        n_envs=num_envs,
        seed=None,  # Seeding handled in make_single_env
        vec_env_cls=vec_env_cls,
        env_kwargs=None
    )

    # --- Apply VecNormalize --- #
    vec_normalize_stats_path = None
    if config.get("load_model"):
        potential_stats_path = config["load_model"].replace(
            ".zip", "_vecnormalize.pkl"
        )
        if os.path.exists(potential_stats_path):
            vec_normalize_stats_path = potential_stats_path
            logger.info(f"Found VecNormalize stats: {vec_normalize_stats_path}")

    if vec_normalize_stats_path:
        logger.info(f"Loading VecNormalize stats: {vec_normalize_stats_path}")
        train_env = VecNormalize.load(vec_normalize_stats_path, train_env)
        train_env.training = True  # Ensure it continues training
    else:
        logger.info("Applying new VecNormalize (norm_obs=True, norm_reward=False).")
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.,
            gamma=config["gamma"]
        )

    # Create validation environment if specified
    eval_env = None
    if config.get("val_data_path"):
        logger.info(f"Creating validation environment: {config['val_data_path']}")
        eval_env_config = config.copy()
        eval_env_config["data_path"] = config["val_data_path"]

        eval_env = make_vec_env(
            env_id=make_single_env(rank=0, base_seed=base_seed_from_config),
            n_envs=1, # Use single env for validation
            seed=None,
            vec_env_cls=DummyVecEnv, # Use DummyVecEnv for validation
            env_kwargs=None
        )

        # Apply VecNormalize to eval_env
        if vec_normalize_stats_path:
            logger.info("Applying loaded VecNormalize stats to eval_env.")
            eval_env = VecNormalize.load(vec_normalize_stats_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            logger.info("Applying new VecNormalize wrapper to eval_env.")
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.,
                gamma=config["gamma"],
                training=False
            )

    # --- Model Creation / Loading --- #
    if config.get("load_model"):
        load_path = config["load_model"]
        logger.info(f"Loading model from: {load_path}")
        if not os.path.exists(load_path):
            logger.error(f"Model path not found: {load_path}")
            sys.exit(1)

        model_cls = {
            "dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
            "lstm_dqn": DQN, "qrdqn": QRDQN, "recurrentppo": RecurrentPPO
        }
        model = model_cls[config["model_type"]].load(load_path, env=train_env)

        # Override loaded learning rate if specified
        if 'learning_rate' in config and config['learning_rate'] is not None:
            new_lr = config['learning_rate']
            if hasattr(model.policy, 'optimizer') and model.policy.optimizer:
                logger.info(f"Overriding loaded LR. Setting to: {new_lr}")
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                logger.warning("Could not find optimizer to override LR.")

    else:
        logger.info(f"Creating new {config['model_type']} model")
        model = create_model(env=train_env, config=config)

    # Set logger for the model
    model.set_logger(sb3_logger_instance)

    # --- Callbacks --- #
    checkpoint_save_path = os.path.join(
        config["checkpoint_dir"], config["model_name"]
    )
    ensure_dir_exists(checkpoint_save_path)
    callbacks = get_callback_list(
        eval_env=eval_env,
        log_dir=log_path,
        eval_freq=max(config["eval_freq"], 5000),
        n_eval_episodes=config["n_eval_episodes"],
        save_freq=config["save_freq"],
        keep_checkpoints=config["keep_checkpoints"],
        resource_check_freq=config["resource_check_freq"],
        metrics_log_freq=config["metrics_log_freq"],
        early_stopping_patience=max(20, config["early_stopping_patience"]),
        checkpoint_save_path=checkpoint_save_path,
        model_name=config["model_type"],
        custom_callbacks=[]
    )

    # --- Training --- #
    logger.info(f"Starting training: {config['total_timesteps']} timesteps...")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=1,
            reset_num_timesteps=not config.get("load_model")
        )
    except Exception as e:
        logger.critical(f"Training failed: {e}", exc_info=True)
        error_save_path = os.path.join(log_path, "model_on_error.zip")
        try:
            model.save(error_save_path)
            logger.info(f"Model state saved to {error_save_path}.")
        except Exception as save_e:
            logger.error(f"Could not save model after error: {save_e}")
        sys.exit(1)
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()

    training_time = time.time() - training_start_time
    logger.info(f"Training finished in {training_time:.2f} seconds.")

    # Save final model
    final_model_path = os.path.join(log_path, "final_model.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save VecNormalize stats
    if isinstance(train_env, VecNormalize):
        stats_path = final_model_path.replace(".zip", "_vecnormalize.pkl")
        train_env.save(stats_path)
        logger.info(f"VecNormalize stats saved to {stats_path}")

    metrics = {
        "training_time": training_time,
        "total_timesteps": model.num_timesteps,
        "model_type": config["model_type"],
    }

    return model, metrics


# Updated main function to handle different models and eval mode
def main():
    """Main function: parse args, setup, run train/eval."""
    args = parse_args()
    config = args_to_config(args)

    # --- Config Loading --- #
    if args.load_config is not None:
        if os.path.exists(args.load_config):
            print(f"Loading configuration from {args.load_config}")
            file_config = load_config(args.load_config)
            cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
            file_config.update(cli_overrides)
            config = file_config
            if 'features' in config and isinstance(config['features'], str):
                config['features'] = config['features'].split(",")
        else:
            print(f"Error: Config file not found at {args.load_config}")
            sys.exit(1)

    # --- Directory Setup --- #
    log_base_dir = config["log_dir"]
    model_name = config["model_name"]
    checkpoint_base_dir = config["checkpoint_dir"]
    ensure_dir_exists(log_base_dir)
    ensure_dir_exists(checkpoint_base_dir)
    ensure_dir_exists(os.path.join(log_base_dir, model_name))
    ensure_dir_exists(os.path.join(checkpoint_base_dir, model_name))

    # --- Mode Selection --- #
    # print(f"DEBUG: eval_only = {config.get('eval_only')}") # Debug print
    if config.get("eval_only", False):
        # --- Evaluation Mode --- #
        print("Running in Evaluation-Only Mode")
        if config.get("load_model") is None:
            print("Error: Must provide --load_model for evaluation mode.")
            sys.exit(1)
        if config.get("test_data_path") is None:
            print("Error: Must provide --test_data_path for evaluation mode.")
            sys.exit(1)
        evaluate(config)

    else:
        # --- Training Mode --- #
        print(f"Running in Training Mode for model: {config['model_type']}")
        model, train_metrics = train(config)

        # Evaluate on test data if provided after training
        if config.get("test_data_path") is not None:
            print("\nStarting final evaluation on test data...")
            final_model_path = os.path.join(
                log_base_dir, model_name, "final_model.zip"
            )
            eval_config = config.copy()
            eval_config["load_model"] = final_model_path
            eval_config["test_data_path"] = config["test_data_path"]

            test_metrics = evaluate(eval_config)

            print("\n--- Training Summary ---")
            print(f"Training time: {train_metrics['training_time']:.2f} sec")
            print(f"Total steps: {train_metrics['total_timesteps']}")
            print(f"Final model: {final_model_path}")
            print("\n--- Test Set Evaluation Results ---")
            for key, value in test_metrics.items():
                 print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        else:
            final_model_path = os.path.join(
                log_base_dir, model_name, "final_model.zip"
            )
            print("\n--- Training Completed --- ")
            print(f"Training time: {train_metrics['training_time']:.2f} sec")
            print(f"Total steps: {train_metrics['total_timesteps']}")
            print(f"Final model: {final_model_path}")
            print("No test data provided for final evaluation.")


if __name__ == "__main__":
    main() 