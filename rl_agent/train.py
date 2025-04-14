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

# Import SB3 models
from stable_baselines3 import DQN, PPO, A2C, SAC  # Add A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv 
# Add SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env # For vectorized envs
# from stable_baselines3.common.vec_env import VecNormalize # Add VecNormalize
from stable_baselines3.dqn.policies import MlpPolicy as DqnMlpPolicy#, CnnPolicy as DqnCnnPolicy
from stable_baselines3.common.policies import ActorCriticPolicy # For PPO/A2C
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy # For SAC

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
# pylint: disable=wrong-import-position
from rl_agent.callbacks import get_callback_list
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
# Placeholder for environment patching utilities (to be added)
# from rl_agent.env_patching import comprehensive_environment_patch
from rl_agent.environment import TradingEnvironment # Ensure this is the correct env class
from rl_agent.models import LSTMFeatureExtractor, AntiHoldPolicy # Import custom components
# Remove LSTMDQN import if handled within create_model
# from rl_agent.models.lstm_dqn import LSTMDQN
from rl_agent.data.data_loader import DataLoader

# Define technical indicators - ensure these match the environment needs
# INDICATORS = ['macd', 'rsi', 'cci', 'dx', 'bb_upper', 'bb_lower', 'bb_middle', 'volume']

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
        choices=["dqn", "ppo", "a2c", "sac", "lstm_dqn"],  # Add lstm_dqn choice
        help="RL algorithm to use (default: dqn). Use lstm_dqn for DQN with "
             "LSTM features."
    )
    parser.add_argument(
        "--load_model", type=str, default=None,
        help="Path to a saved model to continue training from"
    )
    parser.add_argument(
        "--lstm_model_path", type=str, default=None,
        help="Path to a saved LSTM model state_dict for feature extraction"
    )

    # --- Data Parameters --- #
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the data file (CSV or HDF5)"
    )
    parser.add_argument(
        "--val_data_path", type=str, default=None,
        help="Path to the validation data file (optional)"
    )
    parser.add_argument(
        "--test_data_path", type=str, default=None,
        help="Path to the test data file (optional)"
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
        # Updated default features
        default="close,volume,open,high,low,rsi_14,macd,ema_9,ema_21", 
        help="Comma-separated list of features/indicators to use"
    )

    # --- Environment Parameters --- #
    parser.add_argument(
        "--initial_balance", type=float, default=10000,
        help="Initial balance for the trading environment (default: 10000)"
    )
    parser.add_argument(
        "--commission", type=float, default=0.001,
        help="Trading commission percentage (default: 0.001 = 0.1%)"
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
        help="Batch size for training (default: 2048 for PPO/A2C/SAC, "
             "DQN may use smaller)"
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
        help="Patience for early stopping based on eval reward "
             "(0 to disable, default: 10)"
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
        help="Number of steps per update for PPO/A2C (default: 2048)"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01,
        help="Entropy coefficient for PPO/A2C (default: 0.01)"
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

    # --- DQN/SAC Specific --- #
    parser.add_argument(
        "--buffer_size", type=int, default=100000,
        help="Replay buffer size for DQN/SAC (default: 100000)"
    )
    parser.add_argument(
        "--exploration_fraction", type=float, default=0.1,
        help="Fraction of training time for exploration decay in DQN (default: 0.1)"
    )
    parser.add_argument(
        "--exploration_initial_eps", type=float, default=1.0,
        help="Initial exploration rate (epsilon) for DQN (default: 1.0)"
    )
    parser.add_argument(
        "--exploration_final_eps", type=float, default=0.05,
        help="Final exploration rate (epsilon) for DQN (default: 0.05)"
    )
    parser.add_argument(
        "--target_update_interval", type=int, default=10000,
        help="Update frequency for target network in DQN/SAC (default: 10000)"
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
        help="Name for the model/log folder (default: auto-generated)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level (0: no output, 1: info, 2: debug)"
    )
    parser.add_argument(
        "--load_config", type=str, default=None,
        help="Path to configuration file to load (overrides defaults, "
             "overridden by CLI)"
    )
    parser.add_argument(
        "--cpu_only", action="store_true",
        help="Force using CPU even if GPU is available"
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Only evaluate a trained model (--load_model required), "
             "no training"
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
        is_eval: Flag indicating if this is for evaluation (might affect seeding).
    
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
    base_seed = config.get("seed")
    # env_seed = None # Unused variable
    if base_seed is not None:
        # Use different seeds for train vs eval vs parallel envs
        if is_eval:
            pass # env_seed = base_seed + 1  # Simple offset for eval
        else:
            pass # env_seed = base_seed # Use base seed for main training env
        # Note: For SubprocVecEnv, seeding is handled differently

    # Create environment instance
    # Ensure TradingEnvironment accepts these args or adapt
    env_kwargs = {
        "data": data,
        # Arguments expected by TradingEnvironment.__init__:
        "features": config.get("features"),  # Pass the list of feature names
        "sequence_length": config["sequence_length"],
        "initial_balance": config["initial_balance"],
        # Use commission for transaction_fee
        "transaction_fee": config["commission"],  
        "reward_scaling": config["reward_scaling"],
        # Optional args from TradingEnvironment - using defaults or config if available
        "window_size": config.get("window_size", 20), # Default 20 if not in config
        "max_position": config.get("max_position", 1.0), # Default 1.0
        "max_steps": config.get("max_steps"),
        "random_start": config.get("random_start", True) # Default True
        # Removed args not accepted by TradingEnvironment:
        # "buy_cost_pct", "sell_cost_pct", "state_space", "tech_indicator_list",
        # "max_holding_steps", "take_profit_pct", "target_cash_ratio", "symbol", 
        # "seed"
    }
    env = TradingEnvironment(**env_kwargs)

    # Apply wrappers (e.g., SafeTradingEnvWrapper)
    # env = SafeTradingEnvWrapper(env, ...)
    # env = TimeLimit(env, max_episode_steps=config["max_steps"])
    
    return env


# Update create_model to handle more algorithms and LSTM features
def create_model(
    env: gym.Env,
    config: Dict[str, Any],  # Use config dict
) -> Any:
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
    device = "cpu" if config["cpu_only"] else "auto"
    policy_kwargs = {}
    model_kwargs = {
        "policy": None,  # Determined below
        "env": env,
        "learning_rate": learning_rate,
        "gamma": config["gamma"],
        "seed": seed,
        "device": device,
        "verbose": 0,  # Handled by callbacks
        "policy_kwargs": policy_kwargs,  # Updated below
        "tensorboard_log": os.path.join(config["log_dir"], 
                                      config["model_name"], "sb3_logs")
    }

    # --- LSTM Feature Extractor Setup --- #
    use_lstm_features = model_type == "lstm_dqn" or config.get("lstm_model_path")
    if use_lstm_features:
        lstm_state_dict = None
        # Use LSTM hidden size as features_dim
        if config.get("lstm_model_path"):
            # Pass the path to let the extractor handle loading
            lstm_state_dict = config["lstm_model_path"]
            logger.info(f"Using LSTM model path: {config['lstm_model_path']}")
            
        policy_kwargs["features_extractor_class"] = LSTMFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "lstm_state_dict": lstm_state_dict,
            "features_dim": config.get("lstm_hidden_size", 128)
        }
        # Note: Environment patching for LSTM dimensions might be needed here
        # or before env creation
        # comprehensive_environment_patch(env, features_dim, logger)

        # Adjust network architecture if FC layers are specified
        if "fc_hidden_size" in config and config["fc_hidden_size"] > 0:
            # For ActorCritic policies (PPO/A2C)
            if model_type in ["ppo", "a2c"]:
                # SB3 uses net_arch=[dict(pi=[...], vf=[...])] or 
                # net_arch=[...] for shared
                # Assuming shared network for simplicity here
                policy_kwargs["net_arch"] = [config["fc_hidden_size"], 
                                             config["fc_hidden_size"]]
            # For DQN/SAC, net_arch might be handled differently 
            # or within policy class
            # We might need custom policies for LSTM + specific FC layers 
            # for DQN/SAC
            pass  # Placeholder - needs refinement for DQN/SAC

    # --- Algorithm Specific Setup --- #
    if model_type == "dqn" or model_type == "lstm_dqn":
        model_kwargs["policy"] = DqnMlpPolicy  # Base policy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config.get("learning_starts", 10000)
        model_kwargs["train_freq"] = config.get("train_freq", 1)
        model_kwargs["gradient_steps"] = config.get("gradient_steps", 1)
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["exploration_fraction"] = config["exploration_fraction"]
        model_kwargs["exploration_initial_eps"] = config["exploration_initial_eps"]
        model_kwargs["exploration_final_eps"] = config["exploration_final_eps"]
        # Use AntiHoldPolicy if model_type is DQN 
        # (or LSTM-DQN if desired)
        if not use_lstm_features:  # Apply AntiHold only if not using LSTM
            model_kwargs["policy"] = AntiHoldPolicy
            policy_kwargs["hold_action_bias"] = config.get("hold_action_bias", 
                                                         -1.0)
        if "net_arch" not in policy_kwargs:  # Set default arch if not set by LSTM
            policy_kwargs["net_arch"] = [config["fc_hidden_size"]] * 2 \
                if config["fc_hidden_size"] > 0 else [64, 64]
        # Remove kwargs not used by DQN
        model_kwargs.pop("tensorboard_log", None)  # DQN uses learn's tb_log_name
        model = DQN(**model_kwargs)

    elif model_type == "ppo":
        model_kwargs["policy"] = ActorCriticPolicy
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["n_epochs"] = config["n_epochs"]
        model_kwargs["ent_coef"] = config["ent_coef"]
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["clip_range"] = config["clip_range"]
        model_kwargs["gae_lambda"] = config.get("gae_lambda", 0.95)
        model_kwargs["max_grad_norm"] = config.get("max_grad_norm", 0.5)
        if "net_arch" not in policy_kwargs:  # Set default arch if not set by LSTM
            policy_kwargs["net_arch"] = [config["fc_hidden_size"]] * 2 \
                if config["fc_hidden_size"] > 0 else [64, 64]
        model = PPO(**model_kwargs)

    elif model_type == "a2c":
        model_kwargs["policy"] = ActorCriticPolicy
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["ent_coef"] = config["ent_coef"]
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["gae_lambda"] = config.get("gae_lambda", 0.95)
        model_kwargs["max_grad_norm"] = config.get("max_grad_norm", 0.5)
        model_kwargs["rms_prop_eps"] = config.get("rms_prop_eps", 1e-5)
        if "net_arch" not in policy_kwargs:  # Set default arch if not set by LSTM
            policy_kwargs["net_arch"] = [config["fc_hidden_size"]] * 2 \
                if config["fc_hidden_size"] > 0 else [64, 64]
        model = A2C(**model_kwargs)

    elif model_type == "sac":
        model_kwargs["policy"] = SacMlpPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config.get("learning_starts", 1000)
        model_kwargs["train_freq"] = config.get("train_freq", 1)
        model_kwargs["gradient_steps"] = config.get("gradient_steps", 1)
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["ent_coef"] = config.get("ent_coef", "auto")
        model_kwargs["target_entropy"] = config.get("target_entropy", "auto")
        model_kwargs["tau"] = config.get("tau", 0.005)
        # SAC needs policy_kwargs adjusted if fc_hidden_size is used
        if "fc_hidden_size" in config and config["fc_hidden_size"] > 0 and \
           "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = [config["fc_hidden_size"]] * 2
        model = SAC(**model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type.upper()} model with policy "
               f"{model_kwargs['policy'].__name__}")
    return model


# Update evaluate_model to use config
def evaluate_model(
    model: Any,
    env: gym.Env,  # Expects a single, unwrapped env for evaluation
    config: Dict[str, Any],
    n_episodes: int = 1,
    deterministic: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a model on an environment.
    """
    episode_rewards = []
    episode_lengths = []
    portfolio_values_list = []
    actions_list = []
    rewards_list = []
    seed = config.get("seed")
    
    for i in range(n_episodes):
        # Use offset seed for eval
        eval_seed = seed + i + 100 if seed is not None else None  
        obs, info = env.reset(seed=eval_seed)
        done = False
        episode_reward = 0
        episode_length = 0
        # Get initial balance correctly
        initial_balance = getattr(env, 'initial_balance', 
                                config["initial_balance"])
        current_portfolio_values = [info.get('portfolio_value', initial_balance)]
        current_actions = []
        current_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            portfolio_value = info.get('portfolio_value', 
                                       current_portfolio_values[-1])
            current_portfolio_values.append(portfolio_value)
            current_actions.append(int(action))
            current_rewards.append(float(reward))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        portfolio_values_list.append(current_portfolio_values)
        actions_list.append(current_actions)
        rewards_list.append(current_rewards)

    mean_reward = np.mean(episode_rewards)
    portfolio_values = np.array(portfolio_values_list[0])
    actions = np.array(actions_list[0])
    rewards = np.array(rewards_list[0])
    
    return mean_reward, portfolio_values, actions, rewards


# Update train function
def train(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a reinforcement learning agent based on config.
    """
    # Setup logger
    log_path = os.path.join(config["log_dir"], config["model_name"])
    ensure_dir_exists(log_path)
    setup_logger(
        log_dir=log_path,
        log_level=logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO,
    )

    # Setup SB3 logger (adjust path)
    sb3_log_path = log_path  # Log directly into model folder
    sb3_logger_instance = setup_sb3_logger(log_dir=sb3_log_path)
    
    # Save configuration
    save_config(
        config=config,
        log_dir=log_path,
        filename="config.json",
    )
    
    # Set random seeds
    if config.get("seed") is not None:
        set_seeds(config["seed"])
    
    # Log system information
    logger.info(f"Starting training with model: {config['model_name']}")
    device = "cpu" if config["cpu_only"] else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    check_resources(logger)

    # --- Environment Creation --- #
    num_envs = config.get("num_envs", 1)
    logger.info(f"Creating {num_envs} parallel environment(s)...")

    # Function to create a single env instance
    # Ensure seed passed here can be None initially
    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            env_config = config.copy()
            # Calculate a unique seed for this env instance
            # If base_seed is None, avoid TypeError by passing None
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed # Pass the calculated seed
            env = create_env(config=env_config, is_eval=False)
            # Apply wrappers if needed (e.g., Monitor)
            # env = Monitor(env, 
            #               filename=os.path.join(log_path, f'monitor_{rank}.csv'))
            # Don't explicitly seed here, rely on seed passed to __init__ 
            # via env_config
            return env
        return _init

    # Create vectorized environment
    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    base_seed_from_config = config.get("seed")  # This can be None
    train_env = make_vec_env(
        env_id=make_single_env(rank=0, 
                                base_seed=base_seed_from_config), # Pass base seed
        n_envs=num_envs,
        seed=None,  # Let make_single_env handle seed generation per instance
        vec_env_cls=vec_env_cls,
        env_kwargs=None  # Pass None here as we use a lambda for env_id
    )

    # Apply VecNormalize if desired (needs careful handling with LSTM)
    # train_env = VecNormalize(train_env, norm_obs=True, 
    #                          norm_reward=False, clip_obs=10.)

    # Create validation environment if specified
    eval_env = None
    if config.get("val_data_path"):
        logger.info(f"Creating validation environment from: "
                   f"{config['val_data_path']}")
        eval_env_config = config.copy()
        eval_env_config["data_path"] = config["val_data_path"]
        # Create single eval env instance, then potentially vectorize if 
        # needed by callback
        eval_env_instance = create_env(config=eval_env_config, is_eval=True)
        # Wrap if necessary (e.g., Monitor)
        # eval_env_instance = Monitor(eval_env_instance)
        eval_env = DummyVecEnv([lambda: eval_env_instance]) # Wrap for SB3 Eval
        # Apply VecNormalize if train_env is normalized 
        # (MUST use stats from train_env)
        # if isinstance(train_env, VecNormalize):
        #     eval_env = VecNormalize.load(os.path.join(log_path, 
        #                                             "vecnormalize.pkl"), eval_env)
        #     eval_env.training = False
        #     eval_env.norm_reward = False

    # --- Model Creation / Loading --- #
    if config.get("load_model"):
        load_path = config["load_model"]
        logger.info(f"Loading model from: {load_path}")
        if not os.path.exists(load_path):
            logger.error(f"Model path not found: {load_path}")
            sys.exit(1)

        model_cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
                       "lstm_dqn": DQN}  # Load LSTM-DQN as DQN
        model = model_cls[config["model_type"]].load(load_path, env=train_env)
        # Reset timesteps? Usually False when loading unless specified
    else:
        logger.info(f"Creating new {config['model_type']} model")
        model = create_model(env=train_env, config=config)

    # Set logger for the model
    model.set_logger(sb3_logger_instance)

    # --- Callbacks --- #
    checkpoint_save_path = os.path.join(config["checkpoint_dir"], 
                                      config["model_name"])
    ensure_dir_exists(checkpoint_save_path)
    callbacks = get_callback_list(
        eval_env=eval_env,
        log_dir=log_path,  # Base log dir for metrics/best model
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        save_freq=config["save_freq"],  # Checkpoint freq
        keep_checkpoints=config["keep_checkpoints"],
        resource_check_freq=config["resource_check_freq"],
        metrics_log_freq=config["metrics_log_freq"],
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_save_path=checkpoint_save_path,  # Pass the path
        model_name=config["model_type"], # Pass model type as name prefix
        custom_callbacks=[]  # Keep empty or pass actual custom ones if needed
    )

    # --- Training --- #
    logger.info(f"Starting training for {config['total_timesteps']} "
               f"total timesteps...")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=1,  # Log basic SB3 stats every episode
            reset_num_timesteps=False if config.get("load_model") else True
        )
    except Exception as e:
        logger.critical(f"Training failed: {e}", exc_info=True)
        error_save_path = os.path.join(log_path, "model_on_error.zip")
        try:
            model.save(error_save_path)
            logger.info(f"Model state saved to {error_save_path} due to error.")
        except Exception as save_e:
            logger.error(f"Could not save model after error: {save_e}")
        sys.exit(1)
    finally:
        # Close environments
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
    # Save VecNormalize stats if used
    # if isinstance(train_env, VecNormalize):
    #     train_env.save(os.path.join(log_path, "vecnormalize.pkl"))
    
    # --- Final Metrics --- #
    metrics = {
        "training_time": training_time,
        "total_timesteps": model.num_timesteps,
        "model_type": config["model_type"],
        # Add other high-level training outcomes if available
    }
    
    return model, metrics


# Update evaluate function
def evaluate(
    config: Dict[str, Any],  # Use config dict
) -> Dict[str, Any]:
    """
    Evaluate a trained model based on config.
    Requires load_model and test_data_path in config.
    """
    load_path = config.get("load_model")
    test_data_path = config.get("test_data_path")
    n_episodes = config.get("n_eval_episodes", 1)
    model_name = config["model_name"]
    log_dir = config["log_dir"]

    if not load_path or not test_data_path:
        # Use print here as logger might not be configured yet
        print("Error: Evaluation requires --load_model and --test_data_path.")
        sys.exit(1)
    if not os.path.exists(load_path):
        print(f"Error: Model path not found: {load_path}")
        sys.exit(1)

    # Setup logger for evaluation
    eval_log_dir = os.path.join(log_dir, model_name, "evaluation")
    ensure_dir_exists(eval_log_dir)
    setup_logger(
        log_dir=eval_log_dir,
        log_filename="evaluation.log",
    )
    
    logger.info(f"Evaluating model: {load_path}")
    logger.info(f"Using test data from: {test_data_path}")
    
    # Create test environment
    eval_config = config.copy()
    eval_config["data_path"] = test_data_path
    test_env_instance = create_env(config=eval_config, is_eval=True)
    # Potentially wrap with Monitor
    # test_env_instance = Monitor(test_env_instance)

    # Load model (don't wrap env here, load needs unwrapped)
    logger.info(f"Loading model from {load_path}")
    model_cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
                   "lstm_dqn": DQN}
    try:
        # Load without env first to check custom objects if necessary
        # model = model_cls[config["model_type"]].load(load_path, 
        #                                             custom_objects={...})
        model = model_cls[config["model_type"]].load(load_path)
        # Set the environment AFTER loading if necessary 
        # (SB3 usually handles this)
        # model.set_env(DummyVecEnv([lambda: test_env_instance]))
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)

    # Evaluate model using the single instance env
    logger.info(f"Starting evaluation for {n_episodes} episodes")
    evaluation_start_time = time.time()
    mean_reward, portfolio_values, actions, rewards = evaluate_model(
        model=model,
        env=test_env_instance,  # Pass the single instance
        config=config,
        n_episodes=n_episodes,
        deterministic=True,
    )
    evaluation_time = time.time() - evaluation_start_time
    test_env_instance.close()  # Close the single env instance
    
    # Calculate trading metrics
    logger.info("Calculating trading metrics")
    trading_metrics = calculate_trading_metrics(
        portfolio_values=portfolio_values,
        risk_free_rate=0.0,
    )
    
    # Create evaluation plots
    logger.info("Creating evaluation plots")
    plot_path = os.path.join(eval_log_dir, f"{model_name}_evaluation_plot.png")
    create_evaluation_plots(
        portfolio_values=portfolio_values,
        actions=actions,
        rewards=rewards,
        save_path=plot_path,
        show_plot=False,
    )
    
    # Prepare evaluation metrics
    metrics = {
        "mean_reward": float(mean_reward),
        "final_portfolio_value": float(portfolio_values[-1]),
        "total_return": float((portfolio_values[-1] - portfolio_values[0]) /
                              (portfolio_values[0] + 1e-10)),
        "evaluation_time": evaluation_time,
        "n_episodes": n_episodes,
        **trading_metrics,
    }
    
    # Save metrics to file
    metrics_path = os.path.join(eval_log_dir, "metrics.json")
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
            logger.info(f"Evaluation metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation metrics: {e}")
    
    # Log metrics
    logger.info(f"Evaluation plot saved to {plot_path}")
    logger.info(f"Mean reward: {mean_reward:.2f}")
    logger.info(f"Final portfolio value: {portfolio_values[-1]:.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    
    return metrics


# Updated main function to handle different models and eval mode
def main():
    """Main function to parse args, setup, and run training or evaluation."""
    args = parse_args()
    config = args_to_config(args)
    
    # --- Config Loading --- #
    if args.load_config is not None:
        if os.path.exists(args.load_config):
            # Use print before logger setup
            print(f"Loading configuration from {args.load_config}") 
            file_config = load_config(args.load_config)
            # Update file_config with non-None CLI args
            cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
            file_config.update(cli_overrides)
            config = file_config
            # Re-process features if loaded from config and is string
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
    if config["eval_only"]:
        # --- Evaluation Mode --- #
        print("Running in Evaluation-Only Mode")
        if config.get("load_model") is None:
            print("Error: Must provide --load_model for evaluation mode.")
            sys.exit(1)
        if config.get("test_data_path") is None:
            print("Error: Must provide --test_data_path for evaluation mode.")
            sys.exit(1)
        evaluate(config=config)

    else:
        # --- Training Mode --- #
        print(f"Running in Training Mode for model: {config['model_type']}")
        model, train_metrics = train(config)
        
        # Evaluate on test data if provided after training
        if config.get("test_data_path") is not None:
            print("\nStarting final evaluation on test data...")
            # Ensure the final model path is correctly formed
            final_model_path = os.path.join(
                log_base_dir, model_name, "final_model.zip"
            )
            # Create a config for evaluation based on final state
            eval_config = config.copy()
            eval_config["load_model"] = final_model_path
            eval_config["test_data_path"] = config["test_data_path"]

            test_metrics = evaluate(config=eval_config)

            print("\n--- Training Summary ---")
            print(f"Training time: {train_metrics['training_time']:.2f} seconds")
            print(f"Total timesteps trained: {train_metrics['total_timesteps']}")
            print(f"Final model saved to: {final_model_path}")
            print("\n--- Test Set Evaluation Results ---")
            print(f"Mean reward: {test_metrics['mean_reward']:.2f}")
            print(f"Final portfolio value: "
                  f"{test_metrics['final_portfolio_value']:.2f}")
            print(f"Total Return: {test_metrics['total_return']:.2%}")
            print(f"Sharpe ratio: {test_metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown: {test_metrics['max_drawdown']:.2%}")
        else:
            # No test evaluation
            final_model_path = os.path.join(log_base_dir, model_name, 
                                        "final_model.zip")
            print("\n--- Training Completed --- ")
            print(f"Training time: {train_metrics['training_time']:.2f} seconds")
            print(f"Total timesteps trained: {train_metrics['total_timesteps']}")
            print(f"Final model saved to: {final_model_path}")
            print("No test data provided (--test_data_path) for final "
                  "evaluation.")


if __name__ == "__main__":
    main() 