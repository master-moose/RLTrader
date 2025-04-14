#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration for the RL agent.

This module contains all the configuration parameters and logging setup
for the reinforcement learning agent.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up logging
def setup_logger(name=None, level=logging.INFO, log_dir="logs"):
    """Setup logger with file and console handlers"""
    if name is None:
        name = __name__
    
    # Create logger
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    _logger.propagate = False  # Don't propagate to parent loggers
    
    # Remove handlers if they already exist
    if _logger.hasHandlers():
        _logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    _logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    _logger.addHandler(console_handler)
    
    return _logger

# Initialize the logger
logger = setup_logger("rl_agent", logging.INFO)

# Environment settings
ENV_SETTINGS = {
    # Basic environment settings
    "lookback_window_size": 50,
    "initial_balance": 10000,
    "commission": 0.001,  # 0.1% commission on trades
    
    # Risk management parameters
    "max_risk_per_trade": 0.02,  # Maximum 2% risk per trade
    "take_profit_pct": 0.05,  # 5% take profit
    "stop_loss_pct": 0.03,  # 3% stop loss
    
    # Environment wrapper settings
    "trade_cooldown": 5,  # Minimum timesteps between trades
    "oscillation_penalty": 0.1,  # Penalty for oscillating between actions
    "consistency_reward": 0.01,  # Reward for maintaining consistent actions
}

# Model settings
MODEL_SETTINGS = {
    # Network architecture
    "policy_type": "MlpPolicy",
    "net_arch": [256, 256],
    "activation_fn": "relu",
    
    # Learning parameters
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 64,
    "tau": 0.005,  # Target network update rate
    "gamma": 0.99,  # Discount factor
    
    # Exploration parameters
    "exploration_fraction": 0.3,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    
    # Prioritized replay
    "prioritized_replay": True,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta0": 0.4,
    "prioritized_replay_beta_iters": None,  # Auto-calculate
    "prioritized_replay_eps": 1e-6,
    
    # Gradient clipping
    "max_grad_norm": 10,
}

# Training settings
TRAINING_SETTINGS = {
    # Basic training parameters
    "total_timesteps": 100000,
    "eval_freq": 5000,
    "n_eval_episodes": 10,
    "save_freq": 10000,
    
    # Callback parameters
    "patience": 20,  # Number of evaluations without improvement before early stopping
    "eval_log_path": "./logs/evaluations/",
    
    # Performance tracking
    "min_reward_threshold": -100,  # Minimum reward before resetting if doing poorly
    "portfolio_goal": 1.2,  # Target 20% portfolio growth
    
    # Checkpointing
    "checkpoint_dir": "./checkpoints/",
    "checkpoint_freq": 10000,  # Save checkpoints every 10000 steps
    "keep_checkpoints": 3,  # Keep only the last 3 checkpoints
}

# Model paths
def get_model_path(model_name, is_checkpoint=False):
    """Get the path to save/load a model"""
    base_dir = TRAINING_SETTINGS["checkpoint_dir"] if is_checkpoint else "./models"
    os.makedirs(base_dir, exist_ok=True)
    
    if is_checkpoint:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{model_name}_{timestamp}")
    
    return os.path.join(base_dir, model_name)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TENSORBOARD_DIR = os.path.join(LOGS_DIR, "tensorboard")

# Create directories
for directory in [DATA_DIR, LOGS_DIR, MODELS_DIR, TENSORBOARD_DIR, TRAINING_SETTINGS["eval_log_path"], TRAINING_SETTINGS["checkpoint_dir"]]:
    os.makedirs(directory, exist_ok=True)

# Tensorboard settings
TENSORBOARD_SETTINGS = {
    "log_dir": TENSORBOARD_DIR,
    "update_freq": 100,  # Update tensorboard every 100 steps
}

# Enable GPU if available
if torch := sys.modules.get("torch"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    logger.info("PyTorch not imported. Using CPU.") 