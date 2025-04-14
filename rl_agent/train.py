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
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from rl_agent.callbacks import get_callback_list
from rl_agent.utils import (
    setup_logger, 
    setup_sb3_logger,
    check_resources, 
    save_config,
    load_config,
    create_evaluation_plots,
    calculate_trading_metrics,
    ensure_dir_exists
)
# These will be defined in future files
from rl_agent.environment import TradingEnvironment
from rl_agent.models.lstm_dqn import LSTMDQN
from rl_agent.data.data_loader import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LSTM-DQN agent for trading")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training data CSV file")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to the validation data CSV file (optional)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to the test data CSV file (optional)")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Length of price history sequence for LSTM")
    
    # Environment parameters
    parser.add_argument("--features", type=str, default="close,volume,open,high,low,rsi,macd,ema",
                        help="Comma-separated list of features to use")
    parser.add_argument("--initial_balance", type=float, default=10000,
                        help="Initial balance for the trading environment")
    parser.add_argument("--transaction_fee", type=float, default=0.001,
                        help="Transaction fee as a percentage (0.001 = 0.1%)")
    parser.add_argument("--reward_scaling", type=float, default=1.0,
                        help="Scaling factor for rewards")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="lstm_dqn", choices=["lstm_dqn", "dqn"],
                        help="Type of model to use")
    parser.add_argument("--lstm_hidden_size", type=int, default=128,
                        help="Hidden size of LSTM layer")
    parser.add_argument("--fc_hidden_size", type=int, default=64,
                        help="Hidden size of fully connected layers")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor for future rewards")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--target_update_interval", type=int, default=500,
                        help="Update interval for target network")
    parser.add_argument("--exploration_fraction", type=float, default=0.2,
                        help="Fraction of training steps for exploration schedule")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0,
                        help="Initial value of epsilon for exploration")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05,
                        help="Final value of epsilon for exploration")
    
    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=100000,
                        help="Total number of training timesteps")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Evaluation frequency during training")
    parser.add_argument("--n_eval_episodes", type=int, default=1,
                        help="Number of episodes for evaluation")
    parser.add_argument("--save_freq", type=int, default=10000,
                        help="Model saving frequency during training")
    parser.add_argument("--keep_checkpoints", type=int, default=3,
                        help="Number of model checkpoints to keep")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Patience for early stopping (0 to disable)")
    
    # Resource management parameters
    parser.add_argument("--resource_check_freq", type=int, default=2000,
                        help="Frequency of resource usage checks")
    parser.add_argument("--metrics_log_freq", type=int, default=1000,
                        help="Frequency of metrics logging")
    
    # General parameters
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for saving logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for saving model checkpoints")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name for the model (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level (0: no output, 1: info, 2: debug)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to model to resume training from")
    parser.add_argument("--load_config", type=str, default=None,
                        help="Path to configuration file to load")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate a trained model, no training")
    
    args = parser.parse_args()
    
    # Auto-generate model name if not provided
    if args.model_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.model_name = f"{args.model_type}_{timestamp}"
    
    # Process features
    args.features = args.features.split(",")
    
    return args


def args_to_config(args) -> Dict[str, Any]:
    """Convert argparse arguments to config dictionary."""
    return vars(args)


def create_env(
    data_path: str,
    features: List[str],
    sequence_length: int = 60,
    initial_balance: float = 10000,
    transaction_fee: float = 0.001,
    reward_scaling: float = 1.0,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    Create a trading environment.
    
    Args:
        data_path: Path to the data file
        features: List of features to use
        sequence_length: Length of history sequence
        initial_balance: Initial balance
        transaction_fee: Transaction fee as a percentage
        reward_scaling: Scaling factor for rewards
        seed: Random seed
    
    Returns:
        Trading environment
    """
    # Load data
    data_loader = DataLoader(data_path=data_path)
    data = data_loader.load_data()
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        features=features,
        sequence_length=sequence_length,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        reward_scaling=reward_scaling,
    )
    
    # Set seed if provided
    if seed is not None:
        env.seed(seed)
    
    return env


def create_model(
    env: gym.Env,
    model_type: str = "lstm_dqn",
    lstm_hidden_size: int = 128,
    fc_hidden_size: int = 64,
    learning_rate: float = 0.0001,
    gamma: float = 0.99,
    buffer_size: int = 100000,
    batch_size: int = 64,
    target_update_interval: int = 500,
    exploration_fraction: float = 0.2,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    seed: Optional[int] = None,
    device: str = "auto",
) -> Any:
    """
    Create a reinforcement learning model.
    
    Args:
        env: Training environment
        model_type: Type of model to use
        lstm_hidden_size: Hidden size of LSTM layer
        fc_hidden_size: Hidden size of fully connected layers
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        buffer_size: Size of the replay buffer
        batch_size: Batch size for training
        target_update_interval: Update interval for target network
        exploration_fraction: Fraction of training steps for exploration schedule
        exploration_initial_eps: Initial value of epsilon for exploration
        exploration_final_eps: Final value of epsilon for exploration
        seed: Random seed
        device: Device to use ('auto', 'cuda', or 'cpu')
    
    Returns:
        Reinforcement learning model
    """
    # Common model parameters
    model_kwargs = {
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "learning_starts": batch_size * 2,
        "batch_size": batch_size,
        "gamma": gamma,
        "target_update_interval": target_update_interval,
        "exploration_fraction": exploration_fraction,
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "verbose": 0,  # We'll handle logging with callbacks
    }
    
    # Set seed if provided
    if seed is not None:
        model_kwargs["seed"] = seed
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs["device"] = device
    
    # Create model based on type
    if model_type == "lstm_dqn":
        # Create LSTM-DQN model
        policy_kwargs = {
            "lstm_hidden_size": lstm_hidden_size,
            "fc_hidden_size": fc_hidden_size,
        }
        
        model = LSTMDQN(
            env=env,
            policy="MlpPolicy",  # This will be overridden in LSTMDQN
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
    elif model_type == "dqn":
        # Use standard DQN model
        policy_kwargs = {
            "net_arch": [fc_hidden_size, fc_hidden_size],
        }
        
        model = DQN(
            env=env,
            policy=MlpPolicy,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def evaluate_model(
    model: Any,
    env: gym.Env,
    n_episodes: int = 1,
    deterministic: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a model on an environment.
    
    Args:
        model: Model to evaluate
        env: Environment to evaluate on
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
    
    Returns:
        Tuple containing:
        - mean_reward: Mean reward across episodes
        - portfolio_values: Array of portfolio values
        - actions: Array of actions taken
        - rewards: Array of rewards received
    """
    episode_rewards = []
    episode_lengths = []
    portfolio_values_list = []
    actions_list = []
    rewards_list = []
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        portfolio_values = [env.portfolio_value]
        actions = []
        rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            portfolio_values.append(env.portfolio_value)
            actions.append(int(action))
            rewards.append(float(reward))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        portfolio_values_list.append(portfolio_values)
        actions_list.append(actions)
        rewards_list.append(rewards)
    
    # Calculate mean reward
    mean_reward = np.mean(episode_rewards)
    
    # Combine episode data (for simplicity, use the first episode if multiple)
    portfolio_values = np.array(portfolio_values_list[0])
    actions = np.array(actions_list[0])
    rewards = np.array(rewards_list[0])
    
    return mean_reward, portfolio_values, actions, rewards


def train(config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a reinforcement learning agent.
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple containing trained model and metrics dictionary
    """
    # Setup logger
    logger = setup_logger(
        log_dir=os.path.join(config["log_dir"], config["model_name"]),
        log_level=logging.DEBUG if config["verbose"] >= 2 else logging.INFO,
    )
    
    # Setup SB3 logger
    sb3_logger = setup_sb3_logger(
        log_dir=os.path.join(config["log_dir"], config["model_name"], "sb3_logs")
    )
    
    # Save configuration
    save_config(
        config=config,
        log_dir=os.path.join(config["log_dir"], config["model_name"]),
        filename="config.json",
    )
    
    # Set random seeds for reproducibility
    if config["seed"] is not None:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config["seed"])
    
    # Force CPU if requested
    if config["cpu_only"]:
        device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = "auto"
    
    # Log system information
    logger.info(f"Starting training with model: {config['model_name']}")
    logger.info(f"Training on device: {device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')}")
    resource_info = check_resources(logger)
    
    # Create training environment
    logger.info(f"Creating training environment with data from: {config['data_path']}")
    env = create_env(
        data_path=config["data_path"],
        features=config["features"],
        sequence_length=config["sequence_length"],
        initial_balance=config["initial_balance"],
        transaction_fee=config["transaction_fee"],
        reward_scaling=config["reward_scaling"],
        seed=config["seed"],
    )
    
    # Create validation environment if validation data is provided
    eval_env = None
    if config["val_data_path"] is not None:
        logger.info(f"Creating validation environment with data from: {config['val_data_path']}")
        eval_env = create_env(
            data_path=config["val_data_path"],
            features=config["features"],
            sequence_length=config["sequence_length"],
            initial_balance=config["initial_balance"],
            transaction_fee=config["transaction_fee"],
            reward_scaling=config["reward_scaling"],
            seed=config["seed"],
        )
    
    # Prepare callback parameters
    callback_params = {
        "eval_env": eval_env,
        "log_dir": os.path.join(config["log_dir"], config["model_name"]),
        "eval_freq": config["eval_freq"],
        "n_eval_episodes": config["n_eval_episodes"],
        "save_freq": config["save_freq"],
        "keep_checkpoints": config["keep_checkpoints"],
        "resource_check_freq": config["resource_check_freq"],
        "metrics_log_freq": config["metrics_log_freq"],
        "early_stopping_patience": config["early_stopping_patience"],
    }
    
    # Create or load model
    if config["resume_from"] is not None:
        logger.info(f"Resuming training from model: {config['resume_from']}")
        if config["model_type"] == "lstm_dqn":
            model = LSTMDQN.load(config["resume_from"], env=env)
        else:
            model = DQN.load(config["resume_from"], env=env)
    else:
        logger.info(f"Creating new {config['model_type']} model")
        model = create_model(
            env=env,
            model_type=config["model_type"],
            lstm_hidden_size=config["lstm_hidden_size"],
            fc_hidden_size=config["fc_hidden_size"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            target_update_interval=config["target_update_interval"],
            exploration_fraction=config["exploration_fraction"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            seed=config["seed"],
            device=device,
        )
    
    # Get callbacks
    callbacks = get_callback_list(**callback_params)
    
    # Train model
    logger.info(f"Starting training for {config['total_timesteps']} steps")
    training_start_time = time.time()
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        log_interval=config["metrics_log_freq"],
    )
    training_time = time.time() - training_start_time
    
    # Save final model
    final_model_path = os.path.join(config["log_dir"], config["model_name"], "final_model.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Calculate training metrics
    metrics = {
        "training_time": training_time,
        "total_timesteps": config["total_timesteps"],
        "model_type": config["model_type"],
    }
    
    return model, metrics


def evaluate(
    model_path: str,
    data_path: str,
    features: List[str],
    sequence_length: int = 60,
    initial_balance: float = 10000,
    transaction_fee: float = 0.001,
    reward_scaling: float = 1.0,
    n_episodes: int = 1,
    model_type: str = "lstm_dqn",
    log_dir: str = "./logs",
    model_name: str = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the test data
        features: List of features to use
        sequence_length: Length of history sequence
        initial_balance: Initial balance
        transaction_fee: Transaction fee as a percentage
        reward_scaling: Scaling factor for rewards
        n_episodes: Number of episodes to evaluate
        model_type: Type of model
        log_dir: Directory for saving logs
        model_name: Name for the model
        seed: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Auto-generate model name if not provided
    if model_name is None:
        model_name = os.path.basename(model_path).split(".")[0]
    
    # Setup logger
    eval_log_dir = os.path.join(log_dir, model_name, "evaluation")
    ensure_dir_exists(eval_log_dir)
    logger = setup_logger(
        log_dir=eval_log_dir,
        log_filename="evaluation.log",
    )
    
    logger.info(f"Evaluating model: {model_path}")
    logger.info(f"Using test data from: {data_path}")
    
    # Create test environment
    test_env = create_env(
        data_path=data_path,
        features=features,
        sequence_length=sequence_length,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        reward_scaling=reward_scaling,
        seed=seed,
    )
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    if model_type == "lstm_dqn":
        model = LSTMDQN.load(model_path)
    else:
        model = DQN.load(model_path)
    
    # Evaluate model
    logger.info(f"Starting evaluation for {n_episodes} episodes")
    evaluation_start_time = time.time()
    mean_reward, portfolio_values, actions, rewards = evaluate_model(
        model=model,
        env=test_env,
        n_episodes=n_episodes,
        deterministic=True,
    )
    evaluation_time = time.time() - evaluation_start_time
    
    # Calculate trading metrics
    logger.info("Calculating trading metrics")
    trading_metrics = calculate_trading_metrics(
        portfolio_values=portfolio_values,
        risk_free_rate=0.0,  # Assume zero risk-free rate
    )
    
    # Create evaluation plots
    logger.info("Creating evaluation plots")
    plot_path = os.path.join(eval_log_dir, "evaluation_plot.png")
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
        "return": float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]),
        "evaluation_time": evaluation_time,
        "n_episodes": n_episodes,
        **trading_metrics,
    }
    
    # Save metrics to file
    metrics_path = os.path.join(eval_log_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Log metrics
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    logger.info(f"Evaluation plot saved to {plot_path}")
    logger.info(f"Mean reward: {mean_reward:.2f}")
    logger.info(f"Final portfolio value: {portfolio_values[-1]:.2f}")
    logger.info(f"Return: {metrics['return']:.2%}")
    logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
    
    return metrics


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Convert arguments to config dictionary
    config = args_to_config(args)
    
    # Load config from file if provided
    if args.load_config is not None:
        file_config = load_config(args.load_config)
        # Override file config with command line arguments
        file_config.update({k: v for k, v in config.items() if v is not None})
        config = file_config
    
    # Create log directory
    ensure_dir_exists(os.path.join(config["log_dir"], config["model_name"]))
    
    if args.eval_only and args.resume_from is not None:
        # Evaluate mode
        evaluate(
            model_path=args.resume_from,
            data_path=args.test_data_path or args.data_path,
            features=config["features"],
            sequence_length=config["sequence_length"],
            initial_balance=config["initial_balance"],
            transaction_fee=config["transaction_fee"],
            reward_scaling=config["reward_scaling"],
            n_episodes=config["n_eval_episodes"],
            model_type=config["model_type"],
            log_dir=config["log_dir"],
            model_name=config["model_name"],
            seed=config["seed"],
        )
    else:
        # Training mode
        model, train_metrics = train(config)
        
        # Evaluate on test data if provided
        if args.test_data_path is not None:
            test_metrics = evaluate(
                model_path=os.path.join(config["log_dir"], config["model_name"], "final_model.zip"),
                data_path=args.test_data_path,
                features=config["features"],
                sequence_length=config["sequence_length"],
                initial_balance=config["initial_balance"],
                transaction_fee=config["transaction_fee"],
                reward_scaling=config["reward_scaling"],
                n_episodes=config["n_eval_episodes"],
                model_type=config["model_type"],
                log_dir=config["log_dir"],
                model_name=config["model_name"],
                seed=config["seed"],
            )
            
            # Print final results
            print("\nTraining completed!")
            print(f"Training time: {train_metrics['training_time']:.2f} seconds")
            print(f"Final model saved to: {os.path.join(config['log_dir'], config['model_name'], 'final_model.zip')}")
            print("\nTest Results:")
            print(f"Mean reward: {test_metrics['mean_reward']:.2f}")
            print(f"Final portfolio value: {test_metrics['final_portfolio_value']:.2f}")
            print(f"Return: {test_metrics['return']:.2%}")
            print(f"Sharpe ratio: {test_metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown: {test_metrics['max_drawdown']:.2%}")


if __name__ == "__main__":
    main() 