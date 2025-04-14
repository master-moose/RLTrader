#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script to demonstrate LSTM-DQN for cryptocurrency trading.

This script shows how to:
1. Load and preprocess cryptocurrency data
2. Create a trading environment
3. Train an LSTM-DQN agent
4. Evaluate the agent's performance
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import logging

# Import our modules
from rl_agent.data.data_loader import DataLoader
from rl_agent.environment.trading_env import TradingEnvironment
from rl_agent.models.lstm_dqn import LSTMDQN
from rl_agent.callbacks import get_callback_list
from rl_agent.utils import (
    setup_logger, 
    create_evaluation_plots,
    calculate_trading_metrics,
    ensure_dir_exists
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LSTM-DQN Trading Example")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the cryptocurrency data CSV file")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")
    parser.add_argument("--total_timesteps", type=int, default=10000,
                        help="Total number of timesteps to train for")
    parser.add_argument("--eval_freq", type=int, default=1000,
                        help="Evaluation frequency during training")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for training")
    parser.add_argument("--lstm_hidden_size", type=int, default=128,
                        help="Hidden size of the LSTM layer")
    parser.add_argument("--sequence_length", type=int, default=30,
                        help="Sequence length for the LSTM")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0, 1, or 2)")
    
    return parser.parse_args()


def load_and_preprocess_data(data_path):
    """Load and preprocess the cryptocurrency data."""
    # Create data loader
    data_loader = DataLoader(
        data_path=data_path,
        timestamp_column="timestamp",
        price_column="close",
        volume_column="volume",
        drop_na=True,
        fill_method="ffill"
    )
    
    # Load data
    data = data_loader.load_data()
    
    # Add technical indicators
    data = data_loader.add_technical_indicators(
        data=data,
        indicators=["rsi", "macd", "bollinger", "ema"]
    )
    
    # Fill any remaining NaN values
    data = data.fillna(method="ffill").fillna(method="bfill")
    
    return data


def create_environment(data, sequence_length=30):
    """Create a trading environment."""
    # Define features to use
    features = ["close", "volume", "open", "high", "low", "rsi", "macd", "bb_upper", "bb_lower"]
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        features=features,
        sequence_length=sequence_length,
        initial_balance=10000.0,
        transaction_fee=0.001,
        reward_scaling=1.0,
        window_size=20,
        max_position=1.0,
        random_start=True
    )
    
    return env


def train_agent(env, args, log_dir):
    """Train the LSTM-DQN agent."""
    # Create model
    model = LSTMDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={
            "lstm_hidden_size": args.lstm_hidden_size,
            "fc_hidden_size": 64,
            "hold_action_bias": -0.2  # Slightly discourage holding
        },
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        verbose=args.verbose
    )
    
    # Setup callbacks
    callbacks = get_callback_list(
        eval_env=env,
        log_dir=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=1,
        save_freq=args.eval_freq,
        resource_check_freq=500,
        metrics_log_freq=500
    )
    
    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=500
    )
    
    # Save the final model
    model_path = os.path.join(log_dir, "final_model.zip")
    model.save(model_path)
    
    return model, model_path


def evaluate_agent(model, env, args, log_dir):
    """Evaluate the trained agent."""
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Lists to store evaluation data
    portfolio_values = [env.portfolio_value]
    rewards = []
    actions = []
    positions = []
    
    # Run evaluation episode
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        # Record data
        portfolio_values.append(env.portfolio_value)
        rewards.append(reward)
        actions.append(int(action))
        positions.append(env.shares_held * env.data["close"].iloc[env.current_step])
    
    # Calculate metrics
    metrics = calculate_trading_metrics(portfolio_values)
    
    # Create evaluation plot
    plot_path = os.path.join(log_dir, "evaluation_plot.png")
    create_evaluation_plots(
        portfolio_values=portfolio_values,
        actions=actions,
        rewards=rewards,
        positions=positions,
        save_path=plot_path,
        show_plot=False
    )
    
    return metrics, portfolio_values, actions, rewards


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    log_dir = os.path.join(args.output_dir, "lstm_dqn_example")
    ensure_dir_exists(log_dir)
    
    # Setup logger
    logger = setup_logger(
        log_dir=log_dir,
        log_level=logging.INFO,
        console_level=logging.INFO
    )
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data = load_and_preprocess_data(args.data_path)
    logger.info(f"Loaded data with shape: {data.shape}")
    
    # Create environment
    logger.info("Creating trading environment...")
    env = create_environment(data, args.sequence_length)
    
    # Train agent
    logger.info(f"Training LSTM-DQN agent for {args.total_timesteps} timesteps...")
    model, model_path = train_agent(env, args, log_dir)
    logger.info(f"Training completed. Model saved to {model_path}")
    
    # Evaluate agent
    logger.info("Evaluating trained agent...")
    metrics, portfolio_values, actions, rewards = evaluate_agent(model, env, args, log_dir)
    
    # Print evaluation results
    logger.info("\nEvaluation Results:")
    logger.info(f"Initial Portfolio Value: {portfolio_values[0]:.2f}")
    logger.info(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Evaluation plot saved to {os.path.join(log_dir, 'evaluation_plot.png')}")
    
    # Optional: Display example trades
    buy_count = actions.count(2)
    sell_count = actions.count(0)
    hold_count = actions.count(1)
    logger.info(f"\nAction Distribution: {buy_count} buys, {sell_count} sells, {hold_count} holds")


if __name__ == "__main__":
    main() 