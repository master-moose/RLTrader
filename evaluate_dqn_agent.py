#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a trained DQN agent for cryptocurrency trading.
"""

import os
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
from datetime import datetime

from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.lstm_lightning import LightningTimeSeriesModel as LSTMModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a DQN agent for cryptocurrency trading')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                        help='Directory containing the test data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained DQN agent')
    parser.add_argument('--lstm_model_path', type=str, required=True,
                        help='Path to the trained LSTM model used for state representation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    # Evaluation parameters
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for observations')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for trading')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda, None for auto-detection)')
    
    return parser.parse_args()

def load_agent(model_path, lstm_model_path, state_dim, action_dim, device=None):
    """
    Load the trained DQN agent and LSTM model.
    
    Parameters:
    -----------
    model_path : str
        Path to the DQN agent checkpoint
    lstm_model_path : str
        Path to the LSTM model checkpoint
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    device : str
        Device to use for computation ('cpu' or 'cuda')
        
    Returns:
    --------
    DQNAgent
        Loaded DQN agent
    """
    logger.info(f"Loading DQN agent from {model_path}")
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DQN agent not found at {model_path}")
    if not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found at {lstm_model_path}")
    
    try:
        # Load LSTM model
        lstm_checkpoint = torch.load(lstm_model_path, map_location=device)
        
        # Load model configuration or use defaults
        if 'config' in lstm_checkpoint:
            lstm_config = lstm_checkpoint['config']
        else:
            # Default configuration if not found in checkpoint
            lstm_config = {
                'input_dim': 5,
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'output_dim': 3
            }
            logger.warning("Using default LSTM configuration as it was not found in the checkpoint")
        
        # Initialize LSTM model
        lstm_model = LSTMModel(**lstm_config)
        
        # Load LSTM state dictionary
        if 'state_dict' in lstm_checkpoint:
            lstm_state_dict = lstm_checkpoint['state_dict']
            # Clean state dict (remove 'model.' prefix if present)
            clean_lstm_state_dict = {}
            for k, v in lstm_state_dict.items():
                if k.startswith('model.'):
                    clean_lstm_state_dict[k[6:]] = v
                else:
                    clean_lstm_state_dict[k] = v
            lstm_model.load_state_dict(clean_lstm_state_dict)
        else:
            lstm_model.load_state_dict(lstm_checkpoint)
        
        # Set LSTM model to evaluation mode
        lstm_model.eval()
        lstm_model.to(device)
        
        logger.info(f"LSTM model loaded successfully")
        
        # Load DQN agent
        # Initialize DQN agent with the LSTM model
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_model=lstm_model,
            device=device
        )
        
        # Load agent state dict
        agent_checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(agent_checkpoint)
        
        logger.info(f"DQN agent loaded successfully")
        
        return agent
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def evaluate_agent(args):
    """
    Evaluate the DQN agent.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Prepare data path
    test_data_path = os.path.join(args.data_dir, "test_data.h5")
    
    # Check if the file exists
    if not os.path.exists(test_data_path):
        test_data_path = os.path.join(args.data_dir, "test_data.h5")
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    # Create trading environment
    env = TradingEnvironment(
        data_path=test_data_path,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        device=device
    )
    
    # Load agent
    agent = load_agent(
        model_path=args.model_path,
        lstm_model_path=args.lstm_model_path,
        state_dim=env.state_dim,
        action_dim=env.action_space,
        device=device
    )
    
    # Evaluation loop
    state = env.reset()
    done = False
    rewards = []
    balances = []
    portfolio_values = []
    actions = []
    prices = []
    positions = []
    timestamps = []
    
    # Setup progress bar
    progress_bar = tqdm(total=env.data_length - env.current_step, 
                       desc=f"Evaluating agent", 
                       disable=False)
    
    # Evaluate the agent
    while not done:
        # Get action from agent (no exploration)
        action = agent.select_action(state, explore=False)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        
        # Render if requested
        if args.render:
            env.render()
        
        # Store data
        rewards.append(reward)
        balances.append(info['balance'])
        portfolio_values.append(env._calculate_portfolio_value(info['price']))
        actions.append(action)
        prices.append(info['price'])
        positions.append(info['position'])
        
        # Extract timestamp if available
        try:
            timestamp = env.market_data[env.primary_tf]['timestamp'].iloc[env.current_step]
            timestamps.append(timestamp)
        except (KeyError, IndexError):
            timestamps.append(env.current_step)
        
        # Update state
        state = next_state
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            'balance': f"{info['balance']:.2f}",
            'profit': f"{env.total_profit:.2f}",
            'trades': info['total_trades']
        })
    
    progress_bar.close()
    
    # Log evaluation results
    total_reward = sum(rewards)
    final_balance = balances[-1]
    final_portfolio_value = portfolio_values[-1]
    total_trades = len([i for i in range(1, len(actions)) if actions[i] != actions[i-1] and actions[i] != 0])
    profit_percentage = (final_portfolio_value - args.initial_balance) / args.initial_balance * 100
    
    logger.info(f"Evaluation completed:")
    logger.info(f"  Total reward: {total_reward:.2f}")
    logger.info(f"  Final balance: {final_balance:.2f}")
    logger.info(f"  Final portfolio value: {final_portfolio_value:.2f}")
    logger.info(f"  Total trades: {total_trades}")
    logger.info(f"  Profit: {final_portfolio_value - args.initial_balance:.2f} ({profit_percentage:.2f}%)")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'action': actions,
        'position': positions,
        'reward': rewards,
        'balance': balances,
        'portfolio_value': portfolio_values
    })
    
    results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    
    # Plot and save evaluation results
    plot_evaluation_results(
        timestamps, prices, actions, positions, portfolio_values, args.initial_balance, args.output_dir
    )
    
    # Save evaluation metrics
    metrics = {
        'total_reward': float(total_reward),
        'final_balance': float(final_balance),
        'final_portfolio_value': float(final_portfolio_value),
        'initial_balance': float(args.initial_balance),
        'total_trades': int(total_trades),
        'profit': float(final_portfolio_value - args.initial_balance),
        'profit_percentage': float(profit_percentage)
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_evaluation_results(timestamps, prices, actions, positions, portfolio_values, initial_balance, output_dir):
    """
    Plot and save evaluation results.
    
    Parameters:
    -----------
    timestamps : list
        List of timestamps
    prices : list
        List of prices
    actions : list
        List of actions taken by the agent
    positions : list
        List of positions
    portfolio_values : list
        List of portfolio values
    initial_balance : float
        Initial balance
    output_dir : str
        Directory to save plots
    """
    # Convert timestamps to datetime objects if they're not already
    if isinstance(timestamps[0], (int, float)):
        datetime_timestamps = [datetime.fromtimestamp(ts) if isinstance(ts, (int, float)) else ts for ts in timestamps]
    else:
        datetime_timestamps = timestamps
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot price
    ax1.plot(datetime_timestamps, prices, label='Price')
    ax1.set_title('Price')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Format x-axis for dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 10)))
    
    # Plot positions and actions
    buy_idx = [i for i, a in enumerate(actions) if a == 1]
    sell_idx = [i for i, a in enumerate(actions) if a == 2]
    
    ax2.plot(datetime_timestamps, positions, label='Position')
    ax2.scatter([datetime_timestamps[i] for i in buy_idx], [positions[i] for i in buy_idx], 
               color='green', marker='^', s=100, label='Buy')
    ax2.scatter([datetime_timestamps[i] for i in sell_idx], [positions[i] for i in sell_idx], 
               color='red', marker='v', s=100, label='Sell')
    ax2.set_title('Positions and Actions')
    ax2.set_ylabel('Position')
    ax2.grid(True)
    ax2.legend()
    
    # Plot portfolio value
    portfolio_returns = [pv / initial_balance for pv in portfolio_values]
    ax3.plot(datetime_timestamps, portfolio_returns, label='Portfolio Value')
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Initial Balance')
    ax3.set_title('Portfolio Value (Return)')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Return')
    ax3.grid(True)
    ax3.legend()
    
    # Rotate date labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_results.png'))
    plt.close()
    
    # Create separate plot for comparing price and portfolio value
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize price and portfolio value for comparison
    normalized_price = [p / prices[0] for p in prices]
    normalized_portfolio = [pv / portfolio_values[0] for pv in portfolio_values]
    
    ax.plot(datetime_timestamps, normalized_price, label='Price (normalized)')
    ax.plot(datetime_timestamps, normalized_portfolio, label='Portfolio Value (normalized)')
    ax.set_title('Comparison of Price and Portfolio Value')
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Value')
    ax.grid(True)
    ax.legend()
    
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps) // 10)))
    
    # Rotate date labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_vs_portfolio.png'))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(args) 