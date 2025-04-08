#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a DQN agent for cryptocurrency trading using the LSTM model for state representation.
"""

import os
import argparse
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.lstm_lightning import LightningTimeSeriesModel as LSTMModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a DQN agent for cryptocurrency trading')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                        help='Directory containing the training data')
    parser.add_argument('--lstm_model_path', type=str, required=True,
                        help='Path to the trained LSTM model checkpoint')
    parser.add_argument('--output_dir', type=str, default='models/dqn',
                        help='Directory to save trained models and results')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Maximum steps per episode (None uses full data)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for experience replay')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.01,
                        help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Exploration rate decay factor')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for the Q-network')
    parser.add_argument('--update_target_frequency', type=int, default=10,
                        help='Frequency of target network updates (episodes)')
    
    # Environment parameters
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for observations')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for trading')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage')
    parser.add_argument('--reward_scaling', type=float, default=0.01,
                        help='Scaling factor for rewards')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda, None for auto-detection)')
    
    # Saving options
    parser.add_argument('--save_frequency', type=int, default=50,
                        help='Frequency to save model checkpoints (episodes)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    return parser.parse_args()

def load_lstm_model(model_path, device=None):
    """
    Load the trained LSTM model.
    
    Parameters:
    -----------
    model_path : str
        Path to the LSTM model checkpoint
    device : str
        Device to use for computation ('cpu' or 'cuda')
        
    Returns:
    --------
    LSTMModel
        Loaded LSTM model
    """
    logger.info(f"Loading LSTM model from {model_path}")
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found at {model_path}")
    
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load model configuration or use defaults
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            # Default configuration if not found in checkpoint
            model_config = {
                'input_dims': {'15m': 5, '4h': 5, '1d': 5},  # Dictionary of input dimensions for each timeframe
                'hidden_dims': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'num_classes': 3  # Changed from output_dim to num_classes
            }
            logger.warning("Using default LSTM configuration as it was not found in the checkpoint")
        
        # Initialize model
        model = LSTMModel(**model_config)
        
        # Load state dictionary
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Clean state dict (remove 'model.' prefix if present)
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    clean_state_dict[k[6:]] = v
                else:
                    clean_state_dict[k] = v
            model.load_state_dict(clean_state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        # Set model to evaluation mode
        model.eval()
        model.to(device)
        
        logger.info(f"LSTM model loaded successfully, hidden_dim={model_config['hidden_dims']}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading LSTM model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def train_dqn_agent(args):
    """
    Train the DQN agent.
    
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
    
    # Load LSTM model
    lstm_model = load_lstm_model(args.lstm_model_path, device)
    
    # Prepare data path
    train_data_path = os.path.join(args.data_dir, "train_data.h5")
    
    # Check if the file exists
    if not os.path.exists(train_data_path):
        train_data_path = os.path.join(args.data_dir, "train_data.h5")
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    # Create trading environment
    env = TradingEnvironment(
        data_path=train_data_path,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        reward_scaling=args.reward_scaling,
        device=device
    )
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        lstm_model=lstm_model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=10000,
        batch_size=args.batch_size,
        device=device
    )
    
    # Training loop
    episode_rewards = []
    episode_profits = []
    episode_trades = []
    best_reward = float('-inf')
    
    # Save training arguments
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    logger.info("Starting training...")
    for episode in range(1, args.episodes + 1):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Limit steps if specified
        max_steps = args.max_steps if args.max_steps is not None else float('inf')
        
        # Episode loop
        progress_bar = tqdm(total=min(max_steps, env.data_length - env.current_step), 
                           desc=f"Episode {episode}/{args.episodes}", 
                           disable=False)
        
        while not done and steps < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.replay_buffer) > args.batch_size:
                loss = agent.update()
                if args.debug and steps % 100 == 0:
                    logger.debug(f"Step {steps}, Loss: {loss:.4f}")
            
            # Update state and rewards
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'reward': f"{episode_reward:.2f}",
                'balance': f"{info['balance']:.2f}",
                'trades': info['total_trades']
            })
        
        progress_bar.close()
        
        # Decay epsilon
        agent.epsilon = max(args.epsilon_end, agent.epsilon * args.epsilon_decay)
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_profits.append(env.total_profit)
        episode_trades.append(env.total_trades)
        
        logger.info(f"Episode {episode}/{args.episodes} - "
                   f"Reward: {episode_reward:.2f}, "
                   f"Profit: {env.total_profit:.2f}, "
                   f"Trades: {env.total_trades}, "
                   f"Epsilon: {agent.epsilon:.4f}")
        
        # Update target network
        if episode % args.update_target_frequency == 0:
            agent.update_target_network()
            logger.info(f"Target network updated at episode {episode}")
        
        # Save model checkpoint
        if episode % args.save_frequency == 0 or episode == args.episodes:
            checkpoint_path = os.path.join(args.output_dir, f"dqn_agent_episode_{episode}.pt")
            agent.save(checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(args.output_dir, "dqn_agent_best.pt")
            agent.save(best_model_path)
            logger.info(f"New best model saved with reward {best_reward:.2f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "dqn_agent_final.pt")
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot and save training results
    plot_training_results(episode_rewards, episode_profits, episode_trades, args.output_dir)
    
    logger.info("Training completed.")

def plot_training_results(rewards, profits, trades, output_dir):
    """
    Plot and save training results.
    
    Parameters:
    -----------
    rewards : list
        List of episode rewards
    profits : list
        List of episode profits
    trades : list
        List of episode trades
    output_dir : str
        Directory to save plots
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot profits
    ax2.plot(profits)
    ax2.set_title('Episode Profits')
    ax2.set_ylabel('Profit')
    ax2.grid(True)
    
    # Plot trades
    ax3.plot(trades)
    ax3.set_title('Episode Trades')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Trades')
    ax3.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    plt.close()
    
    # Save data as numpy arrays
    np.save(os.path.join(output_dir, 'rewards.npy'), np.array(rewards))
    np.save(os.path.join(output_dir, 'profits.npy'), np.array(profits))
    np.save(os.path.join(output_dir, 'trades.npy'), np.array(trades))

if __name__ == "__main__":
    args = parse_args()
    train_dqn_agent(args) 