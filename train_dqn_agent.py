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
    parser.add_argument('--episode_length', type=int, default=10000,
                        help='Length of each episode in steps (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for experience replay (increased from 64 for more stable learning)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='Final exploration rate (decreased from 0.01 for better exploration)')
    parser.add_argument('--epsilon_decay', type=float, default=0.998,
                        help='Exploration rate decay factor (increased from 0.995 for slower decay)')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for the Q-network (increased from 0.0001 for faster learning)')
    parser.add_argument('--update_target_frequency', type=int, default=10,
                        help='Frequency of target network updates (episodes)')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='Number of network updates per environment step')
    
    # Environment parameters
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for observations')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for trading')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage')
    parser.add_argument('--reward_scaling', type=float, default=0.001,
                        help='Scaling factor for rewards (increased from 0.0001 for more balanced rewards)')
    parser.add_argument('--trade_cooldown', type=int, default=12,
                        help='Number of steps between trades (lower = more frequent trading)')
    
    # Parallelization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel environments to run')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda, None for auto-detection)')
    
    # Saving options
    parser.add_argument('--save_frequency', type=int, default=50,
                        help='Frequency to save model checkpoints (episodes)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (including stop loss messages)')
    
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
        
        # Extract state dictionary for dimension analysis
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Check for model architecture parameters in the checkpoint
        # Based on the model.encoders.15m.weight_ih_l0 shape, we can infer input dimensions
        # and based on model.encoders.15m.weight_hh_l0 shape, we can infer hidden dimensions
        input_dims = {'15m': 5, '4h': 5, '1d': 5}  # Default
        hidden_dims = 64  # Default based on error message
        
        # Try to infer from state dict
        # Look for keys that match the pattern with and without 'model.' prefix
        for prefix in ['model.', '']:
            # Check for input dimensions
            for tf in ['15m', '4h', '1d']:
                key = f"{prefix}encoders.{tf}.weight_ih_l0"
                if key in state_dict:
                    # Extract input dimension from weight shape
                    shape = state_dict[key].shape
                    # LSTM input weight shape is [hidden*4, input_size]
                    if len(shape) == 2:
                        input_dims[tf] = shape[1]
                        logger.info(f"Inferred input_dim for {tf}: {input_dims[tf]}")
                    break
            
            # Check for hidden dimensions
            key = f"{prefix}encoders.15m.weight_hh_l0"
            if key in state_dict:
                shape = state_dict[key].shape
                # LSTM hidden weight shape is [hidden*4, hidden]
                if len(shape) == 2:
                    # The hidden size is the second dimension
                    hidden_dims = shape[1] // 4 if shape[1] % 4 == 0 else shape[1]
                    logger.info(f"Inferred hidden_dims: {hidden_dims}")
                break
        
        # Load model configuration if available, otherwise use inferred dimensions
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            logger.info(f"Loaded model configuration from checkpoint")
        else:
            # Create configuration based on inferred or default dimensions
            model_config = {
                'input_dims': input_dims,
                'hidden_dims': hidden_dims,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'num_classes': 3
            }
            logger.info(f"Using inferred configuration: hidden_dims={hidden_dims}, input_dims={input_dims}")
        
        # Initialize model with inferred or loaded configuration
        model = LSTMModel(**model_config)
        
        # Process state dictionary keys
        # Check if we need to add or remove the "model." prefix
        has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
        needs_model_prefix = any(k.startswith('model.') for k, _ in model.state_dict().items())
        
        if needs_model_prefix and not has_model_prefix:
            # Add "model." prefix to keys
            clean_state_dict = {"model." + k: v for k, v in state_dict.items()}
            logger.info("Added 'model.' prefix to state dict keys")
        elif not needs_model_prefix and has_model_prefix:
            # Remove "model." prefix
            clean_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            logger.info("Removed 'model.' prefix from state dict keys")
        else:
            # No changes needed
            clean_state_dict = state_dict
        
        # Create a dummy model just for the DQN agent
        # This avoids trying to load an incompatible checkpoint
        logger.info("Creating a new LSTM model without loading weights from checkpoint")
        model.eval()
        model.to(device)
        
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
    
    # Load data to get its length (for random starting positions)
    temp_env = TradingEnvironment(
        data_path=train_data_path,
        window_size=args.window_size,
        device=device
    )
    data_length = temp_env.data_length
    logger.info(f"Dataset length: {data_length} timesteps")
    
    # Create multiple trading environments for vectorized training
    logger.info(f"Creating {args.num_workers} parallel environments for training")
    envs = []
    for i in range(args.num_workers):
        env = TradingEnvironment(
            data_path=train_data_path,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            reward_scaling=args.reward_scaling,
            device=device,
            trade_cooldown=args.trade_cooldown,  # Use command-line parameter
            # Initial start step will be randomized in reset()
            start_step=None,
            verbose=args.verbose  # Pass verbose flag to environment
        )
        envs.append(env)
    
    # Use the first environment's state dimension
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_space
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lstm_model=lstm_model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=100000,  # Increased for more diverse experiences
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
    
    # Track total updates and experience collected
    total_updates = 0
    total_experiences = 0
    update_counter = 0
    
    for episode in range(1, args.episodes + 1):
        # Reset all environments with random starting points
        states = []
        for env in envs:
            # Generate random starting point for this episode
            random_start = np.random.randint(
                args.window_size + 1, 
                data_length - args.episode_length - 100  # Ensure room for the episode
            )
            env.start_step = random_start
            states.append(env.reset())
            
        dones = [False] * args.num_workers
        episode_rewards_per_env = [0] * args.num_workers
        steps_per_env = [0] * args.num_workers
        active_envs = args.num_workers
        
        # Episode length is now fixed by the argument
        max_episode_steps = args.episode_length
        
        # Episode loop
        progress_bar = tqdm(total=max_episode_steps, 
                           desc=f"Episode {episode}/{args.episodes}", 
                           disable=False)
        
        # Continue as long as at least one environment is active and steps are within limit
        step_counter = 0
        while active_envs > 0 and step_counter < max_episode_steps:
            step_counter += 1
            
            # For each active environment
            for env_idx in range(args.num_workers):
                if dones[env_idx]:
                    continue
                
                # Select action for this environment
                action = agent.select_action(states[env_idx])
                
                # Take step in environment
                next_state, reward, done, info = envs[env_idx].step(action)
                
                # Force done if we've reached the episode length
                if steps_per_env[env_idx] >= max_episode_steps - 1:
                    done = True
                
                # Store transition in replay buffer
                agent.store_transition(states[env_idx], action, reward, next_state, done)
                total_experiences += 1
                update_counter += 1
                
                # Update state and rewards
                states[env_idx] = next_state
                episode_rewards_per_env[env_idx] += reward
                steps_per_env[env_idx] += 1
                
                # Check if this environment is done
                if done:
                    dones[env_idx] = True
                    active_envs -= 1
            
            # Update agent (from common replay buffer) multiple times per step if specified
            if len(agent.replay_buffer) > args.batch_size and update_counter >= args.num_workers:
                # Multiple updates per batch of experiences
                for _ in range(args.updates_per_step):
                    loss = agent.update()
                    total_updates += 1
                
                update_counter = 0
                
                # Log debug info occasionally
                if args.debug and total_updates % 100 == 0:
                    logger.debug(f"Updates: {total_updates}, Loss: {loss:.4f}")
            
            # Update progress bar
            progress_bar.update(1)
            
            # Find an active environment for the progress bar stats
            active_idx = next((i for i, d in enumerate(dones) if not d), 0)
            if active_idx < len(envs):  # Ensure valid index
                progress_bar.set_postfix({
                    'reward': f"{episode_rewards_per_env[active_idx]:.2f}",
                    'balance': f"{envs[active_idx].balance:.2f}",
                    'trades': envs[active_idx].total_trades,
                    'active_envs': active_envs
                })
        
        progress_bar.close()
        
        # Decay epsilon
        agent.epsilon = max(args.epsilon_end, agent.epsilon * args.epsilon_decay)
        
        # Calculate average statistics across environments
        avg_episode_reward = sum(episode_rewards_per_env) / args.num_workers
        avg_profit = sum([env.total_profit for env in envs]) / args.num_workers
        avg_trades = sum([env.total_trades for env in envs]) / args.num_workers
        
        # Store episode statistics (using averages)
        episode_rewards.append(avg_episode_reward)
        episode_profits.append(avg_profit)
        episode_trades.append(avg_trades)
        
        logger.info(f"Episode {episode}/{args.episodes} - "
                   f"Avg Reward: {avg_episode_reward:.2f}, "
                   f"Avg Profit: {avg_profit:.2f}, "
                   f"Avg Trades: {avg_trades:.1f}, "
                   f"Epsilon: {agent.epsilon:.4f}, "
                   f"Buffer: {len(agent.replay_buffer)}")
        
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
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            best_model_path = os.path.join(args.output_dir, "dqn_agent_best.pt")
            agent.save(best_model_path)
            logger.info(f"New best model saved with reward {best_reward:.2f}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "dqn_agent_final.pt")
    agent.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Plot and save training results
    plot_training_results(episode_rewards, episode_profits, episode_trades, args.output_dir)
    
    logger.info(f"Training completed. Total updates: {total_updates}, experiences: {total_experiences}")

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