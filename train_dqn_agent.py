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
import traceback
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.models.time_series.model import MultiTimeframeModel

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
    
    # Performance options
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for faster training on compatible GPUs')
    
    # Saving options
    parser.add_argument('--save_frequency', type=int, default=50,
                        help='Frequency to save model checkpoints (episodes)')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (including stop loss messages)')
    
    return parser.parse_args()

def load_lstm_model(model_path, device):
    """Load the trained LSTM model."""
    try:
        logger.info(f"Loading LSTM model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found at {model_path}")
        
        # Load model configuration
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Loaded model configuration from config.json")
        else:
            logger.warning("No config.json found, using default configuration")
            config = {
                "model": {
                    "hidden_dims": 64,
                    "num_layers": 1,
                    "dropout": 0.7,
                    "bidirectional": True,
                    "attention": False,
                    "num_classes": 3,
                    "use_batch_norm": True
                }
            }
        
        # Create a new model instance with the saved configuration
        model = MultiTimeframeModel(
            input_dims={"15m": 34, "4h": 34, "1d": 34},  # Match dataset feature count
            hidden_dims=config["model"]["hidden_dims"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            bidirectional=config["model"]["bidirectional"],
            attention=config["model"]["attention"],
            num_classes=config["model"]["num_classes"],
            use_batch_norm=config["model"].get("use_batch_norm", True)
        )
        
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle missing/extra keys
        logger.info("Model state dict loaded successfully")
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        logger.info("LSTM model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading LSTM model: {str(e)}")
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
    
    # AMP verification
    if args.use_amp:
        logger.info("AMP requested via command line")
        if device == "cuda" and torch.cuda.is_available():
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "unknown"
            logger.info(f"CUDA is available (version: {cuda_version}) - AMP should work")
        else:
            logger.warning("AMP requested but CUDA not available - AMP will be disabled")
    
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
            reward_scaling=0.01,  # Increased from 0.001 for more meaningful rewards
            device=device,
            trade_cooldown=args.trade_cooldown,
            start_step=None,
            verbose=args.verbose
        )
        envs.append(env)
    
    # Use the first environment's state dimension
    state_dim = envs[0].state_dim
    action_dim = envs[0].action_space
    
    # Create DQN agent with optimized parameters
    logger.info(f"Creating DQN agent with AMP: {args.use_amp}")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=100000,
        batch_size=2048,  # Reduced from 8192 to 2048
        device=device,
        target_update=100,  # Changed from update_target_frequency to target_update
        verbose=args.verbose  # Pass verbose flag to control logging
    )
    
    # Verify AMP setup
    if hasattr(agent, 'scaler') and agent.scaler is not None:
        logger.info("AMP GradScaler initialized - Mixed precision training ENABLED")
    elif args.use_amp:
        logger.warning("AMP was requested but GradScaler not initialized - check if CUDA is available")
    
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
    
    # Pre-allocate tensors for states to avoid repeated allocations
    states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    actions_tensor = torch.zeros(args.num_workers, dtype=torch.long, device=device)
    rewards_tensor = torch.zeros(args.num_workers, device=device)
    next_states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    dones_tensor = torch.zeros(args.num_workers, dtype=torch.bool, device=device)
    
    # Pre-allocate replay buffer tensors
    replay_states = torch.zeros((args.batch_size, state_dim), device=device)
    replay_actions = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    replay_rewards = torch.zeros(args.batch_size, device=device)
    replay_next_states = torch.zeros((args.batch_size, state_dim), device=device)
    replay_dones = torch.zeros(args.batch_size, dtype=torch.bool, device=device)
    
    # Enable CUDA graph for faster training
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    for episode in range(1, args.episodes + 1):
        # Reset all environments with random starting points
        states = []
        for env in envs:
            random_start = np.random.randint(
                args.window_size + 1, 
                data_length - args.episode_length - 100
            )
            env.start_step = random_start
            states.append(env.reset())
            
        # Convert states to tensor for faster processing
        for i, state in enumerate(states):
            if isinstance(state, np.ndarray):
                states_tensor[i] = torch.from_numpy(state).float()
            else:
                states_tensor[i] = state
                
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
        
        step_counter = 0
        while active_envs > 0 and step_counter < max_episode_steps:
            # Select actions for all environments at once using vectorized operations
            with torch.no_grad():
                q_values = agent.policy_net(states_tensor)
                actions = torch.where(
                    torch.rand(args.num_workers, device=device) < agent.epsilon,
                    torch.randint(0, action_dim, (args.num_workers,), device=device),
                    q_values.argmax(dim=1)
                )
            
            # Step all environments
            next_states = []
            rewards = []
            new_dones = []
            balances = []
            trade_counts = []
            
            for i in range(args.num_workers):
                if not dones[i]:
                    next_state, reward, done, info = envs[i].step(actions[i].item())
                    next_states.append(next_state)
                    rewards.append(reward)
                    new_dones.append(done)
                    balances.append(info.get('balance', 0))
                    trade_counts.append(info.get('trades', 0))
                    
                    episode_rewards_per_env[i] += reward
                    steps_per_env[i] += 1
                    
                    if isinstance(next_state, np.ndarray):
                        next_states_tensor[i] = torch.from_numpy(next_state).float()
                    else:
                        next_states_tensor[i] = next_state
                else:
                    next_states.append(states[i])
                    rewards.append(0)
                    new_dones.append(True)
                    balances.append(0)
                    trade_counts.append(0)
            
            # Store transitions in replay buffer using vectorized operations
            if len(agent.memory) >= agent.batch_size:
                # Sample batch indices
                indices = np.random.choice(len(agent.memory), args.batch_size, replace=False)
                batch = [agent.memory[i] for i in indices]
                
                # Convert batch to tensors efficiently
                for i, (state, action, reward, next_state, done) in enumerate(batch):
                    replay_states[i] = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state
                    replay_actions[i] = action
                    replay_rewards[i] = reward
                    replay_next_states[i] = torch.from_numpy(next_state).float() if isinstance(next_state, np.ndarray) else next_state
                    replay_dones[i] = done
                
                # Update networks using vectorized operations
                with torch.cuda.amp.autocast():
                    current_q_values = agent.policy_net(replay_states).gather(1, replay_actions.unsqueeze(1))
                    with torch.no_grad():
                        next_q_values = agent.target_net(replay_next_states).max(1)[0]
                        expected_q_values = replay_rewards + (1 - replay_dones.float()) * agent.gamma * next_q_values
                    
                    loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
                
                # Optimize
                agent.optimizer.zero_grad()
                if agent.scaler is not None:
                    agent.scaler.scale(loss).backward()
                    agent.scaler.step(agent.optimizer)
                    agent.scaler.update()
                else:
                    loss.backward()
                    agent.optimizer.step()
                
                total_updates += 1
                update_counter += 1
            
            # Store transitions in replay buffer
            for i in range(args.num_workers):
                if not dones[i]:
                    agent.memory.append((states[i], actions[i].item(), rewards[i], next_states[i], new_dones[i]))
                    total_experiences += 1
            
            # Update states and dones
            states = next_states
            dones = new_dones
            states_tensor.copy_(next_states_tensor)
            
            # Count active environments
            active_envs = sum(1 for d in dones if not d)
            
            # Update progress bar with average metrics
            avg_reward = sum(r for i, r in enumerate(rewards) if not dones[i]) / max(1, active_envs)
            avg_balance = sum(b for i, b in enumerate(balances) if not dones[i]) / max(1, active_envs)
            avg_trades = sum(t for i, t in enumerate(trade_counts) if not dones[i]) / max(1, active_envs)
            
            progress_bar.set_postfix({
                'reward': f"{avg_reward:.2f}",
                'balance': f"{avg_balance:.2f}",
                'trades': f"{avg_trades:.0f}",
                'active_envs': active_envs,
                'updates': total_updates
            })
            progress_bar.update(1)
            
            step_counter += 1
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate episode metrics
        episode_reward = sum(episode_rewards_per_env) / args.num_workers
        episode_profit = sum(env.balance - args.initial_balance for env in envs) / args.num_workers
        episode_trade_count = sum(env.info.get('trades', 0) if hasattr(env, 'info') else 0 for env in envs) / args.num_workers
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_profits.append(episode_profit)
        episode_trades.append(episode_trade_count)
        
        # Log episode results
        logger.info(
            f"Episode {episode}/{args.episodes} - "
            f"Avg Reward: {episode_reward:.2f}, "
            f"Avg Profit: {episode_profit:.2f}, "
            f"Avg Trades: {episode_trade_count:.1f}, "
            f"Epsilon: {agent.epsilon:.4f}, "
            f"Buffer: {len(agent.memory)}"
        )
        
        # Save model if it's the best so far
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.output_dir, 'dqn_agent_best.pt'))
            logger.info(f"New best model saved with reward {episode_reward:.2f}")
        
        # Save checkpoint periodically
        if episode % args.save_frequency == 0:
            agent.save(os.path.join(args.output_dir, f'dqn_agent_episode_{episode}.pt'))
            logger.info(f"Checkpoint saved at episode {episode}")
    
    # Save final model
    agent.save(os.path.join(args.output_dir, 'dqn_agent_final.pt'))
    logger.info("Training completed. Final model saved.")
    
    # Plot training metrics
    plot_training_metrics(
        episode_rewards, 
        episode_profits, 
        episode_trades, 
        args.output_dir
    )
    
    return agent

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