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
from concurrent.futures import ThreadPoolExecutor
import psutil

from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds
from crypto_trading_model.data_loaders import load_crypto_data

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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
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
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Set device
    use_cuda = torch.cuda.is_available() and (args.device == 'cuda' or args.use_amp)
    device = "cuda" if use_cuda else "cpu"
    logger.info(f"Using device: {device}")
    
    # Advanced GPU optimizations
    if device == "cuda":
        # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} with {gpu_memory:.1f} GB memory")

    # Load data from specified path
    data_path = args.data_path
    logger.info(f"Loading data from {data_path}")
    market_data = load_crypto_data(data_path)
    data_length = len(market_data['1h'])  # Use 1h as the primary timeframe
    logger.info(f"Data loaded, {data_length} samples")

    # Create multiple environments for parallel training
    logger.info(f"Creating {args.num_workers} environments")
    envs = []
    for i in range(args.num_workers):
        env = TradingEnvironment(
            market_data=market_data,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            window_size=args.window_size,
            primary_timeframe=args.primary_timeframe,
            trade_cooldown=args.trade_cooldown,
            use_indicators=args.use_indicators,
            use_position_features=args.use_position_features,
            lookback_window=args.lookback_window
        )
        envs.append(env)
    
    # Get state and action dimensions
    state_dim = envs[0]._calculate_state_dim()
    action_dim = 3  # hold, buy, sell
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create DQN agent
    logger.info("Creating DQN agent")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[512, 256], # Increase network capacity
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=device,
        verbose=True
    )
    
    # Use torch.compile for faster network computations if available
    if hasattr(torch, 'compile') and device == "cuda":
        logger.info("Using torch.compile to optimize networks")
        agent.policy_net = torch.compile(agent.policy_net, mode="reduce-overhead")
        agent.target_net = torch.compile(agent.target_net, mode="reduce-overhead")
    
    # Initialize metrics
    episode_rewards = []
    episode_balances = []
    episode_trade_counts = []
    
    # Save directory
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Model checkpoints will be saved to {save_dir}")
    
    # Training loop
    logger.info("Starting training...")
    total_updates = 0
    total_experiences = 0
    update_counter = 0
    
    # Pre-allocate tensors for states to avoid repeated allocations
    states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    actions_tensor = torch.zeros(args.num_workers, dtype=torch.long, device=device)
    rewards_tensor = torch.zeros(args.num_workers, device=device)
    next_states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    dones_tensor = torch.zeros(args.num_workers, dtype=torch.bool, device=device)
    
    # Pre-allocate replay buffer tensors with dynamic sizing capability
    def get_optimal_batch_size():
        """Dynamically determine optimal batch size based on available memory"""
        if device == "cuda":
            # Get available GPU memory (in bytes)
            free_mem = torch.cuda.mem_get_info()[0]
            # Estimate memory per sample (state_dim * 4 bytes per float32)
            mem_per_sample = state_dim * 4 * 4  # states, actions, rewards, next_states
            # Use up to 30% of free memory for batches
            return min(args.batch_size, max(64, int(free_mem * 0.3 / mem_per_sample)))
        else:
            # On CPU, consider system memory
            free_mem = psutil.virtual_memory().available
            mem_per_sample = state_dim * 4 * 4
            return min(args.batch_size, max(64, int(free_mem * 0.3 / mem_per_sample)))
    
    optimal_batch_size = get_optimal_batch_size()
    logger.info(f"Using optimal batch size: {optimal_batch_size}")
    
    # Allocate with optimal batch size
    replay_states = torch.zeros((optimal_batch_size, state_dim), device=device)
    replay_actions = torch.zeros(optimal_batch_size, dtype=torch.long, device=device)
    replay_rewards = torch.zeros(optimal_batch_size, device=device)
    replay_next_states = torch.zeros((optimal_batch_size, state_dim), device=device)
    replay_dones = torch.zeros(optimal_batch_size, dtype=torch.bool, device=device)
    
    # Helper function to step environments in parallel
    def step_environments_in_parallel(envs, actions, dones):
        """Execute steps across multiple environments in parallel"""
        next_states = []
        rewards = []
        new_dones = []
        infos = []
        
        # Only process active environments
        active_indices = [i for i, done in enumerate(dones) if not done]
        
        # Use threads to parallelize environment steps
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(active_indices))) as executor:
            futures = []
            for i in active_indices:
                futures.append(executor.submit(envs[i].step, actions[i].item()))
            
            # Collect results
            results = [future.result() for future in futures]
            
            # Distribute results back to full lists
            result_idx = 0
            for i in range(len(envs)):
                if i in active_indices:
                    ns, r, d, info = results[result_idx]
                    next_states.append(ns)
                    rewards.append(r)
                    new_dones.append(d)
                    infos.append(info)
                    result_idx += 1
                else:
                    # Keep existing state for done environments
                    next_states.append(None)
                    rewards.append(0)
                    new_dones.append(True)
                    infos.append({})
        
        return next_states, rewards, new_dones, infos
    
    # Enable CUDA graph for faster training
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Determine optimal update frequency
    update_frequency = 4  # Update every 4 steps to batch updates
    
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
        
        # Episode loop
        progress_bar = tqdm(total=args.episode_length, 
                           desc=f"Episode {episode}/{args.episodes}", 
                           disable=False)
        
        step_counter = 0
        while active_envs > 0 and step_counter < args.episode_length:
            # Select actions for all environments at once using vectorized operations
            with torch.no_grad():
                q_values = agent.policy_net(states_tensor)
                actions = torch.where(
                    torch.rand(args.num_workers, device=device) < agent.epsilon,
                    torch.randint(0, action_dim, (args.num_workers,), device=device),
                    q_values.argmax(dim=1)
                )
            
            # Step all environments in parallel
            next_states, rewards, new_dones, infos = step_environments_in_parallel(
                envs, actions, dones
            )
            
            # Process environment step results
            balances = []
            trade_counts = []
            
            for i in range(args.num_workers):
                if not dones[i]:
                    balances.append(infos[i].get('balance', 0))
                    # Get trades from the environment's info dictionary
                    trade_counts.append(infos[i].get('total_trades', 0))
                    
                    episode_rewards_per_env[i] += rewards[i]
                    steps_per_env[i] += 1
                    
                    if isinstance(next_states[i], np.ndarray):
                        next_states_tensor[i] = torch.from_numpy(next_states[i]).float()
                    else:
                        next_states_tensor[i] = next_states[i]
                else:
                    balances.append(0)
                    trade_counts.append(0)
            
            # Log balance changes for debugging
            if step_counter % 100 == 0 and active_envs > 0:
                i = next((i for i, d in enumerate(dones) if not d), 0)
                logger.info(f"Environment {i} - Step {step_counter}: Action={actions[i].item()}, "
                           f"Balance={infos[i].get('balance', 0):.2f}, "
                           f"Position={infos[i].get('position', 0)}, "
                           f"Trades={infos[i].get('total_trades', 0)}, "
                           f"Price={infos[i].get('price', 0):.2f}")
            
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
            
            # Batch updates every few steps for better efficiency
            if step_counter % update_frequency == 0 and len(agent.memory) >= optimal_batch_size:
                # Perform multiple updates in a batch
                for _ in range(update_frequency * max(1, args.updates_per_step)):
                    # Dynamically adjust batch size based on available replay buffer samples
                    current_batch_size = min(len(agent.memory), optimal_batch_size)
                    
                    # Sample batch indices - ensure we don't sample more than available items
                    indices = np.random.choice(len(agent.memory), current_batch_size, replace=False)
                    batch = [agent.memory[i] for i in indices]
                    
                    # Convert batch to tensors efficiently
                    for i, (state, action, reward, next_state, done) in enumerate(batch):
                        replay_states[i] = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state
                        replay_actions[i] = action
                        replay_rewards[i] = reward
                        replay_next_states[i] = torch.from_numpy(next_state).float() if isinstance(next_state, np.ndarray) else next_state
                        replay_dones[i] = done
                    
                    # Update networks using vectorized operations
                    with torch.amp.autocast('cuda'):
                        current_q_values = agent.policy_net(replay_states[:current_batch_size]).gather(1, replay_actions[:current_batch_size].unsqueeze(1))
                        with torch.no_grad():
                            next_q_values = agent.target_net(replay_next_states[:current_batch_size]).max(1)[0]
                            expected_q_values = replay_rewards[:current_batch_size] + (1 - replay_dones[:current_batch_size].float()) * agent.gamma * next_q_values
                        
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
            
            # Calculate portfolio values (including unrealized gains/losses)
            portfolio_values = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    # Get current price and calculate portfolio value
                    current_price = env._get_current_price()
                    portfolio_value = env._calculate_portfolio_value(current_price)
                    portfolio_values.append(portfolio_value)
            
            avg_portfolio = sum(portfolio_values) / max(1, len(portfolio_values)) if portfolio_values else 0
            
            # Update progress bar with average metrics
            avg_reward = sum(episode_rewards_per_env) / max(1, active_envs)
            avg_balance = sum(balances) / max(1, active_envs) if balances else 0
            avg_trades = sum(trade_counts) / max(1, active_envs) if trade_counts else 0
            
            gpu_util = ""
            if device == "cuda" and step_counter % 50 == 0:
                # Get GPU utilization percentage
                try:
                    gpu_mem = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory * 100
                    gpu_util = f", GPU:{gpu_mem:.1f}%"
                except:
                    pass
            
            progress_bar.set_postfix({
                'reward': f"{avg_reward:.2f}",
                'balance': f"{avg_balance:.2f}",
                'portfolio': f"{avg_portfolio:.2f}",
                'trades': f"{avg_trades:.0f}",
                'active_envs': active_envs,
                'updates': total_updates,
                'info': f"bs:{current_batch_size}{gpu_util}"
            })
            progress_bar.update(1)
            
            step_counter += 1
        
        progress_bar.close()
        
        # Episode summary
        episode_reward = sum(episode_rewards_per_env) / args.num_workers
        episode_profit = sum(env.balance - args.initial_balance for env in envs) / args.num_workers
        episode_trade_count = sum(env.info.get('total_trades', 0) if hasattr(env, 'info') else 0 for env in envs) / args.num_workers
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_balances.append(episode_profit)
        episode_trade_counts.append(episode_trade_count)
        
        # Update networks and exploration
        if episode % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            logger.info(f"Target network updated at episode {episode}")
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Save model
        if episode % args.save_interval == 0 or episode == args.episodes:
            model_path = os.path.join(save_dir, f"dqn_agent_episode_{episode}.pt")
            state_dict = {
                'policy_net': agent.policy_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'episode': episode,
                'epsilon': agent.epsilon,
            }
            torch.save(state_dict, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_balances': episode_balances,
                'episode_trade_counts': episode_trade_counts,
            }
            metrics_path = os.path.join(save_dir, f"metrics_episode_{episode}.pt")
            torch.save(metrics, metrics_path)
        
        # Log episode results
        logger.info(f"Episode {episode}/{args.episodes} - "
                   f"Reward: {episode_reward:.2f}, Profit: {episode_profit:.2f}, "
                   f"Trades: {episode_trade_count:.2f}, Epsilon: {agent.epsilon:.4f}, "
                   f"Memory: {len(agent.memory)}, Updates: {total_updates}")
        
        # Evaluate model every few episodes
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            # TODO: Implement evaluation
            pass
    
    logger.info("Training completed!")
    return episode_rewards, episode_balances, episode_trade_counts

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