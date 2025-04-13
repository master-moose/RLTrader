#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN and PPO Agent Training for Cryptocurrency Trading

This script trains DQN and PPO agents for cryptocurrency trading using parallel environments
with safeguards against rapid trading and action oscillation issues.

Ensures compatibility with:
- FinRL 0.3.7+
- Gymnasium and Gym
- Stable Baselines 3
"""

import os
import sys
import time
import logging
import argparse
import random
import numpy as np
import pandas as pd
import torch
import gymnasium
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
from crypto_trading_model.environment.crypto_env import CryptocurrencyTradingEnv
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define technical indicators to ensure they're included in the environment
INDICATORS = ['macd', 'rsi', 'cci', 'dx', 'bb_upper', 'bb_lower', 'bb_middle', 'volume']

# Constants for trading safeguards
TRADE_COOLDOWN_PERIOD = 50  # Minimum steps between trades
OSCILLATION_PENALTY = 100.0  # Penalty for oscillating between buys and sells
SAME_PRICE_TRADE_PENALTY = 200.0  # Penalty for trading at same price

class SafeTradingEnvWrapper(gymnasium.Wrapper):
    """
    A wrapper for trading environments that adds safeguards against:
    1. Rapid trading (enforces a cooldown period)
    2. Action oscillation (penalizes rapid flipping between positions)
    3. Same-price trading (penalizes trades at the same price)
    
    This wrapper supports both Gymnasium and older Gym environments.
    """
    
    def __init__(self, env, trade_cooldown=TRADE_COOLDOWN_PERIOD, max_history_size=100):
        """Initialize the wrapper with safeguards against harmful trading patterns"""
        super().__init__(env)
        
        # Trading safeguards
        self.trade_cooldown = trade_cooldown
        self.max_history_size = max_history_size
        
        # Trading history tracking
        self.last_trade_step = -self.trade_cooldown  # Start with cooldown already passed
        self.last_trade_price = None
        self.action_history = []
        self.position_history = []
        self.same_price_trades = 0
        self.cooldown_violations = 0
        self.oscillation_count = 0
        
        # State tracking
        self.previous_position = 0
        self.current_position = 0
        self.forced_actions = 0
        
        logger.info(f"SafeTradingEnvWrapper initialized with {trade_cooldown} step cooldown")
    
    def reset(self, **kwargs):
        """Reset the environment and all trading history"""
        observation, info = self.env.reset(**kwargs)
        
        # Reset tracking variables
        self.last_trade_step = -self.trade_cooldown
        self.last_trade_price = None
        self.action_history = []
        self.position_history = []
        self.previous_position = 0
        self.current_position = 0
        self.forced_actions = 0
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment with added safeguards
        
        Parameters:
        -----------
        action : int
            Action to take (0: sell, 1: hold, 2: buy)
        
        Returns:
        --------
        observation : numpy.ndarray
            Observation from the environment
        reward : float
            Reward with penalties applied if needed
        terminated : bool
            Whether episode is done
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        # Track step number from parent class if available
        current_step = getattr(self.env, 'day', 0)
        if hasattr(self.env, 'current_step'):
            current_step = self.env.current_step
        
        # Store original action for tracking
        original_action = action
        
        # Determine if we're in a trade cooldown period
        in_cooldown = (current_step - self.last_trade_step) < self.trade_cooldown
        attempted_trade_during_cooldown = False
        
        if in_cooldown and action != 1:  # Not a hold action
            # Agent is trying to trade during cooldown
            attempted_trade_during_cooldown = True
            # Force a hold action instead
            action = 1
            self.forced_actions += 1
            
            if self.forced_actions % 10 == 0:  # Log periodically to avoid spamming
                logger.warning(f"Forced hold action during cooldown at step {current_step}, " 
                              f"{current_step - self.last_trade_step}/{self.trade_cooldown} steps since last trade")
        
        # Check for oscillation in action history
        if len(self.action_history) >= 4:
            # Look for buy-sell-buy-sell (2-0-2-0) or sell-buy-sell-buy (0-2-0-2) patterns
            last_four = self.action_history[-4:]
            if last_four == [2, 0, 2, 0] or last_four == [0, 2, 0, 2]:
                logger.warning(f"Detected action oscillation at step {current_step}: {last_four}")
                # Force a hold action and prepare for penalty
                action = 1
                self.oscillation_count += 1
        
        # Take the step in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current price and position after the step
        current_price = None
        current_position = 0
        
        # Extract the current price from the environment if available
        if hasattr(self.env, '_get_current_price'):
            current_price = self.env._get_current_price()
        elif 'close_price' in info:
            current_price = info['close_price']
        elif hasattr(self.env, 'current_price'):
            current_price = self.env.current_price
            
        # Extract current position from the environment if available
        if hasattr(self.env, 'position'):
            current_position = self.env.position
        elif 'position' in info:
            current_position = info['position']
        
        # Record the action taken
        self.action_history.append(original_action)
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]
        
        # Record position
        self.previous_position = self.current_position
        self.current_position = current_position
        self.position_history.append(current_position)
        if len(self.position_history) > self.max_history_size:
            self.position_history = self.position_history[-self.max_history_size:]
        
        # Detect if a trade occurred by checking position change
        trade_occurred = self.previous_position != self.current_position
        
        # Apply penalties for concerning trading patterns
        additional_penalty = 0.0
        
        # Penalty for attempted trades during cooldown
        if attempted_trade_during_cooldown:
            cooldown_penalty = OSCILLATION_PENALTY * (0.5 + min(self.cooldown_violations, 10) / 10)
            additional_penalty -= cooldown_penalty
            self.cooldown_violations += 1
            if self.cooldown_violations % 5 == 0:
                logger.warning(f"Applied cooldown violation penalty: {cooldown_penalty:.2f} " 
                              f"(#{self.cooldown_violations})")
            
            # Add cooldown violation to info dict
            info['cooldown_violation'] = True
            info['cooldown_penalty'] = cooldown_penalty
        
        # Penalty for action oscillation
        if len(self.action_history) >= 4:
            last_four = self.action_history[-4:]
            if last_four == [2, 0, 2, 0] or last_four == [0, 2, 0, 2]:
                oscillation_penalty = OSCILLATION_PENALTY * (1.0 + min(self.oscillation_count, 5))
                additional_penalty -= oscillation_penalty
                logger.warning(f"Applied oscillation penalty: {oscillation_penalty:.2f} (#{self.oscillation_count})")
                
                # Add oscillation detection to info dict
                info['oscillation_detected'] = True
                info['oscillation_penalty'] = oscillation_penalty
        
        # Penalty for same-price trades
        if trade_occurred and current_price is not None and self.last_trade_price is not None:
            # Check if trade price is very close to the last trade price
            if abs(current_price - self.last_trade_price) < 0.0001:
                self.same_price_trades += 1
                same_price_penalty = SAME_PRICE_TRADE_PENALTY * (1.0 + min(self.same_price_trades, 5))
                additional_penalty -= same_price_penalty
                logger.warning(f"Same price trade detected! Penalty: {same_price_penalty:.2f} (#{self.same_price_trades})")
                
                # Add same-price trade to info dict
                info['same_price_trade'] = True
                info['same_price_penalty'] = same_price_penalty
        
        # If a trade occurred, update the last trade step and price
        if trade_occurred:
            self.last_trade_step = current_step
            if current_price is not None:
                self.last_trade_price = current_price
            
            # Reset cooldown violation count after successful trade
            if self.cooldown_violations > 0:
                logger.info(f"Resetting cooldown violation count from {self.cooldown_violations} to 0")
                self.cooldown_violations = 0
        
        # Apply any penalties to the reward
        if additional_penalty != 0:
            reward += additional_penalty
            info['additional_penalty'] = additional_penalty
        
        # Add cooldown information to info dict
        info['steps_since_last_trade'] = current_step - self.last_trade_step
        info['in_cooldown'] = in_cooldown
        
        return observation, reward, terminated, truncated, info


class TensorboardCallback(BaseCallback):
    """Custom callback for tracking metrics during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Environment metrics
        self.returns = []
        self.portfolio_values = []
        self.episode_rewards = []
        self.trade_count = 0
        self.successful_trades = 0
        
        # Trading metrics
        self.trade_returns = []
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Sell, Hold, Buy
        
        # Safety metrics
        self.cooldown_violations = 0
        self.oscillation_count = 0
        self.same_price_trades = 0
        
        # Debug counter
        self.debug_steps = 0
        self.debug_frequency = 1000
        self.last_debug_output = 0
    
    def _on_step(self) -> bool:
        """Called at each step"""
        self.debug_steps += 1
        
        # Debug output every N steps
        if self.debug_steps % self.debug_frequency == 0 and self.debug_steps > self.last_debug_output:
            self.last_debug_output = self.debug_steps
            logger.info(f"Training progress at step {self.debug_steps}")
        
        # Log rewards when episodes complete
        if self.locals is not None and 'dones' in self.locals and 'rewards' in self.locals:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    reward = self.locals['rewards'][i]
                    self.episode_rewards.append(reward)
                    self.logger.record('environment/reward', reward)
                    
                    # If info dict available, extract additional metrics
                    if 'infos' in self.locals and len(self.locals['infos']) > i:
                        info = self.locals['infos'][i]
                        
                        # Track portfolio value
                        if 'portfolio_value' in info:
                            portfolio_value = info['portfolio_value']
                            self.portfolio_values.append(portfolio_value)
                            self.logger.record('portfolio/value', portfolio_value)
                        
                        # Track cooldown violations
                        if 'cooldown_violation' in info and info['cooldown_violation']:
                            self.cooldown_violations += 1
                            self.logger.record('safety/cooldown_violations', self.cooldown_violations)
                        
                        # Track oscillations
                        if 'oscillation_detected' in info and info['oscillation_detected']:
                            self.oscillation_count += 1
                            self.logger.record('safety/oscillation_count', self.oscillation_count)
                        
                        # Track same price trades
                        if 'same_price_trade' in info and info['same_price_trade']:
                            self.same_price_trades += 1
                            self.logger.record('safety/same_price_trades', self.same_price_trades)
                        
                        # Track actions if available
                        if 'action' in info:
                            action = info['action']
                            if action in self.action_counts:
                                self.action_counts[action] += 1
                                
                        # Track trades
                        if 'trade' in info and info['trade']:
                            self.trade_count += 1
                            self.logger.record('trades/count', self.trade_count)
                            
                            # Trade metrics if available
                            if 'trade_profit' in info:
                                profit = info['trade_profit']
                                self.trade_returns.append(profit)
                                
                                if profit > 0:
                                    self.successful_trades += 1
                                    self.total_profit += profit
                                else:
                                    self.total_loss += abs(profit)
                                
                                # Calculate win rate and profit factor
                                if self.trade_count > 0:
                                    win_rate = (self.successful_trades / self.trade_count) * 100
                                    self.logger.record('trades/win_rate', win_rate)
                                    
                                if self.total_loss > 0:
                                    profit_factor = self.total_profit / max(self.total_loss, 1e-6)
                                    self.logger.record('trades/profit_factor', profit_factor)
        
        return True


def setup_logging(log_dir=None):
    """Configure logging to file and console"""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    else:
        os.makedirs('logs', exist_ok=True)
        log_path = os.path.join('logs', f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Return path to log file for reference
    return log_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DQN or PPO agent for cryptocurrency trading")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of timesteps for training")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Model parameters
    parser.add_argument("--finrl_model", type=str, default="ppo", 
                        choices=["ppo", "a2c", "dqn"], help="Type of RL model to use")
    parser.add_argument("--use_finrl", action="store_true", help="Use FinRL library for training")
    parser.add_argument("--lstm_model_path", type=str, default=None, 
                        help="Path to pre-trained LSTM model for state representation")
    
    # Environment parameters
    parser.add_argument("--data_path", type=str, default="data/synthetic/synthetic_dataset.h5", 
                       help="Path to market data (HDF5 format)")
    parser.add_argument("--data_key", type=str, default=None,
                       help="Specific key/group to use when HDF5 file contains multiple datasets")
    parser.add_argument("--state_space", type=int, default=16, help="State space dimensionality")
    parser.add_argument("--initial_balance", type=float, default=10000.0, help="Initial portfolio balance")
    parser.add_argument("--transaction_cost_pct", type=float, default=0.001, 
                       help="Transaction cost percentage")
    parser.add_argument("--reward_scaling", type=float, default=0.01, help="Reward scaling factor")
    parser.add_argument("--normalize_observations", type=bool, default=True, 
                       help="Whether to normalize observations")
    
    # Parallelization parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--num_envs_per_worker", type=int, default=1, 
                       help="Number of environments per worker")
    
    # PPO-specific parameters
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    
    # DQN-specific parameters
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--exploration_fraction", type=float, default=0.1, 
                       help="Fraction of training time for exploration")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, 
                       help="Initial exploration rate")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, 
                       help="Final exploration rate")
    parser.add_argument("--target_update_interval", type=int, default=10000, 
                       help="Update frequency for target network")
    
    # Trading safeguards
    parser.add_argument("--trade_cooldown", type=int, default=50, 
                       help="Minimum steps between trades")
    
    # Misc
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--checkpoint_freq", type=int, default=100000, 
                       help="Frequency of model checkpoints")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (cuda, cpu, or auto to automatically detect)")
    
    args = parser.parse_args()
    
    # Auto-detect device if set to 'auto'
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def ensure_technical_indicators(df, indicators):
    """Ensure the dataframe has all required technical indicators"""
    missing_indicators = [ind for ind in indicators if ind not in df.columns]
    
    if missing_indicators:
        logger.info(f"Adding missing technical indicators: {missing_indicators}")
        import pandas_ta as ta
        
        # Calculate basic technical indicators if missing
        if 'macd' in missing_indicators:
            # MACD
            macd = ta.macd(df['close'])
            df = pd.concat([df, macd], axis=1)
        
        if 'rsi' in missing_indicators:
            # RSI
            df['rsi'] = ta.rsi(df['close'])
        
        if 'cci' in missing_indicators:
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        
        if 'dx' in missing_indicators:
            # DX (Directional Movement Index)
            dx = ta.adx(df['high'], df['low'], df['close'])
            df = pd.concat([df, dx], axis=1)
        
        if any(bb in missing_indicators for bb in ['bb_upper', 'bb_lower', 'bb_middle']):
            # Bollinger Bands
            bb = ta.bbands(df['close'])
            df = pd.concat([df, bb], axis=1)
    
    # Ensure no NaN values
    df = df.ffill().bfill().fillna(0)
    
    return df


def load_and_preprocess_market_data(args):
    """Load and preprocess market data for training"""
    data_path = args.data_path
    
    try:
        # Check if data is in HDF5 format
        if data_path.endswith('.h5'):
            logger.info(f"Loading market data from HDF5 file: {data_path}")
            
            # Try to load with key if specified
            if args.data_key:
                logger.info(f"Using specified key: {args.data_key}")
                market_data = pd.read_hdf(data_path, key=args.data_key)
            else:
                # Try loading directly first (for single dataset files)
                try:
                    market_data = pd.read_hdf(data_path)
                except ValueError as e:
                    if "key must be provided" in str(e):
                        # File has multiple datasets, list them and use the first one
                        import h5py
                        with h5py.File(data_path, 'r') as f:
                            keys = list(f.keys())
                            logger.info(f"File contains multiple datasets with keys: {keys}")
                            if keys:
                                first_key = keys[0]
                                logger.info(f"Using first key automatically: {first_key}")
                                market_data = pd.read_hdf(data_path, key=first_key)
                            else:
                                raise ValueError("HDF5 file doesn't contain any datasets")
                    else:
                        raise
        else:
            # Assume CSV format
            logger.info(f"Loading market data from CSV file: {data_path}")
            market_data = pd.read_csv(data_path)
            
            # Convert to datetime if there's a timestamp column
            if 'timestamp' in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                market_data.set_index('timestamp', inplace=True)
        
        # Ensure required technical indicators
        market_data = ensure_technical_indicators(market_data, INDICATORS)
        
        # Log data information
        logger.info(f"Market data loaded with shape: {market_data.shape}")
        logger.info(f"Sample of market data:\n{market_data.head(2)}")
        
        # Train/test split
        train_size = int(len(market_data) * 0.8)
        train_data = market_data[:train_size]
        
        # Return processed data
        return train_data
        
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def create_vec_env(df, args, num_envs=4):
    """Create a vectorized environment for parallel training"""
    
    def make_env(rank, seed=None):
        """Factory function to create a single environment"""
        def _init():
            # Create base environment
            env = CryptocurrencyTradingEnv(
                df=df,
                initial_amount=args.initial_balance,
                state_space=args.state_space,
                buy_cost_pct=args.transaction_cost_pct,
                sell_cost_pct=args.transaction_cost_pct,
                reward_scaling=args.reward_scaling,
                tech_indicator_list=list(set(INDICATORS).intersection(set(df.columns)))
            )
            
            # Add safeguards against rapid trading
            env = SafeTradingEnvWrapper(env, trade_cooldown=args.trade_cooldown)
            
            # Add Monitor wrapper for logging
            log_dir = os.path.join('logs', 'env_logs')
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, os.path.join(log_dir, f'env_{rank}'))
            
            # Set random seed if provided
            if seed is not None:
                env.seed(seed + rank)
                
            return env
        return _init
    
    # Create environment functions based on number of environments requested
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Create appropriate vectorized environment
    if num_envs == 1:
        # Use DummyVecEnv for single environment
        vec_env = DummyVecEnv(env_fns)
    else:
        # Use SubprocVecEnv for parallel environments
        vec_env = SubprocVecEnv(env_fns)
    
    # Add observation normalization if requested
    if args.normalize_observations:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    return vec_env


def load_lstm_model(model_path):
    """Load pre-trained LSTM model for state representation"""
    if model_path and os.path.exists(model_path):
        try:
            logger.info(f"Loading pre-trained LSTM model from {model_path}")
            
            # Load the saved model checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Extract model configuration or use defaults
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                logger.info(f"Using configuration from saved model: {model_config}")
            elif 'hparams' in checkpoint:
                # Handle Lightning checkpoint format
                model_config = {
                    'input_dims': checkpoint['hparams'].get('input_dims', {'15m': 39, '4h': 39, '1d': 39}),
                    'hidden_dims': checkpoint['hparams'].get('hidden_dims', 128),
                    'num_layers': checkpoint['hparams'].get('num_layers', 2),
                    'dropout': checkpoint['hparams'].get('dropout', 0.2),
                    'bidirectional': checkpoint['hparams'].get('bidirectional', True),
                    'attention': checkpoint['hparams'].get('attention', True),
                    'num_classes': checkpoint['hparams'].get('num_classes', 3)
                }
                logger.info(f"Using configuration from Lightning checkpoint: {model_config}")
            else:
                # Use default configuration
                model_config = {
                    'input_dims': {'15m': 39, '4h': 39, '1d': 39},
                    'hidden_dims': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'bidirectional': True,
                    'attention': True,
                    'num_classes': 3
                }
                logger.info(f"Using default model configuration: {model_config}")
            
            # Create a new model instance
            model = MultiTimeframeModel(**model_config)
            
            # Load the state dictionary
            if 'state_dict' in checkpoint:
                # Handle Lightning checkpoint format
                state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded model state from Lightning checkpoint")
            else:
                # Regular PyTorch model
                model.load_state_dict(checkpoint, strict=False)
                logger.info("Loaded model state from regular checkpoint")
            
            model.eval()  # Set to evaluation mode
            return model
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.warning("No valid LSTM model provided or found. Proceeding without LSTM.")
    return None


def train_dqn(env, args):
    """Train a DQN agent"""
    logger.info("Starting DQN training...")
    
    # Setup callbacks
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Add tensorboard callback
    tensorboard_callback = TensorboardCallback()
    callbacks.append(tensorboard_callback)
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        target_update_interval=args.target_update_interval,
        verbose=1 if args.verbose else 0,
        tensorboard_log="tensorboard_log",
        device=args.device
    )
    
    # Train model
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="dqn_run"
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"dqn_model_{int(time.time())}")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during DQN training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def train_ppo(env, args):
    """Train a PPO agent"""
    logger.info("Starting PPO training...")
    
    # Setup callbacks
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_model",
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Add tensorboard callback
    tensorboard_callback = TensorboardCallback()
    callbacks.append(tensorboard_callback)
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Create PPO model with custom policy kwargs and improved stability parameters
    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Smaller networks for stability
        activation_fn=torch.nn.Tanh,  # Tanh keeps values bounded between -1 and 1
        log_std_init=-2.0  # More conservative initial exploration
    )
    
    # Lower learning rate and use a scheduler for stability
    learning_rate = args.learning_rate
    if learning_rate > 0.001:
        logger.warning(f"Reducing learning rate from {learning_rate} to 0.001 for stability")
        learning_rate = 0.001
    
    # Calculate appropriate clip range based on action space
    clip_range = min(args.clip_range, 0.1)  # Conservative clipping
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=min(args.n_steps, 2048),  # Cap n_steps for stability
        batch_size=min(args.batch_size, 64),  # Smaller batch size for more stable updates
        n_epochs=max(args.n_epochs, 5),  # At least 5 epochs
        gamma=args.gamma,
        ent_coef=max(args.ent_coef, 0.01),  # Ensure sufficient exploration
        clip_range=clip_range,
        clip_range_vf=clip_range,  # Also clip value function for stability
        normalize_advantage=True,  # Normalize advantages for stability
        max_grad_norm=0.5,  # Add strong gradient clipping
        policy_kwargs=policy_kwargs,
        verbose=1 if args.verbose else 0,
        tensorboard_log="tensorboard_log",
        device=args.device
    )
    
    # Apply torch float32 precision for better numerical stability
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=10)
    
    # Log the training parameters
    logger.info(f"Training PPO with parameters:")
    logger.info(f" - Learning rate: {learning_rate}")
    logger.info(f" - Batch size: {model.batch_size}")
    logger.info(f" - n_steps: {model.n_steps}")
    logger.info(f" - clip_range: {clip_range}")
    logger.info(f" - max_grad_norm: 0.5")
    
    # Train model
    try:
        # Create a learning rate scheduler for stability
        def lr_schedule(remaining_progress):
            # Start with the base learning rate and decay to 10% of the initial value
            return learning_rate * (0.1 + 0.9 * remaining_progress)
        
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="ppo_run",
            progress_bar=True
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"ppo_model_{int(time.time())}")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during PPO training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def train_a2c(env, args):
    """Train an A2C agent"""
    logger.info("Starting A2C training...")
    
    # Setup callbacks
    checkpoint_dir = os.path.join('models', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="a2c_model",
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Add tensorboard callback
    tensorboard_callback = TensorboardCallback()
    callbacks.append(tensorboard_callback)
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Create model
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps // 4,  # A2C typically uses smaller n_steps
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1 if args.verbose else 0,
        tensorboard_log="tensorboard_log",
        device=args.device
    )
    
    # Train model
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="a2c_run"
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"a2c_model_{int(time.time())}")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during A2C training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_path = setup_logging()
    logger.info(f"Log file created at: {log_path}")
    
    # Log environment information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Gym/Gymnasium version: {gymnasium.__version__}")
    logger.info(f"Number of available CPUs: {psutil.cpu_count(logical=False)}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Set random seeds if provided
    if args.seed is not None:
        set_random_seed(args.seed)
        logger.info(f"Setting random seed to: {args.seed}")
    
    # Log key parameters
    logger.info(f"Training with model: {args.finrl_model}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Trade cooldown period: {args.trade_cooldown}")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('tensorboard_log', exist_ok=True)
    
    # Load market data
    market_data = load_and_preprocess_market_data(args)
    
    # Create vectorized environment
    total_envs = args.num_workers * args.num_envs_per_worker
    logger.info(f"Creating {total_envs} parallel environments")
    vec_env = create_vec_env(market_data, args, num_envs=total_envs)
    
    # Load pre-trained LSTM model if provided
    lstm_model = None
    if args.lstm_model_path:
        lstm_model = load_lstm_model(args.lstm_model_path)
    
    # Train the specified model
    trained_model = None
    
    try:
        if args.finrl_model.lower() == "dqn":
            trained_model = train_dqn(vec_env, args)
        elif args.finrl_model.lower() == "ppo":
            trained_model = train_ppo(vec_env, args)
        elif args.finrl_model.lower() == "a2c":
            trained_model = train_a2c(vec_env, args)
        else:
            logger.error(f"Unknown model type: {args.finrl_model}")
            return
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # Clean up resources
        vec_env.close()
        logger.info("Environment resources cleaned up")
        
        # Save any final models if training was interrupted
        if trained_model is not None:
            final_save_path = os.path.join('models', f"final_{args.finrl_model}_model")
            trained_model.save(final_save_path)
            logger.info(f"Final model saved to {final_save_path}")


if __name__ == "__main__":
    # Run the main function
    main()
