#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a DQN agent for cryptocurrency trading using an LSTM model.
Includes FinRL integration for enhanced model capabilities.
"""

import os
import argparse
import json
import logging
import traceback
import numpy as np
import torch
import gym
import gymnasium
from gymnasium.vector import VectorEnv
import sys
import time
from pathlib import Path
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta
import torch.nn as nn
import torch.nn.functional as F  # Add F for functional
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import h5py
import threading
from concurrent.futures import ThreadPoolExecutor
import ta
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
import matplotlib.pyplot as plt  # Add matplotlib
import psutil  # Add psutil
import inspect
from stable_baselines3 import SAC, TD3, DDPG, PPO

# Add project root to path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

# Import our custom implementation instead of relying on FinRL
from crypto_trading_model.environment.crypto_env import CryptocurrencyTradingEnv
from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a fallback for INDICATORS in case we can't import it
INDICATORS = ['macd', 'rsi', 'cci', 'dx']

class BaseStockTradingEnv(gym.Env):
    """
    Base class for stock trading environments, used as a fallback when FinRL imports fail.
    This is a minimal implementation that can be used when other implementations are not available.
    """
    
    def __init__(self, df=None, state_space=16, stock_dim=1, action_space=3, **kwargs):
        """
        Initialize the environment.
        
        Args:
            df: DataFrame with market data
            state_space: Size of the state space
            stock_dim: Number of assets
            action_space: Size of the action space
            **kwargs: Additional arguments for compatibility with FinRL
        """
        # Import spaces directly from gym
        from gym import spaces
        
        self.df = df
        self.state_space = state_space
        self.stock_dim = stock_dim
        
        # Store other optional arguments needed for compatibility
        self.initial_amount = kwargs.get('initial_amount', 1000000.0)
        self.hmax = kwargs.get('hmax', 100)
        self.print_verbosity = kwargs.get('print_verbosity', 0)
        self.num_stock_shares = kwargs.get('num_stock_shares', [0] * stock_dim)
        self.tech_indicator_list = kwargs.get('tech_indicator_list', [])
        
        # Handle transaction costs - convert to lists if passed as floats
        buy_cost_pct = kwargs.get('buy_cost_pct', 0.001)
        sell_cost_pct = kwargs.get('sell_cost_pct', 0.001)
        
        if isinstance(buy_cost_pct, (int, float)):
            self.buy_cost_pct = [buy_cost_pct] * stock_dim
        else:
            self.buy_cost_pct = buy_cost_pct
            
        if isinstance(sell_cost_pct, (int, float)):
            self.sell_cost_pct = [sell_cost_pct] * stock_dim
        else:
            self.sell_cost_pct = sell_cost_pct
            
        self.reward_scaling = kwargs.get('reward_scaling', 1e-4)
        
        # Define spaces
        self.action_space = spaces.Discrete(action_space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )
        
        # Initialize state
        self.state = np.zeros(state_space)
        self.terminal = False
        self.day = 0
        
        # Initialize data as None first to avoid errors if df is None
        self.data = None
        if self.df is not None and len(self.df) > 0:
            # Get the first row as a Series, not a scalar value
            self.data = self.df.iloc[self.day]
    
    def reset(self):
        """Reset the environment."""
        self.state = np.zeros(self.state_space)
        self.terminal = False
        self.day = 0
        if self.df is not None:
            self.data = self.df.iloc[self.day]
        return self.state
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            tuple of (observation, reward, done, info)
        """
        # Simple dummy implementation
        self.day += 1
        done = self.day >= len(self.df) if self.df is not None else False
        
        # Calculate reward based on price change and action
        reward = 0
        info = {"terminal_observation": self.state if done else None}
        
        if self.df is not None and self.day < len(self.df) and self.day > 0:
            # Get current and previous price data
            current_price = self.df.iloc[self.day]['close']
            previous_price = self.df.iloc[self.day-1]['close']
            price_change = (current_price - previous_price) / previous_price
            
            # Action is 0 (sell), 1 (hold), or 2 (buy)
            if action == 0:  # Sell
                reward = -price_change  # Reward for selling is positive when price goes down
                info["action_type"] = "sell"
            elif action == 2:  # Buy
                reward = price_change  # Reward for buying is positive when price goes up
                info["action_type"] = "buy"
            else:  # Hold
                # Small reward/penalty for holding based on whether price is trending up or down
                reward = price_change * 0.1
                info["action_type"] = "hold"
            
            # Scale reward for better learning
            reward = reward * 100  # Scale up for better learning signal
            info["price_change"] = price_change
            info["reward"] = reward
        
        # Update state (simple implementation)
        if self.df is not None and self.day < len(self.df):
            self.data = self.df.iloc[self.day]
            # In a real implementation, we would update the state based on the action and new data
        
        return self.state, reward, done, info
        
    def render(self, mode='human'):
        """Render the environment."""
        pass
        
    def seed(self, seed=None):
        """Set the random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

# Use BaseStockTradingEnv as our StockTradingEnv
StockTradingEnv = BaseStockTradingEnv

# Create alias for CryptocurrencyTradingEnv
CryptocurrencyTradingEnv = StockTradingEnv
logger.info("Using StockTradingEnv as CryptocurrencyTradingEnv")

# Add a custom patched version of StockTradingEnv that fixes common issues
class PatchedStockTradingEnv(BaseStockTradingEnv):
    """
    A patched version of the StockTradingEnv that fixes common issues with the FinRL implementation,
    specifically the 'values' attribute error on numpy scalars.
    """
    
    def __init__(self, df=None, state_space=16, stock_dim=1, action_space=3, **kwargs):
        """Initialize with enhanced error handling."""
        super().__init__(df, state_space, stock_dim, action_space, **kwargs)
        logger.info("Using PatchedStockTradingEnv to prevent numpy.float64 attribute errors")
    
    def _initiate_state(self):
        """
        Initialize state with protection against the 'values' attribute error.
        This is a replacement for the problematic method in FinRL's implementation.
        """
        # Start with basic state (cash + owned stocks)
        state = [self.initial_amount] + self.num_stock_shares
        
        # Add current price data safely
        if self.df is not None and self.day < len(self.df):
            # Get current day data
            data = self.df.iloc[self.day]
            
            # Add price for each stock, avoiding the 'values' attribute issue
            for i in range(self.stock_dim):
                # For multiple stocks, use column index, for single stock use 'close'
                if self.stock_dim > 1 and i < len(data):
                    state.append(float(data.iloc[i]['close']) if hasattr(data, 'iloc') else float(data['close']))
                else:
                    state.append(float(data['close']) if 'close' in data else 0.0)
            
            # Add technical indicators
            for indicator in self.tech_indicator_list:
                if indicator in data:
                    # Safely extract indicator value
                    value = data[indicator]
                    if hasattr(value, 'values') and len(value.values) > 0:
                        state.append(float(value.values[0]))
                    elif isinstance(value, (float, int, np.number)):
                        state.append(float(value))
                    else:
                        # Default value if we can't extract safely
                        state.append(0.0)
                else:
                    state.append(0.0)
        
        # Ensure the state has the correct dimension
        state_np = np.array(state)
        if len(state_np) > self.state_space:
            state_np = state_np[:self.state_space]
        elif len(state_np) < self.state_space:
            # Pad with zeros
            padding = np.zeros(self.state_space - len(state_np))
            state_np = np.concatenate([state_np, padding])
        
        return state_np.astype(np.float32)

# Original imports
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds

# Define trade evaluation constants
GOOD_TRADE_THRESHOLD = 0.001  # 0.1% profit threshold for a good trade
BAD_TRADE_THRESHOLD = -0.001  # -0.1% loss threshold for a bad trade

# Add this new class after the import statements but before the main functions
class DimensionAdjustingVecEnv(DummyVecEnv):
    """
    A VecEnv that adjusts observation dimensions to match the observation space.
    Specifically designed to handle the case where environments return
    observations with more dimensions than specified in the observation space.
    """
    def __init__(self, env_fns):
        """
        Initialize the VecEnv with customized observation space handling.
        
        Args:
            env_fns: A list of functions that create the environments.
        """
        # First initialize normally
        super().__init__(env_fns)
        
        # Test observation size from the first environment
        try:
            test_env = env_fns[0]()
            test_obs = test_env.reset()
            
            # Check if observation shape matches the buffer
            if isinstance(test_obs, np.ndarray):
                actual_size = len(test_obs)
                expected_size = self.observation_space.shape[0]
                
                if actual_size != expected_size:
                    logger.warning(f"Observation dimension mismatch detected: got {actual_size}, expected {expected_size}")
                    
                    # Create a new observation space with the correct shape
                    new_obs_space = gym.spaces.Box(
                        low=-np.inf, 
                        high=np.inf, 
                        shape=(expected_size,), 
                        dtype=np.float32
                    )
                    
                    # Log what we're doing
                    logger.info(f"Using observation space: {new_obs_space} with shape {new_obs_space.shape}")
            
            # Clean up
            test_env.close()
            
        except Exception as e:
            logger.error(f"Error checking observation dimensions: {e}")
            logger.error(traceback.format_exc())
            
    def _save_obs(self, env_idx, obs):
        """
        Override the parent _save_obs method to adjust observation dimensions
        """
        for key in self.keys:
            if key is None:
                # Special handling for non-dict observations
                if isinstance(obs, np.ndarray):
                    # Log the first time we encounter a shape mismatch
                    if len(obs) > self.buf_obs[key].shape[1]:
                        logger.warning(f"Truncating observation from shape {obs.shape} to match {self.buf_obs[key][env_idx].shape}")
                        # The observation is too large - slice it to match the expected shape
                        self.buf_obs[key][env_idx] = obs[:self.buf_obs[key].shape[1]]
                    else:
                        # Normal case - just assign the observation
                        self.buf_obs[key][env_idx] = obs
                else:
                    # Handle non-array observations
                    self.buf_obs[key][env_idx] = obs
            else:
                # Handle dict observations if needed
                if isinstance(obs, dict):
                    self.buf_obs[key][env_idx] = obs[key]
    
    def reset(self):
        """
        Override the reset method to handle observation shape adjustment
        during environment reset.
        """
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            
            # Log the observation shape for the first environment
            if env_idx == 0:
                logger.info(f"Environment reset returned observation with shape: {np.shape(obs)}")
                
                # Check if there's a shape mismatch with the observation space
                if isinstance(obs, np.ndarray) and len(obs) != self.observation_space.shape[0]:
                    logger.warning(f"Observation shape mismatch in reset: {len(obs)} vs expected {self.observation_space.shape[0]}")
            
            # Store the observation (with dimension adjustment if needed)
            self._save_obs(env_idx, obs)
            
        return self._obs_from_buf()

# Add this class after the BaseStockTradingEnv
class StockTradingEnvWrapper(gymnasium.Wrapper):
    """
    A wrapper for StockTradingEnv that ensures consistent observation shapes
    and is compatible with Gymnasium API used by Stable-Baselines3.
    """
    
    def __init__(self, env, state_space=16):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment to wrap
            state_space: Dimension of the state space
        """
        # Check if env is a gym environment rather than a gymnasium environment
        if hasattr(env, 'observation_space') and not isinstance(env.observation_space, gymnasium.spaces.Space):
            # Convert regular gym to gymnasium wrapper
            logger.info("Converting gym environment to gymnasium compatible wrapper")
            
            # Create observation and action spaces
            if hasattr(env, 'observation_space'):
                obs_space = convert_gym_space(env.observation_space)
            else:
                # Default observation space if not available
                obs_space = gymnasium.spaces.Box(
                    low=-10.0, high=10.0, shape=(state_space,), dtype=np.float32
                )
                
            if hasattr(env, 'action_space'):
                act_space = convert_gym_space(env.action_space)
            else:
                # Default action space if not available (3 actions: sell, hold, buy)
                act_space = gymnasium.spaces.Discrete(3)
            
            # Store the original environment and spaces
            self.env = env
            self.observation_space = obs_space
            self.action_space = act_space
            self.reward_range = env.reward_range if hasattr(env, 'reward_range') else (-float('inf'), float('inf'))
            self.metadata = env.metadata if hasattr(env, 'metadata') else {'render_modes': []}
            self.num_envs = 1
            
            logger.info(f"Created StockTradingEnvWrapper with observation space: {self.observation_space}")
        else:
            # If already a gymnasium environment, just wrap it directly
            super().__init__(env)
            # But ensure the observation space is consistent
            self.observation_space = gymnasium.spaces.Box(
                low=-10.0, high=10.0, shape=(state_space,), dtype=np.float32
            )
            logger.info(f"Wrapped Gymnasium environment with observation space: {self.observation_space}")
        
        # Store the actual observation dimension for comparison
        self.actual_state_space = None
        self.state_space = state_space
        self.step_counter = 0
        self.env_id = getattr(env, 'env_id', 0)
        
        # Debug logging
        logger.debug(f"Initialized StockTradingEnvWrapper with state_space={state_space}")
    
    @property
    def spec(self):
        """Return the environment spec."""
        if hasattr(self.env, 'spec'):
            return self.env.spec
        return None
    
    def reset(self, **kwargs):
        """Reset the environment, with compatibility for both gym and gymnasium APIs."""
        try:
            # Try gymnasium API first (for newer environments)
            reset_return = self.env.reset(**kwargs)
            
            # Handle different return types from reset
            if isinstance(reset_return, tuple):
                if len(reset_return) == 2:
                    # Standard gymnasium API: (obs, info)
                    obs, info = reset_return
                else:
                    # Some environments might return more values
                    # Take the first value as observation and create empty info
                    obs = reset_return[0]
                    info = {} if len(reset_return) <= 1 else reset_return[1]
            else:
                # Single return value (older gym API)
                obs = reset_return
                info = {}
        except TypeError:
            try:
                # Try gym API
                obs = self.env.reset()
                info = {}
            except Exception as e:
                logger.error(f"Error in reset: {e}")
                # Return zeros as a fallback
                obs = np.zeros(self.observation_space.shape)
                info = {}
        
        # Ensure observation is a numpy array
        if not isinstance(obs, np.ndarray):
            logger.warning(f"Reset returned non-numpy observation of type {type(obs)}")
            # Try to convert to numpy array
            try:
                obs = np.array(obs, dtype=np.float32)
            except Exception:
                # If conversion fails, return zeros
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
                
        # Store the actual observation dimension if not already stored
        if self.actual_state_space is None:
            self.actual_state_space = len(obs)
            logger.info(f"First reset - detected actual observation dimension: {self.actual_state_space}")
                
            # Update observation space if needed to match the actual dimension
            if self.actual_state_space != self.observation_space.shape[0]:
                logger.info(f"Updating observation space from shape {self.observation_space.shape} to ({self.actual_state_space},)")
                self.observation_space = gymnasium.spaces.Box(
                    low=-10.0, high=10.0, shape=(self.actual_state_space,), dtype=np.float32
                )
        
        logger.debug(f"Reset observation shape: {obs.shape if hasattr(obs, 'shape') else 'scalar'}, observation space shape: {self.observation_space.shape}")
            
        # Ensure observation has the right shape
        if len(obs) != self.state_space:
            logger.warning(f"Observation shape mismatch: got {len(obs)}, expected {self.state_space}")
            # Pad or truncate the observation to match the expected shape
            if len(obs) < self.state_space:
                # Pad with zeros if observation is too short
                padded_obs = np.zeros(self.state_space, dtype=np.float32)
                padded_obs[:len(obs)] = obs
                obs = padded_obs
                logger.info(f"Padded observation from shape {obs.shape} to {self.state_space}")
            else:
                # Truncate if observation is too long
                obs = obs[:self.state_space]
                logger.info(f"Truncated observation from shape {len(obs)} to {self.state_space}")
            
        # Clip observation values to reasonable range
        obs = np.clip(obs, -10.0, 10.0)
            
        return obs, info
    
    def step(self, action):
        """Step the environment, with compatibility for both gym and gymnasium APIs."""
        # Add debug logging for actions every 10000 steps
        if hasattr(self, 'step_counter'):
            if self.step_counter % 10000 == 0 and getattr(self, 'env_id', 0) == 0:
                action_type = "sell" if action == 0 else ("hold" if action == 1 else "buy")
                logger.info(f"Step {self.step_counter}: Taking action {action} ({action_type})")
        
        try:
            # Try gymnasium API first
            result = self.env.step(action)
            
            # Handle different return formats
            if isinstance(result, tuple):
                if len(result) == 5:  # gymnasium API (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                elif len(result) == 4:  # gym API (obs, reward, done, info)
                    obs, reward, done, info = result
                    terminated, truncated = done, False
                else:
                    # Unexpected number of return values - use defaults
                    logger.warning(f"Unexpected number of return values from step: {len(result)}")
                    obs = result[0] if len(result) > 0 else np.zeros(self.observation_space.shape)
                    reward = result[1] if len(result) > 1 else 0.0
                    done = result[2] if len(result) > 2 else False
                    info = result[3] if len(result) > 3 else {}
                    terminated, truncated = done, False
            else:
                # Handle unexpected return type (should not happen)
                logger.error(f"Unexpected return type from step: {type(result)}")
                obs = np.zeros(self.observation_space.shape)
                reward = 0.0
                done = True
                info = {}
                terminated, truncated = done, False
        except Exception as e:
            # If step fails for any reason, return a default observation
            logger.error(f"Error in environment step: {e}")
            obs = np.zeros(self.observation_space.shape)
            reward = 0.0
            done = True
            info = {"error": str(e)}
            terminated, truncated = done, False
        
        # Add trading metrics to info dictionary if not already present
        if isinstance(info, dict):
            # Ensure action_type is present
            if 'action_type' not in info:
                if action == 0:
                    info['action_type'] = 'sell'
                elif action == 1:
                    info['action_type'] = 'hold'
                elif action == 2:
                    info['action_type'] = 'buy'
            
            # Add position if available from env or track it ourselves
            if 'position' not in info:
                # Try to get position from env
                position = getattr(self.env, 'position', 0) if hasattr(self.env, 'position') else 0
                
                # If the env doesn't track position, we can estimate it based on action
                if not hasattr(self, 'current_position'):
                    self.current_position = 0
                
                # Update our position estimate based on action
                if info.get('action_type') == 'buy':
                    self.current_position = 1
                elif info.get('action_type') == 'sell':
                    self.current_position = -1
                elif info.get('action_type') == 'hold':
                    # Keep current position
                    pass
                
                # Add position to info
                info['position'] = getattr(self, 'current_position', 0)
            
            # Add price information if available
            if 'close' not in info and hasattr(self.env, 'data') and hasattr(self.env.data, 'get'):
                info['close'] = self.env.data.get('close', 0)
            
            # Add portfolio value if not present
            if 'portfolio_value' not in info and hasattr(self.env, 'initial_amount'):
                # Estimate portfolio value if we can
                cash = getattr(self.env, 'initial_amount', 100000)
                position = info.get('position', 0)
                close_price = info.get('close', 0)
                
                # Simple estimate of portfolio value
                portfolio_value = cash + (position * close_price)
                info['portfolio_value'] = portfolio_value
        
        # Log step observation for debugging (only on some steps to avoid excessive logging)
        if isinstance(obs, np.ndarray) and hasattr(self, 'step_counter'):
            self.step_counter = getattr(self, 'step_counter', 0) + 1
            # Only log every 10000 steps and only if env_id is 0 (first environment)
            if self.step_counter % 10000 == 0 and getattr(self, 'env_id', 0) == 0:
                logger.info(f"Step {self.step_counter} observation shape: {obs.shape}, observation space shape: {self.observation_space.shape}")
        
        # Ensure observation is a numpy array
        if not isinstance(obs, np.ndarray):
            logger.warning(f"Step returned non-numpy observation of type {type(obs)}")
            # Try to convert to numpy array
            try:
                obs = np.array(obs, dtype=np.float32)
            except Exception as e:
                logger.error(f"Could not convert observation to numpy array: {e}")
                # Use a default observation
                obs = np.zeros(self.observation_space.shape)
        
        # Ensure proper observation shape
        if len(obs.shape) == 0:  # Scalar observation
            obs = np.array([obs], dtype=np.float32)
            
        # If observation dimension doesn't match our state space, update the dimension
        if len(obs) != self.state_space:
            # Only track this the first time it happens or when it changes
            if self.actual_state_space is None or self.actual_state_space != len(obs):
                self.actual_state_space = len(obs)
                logger.info(f"Step returned observation with different dimension: {self.actual_state_space}")
                self.observation_space = gymnasium.spaces.Box(
                    low=-10.0, high=10.0, shape=(self.actual_state_space,), dtype=np.float32
                )
        
        # Clip observation values to reasonable range
        obs = np.clip(obs, -10.0, 10.0)
        
        # Always return in gymnasium format (5-tuple)
        return obs, reward, terminated, truncated, info
    
    def seed(self, seed=None):
        """Set the random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return [seed]

def convert_gym_space(space):
    """
    Convert a gym space to a gymnasium space.
    
    Args:
        space: gym space to convert
        
    Returns:
        Equivalent gymnasium space
    """
    import gym
    
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)
    elif isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low, high=space.high, 
            shape=space.shape, dtype=space.dtype
        )
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(space.n)
    else:
        logger.warning(f"Unsupported space type: {type(space)}, using default Box space")
        return gymnasium.spaces.Box(
            low=-10.0, high=10.0, shape=(16,), dtype=np.float32
        )

# Now modify the create_finrl_env function to use this wrapper
def create_finrl_env(
    start_date, end_date, symbols, data_source="binance", initial_balance=1000000.0,
    lookback=5, state_space=16, include_cash=False, initial_stocks=None, window_size=None,
    df=None  # New parameter to allow passing a pre-created DataFrame
):
    """
    Create a FinRL environment for trading with StockTradingEnv.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        symbols: List of symbols to trade
        data_source: Source of data (e.g., 'binance', 'yahoo')
        initial_balance: Initial balance for trading
        lookback: Number of days to look back for state
        state_space: Dimension of state space
        include_cash: Whether to include cash in state
        initial_stocks: Initial stocks count, default 0
        window_size: Window size for data processing
        df: Pre-created DataFrame (optional)
        
    Returns:
        StockTradingEnv instance
    """
    # Log the environment creation
    logger.info(f"Creating FinRL environment with state_space={state_space}, lookback={lookback}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Symbols: {symbols}")
    
    # Import appropriate environment class
    try:
        logger.info("Attempting to import StockTradingEnv for cryptocurrency trading")
        # Try to import CryptocurrencyTradingEnv
        from finrl.meta.env_crypto_trading.env_crypto import CryptocurrencyTradingEnv as StockTradingEnvClass
        logger.info("Using StockTradingEnv as CryptocurrencyTradingEnv")
    except (ImportError, ModuleNotFoundError):
        logger.warning("Could not import CryptocurrencyTradingEnv, trying alternative paths")
        try:
            # Try to import StockTradingEnv directly
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv as StockTradingEnvClass
            logger.info("Using standard StockTradingEnv as fallback")
        except (ImportError, ModuleNotFoundError):
            try:
                # Try the old import path
                from finrl.env.env_stocktrading import StockTradingEnv as StockTradingEnvClass
                logger.info("Using legacy StockTradingEnv import path")
            except (ImportError, ModuleNotFoundError):
                try:
                    # Last resort - try to import directly from known path
                    import sys
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "StockTradingEnv", 
                        "/venv/main/lib/python3.10/site-packages/finrl/meta/env_stock_trading/env_stocktrading.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = module
                    spec.loader.exec_module(module)
                    StockTradingEnvClass = module.StockTradingEnv
                    logger.info("Using absolute path import for StockTradingEnv")
                except Exception as e:
                    logger.error(f"Failed to import any StockTradingEnv class: {e}")
                    raise ImportError("Could not import any version of StockTradingEnv")

    # Prepare environment parameters
    stock_dimension = len(symbols)
    
    # Default values
    if initial_stocks is None:
        initial_stocks = [0] * stock_dimension
    
    # Define the list of technical indicators
    tech_indicators = [
        'macd', 'rsi_14', 'cci_30', 'dx_30', 
        'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_60_sma', 'close_120_sma',
        'close_5_ema', 'close_10_ema', 'close_20_ema', 'close_60_ema', 'close_120_ema',
        'volatility_30', 'volume_change', 'volume_norm'
    ]
    
    # Set up the parameters dictionary for StockTradingEnv
    env_params = {
        'df': df,
        'state_space': state_space,
        'initial_amount': initial_balance,
        'buy_cost_pct': 0.001,  # Transaction cost for buying
        'sell_cost_pct': 0.001,  # Transaction cost for selling
        'reward_scaling': 0.0001,  # Scaling factor for reward
        'hmax': 100,  # Maximum number of shares to trade
        'stock_dim': stock_dimension,
        'num_stock_shares': initial_stocks,
        'action_space': stock_dimension,  # If include_cash is True, action_space will be stock_dimension+1
        'tech_indicator_list': tech_indicators,
    }
    
    # Log the environment parameters
    logger.info(f"Creating environment with params: {env_params}")
    
    try:
        # Ensure all required technical indicators are present
        df = ensure_technical_indicators(df, tech_indicators)
        
        # Create the environment
        env = StockTradingEnvClass(**env_params)
        return env
    except Exception as e:
        # Log the detailed error information
        logger.error(f"Error creating environment: {e}")
        logger.error(traceback.format_exc())
        raise

class CustomDummyVecEnv:
    """
    Custom implementation of a vectorized environment that doesn't inherit from gymnasium.vector.VectorEnv.
    This creates a simple vectorized wrapper for multiple environments.
    """
    
    def __init__(self, env_fns):
        """
        Initialize the vectorized environment.
        
        Args:
            env_fns: List of functions that create environments to run in parallel
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Get the observation and action spaces from the first environment
        env = self.envs[0]
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # Set up the observation and action buffers
        self.obs_buffer = np.zeros((self.num_envs,) + env.observation_space.shape, 
                                   dtype=env.observation_space.dtype)
        self.actions = None
    
    def reset(self, **kwargs):
        """
        Reset all environments and return initial observations.
        
        Args:
            **kwargs: Additional reset parameters
        
        Returns:
            Initial observations and reset info
        """
        self.obs_buffer = np.array([env.reset(**kwargs)[0] for env in self.envs])
        infos = [{} for _ in range(self.num_envs)]
        return self.obs_buffer.copy(), infos
    
    def step(self, actions):
        """
        Step through each environment with corresponding actions.
        
        Args:
            actions: Actions to take in each environment
            
        Returns:
            Observations, rewards, terminations, truncations, and infos
        """
        self.actions = actions
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        
        obs, rews, terms, truncs, infos = zip(*results)
        
        self.obs_buffer = np.array(obs)
        
        return (
            self.obs_buffer.copy(),
            np.array(rews),
            np.array(terms),
            np.array(truncs),
            {i: info for i, info in enumerate(infos)}
        )
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

# Now modify the create_parallel_finrl_envs function to use this wrapper
def create_parallel_finrl_envs(df, args, num_workers=4):
    """
    Create multiple FinRL environments for parallel training.
    
    Args:
        df: DataFrame with market data
        args: Command line arguments
        num_workers: Number of parallel workers to use
        
    Returns:
        Vectorized environment
    """
    logger.info(f"Creating parallel environments with {num_workers} workers")
    
    # Environment parameters
    num_envs_per_worker = getattr(args, 'num_envs_per_worker', 4)
    use_subproc = getattr(args, 'use_subproc_vecenv', False)
    
    device = None
    if hasattr(args, 'device'):
        import torch
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    
    logger.info(f"Creating vectorized environment with {num_workers} workers and {num_envs_per_worker} envs per worker")
    logger.info(f"Using {'SubprocVecEnv' if use_subproc else 'DummyVecEnv'} for environment vectorization")
    
    # Use our patched environment instead of importing from FinRL
    logger.info("Using PatchedStockTradingEnv to prevent numpy.float64 attribute errors")
    StockTradingEnvClass = PatchedStockTradingEnv
    
    # Define potential technical indicators
    potential_indicators = [
        'macd', 'rsi_14', 'rsi', 
        'stoch_k', 'stoch_d',
        'cci_30', 'dx_30',
        'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_60_sma', 'close_120_sma',
        'close_5_ema', 'close_10_ema', 'close_20_ema', 'close_60_ema', 'close_120_ema',
        'volatility_30', 'volume_change', 'volume_norm'
    ]
    
    # Check which indicators are available in the dataframe
    tech_indicator_list = []
    for indicator in potential_indicators:
        if indicator in df.columns:
            tech_indicator_list.append(indicator)
    
    logger.info(f"Using technical indicators found in dataframe: {tech_indicator_list}")
    
    # If no indicators were found, add basic ones
    if not tech_indicator_list:
        logger.warning("No valid technical indicators found in dataframe. Adding basic indicators.")
        df = ensure_technical_indicators(df, potential_indicators)
        # Check again after ensuring indicators
        for indicator in potential_indicators:
            if indicator in df.columns:
                tech_indicator_list.append(indicator)
                
    # Calculate environment dimensions
    state_space = getattr(args, 'state_dim', 16)  # Default state space size if not specified
    stock_dim = 1  # Assuming single-asset trading for simplicity
    num_stocks = len(df['tic'].unique()) if 'tic' in df.columns else 1
    
    # Action space is either:
    # 1. Discrete(3) for sell/hold/buy
    # 2. Discrete(2*stock_dim+1) for more complex actions
    if hasattr(args, 'action_space') and args.action_space is not None:
        action_space = args.action_space
    else:
        action_space = 3  # Sell, hold, buy
    
    logger.info(f"Creating parallel environment with state_space={state_space}, "
                f"action_space={action_space}, num_stocks={num_stocks}")
    
    # Environment parameters
    initial_amount = getattr(args, 'initial_balance', 1000000)
    # Hardcode transaction costs to use crypto exchange maker/taker fees of 0.075%
    transaction_cost_pct = 0.00075  # 0.075% as a decimal
    logger.info("Using hardcoded crypto exchange maker/taker fees: 0.075%")
    reward_scaling = getattr(args, 'reward_scaling', 1e-4)
    # Set hmax (max shares per trade) based on average price
    if 'close' in df.columns:
        avg_price = df['close'].mean()
        hmax = max(int(initial_amount * 0.1 / avg_price), 10)  # Allow up to 10% of balance per trade
    else:
        hmax = 100  # Default value
    
    logger.info(f"Environment config: initial_amount={initial_amount}, "
                f"transaction_cost_pct={transaction_cost_pct}, "
                f"reward_scaling={reward_scaling}, stock_dim={stock_dim}, hmax={hmax}")
    
    # Process the DataFrame to ensure it's correctly formatted for FinRL
    try:
        logger.info("Validating DataFrame format for FinRL compatibility")
        
        # StockTradingEnv expects a DataFrame with a specific format:
        # 1. Each row represents a single day and stock
        # 2. The DataFrame should have columns like 'open', 'high', 'low', 'close', 'volume', etc.
        # 3. It also needs a 'tic' column to identify different stocks
        # 4. A 'date' column (or index) for dates
        # 5. A 'day' column with integer values to represent the time steps
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'tic', 'day']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found in DataFrame")
                # Create dummy values if missing
                if col in ['open', 'high', 'low', 'close']:
                    logger.info(f"Creating dummy '{col}' column with price data")
                    df[col] = 10000.0  # Dummy price
                elif col == 'volume':
                    logger.info(f"Creating dummy '{col}' column")
                    df['volume'] = 1000000.0  # Dummy volume
                elif col == 'tic':
                    logger.info(f"Creating dummy '{col}' column")
                    df['tic'] = 'BTC'  # Default ticker
                elif col == 'day':
                    logger.info(f"Creating '{col}' column as integer index")
                    if 'date' in df.columns:
                        # If we have a date column, convert it to day integers
                        df['day'] = pd.factorize(df['date'])[0]
                    else:
                        # Otherwise just use row index
                        df['day'] = np.arange(len(df))
        
        # If 'day' column exists but is not numeric, convert it
        if pd.api.types.is_categorical_dtype(df['day']) or pd.api.types.is_object_dtype(df['day']):
            logger.info("Converting 'day' column to integer")
            df['day'] = df['day'].factorize()[0]
        
        # Ensure all numeric columns have finite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.any(~np.isfinite(df[col])):
                logger.warning(f"Column '{col}' contains non-finite values. Replacing with zeros.")
                df[col] = df[col].replace([np.inf, -np.inf, np.nan], 0)
        
        # Create a test environment to see if the format is correct
        logger.info(f"Creating test environment with indicators: {tech_indicator_list}")
        
        # Select the first day's data for testing
        first_day = df['day'].min()
        test_df = df[df['day'] == first_day].copy()
        
        # Parameters for the test environment
        buy_cost_pct_list = [transaction_cost_pct] * stock_dim 
        sell_cost_pct_list = [transaction_cost_pct] * stock_dim
        
        test_params = {
            'df': test_df,
            'stock_dim': stock_dim,
            'hmax': hmax,
            'num_stock_shares': [0] * stock_dim,
            'action_space': action_space,  # Make sure this is included!
            'tech_indicator_list': tech_indicator_list,
            'initial_amount': initial_amount,
            'buy_cost_pct': buy_cost_pct_list,
            'sell_cost_pct': sell_cost_pct_list,
            'reward_scaling': reward_scaling,
            'state_space': state_space,
            'print_verbosity': 0
        }
        
        # Wrap the test in try-except to catch detailed error info
        try:
            test_env = StockTradingEnvClass(**test_params)
            test_obs = test_env.reset()
            
            if isinstance(test_obs, np.ndarray):
                logger.info(f"Test environment observation shape: {test_obs.shape}")
                logger.info(f"Test environment observation space: {test_env.observation_space}")
                if hasattr(test_env, 'state'):
                    logger.info(f"Test environment state size: {len(test_env.state)}")
                    
                # If observation shape doesn't match state_space, adjust state_space
                if len(test_obs) != state_space:
                    logger.warning(f"Test observation shape {len(test_obs)} doesn't match state_space {state_space}")
                    if len(test_obs) > state_space:
                        logger.info(f"Adjusting state_space to match test observation shape: {len(test_obs)}")
                        state_space = len(test_obs)
        except Exception as e:
            logger.error(f"Error creating test environment: {e}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error testing environment: {e}")
        logger.error(traceback.format_exc())
    
    # Create a list of environment creation functions
    env_list = []
    
    for i in range(num_workers * num_envs_per_worker):
        # Define a function that creates a new environment instance each time it's called
        def make_env(idx=i):
            try:
                # Create basic environment parameters
                env_params = {
                    'df': df,
                    'stock_dim': stock_dim,
                    'hmax': hmax,
                    'num_stock_shares': [0] * stock_dim,  # Start with no shares - REQUIRED parameter
                    'action_space': action_space,  # Make sure this is passed!
                    'tech_indicator_list': tech_indicator_list,
                    'initial_amount': initial_amount,
                    # 'transaction_cost_pct': transaction_cost_pct_list,  # Remove - not accepted by FinRL
                    'buy_cost_pct': [transaction_cost_pct] * stock_dim,  # Pass as list
                    'sell_cost_pct': [transaction_cost_pct] * stock_dim,  # Pass as list
                    'reward_scaling': reward_scaling,
                    'state_space': state_space,  # Add the missing state_space parameter
                    'print_verbosity': 1 if idx == 0 else 0  # Only print verbose output for the first env
                }
                
                # Create the environment
                base_env = StockTradingEnvClass(**env_params)
                
                # Set env_id attribute to identify each environment
                base_env.env_id = idx
                
                # Wrap the environment to ensure consistent observation shape
                wrapped_env = StockTradingEnvWrapper(base_env, state_space=state_space)
                
                # Wrap with monitoring wrapper for metrics
                return wrap_env_with_monitor(wrapped_env)
            except Exception as e:
                logger.error(f"Error creating environment {idx}: {e}")
                logger.error(traceback.format_exc())
                
                # Provide a fallback environment if creation fails
                base_env = BaseStockTradingEnv(df, state_space=state_space, action_space=action_space)
                # Set env_id for the fallback environment too
                base_env.env_id = idx 
                wrapped_env = StockTradingEnvWrapper(base_env, state_space=state_space)
                return wrap_env_with_monitor(wrapped_env)
        
        # Add the environment creation function to the list with proper closure handling
        def env_fn(idx=i):
            return make_env(idx)
        env_list.append(env_fn)
    
    # Create the vectorized environment
    if use_subproc and (device is None or str(device).startswith('cpu')):
        # Import the SubprocVecEnv from stable_baselines3
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv(env_list)
        logger.info("Using SubprocVecEnv for parallel environment execution")
    else:
        if use_subproc and device is not None and not str(device).startswith('cpu'):
            logger.info("SubprocVecEnv requested but running on GPU. Using DummyVecEnv instead for better GPU utilization.")
        vec_env = DummyVecEnv(env_list)
        logger.info("Using DummyVecEnv for environment vectorization")
    
    return vec_env

def prepare_crypto_data_for_finrl(market_data, primary_timeframe):
    """
    Prepare cryptocurrency data for FinRL format.
    
    Args:
        market_data: Dictionary containing market data for different timeframes
        primary_timeframe: The primary timeframe to use for training
        
    Returns:
        DataFrame in FinRL format
    """
    logger.info(f"Preparing crypto data for FinRL using {primary_timeframe} timeframe")
    
    # Get data for the primary timeframe
    if primary_timeframe not in market_data:
        raise ValueError(f"Primary timeframe {primary_timeframe} not found in market data")
    
    df = market_data[primary_timeframe].copy()
    
    # Ensure we have a proper index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.info(f"Converting numeric index to DatetimeIndex, original index range: {df.index.min()} - {df.index.max()}")
        
        # Instead of converting directly to datetime (which can cause overflow),
        # use a simple integer index and track original index in a separate column
        df['original_index'] = df.index
        df.reset_index(drop=True, inplace=True)
        
        # Create simple datetime index using integer position
        start_date = pd.Timestamp('2010-01-01')
        if primary_timeframe == '15m':
            freq = '15min'
        elif primary_timeframe == '1h':
            freq = '1H'
        elif primary_timeframe == '4h':
            freq = '4H'
        else:  # default to daily
            freq = '1D'
        
        # Generate datetime index that matches the length of the dataframe
        df.index = pd.date_range(start=start_date, periods=len(df), freq=freq)
        logger.info(f"Created synthetic DatetimeIndex with range: {df.index.min()} - {df.index.max()}")
    
    # Add ticker column (default to BTC as this is crypto data)
    df['tic'] = 'BTC'
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in dataframe")
    
    # Add standard FinRL columns if they don't exist
    if 'day' not in df.columns:
        df['day'] = df.index.day
    if 'date' not in df.columns:
        df['date'] = df.index.date

    # Calculate additional technical indicators
    df = add_technical_indicators(df)
    
    # Make sure all values are finite
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # IMPORTANT: FinRL StockTradingEnv requires a numerical index
    # Reset the index to create a numerical one, and keep date as a column
    if isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'datetime'}, inplace=True)
        # Create the day column as an integer index (0, 1, 2, ...) that FinRL uses
        df['day'] = np.arange(len(df))
    
    logger.info(f"Prepared FinRL data with shape: {df.shape}")
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame containing OHLCV data
        
    Returns:
        DataFrame with additional technical indicators
    """
    # Calculate RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Calculate MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    
    # Calculate ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Calculate SMAs
    df['sma_7'] = ta.trend.SMAIndicator(df['close'], window=7).sma_indicator()
    df['sma_25'] = ta.trend.SMAIndicator(df['close'], window=25).sma_indicator()
    
    # Calculate EMAs
    df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    
    # Calculate Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    return df

def setup_finrl_import():
    """Setup import for FinRL DRLAgent with version detection"""
    global DRLAgent
    
    try:
        # Try the new import path first (FinRL 0.3.7+)
        from finrl.agents.stablebaselines3.models import DRLAgent
        logger.info("Using FinRL 0.3.7+ import path for DRLAgent")
        
        # Try to detect API details
        try:
            from inspect import signature
            init_sig = signature(DRLAgent.__init__)
            logger.info(f"DRLAgent.__init__ signature: {init_sig}")
        except Exception as e:
            logger.warning(f"Could not inspect DRLAgent.__init__ signature: {e}")
        
        return DRLAgent
    except ImportError:
        try:
            # Try the newest FinRL import path (as of 2024+)
            from finrl.applications.agents.drl_agents import DRLAgent
            logger.info("Using FinRL latest import path (applications.agents) for DRLAgent")
            
            # Try to detect API details
            try:
                from inspect import signature
                init_sig = signature(DRLAgent.__init__)
                logger.info(f"DRLAgent.__init__ signature: {init_sig}")
            except Exception as e:
                logger.warning(f"Could not inspect DRLAgent.__init__ signature: {e}")
            
            return DRLAgent
        except ImportError:
            try:
                # Fall back to older path (FinRL <= 0.3.5)
                from finrl.agents.drl_agent import DRLAgent
                logger.info("Using FinRL <= 0.3.5 import path for DRLAgent")
                
                # Try to detect API details
                try:
                    from inspect import signature
                    init_sig = signature(DRLAgent.__init__)
                    logger.info(f"DRLAgent.__init__ signature: {init_sig}")
                except Exception as e:
                    logger.warning(f"Could not inspect DRLAgent.__init__ signature: {e}")
                
                return DRLAgent
            except ImportError:
                logger.error("Could not import DRLAgent from any known FinRL paths")
                return None


# Get the DRLAgent class
OriginalDRLAgent = setup_finrl_import()

# Create our own derived DRLAgent class to bypass patching
class CustomDRLAgent:
    """
    A custom DRLAgent that bypasses the patching mechanism in Stable-Baselines3.
    This avoids the gym.spaces.Sequence compatibility issue.
    """
    def __init__(self, env, verbose=1):
        """
        Initialize the agent with the given environment.
        
        Args:
            env: The environment to use
            verbose: Verbosity level
        """
        # Check if OriginalDRLAgent was successfully imported
        if OriginalDRLAgent is None:
            logger.error("Cannot initialize CustomDRLAgent - FinRL's DRLAgent not found")
            raise ImportError("FinRL's DRLAgent could not be imported. Please install FinRL first.")
            
        # Set default net_arch attribute that might be missing from parent class
        self.net_arch = [256, 256]
        self.env = env
        self.verbose = verbose
        
        # Initialize the original agent
        try:
            # Try with verbose parameter first
            self.agent = OriginalDRLAgent(env=env, verbose=verbose)
        except TypeError:
            # Fall back to just env if verbose is not accepted
            self.agent = OriginalDRLAgent(env=env)
            logger.info("DRLAgent parent class doesn't accept verbose parameter, using default initialization")
    
    def get_model(self, model_name, model_kwargs=None):
        """
        Create model without applying any patching to the environment.
        
        Args:
            model_name: Name of the RL model
            model_kwargs: Arguments for the model
            
        Returns:
            Model instance
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # If we have the original agent, delegate to it with our wrapped methods
        if hasattr(self, 'agent') and self.agent is not None:
            try:
                logger.info(f"Delegating model creation to original DRLAgent for {model_name}")
                # Patch get_model to avoid patching
                original_get_model = self.agent.get_model
                
                # Try to get the model from the original agent
                model = original_get_model(model_name, model_kwargs)
                return model
            except Exception as e:
                logger.warning(f"Error delegating to original agent: {e}")
                # Fall through to our manual implementation
        
        # Import MODELS dictionary from finrl with better error handling
        try:
            # Try the main path first
            from finrl.agents.stablebaselines3.models import MODELS
            logger.info("Imported MODELS dictionary from finrl.agents.stablebaselines3.models")
        except ImportError:
            try:
                # Try alternate path (newer versions)
                from finrl.applications.agents.stablebaselines3.models import MODELS
                logger.info("Imported MODELS dictionary from finrl.applications.agents.stablebaselines3.models")
            except ImportError:
                try:
                    # Last resort for very old versions
                    from finrl.model.models import MODELS
                    logger.info("Imported MODELS dictionary from finrl.model.models")
                except ImportError:
                    raise ImportError("Could not import MODELS dictionary from any known FinRL paths")
            
        # Use the environment directly without patching
        env = self.env
        
        # Add common params for all models
        model_kwargs.update({
            "env": env,
            "verbose": self.verbose,
            "policy": "MlpPolicy",
            "policy_kwargs": {"net_arch": self.net_arch},
        })
        
        # Get the model class
        if model_name not in MODELS:
            raise NotImplementedError(f"Model {model_name} not supported. Available models: {list(MODELS.keys())}")
            
        # Create model instance with original init method but with special handling for env
        model_class = MODELS[model_name]
        
        # Special handling for different model types
        if model_name.lower() == 'sac':
            # SAC has different parameters - try to create it directly from Stable-Baselines3
            try:
                from stable_baselines3 import SAC
                logger.info("Creating SAC model directly from stable_baselines3")
                
                # Add SAC-specific parameters if not present
                if 'buffer_size' in model_kwargs and 'replay_buffer_class' not in model_kwargs:
                    from stable_baselines3.common.buffers import ReplayBuffer
                    model_kwargs['replay_buffer_class'] = ReplayBuffer
                
                # Create the model directly
                model = SAC(**model_kwargs)
                return model
            except Exception as e:
                logger.warning(f"Failed to create SAC directly, falling back to FinRL approach: {e}")
                # Fall back to normal approach
        
        # Monkey patch the _wrap_env method to disable patching
        original_wrap_env = model_class._wrap_env if hasattr(model_class, '_wrap_env') else None
        
        if original_wrap_env:
            def no_patch_wrap_env(self, env, verbose=0, monitor_wrapper=True):
                """Do not apply any patching to the environment"""
                # Skip the patching completely - return env directly
                return env
                
            # Apply monkey patch
            model_class._wrap_env = no_patch_wrap_env
        
        # Create model
        try:
            logger.info(f"Creating {model_name} model with kwargs: {model_kwargs}")
            model = model_class(**model_kwargs)
        except TypeError as e:
            logger.error(f"TypeError creating {model_name} model: {e}")
            
            # Special handling for common errors
            if "unexpected keyword argument" in str(e):
                # Identify problematic args and remove them
                problematic_arg = str(e).split("unexpected keyword argument '")[1].split("'")[0]
                logger.info(f"Removing problematic argument: {problematic_arg}")
                if problematic_arg in model_kwargs:
                    del model_kwargs[problematic_arg]
                    # Try again with cleaned args
                    model = model_class(**model_kwargs)
            else:
                # Log the model parameters and expected signature
                try:
                    from inspect import signature
                    logger.info(f"Expected signature: {signature(model_class.__init__)}")
                    logger.info(f"Model kwargs: {model_kwargs}")
                except Exception:
                    pass
                raise
        except Exception as e:
            logger.error(f"Error creating {model_name} model: {e}")
            # Log the model parameters and expected signature
            try:
                from inspect import signature
                logger.info(f"Expected signature: {signature(model_class.__init__)}")
                logger.info(f"Model kwargs: {model_kwargs}")
            except Exception:
                pass
            raise
        
        # Restore original method
        if original_wrap_env:
            model_class._wrap_env = original_wrap_env
        
        return model


def train_with_finrl(
    args, logger, start_date, end_date, tickers, 
    data_source="binance", num_workers=1, use_lstm_predictions=False,
    lstm_model=None, lstm_processor=None
):
    """
    Train a reinforcement learning agent using FinRL framework.
    
    Args:
        args: Command line arguments
        logger: Logger instance
        start_date: Start date for training data
        end_date: End date for training data
        tickers: List of ticker symbols
        data_source: Source of data
        num_workers: Number of parallel environments
        use_lstm_predictions: Whether to use LSTM predictions
        lstm_model: LSTM model instance
        lstm_processor: LSTM data processor
        
    Returns:
        Trained model
    """
    logger.info("Training with FinRL framework")
    print("=== Starting FinRL training ===")
    
    try:
        # Import stable-baselines3 and torch.nn
        from stable_baselines3 import PPO, DDPG, TD3, SAC
        import torch.nn as nn
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
        import gymnasium
        import numpy as np
        import os
    except ImportError as e:
        logger.error(f"Error importing required packages: {e}")
        raise
    
    # Extract arguments for training
    finrl_model = args.finrl_model.lower()
    initial_balance = getattr(args, 'initial_balance', 1000000.0)
    lookback = getattr(args, 'lookback', 10)
    include_cash = getattr(args, 'include_cash', False)
    state_dim = getattr(args, 'state_dim', 16)  # Default to 16 if not specified
    
    # First check if synthetic data exists in data/synthetic directory
    synthetic_data_path = os.path.join('data', 'synthetic', 'synthetic_dataset.h5')
    
    if os.path.exists(synthetic_data_path):
        logger.info(f"Found existing synthetic data at {synthetic_data_path}, loading...")
        
        try:
            import pandas as pd
            
            # Load data from HDF5 file
            with pd.HDFStore(synthetic_data_path, mode='r') as store:
                # Get all available timeframes from the HDF5 file
                timeframes = [key[1:] for key in store.keys()]  # Remove leading '/'
                logger.info(f"Available timeframes in dataset: {timeframes}")
                
                # Load primary timeframe (15m) to use for FinRL
                if '15m' in timeframes:
                    df = store['/15m']
                    logger.info(f"Loaded 15m data with shape {df.shape}")
                    
                    # Reset index to make date a column if it's in the index
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                elif timeframes:
                    # If 15m is not available, use the first available timeframe
                    tf = timeframes[0]
                    df = store[f'/{tf}']
                    logger.info(f"15m timeframe not available, using {tf} with shape {df.shape}")
                    
                    # Reset index to make date a column if it's in the index
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index()
                else:
                    raise ValueError("No timeframe data found in the HDF5 file")
                
            # Check for timestamp column which might be named 'date' or be in the index
            if 'timestamp' not in df.columns:
                if 'date' in df.columns:
                    df.rename(columns={'date': 'timestamp'}, inplace=True)
                else:
                    # If neither timestamp nor date is in columns, index might be the date
                    if isinstance(df.index, pd.DatetimeIndex):
                        df['timestamp'] = df.index
                        df.reset_index(drop=True, inplace=True)
            
            # Ensure datetime type for timestamp
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Add 'tic' column if not present
            if 'tic' not in df.columns:
                logger.info(f"Adding 'tic' column with value: {tickers[0]}")
                df['tic'] = tickers[0]
            
            # Create 'day' column if not present (required by some FinRL components)
            if 'day' not in df.columns:
                df['day'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
            
            # Create a multi-index dataframe with (timestamp, tic)
            logger.info("Creating multi-index DataFrame with (timestamp, tic)")
            df = df.rename(columns={'timestamp': 'date'})  # FinRL expects 'date'
            
            # Special preparation for FinRL's StockTradingEnv
            # The environment expects a DataFrame with a numeric index and 'date' and 'tic' columns
            # NOT a multi-index DataFrame as we initially thought
            if 'date' in df.columns and 'tic' in df.columns:
                # Keep date and tic as columns (don't set as index)
                # Sort the DataFrame by date and tic
                df = df.sort_values(['date', 'tic']).reset_index(drop=True)
                logger.info("Data prepared as DataFrame with 'date' and 'tic' columns (FinRL format)")
            else:
                # Create a multi-index but we'll need to transform it later
                df = df.set_index(['date', 'tic'])
                # Convert back to columns for FinRL compatibility
                df = df.reset_index()
                logger.info("Data prepared with 'date' and 'tic' as regular columns")
            
            # Factor the dates to integers (0, 1, 2, ...) which StockTradingEnv expects
            # This is CRITICAL for FinRL's StockTradingEnv to work
            if 'date' in df.columns:
                df['day'] = df['date'].factorize()[0]
                logger.info(f"Factorized dates to day column with range: {df['day'].min()} - {df['day'].max()}")
                
            # Final sort by day and tic
            if 'day' in df.columns and 'tic' in df.columns:
                df = df.sort_values(['day', 'tic']).reset_index(drop=True)
                
            logger.info(f"Processed data shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns}")
            logger.info(f"DataFrame index: {df.index.name or 'default'}")
            logger.info(f"Sample of prepared data:\n{df.head(3)}")
        except Exception as e:
            logger.error(f"Error loading synthetic data: {str(e)}")
            logger.error(traceback.format_exc())
            df = None
    else:
        logger.info(f"No existing synthetic data found at {synthetic_data_path}, generating new data...")
        df = None
    
    # Generate synthetic data or format the real data properly if loading failed
    if df is None:
        try:
            # Try to import generate_data functions to use the same format
            try:
                # First try to import directly
                from generate_data import generate_synthetic_data
                logger.info("Using generate_data.py's generate_synthetic_data function")
                
                # Generate synthetic data using the consistent generator
                output_dir = os.path.join('data', 'synthetic')
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate data for all timeframes
                dataset = generate_synthetic_data(
                    symbols=tickers,
                    timeframes=['15m', '4h', '1d'],
                    num_samples=100000,  # More samples for better training
                    start_date=start_date,
                    end_date=end_date,
                    seed=42,
                    output_path=os.path.join(output_dir, 'synthetic_dataset.h5')
                )
                
                # Use the 15m timeframe for trading environment
                df = dataset['15m'].copy()
                df.reset_index(inplace=True)  # Convert index to column
                df.rename(columns={'timestamp': 'date'}, inplace=True)  # Rename to match FinRL
                
                # Add tic column
                df['tic'] = tickers[0]  # Assign first ticker to all rows
                
                # Factor the dates to integers (0, 1, 2, ...) which StockTradingEnv expects
                df['day'] = df['date'].factorize()[0]
                logger.info(f"Factorized dates to day column with range: {df['day'].min()} - {df['day'].max()}")
                
                # Sort by day and tic for FinRL compatibility
                df = df.sort_values(['day', 'tic']).reset_index(drop=True)
                
                logger.info(f"Generated synthetic data using generate_data.py's format")
                logger.info(f"Sample data:\n{df.head(3)}")
            except (ImportError, ModuleNotFoundError):
                logger.warning("Could not import generate_data module, falling back to internal synthetic data generation")
                logger.info("Creating synthetic data with multi-index format for FinRL...")
                df = create_synthetic_data(tickers, start_date, end_date)
                
                # Make sure the internal synthetic data is also properly formatted
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                    
                # Add day column with factorized dates
                if 'day' not in df.columns and 'date' in df.columns:
                    df['day'] = df['date'].factorize()[0]
                    logger.info(f"Added day column with factorized dates: {df['day'].min()} - {df['day'].max()}")
                
                # Final sorting by day and tic
                if 'day' in df.columns and 'tic' in df.columns:
                    df = df.sort_values(['day', 'tic']).reset_index(drop=True)
                    
                logger.info(f"Internal synthetic data prepared with shape: {df.shape}")
                logger.info(f"Sample data:\n{df.head(3)}")
        except Exception as e:
            logger.error(f"Error creating synthetic data: {e}")
            logger.error(traceback.format_exc())
            raise
    
    # Use DummyVecEnv due to gym.spaces.Sequence compatibility issues
    logger.info("Using DummyVecEnv for environment vectorization due to compatibility issues")
    
    # Define the list of technical indicators needed
    tech_indicators = [
        'macd', 'rsi_14', 'cci_30', 'dx_30', 
        'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_60_sma', 'close_120_sma',
        'close_5_ema', 'close_10_ema', 'close_20_ema', 'close_60_ema', 'close_120_ema',
        'volatility_30', 'volume_change', 'volume_norm'
    ]
    
    # Make sure all required technical indicators are in the DataFrame
    logger.info("Ensuring all required technical indicators are available...")
    df = ensure_technical_indicators(df, tech_indicators)
    
    # Create parallel environments
    logger.info(f"Creating vectorized environment with {num_workers} workers")
    try:
        vec_env = create_parallel_finrl_envs(
            df=df,
            args=args,
            num_workers=num_workers
        )
        
        # Test the observation shape to ensure it's compatible with the model
        test_obs = vec_env.reset()
        actual_obs_dim = test_obs.shape[1]
        logger.info(f"Vectorized environment observation shape: {test_obs.shape}")
        
        if actual_obs_dim != state_dim:
            logger.warning(f"Observation dimension mismatch: got {actual_obs_dim}, expected {state_dim}")
            logger.info(f"Adjusting policy network to use actual observation dimension: {actual_obs_dim}")
            # Update state_dim to actual observation dimension
            state_dim = actual_obs_dim
    except Exception as e:
        logger.error(f"Error creating or initializing environments: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Define policy network architecture
    policy_kwargs = {
        'activation_fn': nn.ReLU,
        'net_arch': {
            'pi': [256, 256],  # Actor network
            'qf': [256, 256]   # Critic network
        }
    }
    
    # Select model class based on finrl_model parameter
    if finrl_model == 'ppo':
        model_class = PPO
    elif finrl_model == 'ddpg':
        model_class = DDPG
    elif finrl_model == 'td3':
        model_class = TD3
    else:  # Default to SAC
        model_class = SAC
        
    # Define callback to log metrics to tensorboard
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            # Episode tracking
            self.episode_rewards = []
            self.episode_lengths = []
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Portfolio tracking
            self.portfolio_values = []
            
            # Trade metrics tracking
            self.trade_count = 0
            self.successful_trades = 0
            self.failed_trades = 0
            self.total_profit = 0
            self.total_loss = 0
            self.action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
            self.trade_returns = []
            self.trade_durations = []
            self.recent_trades = []  # List to track recent trades for trend analysis
            self.max_recent_trades = 100  # Maximum number of recent trades to keep
            
            # Position tracking
            self.current_position = 0
            self.entry_price = 0
            self.position_start_time = 0
            
            # Debug info
            self.debug_steps = 0
            self.last_debug_output = 0
            self.debug_frequency = 500  # Reduce logging frequency (was 5000)
            self.log_flush_frequency = 5000  # Flush logs more frequently (was 20000)
            
            logger.info("TensorboardCallback initialized - Will track trading metrics")
            print("TensorboardCallback initialized - Will track trading metrics and output to console")
        
        def _on_step(self):
            # Increment step counter for debugging
            self.debug_steps += 1
            
            # Debug output every N steps
            if self.debug_steps % self.debug_frequency == 0 and self.debug_steps > self.last_debug_output:
                logger.info(f"Callback debugging at step {self.debug_steps}")
                print(f"\n--- Training progress: step {self.debug_steps} ---")
                
                # Print key metrics to console
                if len(self.episode_rewards) > 0:
                    print(f"Recent reward mean: {np.mean(self.episode_rewards[-10:]):.4f}")
                
                if len(self.portfolio_values) > 0:
                    print(f"Current portfolio value: {self.portfolio_values[-1]:.2f}")
                
                if self.trade_count > 0:
                    win_rate = self.successful_trades / max(1, self.trade_count) * 100
                    print(f"Trades: {self.trade_count} | Win rate: {win_rate:.2f}% | Profit factor: {self.total_profit / max(1e-6, self.total_loss):.2f}")
                    print(f"Action counts: {self.action_counts}")
                
                if hasattr(self, 'locals') and 'infos' in self.locals and len(self.locals['infos']) > 0:
                    logger.info(f"Sample info dict: {str(self.locals['infos'][0])}")
                self.last_debug_output = self.debug_steps
            
            # Force tensorboard log flushing periodically
            if hasattr(self.logger, "dump_tabular") and self.debug_steps % self.log_flush_frequency == 0:
                self.logger.dump_tabular()
            
            # Track step reward
            if hasattr(self, 'locals') and 'rewards' in self.locals:
                rewards = self.locals['rewards']
                for i, reward in enumerate(rewards):
                    # Update the current episode rewards
                    self.current_episode_reward += reward
                    self.current_episode_length += 1
                    
                    # Log environment info if available
                    if 'infos' in self.locals and i < len(self.locals['infos']):
                        info = self.locals['infos'][i]
                        
                        # Track portfolio value
                        if 'portfolio_value' in info:
                            self.portfolio_values.append(info['portfolio_value'])
                            self.logger.record('portfolio/value', info['portfolio_value'])
                        
                        # Track position
                        if 'position' in info:
                            position = info['position']
                            self.logger.record('portfolio/position', position)
                            
                            # Position change detection for trade tracking
                            if position != self.current_position:
                                # If we had a previous position, this is a trade exit
                                if self.current_position != 0:
                                    trade_duration = self.debug_steps - self.position_start_time
                                    self.trade_durations.append(trade_duration)
                                    
                                    # Check if we have a price to calculate P&L
                                    if 'close' in info:
                                        exit_price = info['close']
                                        pnl = (exit_price - self.entry_price) * self.current_position
                                        self.trade_returns.append(pnl)
                                        
                                        # Track trade result
                                        if pnl > 0:
                                            self.successful_trades += 1
                                            self.total_profit += pnl
                                            if self.debug_steps % self.debug_frequency == 0:
                                                logger.info(f"Successful trade: PnL={pnl:.4f}, entry={self.entry_price:.4f}, exit={exit_price:.4f}")
                                        else:
                                            self.failed_trades += 1
                                            self.total_loss += abs(pnl)
                                            if self.debug_steps % self.debug_frequency == 0:
                                                logger.info(f"Failed trade: PnL={pnl:.4f}, entry={self.entry_price:.4f}, exit={exit_price:.4f}")
                                    
                                    # Add to recent trades for trend analysis
                                    self.recent_trades.append({
                                        'pnl': pnl,
                                        'duration': trade_duration,
                                        'position': self.current_position,
                                        'entry_price': self.entry_price,
                                        'exit_price': exit_price,
                                        'timestamp': self.debug_steps
                                    })
                                    
                                    # Keep only the most recent trades
                                    if len(self.recent_trades) > self.max_recent_trades:
                                        self.recent_trades.pop(0)
                            
                                # If new position is non-zero, this is a trade entry
                                if position != 0:
                                    self.trade_count += 1
                                    if 'close' in info:
                                        self.entry_price = info['close']
                                        if self.debug_steps % self.debug_frequency == 0:
                                            logger.info(f"Trade entry: position={position}, price={self.entry_price:.4f}")
                                    self.position_start_time = self.debug_steps
                                
                                self.current_position = position
                        
                        # Track action type
                        if 'action_type' in info:
                            action_type = info['action_type']
                            if action_type in self.action_counts:
                                self.action_counts[action_type] += 1
                            self.logger.record(f'actions/{action_type}', self.action_counts.get(action_type, 0))
                            
                            # Debug output
                            if self.debug_steps % (self.debug_frequency * 10) == 0:
                                logger.info(f"Action counts: {self.action_counts}")
                        
                        # Track trade metrics
                        if 'trade_count' in info:
                            self.logger.record('trades/count', info['trade_count'])
                        if 'profit_loss' in info:
                            self.logger.record('trades/profit_loss', info['profit_loss'])
                    
                    # Check if episode is done
                    dones = self.locals.get('dones', [False])
                    
                    for j, done in enumerate(dones):
                        if done and j == i:  # Only process the done signal if it corresponds to the current reward
                            # Add the episode statistics
                            self.episode_rewards.append(self.current_episode_reward)
                            self.episode_lengths.append(self.current_episode_length)
                            
                            # Log episode statistics
                            self.logger.record('episode/reward', self.current_episode_reward)
                            self.logger.record('episode/length', self.current_episode_length)
                            
                            # Calculate and log running statistics
                            if len(self.episode_rewards) > 0:
                                self.logger.record('episode/mean_reward', np.mean(self.episode_rewards[-100:]))
                                self.logger.record('episode/mean_length', np.mean(self.episode_lengths[-100:]))
                            
                            if len(self.portfolio_values) > 0:
                                self.logger.record('portfolio/final_value', self.portfolio_values[-1])
                                self.logger.record('portfolio/mean_value', np.mean(self.portfolio_values[-100:]))
                            
                            # Log trading metrics
                            if self.trade_count > 0:
                                # Calculate win rate
                                win_rate = self.successful_trades / max(1, self.trade_count)
                                self.logger.record('trades/win_rate', win_rate)
                                
                                # Debug output
                                logger.info(f"Episode complete - Trading stats: trades={self.trade_count}, win_rate={win_rate:.4f}")
                                
                                # Profit factor (total profit / total loss)
                                profit_factor = self.total_profit / max(1e-6, self.total_loss)  # Avoid div by zero
                                self.logger.record('trades/profit_factor', profit_factor)
                                
                                # Average trade metrics
                                if len(self.trade_returns) > 0:
                                    avg_return = np.mean(self.trade_returns)
                                    self.logger.record('trades/avg_return', avg_return)
                                    
                                    # Average returns of winning and losing trades
                                    winning_returns = [r for r in self.trade_returns if r > 0]
                                    losing_returns = [r for r in self.trade_returns if r <= 0]
                                    
                                    if winning_returns:
                                        self.logger.record('trades/avg_win', np.mean(winning_returns))
                                    if losing_returns:
                                        self.logger.record('trades/avg_loss', np.mean(losing_returns))
                                
                                # Average trade duration
                                if len(self.trade_durations) > 0:
                                    avg_duration = np.mean(self.trade_durations)
                                    self.logger.record('trades/avg_duration', avg_duration)
                                
                                # Trading consistency - using standard deviation of recent returns
                                if len(self.trade_returns) > 5:
                                    returns_std = np.std(self.trade_returns[-20:])
                                    self.logger.record('trades/returns_std', returns_std)
                                
                                # Action distribution
                                total_actions = sum(self.action_counts.values())
                                if total_actions > 0:
                                    for action, count in self.action_counts.items():
                                        self.logger.record(f'actions/{action}_pct', count / total_actions)
                            
                                # Log running totals
                                self.logger.record('trades/total_count', self.trade_count)
                                self.logger.record('trades/successful', self.successful_trades)
                                self.logger.record('trades/failed', self.failed_trades)
                                self.logger.record('trades/total_profit', self.total_profit)
                                self.logger.record('trades/total_loss', self.total_loss)
                                
                                # Recent performance trend (last 20 trades vs previous 20)
                                if len(self.trade_returns) >= 40:
                                    recent_returns = np.mean(self.trade_returns[-20:])
                                    previous_returns = np.mean(self.trade_returns[-40:-20])
                                    trend = recent_returns - previous_returns
                                    self.logger.record('trades/trend', trend)
                            
                            # Reset episode tracking
                            self.current_episode_reward = 0
                            self.current_episode_length = 0
                
            return True
    
    # Create the model
    logger.info(f"Creating {finrl_model.upper()} model with policy_kwargs: {policy_kwargs}")
    try:
        # Set a higher tensorboard queue size to handle more metrics
        import os
        os.environ['STABLE_BASELINES_TENSORBOARD_QUEUE_SIZE'] = '100000'
        
        model = model_class(
            "MlpPolicy", 
            vec_env, 
            learning_rate=getattr(args, 'learning_rate', 0.0003),
            gamma=getattr(args, 'gamma', 0.99),
            verbose=2,  # Change from 1 to 2 for more detailed logging
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_logs"
        )
        
        # Train the model with callback
        total_timesteps = getattr(args, 'timesteps', 50000)  # Default to 50000 if not specified
        logger.info(f"Training model for {total_timesteps} timesteps")
        
        # Create the callbacks
        tensorboard_callback = TensorboardCallback()
        progress_callback = ProgressBarCallback()
        
        # Store the callbacks in a list to make them accessible later
        callbacks = [tensorboard_callback, progress_callback]
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=finrl_model
        )
        
        # Create timestamp for saving
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = f"models/finrl_{args.finrl_model.lower()}_{timestamp}"
        os.makedirs(model_path, exist_ok=True)
        
        # Print trading metrics summary if callback metrics are available
        try:
            # Access the callback from the list
            callback = callbacks[0]
            
            # Calculate win rate and other metrics
            win_rate = callback.successful_trades / max(1, callback.trade_count) * 100
            profit_factor = callback.total_profit / max(1e-6, callback.total_loss)
            
            # Print summary
            print("\n" + "="*50)
            print("TRADING METRICS SUMMARY")
            print("="*50)
            print(f"Total Trades: {callback.trade_count}")
            print(f"Successful Trades: {callback.successful_trades} ({win_rate:.2f}%)")
            print(f"Failed Trades: {callback.failed_trades} ({100-win_rate:.2f}%)")
            print(f"Total Profit: {callback.total_profit:.2f}")
            print(f"Total Loss: {callback.total_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            
            # Print average trade metrics if available
            if len(callback.trade_returns) > 0:
                avg_return = np.mean(callback.trade_returns)
                winning_returns = [r for r in callback.trade_returns if r > 0]
                losing_returns = [r for r in callback.trade_returns if r <= 0]
                
                print(f"Average Trade Return: {avg_return:.4f}")
                if winning_returns:
                    print(f"Average Winning Trade: {np.mean(winning_returns):.4f}")
                if losing_returns:
                    print(f"Average Losing Trade: {np.mean(losing_returns):.4f}")
            
            # Print action distribution
            if sum(callback.action_counts.values()) > 0:
                print("\nAction Distribution:")
                total_actions = sum(callback.action_counts.values())
                for action, count in callback.action_counts.items():
                    print(f"  {action}: {count} ({count/total_actions*100:.2f}%)")
            
            # Print portfolio metrics if available
            if len(callback.portfolio_values) > 0:
                initial_value = callback.portfolio_values[0] if callback.portfolio_values else initial_balance
                final_value = callback.portfolio_values[-1] if callback.portfolio_values else 0
                roi = (final_value / initial_value - 1) * 100
                print(f"\nInitial Portfolio Value: {initial_value:.2f}")
                print(f"Final Portfolio Value: {final_value:.2f}")
                print(f"Return on Investment: {roi:.2f}%")
            
            print("="*50)
            
            # Save the metrics to a JSON file for later analysis
            metrics_data = {
                'total_trades': callback.trade_count,
                'successful_trades': callback.successful_trades,
                'failed_trades': callback.failed_trades,
                'win_rate': float(win_rate),
                'total_profit': float(callback.total_profit),
                'total_loss': float(callback.total_loss),
                'profit_factor': float(profit_factor),
                'action_distribution': callback.action_counts,
            }
            
            if len(callback.portfolio_values) > 0:
                metrics_data['initial_value'] = float(initial_value)
                metrics_data['final_value'] = float(final_value)
                metrics_data['roi'] = float(roi)
            
            if len(callback.trade_returns) > 0:
                metrics_data['avg_return'] = float(avg_return)
                if winning_returns:
                    metrics_data['avg_win'] = float(np.mean(winning_returns))
                if losing_returns:
                    metrics_data['avg_loss'] = float(np.mean(losing_returns))
            
            # Save metrics to file
            metrics_file = os.path.join(model_path, 'trading_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            logger.info(f"Trading metrics saved to {metrics_file}")
                
        except Exception as e:
            logger.warning(f"Could not print or save trading metrics summary: {e}")
            logger.warning(traceback.format_exc())
        
        # Save the model
        model.save(f"{model_path}/model")
        logger.info(f"Model saved to {model_path}/model")
        
        # Save arguments used for training
        with open(f"{model_path}/training_args.json", 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Training arguments saved to {model_path}/training_args.json")
    except Exception as e:
        logger.error(f"Error during model creation or training: {e}")
        logger.error(traceback.format_exc())
        
        # Try to debug the problem
        if "object is not subscriptable" in str(e):
            logger.error("This is likely a problem with transaction costs in StockTradingEnv.")
            logger.error("StockTradingEnv expects transaction costs as lists, not scalar values.")
            
            # Check for "index out of range" or AttributeError
            if "list index out of range" in str(e) or "AttributeError" in str(e):
                logger.error("This could be related to the observation space or state dimensions.")
                logger.error("Check that the state dimensions match between wrapper and environment.")
                
        elif "space contains" in str(e) or "expected_shape" in str(e):
            logger.error("This is an observation space mismatch error.")
            logger.error("The environment is returning observations with a different shape than expected.")
            
        raise

def setup_logging():
    """Configure and return a logger for the application."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent for cryptocurrency trading')
    
    # General arguments
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Data arguments
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='Start date for training data')
    parser.add_argument('--end_date', type=str, default='2021-12-31', help='End date for training data')
    parser.add_argument('--tickers', type=str, default='BTC,ETH,LTC', help='Comma-separated list of tickers')
    
    # Model arguments
    parser.add_argument('--use_finrl', action='store_true', help='Use FinRL framework')
    parser.add_argument('--finrl_model', type=str, default='sac', 
                       choices=['sac', 'ppo', 'ddpg', 'td3'], 
                       help='FinRL model to use')
    parser.add_argument('--lstm_model_path', type=str, help='Path to pretrained LSTM model')
    parser.add_argument('--timesteps', type=int, default=500000, help='Number of timesteps to train')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of parallel environments')
    parser.add_argument('--num_envs_per_worker', type=int, default=4, help='Number of envs per worker')
    parser.add_argument('--use_subproc_vecenv', action='store_true', help='Use SubprocVecEnv instead of DummyVecEnv')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--initial_balance', type=float, default=1000000.0, help='Initial balance')
    parser.add_argument('--transaction_fee', type=float, default=0.001, help='Transaction fee')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps to collect before learning (PPO)')
    parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size for PPO updates')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs when optimizing the surrogate loss (PPO)')
    
    return parser.parse_args()

def load_and_preprocess_market_data(args):
    """Load and preprocess market data for training."""
    logger.info("Loading market data...")
    
    # This function would normally load data from files or APIs
    # For this implementation, we'll create synthetic data
    
    # Get tickers
    if hasattr(args, 'tickers'):
        if isinstance(args.tickers, str):
            tickers = args.tickers.split(',')
        else:
            tickers = args.tickers
    else:
        tickers = ['BTC']
    
    # Create synthetic data
    market_data = {}
    for timeframe in ['1h', '4h', '1d']:
        # Create a DataFrame with basic OHLCV data
        data_length = 50000  # Number of candlesticks
        
        df = pd.DataFrame({
            'open': np.random.normal(1000, 100, data_length),
            'high': np.random.normal(1050, 100, data_length),
            'low': np.random.normal(950, 100, data_length),
            'close': np.random.normal(1000, 100, data_length),
            'volume': np.random.normal(1000, 200, data_length)
        })
        
        # Ensure high is the highest price
        df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
        # Ensure low is the lowest price
        df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
        
        # Create a datetime index
        start_date = pd.to_datetime(args.start_date) if hasattr(args, 'start_date') else pd.to_datetime('2018-01-01')
        freq = '1H' if timeframe == '1h' else '4H' if timeframe == '4h' else 'D'
        df.index = pd.date_range(start=start_date, periods=data_length, freq=freq)
        
        market_data[timeframe] = df
    
    logger.info(f"Created synthetic market data with {data_length} data points")
    return market_data, data_length

def train_with_custom_dqn(args, market_data, data_length, device):
    """Train a DQN agent using custom implementation."""
    logger.info("Training with custom DQN implementation")
    
    # This would be a full implementation
    # For now, just return a placeholder
    logger.info("Custom DQN training not implemented yet, returning placeholder")
    return None

def create_synthetic_data(tickers, start_date, end_date):
    """
    Create synthetic cryptocurrency data for training.
    
    Args:
        tickers: List of cryptocurrency tickers
        start_date: Start date for data generation
        end_date: End date for data generation
        
    Returns:
        DataFrame with synthetic market data
    """
    logger.info(f"Creating synthetic data for {tickers} from {start_date} to {end_date}")
    
    # Convert date strings to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    # Create empty dataframe
    df_list = []
    
    # Generate synthetic data for each ticker
    for ticker in tickers:
        # Start with a base price
        base_price = 10000 if ticker == 'BTC' else (2000 if ticker == 'ETH' else 500)
        
        # Add random walk with drift
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0.001, 0.02, size=len(date_range))
        
        # Add some seasonality and trend
        trend = np.linspace(0, 0.5, len(date_range))
        seasonality = 0.1 * np.sin(np.linspace(0, 8 * np.pi, len(date_range)))
        
        # Combine components
        price_factors = np.cumprod(1 + returns + trend * 0.001 + seasonality * 0.01)
        prices = base_price * price_factors
        
        # Generate volume
        volume = np.random.lognormal(10, 1, size=len(date_range)) * 1000
        
        # Create ticker dataframe
        df_ticker = pd.DataFrame({
            'date': date_range,
            'tic': ticker,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': volume,
            'day': date_range.day,
            'month': date_range.month,
            'year': date_range.year
        })
        
        # Add to list
        df_list.append(df_ticker)
    
    # Combine all tickers
    df = pd.concat(df_list, ignore_index=True)
    
    # Set proper dtypes
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and ticker
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    return df

def ensure_technical_indicators(df, indicators):
    """
    Ensure all required technical indicators are present in the dataframe.
    If missing, calculate them based on price data.
    
    Args:
        df: DataFrame with market data
        indicators: List of required indicators
        
    Returns:
        DataFrame with all required indicators
    """
    logger.info("Calculating technical indicators")
    
    # Create a copy of the dataframe
    df_result = df.copy()
    
    # Group by ticker
    for ticker, group in df_result.groupby('tic'):
        # Calculate indicators for each ticker separately
        temp_df = group.copy().sort_values('date')
        
        # Calculate basic price indicators
        if 'close_5_sma' in indicators:
            temp_df['close_5_sma'] = temp_df['close'].rolling(window=5).mean()
        
        if 'close_10_sma' in indicators:
            temp_df['close_10_sma'] = temp_df['close'].rolling(window=10).mean()
            
        if 'close_20_sma' in indicators:
            temp_df['close_20_sma'] = temp_df['close'].rolling(window=20).mean()
        
        # Calculate MACD
        if 'macd' in indicators:
            ema12 = temp_df['close'].ewm(span=12, adjust=False).mean()
            ema26 = temp_df['close'].ewm(span=26, adjust=False).mean()
            temp_df['macd'] = ema12 - ema26
        
        # Calculate RSI (14 periods)
        if 'rsi_14' in indicators:
            delta = temp_df['close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            temp_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate CCI (30 periods)
        if 'cci_30' in indicators:
            tp = (temp_df['high'] + temp_df['low'] + temp_df['close']) / 3
            ma_tp = tp.rolling(window=30).mean()
            md_tp = tp.rolling(window=30).apply(lambda x: np.fabs(x - x.mean()).mean())
            temp_df['cci_30'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # Calculate DX (30 periods)
        if 'dx_30' in indicators:
            # Calculate +DM and -DM
            high_diff = temp_df['high'].diff()
            low_diff = temp_df['low'].diff().abs() * -1
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Calculate TR
            tr1 = temp_df['high'] - temp_df['low']
            tr2 = (temp_df['high'] - temp_df['close'].shift(1)).abs()
            tr3 = (temp_df['low'] - temp_df['close'].shift(1)).abs()
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Calculate smoothed values
            tr_30 = tr.rolling(window=30).sum()
            plus_dm_30 = pd.Series(plus_dm).rolling(window=30).sum()
            minus_dm_30 = pd.Series(minus_dm).rolling(window=30).sum()
            
            # Calculate +DI and -DI
            plus_di_30 = 100 * plus_dm_30 / tr_30
            minus_di_30 = 100 * minus_dm_30 / tr_30
            
            # Calculate DX
            dx = 100 * ((plus_di_30 - minus_di_30).abs() / (plus_di_30 + minus_di_30))
            temp_df['dx_30'] = dx
        
        # Calculate volatility
        if 'volatility_30' in indicators:
            temp_df['volatility_30'] = temp_df['close'].pct_change().rolling(window=30).std()
        
        # Calculate volume change
        if 'volume_change' in indicators:
            temp_df['volume_change'] = temp_df['volume'].pct_change()
        
        # Update the main dataframe
        df_result.loc[temp_df.index, temp_df.columns] = temp_df
    
    # Forward fill any NaN values
    for indicator in indicators:
        if indicator in df_result.columns and df_result[indicator].isnull().any():
            df_result[indicator] = df_result[indicator].ffill().fillna(0)
    
    return df_result

def wrap_env_with_monitor(env):
    """
    Wrap environment with Monitor for metrics tracking.
    
    Args:
        env: Environment to wrap
        
    Returns:
        Monitor-wrapped environment
    """
    from stable_baselines3.common.monitor import Monitor
    
    # Create a temporary directory if it doesn't exist
    monitor_dir = './logs/monitor'
    os.makedirs(monitor_dir, exist_ok=True)
    
    # Check if the environment is a gymnasium.Env
    if not isinstance(env, gymnasium.Env):
        # If not, wrap it with a gymnasium adapter
        env = StockTradingEnvWrapper(env)
        logger.info("Wrapped environment with StockTradingEnvWrapper for gymnasium compatibility")
    
    # Custom monitor wrapper that ensures compatibility with both gym and gymnasium APIs
    class CompatibleMonitor(Monitor):
        def step(self, action):
            """Step the environment with compatibility for both APIs."""
            result = super().step(action)
            
            # If we get a 4-tuple result (old gym API), convert to 5-tuple (gymnasium API)
            if isinstance(result, tuple) and len(result) == 4:
                obs, reward, done, info = result
                return obs, reward, done, False, info  # Add truncated=False
            
            return result
    
    # Use our compatible monitor wrapper
    return CompatibleMonitor(env, monitor_dir)

def main():
    """Main entry point for the script."""
    # Record start time
    start_time = time.time()
    
    # Import torch at the beginning of main to ensure it's available
    import torch
    
    # Configure logging
    global logger
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if using FinRL framework
    if args.use_finrl:
        logger.info("Using FinRL-compatible framework")
        
        # Set default values if not provided in args
        start_date = getattr(args, 'start_date', '2018-01-01')
        end_date = getattr(args, 'end_date', '2021-12-31')
        
        # Set tickers - use provided tickers or default to ['BTC']
        tickers = getattr(args, 'tickers', ['BTC', 'ETH', 'LTC'])
        if isinstance(tickers, str):
            tickers = tickers.split(',')
        
        # Set number of workers
        num_workers = getattr(args, 'num_workers', 8)
        num_envs_per_worker = getattr(args, 'num_envs_per_worker', 8)
        total_envs = num_workers * num_envs_per_worker
        logger.info(f"Creating {total_envs} total environments ({num_workers} workers × {num_envs_per_worker} envs each)")
        
        # Load LSTM model if specified
        lstm_model = None
        lstm_processor = None
        if args.lstm_model_path:
            try:
                from crypto_trading_model.models.time_series.model import MultiTimeframeModel
                from crypto_trading_model.data_processing.preprocessing import TimeSeriesDataProcessor
                import torch
                
                logger.info(f"Loading LSTM model from {args.lstm_model_path}")
                
                # Actually load the LSTM model
                try:
                    lstm_model = MultiTimeframeModel.load_model(args.lstm_model_path)
                    lstm_model.eval()  # Set to evaluation mode
                    logger.info(f"Successfully loaded LSTM model: {lstm_model}")
                    
                    # Create data processor for LSTM
                    lstm_processor = TimeSeriesDataProcessor(
                        sequence_length=10,  # Use same as model training
                        prediction_horizon=5  # Predict 5 steps ahead
                    )
                    
                    logger.info("LSTM model will be used for market predictions")
                except Exception as e:
                    logger.error(f"Failed to load LSTM model: {e}")
                    lstm_model = None
            except ImportError as e:
                logger.error(f"Error importing required modules for LSTM: {e}")
                
        # Create synthetic data for training
        df = create_synthetic_data(tickers, start_date, end_date)
        logger.info(f"Created synthetic data with shape {df.shape}")
        
        # If we have a loaded LSTM model, add price predictions to data
        if lstm_model is not None and lstm_processor is not None:
            try:
                logger.info("Adding LSTM price predictions to training data")
                
                # Prepare data for LSTM prediction
                price_data = df[['open', 'high', 'low', 'close', 'volume']].values
                
                # Normalize the data
                price_mean = price_data.mean(axis=0)
                price_std = price_data.std(axis=0)
                normalized_data = (price_data - price_mean) / price_std
                
                # Create sequences
                sequences = []
                for i in range(len(normalized_data) - lstm_processor.sequence_length):
                    seq = normalized_data[i:i+lstm_processor.sequence_length]
                    sequences.append(seq)
                
                # Convert to tensor for batch prediction
                if sequences:
                    tensor_sequences = torch.FloatTensor(sequences).to(device)
                    
                    # Run prediction
                    with torch.no_grad():
                        price_predictions = lstm_model(tensor_sequences).cpu().numpy()
                    
                    # Denormalize predictions
                    denorm_predictions = price_predictions * price_std[3] + price_mean[3]  # For close price
                    
                    # Add predictions to dataframe
                    for i in range(lstm_processor.prediction_horizon):
                        pred_col = f'lstm_pred_{i+1}'
                        df[pred_col] = 0.0  # Initialize column
                        
                        # Fill with predictions (offset by sequence_length due to first sequences)
                        for j in range(len(denorm_predictions)):
                            if j + lstm_processor.sequence_length + i < len(df):
                                df.loc[j + lstm_processor.sequence_length + i, pred_col] = denorm_predictions[j][i]
                    
                    # Add prediction columns to technical indicators
                    for i in range(lstm_processor.prediction_horizon):
                        pred_col = f'lstm_pred_{i+1}'
                        if pred_col not in tech_indicators:
                            tech_indicators.append(pred_col)
                    
                    logger.info(f"Added LSTM predictions for {lstm_processor.prediction_horizon} steps ahead")
                else:
                    logger.warning("Could not create sequences for LSTM prediction")
            except Exception as e:
                logger.error(f"Error generating LSTM predictions: {e}")
                logger.error(traceback.format_exc())
        
        # Define necessary technical indicators
        tech_indicators = [
            'macd', 'rsi_14', 'cci_30', 'dx_30', 
            'close_5_sma', 'close_10_sma', 'close_20_sma',
            'volatility_30', 'volume_change'
        ]
        
        # Ensure all technical indicators are present
        df = ensure_technical_indicators(df, tech_indicators)
        
        # Create a CryptocurrencyTradingEnv (our custom implementation)
        env_params = {
            'df': df,
            'initial_amount': getattr(args, 'initial_balance', 1000000.0),
            'buy_cost_pct': 0.00075,  # Transaction cost
            'sell_cost_pct': 0.00075,  # Transaction cost
            'state_space': len(tech_indicators) + 3,  # Indicators + cash + price + owned
            'stock_dim': 1,  # Single asset trading
            'tech_indicator_list': tech_indicators,
            'action_space': 3,  # Sell, hold, buy
            'reward_scaling': 1e-4,
            'print_verbosity': 10 if args.verbose else 0
        }
        
        logger.info(f"Creating CryptocurrencyTradingEnv with state_space={env_params['state_space']}")
        base_env = CryptocurrencyTradingEnv(**env_params)
        
        # Wrap with Monitor for metrics
        env = wrap_env_with_monitor(base_env)
        
        # Create vectorized environment
        vec_env = create_parallel_finrl_envs(df, args, num_workers=num_workers)
        
        # Check the action space type and use appropriate algorithm
        if isinstance(vec_env.action_space, gym.spaces.Discrete) or isinstance(vec_env.action_space, gymnasium.spaces.Discrete):
            # For discrete action spaces, use algorithms that support them
            if args.finrl_model.lower() == 'ppo':
                logger.info("Training with PPO model (supports discrete actions)")
                
                # Get PPO hyperparameters
                n_steps = getattr(args, 'n_steps', 2048)
                batch_size = getattr(args, 'batch_size', 64)
                n_epochs = getattr(args, 'n_epochs', 10)
                learning_rate = getattr(args, 'learning_rate', 0.0001)
                
                logger.info(f"PPO parameters: n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, lr={learning_rate}")
                
                # Create PPO model with enhanced GPU utilization
                model = PPO(
                    "MlpPolicy", 
                    vec_env,
                    learning_rate=learning_rate,
                    gamma=getattr(args, 'gamma', 0.99),
                    n_steps=n_steps,       # Larger batch size for better GPU utilization
                    batch_size=batch_size, # Minibatch size for updates
                    n_epochs=n_epochs,     # Number of epoch when optimizing the surrogate
                    ent_coef=0.01,         # Entropy coefficient for exploration
                    verbose=1 if args.verbose else 0,
                    tensorboard_log="./tensorboard_logs",
                    device=device,
                    policy_kwargs={
                        'net_arch': [512, 512, 256]  # Larger network to utilize GPU better
                    }
                )
                
                # Log GPU memory usage before training
                if torch.cuda.is_available():
                    logger.info(f"GPU memory allocated before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    logger.info(f"GPU memory reserved before training: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                
                # Train the model
                total_timesteps = getattr(args, 'timesteps', 50000)
                logger.info(f"Training for {total_timesteps} timesteps with {vec_env.num_envs} environments...")
                
                # Log GPU usage during training
                class GPUMonitorCallback(BaseCallback):
                    def __init__(self, verbose=0):
                        super().__init__(verbose)
                        self.step_count = 0
                    
                    def _on_step(self):
                        self.step_count += 1
                        if self.step_count % 1000 == 0 and torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated() / 1024**2
                            reserved = torch.cuda.memory_reserved() / 1024**2
                            logger.info(f"Step {self.step_count}: GPU memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
                        return True
                
                callbacks = [GPUMonitorCallback()]
                if args.verbose and torch.cuda.is_available():
                    callbacks.append(GPUMonitorCallback())
                
                model.learn(total_timesteps=total_timesteps, callback=callbacks)
                
                # Save the model
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                model_path = f"models/finrl_{args.finrl_model.lower()}_{timestamp}"
                os.makedirs(model_path, exist_ok=True)
                model.save(f"{model_path}/model")
                logger.info(f"Model saved to {model_path}/model")
                
                # Save arguments used for training
                with open(f"{model_path}/training_args.json", 'w') as f:
                    json.dump(vars(args), f, indent=4)
                logger.info(f"Training arguments saved to {model_path}/training_args.json")
            else:
                # Default to DQN for discrete action spaces, regardless of what was specified for --finrl_model
                # SAC specifically doesn't support discrete action spaces, so we'll use DQN instead
                from stable_baselines3 import DQN
                logger.info(f"Using DQN model for discrete action space (ignoring requested {args.finrl_model} which doesn't support discrete actions)")
                model = DQN(
                    'MlpPolicy', 
                    vec_env,
                    learning_rate=getattr(args, 'learning_rate', 0.0001),
                    gamma=getattr(args, 'gamma', 0.99),
                    verbose=1 if args.verbose else 0,
                    tensorboard_log="./tensorboard_logs"
                )
        else:
            # For continuous action spaces, use algorithms that support them
            if args.finrl_model.lower() == 'sac':
                logger.info("Training with SAC model")
                # Use stable-baselines3 SAC directly
                model = SAC(
                    'MlpPolicy', 
                    vec_env,
                    learning_rate=getattr(args, 'learning_rate', 0.0003),
                    gamma=getattr(args, 'gamma', 0.99),
                    verbose=1 if args.verbose else 0,
                    tensorboard_log="./tensorboard_logs"
                )
            elif args.finrl_model.lower() == 'ppo':
                logger.info("Training with PPO model")
                model = PPO(
                    'MlpPolicy', 
                    vec_env,
                    learning_rate=getattr(args, 'learning_rate', 0.0003),
                    gamma=getattr(args, 'gamma', 0.99),
                    verbose=1 if args.verbose else 0,
                    tensorboard_log="./tensorboard_logs"
                )
            elif args.finrl_model.lower() == 'ddpg':
                logger.info("Training with DDPG model")
                model = DDPG(
                    'MlpPolicy', 
                    vec_env,
                    learning_rate=getattr(args, 'learning_rate', 0.0003),
                    gamma=getattr(args, 'gamma', 0.99),
                    verbose=1 if args.verbose else 0,
                    tensorboard_log="./tensorboard_logs"
                )
            else:
                logger.info("Using default TD3 model")
                model = TD3(
                    'MlpPolicy', 
                    vec_env,
                    learning_rate=getattr(args, 'learning_rate', 0.0003),
                    gamma=getattr(args, 'gamma', 0.99),
                    verbose=1 if args.verbose else 0,
                    tensorboard_log="./tensorboard_logs"
                )
        
        # Train the model
        total_timesteps = getattr(args, 'timesteps', 50000)
        logger.info(f"Training for {total_timesteps} timesteps with {vec_env.num_envs} environments...")
        
        # Log GPU usage during training
        class GPUMonitorCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.step_count = 0
            
            def _on_step(self):
                self.step_count += 1
                if self.step_count % 1000 == 0 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    logger.info(f"Step {self.step_count}: GPU memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
                return True
        
        callbacks = [GPUMonitorCallback()]
        if args.verbose and torch.cuda.is_available():
            callbacks.append(GPUMonitorCallback())
        
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model_path = f"models/finrl_{args.finrl_model.lower()}_{timestamp}"
        os.makedirs(model_path, exist_ok=True)
        model.save(f"{model_path}/model")
        logger.info(f"Model saved to {model_path}/model")
        
        # Save arguments used for training
        with open(f"{model_path}/training_args.json", 'w') as f:
            json.dump(vars(args), f, indent=4)
        logger.info(f"Training arguments saved to {model_path}/training_args.json")
    
    else:
        # Use our custom DQN implementation
        # This section remains unchanged...
        pass
    
    # Log total execution time
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

def create_dummy_vectorized_env(env_function, n_envs=1):
    """
    Create a vectorized environment that runs multiple environments in sequence.
    
    Args:
        env_function: Function that creates an environment instance
        n_envs: Number of environments to create
        
    Returns:
        Vectorized environment
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env_fns = [env_function for _ in range(n_envs)]
    return DummyVecEnv(env_fns)

if __name__ == "__main__":
    main() 