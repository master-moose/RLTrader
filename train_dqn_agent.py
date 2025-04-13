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
            
            # Transaction costs - add penalties for trading to discourage excessive trading
            # Default transaction cost is 0.1% of the trade value
            transaction_cost = getattr(self, 'transaction_cost_pct', 0.001)
            trading_penalty = 0
            
            # Action is 0 (sell), 1 (hold), or 2 (buy)
            if action == 0:  # Sell
                reward = -price_change  # Reward for selling is positive when price goes down
                info["action_type"] = "sell"
                # Apply transaction cost penalty for selling
                trading_penalty = -transaction_cost * 10
            elif action == 2:  # Buy
                reward = price_change  # Reward for buying is positive when price goes up
                info["action_type"] = "buy"
                # Apply transaction cost penalty for buying
                trading_penalty = -transaction_cost * 10
            else:  # Hold
                # Small reward/penalty for holding based on whether price is trending up or down
                reward = price_change * 0.1
                info["action_type"] = "hold"
                # No transaction cost for holding
            
            # Add trading penalty to reward
            reward += trading_penalty
            info["trading_penalty"] = trading_penalty
            
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
    Enhanced version of the StockTradingEnv to fix issues and add features
    """
    def __init__(self, *args, **kwargs):
        self.debug_mode = kwargs.pop('debug_mode', False)
        self.trade_cooldown = kwargs.pop('trade_cooldown', 10)  # Default cooldown of 10 steps
        
        # Initialize trading tracking variables
        self._last_trade_step = None
        self._last_trade_price = {}  # Track last trade price per asset
        self.info_buffer = {}  # Buffer for additional info
        
        super().__init__(*args, **kwargs)
        
        # Log creation of environment
        logger.info(f"Created PatchedStockTradingEnv with {self.stock_dim} assets and trade cooldown of {self.trade_cooldown}")
        
    def step(self, actions):
        """
        Override step method to track trades and provide enhanced logging
        """
        begin_total_asset = self.get_total_asset_value()
        
        # Detect if this step includes a trade
        current_holdings = self.state[1:self.stock_dim+1].copy()
        
        # Call parent step method
        next_state, reward, done, info = super().step(actions)
        
        # Update reward based on our custom calculation
        end_total_asset = self.get_total_asset_value()
        reward = self._calculate_reward(begin_total_asset, end_total_asset)
        
        # Check if trade occurred by comparing holdings before and after
        new_holdings = self.state[1:self.stock_dim+1]
        trade_occurred = not np.array_equal(current_holdings, new_holdings)
        
        if trade_occurred:
            # Update last trade step
            self._last_trade_step = self.current_step
            
            # Update last trade prices
            for i in range(self.stock_dim):
                if current_holdings[i] != new_holdings[i]:
                    # Trade occurred for this asset
                    self._last_trade_price[i] = self.state[self.stock_dim + 1 + i * self.feature_dimension]
                    
                    if self.debug_mode and self.current_step % 500 == 0:
                        change = new_holdings[i] - current_holdings[i]
                        action_type = "BUY" if change > 0 else "SELL"
                        logger.info(f"Trade detected at step {self.current_step}: {action_type} asset {i} at price {self._last_trade_price[i]:.4f}")
        
        # Enhance info dictionary
        info.update(self.info_buffer)
        self.info_buffer = {}  # Reset buffer
        
        # Add trade tracking info
        info['trade_occurred'] = trade_occurred
        if trade_occurred:
            info['steps_since_last_trade'] = 0
        elif self._last_trade_step is not None:
            info['steps_since_last_trade'] = self.current_step - self._last_trade_step
            
        return next_state, reward, done, info
        
    def reset(self):
        """
        Override reset to initialize trading variables
        """
        self._last_trade_step = None
        self._last_trade_price = {}
        self.info_buffer = {}
        return super().reset()
        
    def get_total_asset_value(self):
        """
        Calculate the total asset value (cash + stocks)
        """
        if self.stock_dim > 0:
            return self.state[0] + sum(
                self.state[1:self.stock_dim+1] * self.state[self.stock_dim+1:self.stock_dim+1+self.stock_dim*self.feature_dimension:self.feature_dimension]
            )
        return self.state[0]  # Just cash

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
        'reward_scaling': getattr(args, 'reward_scaling', 1e-4),  # Get reward scaling from args
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
        df: DataFrame with market data in FinRL format
        args: Training arguments
        num_workers: Number of workers to use
        
    Returns:
        Vectorized environment containing multiple FinRL environments
    """
    logger.info(f"Creating parallel FinRL environment with {num_workers} workers")
    
    # Determine if we should use subprocess vectorization
    use_subproc = getattr(args, 'use_subproc', False)
    num_envs_per_worker = getattr(args, 'num_envs_per_worker', 1)
    
    # Initialize list of environment functions
    env_list = []
    
    # Set device based on args
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
        'macd', 'rsi_14', 
        'stoch_k', 'stoch_d',
        'cci_30', 'dx_30',
        'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_60_sma', 'close_120_sma',
        'close_5_ema', 'close_10_ema', 'close_20_ema', 'close_60_ema', 'close_120_ema',
        'volatility_30', 'volume_change', 'volume_norm'
    ]
    
    # Find which indicators are present in the dataframe
    tech_indicator_list = []
    
    logger.info("Checking for existing technical indicators in the dataframe")
    existing_indicators = [ind for ind in potential_indicators if ind in df.columns]
    if existing_indicators:
        tech_indicator_list = existing_indicators
        logger.info(f"Found {len(existing_indicators)} technical indicators already in dataframe: {existing_indicators}")
    else:
        logger.warning("No valid technical indicators found in dataframe. Adding basic indicators.")
        df = ensure_technical_indicators(df, potential_indicators)
        # Check again after ensuring indicators
        tech_indicator_list = [ind for ind in potential_indicators if ind in df.columns]
        logger.info(f"Added technical indicators: {tech_indicator_list}")
                
    # Calculate environment dimensions
    state_space = getattr(args, 'state_dim', 16)  # Default state space size if not specified
    stock_dim = 1  # Assuming single-asset trading for simplicity
    num_stocks = len(df['tic'].unique()) if 'tic' in df.columns else 1
    
    # Calculate action space size
    include_cash = getattr(args, 'include_cash', False)
    if include_cash:
        action_space = num_stocks + 1  # +1 for cash position
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
    
    # Calculate maximum number of shares to trade per step (hmax)
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
        # The DataFrame should be sorted by 'day' and 'tic'
        if 'day' in df.columns and 'tic' in df.columns:
            df = df.sort_values(['day', 'tic']).reset_index(drop=True)
            
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
                    df[col] = 10000.0  # Dummy volume
                elif col == 'tic':
                    logger.info(f"Creating '{col}' column with default ticker")
                    df[col] = 'BTC'  # Default ticker
                elif col == 'day':
                    logger.info(f"Creating '{col}' column with sequential days")
                    df[col] = np.arange(len(df))  # Sequential days
        
        # If 'day' column exists but is not numeric, convert it
        if pd.api.types.is_categorical_dtype(df['day']) or pd.api.types.is_object_dtype(df['day']):
            logger.info("Converting 'day' column to integer")
            df['day'] = df['day'].factorize()[0]
        
        # Ensure numeric columns have finite values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume'] + tech_indicator_list
        for col in numeric_cols:
            if np.any(~np.isfinite(df[col])):
                logger.warning(f"Column '{col}' contains non-finite values. Replacing with zeros.")
                df[col] = df[col].replace([np.inf, -np.inf, np.nan], 0)
        
        # Create a test environment to see if the format is correct
        logger.info(f"Creating test environment with indicators: {tech_indicator_list}")
        
        # Select the first day's data for testing
        first_day = df['day'].min()
        test_df = df[df['day'] == first_day].reset_index(drop=True)
        
        try:
            # Create test environment
            test_env = StockTradingEnvClass(
                df=test_df,
                stock_dim=stock_dim,
                hmax=hmax,
                initial_amount=initial_amount,
                transaction_cost_pct=[transaction_cost_pct] * stock_dim,  # List of costs for each stock
                buy_cost_pct=[transaction_cost_pct] * stock_dim,  # Buy costs
                sell_cost_pct=[transaction_cost_pct] * stock_dim,  # Sell costs
                reward_scaling=reward_scaling,
                state_space=state_space,
                action_space=action_space,
                tech_indicator_list=tech_indicator_list,
                print_verbosity=1,
            )
            
            # Try to reset the environment to see if it works
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
            traceback.print_exc()
    except Exception as e:
        logger.error(f"Error processing DataFrame: {e}")
        traceback.print_exc()
    
    # Create environment functions
    
    for i in range(num_workers * num_envs_per_worker):
        # Define a function that creates a new environment instance each time it's called
        def make_env(idx=i):
            try:
                # Create basic environment parameters
                env_config = {
                    'df': df.copy(),
                    'stock_dim': stock_dim,
                    'hmax': hmax,
                    'num_stock_shares': [0] * stock_dim,  # Start with no shares - REQUIRED parameter
                    'action_space': action_space,  # Make sure this is passed!
                    'tech_indicator_list': tech_indicator_list,
                    'initial_amount': initial_amount,
                    # 'transaction_cost_pct': transaction_cost_pct_list,  # Remove - not accepted by FinRL
                    'buy_cost_pct': [transaction_cost_pct] * stock_dim,  # Pass as list
                    'sell_cost_pct': [transaction_cost_pct] * stock_dim,  # Pass as list
                    'reward_scaling': getattr(args, 'reward_scaling', 1e-4),  # Get reward scaling from args
                    'state_space': state_space,  # Add the missing state_space parameter
                    'print_verbosity': 1 if idx == 0 else 0  # Only print verbose output for the first env
                }
                
                # Create the environment
                base_env = StockTradingEnvClass(**env_config)
                
                # Set environment ID for debugging
                base_env.env_id = idx  # Add an env_id attribute for logging
                
                # Wrap the environment to ensure consistent observation shape
                wrapped_env = StockTradingEnvWrapper(base_env, state_space=state_space)
                
                # Wrap with monitoring wrapper for metrics
                return wrap_env_with_monitor(wrapped_env)
            except Exception as e:
                # Log the error but provide a fallback environment
                logger.error(f"Error creating environment {idx}: {e}")
                
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
        market_data: Dictionary of DataFrames by timeframe
        primary_timeframe: Primary timeframe to use
        
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
    df,
    args,
    model_name="ppo",
    timesteps=10000,
    tensorboard_log="./tensorboard_log",
    device=None
):
    """Train a model using the FinRL library.
    
    Args:
        df: DataFrame with market data
        args: Training arguments
        model_name: Name of model to use (ppo, ddpg, td3, sac)
        timesteps: Number of timesteps to train for
        tensorboard_log: Directory for tensorboard logs
        device: PyTorch device to use
        
    Returns:
        Trained model
    """
    try:
        # Import stable-baselines3 and torch.nn
        from stable_baselines3 import PPO, DDPG, TD3, SAC
        import torch.nn as nn
        import torch
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
        import gymnasium
        import numpy as np
        import os
        
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        raise
        
    # Get parameters from args
    logger.info(f"Training with FinRL using model: {model_name}")
    finrl_model = args.finrl_model.lower()

    # Environment parameters 
    initial_balance = getattr(args, 'initial_balance', 1000000.0)
    state_dim = getattr(args, 'state_dim', 16)  # Default to 16 if not specified
    
    # Use the preprocessed data
    if df is not None:
        logger.info(f"Using provided DataFrame for training with shape: {df.shape}")
        logger.info(f"Sample of input data:\n{df.head(2)}")
    else:
        logger.error("No training data provided")
        return None
    
    # Create parallel environments for training
    num_envs = getattr(args, 'num_envs', 4)
    logger.info(f"Creating {num_envs} parallel environments for training")
    vec_env = create_parallel_finrl_envs(df, args, num_workers=num_envs)
    
    try:
        # Test the vectorized environment
        test_obs = vec_env.reset()
        actual_obs_dim = test_obs.shape[1]
        logger.info(f"Vectorized environment observation shape: {test_obs.shape}")
        
        if actual_obs_dim != state_dim:
            logger.warning(f"Observation dimension mismatch: got {actual_obs_dim}, expected {state_dim}")
            logger.info(f"Adjusting policy network to use actual observation dimension: {actual_obs_dim}")
            # Update state_dim to actual observation dimension
            state_dim = actual_obs_dim
    except Exception as e:
        logger.error(f"Error testing vectorized environment: {e}")
    
    # Set up network architecture
    net_arch = {
        'pi': [128, 128],  # Actor network - smaller than default
        'qf': [128, 128]   # Critic network - smaller than default
    }
    
    # Improved PPO parameters for better exploration and learning
    ppo_params = {
        'learning_rate': 3e-4,  # Increased from default
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,  # Don't clip value function
        'normalize_advantage': True,
        'ent_coef': 0.01,  # Increase entropy coefficient for more exploration
        'vf_coef': 0.5,  # Increase value function coefficient
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': 0.01,  # Early stopping based on KL divergence
        'policy_kwargs': {
            'activation_fn': nn.ReLU,
            'net_arch': net_arch,
            'log_std_init': -2.0  # Set initial log standard deviation for more controlled exploration
        }
    }
    
    # DDPG parameters
    ddpg_params = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': (1, 'episode'),
        'gradient_steps': -1,
        'policy_kwargs': {
            'activation_fn': nn.ReLU,
            'net_arch': [256, 256]
        }
    }
    
    # TD3 parameters
    td3_params = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'policy_delay': 2,
        'train_freq': (1, 'episode'),
        'gradient_steps': -1,
        'policy_kwargs': {
            'activation_fn': nn.ReLU,
            'net_arch': [400, 300]
        }
    }
    
    # SAC parameters
    sac_params = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
        'policy_kwargs': {
            'activation_fn': nn.ReLU,
            'net_arch': [256, 256]
        }
    }
    
    # Select parameters based on model
    if finrl_model == 'ppo':
        model_params = ppo_params
        model_class = PPO
    elif finrl_model == 'ddpg':
        model_params = ddpg_params
        model_class = DDPG
    elif finrl_model == 'td3':
        model_params = td3_params
        model_class = TD3
    else:  # Default to SAC
        model_params = sac_params
        model_class = SAC
        
    # Define callback to log metrics to tensorboard
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)
            # Track episode rewards and metadata
            self.episode_rewards = []
            self.portfolio_values = []
            self.total_profit = 0
            self.total_loss = 0
            self.successful_trades = 0
            self.failed_trades = 0
            self.trade_count = 0
            self.current_position = 0
            self.position_start_time = 0
            self.entry_price = 0
            self.action_counts = {"buy": 0, "sell": 0, "hold": 0}
            
            # New tracking metrics for rapid trades
            self.trade_returns = []
            self.trade_durations = []
            self.recent_trades = []  # List to track recent trades for trend analysis
            self.max_recent_trades = 100  # Maximum number of recent trades to keep
            self.rapid_trade_threshold = 5  # Consider trades within 5 steps as rapid
            self.rapid_trades_detected = 0
            self.last_trade_step = 0
            self.same_price_trades = 0  # Count trades at exactly the same price (problematic)
            self.last_trade_price = None
            
            # Position tracking
            self.current_position = 0
            
            # For debugging
            self.debug_steps = 0
            self.last_debug_output = 0
            self.debug_frequency = 500  # Output debug info every 500 steps
            self.log_flush_frequency = 5000  # Flush logs more frequently
            
            logger.info("Enhanced TensorboardCallback initialized - Will track trading metrics with rapid trade detection")
        
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
                    if self.rapid_trades_detected > 0:
                        print(f"WARNING: {self.rapid_trades_detected} rapid trades detected! {self.same_price_trades} at same price!")
                
                if hasattr(self, 'locals') and 'infos' in self.locals and len(self.locals['infos']) > 0:
                    # Log a sample of the info dictionary to understand what's available
                    logger.info(f"Sample info dict: {str(self.locals['infos'][0])}")
                self.last_debug_output = self.debug_steps
            
            # Force tensorboard log flushing periodically
            if hasattr(self.logger, "dump_tabular") and self.debug_steps % self.log_flush_frequency == 0:
                self.logger.dump_tabular()
            
            # Track step reward
            # Collect information from each environment
            if hasattr(self, 'locals') and 'dones' in self.locals:
                for i, done in enumerate(self.locals['dones']):
                    
                    # Log environment info if available
                    if 'infos' in self.locals and i < len(self.locals['infos']):
                        info = self.locals['infos'][i]
                        
                        # Initialize variables that might be used later to avoid UnboundLocalError
                        exit_price = 0
                        pnl = 0
                        trade_duration = 0
                        
                        # Track portfolio value
                        if 'portfolio_value' in info:
                            self.portfolio_values.append(info['portfolio_value'])
                            self.logger.record('portfolio/value', info['portfolio_value'])
                        
                        # Track position
                        if 'position' in info:
                            position = info['position']
                            
                            # Position change detection for trade tracking
                            if position != self.current_position:
                                # If we had a previous position, this is a trade exit
                                if self.current_position != 0:
                                    trade_duration = self.debug_steps - self.position_start_time
                                    self.trade_durations.append(trade_duration)
                                    
                                    # Check for rapid trading
                                    steps_since_last_trade = self.debug_steps - self.last_trade_step
                                    if steps_since_last_trade < self.rapid_trade_threshold:
                                        self.rapid_trades_detected += 1
                                        if self.rapid_trades_detected % 10 == 0:  # Log every 10th rapid trade
                                            logger.warning(f"Rapid trade detected! Only {steps_since_last_trade} steps since last trade.")
                                    
                                    # Initialize variables that might be used later
                                    exit_price = 0
                                    pnl = 0
                                    
                                    # Calculate trade results
                                    if 'close' in info:
                                        exit_price = info['close']
                                        
                                        # Check for same price trades (sign of environment issues)
                                        if self.last_trade_price is not None and abs(exit_price - self.last_trade_price) < 0.0001:
                                            self.same_price_trades += 1
                                            if self.same_price_trades % 10 == 0:  # Log every 10th same-price trade
                                                logger.warning(f"Same price trade detected! Price: {exit_price:.2f}")
                                        
                                        self.last_trade_price = exit_price
                                        pnl = (exit_price - self.entry_price) * self.current_position
                                        self.trade_returns.append(pnl)
                                        
                                        # Track trade result
                                        if pnl > 0:
                                            self.successful_trades += 1
                                            self.total_profit += pnl
                                            if self.debug_steps % self.debug_frequency == 0:
                                                logger.info(f"Successful trade: PnL={pnl:.4f}, entry={self.entry_price:.4f}, exit={exit_price:.4f}, duration={trade_duration}")
                                        else:
                                            self.failed_trades += 1
                                            self.total_loss += abs(pnl)
                                            if self.debug_steps % self.debug_frequency == 0:
                                                logger.info(f"Failed trade: PnL={pnl:.4f}, entry={self.entry_price:.4f}, exit={exit_price:.4f}, duration={trade_duration}")
                                    
                                    # Add to recent trades for trend analysis
                                    self.recent_trades.append({
                                        'entry_price': self.entry_price,
                                        'exit_price': exit_price if 'close' in info else 0,
                                        'position': self.current_position,
                                        'duration': trade_duration,
                                        'pnl': pnl if 'close' in info else 0
                                    })
                                    
                                    # Keep only the most recent trades
                                    if len(self.recent_trades) > self.max_recent_trades:
                                        self.recent_trades.pop(0)
                            
                                # If new position is non-zero, this is a trade entry
                                if position != 0:
                                    self.trade_count += 1
                                    self.last_trade_step = self.debug_steps
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
                            
                            # Log action distribution periodically
                            if self.debug_steps % (self.debug_frequency * 10) == 0:
                                total_actions = sum(self.action_counts.values())
                                if total_actions > 0:
                                    buy_pct = self.action_counts.get('buy', 0) / total_actions * 100
                                    sell_pct = self.action_counts.get('sell', 0) / total_actions * 100
                                    hold_pct = self.action_counts.get('hold', 0) / total_actions * 100
                                    logger.info(f"Action distribution: Buy: {buy_pct:.1f}%, Sell: {sell_pct:.1f}%, Hold: {hold_pct:.1f}%")
                        
                        # Track trade metrics
                        if 'trade_count' in info:
                            self.logger.record('trades/count', info['trade_count'])
                        if 'profit_loss' in info:
                            self.logger.record('trades/profit_loss', info['profit_loss'])
                        
                        # Track individual reward components if available
                        if 'reward_components' in info:
                            components = info['reward_components']
                            for comp_name, comp_value in components.items():
                                self.logger.record(f'reward_components/{comp_name}', comp_value)
                    
                    # Check if episode is done
                    dones = self.locals.get('dones', [False])
                    if done and i < len(dones) and dones[i]:
                        if 'episode' in self.locals and i < len(self.locals['episode'].rewards):
                            ep_rew = self.locals['episode'].rewards[i]
                            self.episode_rewards.append(ep_rew)
                            self.logger.record('episode/reward', ep_rew)
                            
                            # Reset rapid trade counter at episode end
                            if self.rapid_trades_detected > 0:
                                logger.warning(f"Episode ended with {self.rapid_trades_detected} rapid trades detected")
                                self.logger.record('trades/rapid_trades', self.rapid_trades_detected)
                                self.logger.record('trades/same_price_trades', self.same_price_trades)
                                
            return True  # Continue training
    
    # Create the TensorboardCallback
    tensorboard_callback = TensorboardCallback()
    
    # Try to create and train the model
    try:
        logger.info(f"Creating {finrl_model} model with parameters: {model_params}")
        
        # Additional parameters to pass to the model
        if device is not None:
            model_params['device'] = device
        
        # Instantiate the model
        model = model_class(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            **model_params
        )
        
        # Log the model parameters
        logger.info(f"Model created with parameters: {model.get_parameters()}")
        
        # Set the random seed if provided
        if hasattr(args, 'seed') and args.seed is not None:
            logger.info(f"Setting random seed: {args.seed}")
            model.set_random_seed(args.seed)
        
        # Train the model
        logger.info(f"Starting training for {timesteps} timesteps")
        model.learn(
            total_timesteps=timesteps,
            callback=tensorboard_callback,
            tb_log_name=f"{finrl_model}_logs",
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # Save the model
        if hasattr(args, 'model_save_path') and args.model_save_path is not None:
            save_path = args.model_save_path
        else:
            save_path = f"./models/{finrl_model}_{int(time.time())}"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        model.save(save_path)
        
        return model
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        traceback.print_exc()
        return None

def setup_logging():
    """Configure logging."""
    import logging
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler('./logs/training.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a DQN agent for cryptocurrency trading')
    
    # Environment parameters
    parser.add_argument('--use_finrl', action='store_true', help='Use FinRL framework for environments and agents')
    parser.add_argument('--finrl_model', type=str, default='ppo', help='FinRL model to use (ppo, a2c, ddpg, td3, sac)')
    parser.add_argument('--start_date', type=str, default='2018-01-01', help='Start date for data')
    parser.add_argument('--end_date', type=str, default='2021-12-31', help='End date for data')
    parser.add_argument('--tickers', type=str, default='BTC', help='Comma-separated list of tickers')
    parser.add_argument('--lstm_model_path', type=str, help='Path to pre-trained LSTM model for market predictions')
    parser.add_argument('--data_path', type=str, default='data/synthetic/synthetic_dataset.h5', 
                      help='Path to preprocessed market data file (HDF5 format preferred)')
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu, cuda:0, etc.)')
    parser.add_argument('--initial_balance', type=float, default=1000000.0, help='Initial balance for trading environment')
    parser.add_argument('--timesteps', type=int, default=50000, help='Number of timesteps to train')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--reward_scaling', type=float, default=1e-4, help='Reward scaling factor')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--num_envs_per_worker', type=int, default=1, help='Number of environments per worker')
    parser.add_argument('--normalize_observations', type=str, default='true', help='Whether to normalize observations')
    
    # Debugging/verbosity
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def load_and_preprocess_market_data(args):
    """
    Load market data from the specified source and preprocess it.
    
    Args:
        args: Command line arguments
        
    Returns:
        Preprocessed market data
    """
    # Use default path if not provided
    data_path = args.data_path if args.data_path else 'data/synthetic/synthetic_dataset.h5'
    logger.info(f"Loading market data from {data_path}")
    
    try:
        # Try to load HDF5 format first (preferred)
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            logger.info(f"Loading HDF5 market data from {data_path}")
            # Read from the HDF5 file
            with pd.HDFStore(data_path, mode='r') as store:
                # Get keys - they typically start with '/'
                keys = store.keys()
                if not keys:
                    raise ValueError(f"No datasets found in HDF5 file {data_path}")
                
                # Try to load the '15m' timeframe first as it has the most data
                if '/15m' in keys:
                    df = store['/15m']
                    logger.info(f"Loaded 15m timeframe data with shape {df.shape}")
                else:
                    # Otherwise, load the first key
                    first_key = keys[0]
                    df = store[first_key]
                    logger.info(f"Loaded {first_key} timeframe data with shape {df.shape}")
                    
                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Make sure we have a 'tic' column
                if 'tic' not in df.columns:
                    df['tic'] = 'BTC'  # Default to BTC
                    logger.info("Added 'tic' column with default value 'BTC'")
                
                # Make sure we have a 'day' column
                if 'day' not in df.columns and 'date' in df.columns:
                    df['day'] = df['date'].factorize()[0]
                    logger.info(f"Added 'day' column with factorized dates")
                
                return df
        
        # Fallback to CSV if not HDF5
        elif data_path.endswith('.csv'):
            logger.warning("CSV format is not preferred. Consider using HDF5 format for better performance.")
            df = pd.read_csv(data_path)
            logger.info(f"Loaded CSV market data with shape {df.shape}")
            
            # Process CSV data for compatibility
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'tic' not in df.columns:
                df['tic'] = 'BTC'  # Default to BTC
            if 'day' not in df.columns and 'date' in df.columns:
                df['day'] = df['date'].factorize()[0]
                
            return df
        else:
            raise ValueError(f"Unsupported file format: {data_path}. Use .h5, .hdf5, or .csv file.")
        
    except Exception as e:
        logger.error(f"Failed to load market data: {e}")
        logger.error(traceback.format_exc())
        return None

def train_with_custom_dqn(args, market_data, data_length, device):
    """
    Train a custom DQN agent with the provided market data.
    
    Args:
        args: Command line arguments
        market_data: DataFrame with market data
        data_length: Length of market data
        device: Device to use for training
    
    Returns:
        Trained model or None
    """
    # This would be a full implementation
    # For now, just return a placeholder
    logger.info("Custom DQN training not implemented yet, returning placeholder")
    return None

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
    logger.info("Checking for required technical indicators")
    
    # Check if indicators already exist in the dataframe
    existing_indicators = [ind for ind in indicators if ind in df.columns]
    missing_indicators = [ind for ind in indicators if ind not in df.columns]
    
    if not missing_indicators:
        logger.info("All required technical indicators already present in the dataframe")
        return df
    
    logger.info(f"Calculating missing technical indicators: {missing_indicators}")
    
    # Create a copy of the dataframe
    df_result = df.copy()
    
    # Group by ticker
    for ticker, group in df_result.groupby('tic'):
        # Calculate indicators for each ticker separately
        temp_df = group.copy().sort_values('date')
        
        # Calculate basic price indicators
        if 'close_5_sma' in missing_indicators:
            temp_df['close_5_sma'] = temp_df['close'].rolling(window=5).mean()
        
        if 'close_10_sma' in missing_indicators:
            temp_df['close_10_sma'] = temp_df['close'].rolling(window=10).mean()
            
        if 'close_20_sma' in missing_indicators:
            temp_df['close_20_sma'] = temp_df['close'].rolling(window=20).mean()
        
        # Calculate MACD
        if 'macd' in missing_indicators:
            ema12 = temp_df['close'].ewm(span=12, adjust=False).mean()
            ema26 = temp_df['close'].ewm(span=26, adjust=False).mean()
            temp_df['macd'] = ema12 - ema26
        
        # Calculate RSI (14 periods)
        if 'rsi_14' in missing_indicators:
            delta = temp_df['close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            temp_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate CCI (30 periods)
        if 'cci_30' in missing_indicators:
            tp = (temp_df['high'] + temp_df['low'] + temp_df['close']) / 3
            ma_tp = tp.rolling(window=30).mean()
            md_tp = tp.rolling(window=30).apply(lambda x: np.fabs(x - x.mean()).mean())
            temp_df['cci_30'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # Calculate DX (30 periods)
        if 'dx_30' in missing_indicators:
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
        if 'volatility_30' in missing_indicators:
            temp_df['volatility_30'] = temp_df['close'].pct_change().rolling(window=30).std()
        
        # Calculate volume change
        if 'volume_change' in missing_indicators:
            temp_df['volume_change'] = temp_df['volume'].pct_change()
        
        # Update the main dataframe
        df_result.loc[temp_df.index, temp_df.columns] = temp_df
    
    # Forward fill any NaN values for the newly calculated indicators
    for indicator in missing_indicators:
        if indicator in df_result.columns and df_result[indicator].isnull().any():
            df_result[indicator] = df_result[indicator].ffill().fillna(0)
    
    logger.info(f"Added {len(missing_indicators)} missing technical indicators")
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
    # Set up logging
    logger = setup_logging()
    logger.info("Starting cryptocurrency trading DQN agent training")
    
    # Parse arguments
    args = parse_args()
    logger.info(f"Command line arguments: {args}")
    
    # Set device for training
    device = 'cpu'
    if args.device and args.device.startswith('cuda'):
        import torch
        if torch.cuda.is_available():
            device = args.device
            logger.info(f"Using device: {device}")
        else:
            logger.warning("CUDA requested but not available. Using CPU instead.")
    
    # Load and preprocess market data
    market_data = load_and_preprocess_market_data(args)
    
    if market_data is None:
        logger.error("Failed to load market data, exiting")
        return
    
    logger.info(f"Loaded market data with shape: {market_data.shape}")
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(',')]
    logger.info(f"Using tickers: {tickers}")
    
    # Parse dates
    start_date = args.start_date
    end_date = args.end_date
    logger.info(f"Training period: {start_date} to {end_date}")
    
    # Create the training environment
    try:
        if args.use_finrl:
            # Train with FinRL
            train_with_finrl(
                df=market_data,
                args=args,
                model_name=args.finrl_model,
                timesteps=args.timesteps,
                tensorboard_log="./tensorboard_log",
                device=device
            )
        else:
            # Train with our custom DQN implementation
            data_length = len(market_data)
            logger.info(f"Total data length: {data_length}")
            train_with_custom_dqn(args, market_data, data_length, device)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        raise
        
    logger.info("Training completed successfully")
    
    # Add metadata logging
    import psutil
    memory_info = psutil.virtual_memory()
    logger.info(f"System memory usage: {memory_info.percent}%")
    
    # Try to get GPU info if available
    try:
        import torch
        import subprocess
        if torch.cuda.is_available():
            # Log GPU memory info
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            logger.info(f"GPU memory allocated: {gpu_memory:.2f} GB")
            
            # Create a callback that monitors GPU memory
            class GPUMonitorCallback(BaseCallback):
                def __init__(self, verbose=0):
                    super(GPUMonitorCallback, self).__init__(verbose)
                
                def _on_step(self):
                    if self.n_calls % 1000 == 0:  # Log every 1000 steps
                        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
                        logger.info(f"Step {self.n_calls}: GPU memory: {gpu_memory:.2f} GB")
                    return True
            
            # Get GPU info from nvidia-smi
            try:
                gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
                logger.info(f"GPU Info:\n{gpu_info}")
            except:
                logger.info("Could not get GPU info from nvidia-smi")
    except Exception as e:
        logger.info(f"Could not log GPU info: {e}")
        
    # Create metrics dictionary
    metrics = {
        "data_length": data_length,
        "training_duration": 0,  # Will be filled by the training function
        "model_size": 0,  # Will be filled by the training function
        "memory_usage": memory_info.percent,
    }
    
    # Save metrics to JSON
    try:
        import json
        metrics_file = 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        
    logger.info("Exiting successfully")
    return 0

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