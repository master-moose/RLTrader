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
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt  # Add matplotlib
import psutil  # Add psutil
import inspect
from stable_baselines3 import SAC, TD3, DDPG, PPO

# Add project root to path
project_root = str(Path(__file__).parent)
sys.path.append(project_root)

from crypto_trading_model.environment.crypto_env import CryptocurrencyTradingEnv
from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds

# Import FinRL components
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
# Update the import path for DRLAgent for FinRL 0.3.7
try:
    # Try the new import path first (FinRL 0.3.7+)
    from finrl.applications.agents.drl_agents import DRLAgent
except ImportError:
    try:
        # Try alternative path
        from finrl.agents.stablebaselines3.models import DRLAgent
    except ImportError:
        # Fall back to older path (FinRL <= 0.3.5)
        from finrl.agents.drl_agent import DRLAgent
from finrl.config import INDICATORS

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
            df: DataFrame with stock data
            state_space: Dimension of the state space
            stock_dim: Number of stocks
            action_space: Dimension of the action space
            **kwargs: Additional arguments
        """
        self.df = df
        self.state_space = state_space
        self.stock_dim = stock_dim
        self.action_dim = action_space
        self.current_step = 0
        self.max_steps = len(df) if df is not None else 1000
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Set up observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )
        
        # Action space: -1 (sell), 0 (hold), 1 (buy) for each stock
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(stock_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.state = np.zeros(state_space)
        
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.state = np.zeros(self.state_space)
        
        # Generate a simple observation with random values
        obs = np.random.normal(0, 1, size=self.state_space)
        return obs
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.current_step += 1
        
        # Simple reward: random for demonstration
        reward = np.random.normal(0, 0.1)
        
        # Update state: random for demonstration
        self.state = np.random.normal(0, 1, size=self.state_space)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Info dictionary
        info = {'current_step': self.current_step}
        
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        pass
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]

# Use FinRL's StockTradingEnv if available, otherwise use placeholder
StockTradingEnv = StockTradingEnv if StockTradingEnv else BaseStockTradingEnv

# Create alias for CryptocurrencyTradingEnv
CryptocurrencyTradingEnv = StockTradingEnv
logger.info("Using StockTradingEnv as CryptocurrencyTradingEnv")

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
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
            state_space: The expected dimension of the observation space
        """
        # Convert to Gymnasium environment first if it's not already
        if not isinstance(env, gymnasium.Env):
            # Create a proper gymnasium-compatible observation space
            self.observation_space = gymnasium.spaces.Box(
                low=-10.0, high=10.0, shape=(state_space,), dtype=np.float32
            )
            self.action_space = env.action_space
            if isinstance(self.action_space, gym.spaces.Box):
                # Convert gym Box to gymnasium Box
                self.action_space = gymnasium.spaces.Box(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    shape=env.action_space.shape,
                    dtype=env.action_space.dtype
                )
            # We'll initialize the wrapper later
            self.env = env
            # Add num_envs attribute required by SB3
            self.num_envs = 1
            
            logger.info(f"Created StockTradingEnvWrapper with observation space: {self.observation_space}")
        else:
            # If already a gymnasium environment, just wrap it directly
            super().__init__(env)
            # Override the observation space to have finite bounds
            self.observation_space = gymnasium.spaces.Box(
                low=-10.0, high=10.0, shape=(state_space,), dtype=np.float32
            )
            logger.info(f"Wrapped Gymnasium environment with observation space: {self.observation_space}")
        
        # Store the actual observation dimension for comparison
        self.actual_state_space = None
    
    def reset(self, **kwargs):
        """Reset the environment, with compatibility for both gym and gymnasium APIs."""
        try:
            # Try gymnasium API first (for newer environments)
            obs, info = self.env.reset(**kwargs)
            reset_info = info
        except (ValueError, TypeError):
            # Fall back to old gym API
            obs = self.env.reset()
            reset_info = {}
        
        # Log the observation content for debugging
        if isinstance(obs, np.ndarray):
            if self.actual_state_space is None:
                self.actual_state_space = len(obs)
                logger.info(f"First reset - detected actual observation dimension: {self.actual_state_space}")
                
                # Update observation space if needed to match the actual dimension
                if self.actual_state_space != self.observation_space.shape[0]:
                    logger.info(f"Updating observation space from shape {self.observation_space.shape} to ({self.actual_state_space},)")
                    self.observation_space = gymnasium.spaces.Box(
                        low=-10.0, high=10.0, shape=(self.actual_state_space,), dtype=np.float32
                    )
            
            logger.info(f"Reset observation shape: {obs.shape}, observation space shape: {self.observation_space.shape}")
            
            # No need to reshape the observation, just clip it to the bounds
            obs = np.clip(obs, -10.0, 10.0)
        
        return obs, reset_info
    
    def step(self, action):
        """Take a step, with compatibility for both gym and gymnasium APIs."""
        try:
            # Try with gymnasium API (5 values)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        except ValueError:
            # Fall back to old gym API (4 values)
            obs, reward, done, info = self.env.step(action)
            terminated, truncated = done, False
        
        # Log step observation for debugging (only on some steps to avoid excessive logging)
        if isinstance(obs, np.ndarray) and hasattr(self, 'step_counter'):
            self.step_counter = getattr(self, 'step_counter', 0) + 1
            if self.step_counter % 100 == 0:  # Log only every 100 steps
                logger.info(f"Step {self.step_counter} observation shape: {obs.shape}, observation space shape: {self.observation_space.shape}")
        
        # Just clip the observation to the bounds without reshaping
        if isinstance(obs, np.ndarray):
            obs = np.clip(obs, -10.0, 10.0)
        
        # Return in gymnasium format
        return obs, reward, terminated, truncated, info
    
    def seed(self, seed=None):
        """Seed the environment if the underlying env supports it."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        # For gymnasium API, implement a basic seed method
        np.random.seed(seed)
        return [seed]

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
    Custom implementation of DummyVecEnv that doesn't use the patching mechanism
    which causes compatibility issues with gym.spaces.Sequence.
    
    This is a simplified version that only implements the essential functionality
    needed for our use case.
    """
    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: A list of functions that create environments to run in parallel
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        
        # Get observation and action spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Initialize buffers for observations
        obs_shape = self.observation_space.shape
        self.buf_obs = np.zeros((self.num_envs,) + obs_shape, dtype=np.float32)
        
        # Track done states
        self.dones = np.array([False] * self.num_envs)
        
        logger.info(f"Created CustomDummyVecEnv with {self.num_envs} environments")
        logger.info(f"Observation space: {self.observation_space}, Action space: {self.action_space}")
    
    def reset(self):
        """
        Reset all environments and return initial observations
        """
        for i in range(self.num_envs):
            obs = self.envs[i].reset()
            self._save_obs(i, obs)
        
        # Reset done states
        self.dones = np.array([False] * self.num_envs)
        
        return self.buf_obs.copy()
    
    def step(self, actions):
        """
        Step all environments with the given actions
        
        Arguments:
        actions: Actions to take in each environment
        
        Returns:
        observations, rewards, dones, infos
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.array([False] * self.num_envs)
        infos = [{} for _ in range(self.num_envs)]
        
        for i in range(self.num_envs):
            if not self.dones[i]:
                # Convert scalar actions to the correct type
                if isinstance(actions, np.ndarray) and actions.ndim > 1:
                    action = actions[i]
                else:
                    action = actions[i] if isinstance(actions, list) else actions
                
                obs, rewards[i], dones[i], infos[i] = self.envs[i].step(action)
                if dones[i]:
                    # For environments that automatically reset on done
                    if infos[i].get("terminal_observation") is None and hasattr(self.envs[i], "reset"):
                        terminal_obs = obs
                        if isinstance(obs, np.ndarray):
                            terminal_obs = obs.copy()
                        infos[i]["terminal_observation"] = terminal_obs
                        obs = self.envs[i].reset()
                
                self._save_obs(i, obs)
        
        # Update done states
        self.dones = dones.copy()
        
        return self.buf_obs.copy(), rewards, dones, infos
    
    def _save_obs(self, idx, obs):
        """
        Save observation for the idx-th environment
        
        Handle different types of observations:
        - numpy arrays: direct shape verification
        - tuples: extract the first element if it's array-like
        - other types: try to convert to numpy array
        """
        expected_shape = self.observation_space.shape
        
        # Handle tuple observations (common in recent gym versions)
        if isinstance(obs, tuple):
            logger.info(f"Received tuple observation with length {len(obs)}")
            
            # In gym, tuple observations often have the actual observation as first element
            if len(obs) > 0:
                # Try to get the first element if it's an array-like observation
                first_elem = obs[0]
                if isinstance(first_elem, np.ndarray):
                    logger.info(f"Using first element of tuple as observation with shape {first_elem.shape}")
                    obs = first_elem
                elif isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                    # Try to convert lists to numpy arrays
                    try:
                        logger.info(f"Converting first element of tuple (list/tuple) to numpy array")
                        obs = np.array(first_elem, dtype=np.float32)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert first element of tuple observation to array")
                        # Default to zeros as a fallback
                        self.buf_obs[idx] = np.zeros(expected_shape, dtype=np.float32)
                        return
            else:
                logger.warning("Received empty tuple as observation, using zeros")
                self.buf_obs[idx] = np.zeros(expected_shape, dtype=np.float32)
                return
        
        if isinstance(obs, np.ndarray):
            # Ensure observation matches the expected shape
            if obs.shape != expected_shape:
                logger.warning(f"Observation shape mismatch: got {obs.shape}, expected {expected_shape}")
                
                # Handle dimension mismatch - truncate if too large
                if len(obs) > expected_shape[0]:
                    self.buf_obs[idx] = obs[:expected_shape[0]]
                # Pad with zeros if too small
                elif len(obs) < expected_shape[0]:
                    self.buf_obs[idx] = np.zeros(expected_shape, dtype=np.float32)
                    self.buf_obs[idx][:len(obs)] = obs
                else:
                    self.buf_obs[idx] = obs
            else:
                self.buf_obs[idx] = obs
        elif isinstance(obs, (list, float, int)):
            # Try to convert to numpy array if it's a list, float, or int
            try:
                arr_obs = np.array(obs, dtype=np.float32)
                if arr_obs.shape != expected_shape:
                    if len(arr_obs) > expected_shape[0]:
                        arr_obs = arr_obs[:expected_shape[0]]
                    elif len(arr_obs) < expected_shape[0]:
                        temp = np.zeros(expected_shape, dtype=np.float32)
                        temp[:len(arr_obs)] = arr_obs
                        arr_obs = temp
                self.buf_obs[idx] = arr_obs
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting observation to numpy array: {e}")
                self.buf_obs[idx] = np.zeros(expected_shape, dtype=np.float32)
        else:
            logger.warning(f"Unexpected observation type: {type(obs)}")
            # Use zeros as fallback
            self.buf_obs[idx] = np.zeros(expected_shape, dtype=np.float32)
    
    def close(self):
        """
        Close all environments
        """
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()

# Now modify the create_parallel_finrl_envs function
def create_parallel_finrl_envs(df, args, num_workers=4):
    """
    Create multiple FinRL environments for parallel training.
    
    Args:
        df: DataFrame with FinRL formatted data
        args: Command-line arguments
        num_workers: Number of parallel environments to create
        
    Returns:
        A vectorized environment with multiple environment instances
    """
    # Try to import the most appropriate environment class
    try:
        # Try different import paths for the environment
        StockTradingEnvClass = None
        env_import_paths = [
            # Try cryptocurrency specific environments first
            'finrl.meta.env_cryptocurrency_trading.env_crypto.CryptocurrencyTradingEnv',
            'finrl.applications.env.crypto.CryptocurrencyTradingEnv',
            # Then fall back to stock trading environments
            'finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv',
            'finrl.applications.env.stock.StockTradingEnv',
        ]
        
        for import_path in env_import_paths:
            try:
                module_path, class_name = import_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                StockTradingEnvClass = getattr(module, class_name)
                logger.info(f"Successfully imported {class_name} from {module_path}")
                break
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not import {import_path}: {e}")
                
        if StockTradingEnvClass is None:
            logger.warning("Could not import any trading environment from FinRL, using base implementation")
            StockTradingEnvClass = BaseStockTradingEnv
    except Exception as e:
        logger.error(f"Error importing environment classes: {e}")
        logger.error(traceback.format_exc())
        # Fallback to our base implementation
        StockTradingEnvClass = BaseStockTradingEnv
    
    # Get unique tickers
    unique_tickers = df['tic'].unique()
    num_stocks = len(unique_tickers)
    stock_dim = num_stocks
    
    # Initialize stock shares to 0
    num_stock_shares = [0] * num_stocks
    
    # Create list of technical indicators
    tech_indicator_list = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                          'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                          'sma_7', 'sma_25', 'ema_9', 'ema_21',
                          'stoch_k', 'stoch_d']
    
    # Calculate state space dimension
    # For StockTradingEnv, the observation has:
    # - Account balance (1)
    # - Asset price (1)
    # - Asset shares (1)
    # - Technical indicators (len(tech_indicator_list))
    state_space = 1 + 1 + 1 + len(tech_indicator_list)
    
    # Set action space - Buy, Hold, Sell
    action_space = 3
    
    # Log environment configuration
    logger.info(f"Creating parallel environment with state_space={state_space}, "
                f"action_space={action_space}, num_stocks={num_stocks}")
    
    # Environment parameters
    initial_amount = getattr(args, 'initial_balance', 1000000)
    transaction_cost_pct = getattr(args, 'transaction_fee', 0.001)
    reward_scaling = getattr(args, 'reward_scaling', 1e-4)
    hmax = getattr(args, 'hmax', 100)  # Maximum number of shares to trade
    
    logger.info(f"Environment config: initial_amount={initial_amount}, "
                f"transaction_cost_pct={transaction_cost_pct}, "
                f"reward_scaling={reward_scaling}, stock_dim={stock_dim}, hmax={hmax}")
    
    # Create a test environment to debug the observation space
    try:
        test_env = StockTradingEnvClass(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            num_stock_shares=num_stock_shares.copy(),
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicator_list,
            initial_amount=initial_amount,
            buy_cost_pct=transaction_cost_pct,
            sell_cost_pct=transaction_cost_pct,
            reward_scaling=reward_scaling,
            print_verbosity=1
        )
        
        # Test the environment to see if observation shape matches state_space
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
        logger.error(f"Error testing environment: {e}")
        logger.error(traceback.format_exc())
        
    # Create a list of environment creation functions
    env_list = []
    
    for i in range(num_workers):
        # Define a function that creates a new environment instance each time it's called
        def make_env(idx=i):
            try:
                # Create basic environment parameters
                env_params = {
                    'df': df,
                    'state_space': state_space,
                    'stock_dim': stock_dim,
                    'hmax': hmax,
                    'num_stock_shares': num_stock_shares.copy(),  # Use a copy to avoid sharing state
                    'action_space': action_space,
                    'tech_indicator_list': tech_indicator_list,
                    'initial_amount': initial_amount,
                    'buy_cost_pct': transaction_cost_pct,
                    'sell_cost_pct': transaction_cost_pct,
                    'reward_scaling': reward_scaling,
                    'print_verbosity': 1 if idx == 0 else 0  # Only print verbose output for the first env
                }
                
                # Create the environment
                base_env = StockTradingEnvClass(**env_params)
                
                # Wrap the environment to ensure consistent observation shape
                return StockTradingEnvWrapper(base_env, state_space=state_space)
            except Exception as e:
                logger.error(f"Error creating environment {idx}: {e}")
                logger.error(traceback.format_exc())
                # Provide a fallback environment if creation fails
                base_env = BaseStockTradingEnv(df, state_space=state_space)
                return StockTradingEnvWrapper(base_env, state_space=state_space)
        
        # Add the environment creation function to the list with proper closure handling
        env_fn = lambda idx=i: make_env(idx)
        env_list.append(env_fn)
    
    # Use our custom DummyVecEnv implementation that avoids compatibility issues
    logger.info("Using CustomDummyVecEnv for environment vectorization (to avoid gym compatibility issues)")
    vec_env = CustomDummyVecEnv(env_list)
    
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
class CustomDRLAgent(OriginalDRLAgent):
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
        # Set default net_arch attribute that might be missing from parent class
        self.net_arch = [256, 256]
        
        # Check if parent class accepts verbose parameter
        try:
            # Try with verbose parameter first
            super().__init__(env=env, verbose=verbose)
            self.verbose = verbose
        except TypeError:
            # Fall back to just env if verbose is not accepted
            super().__init__(env=env)
            self.verbose = verbose
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
    Train a FinRL agent with the specified parameters.
    
    Args:
        args: Command line arguments
        logger: Logger
        start_date: Start date for training
        end_date: End date for training
        tickers: List of tickers to trade
        data_source: Source of data
        num_workers: Number of workers for parallel environments
        use_lstm_predictions: Whether to use LSTM predictions
        lstm_model: LSTM model to use for predictions
        lstm_processor: LSTM processor for data preprocessing
        
    Returns:
        trained FinRL model
    """
    logger.info(f"Training FinRL agent with model: {args.finrl_model}")
    logger.info(f"Start date: {start_date}, End date: {end_date}")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Number of workers: {num_workers}")
    
    # Set up parameters
    lookback = 5
    initial_balance = 1000000.0
    include_cash = False
    
    # Get state dimension from args or use default
    state_dim = getattr(args, 'state_dim', 16)  # Default to 16 if not specified
    
    # Generate synthetic data or format the real data properly
    try:
        logger.info("Creating synthetic data with multi-index format for FinRL...")
        df = create_synthetic_data(tickers, start_date, end_date)
        
        # Format the dataframe for StockTradingEnv which requires a multi-index DataFrame
        # with date and tic as the index levels
        logger.info(f"Formatting DataFrame for FinRL environment, original shape: {df.shape}")
        
        # Make sure we have a date column that's datetime
        if 'date' not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("No date column found, using index as date")
            df['date'] = df.index
        elif 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            logger.info("Converting date column to datetime")
            df['date'] = pd.to_datetime(df['date'])
        
        # Reset index if it's already a DatetimeIndex to work with the data
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
        
        # Add a 'day' column which is used by StockTradingEnv
        if 'day' not in df.columns:
            df['day'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
        # Ensure we have a 'tic' column - if not, use the ticker from tickers list
        # This assumes the dataframe has data for only one ticker
        if 'tic' not in df.columns:
            logger.info(f"Adding 'tic' column with value: {tickers[0]}")
            df['tic'] = tickers[0]
            
        # If we have multiple tickers, we need to make sure each row has the correct ticker
        if len(tickers) > 1 and 'tic' not in df.columns:
            # Need to replicate data for each ticker or ensure each ticker has its own data
            logger.warning(f"Multiple tickers {tickers} but no 'tic' column in DataFrame")
            # This is a placeholder - in a real implementation, 
            # you'd need to properly structure data for multiple tickers
        
        # Create a multi-index with date and tic
        logger.info("Creating multi-index DataFrame with (date, tic)")
        df = df.set_index(['date', 'tic'])
        
        # Log the levels of the multi-index
        logger.info(f"DataFrame multi-index levels: {df.index.names}")
        logger.info(f"Final dataframe shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns}")
        logger.info(f"First few rows of the DataFrame:\n{df.head()}")
        
    except Exception as e:
        logger.error(f"Error creating synthetic data: {e}")
        logger.error(traceback.format_exc())
        raise
    
    # Use DummyVecEnv due to gym.spaces.Sequence compatibility issues
    logger.info(f"Using DummyVecEnv for environment vectorization due to compatibility issues")
    
    # Create the environment functions for each worker
    def make_env():
        """Create a single environment for vectorization"""
        env = create_finrl_env(
            start_date=start_date,
            end_date=end_date,
            symbols=tickers,
            data_source=data_source,
            initial_balance=initial_balance,
            lookback=lookback,
            state_space=state_dim,  # Use state_dim instead of args.state_dim
            include_cash=include_cash,
            df=df  # Pass the formatted DataFrame
        )
        # First try using SB3's Monitor which is still maintained
        try:
            from stable_baselines3.common.monitor import Monitor
            logger.info("Using SB3's Monitor wrapper for logging")
            return Monitor(env, None, allow_early_resets=True)
        except (ImportError, AttributeError):
            try:
                # If SB3 Monitor is not available, try gym's RecordEpisodeStatistics
                from gymnasium.wrappers import RecordEpisodeStatistics
                logger.info("Using gymnasium's RecordEpisodeStatistics wrapper")
                return RecordEpisodeStatistics(env)
            except (ImportError, AttributeError):
                try:
                    # Try gym's RecordEpisodeStatistics as a last resort
                    from gym.wrappers import RecordEpisodeStatistics
                    logger.info("Using gym's RecordEpisodeStatistics wrapper")
                    return RecordEpisodeStatistics(env)
                except (ImportError, AttributeError):
                    # If neither is available, just return the unwrapped environment
                    logger.warning("No monitoring wrapper available, returning unwrapped environment")
                    return env
    
    # Create the vectorized environment using DummyVecEnv
    envs = []
    for i in range(num_workers):
        envs.append(make_env)
    
    vec_env = DummyVecEnv(envs)
    
    # Set up model parameters
    # Default hidden dimension for network architecture
    hidden_dim = getattr(args, 'hidden_dim', 256)
    
    # Set up the model parameters
    if args.finrl_model.lower() == "sac":
        model_class = SAC
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(
                pi=[hidden_dim, hidden_dim], 
                qf=[hidden_dim, hidden_dim]
            )
        )
    elif args.finrl_model.lower() == "td3":
        model_class = TD3
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(
                pi=[hidden_dim, hidden_dim], 
                qf=[hidden_dim, hidden_dim]
            )
        )
    elif args.finrl_model.lower() == "ddpg":
        model_class = DDPG
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(
                pi=[hidden_dim, hidden_dim], 
                qf=[hidden_dim, hidden_dim]
            )
        )
    elif args.finrl_model.lower() == "ppo":
        model_class = PPO
        policy_kwargs = dict(
            activation_fn=nn.Tanh,
            net_arch=[dict(
                pi=[hidden_dim, hidden_dim],
                vf=[hidden_dim, hidden_dim]
            )]
        )
    else:
        logger.error(f"Unsupported model: {args.finrl_model}")
        raise ValueError(f"Unsupported model: {args.finrl_model}")
    
    # Create a callback for logging
    class TensorboardCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)
            self.episode_rewards = []
            self.current_episode_reward = 0
            self.episode_lengths = []
            self.current_episode_length = 0
            self.portfolio_values = []
            self.training_iteration = 0
            
        def _on_step(self):
            # Track reward for current step
            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # Record current step reward
            self.logger.record('train/step_reward', reward)
            
            # Record additional info from environment if available
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'portfolio_value' in info:
                    self.portfolio_values.append(info['portfolio_value'])
                    self.logger.record('portfolio/value', info['portfolio_value'])
                if 'position' in info:
                    self.logger.record('portfolio/position', info['position'])
                if 'trade_count' in info:
                    self.logger.record('trades/count', info['trade_count'])
                if 'profit_loss' in info:
                    self.logger.record('trades/profit_loss', info['profit_loss'])
            
            # Check if episode is done
            dones = self.locals.get('dones', [False])
            if dones[0]:
                # Record episode statistics
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
                
                # Reset episode tracking
                self.current_episode_reward = 0
                self.current_episode_length = 0
                self.portfolio_values = []
            
            # Track learning progress
            self.training_iteration += 1
            return True
    
    # Create the model
    logger.info(f"Creating {args.finrl_model.upper()} model with policy_kwargs: {policy_kwargs}")
    model = model_class(
        "MlpPolicy", 
        vec_env, 
        verbose=1 if args.verbose else 0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"./tensorboard_logs/{args.finrl_model}/"
    )
    
    # Train the model with callback
    total_timesteps = getattr(args, 'timesteps', 50000)  # Default to 50000 if not specified
    logger.info(f"Training model for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardCallback()
    )
    
    # Save the model
    save_model = getattr(args, 'save_model', True)  # Default to True if not specified
    if save_model:
        model_path = f"./models/finrl_{args.finrl_model}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")
    
    return model

if __name__ == "__main__":
    main() 