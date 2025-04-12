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
    """Placeholder StockTradingEnv in case FinRL imports fail"""
    def __init__(self, df=None, **kwargs):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # buy, hold, sell
        logger.warning("Using placeholder StockTradingEnv - NOT FUNCTIONAL!")
    
    def reset(self):
        return np.zeros(100)
    
    def step(self, action):
        return np.zeros(100), 0, True, {}

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
    lookback=5, state_space=16, include_cash=False, initial_stocks=None, window_size=None
):
    """
    Create a FinRL-compatible environment for cryptocurrency trading.
    
    Args:
        start_date: The start date for the environment
        end_date: The end date for the environment
        symbols: The list of cryptocurrency symbols to trade
        data_source: The data source to use (default: "binance")
        initial_balance: The initial balance for the environment
        lookback: The number of time steps to look back for features
        state_space: Initial expected dimension of state space (will be updated based on actual observation)
        include_cash: Whether to include cash in the state
        initial_stocks: The initial stocks for the environment
        window_size: The window size for the environment
    
    Returns:
        A StockTradingEnvWrapper instance that is compatible with stable-baselines3
    """
    # Try different import paths for StockTradingEnv, falling back as needed
    logger.info("Attempting to import StockTradingEnv for cryptocurrency trading")
    
    StockTradingEnvClass = None
    
    # Try multiple import paths in order of preference
    try:
        # Try the cryptocurrency-specific environment first
        from finrl.meta.env_cryptocurrency_trading.env_crypto import CryptocurrencyTradingEnv
        logger.info("Successfully imported CryptocurrencyTradingEnv")
        StockTradingEnvClass = CryptocurrencyTradingEnv
    except ImportError:
        logger.warning("Could not import CryptocurrencyTradingEnv, trying alternative paths")
        try:
            # Try the standard stock trading environment as fallback
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
            logger.info("Using standard StockTradingEnv as fallback")
            StockTradingEnvClass = StockTradingEnv
        except ImportError:
            try:
                # Try newer import path (FinRL has changed paths multiple times)
                from finrl.applications.env.crypto import CryptocurrencyTradingEnv
                logger.info("Successfully imported CryptocurrencyTradingEnv from applications.env")
                StockTradingEnvClass = CryptocurrencyTradingEnv
            except ImportError:
                try:
                    # Try newer import path for stock trading environment
                    from finrl.applications.env.stock import StockTradingEnv
                    logger.info("Using StockTradingEnv from applications.env as fallback")
                    StockTradingEnvClass = StockTradingEnv
                except ImportError:
                    # Final fallback - use our base class
                    logger.warning("Could not import any trading environment from FinRL, using base implementation")
                    StockTradingEnvClass = BaseStockTradingEnv
    
    if StockTradingEnvClass is None:
        raise ImportError("Failed to import any suitable trading environment")
    
    logger.info(f"Creating FinRL environment with state_space={state_space}, lookback={lookback}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Symbols: {symbols}")
    
    # Create a dictionary of all parameters for the environment
    env_params = {
        'df': None,  # Will download data if None
        'state_space': state_space,
        'initial_amount': initial_balance,
        'buy_cost_pct': 0.001,
        'sell_cost_pct': 0.001,
        'reward_scaling': 1e-4,
        'hmax': 100,  # Maximum number of shares to trade
    }
    
    # Add parameters that are specific to cryptocurrency environments
    if 'symbols' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['symbols'] = symbols
    if 'data_source' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['data_source'] = data_source
    if 'start_date' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['start_date'] = start_date
    if 'end_date' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['end_date'] = end_date
    if 'lookback' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['lookback'] = lookback
    if 'include_cash' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['include_cash'] = include_cash
    if 'initial_stocks' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['initial_stocks'] = initial_stocks
    if 'stock_dim' in inspect.signature(StockTradingEnvClass.__init__).parameters:
        env_params['stock_dim'] = len(symbols)
    
    # Create environment with appropriate parameters
    try:
        # Log what we're doing
        logger.info(f"Creating environment with params: {env_params}")
        
        # Create the environment
        env = StockTradingEnvClass(**env_params)
        
        # Ensure we get the correct state dimension by doing a test reset
        # We'll then create a proper wrapper with the correct dimension
        try:
            test_obs = env.reset()
            actual_dim = len(test_obs) if isinstance(test_obs, np.ndarray) else state_space
            
            logger.info(f"Detected actual observation dimension: {actual_dim}")
            if actual_dim != state_space:
                logger.info(f"Updating state_space from {state_space} to {actual_dim}")
                state_space = actual_dim
            
            # We need to reset the env again
            env.reset()
        except Exception as e:
            logger.error(f"Error during test reset: {e}")
            logger.error(traceback.format_exc())
        
        # Then wrap it for gymnasium compatibility
        wrapped_env = StockTradingEnvWrapper(env, state_space=state_space)
        
        logger.info(f"Environment created with observation space: {wrapped_env.observation_space}")
        logger.info(f"Action space: {wrapped_env.action_space}")
        
        return wrapped_env
    
    except Exception as e:
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


def train_with_finrl(args):
    """Train a model using FinRL's DRL library"""
    logger.info("Training with FinRL")
    
    # Use consistent paths
    model_dir = os.path.join("models", "finrl", args.finrl_model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up start and end dates from args or defaults
    start_date = getattr(args, 'start_date', '2018-01-01')
    end_date = getattr(args, 'end_date', '2021-12-31')
    
    # Get cryptocurrency symbols from args or default
    symbols = getattr(args, 'tickers', ["BTC", "ETH", "LTC"]) 
    if not isinstance(symbols, list):
        # Split comma-separated string into a list
        symbols = [s.strip() for s in symbols.split(',')]
    
    # Update user with settings
    logger.info(f"Training {args.finrl_model} model with DRL on symbols: {symbols}")
    logger.info(f"Data range: {start_date} to {end_date}")
    
    # Create the environment without data loading first (data will be loaded in the env)
    logger.info("Creating single environment for initial testing...")
    
    # Define common environment parameters
    lookback = getattr(args, 'lookback', 5)
    initial_balance = getattr(args, 'initial_balance', 1000000.0)
    include_cash = getattr(args, 'include_cash', False)
    
    # Create a test environment to validate configuration
    test_env = create_finrl_env(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        data_source="binance",
        initial_balance=initial_balance,
        lookback=lookback,
        include_cash=include_cash
    )
    
    # Create multiple environments for parallel training if num_workers > 1
    num_workers = getattr(args, 'num_workers', 1)
    logger.info(f"Creating {num_workers} parallel environments with DummyVecEnv due to compatibility issues")
    
    # Create a list of environment creation functions
    env_fns = []
    for i in range(num_workers):
        def make_env(idx=i):
            # Each environment gets the same configuration but will sample differently
            env = create_finrl_env(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                data_source="binance",
                initial_balance=initial_balance,
                lookback=lookback,
                include_cash=include_cash
            )
            # Add unique identification to the environment
            env = Monitor(env, os.path.join(model_dir, f'monitor_{idx}'))
            return env
        env_fns.append(make_env)
    
    # Always use DummyVecEnv for compatibility
    env = DummyVecEnv(env_fns)
    logger.info(f"Using DummyVecEnv for environment vectorization due to gym compatibility issues")
    
    # Create the DRL agent
    model_name = args.finrl_model.upper()
    
    logger.info(f"Setting up CustomDRLAgent with model: {model_name}")
    # Get observation and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Set up appropriate network architecture based on model type
    if model_name in ["SAC", "TD3", "DDPG"]:
        # For continuous action space models
        # Actor and critic networks
        actor_network_layers = [256, 256]
        critic_network_layers = [256, 256]
        logger.info(f"Using actor network: {actor_network_layers}, critic network: {critic_network_layers}")
    else:
        # For PPO with policy network
        network_layers = [256, 256]  
        logger.info(f"Using network: {network_layers}")
    
    # Get model parameters based on the chosen model
    if model_name == "SAC":
        model_params = {
            "batch_size": 256,
            "buffer_size": 1000000,
            "learning_rate": 0.0003,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
            "verbose": args.verbose,
        }
    elif model_name == "TD3":
        model_params = {
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_rate": 0.0003,
            "learning_starts": 100,
            "verbose": args.verbose,
        }
    elif model_name == "DDPG":
        model_params = {
            "batch_size": 128,
            "buffer_size": 50000,
            "learning_rate": 0.001,
            "verbose": args.verbose,
        }
    elif model_name == "PPO":
        model_params = {
            "batch_size": 128,
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "verbose": args.verbose,
        }
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from SAC, TD3, DDPG, PPO")
    
    logger.info(f"Model parameters: {model_params}")
    
    # Create DRL agent
    agent = CustomDRLAgent(env=env, model_name=model_name)
    
    # Set up the training callback to log progress
    log_dir = os.path.join(model_dir, 'tb_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    callback = TrainingLoggingCallback(
        check_freq=1000,  # Check progress every 1000 steps
        save_path=model_dir,
        verbose=args.verbose,
        save_freq=5000,  # Save every 5000 steps
        log_dir=log_dir
    )
    
    # Train the model
    logger.info("Starting model training...")
    model_path = os.path.join(model_dir, f"{model_name.lower()}_model")
    model = agent.get_model(model_name, model_kwargs=model_params)
    
    # Train for the specified number of timesteps
    timesteps = getattr(args, 'timesteps', 50000)
    logger.info(f"Training for {timesteps} timesteps")
    
    # Train the model
    model = agent.train_model(model=model, tb_log_name=model_name.lower(), total_timesteps=timesteps, callback=callback)
    
    # Save the trained model
    logger.info(f"Saving model to {model_path}")
    model.save(model_path)
    
    logger.info("Training completed successfully!")
    return model

def train_with_custom_dqn(args, market_data, data_length, device):
    """
    Train using custom DQN implementation.
    This is the original training code.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    market_data : dict
        Market data dictionary with different timeframes
    data_length : int
        Length of data
    device : str
        Device to use (cpu or cuda)
    
    Returns:
    --------
    episode_rewards : list
        List of episode rewards
    episode_balances : list
        List of episode balances
    episode_trade_counts : list
        List of episode trade counts
    """
    # Create multiple environments for parallel training
    logger.info(f"Creating {args.num_workers} environments")
    envs = []
    for i in range(args.num_workers):
        env = TradingEnvironment(
            data_path=os.path.join(args.data_dir, "train_data.h5"),
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            reward_scaling=args.reward_scaling,
            use_position_features=args.use_position_features,
            lookback_window=args.lookback_window,
            trade_cooldown=args.trade_cooldown,
            device=device
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
        hidden_dims=[512, 256],  # Increase network capacity
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
    
    # Add trade quality tracking
    good_trades_count = 0
    bad_trades_count = 0
    trade_returns = []  # Track returns of each trade for overfitting detection
    
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
    next_states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    
    # Pre-allocate replay buffer tensors with dynamic sizing capability
    def get_optimal_batch_size():
        """Dynamically determine optimal batch size based on available memory"""
        if device == "cuda":
            # Get available GPU memory (in bytes)
            free_mem = torch.cuda.mem_get_info()[0]
            # Estimate memory per sample (state_dim * 4 bytes per float32)
            mem_per_sample = state_dim * 4 * 4  # states, actions, rewards, next_states
            # Use up to 40% of free memory for batches
            return min(args.batch_size, max(64, int(free_mem * 0.4 / mem_per_sample)))
        else:
            # On CPU, consider system memory
            free_mem = psutil.virtual_memory().available
            mem_per_sample = state_dim * 4 * 4
            return min(args.batch_size, max(64, int(free_mem * 0.4 / mem_per_sample)))
    
    optimal_batch_size = get_optimal_batch_size()
    logger.info(f"Using optimal batch size: {optimal_batch_size}")
    
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
    update_frequency = 16  # Update less frequently to batch more updates
    
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
        
        # Track trade quality for this episode
        episode_good_trades = 0
        episode_bad_trades = 0
        episode_neutral_trades = 0
        last_trade_prices = [None] * args.num_workers
        last_positions = [0] * args.num_workers
        
        # Episode loop
        progress_bar = tqdm(total=args.episode_length, 
                           desc=f"Episode {episode}/{args.episodes}", 
                           disable=False)
        
        step_counter = 0
        while active_envs > 0 and step_counter < args.episode_length:
            # Select actions for all environments at once using vectorized operations
            with torch.no_grad():
                q_values = agent.policy_net(states_tensor)
                # Use inplace operations where possible
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
            
            # Pre-process batch data collection to reduce redundant tensor operations
            transitions_to_add = []
            
            for i in range(args.num_workers):
                if not dones[i]:
                    balances.append(infos[i].get('balance', 0))
                    trade_counts.append(infos[i].get('total_trades', 0))
                    
                    # Track trade quality
                    current_position = infos[i].get('position', 0)
                    current_price = infos[i].get('price', 0)
                    
                    # Check if a trade was made
                    if current_position != last_positions[i]:
                        # A position change occurred (trade was made)
                        if last_trade_prices[i] is not None and last_positions[i] != 0:
                            # Calculate trade return only when closing a position
                            if current_position == 0:
                                # Calculate return based on position direction
                                if last_positions[i] > 0:  # Was long, now closed
                                    trade_return = (current_price / last_trade_prices[i]) - 1 - args.transaction_fee
                                else:  # Was short, now closed
                                    trade_return = (last_trade_prices[i] / current_price) - 1 - args.transaction_fee
                                
                                # Classify the trade
                                if trade_return > GOOD_TRADE_THRESHOLD:
                                    episode_good_trades += 1
                                    good_trades_count += 1
                                    logger.debug(f"Environment {i} - GOOD TRADE: Return={trade_return:.4f}, "
                                                f"Entry={last_trade_prices[i]:.2f}, Exit={current_price:.2f}, "
                                                f"Position={last_positions[i]}")
                                elif trade_return < BAD_TRADE_THRESHOLD:
                                    episode_bad_trades += 1
                                    bad_trades_count += 1
                                    logger.debug(f"Environment {i} - BAD TRADE: Return={trade_return:.4f}, "
                                                f"Entry={last_trade_prices[i]:.2f}, Exit={current_price:.2f}, "
                                                f"Position={last_positions[i]}")
                                else:
                                    episode_neutral_trades += 1
                                    logger.debug(f"Environment {i} - NEUTRAL TRADE: Return={trade_return:.4f}, "
                                                f"Entry={last_trade_prices[i]:.2f}, Exit={current_price:.2f}, "
                                                f"Position={last_positions[i]}")
                                
                                trade_returns.append(trade_return)
                                
                        # Update last trade price when opening a new position
                        if current_position != 0:
                            last_trade_prices[i] = current_price
                    
                    # Update last position
                    last_positions[i] = current_position
                    
                    episode_rewards_per_env[i] += rewards[i]
                    steps_per_env[i] += 1
                    
                    # Collect transitions for batch addition to memory
                    transitions_to_add.append((states[i], actions[i].item(), rewards[i], next_states[i], new_dones[i]))
                    
                    if isinstance(next_states[i], np.ndarray):
                        next_states_tensor[i] = torch.from_numpy(next_states[i]).float()
                    else:
                        next_states_tensor[i] = next_states[i]
                else:
                    balances.append(0)
                    trade_counts.append(0)
            
            # Batch add transitions to memory
            for transition in transitions_to_add:
                agent.memory.append(transition)
                total_experiences += 1
                
            # Log balance changes for debugging
            if step_counter % 100 == 0 and active_envs > 0:
                i = next((i for i, d in enumerate(dones) if not d), 0)
                logger.info(f"Environment {i} - Step {step_counter}: Action={actions[i].item()}, "
                           f"Balance={infos[i].get('balance', 0):.2f}, "
                           f"Position={infos[i].get('position', 0)}, "
                           f"Trades={infos[i].get('total_trades', 0)}, "
                           f"Price={infos[i].get('price', 0):.2f}, "
                           f"Good/Bad/Neutral={episode_good_trades}/{episode_bad_trades}/{episode_neutral_trades}")
            
            # Update states and dones
            states = next_states
            dones = new_dones
            states_tensor.copy_(next_states_tensor)
            
            # Count active environments
            active_envs = sum(1 for d in dones if not d)
            
            # Batch updates every few steps for better efficiency
            if step_counter % update_frequency == 0 and len(agent.memory) >= optimal_batch_size:
                # Perform multiple updates in a batch
                for _ in range(args.updates_per_step):  # Simplified update loop
                    # Dynamically adjust batch size based on available replay buffer samples
                    current_batch_size = min(len(agent.memory), optimal_batch_size)
                    
                    # Sample batch indices - ensure we don't sample more than available items
                    indices = np.random.choice(len(agent.memory), current_batch_size, replace=False)
                    batch = [agent.memory[i] for i in indices]
                    
                    # Convert batch to tensors efficiently - use a single loop
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
                    for i, (state, action, reward, next_state, done) in enumerate(batch):
                        batch_states.append(torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state)
                        batch_actions.append(action)
                        batch_rewards.append(reward)
                        batch_next_states.append(torch.from_numpy(next_state).float() if isinstance(next_state, np.ndarray) else next_state)
                        batch_dones.append(done)
                    
                    # Stack tensors at once instead of individual assignments
                    replay_states = torch.stack(batch_states)
                    replay_actions = torch.tensor(batch_actions, device=device)
                    replay_rewards = torch.tensor(batch_rewards, device=device)
                    replay_next_states = torch.stack(batch_next_states)
                    replay_dones = torch.tensor(batch_dones, device=device)
                    
                    # Update networks using vectorized operations
                    with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
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
                except Exception:
                    pass
            
            # Initialize current_batch_size for the progress bar
            current_batch_size = optimal_batch_size
            
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
        
        # Calculate trade quality metrics
        total_trades = episode_good_trades + episode_bad_trades + episode_neutral_trades
        good_trade_pct = (episode_good_trades / max(1, total_trades)) * 100
        bad_trade_pct = (episode_bad_trades / max(1, total_trades)) * 100
        
        # Detect overfitting by analyzing trade returns distribution
        if len(trade_returns) > 20:  # Need a minimum number of trades for analysis
            recent_returns = trade_returns[-20:]  # Look at recent trades
            avg_return = sum(recent_returns) / len(recent_returns)
            std_return = np.std(recent_returns) if len(recent_returns) > 1 else 0
            
            # Calculate Sharpe-like ratio (return / volatility)
            sharpe_ratio = avg_return / (std_return + 1e-6)  # Add small constant to avoid division by zero
            
            # Check for overfitting signals
            if avg_return < -0.005 and sharpe_ratio < -0.5:
                logger.warning(f"Potential overfitting detected at episode {episode}: "
                             f"Average return: {avg_return:.4f}, StdDev: {std_return:.4f}, "
                             f"Sharpe: {sharpe_ratio:.2f}")
                
                if args.verbose:
                    logger.info(f"Recent trade returns: {recent_returns}")
        
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
                   f"Trades: {episode_trade_count:.2f}, Good/Bad: {episode_good_trades}/{episode_bad_trades} "
                   f"({good_trade_pct:.1f}%/{bad_trade_pct:.1f}%), "
                   f"Epsilon: {agent.epsilon:.4f}, "
                   f"Memory: {len(agent.memory)}, Updates: {total_updates}")
        
        # Evaluate model every few episodes
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            # TODO: Implement evaluation
            pass
    
    # Final trading statistics
    if len(trade_returns) > 0:
        avg_return_per_trade = sum(trade_returns) / len(trade_returns)
        winning_rate = good_trades_count / max(1, good_trades_count + bad_trades_count) * 100
        
        logger.info("Trading Statistics Summary:")
        logger.info(f"Total Trades: {len(trade_returns)}")
        logger.info(f"Good Trades: {good_trades_count} ({winning_rate:.1f}%)")
        logger.info(f"Bad Trades: {bad_trades_count} ({100-winning_rate:.1f}%)")
        logger.info(f"Average Return per Trade: {avg_return_per_trade:.4f}")
        
        if args.verbose:
            # Calculate additional statistics
            max_return = max(trade_returns) if trade_returns else 0
            min_return = min(trade_returns) if trade_returns else 0
            std_return = np.std(trade_returns) if len(trade_returns) > 1 else 0
            
            logger.info(f"Best Trade: {max_return:.4f}")
            logger.info(f"Worst Trade: {min_return:.4f}")
            logger.info(f"Return StdDev: {std_return:.4f}")
            logger.info(f"Sharpe-like Ratio: {(avg_return_per_trade / (std_return + 1e-6)):.2f}")
            
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a DQN agent for cryptocurrency trading.'
    )
    
    # Model and training arguments
    parser.add_argument(
        '--lstm_model_path',
        type=str,
        help='Path to the trained LSTM model'
    )
    parser.add_argument(
        '--use_finrl',
        action='store_true',
        help='Use FinRL framework for training'
    )
    parser.add_argument(
        '--finrl_model',
        type=str,
        choices=['dqn', 'ppo', 'a2c', 'ddpg', 'td3', 'sac'],
        default='dqn',
        help='FinRL model to use for training'
    )
    parser.add_argument(
        '--primary_timeframe',
        type=str,
        choices=['15m', '1d', '4h'],
        default='15m',
        help='Primary timeframe to use for training'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/synthetic',
        help='Directory containing the data files'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models/dqn',
        help='Directory to save the trained model'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel environments for training'
    )
    
    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor for future rewards'
    )
    parser.add_argument(
        '--epsilon_start',
        type=float,
        default=1.0,
        help='Starting value for epsilon-greedy exploration'
    )
    parser.add_argument(
        '--epsilon_end',
        type=float,
        default=0.01,
        help='Final value for epsilon-greedy exploration'
    )
    parser.add_argument(
        '--epsilon_decay',
        type=float,
        default=0.995,
        help='Decay rate for epsilon-greedy exploration'
    )
    parser.add_argument(
        '--target_update',
        type=int,
        default=1000,
        help='Frequency of target network updates'
    )
    parser.add_argument(
        '--memory_size',
        type=int,
        default=100000,
        help='Size of the replay memory'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of episodes to train'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Maximum number of steps per episode'
    )
    parser.add_argument(
        '--net_arch',
        type=str,
        default='[256, 256]',
        help='Network architecture for the model'
    )
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=100000,
        help='Total timesteps for FinRL training'
    )
    parser.add_argument(
        '--transaction_fee',
        type=float,
        default=0.001,
        help='Transaction fee as a percentage'
    )
    parser.add_argument(
        '--reward_scaling',
        type=float,
        default=1e-2,
        help='Scaling factor for rewards in the FinRL environment'
    )
    parser.add_argument(
        '--hmax',
        type=int,
        default=100,
        help='Maximum number of shares to trade in FinRL environment'
    )
    parser.add_argument(
        '--initial_balance',
        type=float,
        default=1000000,
        help='Initial balance for the agent'
    )
    
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
    if args.use_finrl:
        set_random_seed(args.seed)
    else:
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

    # Load data from specified path using h5py
    data_path = args.data_dir
    logger.info(f"Loading data from {data_path}")
    
    # Check which h5 file to use
    h5_file_path = os.path.join(data_path, "synthetic_dataset.h5")
    if not os.path.exists(h5_file_path):
        h5_file_path = os.path.join(data_path, "train_data.h5")
        
    if not os.path.exists(h5_file_path):
        raise ValueError(f"No HDF5 data file found in {data_path}")
    
    # Load data from h5 file
    market_data = {}
    with h5py.File(h5_file_path, 'r') as h5f:
        timeframes = list(h5f.keys())
        logger.info(f"Found timeframes: {timeframes}")
        
        for tf in timeframes:
            # Get the group for this timeframe
            group = h5f[tf]
            
            # Check if the group has a 'table' dataset
            if 'table' not in group:
                logger.error(f"No 'table' dataset found in group {tf}")
                continue
            
            # Get the table dataset
            table = group['table']
            logger.info(f"Found table dataset for {tf} with shape {table.shape} and dtype {table.dtype}")
            
            try:
                # Convert the structured array to a pandas DataFrame
                # The structured array has a field for each column
                data = table[:]  # Read the entire dataset
                
                # Create a DataFrame from the structured array
                df = pd.DataFrame(data)
                
                # Set the index column if it exists
                if 'index' in df.columns:
                    df.set_index('index', inplace=True)
                
                # Convert timestamp to datetime if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                market_data[tf] = df
                logger.info(f"Successfully loaded data for {tf}: {df.shape}")
                
            except Exception as e:
                logger.error(f"Error loading data for timeframe {tf}: {e}")
                logger.error(f"Table info: shape={table.shape}, dtype={table.dtype}")
                raise
    
    # Check if we successfully loaded any data
    if not market_data:
        raise ValueError("Failed to load market data from H5 files")
        
    data_length = len(next(iter(market_data.values())))  # Get length from first timeframe
    logger.info(f"Data loaded, {data_length} samples found")

    # Choose between FinRL and custom implementation
    if args.use_finrl:
        logger.info("Using FinRL framework for training")
        return train_with_finrl(args)
    else:
        logger.info("Using custom DQN implementation for training")
        return train_with_custom_dqn(args, market_data, data_length, device)

def main():
    """Main function for training the DQN agent."""
    start_time = time.time()
    
    # Set up logging (if not already configured)
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if using FinRL framework
    if args.use_finrl:
        logger.info("Using FinRL framework")
        # No need to load data - environment handles it
        model = train_with_finrl(args)
        
        # Record training duration
        training_duration = time.time() - start_time
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        
        return model
    
    # For custom DQN, load data
    market_data, data_length = load_and_preprocess_market_data(args)
    
    # Train the agent
    agent = train_with_custom_dqn(args, market_data, data_length, device)
    
    # Record training duration
    training_duration = time.time() - start_time
    logger.info(f"Training completed in {training_duration:.2f} seconds")
    
    return agent

# Define setup_logging function - add this before the main function
def setup_logging():
    """Configure logging settings for the application."""
    global logger
    # Only configure if not already configured
    if not len(logger.handlers):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
    return logger

# Add the missing TrainingLoggingCallback class
class TrainingLoggingCallback(BaseCallback):
    """
    Custom callback for logging training progress.
    
    Args:
        check_freq: Frequency at which to log training progress
        save_path: Path to save the model
        verbose: Verbosity level
        save_freq: Frequency at which to save the model
        log_dir: Directory for tensorboard logs
    """
    def __init__(self, check_freq=1000, save_path=None, verbose=1, save_freq=5000, log_dir=None):
        super(TrainingLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.total_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_reward = 0
        
    def _init_callback(self):
        """Initialize the callback."""
        # Create save directory if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        """Called at each step."""
        self.step_count += 1
        
        # Get current environment info
        if self.locals.get("dones", False):
            self.episode_count += 1
            self.total_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            
            # Log every few episodes
            if self.episode_count % 10 == 0:
                mean_reward = np.mean(self.total_rewards[-10:])
                logger.info(f"Episode {self.episode_count} - Mean reward over last 10 episodes: {mean_reward:.2f}")
        else:
            # Add current reward to episode total
            reward = self.locals.get("rewards", [0])[0]
            self.current_episode_reward += reward
                
        # Save the model periodically
        if self.step_count % self.save_freq == 0:
            logger.info(f"Saving model at {self.step_count} steps")
            self.model.save(os.path.join(self.save_path, f'model_{self.step_count}_steps'))
            
        # Log progress periodically
        if self.step_count % self.check_freq == 0:
            logger.info(f"Training progress: {self.step_count} steps completed")
            
        return True

def load_and_preprocess_market_data(args):
    """
    Load market data from the specified path.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of market data dataframes and data length
    """
    # Path to data directory
    data_path = args.data_path
    logger.info(f"Loading data from {data_path}")
    
    market_data = {}
    
    # List available h5 files in the directory
    h5_files = []
    for file_path in Path(data_path).glob("*.h5"):
        h5_files.append(file_path)
    
    if not h5_files:
        # Try looking in subdirectories
        for subdir in Path(data_path).iterdir():
            if subdir.is_dir():
                for file_path in subdir.glob("*.h5"):
                    h5_files.append(file_path)
    
    if not h5_files:
        logger.error(f"No h5 files found in {data_path}")
        raise FileNotFoundError(f"No h5 files found in {data_path}")
    
    # Get first file (assuming all files have similar timeframes)
    h5_path = h5_files[0]
    
    # Get timeframes from the file
    with h5py.File(h5_path, 'r') as f:
        timeframes = list(f.keys())
        logger.info(f"Found timeframes: {timeframes}")
    
    # Load data for each timeframe
    for tf in timeframes:
        # Skip if not in selected timeframes
        if args.timeframes and tf not in args.timeframes:
            continue
            
        with h5py.File(h5_path, 'r') as f:
            # Check if timeframe exists in this file
            if tf not in f:
                logger.warning(f"Timeframe {tf} not found in {h5_path}")
                continue
                
            # Get the timeframe group
            group = f[tf]
            
            # Get the table dataset
            table = group['table']
            logger.info(f"Found table dataset for {tf} with shape {table.shape} and dtype {table.dtype}")
            
            try:
                # Convert the structured array to a pandas DataFrame
                data = np.array(table)
                
                # Create pandas DataFrame
                market_data[tf] = pd.DataFrame({
                    name: data[name] for name in table.dtype.names
                })
                logger.info(f"Successfully loaded data for {tf}: {market_data[tf].shape}")
                
            except Exception as e:
                logger.error(f"Error loading data for timeframe {tf}: {e}")
                logger.error(f"Table info: shape={table.shape}, dtype={table.dtype}")
                raise
    
    # Check if we successfully loaded any data
    if not market_data:
        raise ValueError("Failed to load market data from H5 files")
        
    data_length = len(next(iter(market_data.values())))  # Get length from first timeframe
    logger.info(f"Data loaded, {data_length} samples found")
    
    return market_data, data_length

if __name__ == "__main__":
    main() 