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
            df: DataFrame with market data
            state_space: Size of the state space
            stock_dim: Number of assets
            action_space: Size of the action space
            **kwargs: Additional arguments for compatibility with FinRL
        """
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
        
        # Simple reward
        reward = 0
        
        # Update state (simple implementation)
        if self.df is not None and self.day < len(self.df):
            self.data = self.df.iloc[self.day]
            # In a real implementation, we would update the state based on the action and new data
        
        # For now, just return zeros as the state
        return self.state, reward, done, {"terminal_observation": self.state if done else None}
        
    def render(self, mode='human'):
        """Render the environment."""
        pass
        
    def seed(self, seed=None):
        """Set the random seed."""
        if seed is not None:
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
        Initialize the wrapper.
        
        Args:
            env: Environment to wrap
            state_space: Expected state space dimension
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
                # Default action space if not available
                act_space = gymnasium.spaces.Discrete(3)  # Sell, Hold, Buy
            
            # We are wrapping the underlying gym env manually
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
            
            # Ensure observation space is properly defined
            self.observation_space = gymnasium.spaces.Box(
                low=-10.0, high=10.0, shape=(state_space,), dtype=np.float32
            )
            logger.info(f"Wrapped Gymnasium environment with observation space: {self.observation_space}")
        
        # Store the actual observation dimension for comparison
        self.actual_state_space = None
        # Track step count for logging
        self.step_counter = 0
    
    def reset(self, **kwargs):
        """Reset the environment, with compatibility for both gym and gymnasium APIs."""
        try:
            # Try gymnasium API first (for newer environments)
            obs, info = self.env.reset(**kwargs)
            # Older reset API
            if isinstance(obs, tuple) and len(obs) == 2:
                obs, info = obs[0], obs[1]
        except (ValueError, TypeError):
            try:
                # Fall back to gym API
                obs = self.env.reset(**kwargs)
                info = {}
            except (ValueError, TypeError):
                # Last resort - try without kwargs
                obs = self.env.reset()
                info = {}
        
        # Check and update observation space if needed
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
            
            logger.info(f"Reset observation shape: {obs.shape if hasattr(obs, 'shape') else 'scalar'}, observation space shape: {self.observation_space.shape}")
            
            # No need to reshape the observation, just clip it to the bounds
            obs = np.clip(obs, -10.0, 10.0)
            
        else:
            logger.warning(f"Reset returned non-numpy observation of type {type(obs)}")
            
        return obs, info
    
    def step(self, action):
        """Step the environment, with compatibility for both gym and gymnasium APIs."""
        try:
            # Try gymnasium API first
            result = self.env.step(action)
            if len(result) == 5:  # gymnasium API (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # gym API (obs, reward, done, info)
                obs, reward, done, info = result
                terminated, truncated = done, False
        except (ValueError, TypeError):
            # Fall back to gym API
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
            
            # Update observation space if needed
            if self.actual_state_space is None or self.actual_state_space != len(obs):
                self.actual_state_space = len(obs)
                logger.info(f"Step returned observation with different dimension: {self.actual_state_space}")
                self.observation_space = gymnasium.spaces.Box(
                    low=-10.0, high=10.0, shape=(self.actual_state_space,), dtype=np.float32
                )
        
        return obs, reward, terminated, truncated, info
    
    def seed(self, seed=None):
        """Set random seed."""
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
        df: DataFrame with market data
        args: Command line arguments
        num_workers: Number of parallel environments
        
    Returns:
        VectorEnv with multiple environments
    """
    logger.info(f"Creating vectorized environment with {num_workers} workers")
    
    # First, try to import the environment class
    try:
        StockTradingEnvClass = None
        env_import_paths = [
            # Try cryptocurrency specific environments first
            'finrl.meta.env_cryptocurrency_trading.env_crypto.CryptocurrencyTradingEnv',
            'finrl.applications.env.crypto.CryptocurrencyTradingEnv',
            # Then fall back to stock trading environments
            'finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv',
            'finrl.env.env_stocktrading.StockTradingEnv',
            'finrl.applications.env.stocktrading.StockTradingEnv'
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
    
    # Create per-asset transaction cost arrays
    buy_cost_pct_list = [transaction_cost_pct] * stock_dim
    sell_cost_pct_list = [transaction_cost_pct] * stock_dim
    
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
            buy_cost_pct=buy_cost_pct_list,  # Pass as list
            sell_cost_pct=sell_cost_pct_list,  # Pass as list
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
                    'buy_cost_pct': buy_cost_pct_list,  # Pass as list
                    'sell_cost_pct': sell_cost_pct_list,  # Pass as list
                    'reward_scaling': reward_scaling,
                    'print_verbosity': 1 if idx == 0 else 0  # Only print verbose output for the first env
                }
                
                # Create the environment
                base_env = StockTradingEnvClass(**env_params)
                
                # Wrap the environment to ensure consistent observation shape
                wrapped_env = StockTradingEnvWrapper(base_env, state_space=state_space)
                
                # Wrap with monitoring wrapper for metrics
                return wrap_env_with_monitor(wrapped_env)
            except Exception as e:
                logger.error(f"Error creating environment {idx}: {e}")
                logger.error(traceback.format_exc())
                # Provide a fallback environment if creation fails
                base_env = BaseStockTradingEnv(df, state_space=state_space)
                wrapped_env = StockTradingEnvWrapper(base_env, state_space=state_space)
                return wrap_env_with_monitor(wrapped_env)
        
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
    
    try:
        # Import stable-baselines3 and torch.nn
        from stable_baselines3 import PPO, DDPG, TD3, SAC
        import torch.nn as nn
        from stable_baselines3.common.callbacks import BaseCallback
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
            
        def _on_step(self):
            # Track step reward
            if hasattr(self, 'locals') and 'rewards' in self.locals:
                rewards = self.locals['rewards']
                for i, reward in enumerate(rewards):
                    # Update the current episode rewards
                    self.current_episode_reward += reward
                    self.current_episode_length += 1
                    
                    # Log environment info if available
                    if 'infos' in self.locals:
                        info = self.locals['infos'][i]
                        # Log portfolio value
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
                
                for i, done in enumerate(dones):
                    if done:
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
                        
                        # Reset episode tracking
                        self.current_episode_reward = 0
                        self.current_episode_length = 0
                
            return True
    
    # Create the model
    logger.info(f"Creating {finrl_model.upper()} model with policy_kwargs: {policy_kwargs}")
    try:
        model = model_class(
            "MlpPolicy", 
            vec_env, 
            learning_rate=getattr(args, 'learning_rate', 0.0003),
            gamma=getattr(args, 'gamma', 0.99),
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_logs"
        )
        
        # Train the model with callback
        total_timesteps = getattr(args, 'timesteps', 50000)  # Default to 50000 if not specified
        logger.info(f"Training model for {total_timesteps} timesteps")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=TensorboardCallback(),
            tb_log_name=finrl_model
        )
        
        # Save the model
        save_model = getattr(args, 'save_model', True)  # Default to True if not specified
        if save_model:
            model_path = f"./models/finrl_{finrl_model}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model.save(model_path)
            logger.info(f"Saved model to {model_path}")
        
        return model
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
    parser.add_argument('--timesteps', type=int, default=50000, help='Number of timesteps to train')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel environments')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--initial_balance', type=float, default=1000000.0, help='Initial balance')
    parser.add_argument('--transaction_fee', type=float, default=0.001, help='Transaction fee')
    
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
    Create synthetic data for multiple tickers between start_date and end_date.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
    Returns:
        pandas.DataFrame: Synthetic data formatted for FinRL
    """
    print(f"Creating synthetic data for {len(tickers)} symbols from {start_date} to {end_date}")
    
    # Convert dates to pandas datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create empty list to store data for each ticker
    all_data = []
    
    # Generate data for each ticker
    for ticker in tickers:
        # Create base price and volatility
        base_price = np.random.uniform(100, 10000)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Generate price data
        prices = [base_price]
        for i in range(1, len(dates)):
            # Random walk with drift
            change = np.random.normal(0.0001, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Convert to numpy array for vectorized operations
        prices = np.array(prices)
        
        # Create OHLCV data
        daily_volatility = volatility / np.sqrt(252)
        
        df = pd.DataFrame({
            'date': dates,
            'tic': ticker,
            'open': prices * np.random.uniform(0.98, 1.02, len(dates)),
            'high': prices * np.random.uniform(1.01, 1.04, len(dates)),
            'low': prices * np.random.uniform(0.96, 0.99, len(dates)),
            'close': prices,
            'volume': np.random.uniform(100000, 1000000, len(dates))
        })
        
        # Ensure high is the highest price and low is the lowest price
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Add to the list
        all_data.append(df)
    
    # Concatenate all data
    result_df = pd.concat(all_data, ignore_index=True)
    
    # Add technical indicators
    # Add SMA
    for window in [5, 10, 20, 60, 120]:
        result_df[f'close_{window}_sma'] = result_df.groupby('tic')['close'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    # Add EMA
    for window in [5, 10, 20, 60, 120]:
        result_df[f'close_{window}_ema'] = result_df.groupby('tic')['close'].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
    
    # RSI
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    result_df['rsi_14'] = result_df.groupby('tic')['close'].transform(
        lambda x: calculate_rsi(x, window=14)
    )
    
    # MACD
    def calculate_macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    for ticker in tickers:
        ticker_data = result_df[result_df['tic'] == ticker]
        macd, macd_signal, macd_hist = calculate_macd(ticker_data['close'])
        
        result_df.loc[result_df['tic'] == ticker, 'macd'] = macd.values
        result_df.loc[result_df['tic'] == ticker, 'macd_signal'] = macd_signal.values
        result_df.loc[result_df['tic'] == ticker, 'macd_hist'] = macd_hist.values
    
    # CCI
    def calculate_cci(high, low, close, window=20):
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window, min_periods=1).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - sma_tp) / (0.015 * mad.fillna(0))
        return cci
    
    result_df['cci_30'] = result_df.groupby('tic').apply(
        lambda x: calculate_cci(x['high'], x['low'], x['close'], window=30)
    ).reset_index(level=0, drop=True)
    
    # DX (Directional Index)
    def calculate_dx(high, low, close, window=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0)
        minus_dm = minus_dm.abs().where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(window=window).sum() / tr.rolling(window=window).sum())
        minus_di = 100 * (minus_dm.rolling(window=window).sum() / tr.rolling(window=window).sum())
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).where(plus_di + minus_di != 0, 1))
        return dx
    
    result_df['dx_30'] = result_df.groupby('tic').apply(
        lambda x: calculate_dx(x['high'], x['low'], x['close'], window=30)
    ).reset_index(level=0, drop=True)
    
    # Volatility
    result_df['volatility_30'] = result_df.groupby('tic')['close'].transform(
        lambda x: x.pct_change().rolling(window=30).std()
    )
    
    # Volume indicators
    result_df['volume_change'] = result_df.groupby('tic')['volume'].transform(
        lambda x: x.pct_change()
    )
    
    result_df['volume_norm'] = result_df.groupby('tic')['volume'].transform(
        lambda x: (x - x.rolling(window=30, min_periods=1).min()) / 
                 (x.rolling(window=30, min_periods=1).max() - x.rolling(window=30, min_periods=1).min() + 1e-10)
    )
    
    # Create day column with factorized dates for FinRL compatibility
    result_df['day'] = result_df['date'].factorize()[0]
    
    # Fill NaN values
    result_df = result_df.fillna(0)
    
    # Sort by day and tic for FinRL compatibility
    result_df = result_df.sort_values(['day', 'tic']).reset_index(drop=True)
    
    print(f"Created synthetic data with shape {result_df.shape}")
    print(f"Day range: {result_df['day'].min()} - {result_df['day'].max()}")
    
    return result_df

def ensure_technical_indicators(df, tech_indicator_list):
    """
    Ensure all required technical indicators exist in the dataframe.
    Adds any missing indicators to avoid KeyError when initializing StockTradingEnv.
    
    Args:
        df: DataFrame containing OHLCV data
        tech_indicator_list: List of technical indicators needed by the environment
        
    Returns:
        DataFrame with all required technical indicators
    """
    logger.info(f"Checking for required technical indicators: {tech_indicator_list}")
    
    # Track missing indicators
    missing_indicators = [ind for ind in tech_indicator_list if ind not in df.columns]
    if missing_indicators:
        logger.warning(f"Missing technical indicators: {missing_indicators}")
        logger.info("Adding missing technical indicators...")
        
        # Import ta for technical analysis
        try:
            import ta
        except ImportError:
            logger.error("Technical analysis library 'ta' not installed. Using fallback methods.")
            ta = None
        
        # Add each missing indicator
        for indicator in missing_indicators:
            if indicator == 'cci_30':
                # Calculate CCI with period 30
                if ta:
                    # Use ta library if available
                    cci = ta.trend.CCIIndicator(
                        high=df['high'], low=df['low'], close=df['close'], window=30
                    )
                    df['cci_30'] = cci.cci()
                else:
                    # Fallback implementation
                    tp = (df['high'] + df['low'] + df['close']) / 3
                    ma = tp.rolling(window=30).mean()
                    md = tp.rolling(window=30).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
                    df['cci_30'] = (tp - ma) / (0.015 * md)
                logger.info("Added CCI-30 indicator")
            
            elif indicator == 'dx_30':
                # Calculate DX with period 30
                if ta:
                    adx = ta.trend.ADXIndicator(
                        high=df['high'], low=df['low'], close=df['close'], window=30
                    )
                    df['dx_30'] = adx.adx()
                else:
                    # Simplified DX calculation
                    high_diff = df['high'].diff()
                    low_diff = df['low'].diff()
                    plus_dm = (high_diff > 0) & (high_diff > low_diff.abs())
                    plus_dm = high_diff.where(plus_dm, 0)
                    minus_dm = (low_diff < 0) & (low_diff.abs() > high_diff)
                    minus_dm = low_diff.abs().where(minus_dm, 0)
                    
                    tr = pd.concat([
                        df['high'] - df['low'],
                        (df['high'] - df['close'].shift()).abs(),
                        (df['low'] - df['close'].shift()).abs()
                    ], axis=1).max(axis=1)
                    
                    plus_di = 100 * (plus_dm.rolling(30).sum() / tr.rolling(30).sum())
                    minus_di = 100 * (minus_dm.rolling(30).sum() / tr.rolling(30).sum())
                    df['dx_30'] = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
                logger.info("Added DX-30 indicator")
            
            elif 'close_' in indicator and '_sma' in indicator:
                # Extract window size from indicator name (e.g., close_5_sma -> 5)
                window = int(indicator.split('_')[1])
                df[indicator] = df['close'].rolling(window=window, min_periods=1).mean()
                logger.info(f"Added SMA indicator with window {window}")
            
            elif 'close_' in indicator and '_ema' in indicator:
                # Extract window size from indicator name (e.g., close_5_ema -> 5)
                window = int(indicator.split('_')[1])
                df[indicator] = df['close'].ewm(span=window, adjust=False).mean()
                logger.info(f"Added EMA indicator with window {window}")
            
            elif indicator == 'volatility_30':
                df[indicator] = df['close'].pct_change().rolling(window=30).std()
                logger.info("Added volatility indicator with window 30")
            
            elif indicator == 'volume_change':
                df[indicator] = df['volume'].pct_change()
                logger.info("Added volume change indicator")
            
            elif indicator == 'volume_norm':
                min_vol = df['volume'].rolling(window=30, min_periods=1).min()
                max_vol = df['volume'].rolling(window=30, min_periods=1).max()
                df[indicator] = (df['volume'] - min_vol) / (max_vol - min_vol + 1e-10)
                logger.info("Added normalized volume indicator")
            
            elif indicator == 'macd':
                # Calculate MACD
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema_12 - ema_26
                logger.info("Added MACD indicator")
                
                # Also add signal and histogram if they're in the list
                if 'macd_signal' in missing_indicators:
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    logger.info("Added MACD signal line")
                
                if 'macd_hist' in missing_indicators:
                    if 'macd_signal' not in df.columns:
                        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    logger.info("Added MACD histogram")
            
            elif indicator == 'rsi_14':
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).fillna(0)
                loss = -delta.where(delta < 0, 0).fillna(0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
                df['rsi_14'] = 100 - (100 / (1 + rs))
                logger.info("Added RSI-14 indicator")
            
            else:
                # For any other missing indicator, add a column of zeros as placeholder
                logger.warning(f"Unknown indicator '{indicator}', adding zeros as placeholder")
                df[indicator] = 0
    
    # Fill any NaN values that might have been introduced
    df = df.fillna(0)
    
    # Verify all indicators are now present
    still_missing = [ind for ind in tech_indicator_list if ind not in df.columns]
    if still_missing:
        logger.error(f"Still missing indicators after processing: {still_missing}")
    else:
        logger.info("All required technical indicators are present")
    
    return df

def wrap_env_with_monitor(env):
    """Wrap environment with a monitoring wrapper for logging metrics."""
    try:
        # Try with Gymnasium first since we know it's installed (version 1.1.1)
        from gymnasium.wrappers import RecordEpisodeStatistics
        logger.info("Using gymnasium's RecordEpisodeStatistics wrapper")
        return RecordEpisodeStatistics(env)
    except (ImportError, AttributeError):
        try:
            # Try SB3's Monitor as a fallback
            from stable_baselines3.common.monitor import Monitor
            logger.info("Using SB3's Monitor wrapper for logging")
            return Monitor(env, None, allow_early_resets=True)
        except (ImportError, AttributeError):
            # As a last resort, return the unwrapped environment
            logger.warning("No monitoring wrapper available, returning unwrapped environment")
            return env

def main():
    """Main entry point for the script."""
    # Record start time
    start_time = time.time()
    
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
        logger.info("Using FinRL framework")
        
        # Set default values if not provided in args
        start_date = getattr(args, 'start_date', '2018-01-01')
        end_date = getattr(args, 'end_date', '2021-12-31')
        
        # Set tickers - use provided tickers or default to ['BTC']
        tickers = getattr(args, 'tickers', ['BTC', 'ETH', 'LTC'])
        if not isinstance(tickers, list):
            # Split comma-separated string into list if needed
            tickers = [t.strip() for t in tickers.split(',')]
        
        # Ensure at least BTC is included (as fallback)
        if not tickers:
            tickers = ['BTC']
            
        # Get number of workers
        num_workers = getattr(args, 'num_workers', 4)
        
        # Train the model with required parameters
        model = train_with_finrl(
            args=args,
            logger=logger,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            num_workers=num_workers
        )
        
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

if __name__ == "__main__":
    main() 