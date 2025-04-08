#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading environment for cryptocurrency trading using reinforcement learning.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import h5py
from typing import Dict, List, Tuple, Union, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnvironment:
    """
    Trading environment for cryptocurrency trading.
    Simulates a trading environment for reinforcement learning.
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 20,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        reward_scaling: float = 0.01,
        use_position_features: bool = True,
        lookback_window: int = 5,
        device: str = None
    ):
        """
        Initialize the trading environment.
        
        Parameters:
        -----------
        data_path : str
            Path to the HDF5 file containing market data
        window_size : int
            Size of the observation window (in time steps)
        initial_balance : float
            Initial account balance
        transaction_fee : float
            Transaction fee as a percentage of the trade value
        reward_scaling : float
            Scaling factor for rewards
        use_position_features : bool
            Whether to include position features in the observation
        lookback_window : int
            Number of past positions to include in the state
        device : str
            Device to use for computation ('cpu' or 'cuda')
        """
        self.data_path = data_path
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.use_position_features = use_position_features
        self.lookback_window = lookback_window
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load data
        self._load_data()
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = 3
        
        # Reset the environment
        self.reset()
    
    def _load_data(self):
        """Load market data from HDF5 file"""
        try:
            logger.info(f"Loading market data from {self.data_path}")
            
            # Check if the file exists
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
            # Load data from HDF5 file
            self.market_data = {}
            
            with h5py.File(self.data_path, 'r') as f:
                # Extract timeframes
                self.timeframes = list(f.keys())
                logger.info(f"Found timeframes: {self.timeframes}")
                
                # Load data for each timeframe
                for tf in self.timeframes:
                    # Access the main table
                    table_data = f[tf]['table'][:]
                    
                    # Convert structured array to DataFrame
                    df = pd.DataFrame(table_data)
                    self.market_data[tf] = df
                    logger.info(f"Loaded {tf} data with shape {df.shape}")
            
            # Get the primary timeframe (assumed to be the first one)
            self.primary_tf = self.timeframes[0]
            self.data_length = len(self.market_data[self.primary_tf])
            
            # Calculate state dimension
            self.state_dim = self._calculate_state_dim()
            
            logger.info(f"Data loaded successfully. Total samples: {self.data_length}")
            
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _calculate_state_dim(self):
        """Calculate the dimension of the state space"""
        # Count features across timeframes
        feature_count = 0
        for tf, df in self.market_data.items():
            # Count all columns except index and timestamp
            feature_columns = len([col for col in df.columns if col not in ['index', 'timestamp']])
            # Multiply by window size since we're using a window of data
            feature_count += feature_columns * self.window_size
        
        # Add position features if used
        if self.use_position_features:
            # Current position + position history + balance + unrealized PnL
            feature_count += 1 + self.lookback_window + 2
        
        return feature_count
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
        --------
        numpy.ndarray
            Initial observation
        """
        # Reset position in the data
        self.current_step = self.window_size
        
        # Reset account state
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_price = 0.0
        self.position_history = [0] * self.lookback_window
        
        # Reset metrics
        self.total_trades = 0
        self.total_profit = 0.0
        self.returns = []
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action : int
            Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
        --------
        tuple
            (observation, reward, done, info)
        """
        # Get current price
        current_price = self._get_current_price()
        
        # Execute action and calculate reward
        reward = self._execute_action(action, current_price)
        
        # Update position history
        self.position_history.pop(0)
        self.position_history.append(self.position)
        
        # Move to the next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.data_length - 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate additional info
        info = {
            'step': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit
        }
        
        return observation, reward, done, info
    
    def _get_current_price(self):
        """Get the current price"""
        # Use the close price of the primary timeframe
        return self.market_data[self.primary_tf]['close'].iloc[self.current_step]
    
    def _execute_action(self, action, current_price):
        """
        Execute the trading action and calculate reward.
        
        Parameters:
        -----------
        action : int
            Action to take (0: hold, 1: buy, 2: sell)
        current_price : float
            Current price
            
        Returns:
        --------
        float
            Reward for the action
        """
        reward = 0.0
        
        # Calculate the value before action
        portfolio_value_before = self._calculate_portfolio_value(current_price)
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:  # No position -> Long
                # Calculate the amount to buy (use all available balance)
                amount = self.balance / current_price
                fee = amount * current_price * self.transaction_fee
                
                # Update state
                self.balance -= (amount * current_price + fee)
                self.position = 1
                self.position_price = current_price
                self.total_trades += 1
                
                # Small negative reward for transaction fee
                reward -= fee * self.reward_scaling
                
            elif self.position == -1:  # Short -> No position
                # Close short position
                profit = self.position_price - current_price
                amount = abs(self.position_price)
                fee = amount * current_price * self.transaction_fee
                
                # Update state
                self.balance += amount * profit - fee
                self.position = 0
                self.position_price = 0.0
                self.total_trades += 1
                self.total_profit += profit
                
                # Reward based on profit
                reward += profit * self.reward_scaling - fee * self.reward_scaling
                
        elif action == 2:  # Sell
            if self.position == 0:  # No position -> Short
                # Calculate the amount to sell (use all available balance as collateral)
                amount = self.balance / current_price
                fee = amount * current_price * self.transaction_fee
                
                # Update state
                self.balance -= fee
                self.position = -1
                self.position_price = current_price
                self.total_trades += 1
                
                # Small negative reward for transaction fee
                reward -= fee * self.reward_scaling
                
            elif self.position == 1:  # Long -> No position
                # Close long position
                profit = current_price - self.position_price
                amount = self.balance / self.position_price
                fee = amount * current_price * self.transaction_fee
                
                # Update state
                self.balance += amount * current_price - fee
                self.position = 0
                self.position_price = 0.0
                self.total_trades += 1
                self.total_profit += profit
                
                # Reward based on profit
                reward += profit * self.reward_scaling - fee * self.reward_scaling
        
        # Calculate the value after action
        portfolio_value_after = self._calculate_portfolio_value(current_price)
        
        # Add reward based on change in portfolio value (unrealized PnL)
        value_change = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        reward += value_change * self.reward_scaling
        
        # Track returns
        self.returns.append(value_change)
        
        return reward
    
    def _calculate_portfolio_value(self, current_price):
        """
        Calculate the total portfolio value.
        
        Parameters:
        -----------
        current_price : float
            Current price
            
        Returns:
        --------
        float
            Total portfolio value
        """
        # Calculate the value of the position
        position_value = 0.0
        if self.position == 1:  # Long
            # Calculate the amount of the asset
            amount = self.balance / self.position_price
            position_value = amount * current_price
        elif self.position == -1:  # Short
            # Calculate the profit/loss of the short position
            amount = self.balance / self.position_price
            position_value = amount * (self.position_price - current_price)
        
        # Total portfolio value = cash balance + position value
        return self.balance + position_value
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
        --------
        numpy.ndarray
            Current observation
        """
        # Extract market features
        market_features = self._get_market_features()
        
        # Add position features if used
        if self.use_position_features:
            position_features = self._get_position_features()
            # Combine market and position features
            features = np.concatenate([market_features, position_features])
        else:
            features = market_features
        
        # Normalize features (if needed)
        # features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)
    
    def _get_market_features(self):
        """
        Extract market features from the data.
        
        Returns:
        --------
        numpy.ndarray
            Market features
        """
        # Extract window of data from each timeframe
        features = []
        
        for tf in self.timeframes:
            # Get the dataframe for this timeframe
            df = self.market_data[tf]
            
            # Extract the window of data
            window = df.iloc[self.current_step - self.window_size:self.current_step]
            
            # Extract features (excluding index and timestamp)
            for col in df.columns:
                if col not in ['index', 'timestamp']:
                    # Use the values in the window as a flat array
                    feature_values = window[col].values.flatten()
                    features.append(feature_values)
        
        # Stack all features and flatten to 1D array
        return np.hstack(features).astype(np.float32)
    
    def _get_position_features(self):
        """
        Extract position features.
        
        Returns:
        --------
        numpy.ndarray
            Position features
        """
        # Position features:
        # 1. Current position (-1, 0, 1)
        # 2. Position history
        # 3. Current balance
        # 4. Unrealized PnL
        
        # Calculate unrealized PnL
        current_price = self._get_current_price()
        unrealized_pnl = 0.0
        
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.position_price) / self.position_price
        elif self.position == -1:  # Short
            unrealized_pnl = (self.position_price - current_price) / self.position_price
        
        # Combine position features
        position_features = [
            self.position,                     # Current position
            *self.position_history,            # Position history
            self.balance / self.initial_balance,  # Normalized balance
            unrealized_pnl                    # Unrealized PnL
        ]
        
        return np.array(position_features, dtype=np.float32)
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters:
        -----------
        mode : str
            Rendering mode
        """
        if mode == 'human':
            # Get current price
            current_price = self._get_current_price()
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            
            # Print state
            print(f"Step: {self.current_step}/{self.data_length}")
            print(f"Price: {current_price:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Position: {self.position}")
            print(f"Portfolio Value: {portfolio_value:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Total Profit: {self.total_profit:.2f}")
            print("-" * 50)
    
    def close(self):
        """Close the environment and release resources"""
        pass 