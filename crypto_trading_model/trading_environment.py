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
            # Current position + position history + balance + unrealized PnL + position size
            feature_count += 1 + self.lookback_window + 1 + 1 + 1
        
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
        self.position_size = 0.0  # Store quantity of asset held or sold short
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
        # Calculate portfolio value before action
        portfolio_value_before = self._calculate_portfolio_value(current_price)
        
        # Track if trade was executed
        trade_executed = False
        trade_type = "none"
        profit = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:  # No position -> Long
                # Use 95% of available balance for the position to avoid rounding issues
                use_balance = self.balance * 0.95
                
                # Calculate quantity to buy
                quantity = use_balance / current_price
                fee = quantity * current_price * self.transaction_fee
                
                # Store position size for later calculations
                self.position_size = quantity
                
                # Update state
                self.balance -= (quantity * current_price + fee)
                self.position = 1
                self.position_price = current_price
                self.total_trades += 1
                trade_executed = True
                trade_type = "open_long"
                
            elif self.position == -1:  # Short -> No position (close short)
                # Close short position
                quantity = self.position_size if hasattr(self, 'position_size') else 0
                profit = quantity * (self.position_price - current_price)
                fee = quantity * current_price * self.transaction_fee
                
                # Update state
                self.balance += profit - fee
                self.position = 0
                self.position_price = 0.0
                self.position_size = 0
                self.total_trades += 1
                self.total_profit += profit - fee
                trade_executed = True
                trade_type = "close_short"
                
        elif action == 2:  # Sell
            if self.position == 0:  # No position -> Short
                # Use 95% of available balance as collateral
                use_balance = self.balance * 0.95
                
                # Calculate quantity to sell short
                quantity = use_balance / current_price
                fee = quantity * current_price * self.transaction_fee
                
                # Store position size for later
                self.position_size = quantity
                
                # Update state
                self.balance -= fee
                self.position = -1
                self.position_price = current_price
                self.total_trades += 1
                trade_executed = True
                trade_type = "open_short"
                
            elif self.position == 1:  # Long -> No position (close long)
                # Close long position
                quantity = self.position_size if hasattr(self, 'position_size') else 0
                profit = quantity * (current_price - self.position_price)
                fee = quantity * current_price * self.transaction_fee
                
                # Update state
                self.balance += quantity * current_price - fee
                self.position = 0
                self.position_price = 0.0
                self.position_size = 0
                self.total_trades += 1 
                self.total_profit += profit - fee
                trade_executed = True
                trade_type = "close_long"
        
        # Calculate portfolio value after action
        portfolio_value_after = self._calculate_portfolio_value(current_price)
        
        # Calculate reward components
        
        # 1. Portfolio change component - primary reward signal
        portfolio_change = 0
        if portfolio_value_before > 0:
            portfolio_change = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        # 2. Trade execution penalty - small penalty for excessive trading
        trade_penalty = -0.0005 if trade_executed else 0
        
        # 3. Position holding component - encourage holding profitable positions
        position_reward = 0
        if self.position == 1 and current_price > self.position_price:  # Long position in profit
            position_reward = 0.0002
        elif self.position == -1 and current_price < self.position_price:  # Short position in profit
            position_reward = 0.0002
            
        # 4. Profit realization reward - bonus for taking profits
        profit_reward = 0
        if trade_type == "close_long" and profit > 0:
            profit_reward = 0.001
        elif trade_type == "close_short" and profit > 0:
            profit_reward = 0.001
        
        # 5. Balance maintenance - reward for maintaining/growing account balance
        balance_reward = 0
        if self.balance > self.initial_balance * 0.8:  # Still have most of our balance
            balance_reward = 0.0001
        
        # Combine all reward components
        reward = (
            portfolio_change * 1.0 +  # Main component
            trade_penalty +
            position_reward +
            profit_reward +
            balance_reward
        ) * self.reward_scaling
        
        # Apply reward clipping to handle extreme values
        reward = np.clip(reward, -1.0, 1.0)
        
        # Track returns for visualization
        self.returns.append(reward)
        
        # Log large rewards for debugging
        if abs(reward) > 0.5:
            logger.debug(f"Large reward: {reward:.4f}, Action: {action}, Position: {self.position}, " +
                        f"Portfolio change: {portfolio_change:.4f}")
        
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
        # Calculate position value based on correct position sizing
        position_value = 0.0
        
        if self.position == 1:  # Long
            # Calculate amount of asset based on purchase price
            if self.position_price > 1e-8:
                # For long positions, we bought quantity = balance/price
                quantity = self.position_size if hasattr(self, 'position_size') else 0
                position_value = quantity * current_price
            
        elif self.position == -1:  # Short
            # Calculate profit/loss for short positions
            if self.position_price > 1e-8:
                # For short positions, we track the quantity sold short
                quantity = self.position_size if hasattr(self, 'position_size') else 0
                position_value = quantity * (self.position_price - current_price)
        
        # Total portfolio value = cash balance + position value
        value = self.balance + position_value
        
        # Safety check for numerical stability
        if not np.isfinite(value) or abs(value) > 1e10:
            logger.warning(f"Invalid portfolio value detected: {value}, resetting to balance")
            value = self.balance
            
        return value
    
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
            
            # Handle edge cases near the end of the dataset
            # Ensure we have at least window_size datapoints
            start_idx = max(0, min(self.current_step - self.window_size, len(df) - self.window_size))
            end_idx = min(start_idx + self.window_size, len(df))
            
            # Extract the window of data
            window = df.iloc[start_idx:end_idx]
            
            # If window size is smaller than expected, pad with the last value
            if len(window) < self.window_size:
                padding_needed = self.window_size - len(window)
                # Create padding dataframe with repeated last row
                padding = pd.concat([window.iloc[[-1]]] * padding_needed)
                window = pd.concat([window, padding]).reset_index(drop=True)
            
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
        # 3. Current balance (normalized)
        # 4. Unrealized PnL
        # 5. Position size relative to initial balance
        
        # Calculate unrealized PnL
        current_price = self._get_current_price()
        unrealized_pnl = 0.0
        
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.position_price) / self.position_price if self.position_price > 0 else 0
        elif self.position == -1:  # Short
            unrealized_pnl = (self.position_price - current_price) / self.position_price if self.position_price > 0 else 0
        
        # Position size as percentage of initial balance
        position_size_pct = 0.0
        if self.position != 0:
            position_value = self.position_size * current_price
            position_size_pct = position_value / self.initial_balance
        
        # Combine position features
        position_features = [
            self.position,                                 # Current position
            *self.position_history,                        # Position history
            self.balance / self.initial_balance,           # Normalized balance
            np.clip(unrealized_pnl, -1.0, 1.0),           # Clipped unrealized PnL
            np.clip(position_size_pct, 0.0, 3.0)           # Normalized position size
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
            
            # Calculate position value and unrealized P&L
            position_value = 0.0
            unrealized_pnl = 0.0
            
            if self.position == 1:  # Long
                position_value = self.position_size * current_price
                if self.position_price > 0:
                    unrealized_pnl = self.position_size * (current_price - self.position_price)
            elif self.position == -1:  # Short
                if self.position_price > 0:
                    unrealized_pnl = self.position_size * (self.position_price - current_price)
            
            # Print state with more detailed information
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.data_length}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Cash Balance: ${self.balance:.2f}")
            
            # Position information
            position_type = "NONE" if self.position == 0 else "LONG" if self.position == 1 else "SHORT"
            print(f"\nPosition: {position_type}")
            
            if self.position != 0:
                percent_change = (unrealized_pnl/position_value)*100 if position_value > 0 else 0.00
                print(f"Entry Price: ${self.position_price:.2f}")
                print(f"Size: {self.position_size:.6f} units (${position_value:.2f})")
                print(f"Unrealized P&L: ${unrealized_pnl:.2f} ({percent_change:.2f}%)")
            
            # Portfolio and metrics
            print(f"\nTotal Portfolio Value: ${portfolio_value:.2f}")
            print(f"Total Trades Executed: {self.total_trades}")
            print(f"Total Realized Profit: ${self.total_profit:.2f}")
            
            # Performance metrics
            roi = ((portfolio_value / self.initial_balance) - 1.0) * 100
            print(f"Return on Investment: {roi:.2f}%")
            print(f"{'='*60}\n")
            
        return None
    
    def close(self):
        """Close the environment and release resources"""
        pass 