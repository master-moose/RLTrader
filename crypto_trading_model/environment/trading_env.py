"""
Trading environment for reinforcement learning-based crypto trading.

This module implements a custom OpenAI Gym-compatible environment
for cryptocurrency trading with realistic constraints.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Union, Tuple, Optional
import logging
import math

# Import from parent directory
import sys
sys.path.append('..')
from config import TRADING_ENV_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trading_env')

class CryptoTradingEnv(gym.Env):
    """
    Custom trading environment for cryptocurrency trading.
    
    Supports:
    - Multiple timeframes
    - Long and short positions
    - Realistic transaction costs and slippage
    - Customizable reward functions
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        reward_function: str = 'sharpe',
        window_size: int = 50,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.2,
        stop_loss: float = 0.05,
        take_profit: float = 0.15,
        feature_columns: List[str] = None,
        multiframe_data: Dict[str, pd.DataFrame] = None
    ):
        """
        Initialize the trading environment.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data and technical indicators
        reward_function : str
            Reward function to use ('pnl', 'sharpe', 'sortino', 'calmar')
        window_size : int
            Number of time steps for state window
        initial_balance : float
            Initial account balance
        transaction_fee : float
            Transaction fee as a percentage
        slippage : float
            Slippage as a percentage
        max_position_size : float
            Maximum position size as a fraction of balance
        stop_loss : float
            Stop loss percentage
        take_profit : float
            Take profit percentage
        feature_columns : List[str]
            List of feature columns to include in the state
        multiframe_data : Dict[str, pd.DataFrame]
            Dictionary of additional dataframes for multiple timeframes
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.data = data
        self.reward_function = reward_function
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Use all columns except timestamp, open, high, low, close, volume if not specified
        if feature_columns is None:
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            self.feature_columns = [col for col in data.columns if col not in price_columns]
        else:
            self.feature_columns = feature_columns
        
        self.multiframe_data = multiframe_data
        
        # Calculate number of features for state space
        n_features = len(self.feature_columns)
        
        # Add features from multiframe data if provided
        self.multiframe_features = []
        if self.multiframe_data is not None:
            for timeframe, df in self.multiframe_data.items():
                # Extract only feature columns from each timeframe
                timeframe_features = [f"{timeframe}_{col}" for col in self.feature_columns]
                self.multiframe_features.extend(timeframe_features)
                n_features += len(timeframe_features)
        
        # Calculate state dimension
        self.state_dim = n_features + 5  # OHLCV + features + position info
        
        # Define action and observation spaces
        # Actions: 0 (sell/short), 1 (hold), 2 (buy/long)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: normalized price data and indicators
        # Using finite bounds (-10.0, 10.0) instead of infinite (-np.inf, np.inf) for SB3 compatibility
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        """Reset the environment to its initial state and return the initial observation."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.base_position = 0
        self.quote_position = self.initial_balance
        self.position_history = [self.base_position]
        self.portfolio_value_history = [self.portfolio_value]
        self.positions = []
        
        # Get initial state
        observation = self._get_observation()
        
        # Clip observation to defined bounds for SB3 compatibility
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        
        self.previous_portfolio_value = self.portfolio_value
        
        return observation, {}  # Return initial observation and empty info dict for gym compatibility
    
    def step(self, action):
        """Take an action in the environment and return the next state, reward, and whether
        the episode is done."""
        # Record previous portfolio value for reward calculation
        self.previous_portfolio_value = self.portfolio_value
        
        # Execute the trading action
        self._execute_trade_action(action)
        
        # Update the current step
        self.current_step += 1
        
        # Calculate portfolio value
        current_price = self.data.iloc[self.current_step]['close']
        self.portfolio_value = self.base_position * current_price + self.quote_position
        
        # Record history
        self.position_history.append(self.base_position)
        self.portfolio_value_history.append(self.portfolio_value)
        
        # Get observation, reward, and done
        observation = self._get_observation()
        
        # Clip observation to defined bounds for SB3 compatibility
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        
        reward = self._calculate_reward()
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'current_price': current_price,
            'base_position': self.base_position,
            'quote_position': self.quote_position
        }
        
        return observation, reward, done, False, info  # False is for truncated (gym>=0.26.0)
    
    def _execute_trade_action(self, action):
        """
        Execute trading action.
        
        Parameters:
        -----------
        action : int
            Action to take: 0 (sell/short), 1 (hold), 2 (buy/long)
        """
        # Get current price data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        
        # Track the previous equity for reward calculation
        prev_equity = self.portfolio_value
        
        # Execute the action
        executed_action = self._execute_action(action, current_price)
        
        # Update equity with current positions
        self._update_equity(current_price)
        
        # Update position
        self._update_position(executed_action, current_price)
        
        # Calculate return
        if self.portfolio_value_history:
            prev_equity = self.portfolio_value_history[-1]
            ret = (self.portfolio_value - prev_equity) / prev_equity
        else:
            ret = 0
        
        # Update history
        self.position_history.append(self.base_position)
        self.portfolio_value_history.append(self.portfolio_value)
    
    def _execute_action(self, action, current_price):
        """
        Execute trading action.
        
        Parameters:
        -----------
        action : int
            Action to take
        current_price : float
            Current price
            
        Returns:
        --------
        str
            Description of executed action
        """
        executed_action = "hold"  # Default action
        
        # Check stop loss and take profit if we have a position
        if self.base_position != 0:
            # Calculate current position value
            position_value = self.base_position * current_price
            
            # Calculate unrealized P&L
            if self.base_position > 0:  # Long position
                unrealized_pnl = (current_price - self.position_price) / self.position_price
                # Stop loss
                if unrealized_pnl <= -self.stop_loss:
                    # Close position
                    self._close_position(current_price)
                    executed_action = "stop_loss"
                    return executed_action
                # Take profit
                elif unrealized_pnl >= self.take_profit:
                    # Close position
                    self._close_position(current_price)
                    executed_action = "take_profit"
                    return executed_action
            else:  # Short position
                unrealized_pnl = (self.position_price - current_price) / self.position_price
                # Stop loss
                if unrealized_pnl <= -self.stop_loss:
                    # Close position
                    self._close_position(current_price)
                    executed_action = "stop_loss"
                    return executed_action
                # Take profit
                elif unrealized_pnl >= self.take_profit:
                    # Close position
                    self._close_position(current_price)
                    executed_action = "take_profit"
                    return executed_action
        
        # Process action
        if action == 0:  # Sell/Short
            if self.base_position > 0:  # Close long position
                self._close_position(current_price)
                executed_action = "close_long"
            elif self.base_position == 0:  # Open short position
                self._open_position(current_price, -1)
                executed_action = "open_short"
        
        elif action == 2:  # Buy/Long
            if self.base_position < 0:  # Close short position
                self._close_position(current_price)
                executed_action = "close_short"
            elif self.base_position == 0:  # Open long position
                self._open_position(current_price, 1)
                executed_action = "open_long"
        
        # Update equity with current positions
        self._update_equity(current_price)
        
        return executed_action
    
    def _open_position(self, price, direction):
        """
        Open a new position.
        
        Parameters:
        -----------
        price : float
            Entry price
        direction : int
            Position direction (1 for long, -1 for short)
        """
        # Calculate position size
        position_size = self.quote_position * self.max_position_size
        
        # Calculate number of units
        units = position_size / price
        
        # Apply slippage to price
        executed_price = price * (1 + direction * self.slippage)
        
        # Apply transaction fee
        fee = position_size * self.transaction_fee
        self.quote_position -= fee
        
        # Update position
        self.base_position = units * direction
        self.position_price = executed_price
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'type': 'open',
            'direction': direction,
            'price': executed_price,
            'units': abs(self.base_position),
            'fee': fee
        })
    
    def _close_position(self, price):
        """
        Close the current position.
        
        Parameters:
        -----------
        price : float
            Exit price
        """
        if self.base_position == 0:
            return
        
        direction = 1 if self.base_position > 0 else -1
        
        # Apply slippage to price
        executed_price = price * (1 - direction * self.slippage)
        
        # Calculate position value
        position_value = abs(self.base_position) * executed_price
        
        # Apply transaction fee
        fee = position_value * self.transaction_fee
        
        # Update equity
        self.portfolio_value -= fee
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'type': 'close',
            'direction': -direction,
            'price': executed_price,
            'units': abs(self.base_position),
            'fee': fee
        })
        
        # Reset position
        self.base_position = 0
        self.position_price = 0
    
    def _update_equity(self, current_price):
        """
        Update account equity based on current positions.
        
        Parameters:
        -----------
        current_price : float
            Current price
        """
        position_value = self.base_position * current_price
        self.portfolio_value = self.quote_position + position_value
        
        # Calculate return
        if self.portfolio_value_history:
            prev_equity = self.portfolio_value_history[-1]
            ret = (self.portfolio_value - prev_equity) / prev_equity
        else:
            ret = 0
        
        self.returns.append((self.current_step, self.portfolio_value, ret))
    
    def _get_observation(self):
        """
        Construct the observation (state) for the agent.
        
        Returns:
        --------
        np.ndarray
            Observation array
        """
        # Get the window of data
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Extract price data and normalize
        price_data = window_data[['open', 'high', 'low', 'close', 'volume']].values
        
        # Simple normalization: divide by the first close price
        base_price = price_data[0, 3]  # First close price
        normalized_prices = price_data / base_price
        
        # Extract feature data
        if self.feature_columns:
            feature_data = window_data[self.feature_columns].values
            # Normalize features
            feature_data = (feature_data - np.mean(feature_data, axis=0)) / (np.std(feature_data, axis=0) + 1e-10)
        else:
            feature_data = np.array([])
        
        # Combine price and feature data
        combined_data = np.hstack([normalized_prices, feature_data])
        
        # Get current price for position calculations
        current_price = self.data.iloc[self.current_step]['close']
        
        # Add position information
        # Normalize position size
        normalized_position = self.base_position * current_price / self.initial_balance
        
        # Normalize position price
        normalized_position_price = self.position_price / current_price if self.position_price > 0 else 0
        
        # Current balance ratio to initial balance
        balance_ratio = self.quote_position / self.initial_balance
        
        # Duration of current position
        if self.base_position != 0 and len(self.trades) > 0:
            last_open_trade = next((t for t in reversed(self.trades) if t['type'] == 'open'), None)
            if last_open_trade:
                position_duration = self.current_step - last_open_trade['step']
                normalized_duration = position_duration / self.window_size
            else:
                normalized_duration = 0
        else:
            normalized_duration = 0
        
        # Unrealized P&L
        if self.base_position != 0:
            if self.base_position > 0:  # Long position
                unrealized_pnl = (current_price - self.position_price) / self.position_price
            else:  # Short position
                unrealized_pnl = (self.position_price - current_price) / self.position_price
        else:
            unrealized_pnl = 0
        
        # Position features
        position_features = np.array([
            normalized_position,
            normalized_position_price,
            balance_ratio,
            normalized_duration,
            unrealized_pnl
        ])
        
        # Combine all features into one observation
        observation = np.concatenate([combined_data.flatten(), position_features])
        
        return observation
    
    def _calculate_reward(self):
        """
        Calculate reward based on the chosen reward function.
        
        Returns:
        --------
        float
            Calculated reward
        """
        if self.reward_function == 'pnl':
            # Simple P&L reward
            reward = (self.portfolio_value - self.previous_portfolio_value) / self.initial_balance
        
        elif self.reward_function == 'sharpe':
            # Sharpe ratio-based reward
            if len(self.returns) > 1:
                returns = [r[2] for r in self.returns[-30:]]  # Use last 30 returns
                reward = np.mean(returns) / (np.std(returns) + 1e-10)
            else:
                reward = 0
        
        elif self.reward_function == 'sortino':
            # Sortino ratio-based reward
            if len(self.returns) > 1:
                returns = [r[2] for r in self.returns[-30:]]  # Use last 30 returns
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    reward = np.mean(returns) / (np.std(downside_returns) + 1e-10)
                else:
                    reward = np.mean(returns) * 10  # High reward for no downside
            else:
                reward = 0
        
        elif self.reward_function == 'calmar':
            # Calmar ratio-based reward
            if len(self.returns) > 1:
                # Calculate max drawdown
                equity_values = [r[1] for r in self.returns]
                rolling_max = np.maximum.accumulate(equity_values)
                drawdowns = (np.array(equity_values) - rolling_max) / rolling_max
                max_drawdown = abs(min(drawdowns))
                
                # Calculate returns
                returns = [r[2] for r in self.returns[-30:]]  # Use last 30 returns
                
                # Calmar ratio
                if max_drawdown > 0:
                    reward = np.mean(returns) / max_drawdown
                else:
                    reward = np.mean(returns) * 10  # High reward for no drawdown
            else:
                reward = 0
        
        else:
            # Default to P&L
            reward = (self.portfolio_value - self.previous_portfolio_value) / self.initial_balance
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters:
        -----------
        mode : str
            Rendering mode
            
        Returns:
        --------
        str
            String representation of the environment state
        """
        if mode == 'human':
            step_info = self.data.iloc[self.current_step]
            position_info = f"Position: {self.base_position:.4f} @ {self.position_price:.2f}" if self.base_position != 0 else "No position"
            return (
                f"Step: {self.current_step}\n"
                f"Date: {step_info.name}\n"
                f"Price: {step_info['close']:.2f}\n"
                f"Balance: ${self.quote_position:.2f}\n"
                f"{position_info}\n"
                f"Equity: ${self.portfolio_value:.2f}\n"
            )
        else:
            return

# Create a multi-timeframe trading environment
def create_multi_timeframe_env(
    data_dict: Dict[str, pd.DataFrame],
    primary_timeframe: str = '1h',
    reward_function: str = 'sharpe',
    **kwargs
) -> CryptoTradingEnv:
    """
    Create a trading environment with multiple timeframes.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping timeframe names to DataFrames
    primary_timeframe : str
        Primary timeframe to use for trading
    reward_function : str
        Reward function to use
    **kwargs : dict
        Additional arguments to pass to the environment
        
    Returns:
    --------
    CryptoTradingEnv
        Trading environment instance
    """
    if primary_timeframe not in data_dict:
        raise ValueError(f"Primary timeframe '{primary_timeframe}' not found in data dictionary")
    
    # Get primary data
    primary_data = data_dict[primary_timeframe]
    
    # Create multiframe data dictionary excluding the primary timeframe
    multiframe_data = {tf: df for tf, df in data_dict.items() if tf != primary_timeframe}
    
    # Create environment
    env = CryptoTradingEnv(
        data=primary_data,
        reward_function=reward_function,
        multiframe_data=multiframe_data,
        **kwargs
    )
    
    return env 