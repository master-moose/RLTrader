"""
Cryptocurrency trading environment using Gym.
"""

# Don't import the actual module which has dependencies we might not have
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CryptocurrencyTradingEnv(gym.Env):
    """
    A simplified cryptocurrency trading environment based on Gym.
    This is a replacement for the FinRL StockTradingEnv to avoid dependency issues.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_amount: float = 100000.0,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        state_space: int = 16,
        stock_dim: int = 1,
        tech_indicator_list: List[str] = None,
        action_space: int = 3,
        reward_scaling: float = 1e-4,
        print_verbosity: int = 0,
        **kwargs
    ):
        """
        Initialize the crypto trading environment.
        
        Parameters:
            df: DataFrame with market data
            initial_amount: Initial portfolio value
            buy_cost_pct: Cost percentage for buying
            sell_cost_pct: Cost percentage for selling
            state_space: Dimension of the state space
            stock_dim: Number of stocks/cryptocurrencies
            tech_indicator_list: List of technical indicators to include in state
            action_space: Size of the action space
            reward_scaling: Scaling factor for rewards
            print_verbosity: Frequency of printing info during execution
        """
        self.df = df
        self.day = 0
        self.initial_amount = initial_amount
        self.state_space = state_space
        self.stock_dim = stock_dim
        self.tech_indicator_list = tech_indicator_list or []
        self.action_space_size = action_space
        self.reward_scaling = reward_scaling
        self.print_verbosity = print_verbosity
        
        # Handle lists or floats for transaction costs
        if isinstance(buy_cost_pct, list):
            self.buy_cost_pct = buy_cost_pct
        else:
            self.buy_cost_pct = [buy_cost_pct] * stock_dim
            
        if isinstance(sell_cost_pct, list):
            self.sell_cost_pct = sell_cost_pct
        else:
            self.sell_cost_pct = [sell_cost_pct] * stock_dim
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(action_space)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_space,), dtype=np.float32
        )
        
        # Portfolio state
        self.state = np.zeros(self.state_space)
        self.portfolio_value = self.initial_amount
        self.assets_owned = [0] * self.stock_dim
        self.cost = 0
        self.trades = 0
        self.episode_reward = 0
        
        # Store info for rendering
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        
        logger.info(f"Created CryptocurrencyTradingEnv with {self.stock_dim} assets")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.day = 0
        self.portfolio_value = self.initial_amount
        self.assets_owned = [0] * self.stock_dim
        self.cost = 0
        self.trades = 0
        self.episode_reward = 0
        
        # Reset memory
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        
        # Create observation
        self.state = self._get_observation()
        
        return self.state, {}
    
    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Parameters:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.day += 1
        previous_portfolio_value = self.portfolio_value
        
        # Get data for current day
        if self.day >= len(self.df.index.unique()):
            # End of data
            terminated = True
            truncated = False
            self.state = self._get_observation()
            reward = 0
            info = self._get_info()
            return self.state, reward, terminated, truncated, info
        
        # Convert action to a portfolio decision (e.g., 0: sell, 1: hold, 2: buy)
        if self.action_space_size == 3:
            # Simple case: sell, hold, buy
            action_type = action 
        else:
            # If action space is different, interpret based on number of actions
            action_type = action % 3  # Simplified to 3 basic actions
        
        # Execute the trade
        self._trade(action_type)
        
        # Calculate reward as change in portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        reward = (self.portfolio_value - previous_portfolio_value) * self.reward_scaling
        
        # Update memory
        self.rewards_memory.append(reward)
        self.asset_memory.append(self.portfolio_value)
        
        # Get new state and check for termination
        self.state = self._get_observation()
        terminated = self.day >= len(self.df.index.unique()) - 1
        truncated = False
        info = self._get_info()
        
        return self.state, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get current observation state.
        
        Returns:
            numpy array of state values
        """
        # Get current price data
        if self.day >= len(self.df.index.unique()):
            # End of data, return final state
            return self.state
        
        observation = np.zeros(self.state_space)
        
        # Basic portfolio state - first few dimensions
        observation[0] = self.portfolio_value
        
        # Add owned assets
        for i in range(min(self.stock_dim, len(self.assets_owned))):
            if i + 1 < self.state_space:
                observation[i + 1] = self.assets_owned[i]
        
        # Add market data
        data_index = min(self.day, len(self.df.index.unique()) - 1)
        current_date = self.df.index.unique()[data_index]
        current_data = self.df[self.df.index == current_date]
        
        # Add price data
        offset = self.stock_dim + 1  # Start after portfolio value and assets owned
        
        # Add prices and technical indicators
        if not current_data.empty:
            row = current_data.iloc[0]
            if 'close' in row and offset < self.state_space:
                observation[offset] = row['close']
                offset += 1
            
            # Add technical indicators
            for indicator in self.tech_indicator_list:
                if indicator in row and offset < self.state_space:
                    observation[offset] = row[indicator]
                    offset += 1
        
        return observation
    
    def _trade(self, action_type):
        """
        Execute a trade based on the action type.
        
        Parameters:
            action_type: Type of action (0: sell, 1: hold, 2: buy)
        """
        # Get current price
        data_index = min(self.day, len(self.df.index.unique()) - 1)
        current_date = self.df.index.unique()[data_index]
        current_data = self.df[self.df.index == current_date]
        
        if current_data.empty:
            return
        
        # Assume we're working with the first asset only for simplicity
        asset_index = 0
        price = current_data.iloc[0]['close']
        
        # Execute the order based on action
        if action_type == 0:  # Sell
            if self.assets_owned[asset_index] > 0:
                # Calculate transaction cost
                sell_cost = self.sell_cost_pct[asset_index] * price * self.assets_owned[asset_index]
                # Update portfolio
                self.portfolio_value += price * self.assets_owned[asset_index] - sell_cost
                self.cost += sell_cost
                self.trades += 1
                self.assets_owned[asset_index] = 0
                
        elif action_type == 2:  # Buy
            # Calculate available cash
            available_amount = self.portfolio_value
            # Calculate max shares to buy with available cash
            max_shares = available_amount // (price * (1 + self.buy_cost_pct[asset_index]))
            
            # Buy shares
            if max_shares > 0:
                shares_bought = max_shares
                buy_cost = self.buy_cost_pct[asset_index] * price * shares_bought
                self.assets_owned[asset_index] += shares_bought
                self.portfolio_value -= price * shares_bought + buy_cost
                self.cost += buy_cost
                self.trades += 1
    
    def _calculate_portfolio_value(self):
        """
        Calculate current portfolio value.
        
        Returns:
            float: Current portfolio value
        """
        # Start with cash
        value = self.portfolio_value
        
        # Add value of assets
        data_index = min(self.day, len(self.df.index.unique()) - 1)
        current_date = self.df.index.unique()[data_index]
        current_data = self.df[self.df.index == current_date]
        
        if not current_data.empty:
            price = current_data.iloc[0]['close']
            value = self.portfolio_value + self.assets_owned[0] * price
        
        return value
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            dict: Information about the current state
        """
        return {
            'portfolio_value': self.portfolio_value,
            'assets_owned': self.assets_owned,
            'cost': self.cost,
            'trades': self.trades,
            'day': self.day
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters:
            mode: Rendering mode
        """
        if self.print_verbosity > 0 and self.day % self.print_verbosity == 0:
            print(f"Day: {self.day}, Portfolio Value: {self.portfolio_value}")

# Make the class available outside
StockTradingEnv = CryptocurrencyTradingEnv 