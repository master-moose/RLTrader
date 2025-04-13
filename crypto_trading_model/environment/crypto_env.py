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
        
        # Add a safety maximum for portfolio value (increased from 100x to 200x initial amount)
        self.max_portfolio_value = self.initial_amount * 200
        
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
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -10.0, 10.0)
        
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
        observation[0] = min(self.portfolio_value, self.max_portfolio_value)  # Clip for stability
        
        # Add owned assets
        for i in range(min(self.stock_dim, len(self.assets_owned))):
            if i + 1 < self.state_space:
                observation[i + 1] = min(self.assets_owned[i], 1e6)  # Clip for stability
        
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
                # Clip close price to reasonable range
                observation[offset] = np.clip(row['close'], 0, 1e6)
                offset += 1
            
            # Add technical indicators
            for indicator in self.tech_indicator_list:
                if indicator in row and offset < self.state_space:
                    # Apply bounds to technical indicators
                    if np.isfinite(row[indicator]):
                        observation[offset] = np.clip(row[indicator], -1e6, 1e6)
                    else:
                        observation[offset] = 0.0
                    offset += 1
        
        return observation
    
    def _trade(self, action_type):
        """
        Execute a trade based on the action type with enhanced safety measures
        to prevent numerical issues and extreme trades.
        
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
        
        # Safety check for price
        if not np.isfinite(price) or price <= 0:
            logger.warning(f"Invalid price value detected in _trade: {price}, using 1.0")
            price = 1.0
        
        # Store original portfolio state for reference
        original_portfolio_value = self.portfolio_value
        original_assets_owned = self.assets_owned[asset_index]
        
        # Execute the order based on action, with enhanced risk management
        if action_type == 0:  # Sell
            # Only sell if we actually own assets
            if self.assets_owned[asset_index] > 0:
                # Calculate sell amount with position sizing
                # Never sell more than what we have and apply gradual position sizing
                # Sell at most 50% of position to prevent extreme value changes
                max_sell_pct = 0.5
                sell_amount = min(
                    self.assets_owned[asset_index], 
                    self.assets_owned[asset_index] * max_sell_pct
                )
                
                # Apply a minimum sell amount to avoid dust
                min_sell_value = self.initial_amount * 0.001  # 0.1% of initial capital
                if sell_amount * price < min_sell_value and self.assets_owned[asset_index] * price > min_sell_value:
                    sell_amount = min_sell_value / price
                
                # Calculate transaction cost with safety
                sell_cost = min(
                    self.sell_cost_pct[asset_index] * price * sell_amount,
                    original_portfolio_value * 0.01  # Limit cost to 1% of portfolio
                )
                
                # Update portfolio with safety limits
                sale_value = price * sell_amount - sell_cost
                
                # Don't allow sale value to exceed reasonable limits
                if sale_value > original_portfolio_value:
                    logger.warning(f"Sale value ({sale_value:.2f}) exceeds portfolio value ({original_portfolio_value:.2f}), limiting")
                    sale_value = original_portfolio_value * 0.5
                    sell_amount = sale_value / price
                
                # Update portfolio value with rate limiting
                new_portfolio_value = min(self.portfolio_value + sale_value, self.max_portfolio_value * 0.8)
                
                # Log if significant change
                if abs(new_portfolio_value - self.portfolio_value) / max(1, self.portfolio_value) > 0.5:
                    logger.warning(f"Sell operation resulted in large portfolio change: {self.portfolio_value:.2f} -> {new_portfolio_value:.2f}")
                
                self.portfolio_value = new_portfolio_value
                self.cost += sell_cost
                self.trades += 1
                
                # Update assets owned with safety check
                self.assets_owned[asset_index] -= sell_amount
                if self.assets_owned[asset_index] < 0.00001:
                    self.assets_owned[asset_index] = 0  # Clean up dust
                
        elif action_type == 2:  # Buy
            # Calculate available cash with safety margin
            available_amount = max(0, min(self.portfolio_value * 0.9, self.max_portfolio_value * 0.2))
            
            # Safety check: don't allow purchases if portfolio is already near maximum
            if self.portfolio_value >= self.max_portfolio_value * 0.7:
                logger.warning(f"Portfolio value ({self.portfolio_value:.2f}) too close to max, skipping buy")
                return
                
            # Calculate the maximum buy value with VARIABLE percentage (5-15%) instead of fixed 30%
            # This prevents the model from always using exactly 30% of portfolio
            max_buy_pct = min(0.05 + (self.day / 1000) * 0.1, 0.15)  # Scale from 5% to 15% over time
            max_buy_value = min(available_amount, self.portfolio_value * max_buy_pct)
            
            # Additional check to prevent rapid portfolio growth in early training
            if self.portfolio_value > self.initial_amount * 1.5 and self.day < 100:
                # More conservative buying in early stages if portfolio has already grown
                max_buy_value = min(max_buy_value, self.portfolio_value * 0.05)
                
            # Calculate max shares to buy with conservative limits
            max_shares = max_buy_value / (price * (1 + self.buy_cost_pct[asset_index]))
            
            # Apply further conservative limit (max 100 units instead of 500)
            max_shares = min(max_shares, 100)
            
            # Buy shares with minimum purchase size
            min_purchase = self.initial_amount * 0.001  # 0.1% of initial capital
            if max_shares > 0 and max_shares * price >= min_purchase:
                shares_bought = max_shares
                buy_cost = self.buy_cost_pct[asset_index] * price * shares_bought
                
                # Calculate total purchase amount with safety checks
                purchase_total = price * shares_bought + buy_cost
                
                # Ensure purchase doesn't exceed available funds
                if purchase_total > self.portfolio_value:
                    logger.warning(f"Purchase amount ({purchase_total:.2f}) exceeds available funds ({self.portfolio_value:.2f}), scaling down")
                    # Scale down the purchase
                    scale_factor = (self.portfolio_value * 0.9) / purchase_total
                    shares_bought *= scale_factor
                    purchase_total = price * shares_bought + (self.buy_cost_pct[asset_index] * price * shares_bought)
                
                # Update assets owned with safety check
                new_assets = min(self.assets_owned[asset_index] + shares_bought, 1e2)  # 100 unit absolute max (reduced from 1000)
                if new_assets != self.assets_owned[asset_index] + shares_bought:
                    logger.warning(f"Limiting asset position from {self.assets_owned[asset_index] + shares_bought:.2f} to {new_assets:.2f}")
                    shares_bought = new_assets - self.assets_owned[asset_index]
                    purchase_total = price * shares_bought + (self.buy_cost_pct[asset_index] * price * shares_bought)
                
                self.assets_owned[asset_index] = new_assets
                
                # Update portfolio value with rate limiting
                new_portfolio_value = max(0, self.portfolio_value - purchase_total)
                
                # Log if significant change
                if purchase_total / max(1, self.portfolio_value) > 0.1:  # Reduced threshold from 0.3 to 0.1
                    logger.warning(f"Buy operation is using {(purchase_total/self.portfolio_value)*100:.1f}% of portfolio")
                    
                self.portfolio_value = new_portfolio_value
                self.cost += buy_cost
                self.trades += 1
                
        # Final check to ensure we haven't created an impossible state
        total_value = self.portfolio_value
        if self.assets_owned[asset_index] > 0:
            total_value += self.assets_owned[asset_index] * price
            
        # If total value is more than 1.5x initial (reduced from 3x) or asset position is too large, log and apply limit
        if total_value > self.initial_amount * 1.5 or self.assets_owned[asset_index] > 100:  # Reduced from 1000 to 100
            # Only log the warning if the value is significantly above the threshold (2x instead of 1.5x)
            # or if assets are way above the threshold (150 instead of 100)
            if total_value > self.initial_amount * 2.0 or self.assets_owned[asset_index] > 150:
                logger.warning(f"Post-trade check: Portfolio value {total_value:.2f} > 2.0x initial or assets {self.assets_owned[asset_index]:.2f} > 150")
            
            # Silently apply a limit without spamming logs for normal growth
            if total_value > self.initial_amount * 1.5:
                reduction_factor = (self.initial_amount * 1.5) / total_value
                self.portfolio_value *= reduction_factor
                self.assets_owned[asset_index] *= reduction_factor
    
    def _calculate_portfolio_value(self):
        """
        Calculate current portfolio value with enhanced safety measures to prevent 
        numerical instability during training.
        
        Returns:
            float: Current portfolio value
        """
        # Get current price with robust error handling
        price = self._get_current_price()
        if price is None or not np.isfinite(price) or price <= 0:
            logger.warning(f"Invalid price in _calculate_portfolio_value: using default of 1.0")
            price = 1.0
        
        # Original portfolio value for reference
        original_value = self.portfolio_value
        
        # Start with cash (with safety bound)
        cash_value = min(max(0, self.portfolio_value), self.max_portfolio_value * 0.8)  # Increased from 0.6 to 0.8
        if cash_value != self.portfolio_value and abs(cash_value - self.portfolio_value) > 100:
            logger.warning(f"Clipping cash value from {self.portfolio_value} to {cash_value}")
            self.portfolio_value = cash_value
        
        # Calculate asset value with multiple safety measures
        asset_value = 0
        if self.assets_owned[0] > 0:
            # Use a more generous asset limit (500 units max - increased from 100)
            max_asset_units = 5e2
            
            # Apply a stricter safety limit to prevent overflow
            original_assets = self.assets_owned[0]
            clipped_assets = min(original_assets, max_asset_units)
            
            # If assets dramatically exceed the limit, log warning and fix the stored value
            if original_assets > max_asset_units:
                logger.warning(f"Asset position extremely large: {original_assets:.2f}, clipping to {max_asset_units}")
                self.assets_owned[0] = clipped_assets
            
            # Calculate asset value with overflow prevention
            asset_value = clipped_assets * price
            
            # Detect if asset value is unreasonably large compared to initial amount
            asset_value_ratio = asset_value / self.initial_amount
            if asset_value_ratio > 10:  # Increased from 5 to 10
                logger.warning(f"Asset value extremely high: {asset_value_ratio:.2f}x initial amount")
                # Now actually clip it
                max_reasonable_value = self.initial_amount * 10  # Increased from 5 to 10
                if asset_value > max_reasonable_value:
                    asset_value = max_reasonable_value
                    # Update assets owned to match the clipped value
                    self.assets_owned[0] = asset_value / price
                
            # More generous value limit (25% of max instead of 15%)
            max_asset_value = self.max_portfolio_value * 0.25
            if asset_value > max_asset_value:
                logger.warning(f"Asset value too large: {asset_value:.2f}, clipping to {max_asset_value:.2f}")
                
                # Recalculate assets owned based on the max allowed value
                self.assets_owned[0] = max_asset_value / price
                asset_value = max_asset_value
        
        # Calculate total value with stronger safety checks
        total_value = cash_value + asset_value
        
        # Apply less strict maximum - allow up to 20x initial amount (increased from 10x)
        absolute_max = self.initial_amount * 20
        if total_value > absolute_max:
            # Only log warning if significantly exceeding the limit (25x instead of 15x)
            if total_value > self.initial_amount * 25:
                logger.warning(f"Portfolio value exceeded absolute maximum: {total_value:.2f}, clipping to {absolute_max:.2f}")
            
            # Proportionally reduce cash and assets 
            if asset_value > 0:
                ratio = asset_value / total_value
                new_asset_value = absolute_max * ratio
                new_cash_value = absolute_max - new_asset_value
                
                # Recalculate assets owned 
                if price > 0:
                    self.assets_owned[0] = new_asset_value / price
                    
                # Update balance (portfolio value is cash when no assets owned)
                self.portfolio_value = new_cash_value
                
                # Final portfolio value is our max
                total_value = absolute_max
            else:
                # If no assets, just clip cash
                self.portfolio_value = absolute_max
                total_value = absolute_max
        
        # Add rate limiting - allow changes of up to 75% (increased from 50%) in a single step
        # This is crucial for early training stability
        if original_value > 0 and total_value > 0:
            change_pct = abs(total_value - original_value) / original_value
            
            # If change is too large (over 75%), limit it
            if change_pct > 0.75:  # Increased from 0.5 to 0.75
                # If increasing, limit to 75% growth (up from 50%)
                if total_value > original_value:
                    limited_value = original_value * 1.75  # Increased from 1.5 to 1.75
                    # Only log warning if the change is extremely large (over 100%)
                    if change_pct > 1.0:
                        logger.warning(f"Rate limiting portfolio increase: {total_value:.2f} -> {limited_value:.2f}")
                    
                    # Adjust both the portfolio value and asset balance
                    if asset_value > 0:
                        # Proportionally adjust
                        reduction_ratio = limited_value / total_value
                        # Apply to assets
                        self.assets_owned[0] *= reduction_ratio
                        # Reset portfolio value (cash component)
                        self.portfolio_value = limited_value - (asset_value * reduction_ratio)
                    else:
                        # Just cash, simpler adjustment
                        self.portfolio_value = limited_value
                    
                    total_value = limited_value
                    
                # If decreasing, limit to 25% loss (instead of 30%)
                elif total_value < original_value:
                    limited_value = original_value * 0.75  # Changed from 0.7 to 0.75
                    # Only log warning if the change is extremely large (over 80%)
                    if change_pct > 0.8:
                        logger.warning(f"Rate limiting portfolio decrease: {total_value:.2f} -> {limited_value:.2f}")
                    
                    # Similar adjustment approach for decreases
                    if asset_value > 0:
                        scale_up_ratio = limited_value / total_value
                        self.assets_owned[0] *= scale_up_ratio
                        self.portfolio_value = limited_value - (asset_value * scale_up_ratio)
                    else:
                        self.portfolio_value = limited_value
                        
                    total_value = limited_value
        
        # Add day-based growth limiting for early training (first 300 steps - reduced from 500)
        if self.day < 300 and total_value > self.initial_amount:
            # Linear growth cap: allow max 50% increase over initial amount for first 300 steps (up from 30%)
            max_allowed = self.initial_amount * (1.0 + 0.5 * (self.day / 300))
            if total_value > max_allowed:
                # Only log warning if significantly exceeding the growth cap (more than 3x over)
                if total_value > max_allowed * 3.0:
                    logger.warning(f"Early training growth limiting: {total_value:.2f} -> {max_allowed:.2f} at day {self.day}")
                # Scale everything down proportionally
                scale_factor = max_allowed / total_value
                if asset_value > 0:
                    self.assets_owned[0] *= scale_factor
                self.portfolio_value = max_allowed - (asset_value * scale_factor)
                total_value = max_allowed
        
        # Final sanity check for numerical stability
        if not np.isfinite(total_value) or total_value <= 0:
            logger.warning(f"Invalid portfolio value detected: {total_value:.2f}, resetting to initial amount")
            
            # Complete reset of portfolio to recover from numerical error
            total_value = self.initial_amount
            self.portfolio_value = self.initial_amount
            self.assets_owned[0] = 0
        
        # Track if we made a significant change to the value
        if abs(total_value - original_value) / max(1, original_value) > 1.0 and original_value > 0:  # Increased from 0.8 to 1.0
            logger.warning(f"Portfolio value changed dramatically: {original_value:.2f} -> {total_value:.2f}")
        
        return float(total_value)
    
    def _get_current_price(self):
        """
        Get the current price of the asset.
        
        Returns:
            float: Current price
        """
        if self.day >= len(self.df.index.unique()):
            return 1.0  # Default price if we're past the end of data
            
        data_index = min(self.day, len(self.df.index.unique()) - 1)
        current_date = self.df.index.unique()[data_index]
        current_data = self.df[self.df.index == current_date]
        
        if current_data.empty:
            return 1.0
            
        price = current_data.iloc[0]['close']
        
        # Safety check
        if not np.isfinite(price) or price <= 0:
            return 1.0
            
        return price
    
    def _get_info(self):
        """
        Get additional information about the current state.
        
        Returns:
            dict: Information about the current state
        """
        # Include current price in info
        current_price = self._get_current_price()
        
        return {
            'portfolio_value': self.portfolio_value,
            'assets_owned': self.assets_owned,
            'cost': self.cost,
            'trades': self.trades,
            'day': self.day,
            'close_price': current_price
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