"""
Cryptocurrency trading environment using Gym.

Major fixes in this version:
1. Prevented rewarding hold actions for price appreciation - now uses fixed penalties
2. Improved forced sell mechanism with better logging
3. Added detailed tracking of holding periods
4. Significantly increased hold penalties to combat perpetual holding issue
"""

# Don't import the actual module which has dependencies we might not have
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import List
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
        max_holding_steps: int = 8,
        episode_length: int = None,
        randomize_start: bool = True,
        candles_per_day: int = 1,
        take_profit_pct: float = 0.03,  # Default 3% take profit
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
            max_holding_steps: Maximum number of steps to hold a position before forced selling
            episode_length: Length of each episode in days (if None, uses entire dataset)
            randomize_start: Whether to randomize the start date for each episode
            candles_per_day: Number of candles representing one day (1 for daily, 96 for 15-min)
            take_profit_pct: Take profit percentage
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
        self.randomize_start = randomize_start
        self.candles_per_day = max(1, candles_per_day)  # Ensure at least 1 candle per day
        
        # Calculate actual days in the dataset
        total_candles = len(df.index.unique())
        self.total_days = total_candles // self.candles_per_day
        logger.info(f"Dataset has {total_candles} candles, representing {self.total_days} days with {self.candles_per_day} candles per day")
        
        # Default episode length to 365 days (1 year) if not specified and if enough data
        if episode_length is None:
            # Use 365 days or the full dataset, whichever is smaller
            self.episode_length = min(365, self.total_days)
        else:
            self.episode_length = min(episode_length, self.total_days)
            
        # Make sure episode length is at least 30 days
        self.episode_length = max(30, self.episode_length)
        
        # Calculate episode length in candles
        self.episode_length_candles = self.episode_length * self.candles_per_day
        
        # Calculate maximum start day to ensure full episodes
        self.max_start_day = max(0, total_candles - self.episode_length_candles)
        self.start_day = 0
        
        logger.info(f"Episode length set to {self.episode_length} days ({self.episode_length_candles} candles) out of {self.total_days} days ({total_candles} candles)")
        
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
        
        # Position holding tracking
        self.holding_counter = 0
        self.max_holding_steps = max_holding_steps
        self.forced_sells = 0  # Track forced sells for analytics
        
        # Store info for rendering
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        
        # Add a safety maximum for portfolio value (increased from 100x to 200x initial amount)
        self.max_portfolio_value = self.initial_amount * 200
        
        # Take profit tracking
        self.take_profit_pct = take_profit_pct
        self.entry_price = None
        self.take_profit_price = None
        self.take_profit_sells = 0
        
        logger.info(f"Created CryptocurrencyTradingEnv with {self.stock_dim} assets")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment state at the start of a new episode.
        """
        # Reset internal state
        if seed is not None:
            np.random.seed(seed)
        
        self.terminal = False
        self.day = 0
        self.assets_owned = np.zeros(self.stock_dim)
        self.costs = []
        self.trades = 0
        self.total_trades = 0
        self.holding_counter = 0
        self.forced_sells = 0  # Reset forced sells counter
        
        # Randomize start day if configured
        if self.randomize_start and self.max_start_day > 0:
            self.start_day = np.random.randint(0, self.max_start_day)
            actual_day = self.start_day // self.candles_per_day
            logger.info(f"Starting new episode from candle {self.start_day} (day {actual_day}) out of {len(self.df.index.unique())} candles")
        else:
            self.start_day = 0
        
        self.day = self.start_day
        
        # Get the first day of data
        if self.day >= len(self.df.index.unique()):
            self.day = 0  # Reset if we're past the end of data
        
        # Reset portfolio value to initial amount
        self.portfolio_value = self.initial_amount
        self.cost = 0
        
        # Validate portfolio integrity
        self._validate_portfolio_integrity()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _validate_portfolio_integrity(self):
        """
        Validate that portfolio values are reasonable and consistent.
        This helps prevent artificial manipulation of values.
        """
        # Ensure portfolio value is positive and not too high
        if not np.isfinite(self.portfolio_value) or self.portfolio_value <= 0:
            logger.warning("Reset detected invalid portfolio value, resetting to initial amount")
            self.portfolio_value = self.initial_amount
        
        # Ensure assets owned is reasonable
        for i, asset in enumerate(self.assets_owned):
            if not np.isfinite(asset) or asset < 0:
                logger.warning(f"Reset detected invalid assets owned [{i}]: {asset}, resetting to 0")
                self.assets_owned[i] = 0.0
            
        # Ensure we're not starting with an unreasonable portfolio value
        if self.portfolio_value > self.initial_amount * 3:
            logger.warning(f"Reset detected unusually high portfolio value: {self.portfolio_value}, capping at 3x initial")
            self.portfolio_value = self.initial_amount * 3
        
        # Validate that portfolio calculation is consistent
        calculated_value = self._calculate_portfolio_value()
        difference = abs(calculated_value - self.portfolio_value)
        if difference > self.initial_amount * 0.01:  # More than 1% difference
            logger.warning(f"Portfolio value inconsistency detected: {self.portfolio_value} vs {calculated_value}, correcting")
            self.portfolio_value = calculated_value
    
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
        
        # Check if episode is done (reached end of episode length or end of data)
        episode_end = self.day >= (self.start_day + self.episode_length_candles)
        data_end = self.day >= len(self.df.index.unique())
        
        if episode_end or data_end:
            # End of episode
            terminated = True
            truncated = False
            self.state = self._get_observation()
            
            # Calculate final reward with potential bonus for ending with cash instead of assets
            reward = 0
            
            # Add bonus for ending episode with no assets (encourages selling before end)
            if self.assets_owned[0] == 0:
                cash_bonus = 1.0  # Bonus for ending with cash
                reward += cash_bonus
                logger.info(f"Episode ended with no assets - applying cash bonus: +{cash_bonus:.2f}")
            else:
                # Penalize ending with assets
                asset_penalty = 0.5  # Penalty for not selling before end
                reward -= asset_penalty
                actual_day = (self.day // self.candles_per_day)
                logger.warning(f"Episode ended on candle {self.day} (day {actual_day}) with {self.assets_owned[0]:.2f} assets - applying penalty: -{asset_penalty:.2f}")
            
            info = self._get_info()
            info['end_reason'] = 'episode_length' if episode_end else 'data_end'
            return self.state, reward, terminated, truncated, info
        
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
        
        # Get current price for take profit check
        current_date = self.df.index.unique()[self.day]
        current_data = self.df[self.df.index == current_date]
        current_price = current_data.iloc[0]['close']
        
        # Check for take profit condition if we have assets and an entry price
        original_action_type = action_type
        prev_assets = self.assets_owned[0]
        take_profit_triggered = False
        
        if self.assets_owned[0] > 0 and self.entry_price is not None and self.take_profit_price is not None:
            # Check if the current price exceeds our take profit target
            if current_price >= self.take_profit_price:
                # Force a sell action
                action_type = 0  # Sell
                take_profit_triggered = True
                self.take_profit_sells += 1
                
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                logger.warning(
                    f"TAKE PROFIT TRIGGERED: Original action {original_action_type}, "
                    f"Current price {current_price:.4f} >= take profit price {self.take_profit_price:.4f} "
                    f"(profit: {profit_pct:.2f}%)"
                )
        
        # Force a sell action if we've been holding for too long and have assets
        force_time_sell = False
        if self.assets_owned[0] > 0:
            self.holding_counter += 1
            # Log holding counter more frequently to track progress toward forced sell
            if self.holding_counter >= 4 or self.holding_counter % 2 == 0:  # Start logging from 4+ steps
                logger.info(f"Holding assets for {self.holding_counter}/{self.max_holding_steps} steps (forced sell threshold)")
            
            # Force a sell if holding counter exceeds threshold AND take profit wasn't already triggered
            if self.holding_counter >= self.max_holding_steps and not take_profit_triggered:
                # Force sell with explicit logging that we're overriding the action
                force_time_sell = True
                logger.warning(
                    f"FORCING SELL: Original action {original_action_type}, "
                    f"but holding for {self.holding_counter} steps (max: {self.max_holding_steps})"
                )
                action_type = 0  # Force sell
                self.forced_sells += 1
                # Force a sell of ALL assets (not partial) when we hit the max holding period
                logger.warning(
                    f"Forcing FULL sell at step {self.day} after "
                    f"holding for {self.holding_counter} steps"
                )
        else:
            # Reset counter if we don't have assets
            self.holding_counter = 0
            self.entry_price = None
            self.take_profit_price = None
        
        # Execute the trade
        self._trade(action_type)
        
        # Update entry price and take profit targets after a buy action
        if action_type == 2 and self.assets_owned[0] > prev_assets:
            # Set entry price and take profit targets
            self.entry_price = current_price
            self.take_profit_price = current_price * (1 + self.take_profit_pct)
            logger.info(f"Set entry price to {self.entry_price:.4f} and take profit target to {self.take_profit_price:.4f} (target gain: {self.take_profit_pct*100:.1f}%)")
            
            # Start counter after buying
            self.holding_counter = 1
            logger.info(f"Bought assets at step {self.day}, setting holding counter to 1")
        elif action_type == 0 and self.assets_owned[0] < prev_assets:
            # Reset counter and targets after selling
            self.holding_counter = 0
            self.entry_price = None
            self.take_profit_price = None
            logger.info(f"Sold assets at step {self.day}, resetting holding counter to 0 and clearing price targets")
        
        # Calculate reward as change in portfolio value
        self.portfolio_value = self._calculate_portfolio_value()
        
        # For hold actions, we need to ensure no portfolio growth from price appreciation
        # This is key to prevent the agent from getting rewarded for holding during price increases
        if action_type == 1 and self.assets_owned[0] > 0:
            # If holding assets, don't reward for price appreciation
            # Calculate what the portfolio value would be if the price hadn't changed
            # The agent should only be rewarded/penalized for explicit trades, not market movements during holds
            hold_penalty = 1.0  # Significantly increased from 0.2 to 1.0 - much higher penalty for any hold action
            reward = -hold_penalty  # Apply hold penalty directly instead of based on portfolio change
            logger.debug(f"HOLD action: Not rewarding for price appreciation, applying hold penalty of {hold_penalty:.4f}")
        else:
            # For buys and sells, reward normally based on portfolio change
            reward = (self.portfolio_value - previous_portfolio_value) * self.reward_scaling
        
        # Add extra reward for sells (action_type 0) to encourage selling
        # Give even higher reward for sells
        if action_type == 0 and self.assets_owned[0] >= 0:
            # Apply a multiplier to rewards from sell actions
            sell_reward_multiplier = 6.0  # Increased from 4.0 to 6.0 for even stronger sell incentive
            reward = reward * sell_reward_multiplier
            
            # Add a fixed bonus reward for selling regardless of profit/loss
            # This helps encourage the agent to take more sell actions
            sell_bonus = 6.0  # Doubled from 3.0 to 6.0 - much higher fixed bonus for any sell action 
            reward += sell_bonus
            
            # Provide HIGHER rewards for take profit selling
            if take_profit_triggered:
                take_profit_bonus = 10.0  # High bonus for take profit sells
                reward += take_profit_bonus
                logger.info(f"Applied take profit bonus: +{take_profit_bonus:.2f} at step {self.day}")
            # Provide HIGHER rewards for proactive selling (before being forced)
            elif original_action_type == action_type:  # This was a voluntary sell
                proactive_bonus = 8.0  # Doubled from 4.0 to 8.0 for even stronger proactive sell incentive
                reward += proactive_bonus
                logger.info(f"Applied proactive sell bonus: +{proactive_bonus:.2f} at step {self.day}")
            # Log when forced sells happen
            elif force_time_sell:
                logger.warning(f"Forced time-based sell at step {self.day} resulted in reward: {reward:.4f}")
        
        # Add reward for buy actions to encourage more trading
        elif action_type == 2:
            # Add buy bonus to encourage more buying decisions
            buy_bonus = 4.0  # Increased from 1.5 to 4.0 - much higher bonus for buy actions
            reward += buy_bonus
            logger.info(f"Applied buy bonus: +{buy_bonus:.2f} at step {self.day}")
        
        # Penalize holding to discourage excessive holding
        elif action_type == 1:
            # Reduce reward for holding (significant penalty)
            hold_penalty = 1.0  # Increased from 0.2 to 1.0 - much higher penalty for any hold action
            reward -= hold_penalty
            
            # Add a small penalty for long holds to discourage excessive holding
            if self.holding_counter > 3:  # Penalize holds even earlier
                # Exponential penalty growth to strongly discourage long holds
                hold_steps_over_limit = self.holding_counter - 3
                hold_penalty = min(0.5 * (hold_steps_over_limit ** 2.0), 5.0)  # Much stronger exponential growth with higher cap
                reward -= hold_penalty
                
                # Log significant hold penalties
                if hold_penalty > 0.1 or self.holding_counter > 5:  # Lower threshold to log more penalties
                    logger.warning(f"Applied hold penalty of {hold_penalty:.4f} after {self.holding_counter} steps of holding")
        
        # Clip reward to prevent extreme values, but with wider limits
        reward = np.clip(reward, -20.0, 20.0)  # Keep the existing limits
        
        # Update memory
        self.rewards_memory.append(reward)
        self.asset_memory.append(self.portfolio_value)
        
        # Get new state and check for termination
        self.state = self._get_observation()
        terminated = self.day >= len(self.df.index.unique()) - 1
        truncated = False
        
        # Add info about holding counter and forced sells
        info = self._get_info()
        info['holding_counter'] = self.holding_counter
        info['forced_sells'] = self.forced_sells
        info['original_action'] = original_action_type
        info['executed_action'] = action_type
        
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
        
        # Add holding counter information to the observation - normalized between 0 and 1
        # This helps the agent learn about holding duration
        if self.stock_dim + 2 < self.state_space:
            # Make holding counter more prominent in observation
            holding_counter_normalized = min(self.holding_counter / self.max_holding_steps, 1.0)
            observation[self.stock_dim + 2] = holding_counter_normalized * 2.0  # Multiply by 2 to make it more prominent
            
            # Add a warning signal as holding approaches max limit - strengthen the signal
            # Start warning signal earlier (at 50% of max instead of 80%)
            if self.holding_counter > 0.5 * self.max_holding_steps:
                # Create a stronger signal as we get closer to forced sell
                sell_urgency = (self.holding_counter - 0.5 * self.max_holding_steps) / (0.5 * self.max_holding_steps)
                # Amplify the urgency signal
                observation[self.stock_dim + 3] = sell_urgency * 2.0  # Multiply by 2 for stronger signal
            else:
                observation[self.stock_dim + 3] = 0.0
        
        # Add market data
        data_index = min(self.day, len(self.df.index.unique()) - 1)
        current_date = self.df.index.unique()[data_index]
        current_data = self.df[self.df.index == current_date]
        
        # Add price data
        offset = self.stock_dim + 4  # Start after portfolio value, assets owned, and holding counter
        
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
                # Check if this is a forced sell action (from max holding time)
                if self.holding_counter >= self.max_holding_steps:
                    # If it's a forced sell, sell the entire position
                    max_sell_pct = 1.0  # Sell 100%
                else:
                    # Normal sell - use the standard position sizing
                    # Increase from 0.9 to make normal sells more impactful
                    max_sell_pct = 1.0  # Always sell 100% to make sell actions more decisive
                    
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
                
                # Update portfolio value with rate limiting - but with higher limits
                new_portfolio_value = min(self.portfolio_value + sale_value, self.max_portfolio_value * 0.9)  # Increased from 0.8 to 0.9
                
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
                
            # Calculate the maximum buy value with VARIABLE percentage (10-30%) instead of (5-15%)
            # This allows more aggressive buying
            max_buy_pct = min(0.1 + (self.day / 1000) * 0.2, 0.3)  # Scale from 10% to 30% over time (increased from 5-15%)
            max_buy_value = min(available_amount, self.portfolio_value * max_buy_pct)
            
            # Additional check to prevent rapid portfolio growth in early training
            if self.portfolio_value > self.initial_amount * 2.0 and self.day < 100:  # Increased from 1.5x to 2.0x
                # Still be conservative but allow more buying
                max_buy_value = min(max_buy_value, self.portfolio_value * 0.1)  # Increased from 0.05 to 0.1
                
            # Calculate max shares to buy with conservative limits
            max_shares = max_buy_value / (price * (1 + self.buy_cost_pct[asset_index]))
            
            # Apply further conservative limit (allow up to 300 units instead of 100)
            max_shares = min(max_shares, 300)  # Increased from 100 to 300
            
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
            
        # REMOVED: The hard cap of 2.0x initial that was causing the exact doubling issue
        # Instead, use a much higher cap with proper logging
        if total_value > self.initial_amount * 5.0 or self.assets_owned[asset_index] > 400:
            # Log when portfolio value is extremely high
            if total_value > self.initial_amount * 5.0:
                logger.warning(f"Post-trade check: Portfolio value {total_value:.2f} > 5.0x initial")
            
            # Only apply reduction for truly extreme values
            if total_value > self.initial_amount * 10.0:
                reduction_factor = (self.initial_amount * 10.0) / total_value
                self.portfolio_value *= reduction_factor
                self.assets_owned[asset_index] *= reduction_factor
                logger.warning(f"Extreme portfolio value detected, scaling down: {total_value:.2f} -> {self.portfolio_value + (self.assets_owned[asset_index] * price):.2f}")
    
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
            logger.warning("Invalid price in _calculate_portfolio_value: using default of 1.0")
            price = 1.0
        
        # Original portfolio value for reference
        original_value = self.portfolio_value
        
        # Start with cash (with safety bound)
        cash_value = min(max(0, self.portfolio_value), self.max_portfolio_value * 0.9)
        if cash_value != self.portfolio_value and abs(cash_value - self.portfolio_value) > 100:
            logger.warning(f"Clipping cash value from {self.portfolio_value} to {cash_value}")
            self.portfolio_value = cash_value
        
        # Calculate asset value with multiple safety measures
        asset_value = 0
        if self.assets_owned[0] > 0:
            # Use a reasonable asset limit to prevent overflow
            max_asset_units = 1e3
            
            # Apply safety limit to prevent overflow
            original_assets = self.assets_owned[0]
            clipped_assets = min(original_assets, max_asset_units)
            
            # If assets dramatically exceed the limit, log warning and fix the stored value
            if original_assets > max_asset_units:
                logger.warning(f"Asset position extremely large: {original_assets:.2f}, clipping to {max_asset_units}")
                self.assets_owned[0] = clipped_assets
            
            # Calculate asset value with overflow prevention
            asset_value = clipped_assets * price
            
            # Apply reasonable limits to asset value
            if asset_value > self.initial_amount * 15:
                logger.warning(f"Asset value extremely high: {asset_value:.2f}, clipping to {self.initial_amount * 15:.2f}")
                asset_value = self.initial_amount * 15
                self.assets_owned[0] = asset_value / price
        
        # Calculate total value
        total_value = cash_value + asset_value
        
        # Apply reasonable maximum - but don't artificially manipulate values
        absolute_max = self.initial_amount * 25  # Reduced from 50 to 25
        if total_value > absolute_max:
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
        
        # Final sanity check for numerical stability
        if not np.isfinite(total_value) or total_value <= 0:
            logger.warning(f"Invalid portfolio value detected: {total_value:.2f}, resetting to initial amount")
            
            # Complete reset of portfolio to recover from numerical error
            total_value = self.initial_amount
            self.portfolio_value = self.initial_amount
            self.assets_owned[0] = 0
        
        # Track significant changes to portfolio value
        if abs(total_value - original_value) / max(1, original_value) > 0.3 and original_value > 0:
            logger.warning(f"Portfolio value changed significantly: {original_value:.2f} -> {total_value:.2f}")
        
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
        actual_day = self.day // self.candles_per_day
        
        return {
            'portfolio_value': self.portfolio_value,
            'assets_owned': self.assets_owned,
            'cost': self.cost,
            'trades': self.trades,
            'day': self.day,
            'actual_day': actual_day,
            'close_price': current_price,
            'holding_counter': self.holding_counter,
            'forced_sells': self.forced_sells,
            'take_profit_sells': self.take_profit_sells,
            'take_profit_price': self.take_profit_price,
            'entry_price': self.entry_price
        }
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters:
            mode: Rendering mode
        """
        if self.print_verbosity > 0 and self.day % self.print_verbosity == 0:
            actual_day = self.day // self.candles_per_day
            print(f"Candle: {self.day}, Day: {actual_day}, Portfolio Value: {self.portfolio_value}")

# Make the class available outside
StockTradingEnv = CryptocurrencyTradingEnv 