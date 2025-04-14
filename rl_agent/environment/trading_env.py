"""
Trading Environment for reinforcement learning.

This module implements a cryptocurrency trading environment
compatible with the OpenAI Gym interface.
"""

from gymnasium import Env, spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import matplotlib.pyplot as plt
import logging

# Get module logger
logger = logging.getLogger("rl_agent.environment")


class TradingEnvironment(Env):
    """
    A cryptocurrency trading environment for reinforcement learning.
    
    This environment simulates trading a single cryptocurrency
    with features like transaction fees, portfolio tracking,
    and customizable observation space.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        sequence_length: int = 60,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        reward_scaling: float = 1.0,
        window_size: int = 20,
        max_position: float = 1.0,
        max_steps: Optional[int] = None,
        random_start: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            features: List of column names to use as features
            sequence_length: Length of price history sequences
            initial_balance: Initial cash balance
            transaction_fee: Fee per transaction as a fraction
            reward_scaling: Scaling factor for the reward
            window_size: Size of the trading time window for rendering
            max_position: Maximum position size as a fraction of balance
            max_steps: Maximum number of steps in an episode
            random_start: Start from a random position in the data
            render_mode: Gymnasium render mode ('human', 'rgb_array', None)
        """
        super(TradingEnvironment, self).__init__()
        
        # Store parameters
        self.data = data.copy()
        self.features = features or ["close", "volume", "open", "high", "low"]
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee  # Store the original target fee
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.max_position = max_position
        self.sequence_length = sequence_length
        self.random_start = random_start
        self.render_mode = render_mode
        
        # Data preprocessing
        for feature in self.features:
            if feature not in self.data.columns:
                raise ValueError(f"Feature '{feature}' not found in data columns: "
                               f"{data.columns.tolist()}")
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        
        # Observation space: features + balance + position
        feature_space = len(self.features) * self.sequence_length  # Historical features
        account_space = 2  # Cash balance and asset position
        total_space = feature_space + account_space
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_space,),
            dtype=np.float32
        )
        
        # Set maximum number of steps
        if max_steps is None:
            self.max_steps = len(self.data) - self.sequence_length - 1
        else:
            self.max_steps = min(max_steps, 
                                 len(self.data) - self.sequence_length - 1)
        
        logger.info(f"Created TradingEnvironment with {len(self.data)} data points, "
                   f"{len(self.features)} features, and sequence length "
                   f"{self.sequence_length}")
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial observation and info dictionary
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the position in the data
        if self.random_start:
            self.current_step = np.random.randint(
                self.sequence_length, len(self.data) - self.max_steps
            )
            logger.debug(f"Starting from random position: {self.current_step}")
        else:
            self.current_step = self.sequence_length
            logger.debug(f"Starting from beginning: {self.current_step}")
        
        # Reset the account balance
        self.balance = self.initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.portfolio_value = self.balance + self.asset_value
        self.max_portfolio_value = self.portfolio_value
        
        # Reset trade history
        self.trades = []
        self.buy_prices = []
        self.sell_prices = []
        self.portfolio_values = [self.portfolio_value]
        self.rewards = []
        self.last_buy_price = None
        
        # Reset metrics
        self.total_trades = 0
        self.total_buys = 0
        self.total_sells = 0
        self.total_holds = 0
        self.max_drawdown = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: Sell, 1: Hold, 2: Buy)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Record previous portfolio value
        prev_portfolio_value = self.portfolio_value
        
        # Execute the action
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if the episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, 
                                       self.portfolio_value)
        
        # Calculate drawdown
        drawdown = 0
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) \
                       / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Store portfolio value history
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_portfolio_value)
        self.rewards.append(reward)
        
        # Get new observation
        observation = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        # Gymnasium expects 5 return values: obs, reward, terminated, truncated, info
        terminated = done
        return observation, reward, terminated, truncated, info
    
    def _take_action(self, action):
        """
        Execute the trading action.
        
        Args:
            action: Action to take (0: Sell, 1: Hold, 2: Buy)
        """
        current_price = self.data['close'].iloc[self.current_step]
        
        # Record action counts
        self.total_trades += (action != 1)  # Count non-hold actions as trades
        if action == 0:
            self.total_sells += 1
        elif action == 1:
            self.total_holds += 1
        elif action == 2:
            self.total_buys += 1
        
        if action == 0:  # Sell
            if self.shares_held > 0:
                # Calculate transaction fee
                sell_amount = self.shares_held * current_price
                fee = sell_amount * self.transaction_fee # Use fixed fee
                
                # Update balance and shares
                self.balance += sell_amount - fee
                
                # Log sell
                sell_info = {
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'shares': self.shares_held,
                    'amount': sell_amount,
                    'fee': fee,
                    'balance_after': self.balance
                }
                self.trades.append(sell_info)
                self.sell_prices.append(current_price)
                
                logger.debug(f"Sold {self.shares_held:.6f} shares at "
                           f"{current_price:.2f} for {sell_amount:.2f} "
                           f"(fee: {fee:.2f})")
                
                self.shares_held = 0
                self.asset_value = 0
        
        elif action == 2:  # Buy
            if self.balance > 0:
                # Calculate max shares we can buy
                max_buyable_shares = self.balance * self.max_position / \
                    (current_price * (1 + self.transaction_fee)) # Use fixed fee
                
                # Buy shares (using max_position of balance)
                self.shares_held = max_buyable_shares
                buy_amount = self.shares_held * current_price
                fee = buy_amount * self.transaction_fee  # Use fixed fee
                
                # Update balance and asset value
                self.balance -= (buy_amount + fee)
                self.asset_value = self.shares_held * current_price
                
                # Record buy price
                self.last_buy_price = current_price
                self.buy_prices.append(current_price)
                
                # Log buy
                buy_info = {
                    'step': self.current_step,
                    'type': 'buy',
                    'price': current_price,
                    'shares': self.shares_held,
                    'amount': buy_amount,
                    'fee': fee,
                    'balance_after': self.balance
                }
                self.trades.append(buy_info)
                
                logger.debug(f"Bought {self.shares_held:.6f} shares at "
                           f"{current_price:.2f} for {buy_amount:.2f} "
                           f"(fee: {fee:.2f})")
    
    def _update_portfolio_value(self):
        """Update the portfolio value based on current balance and asset prices."""
        current_price = self.data['close'].iloc[self.current_step]
        self.asset_value = self.shares_held * current_price
        self.portfolio_value = self.balance + self.asset_value
    
    def _calculate_reward(self, action, prev_portfolio_value):
        """
        Calculate the reward based on the action and portfolio change.
        
        Args:
            action: The action taken (0: Sell, 1: Hold, 2: Buy)
            prev_portfolio_value: Portfolio value before action
            
        Returns:
            Scaled reward value
        """
        # Base reward: Change in portfolio value
        portfolio_change = self.portfolio_value - prev_portfolio_value
        reward = portfolio_change * self.reward_scaling
        
        # Discourage holding without position by adding a small negative reward
        if action == 1 and self.shares_held == 0:
            idle_penalty = -0.1 * self.reward_scaling
            reward += idle_penalty
        
        # Encourage selling near peak or buying near bottom for better trading
        if action == 0 and len(self.sell_prices) > 0 and \
           self.last_buy_price is not None:
            # Reward for selling at a profit
            last_sell_price = self.sell_prices[-1]
            buy_sell_ratio = (last_sell_price - self.last_buy_price) / \
                           self.last_buy_price
            if buy_sell_ratio > 0:
                profit_bonus = buy_sell_ratio * 1.0 * self.reward_scaling
                reward += profit_bonus
        
        # Penalize excessive drawdown
        if self.max_drawdown > 0.3:  # Penalize drawdowns over 30%
            drawdown_penalty = (self.max_drawdown - 0.3) * 10 * \
                             self.reward_scaling
            reward -= drawdown_penalty
        
        return reward
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Numpy array containing the observation
        """
        # Get historical feature data (sequence_length prior steps)
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.sequence_length + 1)
        
        # Make sure we have at least sequence_length entries
        if end_idx - start_idx + 1 < self.sequence_length:
            start_idx = max(0, end_idx - self.sequence_length + 1)
        
        historical_data = self.data.iloc[start_idx:end_idx+1]
        
        # Ensure we have exactly sequence_length rows for feature data
        if len(historical_data) < self.sequence_length:
            # Pad with the first row if needed
            padding = pd.DataFrame([historical_data.iloc[0]] * 
                                 (self.sequence_length - len(historical_data)))
            historical_data = pd.concat([padding, historical_data])
        
        # Extract features and flatten
        feature_data = []
        for feature in self.features:
            feature_data.extend(historical_data[feature].values)
        
        # Add account information
        account_info = [
            self.balance / self.initial_balance,  # Normalized balance
            # Normalized position value
            self.shares_held * self.data['close'].iloc[self.current_step] / \
                self.initial_balance  
        ]
        
        # Combine features and account info
        observation = np.array(feature_data + account_info, dtype=np.float32)
        
        return observation
    
    def _get_info(self):
        """
        Get additional information about the environment state.
        
        Returns:
            Dictionary containing environment info
        """
        current_price = self.data['close'].iloc[self.current_step]
        
        info = {
            'step': self.current_step,
            'timestamp': self.data.index[self.current_step],
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'asset_value': self.asset_value,
            'portfolio_value': self.portfolio_value,
            'total_trades': self.total_trades,
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'total_holds': self.total_holds,
            'drawdown': self.max_drawdown,
            'initial_balance': self.initial_balance,
            'cash_ratio': self.balance / self.portfolio_value 
                          if self.portfolio_value > 0 else 1.0,
        }
        
        # Calculate returns if we have enough history
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / \
                      np.array(self.portfolio_values[:-1])
            if len(returns) > 0:
                info['returns_mean'] = float(np.mean(returns))
                info['returns_std'] = float(np.std(returns))
                if info['returns_std'] > 0:
                    info['sharpe_ratio'] = float(info['returns_mean'] / \
                                              info['returns_std'] * \
                                              np.sqrt(252))
                else:
                    info['sharpe_ratio'] = 0.0
            else:
                info['returns_mean'] = 0.0
                info['returns_std'] = 0.0
                info['sharpe_ratio'] = 0.0
        
        return info
    
    def render(self):
        """
        Render the environment (human mode or rgb_array).
        """
        if self.render_mode == 'human':
            # Implement human-readable rendering (e.g., print status)
            print(f"Step: {self.current_step}, "
                  f"Portfolio: {self.portfolio_value:.2f}, "
                  f"Balance: {self.balance:.2f}, "
                  f"Shares: {self.shares_held:.4f}")
            # Potentially add plotting for human mode
            return None  # Return None for human mode

        elif self.render_mode == 'rgb_array':
            # --- Basic RGB Array Rendering --- 
            # Create a simple plot and return as numpy array
            fig, ax = plt.subplots(figsize=(8, 4))
            start = max(0, self.current_step - self.window_size)
            end = self.current_step + 1
            
            # Plot price
            ax.plot(
                self.data.index[start:end],
                self.data['close'].iloc[start:end],
                label='Close Price', color='blue'
            )
            
            # Mark trades
            buy_steps = [t['step'] for t in self.trades if t['type'] == 'buy']
            sell_steps = [t['step'] for t in self.trades if t['type'] == 'sell']
            
            buy_indices = [i for i, step in 
                           enumerate(self.data.index[start:end]) 
                           if step in buy_steps]
            sell_indices = [i for i, step in 
                            enumerate(self.data.index[start:end]) 
                            if step in sell_steps]
            
            if buy_indices:
                ax.scatter(
                     self.data.index[start:end][buy_indices],
                     self.data['close'].iloc[start:end][buy_indices],
                     marker='^', color='green', s=100, label='Buy'
                 )
            if sell_indices:
                ax.scatter(
                    self.data.index[start:end][sell_indices],
                    self.data['close'].iloc[start:end][sell_indices],
                    marker='v', color='red', s=100, label='Sell'
                )
            
            ax.set_title(f"Trading Environment - Step {self.current_step}")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            
            # Draw figure and convert to RGB array
            fig.canvas.draw()
            rgb_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)  # Close the plot to avoid display
            
            return rgb_array
        else:
            # If render_mode is None or unsupported
            return None  # Or raise an error if preferred

    def close(self):
        """Clean up resources."""
        plt.close()
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed] 