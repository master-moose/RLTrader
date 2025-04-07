import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Actions(Enum):
    """Possible actions in the trading environment"""
    HOLD = 0
    BUY = 1
    SELL = 2

class Position(Enum):
    """Possible positions in the trading environment"""
    FLAT = 0
    LONG = 1
    SHORT = 2

class CryptoTradingEnv(gym.Env):
    """
    Trading environment for cryptocurrency
    
    Features:
    - Support for multiple timeframes
    - Long and short positions
    - Transaction costs
    - Realistic market constraints
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                data: Dict[str, pd.DataFrame],
                timeframes: List[str] = ["15m", "4h", "1d"],
                window_size: int = 50,
                initial_balance: float = 10000.0,
                transaction_cost: float = 0.001,
                position_size: float = 0.2,
                reward_function: str = "pnl",
                include_position_info: bool = True):
        """
        Initialize the trading environment
        
        Parameters:
        - data: Dictionary of DataFrames with price data for each timeframe
        - timeframes: List of timeframes to include
        - window_size: Size of the observation window
        - initial_balance: Initial account balance
        - transaction_cost: Transaction cost as a percentage
        - position_size: Size of each position as a percentage of balance
        - reward_function: Type of reward function to use
        - include_position_info: Whether to include position info in state
        """
        super().__init__()
        
        self.data = data
        self.timeframes = timeframes
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.reward_function = reward_function
        self.include_position_info = include_position_info
        
        # Verify data structure
        for tf in timeframes:
            if tf not in data:
                raise ValueError(f"Timeframe {tf} not found in data")
        
        # Set the base timeframe (smallest) for price data
        self.base_timeframe = min(timeframes, key=self._get_minutes)
        
        # Get feature dimensions for each timeframe
        self.feature_dims = {}
        for tf in timeframes:
            self.feature_dims[tf] = data[tf].shape[1]  # Number of features per timeframe
        
        # Calculate total observation space size
        self.total_feature_dims = sum(self.feature_dims.values()) * self.window_size
        if self.include_position_info:
            self.total_feature_dims += 3  # Position type, position size, unrealized PnL
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.total_feature_dims,),
            dtype=np.float32
        )
        
        # Set episode variables
        self.reset()
    
    def _get_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        return 0
    
    def reset(self):
        """
        Reset the environment to the initial state
        
        Returns:
        - Initial observation
        """
        # Reset episode variables
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = Position.FLAT
        self.position_price = 0.0
        self.position_size_usd = 0.0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.trade_history = []
        
        # Create the initial observation
        return self._get_observation()
    
    def step(self, action: int):
        """
        Take a step in the environment
        
        Parameters:
        - action: Action to take
        
        Returns:
        - Observation, reward, done, info
        """
        # Get current price data
        current_price = self._get_current_price()
        
        # Process action
        reward, info = self._process_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data[self.base_timeframe]) - 1
        
        # Create the next observation
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_current_price(self) -> float:
        """
        Get the current price from the base timeframe
        
        Returns:
        - Current price
        """
        # Get the current price from the base timeframe
        # Assuming 'close' is the 4th column (index 3) in OHLCV data
        price_data = self.data[self.base_timeframe]
        if 'close' in price_data.columns:
            return price_data.iloc[self.current_step]['close']
        else:
            return price_data.iloc[self.current_step, 3]  # Default to close column
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation
        
        Returns:
        - Observation array
        """
        features = []
        
        # Get data from each timeframe
        for tf in self.timeframes:
            tf_data = self.data[tf]
            
            # Calculate the corresponding index for this timeframe
            tf_minutes = self._get_minutes(tf)
            base_minutes = self._get_minutes(self.base_timeframe)
            tf_factor = base_minutes / tf_minutes
            tf_index = min(len(tf_data) - 1, int(self.current_step * tf_factor))
            
            # Get window of data
            start_idx = max(0, tf_index - self.window_size + 1)
            end_idx = tf_index + 1
            window_data = tf_data.iloc[start_idx:end_idx].values
            
            # Normalize the data (simple min-max for now)
            window_data = self._normalize_data(window_data)
            
            # Pad if necessary
            if len(window_data) < self.window_size:
                padding = np.zeros((self.window_size - len(window_data), window_data.shape[1]))
                window_data = np.vstack([padding, window_data])
            
            # Flatten and add to features
            features.append(window_data.flatten())
        
        # Add position information
        if self.include_position_info:
            position_info = np.zeros(3)
            # Position type (one-hot)
            position_info[self.position.value] = 1.0
            
            # Position size relative to balance
            if self.position != Position.FLAT:
                position_info[1] = self.position_size_usd / self.balance
            
            # Unrealized PnL as percentage of balance
            current_price = self._get_current_price()
            if self.position == Position.LONG:
                unrealized_pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd
                position_info[2] = unrealized_pnl / self.balance
            elif self.position == Position.SHORT:
                unrealized_pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd
                position_info[2] = unrealized_pnl / self.balance
            
            features.append(position_info)
        
        # Concatenate all features
        return np.concatenate(features)
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to range [0, 1]
        
        Parameters:
        - data: Data to normalize
        
        Returns:
        - Normalized data
        """
        # Simple min-max normalization
        result = data.copy()
        for col in range(data.shape[1]):
            col_min = np.min(data[:, col])
            col_max = np.max(data[:, col])
            if col_max > col_min:
                result[:, col] = (data[:, col] - col_min) / (col_max - col_min)
        
        return result
    
    def _process_action(self, action: int, current_price: float) -> Tuple[float, Dict]:
        """
        Process an action and calculate reward
        
        Parameters:
        - action: Action to take
        - current_price: Current price
        
        Returns:
        - Reward, info dictionary
        """
        info = {
            'step': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'position': self.position.name,
            'position_price': self.position_price,
            'position_size': self.position_size_usd,
            'action': Actions(action).name,
        }
        
        # Calculate reward based on action and current position
        reward = 0.0
        
        if action == Actions.HOLD.value:
            # HOLD action
            # Calculate unrealized PnL
            if self.position == Position.LONG:
                unrealized_pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd
                reward = self._calculate_reward(unrealized_pnl, 0.0)
            elif self.position == Position.SHORT:
                unrealized_pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd
                reward = self._calculate_reward(unrealized_pnl, 0.0)
            
            info['pnl'] = 0.0
            info['fees'] = 0.0
        
        elif action == Actions.BUY.value:
            if self.position == Position.FLAT:
                # Open long position
                self.position_size_usd = self.balance * self.position_size
                fees = self.position_size_usd * self.transaction_cost
                self.position_size_usd -= fees
                self.total_fees += fees
                self.position = Position.LONG
                self.position_price = current_price
                
                info['pnl'] = 0.0
                info['fees'] = fees
                reward = self._calculate_reward(0.0, -fees)
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'price': current_price,
                    'action': 'BUY',
                    'position': 'LONG',
                    'size': self.position_size_usd,
                    'fees': fees
                })
            
            elif self.position == Position.SHORT:
                # Close short position
                price_diff = self.position_price - current_price
                pnl = price_diff / self.position_price * self.position_size_usd
                fees = self.position_size_usd * self.transaction_cost
                net_pnl = pnl - fees
                
                self.balance += net_pnl
                self.total_pnl += pnl
                self.total_fees += fees
                
                info['pnl'] = pnl
                info['fees'] = fees
                reward = self._calculate_reward(pnl, -fees)
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'price': current_price,
                    'action': 'BUY',
                    'position': 'CLOSE_SHORT',
                    'size': self.position_size_usd,
                    'pnl': pnl,
                    'fees': fees
                })
                
                # Reset position
                self.position = Position.FLAT
                self.position_price = 0.0
                self.position_size_usd = 0.0
        
        elif action == Actions.SELL.value:
            if self.position == Position.FLAT:
                # Open short position
                self.position_size_usd = self.balance * self.position_size
                fees = self.position_size_usd * self.transaction_cost
                self.position_size_usd -= fees
                self.total_fees += fees
                self.position = Position.SHORT
                self.position_price = current_price
                
                info['pnl'] = 0.0
                info['fees'] = fees
                reward = self._calculate_reward(0.0, -fees)
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'price': current_price,
                    'action': 'SELL',
                    'position': 'SHORT',
                    'size': self.position_size_usd,
                    'fees': fees
                })
            
            elif self.position == Position.LONG:
                # Close long position
                price_diff = current_price - self.position_price
                pnl = price_diff / self.position_price * self.position_size_usd
                fees = self.position_size_usd * self.transaction_cost
                net_pnl = pnl - fees
                
                self.balance += net_pnl
                self.total_pnl += pnl
                self.total_fees += fees
                
                info['pnl'] = pnl
                info['fees'] = fees
                reward = self._calculate_reward(pnl, -fees)
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'price': current_price,
                    'action': 'SELL',
                    'position': 'CLOSE_LONG',
                    'size': self.position_size_usd,
                    'pnl': pnl,
                    'fees': fees
                })
                
                # Reset position
                self.position = Position.FLAT
                self.position_price = 0.0
                self.position_size_usd = 0.0
        
        # Add more info
        info['total_pnl'] = self.total_pnl
        info['total_fees'] = self.total_fees
        info['portfolio_value'] = self.balance
        if self.position != Position.FLAT:
            # Add unrealized PnL to portfolio value
            if self.position == Position.LONG:
                unrealized_pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd
            else:  # SHORT
                unrealized_pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd
            info['portfolio_value'] += unrealized_pnl
        
        return reward, info
    
    def _calculate_reward(self, pnl: float, fees: float) -> float:
        """
        Calculate reward based on selected reward function
        
        Parameters:
        - pnl: Profit/loss amount
        - fees: Transaction fees
        
        Returns:
        - Reward value
        """
        if self.reward_function == "pnl":
            # Simple PnL reward
            return pnl + fees  # fees are negative
        
        elif self.reward_function == "sharpe":
            # Sharpe-like reward (PnL adjusted for risk)
            # Calculate simple return
            return (pnl + fees) / (self.balance * self.position_size) if self.position_size > 0 else 0
        
        elif self.reward_function == "sortino":
            # Sortino-like reward (PnL adjusted for downside risk)
            if pnl + fees < 0:
                return 2 * (pnl + fees)  # Penalize losses more
            else:
                return pnl + fees
        
        elif self.reward_function == "calmar":
            # Calmar-like reward (PnL adjusted for drawdown)
            # Use simple approximation for now
            if self.balance < self.initial_balance:
                drawdown_factor = self.balance / self.initial_balance
                return (pnl + fees) * drawdown_factor
            else:
                return pnl + fees
        
        # Default to simple PnL reward
        return pnl + fees
    
    def render(self, mode='human'):
        """
        Render the environment state
        
        Parameters:
        - mode: Rendering mode
        """
        if mode != 'human':
            return
        
        current_price = self._get_current_price()
        print(f"Step: {self.current_step}")
        print(f"Price: {current_price}")
        print(f"Balance: {self.balance}")
        print(f"Position: {self.position.name}")
        if self.position != Position.FLAT:
            print(f"Position Price: {self.position_price}")
            print(f"Position Size: {self.position_size_usd}")
            # Calculate unrealized PnL
            if self.position == Position.LONG:
                unrealized_pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd
            else:  # SHORT
                unrealized_pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd
            print(f"Unrealized PnL: {unrealized_pnl}")
        print(f"Total PnL: {self.total_pnl}")
        print(f"Total Fees: {self.total_fees}")
        print("-" * 50)
    
    def close(self):
        """Clean up resources"""
        pass

class MultiTimeframeTradingEnv(CryptoTradingEnv):
    """
    Extended trading environment with more sophisticated handling of multiple timeframes
    """
    
    def __init__(self, 
                data: Dict[str, pd.DataFrame],
                timeframes: List[str] = ["15m", "4h", "1d"],
                window_size: int = 50,
                initial_balance: float = 10000.0,
                transaction_cost: float = 0.001,
                position_size: float = 0.2,
                reward_function: str = "pnl",
                include_position_info: bool = True,
                include_technical_indicators: bool = True,
                observation_type: str = "composed"):
        """
        Initialize the multi-timeframe trading environment
        
        Parameters:
        - data: Dictionary of DataFrames with price data for each timeframe
        - timeframes: List of timeframes to include
        - window_size: Size of the observation window
        - initial_balance: Initial account balance
        - transaction_cost: Transaction cost as a percentage
        - position_size: Size of each position as a percentage of balance
        - reward_function: Type of reward function to use
        - include_position_info: Whether to include position info in state
        - include_technical_indicators: Whether to include technical indicators
        - observation_type: Type of observation to return ("flat", "composed", "dict")
        """
        super().__init__(
            data=data,
            timeframes=timeframes,
            window_size=window_size,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            position_size=position_size,
            reward_function=reward_function,
            include_position_info=include_position_info
        )
        
        self.include_technical_indicators = include_technical_indicators
        self.observation_type = observation_type
        
        # If we're using composed observations, redefine the observation space
        if observation_type == "composed":
            self.observation_space = {}
            for tf in timeframes:
                self.observation_space[tf] = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.window_size, self.feature_dims[tf]),
                    dtype=np.float32
                )
            
            if self.include_position_info:
                self.observation_space["position"] = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(3,),
                    dtype=np.float32
                )
        
        elif observation_type == "dict":
            # Dictionary observation space for DQN/PPO agents that support dict spaces
            self.observation_space = spaces.Dict({
                tf: spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(self.window_size, self.feature_dims[tf]),
                    dtype=np.float32
                ) for tf in timeframes
            })
            
            if self.include_position_info:
                self.observation_space["position"] = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(3,),
                    dtype=np.float32
                )
    
    def _get_observation(self):
        """
        Get the current observation based on observation_type
        
        Returns:
        - Observation (flat, composed, or dict)
        """
        if self.observation_type == "flat":
            # Use the parent class implementation for flat observations
            return super()._get_observation()
        
        # For composed or dict observations, create a dictionary
        observation = {}
        
        # Get data from each timeframe
        for tf in self.timeframes:
            tf_data = self.data[tf]
            
            # Calculate the corresponding index for this timeframe
            tf_minutes = self._get_minutes(tf)
            base_minutes = self._get_minutes(self.base_timeframe)
            tf_factor = base_minutes / tf_minutes
            tf_index = min(len(tf_data) - 1, int(self.current_step * tf_factor))
            
            # Get window of data
            start_idx = max(0, tf_index - self.window_size + 1)
            end_idx = tf_index + 1
            window_data = tf_data.iloc[start_idx:end_idx].values
            
            # Normalize the data
            window_data = self._normalize_data(window_data)
            
            # Pad if necessary
            if len(window_data) < self.window_size:
                padding = np.zeros((self.window_size - len(window_data), window_data.shape[1]))
                window_data = np.vstack([padding, window_data])
            
            # Add to observation
            observation[tf] = window_data
        
        # Add position information
        if self.include_position_info:
            position_info = np.zeros(3)
            # Position type (one-hot)
            position_info[self.position.value] = 1.0
            
            # Position size relative to balance
            if self.position != Position.FLAT:
                position_info[1] = self.position_size_usd / self.balance
            
            # Unrealized PnL as percentage of balance
            current_price = self._get_current_price()
            if self.position == Position.LONG:
                unrealized_pnl = (current_price - self.position_price) / self.position_price * self.position_size_usd
                position_info[2] = unrealized_pnl / self.balance
            elif self.position == Position.SHORT:
                unrealized_pnl = (self.position_price - current_price) / self.position_price * self.position_size_usd
                position_info[2] = unrealized_pnl / self.balance
            
            observation["position"] = position_info
        
        return observation 