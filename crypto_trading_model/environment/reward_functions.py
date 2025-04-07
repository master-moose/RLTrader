"""
Reward functions for reinforcement learning-based crypto trading.

This module provides various reward functions that can be used
in the trading environment to shape the agent's behavior.
"""

import numpy as np
from typing import List, Dict, Union, Callable
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reward_functions')

class RewardFunction:
    """
    Base class for reward functions.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize the reward function.
        
        Parameters:
        -----------
        window_size : int
            Number of steps to consider for reward calculation
        """
        self.window_size = window_size
        self.name = "base"
    
    def calculate(self, 
                  current_equity: float, 
                  prev_equity: float, 
                  initial_capital: float,
                  returns: List[float] = None,
                  equity_curve: List[float] = None,
                  **kwargs) -> float:
        """
        Calculate reward.
        
        Parameters:
        -----------
        current_equity : float
            Current account equity
        prev_equity : float
            Previous account equity
        initial_capital : float
            Initial capital
        returns : List[float], optional
            List of historical returns
        equity_curve : List[float], optional
            List of historical equity values
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        float
            Calculated reward
        """
        raise NotImplementedError("Subclasses must implement this method")

class PnLReward(RewardFunction):
    """
    Profit and Loss based reward function.
    
    Rewards the agent based on the change in equity.
    """
    
    def __init__(self, scale: float = 1.0, **kwargs):
        """
        Initialize the PnL reward function.
        
        Parameters:
        -----------
        scale : float
            Scaling factor for the reward
        **kwargs : dict
            Additional arguments to pass to the base class
        """
        super().__init__(**kwargs)
        self.scale = scale
        self.name = "pnl"
    
    def calculate(self, 
                  current_equity: float, 
                  prev_equity: float, 
                  initial_capital: float,
                  **kwargs) -> float:
        """
        Calculate reward based on profit and loss.
        
        Parameters:
        -----------
        current_equity : float
            Current account equity
        prev_equity : float
            Previous account equity
        initial_capital : float
            Initial capital
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        float
            Calculated reward
        """
        # Calculate absolute and relative PnL
        abs_pnl = current_equity - prev_equity
        rel_pnl = abs_pnl / initial_capital
        
        # Scale reward
        reward = rel_pnl * self.scale
        
        return reward

class SharpeReward(RewardFunction):
    """
    Sharpe ratio based reward function.
    
    Rewards the agent based on the risk-adjusted return.
    Higher Sharpe ratio indicates better risk-adjusted performance.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, annualize: bool = False, **kwargs):
        """
        Initialize the Sharpe ratio reward function.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate of return
        annualize : bool
            Whether to annualize the Sharpe ratio
        **kwargs : dict
            Additional arguments to pass to the base class
        """
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize
        self.name = "sharpe"
    
    def calculate(self, 
                  current_equity: float, 
                  prev_equity: float, 
                  initial_capital: float,
                  returns: List[float] = None,
                  **kwargs) -> float:
        """
        Calculate reward based on Sharpe ratio.
        
        Parameters:
        -----------
        current_equity : float
            Current account equity
        prev_equity : float
            Previous account equity
        initial_capital : float
            Initial capital
        returns : List[float]
            List of historical returns
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        float
            Calculated reward
        """
        if returns is None or len(returns) < 2:
            # Fall back to PnL if not enough returns
            return (current_equity - prev_equity) / initial_capital
        
        # Use the most recent returns for calculation
        recent_returns = returns[-self.window_size:]
        
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        # Avoid division by zero
        if std_return == 0:
            if mean_return > self.risk_free_rate:
                return 1.0  # Positive returns with no volatility is good
            else:
                return 0.0  # Below risk-free rate with no volatility
        
        # Calculate Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Annualize if requested
        if self.annualize:
            # Assuming daily returns
            sharpe *= np.sqrt(252)
        
        return sharpe

class SortinoReward(RewardFunction):
    """
    Sortino ratio based reward function.
    
    Rewards the agent based on the downside risk-adjusted return.
    Higher Sortino ratio indicates better downside risk-adjusted performance.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, annualize: bool = False, **kwargs):
        """
        Initialize the Sortino ratio reward function.
        
        Parameters:
        -----------
        risk_free_rate : float
            Risk-free rate of return
        annualize : bool
            Whether to annualize the Sortino ratio
        **kwargs : dict
            Additional arguments to pass to the base class
        """
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.annualize = annualize
        self.name = "sortino"
    
    def calculate(self, 
                  current_equity: float, 
                  prev_equity: float, 
                  initial_capital: float,
                  returns: List[float] = None,
                  **kwargs) -> float:
        """
        Calculate reward based on Sortino ratio.
        
        Parameters:
        -----------
        current_equity : float
            Current account equity
        prev_equity : float
            Previous account equity
        initial_capital : float
            Initial capital
        returns : List[float]
            List of historical returns
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        float
            Calculated reward
        """
        if returns is None or len(returns) < 2:
            # Fall back to PnL if not enough returns
            return (current_equity - prev_equity) / initial_capital
        
        # Use the most recent returns for calculation
        recent_returns = returns[-self.window_size:]
        
        # Calculate mean return
        mean_return = np.mean(recent_returns)
        
        # Calculate downside deviation
        downside_returns = [min(r - self.risk_free_rate, 0) for r in recent_returns]
        downside_dev = np.sqrt(np.mean(np.square(downside_returns)))
        
        # Avoid division by zero
        if downside_dev == 0:
            if mean_return > self.risk_free_rate:
                return 1.0  # Positive returns with no downside is good
            else:
                return 0.0  # Below risk-free rate with no downside
        
        # Calculate Sortino ratio
        sortino = (mean_return - self.risk_free_rate) / downside_dev
        
        # Annualize if requested
        if self.annualize:
            # Assuming daily returns
            sortino *= np.sqrt(252)
        
        return sortino

class CalmarReward(RewardFunction):
    """
    Calmar ratio based reward function.
    
    Rewards the agent based on the drawdown-adjusted return.
    Higher Calmar ratio indicates better drawdown-adjusted performance.
    """
    
    def __init__(self, annualize: bool = False, **kwargs):
        """
        Initialize the Calmar ratio reward function.
        
        Parameters:
        -----------
        annualize : bool
            Whether to annualize the Calmar ratio
        **kwargs : dict
            Additional arguments to pass to the base class
        """
        super().__init__(**kwargs)
        self.annualize = annualize
        self.name = "calmar"
    
    def calculate(self, 
                  current_equity: float, 
                  prev_equity: float, 
                  initial_capital: float,
                  returns: List[float] = None,
                  equity_curve: List[float] = None,
                  **kwargs) -> float:
        """
        Calculate reward based on Calmar ratio.
        
        Parameters:
        -----------
        current_equity : float
            Current account equity
        prev_equity : float
            Previous account equity
        initial_capital : float
            Initial capital
        returns : List[float]
            List of historical returns
        equity_curve : List[float]
            List of historical equity values
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        float
            Calculated reward
        """
        if returns is None or len(returns) < 2 or equity_curve is None:
            # Fall back to PnL if not enough data
            return (current_equity - prev_equity) / initial_capital
        
        # Use the most recent data for calculation
        recent_returns = returns[-self.window_size:]
        recent_equity = equity_curve[-self.window_size:]
        
        # Calculate mean return
        mean_return = np.mean(recent_returns)
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(recent_equity)
        drawdown = (recent_equity - peak) / peak
        max_drawdown = abs(np.min(drawdown))
        
        # Avoid division by zero
        if max_drawdown == 0:
            return mean_return * 10  # Reward positive returns more if no drawdown
        
        # Calculate Calmar ratio
        calmar = mean_return / max_drawdown
        
        # Annualize if requested
        if self.annualize:
            # Assuming daily returns
            calmar *= 252
        
        return calmar

class CompositeReward(RewardFunction):
    """
    Composite reward function.
    
    Combines multiple reward functions with weights.
    """
    
    def __init__(self, 
                 reward_functions: Dict[RewardFunction, float],
                 **kwargs):
        """
        Initialize the composite reward function.
        
        Parameters:
        -----------
        reward_functions : Dict[RewardFunction, float]
            Dictionary mapping reward functions to their weights
        **kwargs : dict
            Additional arguments to pass to the base class
        """
        super().__init__(**kwargs)
        self.reward_functions = reward_functions
        self.name = "composite"
    
    def calculate(self, 
                  current_equity: float, 
                  prev_equity: float, 
                  initial_capital: float,
                  **kwargs) -> float:
        """
        Calculate composite reward.
        
        Parameters:
        -----------
        current_equity : float
            Current account equity
        prev_equity : float
            Previous account equity
        initial_capital : float
            Initial capital
        **kwargs : dict
            Additional arguments
            
        Returns:
        --------
        float
            Calculated reward
        """
        reward = 0.0
        
        # Calculate weighted sum of rewards
        for func, weight in self.reward_functions.items():
            reward += weight * func.calculate(
                current_equity=current_equity,
                prev_equity=prev_equity,
                initial_capital=initial_capital,
                **kwargs
            )
        
        return reward

# Factory function to create reward function by name
def create_reward_function(name: str, **kwargs) -> RewardFunction:
    """
    Create a reward function by name.
    
    Parameters:
    -----------
    name : str
        Name of the reward function
    **kwargs : dict
        Additional arguments to pass to the reward function
        
    Returns:
    --------
    RewardFunction
        Created reward function
    """
    if name == 'pnl':
        return PnLReward(**kwargs)
    elif name == 'sharpe':
        return SharpeReward(**kwargs)
    elif name == 'sortino':
        return SortinoReward(**kwargs)
    elif name == 'calmar':
        return CalmarReward(**kwargs)
    else:
        logger.warning(f"Unknown reward function: {name}. Using PnL reward.")
        return PnLReward(**kwargs) 