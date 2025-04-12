import numpy as np
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class CryptocurrencyTradingEnv(StockTradingEnv):
    """
    Custom environment for cryptocurrency trading that inherits from FinRL's StockTradingEnv.
    This environment is specifically tailored for cryptocurrency trading with additional
    features and modifications.
    """
    
    def __init__(self, df, **kwargs):
        """
        Initialize the cryptocurrency trading environment.
        
        Args:
            df: DataFrame containing the cryptocurrency data
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(df, **kwargs)
        
    def _process_data(self):
        """
        Process the data for cryptocurrency trading.
        This method overrides the parent class method to handle cryptocurrency-specific data.
        """
        # Call parent method first
        super()._process_data()
        
        # Add cryptocurrency-specific processing if needed
        # For example, handling 24/7 trading, different trading fees, etc.
        
    def step(self, actions):
        """
        Execute one time step within the environment.
        
        Args:
            actions: Actions to take in the environment
            
        Returns:
            observation, reward, done, info
        """
        # Call parent method
        return super().step(actions)
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            observation
        """
        # Call parent method
        return super().reset() 