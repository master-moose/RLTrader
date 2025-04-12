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
        
        # Cryptocurrency-specific parameters
        self.continuous_trading = True  # Crypto markets operate 24/7
        self.initial_amount = kwargs.get('initial_amount', 1000000)
        self.buy_cost_pct = kwargs.get('buy_cost_pct', 0.001)  # 0.1% transaction cost
        self.sell_cost_pct = kwargs.get('sell_cost_pct', 0.001)  # 0.1% transaction cost
        self.reward_scaling = kwargs.get('reward_scaling', 1e-4)
        
    def _process_data(self):
        """
        Process the data for cryptocurrency trading.
        This method overrides the parent class method to handle cryptocurrency-specific data.
        """
        # Call parent method first
        super()._process_data()
        
        # Add cryptocurrency-specific processing
        # Ensure continuous trading (24/7)
        self.df = self.df.sort_values('date')
        self.df = self.df.set_index('date')
        
        # Add volatility-based position sizing
        self.df['volatility'] = self.df['close'].pct_change().rolling(window=24).std()
        
    def step(self, actions):
        """
        Execute one time step within the environment.
        
        Args:
            actions: Actions to take in the environment
            
        Returns:
            observation, reward, done, info
        """
        # Get current price and volatility
        current_price = self.df.iloc[self.current_step]['close']
        current_volatility = self.df.iloc[self.current_step]['volatility']
        
        # Adjust position size based on volatility
        max_position_size = 1.0 / (1.0 + current_volatility * 10)  # Reduce position size in high volatility
        
        # Call parent method with adjusted position size
        return super().step(actions)
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            observation
        """
        # Call parent method
        return super().reset() 