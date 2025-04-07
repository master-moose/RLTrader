"""
Synthetic data generation utilities for the crypto trading model.

This module provides functions for generating synthetic cryptocurrency
price data with various patterns for model training and testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
import logging
from datetime import datetime, timedelta
import random

# Import from parent directory
import sys
sys.path.append('..')
from config import SYNTHETIC_DATA_SETTINGS, PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('synthetic_data')

class SyntheticDataGenerator:
    """
    Class that handles synthetic cryptocurrency price data generation.
    """
    
    def __init__(self, base_price: float = 1000.0, volatility: float = 0.02):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        base_price : float
            Base price level for generated data
        volatility : float
            Base volatility level for random walks
        """
        self.base_price = base_price
        self.volatility = volatility
        self.rng = np.random.RandomState(42)  # For reproducibility
    
    def generate_random_walk(
        self, 
        n_steps: int = 500, 
        drift: float = 0.0, 
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate a random walk price series.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps to generate
        drift : float
            Drift component (positive for uptrend, negative for downtrend)
        noise_level : float
            Level of random noise to add
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV synthetic data
        """
        # Generate returns with drift
        returns = np.random.normal(
            loc=drift, 
            scale=self.volatility, 
            size=n_steps
        )
        
        # Add noise
        noise = np.random.normal(0, noise_level * self.volatility, n_steps)
        returns = returns + noise
        
        # Convert returns to price
        price = self.base_price * (1 + np.cumsum(returns))
        
        # Create OHLCV data from price series
        return self._create_ohlcv_from_price(price)
    
    def _create_ohlcv_from_price(self, price_series: np.ndarray) -> pd.DataFrame:
        """
        Create OHLCV data from a price series.
        
        Parameters:
        -----------
        price_series : np.ndarray
            Array of closing prices
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV synthetic data
        """
        n_points = len(price_series)
        
        # Create timestamp index
        now = datetime.now()
        timestamps = [now - timedelta(hours=n_points-i) for i in range(n_points)]
        
        # Generate OHLC from close prices
        volatility = self.volatility * price_series
        
        # Open typically close to previous close
        open_prices = np.roll(price_series, 1)
        open_prices[0] = price_series[0] * (1 - self.rng.normal(0, self.volatility))
        
        # High and low reflect volatility around close
        high_prices = price_series + self.rng.uniform(0.001, 0.01) * volatility
        low_prices = price_series - self.rng.uniform(0.001, 0.01) * volatility
        
        # Ensure high >= open/close >= low
        for i in range(n_points):
            high_prices[i] = max(high_prices[i], open_prices[i], price_series[i])
            low_prices[i] = min(low_prices[i], open_prices[i], price_series[i])
        
        # Generate synthetic volume (correlates with volatility)
        volume = self.rng.gamma(
            shape=2.0, 
            scale=self.base_price * 10 * (1 + 5 * np.abs(np.diff(np.append(open_prices[0], price_series)) / price_series))
        )
        
        # Create dataframe
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': price_series,
            'volume': volume
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def generate_trend_pattern(
        self, 
        n_steps: int = 200, 
        trend_type: str = 'bullish',
        strength: float = 0.05,
        noise_level: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate a trend pattern (bull/bear trend).
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps to generate
        trend_type : str
            Type of trend ('bullish' or 'bearish')
        strength : float
            Strength of the trend (drift magnitude)
        noise_level : float
            Level of random noise to add
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV synthetic data
        """
        drift = strength if trend_type == 'bullish' else -strength
        return self.generate_random_walk(n_steps, drift, noise_level)
    
    def generate_reversal_pattern(
        self, 
        n_steps: int = 200, 
        reversal_point: float = 0.6,
        pre_reversal_trend: float = 0.05,
        post_reversal_trend: float = -0.05,
        noise_level: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate a trend reversal pattern.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps to generate
        reversal_point : float
            Point at which reversal occurs (0-1 range)
        pre_reversal_trend : float
            Trend before reversal
        post_reversal_trend : float
            Trend after reversal
        noise_level : float
            Level of random noise to add
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV synthetic data
        """
        reversal_idx = int(n_steps * reversal_point)
        
        # Generate pre-reversal trend
        pre_df = self.generate_random_walk(
            reversal_idx, 
            drift=pre_reversal_trend, 
            noise_level=noise_level
        )
        
        # Generate post-reversal trend
        post_base_price = pre_df['close'].iloc[-1]
        self.base_price = post_base_price  # Set new base price
        post_df = self.generate_random_walk(
            n_steps - reversal_idx, 
            drift=post_reversal_trend, 
            noise_level=noise_level
        )
        
        # Combine the two parts
        post_df.index = [pre_df.index[-1] + timedelta(hours=i+1) for i in range(len(post_df))]
        combined_df = pd.concat([pre_df, post_df])
        
        # Reset base price
        self.base_price = 1000.0
        
        return combined_df
    
    def generate_consolidation_pattern(
        self, 
        n_steps: int = 200, 
        range_width: float = 0.03,
        noise_level: float = 0.01
    ) -> pd.DataFrame:
        """
        Generate a consolidation/range pattern.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps to generate
        range_width : float
            Width of the price range as a fraction of base price
        noise_level : float
            Level of random noise to add
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV synthetic data
        """
        # Generate a series with mean-reversion
        price = np.zeros(n_steps)
        price[0] = self.base_price
        
        # Define range bounds
        range_min = self.base_price * (1 - range_width/2)
        range_max = self.base_price * (1 + range_width/2)
        
        # Generate random walk with mean reversion
        for i in range(1, n_steps):
            # Calculate mean reversion pull
            distance_from_mid = price[i-1] - self.base_price
            mean_reversion = -0.2 * distance_from_mid / self.base_price
            
            # Random component
            random_move = np.random.normal(0, self.volatility * noise_level)
            
            # Combined move
            price[i] = price[i-1] * (1 + mean_reversion + random_move)
            
            # Ensure we stay within range
            price[i] = max(min(price[i], range_max), range_min)
        
        return self._create_ohlcv_from_price(price)
    
    def generate_breakout_pattern(
        self, 
        n_steps: int = 200, 
        consolidation_steps: int = 140,
        breakout_direction: str = 'up',
        range_width: float = 0.03,
        breakout_strength: float = 0.06,
        noise_level: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate a breakout pattern.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps to generate
        consolidation_steps : int
            Number of steps in consolidation before breakout
        breakout_direction : str
            Direction of breakout ('up' or 'down')
        range_width : float
            Width of consolidation range
        breakout_strength : float
            Strength of trend after breakout
        noise_level : float
            Level of random noise to add
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV synthetic data
        """
        # Generate consolidation phase
        consolidation_df = self.generate_consolidation_pattern(
            n_steps=consolidation_steps,
            range_width=range_width,
            noise_level=noise_level
        )
        
        # Set new base price for breakout phase
        self.base_price = consolidation_df['close'].iloc[-1]
        
        # Generate breakout phase
        drift = breakout_strength if breakout_direction == 'up' else -breakout_strength
        breakout_df = self.generate_random_walk(
            n_steps - consolidation_steps,
            drift=drift,
            noise_level=noise_level
        )
        
        # Combine the two parts
        breakout_df.index = [
            consolidation_df.index[-1] + timedelta(hours=i+1) 
            for i in range(len(breakout_df))
        ]
        combined_df = pd.concat([consolidation_df, breakout_df])
        
        # Reset base price
        self.base_price = 1000.0
        
        return combined_df
    
    def generate_synthetic_pattern(
        self, 
        pattern_type: str, 
        params: Dict = None,
        noise_level: float = 0.02
    ) -> pd.DataFrame:
        """
        Generate synthetic price data with specified pattern.
        
        Parameters:
        -----------
        pattern_type : str
            Type of pattern to generate ('trend', 'reversal', 'consolidation', 'breakout')
        params : Dict
            Dictionary of parameters for the specific pattern
        noise_level : float
            Amount of random noise to add for realism
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with OHLCV data containing the specified pattern
        """
        if params is None:
            params = {}
        
        params['noise_level'] = noise_level
        
        if pattern_type == 'trend':
            return self.generate_trend_pattern(**params)
        elif pattern_type == 'reversal':
            return self.generate_reversal_pattern(**params)
        elif pattern_type == 'consolidation':
            return self.generate_consolidation_pattern(**params)
        elif pattern_type == 'breakout':
            return self.generate_breakout_pattern(**params)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    def generate_synthetic_dataset(
        self,
        n_samples: int = 1000,
        timeframe: str = '1h',
        pattern_distribution: Dict[str, float] = None,
        save_to_file: bool = True
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset with multiple patterns.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        timeframe : str
            Timeframe label for the generated data
        pattern_distribution : Dict[str, float]
            Distribution of patterns to generate (probabilities must sum to 1)
        save_to_file : bool
            Whether to save the generated data to file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all generated samples
        """
        if pattern_distribution is None:
            pattern_distribution = SYNTHETIC_DATA_SETTINGS['pattern_distribution']
        
        # Validate pattern distribution
        if abs(sum(pattern_distribution.values()) - 1.0) > 0.001:
            logger.warning("Pattern distribution doesn't sum to 1, normalizing")
            total = sum(pattern_distribution.values())
            pattern_distribution = {k: v/total for k, v in pattern_distribution.items()}
        
        # Calculate number of samples for each pattern
        pattern_counts = {
            pattern: int(n_samples * prob)
            for pattern, prob in pattern_distribution.items()
        }
        
        # Adjust for rounding errors
        remaining = n_samples - sum(pattern_counts.values())
        if remaining > 0:
            pattern_counts[list(pattern_counts.keys())[0]] += remaining
        
        # Generate samples for each pattern
        all_samples = []
        labels = []
        
        logger.info(f"Generating {n_samples} synthetic samples with timeframe {timeframe}")
        
        for pattern, count in pattern_counts.items():
            logger.info(f"Generating {count} samples for pattern '{pattern}'")
            
            for i in range(count):
                # Randomize parameters
                if pattern == 'trend':
                    params = {
                        'n_steps': random.randint(180, 220),
                        'trend_type': random.choice(['bullish', 'bearish']),
                        'strength': random.uniform(0.03, 0.07)
                    }
                elif pattern == 'reversal':
                    params = {
                        'n_steps': random.randint(180, 220),
                        'reversal_point': random.uniform(0.4, 0.7),
                        'pre_reversal_trend': random.uniform(0.03, 0.07) * (1 if random.random() > 0.5 else -1),
                        'post_reversal_trend': random.uniform(0.03, 0.07) * (-1 if random.random() > 0.5 else 1)
                    }
                elif pattern == 'consolidation':
                    params = {
                        'n_steps': random.randint(180, 220),
                        'range_width': random.uniform(0.02, 0.05)
                    }
                elif pattern == 'breakout':
                    params = {
                        'n_steps': random.randint(180, 220),
                        'consolidation_steps': random.randint(120, 160),
                        'breakout_direction': random.choice(['up', 'down']),
                        'range_width': random.uniform(0.02, 0.04),
                        'breakout_strength': random.uniform(0.04, 0.08)
                    }
                
                # Generate the sample
                noise_level = random.choice(SYNTHETIC_DATA_SETTINGS['noise_levels'])
                sample = self.generate_synthetic_pattern(pattern, params, noise_level)
                
                # Add metadata
                sample['pattern'] = pattern
                sample['params'] = str(params)
                sample['noise_level'] = noise_level
                
                all_samples.append(sample)
                labels.append(pattern)
        
        # Save to file if requested
        if save_to_file:
            import os
            # Create directory if it doesn't exist
            os.makedirs(PATHS['synthetic_data'], exist_ok=True)
            
            # Save each pattern type to a separate file
            for pattern in pattern_distribution.keys():
                pattern_samples = [s for s, l in zip(all_samples, labels) if l == pattern]
                if pattern_samples:
                    # Concatenate all samples of this pattern
                    pattern_df = pd.concat(pattern_samples, keys=range(len(pattern_samples)))
                    
                    # Save to file
                    filename = f"{PATHS['synthetic_data']}synthetic_{pattern}_{timeframe}.pkl"
                    pattern_df.to_pickle(filename)
                    logger.info(f"Saved {len(pattern_samples)} {pattern} samples to {filename}")
            
            # Save all samples to a single file
            all_df = pd.concat(all_samples, keys=range(len(all_samples)))
            filename = f"{PATHS['synthetic_data']}synthetic_all_{timeframe}.pkl"
            all_df.to_pickle(filename)
            logger.info(f"Saved all {len(all_samples)} samples to {filename}")
        
        return all_samples

# Function to create a multi-timeframe synthetic dataset
def create_multi_timeframe_synthetic_dataset(
    n_samples: int = 1000,
    timeframes: List[str] = None,
    save_to_file: bool = True
) -> Dict[str, List[pd.DataFrame]]:
    """
    Create a synthetic dataset across multiple timeframes.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    timeframes : List[str]
        List of timeframes to generate data for
    save_to_file : bool
        Whether to save the generated data to file
        
    Returns:
    --------
    Dict[str, List[pd.DataFrame]]
        Dictionary mapping timeframes to lists of DataFrames
    """
    if timeframes is None:
        timeframes = DATA_SETTINGS['timeframes']
    
    generator = SyntheticDataGenerator()
    
    result = {}
    for timeframe in timeframes:
        logger.info(f"Generating synthetic dataset for timeframe {timeframe}")
        samples = generator.generate_synthetic_dataset(
            n_samples=n_samples,
            timeframe=timeframe,
            save_to_file=save_to_file
        )
        result[timeframe] = samples
    
    return result

if __name__ == "__main__":
    # Example usage
    logger.info("Generating synthetic datasets...")
    datasets = create_multi_timeframe_synthetic_dataset(
        n_samples=SYNTHETIC_DATA_SETTINGS['num_samples']
    )
    logger.info("Done!")
