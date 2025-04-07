#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path

logger = logging.getLogger('crypto_trading_model.synthetic_data')

def build_synthetic_dataset(num_samples: int, 
                           pattern_distribution: Dict[str, float],
                           with_indicators: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Build a synthetic dataset with price patterns

    Parameters:
    - num_samples: Number of pattern samples to generate
    - pattern_distribution: Dictionary mapping pattern types to their probability
    - with_indicators: Whether to include technical indicators
    
    Returns:
    - Dictionary with dataframes for different timeframes
    """
    logger.info(f"Building synthetic dataset with {num_samples} samples")
    
    # In a real implementation, we would call functions from pattern_generator.py
    # For this simplified version, we'll just create random data
    
    # Generate 15-minute data first (smallest timeframe)
    days = num_samples // (24 * 4)  # 4 15-min periods per hour, 24 hours per day
    logger.info(f"Generating {days} days of synthetic data")
    
    # Create a date range
    date_range = pd.date_range(
        start='2020-01-01', 
        periods=num_samples,
        freq='15min'
    )
    
    # Generate random price data (simple random walk)
    base_price = 10000.0  # Starting price
    returns = np.random.normal(0, 0.002, num_samples)  # Small random returns
    cumulative_returns = np.exp(np.cumsum(returns))  # Log-normal price path
    prices = base_price * cumulative_returns
    
    # Create 15m dataframe
    df_15m = pd.DataFrame({
        'timestamp': date_range,
        'open': prices * (1 - np.random.uniform(0, 0.005, num_samples)),
        'high': prices * (1 + np.random.uniform(0, 0.01, num_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.01, num_samples)),
        'close': prices,
        'volume': np.random.lognormal(10, 1, num_samples)
    })
    df_15m.set_index('timestamp', inplace=True)
    
    # Create 4h dataframe (aggregate from 15m)
    df_4h = df_15m.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Create 1d dataframe (aggregate from 4h)
    df_1d = df_15m.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    logger.info(f"Generated dataframes: 15m ({len(df_15m)} rows), 4h ({len(df_4h)} rows), 1d ({len(df_1d)} rows)")
    
    # Add indicators if requested
    if with_indicators:
        # In a full implementation, this would call indicator_engineering.py
        # Here we'll just add some random indicators
        for df in [df_15m, df_4h, df_1d]:
            df['sma_7'] = df['close'].rolling(7).mean()
            df['sma_25'] = df['close'].rolling(25).mean()
            df['rsi_14'] = 50 + np.random.normal(0, 10, len(df))  # Random RSI
            df['rsi_14'] = df['rsi_14'].clip(0, 100)  # Clip to valid range
    
    return {
        '15m': df_15m,
        '4h': df_4h,
        '1d': df_1d
    }

def save_dataset(dataset: Dict[str, pd.DataFrame], 
                output_dir: str,
                dataset_name: str = "synthetic_dataset") -> None:
    """
    Save the synthetic dataset to files
    
    Parameters:
    - dataset: Dictionary with dataframes for different timeframes
    - output_dir: Directory to save the dataset
    - dataset_name: Name for the dataset files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for timeframe, df in dataset.items():
        # Save CSV
        output_file = os.path.join(output_dir, f"{dataset_name}_{timeframe}.csv")
        df.to_csv(output_file)
        logger.info(f"Saved {timeframe} data to {output_file}")

def create_train_val_test_split(dataset: Dict[str, pd.DataFrame],
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               shuffle: bool = False,
                               output_dir: str = "./data/synthetic") -> None:
    """
    Split the dataset into train, validation, and test sets
    
    Parameters:
    - dataset: Dictionary with dataframes for different timeframes
    - train_ratio: Proportion of data for training
    - val_ratio: Proportion of data for validation
    - test_ratio: Proportion of data for testing
    - shuffle: Whether to shuffle the data before splitting
    - output_dir: Directory to save the split datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for timeframe, df in dataset.items():
        # Time series data is typically split in chronological order
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        
        # Calculate split indices
        total_rows = len(df)
        train_idx = int(total_rows * train_ratio)
        val_idx = train_idx + int(total_rows * val_ratio)
        
        # Split the data
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        # Save the splits
        train_df.to_csv(os.path.join(output_dir, f"train_{timeframe}.csv"))
        val_df.to_csv(os.path.join(output_dir, f"val_{timeframe}.csv"))
        test_df.to_csv(os.path.join(output_dir, f"test_{timeframe}.csv"))
        
        logger.info(f"Split {timeframe} data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}") 