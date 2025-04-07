#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import tables
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_generation.log')
    ]
)
logger = logging.getLogger('crypto_trading_model.data_generation')

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'data/synthetic',
        'output/time_series',
        'output/reinforcement',
        'output/ensemble',
        'output/backtest',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directory setup complete.")

def load_config(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        sys.exit(1)

def generate_synthetic_data(num_samples=140000, output_dir='data/synthetic', config_path=None):
    """Generate synthetic data for training.
    
    Parameters:
    - num_samples: Number of 15-minute intervals to generate
    - output_dir: Directory to save the HDF5 files
    - config_path: Optional path to a configuration file
    """
    # Load config if provided
    if config_path:
        config = load_config(config_path)
        num_samples = config.get('num_samples', num_samples)
        output_dir = config.get('output_dir', output_dir)
        pattern_distribution = config.get('pattern_distribution', {
            "uptrend": 0.25,
            "downtrend": 0.25,
            "sideways": 0.15,
            "reversal_top": 0.1,
            "reversal_bottom": 0.1,
            "support_bounce": 0.075,
            "resistance_bounce": 0.075,
            "support_break": 0.025,
            "resistance_break": 0.025
        })
        include_indicators = config.get('include_indicators', True)
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('val_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)
        shuffle = config.get('shuffle', False)  # Default to chronological order
        
        logger.info(f"Generating {num_samples} samples based on configuration")
    else:
        pattern_distribution = {
            "uptrend": 0.25,
            "downtrend": 0.25,
            "sideways": 0.15,
            "reversal_top": 0.1,
            "reversal_bottom": 0.1,
            "support_bounce": 0.075,
            "resistance_bounce": 0.075,
            "support_break": 0.025,
            "resistance_break": 0.025
        }
        include_indicators = True
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        shuffle = False
        
        logger.info(f"Generating {num_samples} samples with default settings")
    
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
    df_4h = df_15m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Create 1d dataframe (aggregate from 1h)
    df_1d = df_15m.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    logger.info(f"Generated dataframes: 15m ({len(df_15m)} rows), 4h ({len(df_4h)} rows), 1d ({len(df_1d)} rows)")
    
    # Add simple indicators
    if include_indicators:
        for df in [df_15m, df_4h, df_1d]:
            df['sma_7'] = df['close'].rolling(7).mean()
            df['sma_25'] = df['close'].rolling(25).mean()
            df['rsi_14'] = 50 + np.random.normal(0, 10, len(df))  # Random RSI
            df['rsi_14'] = df['rsi_14'].clip(0, 100)  # Clip to valid range
    
    dataset = {
        '15m': df_15m,
        '4h': df_4h,
        '1d': df_1d
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data to HDF5 format
    save_dataset_hdf5(dataset, output_dir)
    
    # Create and save train/val/test splits
    create_and_save_splits(dataset, train_ratio, val_ratio, test_ratio, shuffle, output_dir)
    
    logger.info(f"Synthetic data generation complete. Data saved to {output_dir}")
    return dataset

def save_dataset_hdf5(dataset, output_dir, filename="synthetic_dataset.h5"):
    """Save the entire dataset as a single HDF5 file with multiple groups for timeframes"""
    
    # Define HDF5 file path
    hdf5_path = os.path.join(output_dir, filename)
    
    # Save each timeframe as a separate group in the HDF5 file
    with pd.HDFStore(hdf5_path, mode='w') as store:
        for timeframe, df in dataset.items():
            # Save the DataFrame to the HDF5 file, using timeframe as the group name
            store.put(f'/{timeframe}', df, format='table', data_columns=True)
            logger.info(f"Saved {timeframe} data to {hdf5_path}/{timeframe} ({len(df)} rows)")
    
    # Create metadata file
    metadata = {
        "creation_date": pd.Timestamp.now().isoformat(),
        "timeframes": list(dataset.keys()),
        "rows_per_timeframe": {tf: len(df) for tf, df in dataset.items()},
        "columns": {tf: list(df.columns) for tf, df in dataset.items()}
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved dataset metadata to {metadata_path}")

def create_and_save_splits(dataset, train_ratio, val_ratio, test_ratio, shuffle, output_dir):
    """Create train/val/test splits and save them as separate HDF5 files"""
    
    split_sizes = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_datasets = {split_name: {} for split_name in split_sizes.keys()}
    
    for timeframe, df in dataset.items():
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Shuffle if requested (usually not recommended for time series)
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=False)
        
        # Calculate split indices
        total_rows = len(data)
        train_idx = int(total_rows * train_ratio)
        val_idx = train_idx + int(total_rows * val_ratio)
        
        # Split the data
        split_datasets["train"][timeframe] = data.iloc[:train_idx]
        split_datasets["val"][timeframe] = data.iloc[train_idx:val_idx]
        split_datasets["test"][timeframe] = data.iloc[val_idx:]
        
        logger.info(f"Split {timeframe} data: train={len(split_datasets['train'][timeframe])}, "
                   f"val={len(split_datasets['val'][timeframe])}, "
                   f"test={len(split_datasets['test'][timeframe])}")
    
    # Save each split as a separate HDF5 file
    for split_name, split_data in split_datasets.items():
        split_file = os.path.join(output_dir, f"{split_name}_data.h5")
        
        with pd.HDFStore(split_file, mode='w') as store:
            for timeframe, df in split_data.items():
                store.put(f'/{timeframe}', df, format='table', data_columns=True)
                
        logger.info(f"Saved {split_name} split to {split_file}")

def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for cryptocurrency trading model")
    parser.add_argument("--num_samples", type=int, default=140000,
                      help="Number of 15-minute samples to generate (default: 140000 for ~4 years)")
    parser.add_argument("--output_dir", type=str, default="data/synthetic",
                      help="Directory where the data will be saved (default: data/synthetic)")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Generate synthetic data
    if args.config:
        generate_synthetic_data(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            config_path=args.config
        )
    else:
        generate_synthetic_data(
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
    
    logger.info("Data generation complete.")

if __name__ == "__main__":
    main() 