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
from crypto_trading_model.data_processing.feature_engineering import calculate_multi_timeframe_signal

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
    
    # Add comprehensive set of indicators
    if include_indicators:
        for df in [df_15m, df_4h, df_1d]:
            # Basic OHLCV features (5 features)
            # 'open', 'high', 'low', 'close', 'volume' are already in the DataFrame
            
            # Trend indicators (7 features)
            df['sma_7'] = df['close'].rolling(7).mean()
            df['sma_25'] = df['close'].rolling(25).mean()
            df['sma_99'] = df['close'].rolling(99).mean()
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            # Price position relative to moving averages
            df['price_sma_ratio'] = df['close'] / df['sma_25']
            df['sma_cross_signal'] = ((df['sma_7'] > df['sma_25']).astype(int) * 2 - 1)  # +1 for bullish, -1 for bearish

            # Momentum indicators (6 features)
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + np.finfo(float).eps))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # MACD calculation
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Volatility indicators (6 features)
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * std_20
            df['bb_lower'] = df['bb_middle'] - 2 * std_20
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / ((df['bb_upper'] - df['bb_lower']) + np.finfo(float).eps)
            
            # ATR calculation
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr_14'] = true_range.rolling(14).mean()

            # Volume indicators (4 features)
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On-Balance Volume
            df['price_change'] = df['close'].diff()
            df['price_direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))
            df['obv'] = (df['volume'] * df['price_direction']).cumsum()
            
            # Chaikin Money Flow
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + np.finfo(float).eps)
            money_flow_volume = money_flow_multiplier * df['volume']
            df['cmf_20'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

            # Price-derived features (4 features)
            df['return'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['open'] - df['close']) / df['close']
            
            # Clean up NaN values that resulted from rolling windows
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)

    dataset = {
        '15m': df_15m,
        '4h': df_4h,
        '1d': df_1d
    }
    
    # Generate advanced trading signals using multi-timeframe approach
    try:
        logger.info("Calculating advanced multi-timeframe signals...")
        multi_tf_signals = calculate_multi_timeframe_signal(
            dataset,
            primary_tf='15m',
            threshold_pct=0.015,  # 1.5% threshold for significant moves
            lookforward_periods={'15m': 16, '4h': 4, '1d': 1}  # Look forward periods for each timeframe
        )
        
        # Replace the price_direction in the primary timeframe with the enhanced version
        dataset['15m']['price_direction'] = multi_tf_signals
        
        # Create versions for higher timeframes by downsampling
        # 4-hour timeframe (1 4h candle = 16 15m candles)
        df_4h_indices = list(range(0, len(dataset['15m']), 16))
        dataset['4h']['price_direction'] = multi_tf_signals.iloc[df_4h_indices].values
        
        # Daily timeframe (1 day candle = 96 15m candles)
        df_1d_indices = list(range(0, len(dataset['15m']), 96))
        dataset['1d']['price_direction'] = multi_tf_signals.iloc[df_1d_indices].values
        
        logger.info("Successfully updated price_direction with multi-timeframe signals")
    except Exception as e:
        logger.error(f"Error calculating multi-timeframe signals: {str(e)}")
        logger.info("Falling back to standard price_direction calculation")
    
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