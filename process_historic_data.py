#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('crypto_trading_model.historic_data_processing')

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/historic',
        'data/synthetic',
        'logs',
        'models'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")
    
    logger.info("Directory setup complete.")

def load_config(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)

def calculate_advanced_price_direction(df, threshold_pct=0.01, lookforward_periods=3):
    """
    Calculate advanced price direction labels based on future price movement.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    threshold_pct : float
        Minimum percentage move required to classify as up/down
    lookforward_periods : int
        Number of periods to look forward for price movement
        
    Returns:
    --------
    pd.Series
        Series with price direction labels:
        1 = upward move (buy)
        0 = no significant move (hold)
        -1 = downward move (sell)
    """
    # Calculate future price changes
    future_close = df['close'].shift(-lookforward_periods)
    price_change_pct = (future_close - df['close']) / df['close']
    
    # Apply thresholds to determine direction
    direction = np.zeros(len(df))
    direction[price_change_pct > threshold_pct] = 1     # Buy signal
    direction[price_change_pct < -threshold_pct] = -1   # Sell signal
    
    # Fill NaN values at the end (where we don't have future data)
    direction = pd.Series(direction, index=df.index)
    direction.fillna(0, inplace=True)
    
    return direction

def calculate_multi_timeframe_signal(data_dict, base_timeframe='15m', higher_timeframes=['4h', '1d'], 
                                     threshold_pct=0.01, lookforward_periods={'15m': 4, '4h': 1, '1d': 1}):
    """
    Generate trading signals using information from multiple timeframes.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames for each timeframe
    base_timeframe : str
        Base timeframe to use for signal generation
    higher_timeframes : list
        List of higher timeframes to use
    threshold_pct : float
        Minimum percentage move required to classify as up/down
    lookforward_periods : Dict[str, int]
        Number of periods to look forward for each timeframe
        
    Returns:
    --------
    pd.Series
        Series with trading signals aligned to the base timeframe:
        1 = buy signal
        0 = hold signal
        -1 = sell signal
    """
    # Extract primary dataframe
    primary_df = data_dict[base_timeframe]
    
    # Calculate signals for each timeframe
    signals = {}
    for tf, df in data_dict.items():
        if tf in lookforward_periods:
            # Calculate direction for this timeframe
            direction = calculate_advanced_price_direction(
                df, 
                threshold_pct=threshold_pct, 
                lookforward_periods=lookforward_periods[tf]
            )
            signals[tf] = direction
    
    # Create a common timestamp index based on primary timeframe
    common_index = primary_df.index
    
    # Align all signals to primary timeframe with improved resampling
    aligned_signals = {}
    for tf, signal in signals.items():
        if tf == base_timeframe:
            # Base timeframe remains unchanged
            aligned_signals[tf] = signal
        else:
            # Determine the proper resampling ratio between timeframes
            # This assumes standard timeframe relationships (e.g., 4h = 16 * 15m, 1d = 96 * 15m)
            timeframe_ratios = {
                '15m_4h': 16,  # 16 15-minute candles = 1 4-hour candle
                '15m_1d': 96,  # 96 15-minute candles = 1 day candle
                '4h_1d': 6     # 6 4-hour candles = 1 day candle
            }
            
            # Determine the ratio based on the timeframes we're working with
            ratio_key = f"{base_timeframe}_{tf}"
            reverse_ratio_key = f"{tf}_{base_timeframe}"
            
            # Check if we have a direct ratio or need to reverse it
            if ratio_key in timeframe_ratios:
                ratio = timeframe_ratios[ratio_key]
                # For higher timeframes, we repeat each value 'ratio' times
                # Create a new index that will match the primary timeframe
                resampled = pd.Series(index=common_index, dtype=float)
                
                # Find where each higher timeframe value should be assigned
                for i, idx in enumerate(signal.index):
                    # Find the closest index in the primary timeframe
                    closest_idx = common_index[common_index >= idx]
                    if len(closest_idx) > 0:
                        # Assign this value to the next 'ratio' candles in the primary timeframe
                        segment = common_index[common_index >= closest_idx[0]][:ratio]
                        resampled.loc[segment] = signal.iloc[i]
                
                # Fill any remaining NaN values with forward fill
                aligned_signals[tf] = resampled.ffill()
                
            elif reverse_ratio_key in timeframe_ratios:
                # For lower timeframes to higher (should be rare), use downsampling
                # with a last-value strategy
                ratio = timeframe_ratios[reverse_ratio_key]
                # Downsample by taking the last value of each group
                resampled = signal.reindex(common_index, method='ffill')
                aligned_signals[tf] = resampled
                
            else:
                # Default fallback if ratio not found: use standard resampling with forward fill
                resampled = signal.reindex(common_index, method='ffill')
                aligned_signals[tf] = resampled
    
    # Combine signals with weights (higher timeframes get more weight)
    weights = {'15m': 1.0, '4h': 2.0, '1d': 3.0}
    
    # Initialize combined signal
    combined_signal = pd.Series(0.0, index=common_index)
    total_weight = 0
    
    # Weighted sum of signals
    for tf, signal in aligned_signals.items():
        weight = weights.get(tf, 1.0)  # Default weight is 1.0
        combined_signal += signal * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        combined_signal = combined_signal / total_weight
    
    # Discretize the combined signal
    discretized = np.zeros(len(combined_signal))
    buy_threshold = 0.2   # Adjusted threshold for stronger buy signals
    sell_threshold = -0.2  # Adjusted threshold for stronger sell signals
    
    discretized[combined_signal > buy_threshold] = 1
    discretized[combined_signal < sell_threshold] = -1
    
    return pd.Series(discretized, index=common_index)

def process_historic_data(input_csv_path, output_dir='data/historic', config_path=None):
    """Process historic data from a CSV file.
    
    Parameters:
        input_csv_path: Path to the input CSV file.
        output_dir: Directory where processed data will be saved.
        config_path: Optional path to a configuration file.
    
    Returns:
        Dictionary of DataFrames for different timeframes.
    """
    try:
        if config_path:
            config = load_config(config_path)
        else:
            config = {}
        
        # Extract parameters from config
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('val_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)
        shuffle = config.get('shuffle', False)  # Default to chronological order
        resample = config.get('resample', True)  # Whether to create higher timeframes
        input_tf = config.get('input_tf', '1m')  # Input timeframe of the data
        
        logger.info("Processing historic data based on configuration")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the CSV file
        logger.info(f"Loading data from {input_csv_path}")
        
        try:
            df = pd.read_csv(input_csv_path)
            
            # Check for required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                sys.exit(1)
            
            # Convert timestamp to datetime if needed
            if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                if pd.api.types.is_string_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                elif np.issubdtype(df['timestamp'].dtype, np.number):
                    # If timestamp is a number (Unix timestamp), convert to datetime
                    # For Unix timestamps in seconds (typical for crypto data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"Loaded data: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            sys.exit(1)
        
        # Create dictionary to store dataframes for different timeframes
        dataset = {}
        
        # Store the input data with its original timeframe
        dataset[input_tf] = df
        
        # Resample to higher timeframes if requested
        if resample:
            logger.info("Resampling data to higher timeframes")
            
            # Define resampling function
            def resample_ohlcv(df, timeframe):
                resampled = df.resample(timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                return resampled
            
            # Resample to 15-minute timeframe
            df_15m = resample_ohlcv(df, '15min')
            dataset['15m'] = df_15m
            
            # Resample to 4-hour timeframe
            df_4h = resample_ohlcv(df, '4h')
            dataset['4h'] = df_4h
            
            # Resample to daily timeframe
            df_1d = resample_ohlcv(df, '1D')
            dataset['1d'] = df_1d
            
            logger.info(f"Created dataframes: 15m ({len(df_15m)} rows), 4h ({len(df_4h)} rows), 1d ({len(df_1d)} rows)")
        else:
            # If no resampling, just use the input data with its original timeframe
            logger.info(f"Using only input timeframe: {input_tf} ({len(df)} rows)")
        
        # Calculate features for each timeframe
        logger.info("Calculating technical indicators and features")
        try:
            # Process each timeframe
            for timeframe, df in dataset.items():
                logger.info(f"Adding features for {timeframe} timeframe")
                
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
                df['sma_cross_signal'] = ((df['sma_7'] > df['sma_25']).astype(int) * 2 - 1)

                # Momentum indicators (6 features)
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
                df['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Stochastic oscillator
                low_14 = df['low'].rolling(window=14).min()
                high_14 = df['high'].rolling(window=14).max()
                df['stoch_k'] = 100 * ((df['close'] - low_14) / 
                                     (high_14 - low_14 + np.finfo(float).eps))
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
                df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (
                    (df['bb_upper'] - df['bb_lower']) + np.finfo(float).eps)
                
                # ATR calculation
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr_14'] = tr.rolling(window=14).mean()
                
                # Volume indicators (4 features)
                df['volume_sma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']
                
                # Calculate OBV and CMF
                df['price_change'] = df['close'].diff()
                df['price_direction'] = np.where(df['price_change'] > 0, 1, 
                                              np.where(df['price_change'] < 0, -1, 0))
                df['obv'] = (df['volume'] * df['price_direction']).cumsum()
                
                # Chaikin Money Flow
                money_flow_multiplier = ((df['close'] - df['low']) - 
                                         (df['high'] - df['close'])) / (
                                         df['high'] - df['low'] + np.finfo(float).eps)
                money_flow_volume = money_flow_multiplier * df['volume']
                df['cmf_20'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

                # Price-derived features (4 features)
                df['return'] = df['close'].pct_change(fill_method=None)
                df['log_return'] = np.log1p(df['return'])
                df['high_low_range'] = (df['high'] - df['low']) / df['close']
                df['body_size'] = abs(df['open'] - df['close']) / df['close']
                
                # Add true_volatility estimation based on returns - this helps in RL modeling
                # Use a rolling window of returns to estimate volatility
                df['true_volatility'] = df['return'].rolling(window=30).std()
                
                # Drop rows with NaN values that result from rolling windows
                df.dropna(inplace=True)
                
                # Update the dataset with feature-enriched dataframe
                dataset[timeframe] = df
                logger.info(f"Added features to {timeframe} dataframe: {df.shape}")
            
            # Calculate multi-timeframe signals if we have multiple timeframes
            logger.info("Calculating advanced multi-timeframe signals...")
            
            # For 1-minute data converted to higher timeframes, create lookforward periods
            lookforward_periods = {'15m': 4, '4h': 1, '1d': 1}
                
            multi_tf_signals = calculate_multi_timeframe_signal(
                dataset, 
                base_timeframe='15m', 
                higher_timeframes=['4h', '1d'],
                lookforward_periods=lookforward_periods
            )
            
            # Replace the price_direction in the primary timeframe with the enhanced version
            dataset['15m']['price_direction'] = multi_tf_signals
            
            # Add signal to 4h timeframe by resampling
            df_4h_indices = list(range(0, len(multi_tf_signals), 16))  # 16 15-min periods in 4 hours
            if len(df_4h_indices) <= len(dataset['4h']):
                dataset['4h']['price_direction'] = multi_tf_signals.iloc[df_4h_indices].values[:len(dataset['4h'])]
            
            # Add signal to 1d timeframe by resampling
            df_1d_indices = list(range(0, len(multi_tf_signals), 96))  # 96 15-min periods in 1 day
            if len(df_1d_indices) <= len(dataset['1d']):
                dataset['1d']['price_direction'] = multi_tf_signals.iloc[df_1d_indices].values[:len(dataset['1d'])]
            
            logger.info("Successfully updated price_direction with multi-timeframe signals")
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe signals: {str(e)}")
            logger.info("Falling back to standard price_direction calculation")
            for timeframe, df in dataset.items():
                if 'price_direction' not in df.columns:
                    df['price_direction'] = np.where(df['close'].diff() > 0, 1, 
                                                  np.where(df['close'].diff() < 0, -1, 0))
        
        # Save the processed dataset as HDF5    
        save_dataset_hdf5(dataset, output_dir)
        
        # Create and save train/val/test splits
        create_and_save_splits(dataset, train_ratio, val_ratio, test_ratio, shuffle, output_dir)
        
        logger.info(f"Historic data processing complete. Data saved to {output_dir}")
        return dataset
    except Exception as e:
        logger.error(f"Unexpected error in process_historic_data: {str(e)}")
        raise

def save_dataset_hdf5(dataset, output_dir, filename="historic_dataset.h5"):
    """Save the entire dataset as a single HDF5 file with multiple groups for timeframes"""
    
    # Define HDF5 file path
    hdf5_path = os.path.join(output_dir, filename)
    
    # Save each timeframe as a separate group in the HDF5 file
    with pd.HDFStore(hdf5_path, mode='w') as store:
        for timeframe, df in dataset.items():
            # Save the DataFrame to the HDF5 file, using timeframe as the group name
            store.put(f'/{timeframe}', df, format='table', data_columns=True)
            logger.info(f"Saved {timeframe} data to {hdf5_path}/{timeframe} "
                      f"({len(df)} rows)")
    
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
    """Create train/val/test splits, apply scaling, and save them as separate HDF5 files."""
    
    split_sizes = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_datasets = {split_name: {} for split_name in split_sizes.keys()}
    scalers = {}  # To store scalers for each timeframe
    
    # Define features to scale (adjust this list as needed)
    # These must match what the environment expects
    features_to_scale_base = [
        'open', 'high', 'low', 'close', 'volume',  # Base OHLCV
        'sma_7', 'sma_25', 'sma_99', 'ema_9', 'ema_21',  # Trend
        'price_sma_ratio', 'sma_cross_signal',  # Trend derived
        'rsi_14', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_hist',  # Momentum
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct_b', 'atr_14',  # Volatility
        'volume_sma_20', 'volume_ratio', 'obv', 'cmf_20',  # Volume
        'return', 'log_return', 'high_low_range', 'body_size',  # Price-derived
        'true_volatility'  # Added volatility
    ]
    
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
        
        # Split the data *before* scaling
        train_df = data.iloc[:train_idx].copy()
        val_df = data.iloc[train_idx:val_idx].copy()
        test_df = data.iloc[val_idx:].copy()
        
        # Identify columns present in this dataframe that need scaling
        features_present = [f for f in features_to_scale_base if f in train_df.columns]
        logger.info(f"Scaling features for {timeframe}: {features_present}")
        
        # Check for expected features that are missing
        missing_features = [f for f in features_to_scale_base if f not in features_present]
        if missing_features:
            logger.warning(f"Missing expected features in {timeframe} data: {missing_features}")
        
        # Fit scaler ONLY on training data for these features
        scaler = StandardScaler()
        scaler.fit(train_df[features_present])
        scalers[timeframe] = scaler  # Store the fitted scaler
        
        # Apply scaler to train, val, and test sets
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Create new columns for scaled features
            scaled_feature_names = [f"{f}_scaled" for f in features_present]
            split_df[scaled_feature_names] = scaler.transform(split_df[features_present])
            
            # Log the scaled features that have been created
            logger.info(f"Created scaled features for {timeframe} ({split_name}): {scaled_feature_names}")
            
            # Store the processed dataframe
            split_datasets[split_name][timeframe] = split_df
            
            logger.info(
                f"Split {timeframe} data ({split_name}): {len(split_df)} rows, {len(split_df.columns)} columns"
            )
    
    # Save the scalers
    scaler_save_path = os.path.join(output_dir, "feature_scalers.joblib")
    try:
        joblib.dump(scalers, scaler_save_path)
        logger.info(f"Saved feature scalers to {scaler_save_path}")
    except Exception as e:
        logger.error(f"Error saving scalers: {e}")
    
    # Save each split as a separate HDF5 file
    for split_name, split_data in split_datasets.items():
        split_file = os.path.join(output_dir, f"{split_name}_data.h5")
        
        with pd.HDFStore(split_file, mode='w') as store:
            for timeframe, df_split in split_data.items():
                # Ensure index is suitable for HDFStore (like DatetimeIndex)
                if not isinstance(df_split.index, pd.DatetimeIndex):
                    logger.warning(f"Index for {split_name}/{timeframe} is not DatetimeIndex, attempting conversion.")
                    try:
                        df_split.index = pd.to_datetime(df_split.index)
                    except Exception as e:
                        logger.error(f"Failed to convert index for {split_name}/{timeframe}: {e}. Saving might fail.")
                
                # Check for any NaN values before saving
                nan_count = df_split.isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in {split_name}/{timeframe} data. Filling with zeros.")
                    df_split = df_split.fillna(0)
                
                # Log the columns being saved
                logger.info(f"Saving {split_name}/{timeframe} data with columns: {df_split.columns.tolist()}")
                
                # Save the data
                store.put(f'/{timeframe}', df_split, format='table', data_columns=True)
                
        logger.info(f"Saved {split_name} split to {split_file}")

def main():
    """Main entry point for historic data processing."""
    parser = argparse.ArgumentParser(description="Process historic OHLCV data from CSV to HDF5 format")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file with OHLCV data")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/historic",
        help="Directory where the data will be saved (default: data/historic)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input_tf", 
        type=str, 
        default="1m",
        help="Timeframe of the input data (default: 1m)"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Process historic data
    if args.config:
        process_historic_data(
            input_csv_path=args.input_csv,
            output_dir=args.output_dir,
            config_path=args.config
        )
    else:
        process_historic_data(
            input_csv_path=args.input_csv,
            output_dir=args.output_dir
        )
    
    logger.info("Historic data processing complete.")

if __name__ == "__main__":
    main() 