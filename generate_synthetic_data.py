#!/usr/bin/env python
"""
Synthetic data generation script for the crypto trading model.

This script generates 4 years of synthetic cryptocurrency price data with various
patterns, engineers features, and saves the processed data in HDF5 format.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import h5py
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple

# Import project modules
from crypto_trading_model.utils.synthetic_data import SyntheticDataGenerator
from crypto_trading_model.utils.feature_engineering import FeatureEngineer
from crypto_trading_model.utils.visualization import MarketVisualizer

# Import project configuration
from crypto_trading_model.config import PATHS, SYNTHETIC_DATA_SETTINGS, FEATURE_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate_synthetic_data')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic data for crypto trading model')
    
    parser.add_argument('--years', type=int, default=4,
                        help='Number of years of data to generate')
    parser.add_argument('--timeframes', type=str, default='15m,4h,1d',
                        help='Comma-separated list of timeframes to generate')
    parser.add_argument('--base_price', type=float, default=10000.0,
                        help='Base price for synthetic data')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output directory for generated data')
    
    return parser.parse_args()

def generate_multi_timeframe_data(
    start_date: datetime,
    end_date: datetime,
    timeframes: List[str] = None,
    pattern_distribution: Dict[str, float] = None,
    base_price: float = 10000.0
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic cryptocurrency data for multiple timeframes.
    
    Parameters:
    -----------
    start_date : datetime
        Start date for the data
    end_date : datetime
        End date for the data
    timeframes : List[str]
        List of timeframes to generate
    pattern_distribution : Dict[str, float]
        Distribution of patterns to generate
    base_price : float
        Base price for the data
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames with generated data for each timeframe
    """
    if timeframes is None:
        timeframes = ["15m", "4h", "1d"]
    
    if pattern_distribution is None:
        pattern_distribution = {
            'trend': 0.3,
            'reversal': 0.2,
            'consolidation': 0.3,
            'breakout': 0.2
        }
    
    logger.info(f"Generating synthetic data from {start_date} to {end_date}")
    logger.info(f"Timeframes: {timeframes}")
    
    # Create synthetic data generator
    generator = SyntheticDataGenerator(base_price=base_price)
    
    # Calculate timeframe parameters
    timeframe_params = {
        "15m": {"minutes": 15, "noise_level": 0.015},
        "4h": {"minutes": 240, "noise_level": 0.025},
        "1d": {"minutes": 1440, "noise_level": 0.035}
    }
    
    results = {}
    
    for timeframe in timeframes:
        logger.info(f"Generating {timeframe} data...")
        
        if timeframe not in timeframe_params:
            logger.warning(f"Unknown timeframe: {timeframe}. Skipping.")
            continue
        
        # Calculate number of periods
        minutes_delta = int((end_date - start_date).total_seconds() / 60)
        n_periods = minutes_delta // timeframe_params[timeframe]["minutes"]
        
        # Generate timeframe-specific dates
        dates = [start_date + timedelta(minutes=i * timeframe_params[timeframe]["minutes"]) 
                for i in range(n_periods)]
        
        # Calculate number of samples for each pattern based on distribution
        n_samples = {}
        total_periods = len(dates)
        
        for pattern, prob in pattern_distribution.items():
            n_samples[pattern] = int(total_periods * prob)
        
        # Adjust for rounding errors
        remaining = total_periods - sum(n_samples.values())
        if remaining > 0:
            patterns = list(n_samples.keys())
            n_samples[patterns[0]] += remaining
        
        # Generate data for each pattern
        all_data = []
        current_idx = 0
        
        for pattern, count in n_samples.items():
            logger.info(f"  - Generating {count} periods of '{pattern}' pattern")
            
            while count > 0:
                # Determine segment size (between 50 and 200 periods)
                segment_size = min(count, np.random.randint(50, 201))
                count -= segment_size
                
                # Determine pattern parameters randomly
                if pattern == 'trend':
                    params = {
                        'n_steps': segment_size,
                        'trend_type': np.random.choice(['bullish', 'bearish']),
                        'strength': np.random.uniform(0.01, 0.05)
                    }
                elif pattern == 'reversal':
                    params = {
                        'n_steps': segment_size,
                        'reversal_point': np.random.uniform(0.3, 0.7),
                        'pre_reversal_trend': np.random.uniform(0.01, 0.05) * (1 if np.random.random() > 0.5 else -1),
                        'post_reversal_trend': np.random.uniform(0.01, 0.05) * (-1 if np.random.random() > 0.5 else 1)
                    }
                elif pattern == 'consolidation':
                    params = {
                        'n_steps': segment_size,
                        'range_width': np.random.uniform(0.01, 0.03)
                    }
                elif pattern == 'breakout':
                    params = {
                        'n_steps': segment_size,
                        'consolidation_steps': int(segment_size * np.random.uniform(0.5, 0.8)),
                        'breakout_direction': np.random.choice(['up', 'down']),
                        'range_width': np.random.uniform(0.01, 0.03),
                        'breakout_strength': np.random.uniform(0.02, 0.06)
                    }
                
                # Generate the sample with timeframe-specific noise level
                noise_level = timeframe_params[timeframe]["noise_level"]
                sample = generator.generate_synthetic_pattern(pattern, params, noise_level)
                
                # Update the index to be consecutive dates
                segment_dates = dates[current_idx:current_idx+segment_size]
                current_idx += segment_size
                
                # If not enough dates, break
                if len(segment_dates) < len(sample):
                    break
                
                sample.index = segment_dates[:len(sample)]
                
                # Add pattern label
                sample['pattern'] = pattern
                
                all_data.append(sample)
        
        # Combine all segments
        if all_data:
            combined_df = pd.concat(all_data)
            # Sort by timestamp to ensure proper order
            combined_df = combined_df.sort_index()
            results[timeframe] = combined_df
            logger.info(f"  - Generated {len(combined_df)} periods of {timeframe} data")
        else:
            logger.warning(f"  - No data generated for {timeframe}")
    
    return results

def engineer_features(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Add technical indicators and features to the synthetic data.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with synthetic data for each timeframe
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames with engineered features
    """
    logger.info("Engineering features for all timeframes")
    feature_engineer = FeatureEngineer()
    
    results = {}
    
    for timeframe, df in data_dict.items():
        logger.info(f"Adding features to {timeframe} data...")
        
        # Add trend features
        df = feature_engineer.add_trend_indicators(df)
        
        # Add momentum features
        df = feature_engineer.add_momentum_indicators(df)
        
        # Add volatility features
        df = feature_engineer.add_volatility_indicators(df)
        
        # Add volume features
        df = feature_engineer.add_volume_indicators(df)
        
        # Add cycle features
        df = feature_engineer.add_cycle_indicators(df)
        
        # Add pattern recognition
        df = feature_engineer.add_pattern_recognition(df)
        
        # Add custom features
        df = feature_engineer.add_custom_features(df)
        
        results[timeframe] = df
        logger.info(f"  - Added {len(df.columns) - 6} features to {timeframe} data")  # Subtract OHLCV + pattern
    
    return results

def align_multi_timeframe_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align data from different timeframes to the smallest timeframe.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of aligned DataFrames
    """
    logger.info("Aligning multi-timeframe data")
    
    if not data_dict:
        logger.warning("No data to align")
        return {}
    
    # Find the smallest timeframe (assume it's the one with the most rows)
    smallest_tf = max(data_dict.items(), key=lambda x: len(x[1]))[0]
    smallest_df = data_dict[smallest_tf]
    
    logger.info(f"Aligning all timeframes to {smallest_tf} with {len(smallest_df)} periods")
    
    results = {smallest_tf: smallest_df}
    
    # For each larger timeframe, forward fill to match the smallest timeframe
    for timeframe, df in data_dict.items():
        if timeframe == smallest_tf:
            continue
        
        # Create a new DataFrame with the index of the smallest timeframe
        aligned_df = pd.DataFrame(index=smallest_df.index)
        
        # For each column in the larger timeframe, reindex to the smallest timeframe
        for col in df.columns:
            # Forward fill, then backward fill any remaining NaNs at the beginning
            aligned_df[f"{timeframe}_{col}"] = df[col].reindex(
                smallest_df.index, method='ffill'
            ).fillna(method='bfill')
        
        results[timeframe] = aligned_df
        logger.info(f"  - Aligned {timeframe} data to {smallest_tf}")
    
    return results

def combine_timeframes(aligned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine aligned data from different timeframes into a single DataFrame.
    
    Parameters:
    -----------
    aligned_data : Dict[str, pd.DataFrame]
        Dictionary of aligned DataFrames for each timeframe
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with features from all timeframes
    """
    logger.info("Combining timeframes into a single DataFrame")
    
    if not aligned_data:
        logger.warning("No data to combine")
        return pd.DataFrame()
    
    # Get the base timeframe (should be the one without prefix in column names)
    base_tf = None
    base_df = None
    
    for tf, df in aligned_data.items():
        if df.columns[0].startswith(tf):
            continue
        base_tf = tf
        base_df = df
    
    if base_tf is None:
        logger.error("Could not find base timeframe")
        return pd.DataFrame()
    
    logger.info(f"Using {base_tf} as base timeframe")
    
    # Start with the base timeframe
    combined = base_df.copy()
    
    # Add columns from other timeframes
    for tf, df in aligned_data.items():
        if tf == base_tf:
            continue
        
        # Add all columns from this timeframe
        for col in df.columns:
            combined[col] = df[col]
    
    logger.info(f"Combined DataFrame has {len(combined)} rows and {len(combined.columns)} columns")
    return combined

def save_to_hdf5(data_dict: Dict[str, pd.DataFrame], filepath: str):
    """
    Save multi-timeframe data to HDF5 format.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
    filepath : str
        Path to save the HDF5 file
    """
    logger.info(f"Saving data to HDF5: {filepath}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save each timeframe to a separate group in the HDF5 file
    with pd.HDFStore(filepath, mode='w') as store:
        for timeframe, df in data_dict.items():
            store[f'/{timeframe}'] = df
            logger.info(f"  - Saved {timeframe} data: {len(df)} rows, {len(df.columns)} columns")

def save_combined_to_hdf5(combined_df: pd.DataFrame, filepath: str):
    """
    Save combined multi-timeframe data to HDF5 format.
    
    Parameters:
    -----------
    combined_df : pd.DataFrame
        Combined DataFrame with features from all timeframes
    filepath : str
        Path to save the HDF5 file
    """
    logger.info(f"Saving combined data to HDF5: {filepath}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to HDF5
    combined_df.to_hdf(filepath, key='combined', mode='w')
    logger.info(f"  - Saved combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")

def plot_sample_data(data_dict: Dict[str, pd.DataFrame], save_dir: str):
    """
    Plot sample data from each timeframe and pattern for visual inspection.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
    save_dir : str
        Directory to save the plots
    """
    logger.info("Plotting sample data for visual inspection")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = MarketVisualizer()
    
    for timeframe, df in data_dict.items():
        logger.info(f"Plotting {timeframe} data samples")
        
        # Plot samples for each pattern
        for pattern in df['pattern'].unique():
            # Get sample data for this pattern
            pattern_df = df[df['pattern'] == pattern].iloc[:200]  # Limit to 200 periods
            
            if len(pattern_df) < 50:
                continue
            
            # Plot price chart
            fig = visualizer.plot_price_series(
                data=pattern_df,
                title=f"{timeframe} - {pattern.capitalize()} Pattern",
                volume=True
            )
            
            # Save the plot
            filename = os.path.join(save_dir, f"{timeframe}_{pattern}_price.png")
            fig.savefig(filename)
            plt.close(fig)
            
            # Plot indicators
            indicators = {
                'Trend': ['sma_20', 'sma_50', 'sma_200'],
                'Momentum': ['rsi_14', 'macd', 'macd_signal'],
                'Volatility': ['bbands_upper', 'bbands_middle', 'bbands_lower']
            }
            
            # Check if these indicators exist in the DataFrame
            valid_indicators = {}
            for category, inds in indicators.items():
                valid_inds = [ind for ind in inds if ind in pattern_df.columns]
                if valid_inds:
                    valid_indicators[category] = valid_inds
            
            if valid_indicators:
                fig = visualizer.plot_indicators(
                    data=pattern_df,
                    indicators=valid_indicators,
                    title=f"{timeframe} - {pattern.capitalize()} Indicators"
                )
                
                # Save the plot
                filename = os.path.join(save_dir, f"{timeframe}_{pattern}_indicators.png")
                fig.savefig(filename)
                plt.close(fig)
            
            logger.info(f"  - Plotted {pattern} pattern for {timeframe}")

def main():
    """Main function to generate synthetic data for the crypto trading model."""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("Starting synthetic data generation")
    
    # Set parameters
    years = args.years
    timeframes = args.timeframes.split(',')
    base_price = args.base_price
    
    start_date = datetime.now() - timedelta(days=years*365)  # X years ago
    end_date = datetime.now()
    
    # Create paths
    if args.output:
        data_dir = args.output
    else:
        data_dir = PATHS.get('data', os.path.join('crypto_trading_model', 'data'))
    
    raw_file = os.path.join(data_dir, 'synthetic_raw.h5')
    processed_file = os.path.join(data_dir, 'synthetic_processed.h5')
    combined_file = os.path.join(data_dir, 'synthetic_combined.h5')
    plots_dir = os.path.join(data_dir, 'plots')
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Step 1: Generate multi-timeframe data
    logger.info("Step 1: Generating multi-timeframe synthetic data")
    data_dict = generate_multi_timeframe_data(
        start_date=start_date,
        end_date=end_date,
        timeframes=timeframes,
        base_price=base_price
    )
    
    # Save raw data
    save_to_hdf5(data_dict, raw_file)
    
    # Step 2: Engineer features
    logger.info("Step 2: Engineering features")
    processed_dict = engineer_features(data_dict)
    
    # Save processed data
    save_to_hdf5(processed_dict, processed_file)
    
    # Step 3: Align and combine timeframes
    logger.info("Step 3: Aligning and combining timeframes")
    aligned_dict = align_multi_timeframe_data(processed_dict)
    combined_df = combine_timeframes(aligned_dict)
    
    # Save combined data
    save_combined_to_hdf5(combined_df, combined_file)
    
    # Step 4: Plot sample data for visual inspection
    logger.info("Step 4: Plotting sample data")
    plot_sample_data(processed_dict, plots_dir)
    
    logger.info("Synthetic data generation complete!")
    logger.info(f"Raw data saved to: {raw_file}")
    logger.info(f"Processed data saved to: {processed_file}")
    logger.info(f"Combined data saved to: {combined_file}")
    logger.info(f"Sample plots saved to: {plots_dir}")

if __name__ == "__main__":
    main() 