"""
Data loading utilities for cryptocurrency market data.

This module provides functions for loading different formats of cryptocurrency
market data and converting them to the standardized format used in the project.
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_crypto_data(
    data_path: str, 
    timeframes: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load cryptocurrency market data from the specified path.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory or file
    timeframes : List[str], optional
        List of timeframes to load. If None, loads all available timeframes.
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    """
    logger.info(f"Loading market data from {data_path}")
    
    # If data_path is a directory, assume it contains CSV files
    if os.path.isdir(data_path):
        return _load_data_from_directory(data_path, timeframes)
    
    # If data_path is a file with .json extension, assume it's a JSON file
    elif data_path.endswith('.json'):
        return _load_data_from_json(data_path)
    
    # If data_path is a file with .csv extension, assume it's a single CSV file
    elif data_path.endswith('.csv'):
        return _load_data_from_csv(data_path)
    
    # If data_path is a file with .pickle or .pkl extension, assume it's a pickle
    elif data_path.endswith('.pickle') or data_path.endswith('.pkl'):
        return _load_data_from_pickle(data_path)
    
    else:
        raise ValueError(f"Unsupported data path format: {data_path}")


def _load_data_from_directory(
    data_dir: str, 
    timeframes: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load data from a directory containing CSV files.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing CSV files
    timeframes : List[str], optional
        List of timeframes to load. If None, loads all available timeframes.
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    """
    data_dict = {}
    
    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # If timeframes are specified, filter files
    if timeframes:
        csv_files = [f for f in csv_files if any(tf in f for tf in timeframes)]
    
    for file in csv_files:
        # Extract timeframe from filename (assuming format like "btc_usdt_1h.csv")
        timeframe = None
        for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d', '1w']:
            if tf in file:
                timeframe = tf
                break
        
        if not timeframe:
            logger.warning(f"Could not determine timeframe for file: {file}, skipping")
            continue
        
        # Load data
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            data_dict[timeframe] = df
            logger.info(f"Loaded {len(df)} rows for timeframe {timeframe}")
        except Exception as e:
            logger.error(f"Error loading file {file}: {str(e)}")
    
    if not data_dict:
        raise ValueError(f"No valid data files found in {data_dir}")
    
    return data_dict


def _load_data_from_json(json_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load data from a JSON file.
    
    Parameters:
    -----------
    json_path : str
        Path to the JSON file
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        data_dict = {}
        for timeframe, values in data.items():
            df = pd.DataFrame(values)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            data_dict[timeframe] = df
            logger.info(f"Loaded {len(df)} rows for timeframe {timeframe}")
        
        return data_dict
    
    except Exception as e:
        logger.error(f"Error loading JSON file {json_path}: {str(e)}")
        raise


def _load_data_from_csv(csv_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load data from a single CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with a single key (default timeframe) and DataFrame as value
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Convert timestamp if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Try to extract timeframe from filename
        timeframe = None
        for tf in ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d', '1w']:
            if tf in os.path.basename(csv_path):
                timeframe = tf
                break
        
        # Use default timeframe if none found
        if not timeframe:
            timeframe = '1h'
            logger.warning(
                f"Could not determine timeframe for {csv_path}, using default: {timeframe}"
            )
        
        return {timeframe: df}
    
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {str(e)}")
        raise


def _load_data_from_pickle(pickle_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load data from a pickle file.
    
    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    """
    try:
        data_dict = pd.read_pickle(pickle_path)
        
        # Ensure the loaded object is a dictionary of DataFrames
        if not isinstance(data_dict, dict):
            raise ValueError("Pickle file does not contain a dictionary of DataFrames")
        
        for timeframe, df in data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Value for timeframe {timeframe} is not a DataFrame")
            
            logger.info(f"Loaded {len(df)} rows for timeframe {timeframe}")
        
        return data_dict
    
    except Exception as e:
        logger.error(f"Error loading pickle file {pickle_path}: {str(e)}")
        raise 