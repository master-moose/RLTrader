import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def align_timeframes(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align data from multiple timeframes by timestamp
    
    Parameters:
    - data_dict: Dictionary of DataFrames with different timeframes
    
    Returns:
    - Dictionary of aligned DataFrames with same timestamp index
    """
    # Find the smallest timeframe as base
    timeframes = list(data_dict.keys())
    if not timeframes:
        return {}
    
    # Sort timeframes by granularity (assuming format like "15m", "4h", "1d")
    def get_minutes(tf):
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 60 * 24
        return 0
    
    timeframes.sort(key=get_minutes)
    base_tf = timeframes[0]
    base_df = data_dict[base_tf].copy()
    
    # Ensure index is datetime and sorted
    for tf in timeframes:
        if not isinstance(data_dict[tf].index, pd.DatetimeIndex):
            data_dict[tf].index = pd.to_datetime(data_dict[tf].index)
        data_dict[tf] = data_dict[tf].sort_index()
    
    # Create a dictionary to store aligned dataframes
    aligned_dict = {base_tf: base_df}
    
    # For each larger timeframe, forward fill to match the base timeframe
    for tf in timeframes[1:]:
        larger_tf_df = data_dict[tf].copy()
        
        # Reindex the larger timeframe to match the base timeframe's index
        aligned_df = larger_tf_df.reindex(base_df.index, method='ffill')
        
        # Add suffix to column names to avoid confusion
        aligned_df.columns = [f"{col}_{tf}" for col in aligned_df.columns]
        
        aligned_dict[tf] = aligned_df
    
    return aligned_dict

def merge_aligned_timeframes(aligned_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge aligned DataFrames from different timeframes into a single DataFrame
    
    Parameters:
    - aligned_dict: Dictionary of aligned DataFrames
    
    Returns:
    - Single DataFrame with features from all timeframes
    """
    if not aligned_dict:
        return pd.DataFrame()
    
    # Start with the base timeframe (smallest)
    timeframes = list(aligned_dict.keys())
    base_tf = timeframes[0]
    result = aligned_dict[base_tf].copy()
    
    # Add columns from larger timeframes
    for tf in timeframes[1:]:
        # Exclude duplicated columns like timestamp or date
        for col in aligned_dict[tf].columns:
            if col not in result.columns:
                result[col] = aligned_dict[tf][col]
    
    return result

def ensure_continuous_timestamps(df: pd.DataFrame, 
                                timeframe: str, 
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure DataFrame has continuous timestamps for the given timeframe,
    filling gaps with forward-filled values
    
    Parameters:
    - df: DataFrame with timestamp index
    - timeframe: String representing timeframe ("15m", "4h", "1d", etc.)
    - start_date: Optional start date to expand the range
    - end_date: Optional end date to expand the range
    
    Returns:
    - DataFrame with continuous timestamps
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Determine frequency based on timeframe
    freq_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
        '1d': '1D', '3d': '3D', '1w': '1W'
    }
    
    freq = freq_map.get(timeframe)
    if not freq:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Create a continuous date range
    start = pd.to_datetime(start_date) if start_date else df.index.min()
    end = pd.to_datetime(end_date) if end_date else df.index.max()
    
    continuous_idx = pd.date_range(start=start, end=end, freq=freq)
    
    # Reindex with the continuous range
    return df.reindex(continuous_idx, method='ffill')

def verify_alignment(aligned_dict: Dict[str, pd.DataFrame]) -> bool:
    """
    Verify that all aligned DataFrames have the same index
    
    Parameters:
    - aligned_dict: Dictionary of aligned DataFrames
    
    Returns:
    - True if all DataFrames are properly aligned, False otherwise
    """
    if not aligned_dict:
        return True
    
    timeframes = list(aligned_dict.keys())
    base_index = aligned_dict[timeframes[0]].index
    
    for tf in timeframes[1:]:
        if not aligned_dict[tf].index.equals(base_index):
            return False
    
    return True

def extract_common_timeframe(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Extract the common timeframe range from all DataFrames
    
    Parameters:
    - data_dict: Dictionary of DataFrames with different timeframes
    
    Returns:
    - Dictionary of DataFrames with common date range
    """
    if not data_dict:
        return {}
    
    # Find common start and end dates
    start_dates = [df.index.min() for df in data_dict.values()]
    end_dates = [df.index.max() for df in data_dict.values()]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    # Extract common range for each DataFrame
    result = {}
    for tf, df in data_dict.items():
        result[tf] = df.loc[common_start:common_end].copy()
    
    return result 