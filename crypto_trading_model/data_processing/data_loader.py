"""
Data processing utilities for the crypto trading model.

This module provides functions for fetching historical data,
cleaning and preprocessing, and organizing multi-timeframe data.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple
import ccxt
import logging
from datetime import datetime

# Import from the correct location
from crypto_trading_model.config import DATA_SETTINGS, PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_processing')

def fetch_historical_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    exchange_id: str = 'binance'
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from exchange.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT')
    timeframe : str
        Timeframe of the data (e.g., '1h', '1d')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    exchange_id : str
        Exchange ID (default: 'binance')

    Returns:
    --------
    pd.DataFrame
        DataFrame containing OHLCV data
    """
    try:
        logger.info(
            f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}"
        )
        
        # Initialize exchange
        exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
        # Convert dates to timestamps
        since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        until = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Fetch data in chunks due to exchange limits
        all_candles = []
        while since < until:
            logger.debug(
                f"Fetching chunk starting from {datetime.fromtimestamp(since/1000)}"
            )
            candles = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1  # Next timestamp after the last received
            
            # Respect exchange rate limits
            exchange.sleep(exchange.rateLimit)
            
        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the OHLCV data by handling missing values and outliers.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLCV data

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    logger.info(f"Original data shape: {df_clean.shape}")
    missing_values = df_clean.isnull().sum()
    if missing_values.any():
        logger.info(f"Missing values found: {missing_values}")
        
        # Forward fill for small gaps
        df_clean = df_clean.fillna(method='ffill', limit=3)
        
        # Drop any remaining rows with NaN values
        df_clean = df_clean.dropna()
    
    # Handle outliers (using IQR method for price columns)
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds (using a conservative 3 * IQR)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Flag potential outliers
        outliers = df_clean[
            (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        ]
        if not outliers.empty:
            logger.info(f"Found {len(outliers)} potential outliers in {col}")
            
            # For trading data, we don't remove outliers as they might be legitimate price moves
            # Instead, we log them for manual review
            # But we cap extreme values
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    return df_clean

def resample_data(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing OHLCV data
    target_timeframe : str
        Target timeframe for resampling (e.g., '1h', '4h', '1d')

    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame
    """
    # Map timeframe strings to pandas offset aliases
    timeframe_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1H',
        '4h': '4H',
        '6h': '6H',
        '12h': '12H',
        '1d': '1D',
        '1w': '1W'
    }
    
    if target_timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {target_timeframe}")
    
    rule = timeframe_map[target_timeframe]
    
    # Resample the data
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled.dropna()

def prepare_multi_timeframe_data(
    symbol: str,
    timeframes: List[str] = None,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for multiple timeframes.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT')
    timeframes : List[str]
        List of timeframes to fetch (default: from config)
    start_date : str
        Start date in 'YYYY-MM-DD' format (default: from config)
    end_date : str
        End date in 'YYYY-MM-DD' format (default: from config)

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    """
    # Use default values from config if not provided
    if timeframes is None:
        timeframes = DATA_SETTINGS['timeframes']
    if start_date is None:
        start_date = DATA_SETTINGS['start_date']
    if end_date is None:
        end_date = DATA_SETTINGS['end_date']
    
    exchange_id = DATA_SETTINGS['exchange']
    
    # Create directory for saving if it doesn't exist
    os.makedirs(PATHS['historical_data'], exist_ok=True)
    
    # Prepare data for each timeframe
    data_dict = {}
    
    for tf in timeframes:
        # Define file path for caching
        file_path = os.path.join(
            PATHS['historical_data'],
            f"{symbol.replace('/', '_')}_{tf}_{start_date}_{end_date}.csv"
        )
        
        # Check if data is already cached
        if os.path.exists(file_path):
            logger.info(f"Loading cached data for {symbol} {tf} from {file_path}")
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
        else:
            # Fetch and clean data
            df = fetch_historical_data(symbol, tf, start_date, end_date, exchange_id)
            df = clean_data(df)
            
            # Save to cache
            logger.info(f"Saving data to {file_path}")
            df.to_csv(file_path)
        
        data_dict[tf] = df
    
    return data_dict

def align_multi_timeframe_data(
    data_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Align data from different timeframes to ensure they cover the same period.

    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with aligned DataFrames
    """
    if not data_dict:
        return {}
    
    # Find common date range
    start_dates = [df.index.min() for df in data_dict.values()]
    end_dates = [df.index.max() for df in data_dict.values()]
    
    common_start = max(start_dates)
    common_end = min(end_dates)
    
    logger.info(f"Aligning data from {common_start} to {common_end}")
    
    # Trim all dataframes to the common range
    aligned_dict = {}
    for tf, df in data_dict.items():
        aligned_dict[tf] = df.loc[common_start:common_end].copy()
        logger.info(f"Timeframe {tf}: {len(aligned_dict[tf])} rows after alignment")
    
    return aligned_dict

def save_processed_data(
    data_dict: Dict[str, pd.DataFrame],
    symbol: str,
    prefix: str = "processed"
) -> None:
    """
    Save processed data to disk.

    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT')
    prefix : str
        Prefix for the saved files (default: "processed")
    """
    # Create directory if it doesn't exist
    os.makedirs(PATHS['processed_data'], exist_ok=True)
    
    # Save each dataframe
    for tf, df in data_dict.items():
        file_path = os.path.join(
            PATHS['processed_data'],
            f"{prefix}_{symbol.replace('/', '_')}_{tf}.csv"
        )
        df.to_csv(file_path)
        logger.info(f"Saved processed data to {file_path}")

def load_processed_data(
    symbol: str,
    timeframes: List[str] = None,
    prefix: str = "processed"
) -> Dict[str, pd.DataFrame]:
    """
    Load processed data from disk.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT')
    timeframes : List[str]
        List of timeframes to load (default: from config)
    prefix : str
        Prefix for the saved files (default: "processed")

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    """
    if timeframes is None:
        timeframes = DATA_SETTINGS['timeframes']
    
    data_dict = {}
    
    for tf in timeframes:
        file_path = os.path.join(
            PATHS['processed_data'],
            f"{prefix}_{symbol.replace('/', '_')}_{tf}.csv"
        )
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
            data_dict[tf] = df
            logger.info(f"Loaded processed data from {file_path}")
        else:
            logger.warning(f"No processed data found at {file_path}")
    
    return data_dict

def create_train_test_split(
    data_dict: Dict[str, pd.DataFrame],
    train_size: float = None,
    validation_size: float = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split data into training, validation, and test sets.

    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary with timeframes as keys and DataFrames as values
    train_size : float
        Proportion of data to use for training (default: from config)
    validation_size : float
        Proportion of training data to use for validation (default: from config)

    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Training, validation, and test data dictionaries
    """
    if train_size is None:
        train_size = DATA_SETTINGS['train_test_split']
    if validation_size is None:
        validation_size = DATA_SETTINGS['validation_split']
    
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for tf, df in data_dict.items():
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_size)
        val_end = train_end + int(n * train_size * validation_size)
        
        # Split data
        train_data = df.iloc[:train_end].copy()
        val_data = df.iloc[train_end:val_end].copy()
        test_data = df.iloc[val_end:].copy()
        
        # Store in dictionaries
        train_dict[tf] = train_data
        val_dict[tf] = val_data
        test_dict[tf] = test_data
        
        logger.info(f"Timeframe {tf}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_dict, val_dict, test_dict

if __name__ == "__main__":
    # Example usage
    symbol = "BTC/USDT"
    timeframes = ['15m', '1h', '4h', '1d']
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    
    # Fetch and prepare data
    data = prepare_multi_timeframe_data(symbol, timeframes, start_date, end_date)
    aligned_data = align_multi_timeframe_data(data)
    
    # Split data
    train_data, val_data, test_data = create_train_test_split(aligned_data)
    
    # Save processed data
    save_processed_data(aligned_data, symbol, prefix="aligned")
    save_processed_data(train_data, symbol, prefix="train")
    save_processed_data(val_data, symbol, prefix="val")
    save_processed_data(test_data, symbol, prefix="test") 