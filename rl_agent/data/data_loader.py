"""
Data loader for financial and cryptocurrency time series data.

This module provides functionality for loading, preprocessing, and
feature engineering on financial time series data.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
import matplotlib.pyplot as plt

# Get module logger
logger = logging.getLogger("rl_agent.data")


class DataLoader:
    """
    Data loader for cryptocurrency and financial time series data.
    
    This class loads data from CSV files and provides methods for
    feature engineering, preprocessing, and splitting the data.
    """
    
    def __init__(
        self,
        data_path: str,
        timestamp_column: str = "timestamp",
        datetime_format: Optional[str] = None,
        price_column: str = "close",
        volume_column: str = "volume",
        drop_na: bool = True,
        fill_method: str = "ffill",
        crypto_base: str = "USD",
        data_key: Optional[str] = None,
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV data file
            timestamp_column: Name of the timestamp column
            datetime_format: Format of the datetime in the timestamp column
            price_column: Name of the price column
            volume_column: Name of the volume column
            drop_na: Whether to drop rows with NA values
            fill_method: Method to fill NA values ('ffill', 'bfill', or 'none')
            crypto_base: Base currency for cryptocurrency data
            data_key: Key for HDF5 file group (e.g., '/15m')
        """
        self.data_path = os.path.abspath(os.path.expanduser(data_path))
        self.timestamp_column = timestamp_column
        self.datetime_format = datetime_format
        self.price_column = price_column
        self.volume_column = volume_column
        self.drop_na = drop_na
        self.fill_method = fill_method
        self.crypto_base = crypto_base
        self.data_key = data_key
        
        # Check if data file exists
        if not os.path.exists(self.data_path):
            # Try alternate paths that might work in container environments
            alt_paths = []
            
            # If path starts with /data, try with workspace prefix
            if self.data_path.startswith('/data'):
                alt_paths.append(os.path.join('/workspace', self.data_path[1:]))
            
            # If path starts with /workspace, try with data prefix
            if self.data_path.startswith('/workspace'):
                alt_paths.append(os.path.join('/data', self.data_path[10:]))
            
            # Try different path variations
            found = False
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Original path {self.data_path} not found, using alternative: {alt_path}")
                    self.data_path = alt_path
                    found = True
                    break
            
            if not found:
                logger.error(f"Data file not found at {self.data_path} or any alternatives: {alt_paths}")
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the CSV file.
        
        Returns:
            DataFrame with preprocessed data
        """
        logger.debug(f"Loading data from {self.data_path}")
        
        # Determine file type
        file_extension = os.path.splitext(self.data_path)[1].lower()
        data = None
        
        if file_extension == '.csv':
            logger.debug("Detected CSV file format.")
            try:
                data = pd.read_csv(self.data_path)
            except Exception as e:
                logger.error(f"Error loading CSV file: {e}")
                raise
        elif file_extension in ['.h5', '.hdf5']:
            logger.debug("Detected HDF5 file format.")
            if not self.data_key:
                logger.warning("HDF5 file specified but no data_key provided. Attempting to load default key '/data' or first available key.")
                # Try to infer a key if none provided (optional, can be error-prone)
                try:
                    with pd.HDFStore(self.data_path, mode='r') as store:
                        keys = store.keys()
                        if not keys:
                             raise ValueError("HDF5 file contains no keys.")
                        self.data_key = keys[0] # Use the first key
                        logger.info(f"Using inferred data_key: {self.data_key}")
                except Exception as e:
                    logger.error(f"Could not read keys from HDF5 file {self.data_path}: {e}")
                    raise

            try:
                logger.debug(f"Loading data from HDF5 key: {self.data_key}")
                data = pd.read_hdf(self.data_path, key=self.data_key)
            except KeyError:
                logger.error(f"Key '{self.data_key}' not found in HDF5 file: {self.data_path}")
                raise
            except Exception as e:
                logger.error(f"Error loading HDF5 file with key '{self.data_key}': {e}")
                raise
        else:
            logger.error(f"Unsupported file format: {file_extension}. Please use .csv or .h5/.hdf5")
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        if data is None or data.empty:
            logger.error("Loaded data is empty.")
            raise ValueError("Failed to load non-empty data.")

        # --- Post-loading processing (applies to both CSV and HDF5) ---
        
        # Check if timestamp column exists; if not, assume it's the index
        if self.timestamp_column in data.columns:
            timestamp_source = data[self.timestamp_column]
            is_index = False
        elif data.index.name == self.timestamp_column or isinstance(data.index, pd.DatetimeIndex):
            timestamp_source = data.index
            is_index = True
        else:
            timestamp_source = None
            logger.warning(f"Timestamp column/index '{self.timestamp_column}' not found.")

        # Parse timestamps if found and not already DatetimeIndex
        if timestamp_source is not None and not isinstance(timestamp_source, pd.DatetimeIndex):
            try:
                parsed_timestamps = pd.to_datetime(timestamp_source, format=self.datetime_format)
                if is_index:
                    data.index = parsed_timestamps
                else:
                    data[self.timestamp_column] = parsed_timestamps
                    # Set timestamp as index if not already set
                    if not isinstance(data.index, pd.DatetimeIndex):
                         data.set_index(self.timestamp_column, inplace=True)
            except Exception as e:
                logger.warning(f"Error parsing timestamps: {e}. Proceeding without datetime index.")
        elif isinstance(timestamp_source, pd.DatetimeIndex):
            logger.debug("Data already has a DatetimeIndex.")
        
        # Handle missing values (ensure this happens after potential index setting)
        if data.isnull().values.any():
            original_rows = len(data)
            if self.drop_na:
                data.dropna(inplace=True)
                logger.info(f"Dropped {original_rows - len(data)} rows with NA values.")
            elif self.fill_method.lower() != 'none':
                fill_method_used = self.fill_method.lower()
                if fill_method_used in ['ffill', 'bfill']:
                    data.fillna(method=fill_method_used, inplace=True)
                else:
                    data.fillna(0, inplace=True) # Default fill with 0
                    fill_method_used = 'zero'
                logger.info(f"Filled NA values using method: {fill_method_used}")
            else:
                 logger.info("NA values present but not handled (fill_method='none').")
        else:
            logger.debug("No NA values found in the loaded data.")
        
        # Ensure all numeric columns are float32 for better memory usage
        for col in data.select_dtypes(include=np.number).columns:
            data[col] = data[col].astype(np.float32)
        
        logger.debug(f"Loaded data with shape: {data.shape}")
        
        return data
    
    def add_technical_indicators(
        self, 
        data: pd.DataFrame, 
        indicators: List[str] = None
    ) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with price data
            indicators: List of indicators to add
            
        Returns:
            DataFrame with technical indicators
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Define available indicators
        available_indicators = {
            'rsi': self._add_rsi,
            'macd': self._add_macd,
            'bollinger': self._add_bollinger,
            'sma': self._add_sma,
            'ema': self._add_ema,
            'atr': self._add_atr,
            'momentum': self._add_momentum,
            'roc': self._add_roc,  # Rate of change
            'volatility': self._add_volatility,
        }
        
        # Default indicators if none specified
        if indicators is None:
            indicators = ['rsi', 'macd', 'bollinger', 'ema']
        
        # Add each requested indicator
        for indicator in indicators:
            if indicator.lower() in available_indicators:
                logger.info(f"Adding {indicator} indicator")
                try:
                    df = available_indicators[indicator.lower()](df)
                except Exception as e:
                    logger.error(f"Error adding {indicator}: {e}")
            else:
                logger.warning(f"Indicator '{indicator}' not available. "
                              f"Available indicators: {list(available_indicators.keys())}")
        
        return df
    
    def _add_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index (RSI) indicator."""
        df = data.copy()
        price = df[self.price_column]
        
        # Calculate price change
        delta = price.diff()
        
        # Calculate gain and loss
        gain = delta.copy()
        gain[gain < 0] = 0
        loss = -delta.copy()
        loss[loss < 0] = 0
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        df['rsi'] = rsi
        return df
    
    def _add_macd(
        self, 
        data: pd.DataFrame, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> pd.DataFrame:
        """Add Moving Average Convergence Divergence (MACD) indicator."""
        df = data.copy()
        price = df[self.price_column]
        
        # Calculate EMAs
        ema_fast = price.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price.ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD and signal line
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        return df
    
    def _add_bollinger(
        self, 
        data: pd.DataFrame, 
        window: int = 20, 
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df = data.copy()
        price = df[self.price_column]
        
        # Calculate middle band (SMA)
        middle_band = price.rolling(window=window).mean()
        
        # Calculate standard deviation
        std = price.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        df['bb_middle'] = middle_band
        df['bb_upper'] = upper_band
        df['bb_lower'] = lower_band
        
        # Calculate bandwidth and %B
        bandwidth = (upper_band - lower_band) / middle_band
        percent_b = (price - lower_band) / (upper_band - lower_band + 1e-10)
        
        df['bb_bandwidth'] = bandwidth
        df['bb_percent_b'] = percent_b
        
        return df
    
    def _add_sma(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [5, 20, 50, 200]
    ) -> pd.DataFrame:
        """Add Simple Moving Averages (SMA)."""
        df = data.copy()
        price = df[self.price_column]
        
        for window in windows:
            df[f'sma_{window}'] = price.rolling(window=window).mean()
        
        return df
    
    def _add_ema(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [5, 20, 50, 200]
    ) -> pd.DataFrame:
        """Add Exponential Moving Averages (EMA)."""
        df = data.copy()
        price = df[self.price_column]
        
        for window in windows:
            df[f'ema_{window}'] = price.ewm(span=window, adjust=False).mean()
        
        return df
    
    def _add_atr(
        self, 
        data: pd.DataFrame, 
        window: int = 14
    ) -> pd.DataFrame:
        """Add Average True Range (ATR)."""
        df = data.copy()
        
        # Verify we have high, low, close columns
        required_cols = ['high', 'low', 'close']
        if not all(col.lower() in map(str.lower, df.columns) for col in required_cols):
            logger.warning(f"Cannot calculate ATR - missing required columns {required_cols}")
            return df
        
        # Get high, low, close
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window).mean()
        
        df['atr'] = atr
        return df
    
    def _add_momentum(
        self, 
        data: pd.DataFrame, 
        window: int = 10
    ) -> pd.DataFrame:
        """Add momentum indicator."""
        df = data.copy()
        df['momentum'] = df[self.price_column].diff(window)
        return df
    
    def _add_roc(
        self, 
        data: pd.DataFrame, 
        window: int = 10
    ) -> pd.DataFrame:
        """Add Rate of Change (ROC) indicator."""
        df = data.copy()
        df['roc'] = ((df[self.price_column] / df[self.price_column].shift(window)) - 1) * 100
        return df
    
    def _add_volatility(
        self, 
        data: pd.DataFrame, 
        window: int = 20
    ) -> pd.DataFrame:
        """Add volatility (standard deviation of returns)."""
        df = data.copy()
        price = df[self.price_column]
        
        # Calculate returns
        returns = price.pct_change()
        
        # Calculate rolling standard deviation of returns
        volatility = returns.rolling(window=window).std()
        
        df['volatility'] = volatility
        return df
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize the data using a specified method.
        
        Args:
            data: DataFrame to normalize
            method: Normalization method ('minmax', 'zscore', or 'log')
            
        Returns:
            Normalized DataFrame
        """
        df = data.copy()
        
        if method == 'minmax':
            # Min-max normalization to [0, 1] range
            for col in df.select_dtypes(include=np.number).columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.5  # Default value if all values are the same
        
        elif method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            for col in df.select_dtypes(include=np.number).columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:  # Avoid division by zero
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0  # Default value if all values are the same
        
        elif method == 'log':
            # Log transformation for skewed data
            for col in df.select_dtypes(include=np.number).columns:
                # Make sure all values are positive before log transform
                if (df[col] <= 0).any():
                    min_val = df[col].min()
                    if min_val <= 0:
                        df[col] = df[col] - min_val + 1  # Shift values to be positive
                
                df[col] = np.log(df[col])
        
        else:
            logger.warning(f"Unknown normalization method: {method}. "
                          f"Available methods: 'minmax', 'zscore', 'log'.")
        
        return df
    
    def split_data(
        self, 
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            data: DataFrame to split
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            shuffle: Whether to shuffle the data before splitting
            
        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames
        """
        # Check ratios
        if train_ratio + val_ratio + test_ratio != 1.0:
            logger.warning(f"Split ratios {train_ratio}, {val_ratio}, {test_ratio} don't sum to 1.0. "
                          f"Normalizing...")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        df = data.copy()
        
        if shuffle:
            # Shuffle the data - Note: this isn't typically done for time series
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split the data
        train_data = df.iloc[:train_end]
        val_data = df.iloc[train_end:val_end]
        test_data = df.iloc[val_end:]
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def plot_data(
        self, 
        data: pd.DataFrame, 
        columns: List[str] = None,
        title: str = "Price Data",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot the data.
        
        Args:
            data: DataFrame to plot
            columns: List of columns to plot (default: main price column)
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot (if None, not saved)
            show: Whether to show the plot
        """
        df = data.copy()
        
        if columns is None:
            columns = [self.price_column]
        
        plt.figure(figsize=figsize)
        
        for col in columns:
            if col in df.columns:
                plt.plot(df.index, df[col], label=col)
            else:
                logger.warning(f"Column '{col}' not found in data")
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close() 