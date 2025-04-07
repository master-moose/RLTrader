"""
Feature engineering utilities for the crypto trading model.

This module provides functions for creating technical indicators
and cross-timeframe features used by the trading models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
import talib
import logging
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# Import from parent directory
import sys
sys.path.append('..')
from config import FEATURE_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')

class FeatureEngineer:
    """
    Class that handles feature engineering for the trading model.
    """
    
    def __init__(self, normalization: str = 'robust_scaler'):
        """
        Initialize the feature engineer.
        
        Parameters:
        -----------
        normalization : str
            Type of normalization to use ('robust_scaler', 'minmax_scaler', None)
        """
        self.normalization = normalization
        self.scalers = {}  # Store fitted scalers for each feature
    
    def add_trend_indicators(self, df: pd.DataFrame, window_sizes: List[int] = None) -> pd.DataFrame:
        """
        Add trend indicators to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        window_sizes : List[int]
            List of window sizes for indicators
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added trend indicators
        """
        if window_sizes is None:
            window_sizes = FEATURE_SETTINGS['window_sizes']
        
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Simple Moving Averages
        for window in window_sizes:
            df_features[f'sma_{window}'] = talib.SMA(df_features['close'].values, timeperiod=window)
        
        # Exponential Moving Averages
        for window in window_sizes:
            df_features[f'ema_{window}'] = talib.EMA(df_features['close'].values, timeperiod=window)
        
        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(
            df_features['close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df_features['macd'] = macd
        df_features['macd_signal'] = macd_signal
        df_features['macd_hist'] = macd_hist
        
        # ADX (Average Directional Index)
        df_features['adx'] = talib.ADX(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            timeperiod=14
        )
        
        # PPO (Percentage Price Oscillator)
        df_features['ppo'] = talib.PPO(
            df_features['close'].values,
            fastperiod=12,
            slowperiod=26,
            matype=0
        )
        
        return df_features
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added momentum indicators
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # RSI (Relative Strength Index)
        df_features['rsi_14'] = talib.RSI(df_features['close'].values, timeperiod=14)
        
        # Stochastic Oscillator
        stoch_k, stoch_d = talib.STOCH(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        df_features['stoch_k'] = stoch_k
        df_features['stoch_d'] = stoch_d
        
        # CCI (Commodity Channel Index)
        df_features['cci_14'] = talib.CCI(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            timeperiod=14
        )
        
        # ROC (Rate of Change)
        df_features['roc_10'] = talib.ROC(df_features['close'].values, timeperiod=10)
        
        # Williams %R
        df_features['willr_14'] = talib.WILLR(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            timeperiod=14
        )
        
        return df_features
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added volatility indicators
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df_features['close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        df_features['bb_upper'] = upper
        df_features['bb_middle'] = middle
        df_features['bb_lower'] = lower
        
        # BB Width and %B
        df_features['bb_width'] = (upper - lower) / middle
        df_features['bb_pct_b'] = (df_features['close'] - lower) / (upper - lower)
        
        # ATR (Average True Range)
        df_features['atr_14'] = talib.ATR(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            timeperiod=14
        )
        
        # Normalized ATR (ATR / Close)
        df_features['natr_14'] = talib.NATR(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            timeperiod=14
        )
        
        # Standard Deviation
        df_features['std_20'] = talib.STDDEV(df_features['close'].values, timeperiod=20, nbdev=1)
        
        return df_features
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added volume indicators
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # OBV (On Balance Volume)
        df_features['obv'] = talib.OBV(df_features['close'].values, df_features['volume'].values)
        
        # Volume SMA
        df_features['volume_sma_20'] = talib.SMA(df_features['volume'].values, timeperiod=20)
        
        # Money Flow Index
        df_features['mfi_14'] = talib.MFI(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            df_features['volume'].values,
            timeperiod=14
        )
        
        # Chaikin A/D Line
        df_features['ad'] = talib.AD(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            df_features['volume'].values
        )
        
        # Chaikin A/D Oscillator
        df_features['adosc'] = talib.ADOSC(
            df_features['high'].values,
            df_features['low'].values,
            df_features['close'].values,
            df_features['volume'].values,
            fastperiod=3,
            slowperiod=10
        )
        
        # Calculate VWAP (Volume Weighted Average Price)
        # This is a more advanced indicator that's not directly available in talib
        df_features['vwap'] = (df_features['volume'] * df_features['close']).cumsum() / df_features['volume'].cumsum()
        
        return df_features
    
    def add_price_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-derived features to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added price-derived features
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Price changes and returns
        df_features['price_change'] = df_features['close'].diff()
        df_features['return'] = df_features['close'].pct_change()
        df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        # Daily high/low ranges
        df_features['high_low_range'] = df_features['high'] - df_features['low']
        df_features['high_low_pct'] = (df_features['high'] - df_features['low']) / df_features['low']
        
        # Candle features
        df_features['body_size'] = abs(df_features['open'] - df_features['close'])
        df_features['upper_shadow'] = df_features['high'] - df_features[['open', 'close']].max(axis=1)
        df_features['lower_shadow'] = df_features[['open', 'close']].min(axis=1) - df_features['low']
        
        # Distance from moving averages
        if 'sma_20' in df_features.columns:
            df_features['dist_from_sma_20'] = (df_features['close'] - df_features['sma_20']) / df_features['sma_20']
        if 'ema_50' in df_features.columns:
            df_features['dist_from_ema_50'] = (df_features['close'] - df_features['ema_50']) / df_features['ema_50']
        
        return df_features
    
    def add_custom_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom chart pattern detection to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added pattern detection features
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Add candlestick pattern recognition from TALib
        # Bullish patterns
        df_features['cdl_hammer'] = talib.CDLHAMMER(df_features['open'].values, df_features['high'].values, 
                                                   df_features['low'].values, df_features['close'].values)
        df_features['cdl_morning_star'] = talib.CDLMORNINGSTAR(df_features['open'].values, df_features['high'].values, 
                                                              df_features['low'].values, df_features['close'].values)
        df_features['cdl_engulfing_bullish'] = talib.CDLENGULFING(df_features['open'].values, df_features['high'].values, 
                                                                 df_features['low'].values, df_features['close'].values)
        
        # Bearish patterns
        df_features['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(df_features['open'].values, df_features['high'].values, 
                                                                df_features['low'].values, df_features['close'].values)
        df_features['cdl_evening_star'] = talib.CDLEVENINGSTAR(df_features['open'].values, df_features['high'].values, 
                                                              df_features['low'].values, df_features['close'].values)
        df_features['cdl_hanging_man'] = talib.CDLHANGINGMAN(df_features['open'].values, df_features['high'].values, 
                                                            df_features['low'].values, df_features['close'].values)
        
        # Support/resistance patterns
        df_features['cdl_doji'] = talib.CDLDOJI(df_features['open'].values, df_features['high'].values, 
                                               df_features['low'].values, df_features['close'].values)
        
        return df_features
    
    def add_cross_timeframe_features(
        self, 
        current_tf_df: pd.DataFrame, 
        higher_tf_df: pd.DataFrame, 
        higher_tf_name: str
    ) -> pd.DataFrame:
        """
        Add features from higher timeframe to current timeframe.
        
        Parameters:
        -----------
        current_tf_df : pd.DataFrame
            DataFrame with OHLCV data for current timeframe
        higher_tf_df : pd.DataFrame
            DataFrame with OHLCV data for higher timeframe
        higher_tf_name : str
            Name of the higher timeframe
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added cross-timeframe features
        """
        # Make a copy to avoid modifying the original
        df_features = current_tf_df.copy()
        
        # Ensure both DataFrames have datetime indices
        current_tf_df = current_tf_df.copy()
        higher_tf_df = higher_tf_df.copy()
        
        if not isinstance(current_tf_df.index, pd.DatetimeIndex):
            current_tf_df.index = pd.to_datetime(current_tf_df.index)
        if not isinstance(higher_tf_df.index, pd.DatetimeIndex):
            higher_tf_df.index = pd.to_datetime(higher_tf_df.index)
        
        # Select key features from higher timeframe to bring to current timeframe
        higher_tf_features = higher_tf_df[['close', 'high', 'low']].copy()
        
        # Add indicators if they exist in the higher timeframe
        for col in higher_tf_df.columns:
            if col in ['rsi_14', 'adx', 'atr_14', 'bb_width', 'macd']:
                higher_tf_features[col] = higher_tf_df[col]
        
        # Rename columns to indicate they're from higher timeframe
        higher_tf_features = higher_tf_features.rename(
            columns={col: f"{col}_{higher_tf_name}" for col in higher_tf_features.columns}
        )
        
        # Resample higher timeframe features to match current timeframe
        # Using forward fill to propagate values
        resampled_higher_tf = higher_tf_features.reindex(current_tf_df.index, method='ffill')
        
        # Add higher timeframe features to current timeframe
        for col in resampled_higher_tf.columns:
            df_features[col] = resampled_higher_tf[col]
        
        # Calculate relationship between timeframes
        if 'close' in df_features.columns and f"close_{higher_tf_name}" in df_features.columns:
            # Deviation from higher timeframe close price
            df_features[f'deviation_from_{higher_tf_name}'] = (
                df_features['close'] - df_features[f'close_{higher_tf_name}']
            ) / df_features[f'close_{higher_tf_name}'] * 100
            
            # Position within higher timeframe range
            df_features[f'pos_in_{higher_tf_name}_range'] = (
                df_features['close'] - df_features[f'low_{higher_tf_name}']
            ) / (df_features[f'high_{higher_tf_name}'] - df_features[f'low_{higher_tf_name}'])
        
        return df_features
    
    def create_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create all features for all timeframes.
        
        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary with timeframes as keys and DataFrames as values
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with timeframes as keys and feature-enhanced DataFrames as values
        """
        # Step 1: Add basic indicators to each timeframe
        features_dict = {}
        for tf, df in data_dict.items():
            logger.info(f"Adding features for timeframe {tf}")
            
            # Copy the DataFrame to avoid modifying the original
            df_features = df.copy()
            
            # Add all types of indicators
            df_features = self.add_trend_indicators(df_features)
            df_features = self.add_momentum_indicators(df_features)
            df_features = self.add_volatility_indicators(df_features)
            df_features = self.add_volume_indicators(df_features)
            df_features = self.add_price_derived_features(df_features)
            df_features = self.add_custom_patterns(df_features)
            
            features_dict[tf] = df_features
        
        # Step 2: Add cross-timeframe features
        # Sort timeframes by period length (assuming format like '1m', '5m', '1h', etc.)
        timeframe_order = sorted(features_dict.keys(), key=self._timeframe_to_minutes)
        
        for i, tf in enumerate(timeframe_order[:-1]):  # Skip the longest timeframe
            # Add features from all longer timeframes
            for higher_tf in timeframe_order[i+1:]:
                logger.info(f"Adding cross-timeframe features from {higher_tf} to {tf}")
                features_dict[tf] = self.add_cross_timeframe_features(
                    features_dict[tf], 
                    features_dict[higher_tf], 
                    higher_tf
                )
        
        # Step 3: Handle missing values
        for tf, df in features_dict.items():
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger.info(f"Handling missing values in {tf}: {missing_values.sum()} total")
                
                # For initial missing values, use forward fill with a limit
                features_dict[tf] = df.fillna(method='ffill', limit=5)
                
                # For any remaining NaNs, use fill with median
                features_dict[tf] = features_dict[tf].fillna(features_dict[tf].median())
        
        return features_dict
    
    def normalize_features(
        self, 
        train_dict: Dict[str, pd.DataFrame], 
        val_dict: Optional[Dict[str, pd.DataFrame]] = None, 
        test_dict: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Normalize features across all timeframes.
        
        Parameters:
        -----------
        train_dict : Dict[str, pd.DataFrame]
            Dictionary with timeframes as keys and training DataFrames as values
        val_dict : Optional[Dict[str, pd.DataFrame]]
            Dictionary with timeframes as keys and validation DataFrames as values
        test_dict : Optional[Dict[str, pd.DataFrame]]
            Dictionary with timeframes as keys and test DataFrames as values
            
        Returns:
        --------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            Dictionaries with normalized training, validation, and test data
        """
        if self.normalization is None:
            logger.info("Skipping normalization as requested")
            return train_dict, val_dict, test_dict
        
        normalized_train = {}
        normalized_val = {} if val_dict else None
        normalized_test = {} if test_dict else None
        
        for tf in train_dict.keys():
            logger.info(f"Normalizing features for timeframe {tf}")
            
            train_df = train_dict[tf].copy()
            
            # Columns to normalize (exclude binary pattern signals, timestamps, etc.)
            exclude_cols = []
            for col in train_df.columns:
                # Skip date-related columns
                if 'date' in col.lower() or 'time' in col.lower():
                    exclude_cols.append(col)
                # Skip binary pattern signals (typically have 0, 100, -100 values)
                elif col.startswith('cdl_'):
                    exclude_cols.append(col)
            
            # Select columns to normalize
            normalize_cols = [col for col in train_df.columns if col not in exclude_cols]
            
            # Create and fit scalers for each feature column
            for col in normalize_cols:
                # Skip columns that already look normalized
                if 'normalized' in col or 'zscore' in col:
                    continue
                
                # Only fit on non-NaN values
                clean_data = train_df[col].dropna().values.reshape(-1, 1)
                
                # Initialize the appropriate scaler
                if self.normalization == 'robust_scaler':
                    scaler = RobustScaler()
                elif self.normalization == 'minmax_scaler':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalization}")
                
                # Fit the scaler if we have valid data
                if len(clean_data) > 0:
                    scaler.fit(clean_data)
                    self.scalers[f"{tf}_{col}"] = scaler
                    
                    # Transform training data
                    train_df[col] = self._transform_with_scaler(train_df[col], scaler)
            
            normalized_train[tf] = train_df
            
            # Transform validation data if provided
            if val_dict:
                val_df = val_dict[tf].copy()
                for col in normalize_cols:
                    if f"{tf}_{col}" in self.scalers:
                        val_df[col] = self._transform_with_scaler(val_df[col], self.scalers[f"{tf}_{col}"])
                normalized_val[tf] = val_df
            
            # Transform test data if provided
            if test_dict:
                test_df = test_dict[tf].copy()
                for col in normalize_cols:
                    if f"{tf}_{col}" in self.scalers:
                        test_df[col] = self._transform_with_scaler(test_df[col], self.scalers[f"{tf}_{col}"])
                normalized_test[tf] = test_df
        
        return normalized_train, normalized_val, normalized_test
    
    def _transform_with_scaler(self, series: pd.Series, scaler) -> pd.Series:
        """
        Transform a series using a fitted scaler, handling NaN values.
        
        Parameters:
        -----------
        series : pd.Series
            Data to transform
        scaler : sklearn.preprocessing.Scaler
            Fitted scaler to use
            
        Returns:
        --------
        pd.Series
            Transformed series
        """
        # Preserve the original index and NaN positions
        original_index = series.index
        nan_mask = series.isna()
        
        # Transform only non-NaN values
        clean_values = series.dropna().values.reshape(-1, 1)
        
        if len(clean_values) > 0:
            transformed = scaler.transform(clean_values).flatten()
            
            # Create a new series with transformed values
            result = pd.Series(index=original_index, dtype=float)
            result.loc[~nan_mask] = transformed
            result.loc[nan_mask] = np.nan
            
            return result
        else:
            return series
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes for sorting.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
        --------
        int
            Equivalent minutes
        """
        # Extract the number and unit
        import re
        match = re.match(r'(\d+)([mhdw])', timeframe)
        if not match:
            return 0  # Default for invalid formats
        
        number, unit = int(match.group(1)), match.group(2)
        
        # Convert to minutes
        if unit == 'm':
            return number
        elif unit == 'h':
            return number * 60
        elif unit == 'd':
            return number * 60 * 24
        elif unit == 'w':
            return number * 60 * 24 * 7
        else:
            return 0

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from utils.data_processing import load_processed_data, create_train_test_split
    
    # Load processed data
    symbol = "BTC/USDT"
    data_dict = load_processed_data(symbol, prefix="aligned")
    
    # Split data
    train_dict, val_dict, test_dict = create_train_test_split(data_dict)
    
    # Create features
    feature_engineer = FeatureEngineer(normalization='robust_scaler')
    
    # Add features to each set
    train_features = feature_engineer.create_features(train_dict)
    val_features = feature_engineer.create_features(val_dict)
    test_features = feature_engineer.create_features(test_dict)
    
    # Normalize features
    norm_train, norm_val, norm_test = feature_engineer.normalize_features(
        train_features, val_features, test_features
    )
    
    # Print feature statistics
    for tf in norm_train.keys():
        print(f"Timeframe {tf} features: {norm_train[tf].shape[1]}")
        print(f"Sample features: {list(norm_train[tf].columns[:10])}") 