import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Union, Optional

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for a DataFrame
    
    Parameters:
    - df: DataFrame with OHLCV data
    
    Returns:
    - DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Basic moving averages
    result['sma_7'] = ta.sma(df['close'], length=7)
    result['sma_25'] = ta.sma(df['close'], length=25)
    result['sma_99'] = ta.sma(df['close'], length=99)
    result['ema_9'] = ta.ema(df['close'], length=9)
    result['ema_21'] = ta.ema(df['close'], length=21)
    
    # Oscillators
    result['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    result['macd'] = macd['MACD_12_26_9']
    result['macd_signal'] = macd['MACDs_12_26_9']
    result['macd_hist'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    result['bb_upper'] = bbands['BBU_20_2.0']
    result['bb_middle'] = bbands['BBM_20_2.0']
    result['bb_lower'] = bbands['BBL_20_2.0']
    
    # Volume indicators
    result['obv'] = ta.obv(df['close'], df['volume'])
    result['vwap'] = calculate_vwap(df)
    
    # Volatility indicators
    result['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Momentum indicators
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    result['adx_14'] = adx['ADX_14']
    result['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
    
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    result['stoch_k'] = stoch['STOCHk_14_3_3']
    result['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Support/resistance levels
    result['pivot_point'] = calculate_pivot_points(df)
    
    # Calculate trend identification features
    result['trend_intensity'] = calculate_trend_intensity(df)
    
    return result

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Parameters:
    - df: DataFrame with OHLCV data
    
    Returns:
    - Series with VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_pivot_points(df: pd.DataFrame) -> pd.Series:
    """
    Calculate simple pivot points
    
    Parameters:
    - df: DataFrame with OHLCV data
    
    Returns:
    - Series with pivot point values
    """
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    return pivot

def calculate_trend_intensity(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate trend intensity indicator
    
    Parameters:
    - df: DataFrame with OHLCV data
    - window: Window size for calculation
    
    Returns:
    - Series with trend intensity values (higher values indicate stronger trends)
    """
    direction = np.sign(df['close'].diff())
    trend_intensity = direction.rolling(window=window).apply(lambda x: abs(x.sum()) / window)
    return trend_intensity

def create_multi_timeframe_features(aligned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create features that capture relationships between timeframes
    
    Parameters:
    - aligned_data: Dictionary of aligned DataFrames for different timeframes
    
    Returns:
    - DataFrame with features from all timeframes
    """
    # Assume the smallest timeframe is the base
    timeframes = list(aligned_data.keys())
    timeframes.sort()  # Sort timeframes from smallest to largest
    
    base_tf = timeframes[0]
    base_data = aligned_data[base_tf].copy()
    
    # For each higher timeframe, add features to the base timeframe
    for tf in timeframes[1:]:
        # Add price relationship features
        base_data[f'close_ratio_{tf}_{base_tf}'] = (
            aligned_data[tf]['close'] / base_data['close']
        )
        
        # Add indicator relationships
        if 'sma_25' in aligned_data[tf].columns and 'sma_25' in base_data.columns:
            base_data[f'sma_cross_{tf}_{base_tf}'] = (
                (base_data['sma_25'] > base_data['close']).astype(int) - 
                (aligned_data[tf]['sma_25'] > aligned_data[tf]['close']).astype(int)
            )
        
        # Add RSI divergence
        if 'rsi_14' in aligned_data[tf].columns and 'rsi_14' in base_data.columns:
            # Price making higher highs but RSI making lower highs = bearish divergence
            base_data[f'rsi_divergence_{tf}_{base_tf}'] = calculate_divergence(
                base_data['close'], 
                base_data['rsi_14'],
                aligned_data[tf]['close'],
                aligned_data[tf]['rsi_14']
            )
    
    # Create consolidated trend features
    base_data['multi_tf_trend'] = calculate_multi_timeframe_trend(aligned_data)
    
    return base_data

def calculate_divergence(price_small_tf: pd.Series, 
                         indicator_small_tf: pd.Series,
                         price_large_tf: pd.Series,
                         indicator_large_tf: pd.Series,
                         window: int = 3) -> pd.Series:
    """
    Calculate divergence between price and indicator across timeframes
    
    Parameters:
    - price_small_tf: Price series for smaller timeframe
    - indicator_small_tf: Indicator series for smaller timeframe
    - price_large_tf: Price series for larger timeframe
    - indicator_large_tf: Indicator series for larger timeframe
    - window: Window size for local extrema detection
    
    Returns:
    - Series with divergence values (-1 for bearish, 0 for none, 1 for bullish)
    """
    # Simplified divergence calculation
    price_trend_small = price_small_tf.diff().rolling(window).mean().apply(np.sign)
    ind_trend_small = indicator_small_tf.diff().rolling(window).mean().apply(np.sign)
    
    price_trend_large = price_large_tf.diff().rolling(window).mean().apply(np.sign)
    ind_trend_large = indicator_large_tf.diff().rolling(window).mean().apply(np.sign)
    
    # Bearish: price up, indicator down
    bearish_div = ((price_trend_small > 0) & (ind_trend_small < 0) & 
                  (price_trend_large > 0) & (ind_trend_large < 0)).astype(int) * -1
    
    # Bullish: price down, indicator up
    bullish_div = ((price_trend_small < 0) & (ind_trend_small > 0) & 
                  (price_trend_large < 0) & (ind_trend_large > 0)).astype(int)
    
    return bearish_div + bullish_div

def calculate_multi_timeframe_trend(aligned_data: Dict[str, pd.DataFrame], 
                                   window: int = 14) -> pd.Series:
    """
    Calculate consolidated trend across multiple timeframes
    
    Parameters:
    - aligned_data: Dictionary of aligned DataFrames for different timeframes
    - window: Window size for trend calculation
    
    Returns:
    - Series with multi-timeframe trend values
    """
    timeframes = list(aligned_data.keys())
    base_tf = timeframes[0]
    
    # Start with base timeframe trend
    base_trend = aligned_data[base_tf]['close'].diff().rolling(window).mean().apply(np.sign)
    
    # Weight higher timeframes more
    weight_sum = 1.0
    weighted_trend = base_trend.copy()
    
    for i, tf in enumerate(timeframes[1:], 1):
        weight = (i + 1) * 1.0  # Higher timeframes get higher weights
        tf_trend = aligned_data[tf]['close'].diff().rolling(window).mean().apply(np.sign)
        weighted_trend += tf_trend * weight
        weight_sum += weight
    
    # Normalize
    weighted_trend = weighted_trend / weight_sum
    
    return weighted_trend

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

def calculate_multi_timeframe_signal(data_dict, primary_tf='15m', threshold_pct=0.01, 
                                     lookforward_periods={'15m': 12, '4h': 3, '1d': 1}):
    """
    Generate trading signals using information from multiple timeframes.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames for each timeframe
    primary_tf : str
        Primary timeframe to use for signal generation
    threshold_pct : float
        Minimum percentage move required to classify as up/down
    lookforward_periods : Dict[str, int]
        Number of periods to look forward for each timeframe
        
    Returns:
    --------
    pd.Series
        Series with trading signals aligned to the primary timeframe:
        1 = buy signal
        0 = hold signal
        -1 = sell signal
    """
    # Extract primary dataframe
    primary_df = data_dict[primary_tf]
    
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
        if tf == primary_tf:
            # Primary timeframe remains unchanged
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
            ratio_key = f"{primary_tf}_{tf}"
            reverse_ratio_key = f"{tf}_{primary_tf}"
            
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
                aligned_signals[tf] = resampled.fillna(method='ffill')
                
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