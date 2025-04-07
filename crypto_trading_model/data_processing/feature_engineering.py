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