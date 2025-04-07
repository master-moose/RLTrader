import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import random

def generate_trend_pattern(length: int, 
                          trend_type: str, 
                          noise_level: float = 0.05,
                          volatility_profile: str = "medium") -> pd.DataFrame:
    """
    Generate synthetic price data with trend pattern
    
    Parameters:
    - length: Number of candles to generate
    - trend_type: Type of trend ('uptrend', 'downtrend', 'sideways')
    - noise_level: Amount of noise to add
    - volatility_profile: Volatility characteristics ("low", "medium", "high")
    
    Returns:
    - DataFrame with OHLCV data containing the trend pattern
    """
    # Set random seed for reproducibility (can be removed in production)
    np.random.seed(None)
    
    # Define base parameters
    base_price = 100.0
    
    # Determine trend slope based on type
    if trend_type == 'uptrend':
        slope = np.random.uniform(0.1, 0.3)
    elif trend_type == 'downtrend':
        slope = np.random.uniform(-0.3, -0.1)
    else:  # sideways
        slope = np.random.uniform(-0.05, 0.05)
        
    # Adjust noise level based on volatility profile
    vol_factor = {
        "low": 0.5,
        "medium": 1.0,
        "high": 2.0
    }.get(volatility_profile, 1.0)
    
    adjusted_noise = noise_level * vol_factor
    
    # Generate time component
    time = np.arange(length)
    
    # Generate base trend
    trend = base_price + slope * time
    
    # Add noise
    noise = np.random.normal(0, adjusted_noise * base_price, length)
    close_prices = trend + noise
    
    # Generate realistic OHLC based on close prices
    data = []
    prev_close = close_prices[0]
    
    for i in range(length):
        # Generate realistic candle based on previous close
        close = close_prices[i]
        
        # Calculate high and low with greater range for higher volatility
        high_low_range = abs(close - prev_close) * (1 + np.random.uniform(0.5, 1.5) * vol_factor)
        
        if close > prev_close:
            # Bullish candle
            open_price = prev_close + np.random.uniform(-0.3, 0.3) * (close - prev_close)
            high = max(close, open_price) + np.random.uniform(0, 1) * high_low_range * 0.5
            low = min(close, open_price) - np.random.uniform(0, 1) * high_low_range * 0.2
        else:
            # Bearish candle
            open_price = prev_close + np.random.uniform(-0.3, 0.3) * (close - prev_close)
            high = max(close, open_price) + np.random.uniform(0, 1) * high_low_range * 0.2
            low = min(close, open_price) - np.random.uniform(0, 1) * high_low_range * 0.5
        
        # Generate volume
        if trend_type == 'uptrend':
            # Higher volume on up days
            volume_factor = 1 + 0.5 * (close > prev_close)
        elif trend_type == 'downtrend':
            # Higher volume on down days
            volume_factor = 1 + 0.5 * (close < prev_close)
        else:
            # Random volume for sideways
            volume_factor = np.random.uniform(0.7, 1.3)
        
        base_volume = np.random.uniform(800, 1200) * volume_factor
        volume = base_volume * (1 + vol_factor * np.random.uniform(-0.2, 0.2))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        prev_close = close
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add timestamp starting from now and going back
    end_time = pd.Timestamp.now().floor('min')
    timestamps = [end_time - pd.Timedelta(minutes=i) for i in range(length-1, -1, -1)]
    df['timestamp'] = timestamps
    df.set_index('timestamp', inplace=True)
    
    return df

def generate_reversal_pattern(length: int, 
                             pattern_type: str, 
                             noise_level: float = 0.05,
                             volume_profile: str = "increasing") -> pd.DataFrame:
    """
    Generate synthetic price data with reversal pattern
    
    Parameters:
    - length: Number of candles to generate
    - pattern_type: Type of reversal pattern ('double_top', 'head_shoulders', etc.)
    - noise_level: Amount of noise to add
    - volume_profile: Volume behavior during pattern formation
    
    Returns:
    - DataFrame with OHLCV data containing the reversal pattern
    """
    # Set random seed for reproducibility (can be removed in production)
    np.random.seed(None)
    
    # Base price and arrays
    base_price = 100.0
    close_prices = np.zeros(length)
    
    # Pattern specific generation
    if pattern_type == 'double_top':
        # Initial uptrend
        uptrend_length = length // 3
        for i in range(uptrend_length):
            close_prices[i] = base_price * (1 + 0.1 * i / uptrend_length)
        
        # First top
        first_top_idx = uptrend_length + length // 10
        top_height = close_prices[uptrend_length-1] * 1.05
        
        for i in range(uptrend_length, first_top_idx):
            progress = (i - uptrend_length) / (first_top_idx - uptrend_length)
            close_prices[i] = close_prices[uptrend_length-1] + (top_height - close_prices[uptrend_length-1]) * progress
        
        # Pullback
        pullback_idx = first_top_idx + length // 10
        pullback_low = top_height * 0.97
        
        for i in range(first_top_idx, pullback_idx):
            progress = (i - first_top_idx) / (pullback_idx - first_top_idx)
            close_prices[i] = top_height - (top_height - pullback_low) * progress
        
        # Second top
        second_top_idx = pullback_idx + length // 10
        
        for i in range(pullback_idx, second_top_idx):
            progress = (i - pullback_idx) / (second_top_idx - pullback_idx)
            close_prices[i] = pullback_low + (top_height - pullback_low) * progress
        
        # Breakdown
        for i in range(second_top_idx, length):
            progress = (i - second_top_idx) / (length - second_top_idx)
            target_low = pullback_low * 0.9
            close_prices[i] = top_height - (top_height - target_low) * progress
            
    elif pattern_type == 'head_shoulders':
        # Initial uptrend
        uptrend_length = length // 6
        for i in range(uptrend_length):
            close_prices[i] = base_price * (1 + 0.08 * i / uptrend_length)
        
        # Left shoulder
        left_shoulder_idx = uptrend_length + length // 12
        shoulder_height = close_prices[uptrend_length-1] * 1.03
        
        for i in range(uptrend_length, left_shoulder_idx):
            progress = (i - uptrend_length) / (left_shoulder_idx - uptrend_length)
            close_prices[i] = close_prices[uptrend_length-1] + (shoulder_height - close_prices[uptrend_length-1]) * progress
        
        # First trough
        first_trough_idx = left_shoulder_idx + length // 12
        trough_level = shoulder_height * 0.98
        
        for i in range(left_shoulder_idx, first_trough_idx):
            progress = (i - left_shoulder_idx) / (first_trough_idx - left_shoulder_idx)
            close_prices[i] = shoulder_height - (shoulder_height - trough_level) * progress
        
        # Head
        head_idx = first_trough_idx + length // 12
        head_height = shoulder_height * 1.05
        
        for i in range(first_trough_idx, head_idx):
            progress = (i - first_trough_idx) / (head_idx - first_trough_idx)
            close_prices[i] = trough_level + (head_height - trough_level) * progress
        
        # Second trough
        second_trough_idx = head_idx + length // 12
        
        for i in range(head_idx, second_trough_idx):
            progress = (i - head_idx) / (second_trough_idx - head_idx)
            close_prices[i] = head_height - (head_height - trough_level) * progress
        
        # Right shoulder
        right_shoulder_idx = second_trough_idx + length // 12
        
        for i in range(second_trough_idx, right_shoulder_idx):
            progress = (i - second_trough_idx) / (right_shoulder_idx - second_trough_idx)
            close_prices[i] = trough_level + (shoulder_height - trough_level) * progress
        
        # Breakdown
        neckline = trough_level
        for i in range(right_shoulder_idx, length):
            progress = (i - right_shoulder_idx) / (length - right_shoulder_idx)
            target_low = neckline * 0.92
            close_prices[i] = shoulder_height - (shoulder_height - target_low) * progress
    
    else:  # Default to V-reversal
        # Downtrend section
        mid_point = length // 2
        for i in range(mid_point):
            progress = i / mid_point
            close_prices[i] = base_price * (1 - 0.15 * progress)
        
        # Recovery section
        for i in range(mid_point, length):
            progress = (i - mid_point) / (length - mid_point)
            low_price = close_prices[mid_point-1]
            close_prices[i] = low_price * (1 + 0.1 * progress)
    
    # Add noise
    noise = np.random.normal(0, noise_level * base_price, length)
    close_prices = close_prices + noise
    
    # Generate OHLC data
    data = []
    prev_close = close_prices[0]
    
    for i in range(length):
        close = close_prices[i]
        
        # Calculate candle properties
        high_low_range = abs(close - prev_close) * (1 + np.random.uniform(0.5, 1.5))
        
        if close > prev_close:
            # Bullish candle
            open_price = prev_close + np.random.uniform(-0.3, 0.3) * (close - prev_close)
            high = max(close, open_price) + np.random.uniform(0, 1) * high_low_range * 0.5
            low = min(close, open_price) - np.random.uniform(0, 1) * high_low_range * 0.2
        else:
            # Bearish candle
            open_price = prev_close + np.random.uniform(-0.3, 0.3) * (close - prev_close)
            high = max(close, open_price) + np.random.uniform(0, 1) * high_low_range * 0.2
            low = min(close, open_price) - np.random.uniform(0, 1) * high_low_range * 0.5
        
        # Generate volume based on profile
        if volume_profile == "increasing":
            # Volume increases near key points (tops, bottoms)
            if pattern_type == 'double_top':
                if i == first_top_idx or i == second_top_idx:
                    volume_factor = 1.5
                elif i > second_top_idx:
                    volume_factor = 1.3  # Increased volume on breakdown
                else:
                    volume_factor = 1.0
            elif pattern_type == 'head_shoulders':
                if i == left_shoulder_idx or i == head_idx or i == right_shoulder_idx:
                    volume_factor = 1.3
                elif i > right_shoulder_idx:
                    volume_factor = 1.5  # Increased volume on breakdown
                else:
                    volume_factor = 1.0
            else:  # V-reversal
                if i == mid_point:
                    volume_factor = 2.0  # High volume at reversal point
                else:
                    volume_factor = 1.0
        else:
            # Random volume
            volume_factor = np.random.uniform(0.8, 1.2)
        
        base_volume = np.random.uniform(800, 1200) * volume_factor
        volume = base_volume
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        prev_close = close
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add timestamp starting from now and going back
    end_time = pd.Timestamp.now().floor('min')
    timestamps = [end_time - pd.Timedelta(minutes=i) for i in range(length-1, -1, -1)]
    df['timestamp'] = timestamps
    df.set_index('timestamp', inplace=True)
    
    return df

def generate_support_resistance_reaction(length: int, 
                                        reaction_type: str, 
                                        strength: str = "strong",
                                        noise_level: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic price data reacting to support/resistance
    
    Parameters:
    - length: Number of candles to generate
    - reaction_type: Type of reaction ('bounce', 'breakout')
    - strength: Strength of support/resistance level
    - noise_level: Amount of noise to add
    
    Returns:
    - DataFrame with OHLCV data showing support/resistance reaction
    """
    # Set random seed for reproducibility (can be removed in production)
    np.random.seed(None)
    
    # Base price and arrays
    base_price = 100.0
    close_prices = np.zeros(length)
    
    # Pre-reaction phase (first third)
    pre_length = length // 3
    
    # Set initial trend direction (approaching support/resistance)
    if reaction_type == 'bounce':
        # Approaching support (downtrend)
        for i in range(pre_length):
            progress = i / pre_length
            close_prices[i] = base_price * (1 - 0.1 * progress)
    else:  # breakout
        # Approaching resistance (uptrend)
        for i in range(pre_length):
            progress = i / pre_length
            close_prices[i] = base_price * (1 + 0.1 * progress)
    
    # Establish support/resistance level
    if reaction_type == 'bounce':
        level = close_prices[pre_length-1] * 0.98  # Support slightly below
    else:
        level = close_prices[pre_length-1] * 1.02  # Resistance slightly above
    
    # Reaction phase (testing support/resistance)
    reaction_length = length // 3
    reaction_end = pre_length + reaction_length
    
    # Set strength parameters
    strength_params = {
        "weak": 0.7,
        "medium": 1.0,
        "strong": 1.3
    }
    strength_factor = strength_params.get(strength, 1.0)
    
    # Generate testing of the level
    if reaction_type == 'bounce':
        # Test support with small movements around it
        for i in range(pre_length, reaction_end):
            test_intensity = np.random.uniform(0, 0.02)
            close_prices[i] = level * (1 + test_intensity - 0.01)
    else:  # breakout
        # Test resistance with small movements around it
        for i in range(pre_length, reaction_end):
            test_intensity = np.random.uniform(0, 0.02)
            close_prices[i] = level * (1 - test_intensity + 0.01)
    
    # Post-reaction phase (final third)
    if reaction_type == 'bounce':
        # Bounce from support
        for i in range(reaction_end, length):
            progress = (i - reaction_end) / (length - reaction_end)
            bounce_height = level * 0.1 * strength_factor
            close_prices[i] = level + bounce_height * progress
    else:  # breakout
        # Breakout above resistance
        for i in range(reaction_end, length):
            progress = (i - reaction_end) / (length - reaction_end)
            breakout_height = level * 0.1 * strength_factor
            close_prices[i] = level + breakout_height * progress
    
    # Add noise
    noise = np.random.normal(0, noise_level * base_price, length)
    close_prices = close_prices + noise
    
    # Generate OHLC data
    data = []
    prev_close = close_prices[0]
    
    for i in range(length):
        close = close_prices[i]
        
        # Calculate candle properties
        high_low_range = abs(close - prev_close) * (1 + np.random.uniform(0.5, 1.5))
        
        # Create different candle types based on phase
        if i < pre_length:  # Approach phase
            # Normal candles
            if close > prev_close:
                open_price = prev_close + np.random.uniform(-0.3, 0.3) * (close - prev_close)
                high = max(close, open_price) + np.random.uniform(0, 1) * high_low_range * 0.5
                low = min(close, open_price) - np.random.uniform(0, 1) * high_low_range * 0.2
            else:
                open_price = prev_close + np.random.uniform(-0.3, 0.3) * (close - prev_close)
                high = max(close, open_price) + np.random.uniform(0, 1) * high_low_range * 0.2
                low = min(close, open_price) - np.random.uniform(0, 1) * high_low_range * 0.5
        
        elif i < reaction_end:  # Testing phase
            # Create wicks that test the level
            if reaction_type == 'bounce':
                # Testing support - longer lower wicks
                open_price = close + np.random.uniform(0, 0.5) * (close - level)
                high = max(close, open_price) + np.random.uniform(0, 0.5) * high_low_range
                low = level - np.random.uniform(0, 0.5) * (close - level) * 0.5
            else:
                # Testing resistance - longer upper wicks
                open_price = close - np.random.uniform(0, 0.5) * (level - close)
                high = level + np.random.uniform(0, 0.5) * (level - close) * 0.5
                low = min(close, open_price) - np.random.uniform(0, 0.5) * high_low_range
        
        else:  # Post-reaction phase
            # Strong reaction candles
            if reaction_type == 'bounce':
                # Bullish candles after bounce
                open_price = close * (1 - np.random.uniform(0, 0.01))
                high = close * (1 + np.random.uniform(0, 0.01))
                low = close * (1 - np.random.uniform(0.01, 0.02))
            else:
                # Bullish candles after breakout
                open_price = close * (1 - np.random.uniform(0, 0.01))
                high = close * (1 + np.random.uniform(0.01, 0.02))
                low = close * (1 - np.random.uniform(0, 0.01))
        
        # Adjust high and low to maintain proper order
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        if i < pre_length:
            volume_factor = 1.0
        elif i < reaction_end:
            # Increased volume during testing
            volume_factor = 1.2
        else:
            # Even higher volume during reaction
            volume_factor = 1.5 * strength_factor
        
        base_volume = np.random.uniform(800, 1200) * volume_factor
        volume = base_volume
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        prev_close = close
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add timestamp starting from now and going back
    end_time = pd.Timestamp.now().floor('min')
    timestamps = [end_time - pd.Timedelta(minutes=i) for i in range(length-1, -1, -1)]
    df['timestamp'] = timestamps
    df.set_index('timestamp', inplace=True)
    
    return df

def add_realistic_noise(price_data: pd.DataFrame, noise_profile: str = "market_like") -> pd.DataFrame:
    """
    Add realistic market-like noise to synthetic price data
    
    Parameters:
    - price_data: Clean synthetic price data
    - noise_profile: Type of noise to add ("market_like", "choppy", "trending")
    
    Returns:
    - DataFrame with realistic market noise added
    """
    # Create a copy to avoid modifying the original
    df = price_data.copy()
    
    # Adjust noise parameters based on profile
    if noise_profile == "market_like":
        base_noise = 0.005
        volatility_clustering = True
        mean_reversion = 0.3
    elif noise_profile == "choppy":
        base_noise = 0.008
        volatility_clustering = True
        mean_reversion = 0.7
    elif noise_profile == "trending":
        base_noise = 0.006
        volatility_clustering = True
        mean_reversion = 0.1
    else:
        base_noise = 0.005
        volatility_clustering = False
        mean_reversion = 0.3
    
    # Get price columns
    length = len(df)
    
    # Generate base noise
    noise = np.random.normal(0, base_noise, length)
    
    # Add volatility clustering (GARCH-like effect)
    if volatility_clustering:
        volatility = np.ones(length) * base_noise
        for i in range(1, length):
            # Volatility depends on previous volatility and previous absolute return
            volatility[i] = 0.7 * volatility[i-1] + 0.3 * abs(noise[i-1])
        
        # Normalize and apply clustered volatility
        volatility = volatility / np.mean(volatility) * base_noise
        noise = np.random.normal(0, volatility)
    
    # Add mean reversion tendency
    if mean_reversion > 0:
        for i in range(1, length):
            noise[i] = noise[i] - mean_reversion * noise[i-1]
    
    # Apply noise while maintaining OHLC relationships
    for i in range(length):
        # Calculate noise scale factor based on current price level
        scale_factor = df.iloc[i]['close'] * 1.0
        
        # Apply noise to close price
        close_noise = noise[i] * scale_factor
        df.iloc[i, df.columns.get_indexer(['close'])[0]] += close_noise
        
        # Adjust open with related but different noise
        open_noise = noise[i] * 0.8 * scale_factor  # Slightly less noise for open
        df.iloc[i, df.columns.get_indexer(['open'])[0]] += open_noise
        
        # Adjust high and low to maintain proper relationships
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        current_open = df.iloc[i]['open']
        current_close = df.iloc[i]['close']
        
        # Ensure high is at least as high as the higher of open and close
        new_high = max(current_high, current_open, current_close)
        # Add some extra movement to high
        new_high += abs(np.random.normal(0, base_noise * scale_factor * 0.5))
        
        # Ensure low is at most as low as the lower of open and close
        new_low = min(current_low, current_open, current_close)
        # Add some extra movement to low
        new_low -= abs(np.random.normal(0, base_noise * scale_factor * 0.5))
        
        df.iloc[i, df.columns.get_indexer(['high'])[0]] = new_high
        df.iloc[i, df.columns.get_indexer(['low'])[0]] = new_low
    
    # Ensure all prices are positive
    for col in ['open', 'high', 'low', 'close']:
        df[col] = np.maximum(df[col], 0.01)  # Ensure no negative or zero prices
    
    return df

def create_realistic_volume_profile(price_data: pd.DataFrame, pattern_type: str) -> pd.Series:
    """
    Create realistic volume profile based on price pattern
    
    Parameters:
    - price_data: Synthetic price data
    - pattern_type: Type of pattern in the data
    
    Returns:
    - Series with volume data that matches price behavior
    """
    # Create a copy to avoid modifying the original
    df = price_data.copy()
    length = len(df)
    
    # Initialize base volume
    base_volume = np.random.uniform(800, 1200, length)
    
    # Apply specific volume profiles based on pattern type
    if pattern_type == "uptrend":
        # Higher volume on up days, fading volume on pullbacks
        for i in range(1, length):
            if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                # Up day - higher volume
                base_volume[i] *= np.random.uniform(1.1, 1.3)
            else:
                # Down day - lower volume
                base_volume[i] *= np.random.uniform(0.7, 0.9)
                
    elif pattern_type == "downtrend":
        # Higher volume on down days, lower volume on bounces
        for i in range(1, length):
            if df.iloc[i]['close'] < df.iloc[i-1]['close']:
                # Down day - higher volume
                base_volume[i] *= np.random.uniform(1.1, 1.3)
            else:
                # Up day - lower volume
                base_volume[i] *= np.random.uniform(0.7, 0.9)
                
    elif pattern_type == "double_top" or pattern_type == "head_shoulders":
        # Find high points and create realistic volume profiles
        highs = []
        for i in range(1, length-1):
            if df.iloc[i]['close'] > df.iloc[i-1]['close'] and df.iloc[i]['close'] > df.iloc[i+1]['close']:
                highs.append(i)
        
        # Higher volume at key points
        for i in range(length):
            # Higher volume near high points
            for high_idx in highs:
                dist = abs(i - high_idx)
                if dist < 3:
                    base_volume[i] *= np.random.uniform(1.2, 1.5)
            
            # Higher volume on breakdown
            if i > highs[-1] and df.iloc[i]['close'] < df.iloc[i-1]['close']:
                base_volume[i] *= np.random.uniform(1.3, 1.6)
                
    elif pattern_type == "bounce" or pattern_type == "breakout":
        # Find the support/resistance level
        mid_point = length // 2
        
        # Volume increases during tests of support/resistance
        for i in range(mid_point - 5, mid_point + 5):
            if 0 <= i < length:
                base_volume[i] *= np.random.uniform(1.1, 1.4)
        
        # Volume surge on breakout/bounce
        for i in range(mid_point + 5, length):
            if pattern_type == "breakout" and df.iloc[i]['close'] > df.iloc[i-1]['close']:
                base_volume[i] *= np.random.uniform(1.3, 1.8)
            elif pattern_type == "bounce" and df.iloc[i]['close'] > df.iloc[i-1]['close']:
                base_volume[i] *= np.random.uniform(1.2, 1.6)
    
    # Add daily volume patterns (U-shaped volume)
    for i in range(length):
        time_of_day = i % 24  # Simulate 24-hour cycle
        
        # Higher volume at market open and close (U-shaped)
        if time_of_day < 2 or time_of_day > 21:
            base_volume[i] *= np.random.uniform(1.1, 1.3)
        elif 10 <= time_of_day <= 14:
            base_volume[i] *= np.random.uniform(0.8, 0.9)  # Lower mid-day
    
    # Add clustering and momentum in volume
    for i in range(1, length):
        # Volume tends to follow previous volume (autocorrelation)
        momentum = 0.3  # 30% of previous volume change affects current volume
        prev_change = base_volume[i] / base_volume[i-1] - 1
        base_volume[i] *= (1 + momentum * prev_change)
    
    # Add some random spikes (news events, etc.)
    num_spikes = max(1, length // 30)
    spike_indices = np.random.choice(range(length), size=num_spikes, replace=False)
    
    for idx in spike_indices:
        base_volume[idx] *= np.random.uniform(1.5, 3.0)
    
    # Ensure no negative volume
    base_volume = np.maximum(base_volume, 1)
    
    return pd.Series(base_volume) 