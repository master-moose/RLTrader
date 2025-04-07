import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_multi_timeframe_data(pattern_params: Dict, 
                                 timeframes: List[str] = ["15m", "4h", "1d"]) -> Dict[str, pd.DataFrame]:
    """
    Generate consistent synthetic data across multiple timeframes
    
    Parameters:
    - pattern_params: Parameters for pattern generation
    - timeframes: List of timeframes to generate
    
    Returns:
    - Dictionary of DataFrames for each timeframe with consistent patterns
    """
    from .pattern_generator import generate_trend_pattern, add_realistic_noise
    
    # Sort timeframes by granularity (smallest first)
    def get_minutes(tf):
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 60 * 24
        return 0
    
    sorted_timeframes = sorted(timeframes, key=get_minutes)
    base_tf = sorted_timeframes[0]  # Smallest timeframe
    
    # Calculate required base timeframe length to generate all higher timeframes
    base_length = pattern_params.get('length', 200)
    max_tf_minutes = get_minutes(sorted_timeframes[-1])  # Largest timeframe
    base_tf_minutes = get_minutes(base_tf)  # Smallest timeframe
    required_length = base_length * (max_tf_minutes // base_tf_minutes) + 50  # Add margin
    
    logger.info(f"Generating base timeframe {base_tf} with {required_length} candles")
    
    # Generate the base timeframe data (highest resolution)
    base_pattern_type = pattern_params.get('pattern_type', 'trend')
    base_params = pattern_params.copy()
    base_params['length'] = required_length
    
    # Import pattern generators based on pattern type
    if base_pattern_type == 'trend':
        from .pattern_generator import generate_trend_pattern as generator
        base_data = generator(
            length=required_length,
            trend_type=base_params.get('trend_type', 'uptrend'),
            noise_level=base_params.get('noise_level', 0.05),
            volatility_profile=base_params.get('volatility_profile', 'medium')
        )
    elif base_pattern_type == 'reversal':
        from .pattern_generator import generate_reversal_pattern as generator
        base_data = generator(
            length=required_length,
            pattern_type=base_params.get('reversal_type', 'double_top'),
            noise_level=base_params.get('noise_level', 0.05),
            volume_profile=base_params.get('volume_profile', 'increasing')
        )
    elif base_pattern_type == 'support_resistance':
        from .pattern_generator import generate_support_resistance_reaction as generator
        base_data = generator(
            length=required_length,
            reaction_type=base_params.get('reaction_type', 'bounce'),
            strength=base_params.get('strength', 'strong'),
            noise_level=base_params.get('noise_level', 0.05)
        )
    else:
        logger.warning(f"Unknown pattern type: {base_pattern_type}, defaulting to uptrend")
        from .pattern_generator import generate_trend_pattern as generator
        base_data = generator(
            length=required_length,
            trend_type='uptrend',
            noise_level=base_params.get('noise_level', 0.05)
        )
    
    # Initialize result dictionary with base timeframe
    result = {base_tf: base_data.iloc[-base_length:].copy()}  # Take only the recent portion
    
    # Generate higher timeframes by aggregating the base timeframe
    for tf in sorted_timeframes[1:]:
        logger.info(f"Generating higher timeframe {tf} from base {base_tf}")
        
        # Calculate aggregation factor
        tf_minutes = get_minutes(tf)
        base_tf_minutes = get_minutes(base_tf)
        agg_factor = tf_minutes // base_tf_minutes
        
        if agg_factor <= 0:
            logger.warning(f"Invalid aggregation factor for {tf}, skipping")
            continue
        
        # Resample to higher timeframe
        # For timestamp, use the last timestamp of each group
        resampled_data = aggregate_ohlcv(base_data, agg_factor)
        
        # Ensure we have the right number of candles
        target_length = base_length // agg_factor
        if len(resampled_data) > target_length:
            resampled_data = resampled_data.iloc[-target_length:].copy()
        
        # Add to result
        result[tf] = resampled_data
    
    return result

def aggregate_ohlcv(df: pd.DataFrame, agg_factor: int) -> pd.DataFrame:
    """
    Aggregate OHLCV data to a higher timeframe
    
    Parameters:
    - df: DataFrame with OHLCV data
    - agg_factor: Aggregation factor (number of base candles per higher timeframe candle)
    
    Returns:
    - Aggregated DataFrame
    """
    if agg_factor <= 1:
        return df.copy()
    
    # Create groups
    groups = [df.iloc[i:i+agg_factor] for i in range(0, len(df), agg_factor) if i+agg_factor <= len(df)]
    
    aggregated_data = []
    for group in groups:
        if len(group) == agg_factor:  # Ensure we have a complete group
            agg_candle = {
                'open': group.iloc[0]['open'],
                'high': group['high'].max(),
                'low': group['low'].min(),
                'close': group.iloc[-1]['close'],
                'volume': group['volume'].sum()
            }
            # Use the timestamp of the last candle in the group
            aggregated_data.append({**agg_candle, 'timestamp': group.index[-1]})
    
    if not aggregated_data:
        return pd.DataFrame()
    
    result = pd.DataFrame(aggregated_data)
    result.set_index('timestamp', inplace=True)
    
    return result

def ensure_pattern_visibility(multi_tf_data: Dict[str, pd.DataFrame], 
                             pattern_type: str,
                             primary_timeframe: str) -> Dict[str, pd.DataFrame]:
    """
    Ensure pattern is appropriately visible across timeframes
    
    Parameters:
    - multi_tf_data: Dictionary of DataFrames for different timeframes
    - pattern_type: Type of pattern generated
    - primary_timeframe: Timeframe where pattern should be most obvious
    
    Returns:
    - Modified multi-timeframe data with properly visible patterns
    """
    if primary_timeframe not in multi_tf_data:
        logger.warning(f"Primary timeframe {primary_timeframe} not found in data")
        return multi_tf_data
    
    # Make a copy to avoid modifying the original
    result = {tf: df.copy() for tf, df in multi_tf_data.items()}
    
    # Process according to pattern type
    if pattern_type in ['double_top', 'head_shoulders']:
        # For reversal patterns, ensure the pattern is clearly visible in primary timeframe
        # and also visible but less pronounced in other timeframes
        primary_df = result[primary_timeframe]
        
        # Enhance the pattern in the primary timeframe
        enhance_reversal_pattern(primary_df, pattern_type, enhancement_factor=1.2)
        
        # Scale down the pattern in other timeframes
        for tf, df in result.items():
            if tf != primary_timeframe:
                # Determine if higher or lower timeframe
                tf_minutes = get_timeframe_minutes(tf)
                primary_minutes = get_timeframe_minutes(primary_timeframe)
                
                if tf_minutes > primary_minutes:
                    # Higher timeframe - make pattern more subtle
                    enhance_factor = 0.7
                else:
                    # Lower timeframe - keep pattern somewhat visible but noisy
                    enhance_factor = 0.9
                
                enhance_reversal_pattern(df, pattern_type, enhancement_factor=enhance_factor)
    
    elif pattern_type in ['bounce', 'breakout']:
        # For support/resistance reactions, ensure clear levels across timeframes
        align_support_resistance_levels(result, pattern_type, primary_timeframe)
    
    elif pattern_type == 'uptrend' or pattern_type == 'downtrend':
        # For trends, ensure consistent trend direction with varying strength
        align_trend_strength(result, pattern_type, primary_timeframe)
    
    return result

def enhance_reversal_pattern(df: pd.DataFrame, pattern_type: str, enhancement_factor: float = 1.0) -> None:
    """
    Enhance or diminish a reversal pattern in place
    
    Parameters:
    - df: DataFrame to modify
    - pattern_type: Type of pattern ('double_top', 'head_shoulders')
    - enhancement_factor: Factor to enhance (>1) or diminish (<1) the pattern
    """
    if pattern_type == 'double_top':
        # Find the likely tops
        length = len(df)
        first_half = df.iloc[:length//2]
        second_half = df.iloc[length//2:]
        
        first_top_idx = first_half['close'].idxmax()
        second_top_idx = second_half['close'].idxmax()
        
        if first_top_idx and second_top_idx:
            # Calculate the average top height
            avg_top_height = (df.loc[first_top_idx, 'close'] + df.loc[second_top_idx, 'close']) / 2
            
            # Find the trough between the tops
            trough_section = df.loc[first_top_idx:second_top_idx]
            if not trough_section.empty:
                trough_idx = trough_section['close'].idxmin()
                
                # Enhance or diminish the pattern
                if enhancement_factor != 1.0:
                    # Adjust the height of the tops
                    df.loc[first_top_idx, 'close'] *= enhancement_factor
                    df.loc[second_top_idx, 'close'] *= enhancement_factor
                    
                    # Ensure OHLC integrity
                    df.loc[first_top_idx, 'high'] = max(df.loc[first_top_idx, 'high'], df.loc[first_top_idx, 'close'])
                    df.loc[second_top_idx, 'high'] = max(df.loc[second_top_idx, 'high'], df.loc[second_top_idx, 'close'])
                    
                    # Optionally adjust the trough
                    if enhancement_factor > 1:
                        # Deeper trough for enhanced pattern
                        df.loc[trough_idx, 'close'] *= (2 - enhancement_factor)
                        df.loc[trough_idx, 'low'] = min(df.loc[trough_idx, 'low'], df.loc[trough_idx, 'close'])
    
    elif pattern_type == 'head_shoulders':
        # Find the head and shoulders
        length = len(df)
        first_third = df.iloc[:length//3]
        middle_third = df.iloc[length//3:2*length//3]
        last_third = df.iloc[2*length//3:]
        
        left_shoulder_idx = first_third['close'].idxmax()
        head_idx = middle_third['close'].idxmax()
        right_shoulder_idx = last_third['close'].idxmax()
        
        if left_shoulder_idx and head_idx and right_shoulder_idx:
            # Calculate the average shoulder height
            avg_shoulder_height = (df.loc[left_shoulder_idx, 'close'] + df.loc[right_shoulder_idx, 'close']) / 2
            
            # Enhance or diminish the pattern
            if enhancement_factor != 1.0:
                # Adjust the height of the head relative to shoulders
                if enhancement_factor > 1:
                    # Head becomes relatively higher
                    df.loc[head_idx, 'close'] *= enhancement_factor
                    df.loc[head_idx, 'high'] = max(df.loc[head_idx, 'high'], df.loc[head_idx, 'close'])
                else:
                    # Head becomes relatively closer to shoulders
                    adjustment = 1 - (1 - enhancement_factor) * 0.5
                    df.loc[head_idx, 'close'] *= adjustment
                    # Ensure shoulders are roughly equal height
                    shoulder_target = avg_shoulder_height
                    df.loc[left_shoulder_idx, 'close'] = shoulder_target
                    df.loc[right_shoulder_idx, 'close'] = shoulder_target

def align_support_resistance_levels(multi_tf_data: Dict[str, pd.DataFrame], 
                                   pattern_type: str, 
                                   primary_timeframe: str) -> None:
    """
    Align support/resistance levels across timeframes
    
    Parameters:
    - multi_tf_data: Dictionary of DataFrames for different timeframes
    - pattern_type: Type of pattern ('bounce', 'breakout')
    - primary_timeframe: Timeframe where pattern should be most obvious
    """
    if primary_timeframe not in multi_tf_data:
        return
    
    # Get the key level from primary timeframe
    primary_df = multi_tf_data[primary_timeframe]
    length = len(primary_df)
    mid_point = length // 2
    
    # Find the support/resistance level
    if pattern_type == 'bounce':
        # For a bounce, the level is near the lows
        level_section = primary_df.iloc[mid_point-5:mid_point+5]
        level = level_section['low'].min()
    else:  # breakout
        # For a breakout, the level is near the highs
        level_section = primary_df.iloc[mid_point-5:mid_point+5]
        level = level_section['high'].max()
    
    # Calculate the price ratio for each timeframe
    base_price = primary_df['close'].mean()
    
    # Align levels across timeframes
    for tf, df in multi_tf_data.items():
        if tf != primary_timeframe:
            tf_price = df['close'].mean()
            price_ratio = tf_price / base_price
            
            # Calculate the equivalent level for this timeframe
            tf_level = level * price_ratio
            
            # Adjust prices to create a clear level
            length = len(df)
            mid_point = length // 2
            
            # Find section to adjust
            adjust_start = max(0, mid_point - 5)
            adjust_end = min(length, mid_point + 5)
            
            if pattern_type == 'bounce':
                # Create a clear support level
                for i in range(adjust_start, adjust_end):
                    # Adjust lows to create a clear support
                    closeness = 1 - min(1, abs(i - mid_point) / 5)  # 1 at midpoint, 0 at edges
                    target_low = tf_level * (1 + 0.01 * (1 - closeness))
                    
                    # Only adjust if it would create a clearer level
                    if df.iloc[i]['low'] > target_low:
                        df.iloc[i]['low'] = target_low
                        
                        # Ensure open/close are consistent
                        if df.iloc[i]['close'] < target_low:
                            df.iloc[i]['close'] = target_low * (1 + 0.005)
                        if df.iloc[i]['open'] < target_low:
                            df.iloc[i]['open'] = target_low * (1 + 0.005)
            
            else:  # breakout
                # Create a clear resistance level
                for i in range(adjust_start, adjust_end):
                    # Adjust highs to create a clear resistance
                    closeness = 1 - min(1, abs(i - mid_point) / 5)  # 1 at midpoint, 0 at edges
                    target_high = tf_level * (1 - 0.01 * (1 - closeness))
                    
                    # Only adjust if it would create a clearer level
                    if df.iloc[i]['high'] < target_high:
                        df.iloc[i]['high'] = target_high
                        
                        # Ensure open/close are consistent
                        if df.iloc[i]['close'] > target_high:
                            df.iloc[i]['close'] = target_high * (1 - 0.005)
                        if df.iloc[i]['open'] > target_high:
                            df.iloc[i]['open'] = target_high * (1 - 0.005)

def align_trend_strength(multi_tf_data: Dict[str, pd.DataFrame], 
                        trend_type: str, 
                        primary_timeframe: str) -> None:
    """
    Align trend strength across timeframes
    
    Parameters:
    - multi_tf_data: Dictionary of DataFrames for different timeframes
    - trend_type: Type of trend ('uptrend', 'downtrend')
    - primary_timeframe: Timeframe where trend should be most obvious
    """
    if primary_timeframe not in multi_tf_data:
        return
    
    # Calculate the trend strength in the primary timeframe
    primary_df = multi_tf_data[primary_timeframe]
    primary_returns = primary_df['close'].pct_change().dropna()
    
    # Determine trend direction
    if trend_type == 'uptrend':
        target_strength = 0.2  # 20% return over the period
        direction = 1
    else:  # downtrend
        target_strength = -0.2  # -20% return over the period
        direction = -1
    
    # Calculate required adjustment to match target strength
    primary_strength = (primary_df['close'].iloc[-1] / primary_df['close'].iloc[0] - 1)
    adjustment_factor = (1 + target_strength) / (1 + primary_strength) if primary_strength != -1 else 1
    
    # Apply adjustments to all timeframes
    for tf, df in multi_tf_data.items():
        # Calculate trend strength for this timeframe
        tf_strength = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
        
        # Adjust based on timeframe relationship to primary
        tf_minutes = get_timeframe_minutes(tf)
        primary_minutes = get_timeframe_minutes(primary_timeframe)
        
        # Higher timeframes should have smoother, more consistent trends
        # Lower timeframes should be more volatile but maintain the overall trend
        if tf_minutes > primary_minutes:
            # Higher timeframe - smoother trend
            smoothness = 0.8
            volatility = 0.5
        elif tf == primary_timeframe:
            # Primary timeframe - baseline
            smoothness = 1.0
            volatility = 1.0
        else:
            # Lower timeframe - more volatile
            smoothness = 1.2
            volatility = 1.5
        
        # Apply adjustment
        length = len(df)
        for i in range(1, length):
            progress = i / (length - 1)  # 0 to 1 over the timeframe
            
            # Adjust close price to match desired trend strength
            # Use logarithmic scaling for more realistic price movements
            if trend_type == 'uptrend':
                target_return = np.exp(target_strength * progress * smoothness) - 1
            else:  # downtrend
                target_return = np.exp(-target_strength * progress * smoothness) - 1
                target_return = -target_return  # Convert to negative
            
            base_price = df['close'].iloc[0]
            target_price = base_price * (1 + target_return)
            
            # Add volatility around the trend line
            volatility_factor = 1 + np.random.normal(0, 0.005) * volatility
            df.iloc[i]['close'] = target_price * volatility_factor
            
            # Ensure OHLC integrity
            high_low_range = abs(df.iloc[i]['close'] - df.iloc[i-1]['close']) * (0.2 + 0.2 * volatility)
            
            if df.iloc[i]['close'] > df.iloc[i-1]['close']:
                # Bullish candle
                df.iloc[i]['open'] = df.iloc[i-1]['close'] * (1 + np.random.uniform(-0.3, 0.3) * 0.01)
                df.iloc[i]['high'] = max(df.iloc[i]['close'], df.iloc[i]['open']) + np.random.uniform(0, 1) * high_low_range
                df.iloc[i]['low'] = min(df.iloc[i]['close'], df.iloc[i]['open']) - np.random.uniform(0, 1) * high_low_range * 0.3
            else:
                # Bearish candle
                df.iloc[i]['open'] = df.iloc[i-1]['close'] * (1 + np.random.uniform(-0.3, 0.3) * 0.01)
                df.iloc[i]['high'] = max(df.iloc[i]['close'], df.iloc[i]['open']) + np.random.uniform(0, 1) * high_low_range * 0.3
                df.iloc[i]['low'] = min(df.iloc[i]['close'], df.iloc[i]['open']) - np.random.uniform(0, 1) * high_low_range

def create_timeframe_confluences(multi_tf_data: Dict[str, pd.DataFrame], 
                                confluence_type: str) -> Dict[str, pd.DataFrame]:
    """
    Create confluences between different timeframes
    
    Parameters:
    - multi_tf_data: Dictionary of DataFrames for different timeframes
    - confluence_type: Type of confluence to create
    
    Returns:
    - Modified multi-timeframe data with specific confluences
    """
    # Make a copy to avoid modifying the original
    result = {tf: df.copy() for tf, df in multi_tf_data.items()}
    
    # Get timeframes sorted by granularity
    timeframes = list(result.keys())
    
    # Sort timeframes by granularity (smallest to largest)
    def get_minutes(tf):
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 60 * 24
        return 0
    
    timeframes.sort(key=get_minutes)
    
    # Generate indicators if not present
    for tf, df in result.items():
        if 'rsi_14' not in df.columns or 'macd' not in df.columns:
            from .indicator_engineering import calculate_realistic_indicators
            result[tf] = calculate_realistic_indicators(df)
    
    # Apply specific confluences
    if confluence_type == 'multi_tf_bullish_divergence':
        # Create bullish RSI divergence on higher timeframe with price confirmation on lower
        if len(timeframes) >= 2:
            # Use the two highest timeframes
            higher_tf = timeframes[-1]
            lower_tf = timeframes[-2]
            
            # Create bullish divergence on higher timeframe
            higher_df = result[higher_tf]
            length = len(higher_df)
            mid_point = length // 2
            
            # Find a significant low point in the first half
            first_half = higher_df.iloc[:mid_point]
            first_low_idx = first_half['close'].idxmin()
            
            # Find a low point in the second half
            second_half = higher_df.iloc[mid_point:]
            second_low_idx = second_half['close'].idxmin()
            
            if first_low_idx is not None and second_low_idx is not None:
                # Make the second price low lower than the first
                price_adjustment = higher_df.loc[first_low_idx, 'close'] * 0.05  # 5% lower
                higher_df.loc[second_low_idx, 'close'] = higher_df.loc[first_low_idx, 'close'] - price_adjustment
                
                # But make the second RSI higher than the first (divergence)
                higher_df.loc[second_low_idx, 'rsi_14'] = higher_df.loc[first_low_idx, 'rsi_14'] + 10
                
                # Now create confirmation on lower timeframe
                lower_df = result[lower_tf]
                
                # Find the corresponding section in the lower timeframe
                # Assuming the index is datetime
                if isinstance(second_low_idx, pd.Timestamp):
                    # Find a window around the higher timeframe signal
                    window_start = second_low_idx - pd.Timedelta(hours=4)
                    window_end = second_low_idx + pd.Timedelta(hours=4)
                    
                    # Get the lower timeframe section
                    lower_section = lower_df.loc[window_start:window_end]
                    
                    if not lower_section.empty:
                        # Create a bullish confirmation in the lower timeframe
                        # by making the price form a double bottom with higher lows
                        for i in range(len(lower_section) - 1):
                            if i > len(lower_section) // 2:
                                # Second half - create higher low with strong bullish candle
                                if lower_section.iloc[i]['close'] < lower_section.iloc[i]['open']:
                                    # Convert bearish to bullish
                                    lower_section.iloc[i]['close'] = lower_section.iloc[i]['open'] * 1.02
                                    lower_section.iloc[i]['high'] = max(lower_section.iloc[i]['high'], 
                                                                       lower_section.iloc[i]['close'] * 1.005)
    
    elif confluence_type == 'multi_tf_support':
        # Create support level confirmation across timeframes
        if len(timeframes) >= 2:
            # Start with the primary (middle) timeframe
            primary_idx = len(timeframes) // 2
            primary_tf = timeframes[primary_idx]
            primary_df = result[primary_tf]
            
            # Find a potential support level
            primary_low = primary_df['low'].min()
            primary_close_mean = primary_df['close'].mean()
            support_level = primary_low * 1.02  # Slightly above absolute low
            
            # Create a support test in the primary timeframe
            length = len(primary_df)
            mid_point = length // 2
            
            # Create a clear support test with a bounce
            for i in range(mid_point - 3, mid_point + 3):
                if i >= 0 and i < length:
                    # Make prices test but respect the support
                    primary_df.iloc[i]['low'] = support_level * (1 - 0.01 * np.random.random())
                    if primary_df.iloc[i]['close'] < support_level:
                        primary_df.iloc[i]['close'] = support_level * (1 + 0.01 * np.random.random())
            
            # Confirm the support on higher timeframes
            for tf in timeframes[primary_idx+1:]:
                higher_df = result[tf]
                
                # Set key levels based on price ratio
                price_ratio = higher_df['close'].mean() / primary_close_mean
                tf_support = support_level * price_ratio
                
                # Create subtle support at the same level
                length = len(higher_df)
                mid_point = length // 2
                
                for i in range(mid_point - 2, mid_point + 2):
                    if i >= 0 and i < length:
                        if higher_df.iloc[i]['low'] > tf_support:
                            higher_df.iloc[i]['low'] = tf_support * (1 + 0.005 * np.random.random())
            
            # Show the support being tested more frequently on lower timeframes
            for tf in timeframes[:primary_idx]:
                lower_df = result[tf]
                
                # Set key levels based on price ratio
                price_ratio = lower_df['close'].mean() / primary_close_mean
                tf_support = support_level * price_ratio
                
                # Create multiple support tests
                length = len(lower_df)
                test_points = [length // 4, length // 2, 3 * length // 4]
                
                for point in test_points:
                    for i in range(point - 2, point + 2):
                        if i >= 0 and i < length:
                            # Create a deep test but bounce from support
                            lower_df.iloc[i]['low'] = tf_support * (1 - 0.02 * np.random.random())
                            if lower_df.iloc[i]['close'] < tf_support:
                                lower_df.iloc[i]['close'] = tf_support * (1 + 0.02 * np.random.random())
    
    return result

def get_timeframe_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes
    
    Parameters:
    - timeframe: Timeframe string (e.g., "15m", "4h", "1d")
    
    Returns:
    - Number of minutes
    """
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 24
    return 0 