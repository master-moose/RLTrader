import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional, Union

def calculate_realistic_indicators(synthetic_price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for synthetic data that behave realistically
    
    Parameters:
    - synthetic_price_data: DataFrame with synthetic OHLCV data
    
    Returns:
    - DataFrame with added technical indicators that behave as they would in real markets
    """
    # Create a copy to avoid modifying the original
    df = synthetic_price_data.copy()
    
    # Calculate basic indicators
    # Moving averages
    df['sma_7'] = ta.sma(df['close'], length=7)
    df['sma_25'] = ta.sma(df['close'], length=25)
    df['sma_99'] = ta.sma(df['close'], length=99)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    
    # Oscillators
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # Add realistic noise to RSI (more noise in choppy periods)
    price_volatility = df['close'].pct_change().rolling(window=14).std()
    rsi_noise = np.random.normal(0, 1, len(df)) * price_volatility * 10
    df['rsi_14'] = df['rsi_14'] + rsi_noise
    # Ensure RSI stays in 0-100 range
    df['rsi_14'] = np.clip(df['rsi_14'], 0, 100)
    
    # MACD with realistic lag
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # Add realistic noise to MACD (correlated with price volatility)
    macd_noise = np.random.normal(0, 1, len(df)) * price_volatility * df['close'].mean() * 0.001
    df['macd'] = df['macd'] + macd_noise
    
    # Ensure signal lags a bit more during volatile periods
    signal_lag_adjustment = np.random.normal(0, 1, len(df)) * price_volatility * 0.5
    df['macd_signal'] = df['macd_signal'].shift(1) * 0.05 + df['macd_signal'] * 0.95 + signal_lag_adjustment
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_middle'] = bbands['BBM_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    
    # Add realistic BB noise based on volatility
    bb_adjustment = price_volatility * df['close'].mean() * 0.005
    df['bb_upper'] = df['bb_upper'] + bb_adjustment * np.random.normal(0.5, 0.5, len(df))
    df['bb_lower'] = df['bb_lower'] - bb_adjustment * np.random.normal(0.5, 0.5, len(df))
    
    # Stochastic oscillator with realistic behavior
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Add realistic noise to stochastic (more noise in ranges, less in trends)
    linear_reg = ta.linreg(df['close'], length=14)
    # Calculate slope from linear regression
    trend_strength = abs(linear_reg.diff(1))
    normalized_trend = trend_strength / trend_strength.mean()
    stoch_noise = np.random.normal(0, 2, len(df)) * (1 - normalized_trend.fillna(0))
    df['stoch_k'] = np.clip(df['stoch_k'] + stoch_noise, 0, 100)
    df['stoch_d'] = np.clip(df['stoch_d'] + stoch_noise * 0.5, 0, 100)  # Less noise in signal line
    
    # Add other indicators
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['adx_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    
    # ADX behaves realistically (less sensitive in choppy markets)
    adx_adjustment = np.where(
        df['adx_14'] < 25,  # Weak trend
        np.random.normal(-2, 2, len(df)),  # More random in weak trends
        np.random.normal(0, 1, len(df))    # Less random in strong trends
    )
    df['adx_14'] = np.clip(df['adx_14'] + adx_adjustment, 0, 100)
    
    # Volume-based indicators
    df['obv'] = ta.obv(df['close'], df['volume'])
    
    # Add noise to OBV that occasionally creates false signals
    obv_noise = np.zeros(len(df))
    # Add occasional false signals (5% of the time)
    false_signal_points = np.random.choice(range(len(df)), size=int(len(df)*0.05), replace=False)
    for i in false_signal_points:
        if i > 0:
            obv_noise[i] = df['obv'].iloc[i-1] * np.random.uniform(0.01, 0.03) * np.random.choice([-1, 1])
    
    df['obv'] = df['obv'] + obv_noise.cumsum()
    
    return df

def generate_pattern_with_indicators(pattern_type: str, 
                                    params: Dict, 
                                    include_indicators: bool = True) -> pd.DataFrame:
    """
    Generate synthetic pattern with corresponding indicator behavior
    
    Parameters:
    - pattern_type: Type of pattern to generate
    - params: Parameters for pattern generation
    - include_indicators: Whether to include technical indicators
    
    Returns:
    - DataFrame with price data and indicators
    """
    # Import pattern generators here to avoid circular imports
    from .pattern_generator import (generate_trend_pattern, 
                                   generate_reversal_pattern,
                                   generate_support_resistance_reaction)
    
    # Generate price data based on pattern type
    if pattern_type == 'trend':
        df = generate_trend_pattern(
            length=params.get('length', 100),
            trend_type=params.get('trend_type', 'uptrend'),
            noise_level=params.get('noise_level', 0.05),
            volatility_profile=params.get('volatility_profile', 'medium')
        )
    elif pattern_type == 'reversal':
        df = generate_reversal_pattern(
            length=params.get('length', 100),
            pattern_type=params.get('reversal_type', 'double_top'),
            noise_level=params.get('noise_level', 0.05),
            volume_profile=params.get('volume_profile', 'increasing')
        )
    elif pattern_type == 'support_resistance':
        df = generate_support_resistance_reaction(
            length=params.get('length', 100),
            reaction_type=params.get('reaction_type', 'bounce'),
            strength=params.get('strength', 'strong'),
            noise_level=params.get('noise_level', 0.05)
        )
    else:
        # Default to uptrend if pattern type not recognized
        df = generate_trend_pattern(
            length=params.get('length', 100),
            trend_type='uptrend',
            noise_level=params.get('noise_level', 0.05)
        )
    
    # Add indicators if requested
    if include_indicators:
        df = calculate_realistic_indicators(df)
    
    return df

def create_feature_confluences(base_pattern: pd.DataFrame, 
                              confluence_type: str) -> pd.DataFrame:
    """
    Create specific technical indicator confluences in synthetic data
    
    Parameters:
    - base_pattern: Base price pattern data
    - confluence_type: Type of confluence to create (e.g., 'bullish_divergence', 'support_test')
    
    Returns:
    - Modified pattern with specific indicator confluences
    """
    # Create a copy to avoid modifying the original
    df = base_pattern.copy()
    
    # Calculate necessary indicators if they don't exist
    required_indicators = ['rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
    missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
    
    if missing_indicators:
        df = calculate_realistic_indicators(df)
    
    # Apply specific confluences
    if confluence_type == 'bullish_divergence':
        # Create bullish RSI divergence
        # Price makes lower lows while RSI makes higher lows
        length = len(df)
        mid_point = length // 2
        
        # Find a significant low point in the first half
        first_half = df.iloc[:mid_point]
        first_low_idx = first_half['close'].idxmin()
        
        # Find a low point in the second half
        second_half = df.iloc[mid_point:]
        second_low_idx = second_half['close'].idxmin()
        
        # Make the second price low lower than the first
        price_adjustment = df.loc[first_low_idx, 'close'] * 0.05  # 5% lower
        df.loc[second_low_idx, 'close'] = df.loc[first_low_idx, 'close'] - price_adjustment
        
        # But make the second RSI higher than the first (divergence)
        df.loc[second_low_idx, 'rsi_14'] = df.loc[first_low_idx, 'rsi_14'] + 10
        
        # Adjust nearby points for smooth transitions
        window = 5
        for i in range(1, window + 1):
            # Adjust points before and after the second low
            if second_low_idx - i in df.index:
                weight = (window - i + 1) / window
                df.loc[second_low_idx - i, 'rsi_14'] += 10 * weight
            
            if second_low_idx + i in df.index:
                weight = (window - i + 1) / window
                df.loc[second_low_idx + i, 'rsi_14'] += 10 * weight
    
    elif confluence_type == 'bearish_divergence':
        # Create bearish RSI divergence
        # Price makes higher highs while RSI makes lower highs
        length = len(df)
        mid_point = length // 2
        
        # Find a significant high point in the first half
        first_half = df.iloc[:mid_point]
        first_high_idx = first_half['close'].idxmax()
        
        # Find a high point in the second half
        second_half = df.iloc[mid_point:]
        second_high_idx = second_half['close'].idxmax()
        
        # Make the second price high higher than the first
        price_adjustment = df.loc[first_high_idx, 'close'] * 0.05  # 5% higher
        df.loc[second_high_idx, 'close'] = df.loc[first_high_idx, 'close'] + price_adjustment
        
        # But make the second RSI lower than the first (divergence)
        df.loc[second_high_idx, 'rsi_14'] = df.loc[first_high_idx, 'rsi_14'] - 10
        
        # Adjust nearby points for smooth transitions
        window = 5
        for i in range(1, window + 1):
            # Adjust points before and after the second high
            if second_high_idx - i in df.index:
                weight = (window - i + 1) / window
                df.loc[second_high_idx - i, 'rsi_14'] -= 10 * weight
            
            if second_high_idx + i in df.index:
                weight = (window - i + 1) / window
                df.loc[second_high_idx + i, 'rsi_14'] -= 10 * weight
    
    elif confluence_type == 'macd_crossover_at_support':
        # Create MACD bullish crossover at price support
        length = len(df)
        mid_point = length // 2
        
        # Find a significant low point
        low_idx = df['close'].idxmin()
        
        # Create a MACD crossover at that point
        df.loc[low_idx, 'macd'] = df.loc[low_idx, 'macd_signal'] * 0.95  # Just below signal
        
        # Create crossover in the next few candles
        for i in range(1, 4):
            if low_idx + i in df.index:
                crossover_strength = i / 4  # Progressive crossover
                df.loc[low_idx + i, 'macd'] = df.loc[low_idx + i, 'macd_signal'] * (1 + 0.05 * crossover_strength)
        
        # Adjust surrounding MACD values for smooth transition
        window = 5
        for i in range(1, window + 1):
            if low_idx - i in df.index:
                weight = (window - i + 1) / window
                adjustment = (df.loc[low_idx, 'macd_signal'] - df.loc[low_idx, 'macd']) * weight
                df.loc[low_idx - i, 'macd'] = df.loc[low_idx - i, 'macd'] - adjustment * 0.5
    
    elif confluence_type == 'bollinger_band_squeeze':
        # Create Bollinger Band squeeze (narrowing bands) followed by expansion
        length = len(df)
        squeeze_start = length // 3
        squeeze_end = 2 * length // 3
        
        # Calculate standard Bollinger Band width
        std_bb_width = df['bb_upper'] - df['bb_lower']
        avg_width = std_bb_width.mean()
        
        # Narrow the bands during squeeze period
        for i in range(squeeze_start, squeeze_end):
            squeeze_factor = 1 - 0.6 * ((i - squeeze_start) / (squeeze_end - squeeze_start))
            
            df.iloc[i]['bb_upper'] = df.iloc[i]['bb_middle'] + (df.iloc[i]['bb_upper'] - df.iloc[i]['bb_middle']) * squeeze_factor
            df.iloc[i]['bb_lower'] = df.iloc[i]['bb_middle'] - (df.iloc[i]['bb_middle'] - df.iloc[i]['bb_lower']) * squeeze_factor
        
        # Expand the bands after the squeeze
        for i in range(squeeze_end, min(squeeze_end + 10, length)):
            expansion_factor = 1 + 0.5 * ((i - squeeze_end) / 10)
            
            df.iloc[i]['bb_upper'] = df.iloc[i]['bb_middle'] + (df.iloc[i]['bb_upper'] - df.iloc[i]['bb_middle']) * expansion_factor
            df.iloc[i]['bb_lower'] = df.iloc[i]['bb_middle'] - (df.iloc[i]['bb_middle'] - df.iloc[i]['bb_lower']) * expansion_factor
    
    # Update the histogram based on modified MACD values
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

def simulate_indicator_lag(price_data: pd.DataFrame, 
                          indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Simulate realistic lag in indicators relative to price
    
    Parameters:
    - price_data: Synthetic price data
    - indicators: Dictionary of calculated indicators
    
    Returns:
    - Modified indicators with realistic lag characteristics
    """
    result = {}
    
    # Define lag characteristics for different indicator types
    lag_params = {
        'sma': {'lag_factor': 1.0, 'smoothing': 0.0},  # Significant lag
        'ema': {'lag_factor': 0.5, 'smoothing': 0.2},  # Less lag than SMA
        'rsi': {'lag_factor': 0.3, 'smoothing': 0.1},  # Moderate lag
        'macd': {'lag_factor': 0.7, 'smoothing': 0.15},  # Notable lag
        'macd_signal': {'lag_factor': 1.2, 'smoothing': 0.25},  # Even more lag
        'stoch': {'lag_factor': 0.4, 'smoothing': 0.15},  # Moderate lag
        'bb': {'lag_factor': 0.6, 'smoothing': 0.1},  # Moderate lag
        'adx': {'lag_factor': 0.9, 'smoothing': 0.3},  # Significant lag and smoothing
    }
    
    # Calculate price volatility to adjust lag dynamically
    volatility = price_data['close'].pct_change().rolling(window=14).std().fillna(0)
    normalized_volatility = volatility / volatility.mean() if volatility.mean() > 0 else volatility
    
    # Process each indicator
    for name, series in indicators.items():
        # Determine indicator type
        ind_type = None
        for key in lag_params.keys():
            if key in name:
                ind_type = key
                break
        
        if ind_type is None:
            # If indicator type not recognized, copy as is
            result[name] = series.copy()
            continue
        
        # Get lag parameters
        lag_factor = lag_params[ind_type]['lag_factor']
        smoothing = lag_params[ind_type]['smoothing']
        
        # Apply lag
        lagged_series = series.copy()
        
        # Calculate dynamic lag based on volatility
        dynamic_lag = int(np.ceil(lag_factor * (1 + normalized_volatility * 0.5)))
        
        # Shift the series to simulate lag
        lagged_series = series.shift(dynamic_lag)
        
        # Apply smoothing if needed
        if smoothing > 0:
            # Use exponential smoothing to reduce sharp movements
            for i in range(1, len(lagged_series)):
                if pd.notna(lagged_series.iloc[i-1]) and pd.notna(lagged_series.iloc[i]):
                    smooth_factor = smoothing * (1 + normalized_volatility.iloc[i] * 0.5)
                    lagged_series.iloc[i] = lagged_series.iloc[i-1] * smooth_factor + lagged_series.iloc[i] * (1 - smooth_factor)
        
        result[name] = lagged_series
    
    return result

def add_indicator_noise(indicators: Dict[str, pd.Series], 
                       noise_level: str = "realistic") -> Dict[str, pd.Series]:
    """
    Add realistic noise to technical indicators
    
    Parameters:
    - indicators: Dictionary of calculated indicators
    - noise_level: Amount and type of noise to add
    
    Returns:
    - Modified indicators with realistic noise characteristics
    """
    result = {}
    
    # Define noise parameters for different indicators and noise levels
    noise_params = {
        "minimal": {
            'default': 0.01,
            'oscillator': 0.2,  # For RSI, stochastic (0-100 scale)
            'macd': 0.005,  # For MACD (varies by price)
            'false_signal_freq': 0.01  # 1% of points may have false signals
        },
        "realistic": {
            'default': 0.02,
            'oscillator': 0.5,
            'macd': 0.01,
            'false_signal_freq': 0.05  # 5% of points may have false signals
        },
        "noisy": {
            'default': 0.04,
            'oscillator': 1.0,
            'macd': 0.02,
            'false_signal_freq': 0.1  # 10% of points may have false signals
        }
    }
    
    # Get noise parameters
    params = noise_params.get(noise_level, noise_params["realistic"])
    
    # Process each indicator
    for name, series in indicators.items():
        # Create a copy
        noisy_series = series.copy()
        length = len(noisy_series)
        
        # Determine indicator type and appropriate noise level
        if any(x in name.lower() for x in ['rsi', 'stoch', 'cci']):
            # Oscillator-type indicators (0-100 scale)
            base_noise = params['oscillator']
            noise = np.random.normal(0, base_noise, length)
            
            # Add noise
            noisy_series = noisy_series + noise
            
            # Ensure values stay in valid range (0-100)
            noisy_series = np.clip(noisy_series, 0, 100)
            
        elif any(x in name.lower() for x in ['macd', 'histogram']):
            # MACD-type indicators (scale varies with price)
            base_noise = params['macd']
            # Scale noise with the magnitude of the indicator
            scale_factor = noisy_series.abs().mean() if noisy_series.abs().mean() > 0 else 1
            noise = np.random.normal(0, base_noise * scale_factor, length)
            
            # Add noise
            noisy_series = noisy_series + noise
            
        else:
            # Default case (moving averages, Bollinger Bands, etc.)
            base_noise = params['default']
            # Scale noise with the magnitude of the indicator
            scale_factor = noisy_series.abs().mean() if noisy_series.abs().mean() > 0 else 1
            noise = np.random.normal(0, base_noise * scale_factor, length)
            
            # Add noise
            noisy_series = noisy_series + noise
        
        # Add occasional false signals
        false_signal_freq = params['false_signal_freq']
        false_signal_points = np.random.choice(
            range(length), 
            size=int(length * false_signal_freq),
            replace=False
        )
        
        for i in false_signal_points:
            if i > 0 and i < length - 1:
                # For oscillators
                if any(x in name.lower() for x in ['rsi', 'stoch', 'cci']):
                    # Create false breakouts above 70 or below 30
                    if 30 <= noisy_series.iloc[i] <= 70:
                        if noisy_series.iloc[i] > 50:
                            noisy_series.iloc[i] = np.random.uniform(70, 75)
                        else:
                            noisy_series.iloc[i] = np.random.uniform(25, 30)
                
                # For MACD
                elif any(x in name.lower() for x in ['macd']):
                    # Create false crossover
                    if 'signal' in name.lower():
                        continue  # Skip signal line
                    elif 'histogram' in name.lower():
                        # Flip the histogram value to create false crossover
                        noisy_series.iloc[i] = -noisy_series.iloc[i] * np.random.uniform(0.2, 0.5)
                    else:
                        # Regular MACD line - create false signal by moving it toward signal
                        signal_name = name.replace('macd', 'macd_signal')
                        if signal_name in indicators:
                            signal_value = indicators[signal_name].iloc[i]
                            # Move MACD toward and slightly past signal
                            direction = 1 if noisy_series.iloc[i] < signal_value else -1
                            noisy_series.iloc[i] = signal_value + direction * abs(noisy_series.iloc[i] - signal_value) * 0.1
                
                # For moving averages and price-based indicators
                else:
                    # Create false breakouts or breakdowns
                    change_pct = np.random.uniform(0.003, 0.008)  # 0.3% to 0.8%
                    if np.random.random() > 0.5:
                        noisy_series.iloc[i] *= (1 + change_pct)  # False breakout
                    else:
                        noisy_series.iloc[i] *= (1 - change_pct)  # False breakdown
        
        result[name] = noisy_series
    
    return result 