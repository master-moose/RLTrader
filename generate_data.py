#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy import stats
import random
from crypto_trading_model.data_processing.feature_engineering import (
    calculate_multi_timeframe_signal
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_generation.log')
    ]
)
logger = logging.getLogger('crypto_trading_model.data_generation')

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'data/synthetic',
        'output/time_series',
        'output/reinforcement',
        'output/ensemble',
        'output/backtest',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directory setup complete.")

def load_config(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        sys.exit(1)

class MarketRegime:
    """Represents different market regimes with specific characteristics."""
    
    UPTREND = 'uptrend'
    DOWNTREND = 'downtrend'
    RANGING = 'ranging'
    VOLATILITY_EXPANSION = 'volatility_expansion'
    VOLATILITY_CONTRACTION = 'volatility_contraction'
    
    @staticmethod
    def get_regime_parameters(regime_type, base_volatility=0.002):
        """Get drift and volatility parameters for a specific regime.
        
        Parameters:
        -----------
        regime_type : str
            Type of market regime
        base_volatility : float
            Base volatility level to scale from
            
        Returns:
        --------
        dict
            Dictionary with regime parameters
        """
        params = {
            MarketRegime.UPTREND: {
                'mu': 0.0005,  # Strong positive drift
                'sigma': base_volatility * 1.0,  # Normal volatility
                'mean_reversion_strength': 0.05,  # Low mean reversion (trending)
                'volume_factor': 1.2,  # Higher than normal volume
                'duration_range': (20, 100)  # Tends to last longer
            },
            MarketRegime.DOWNTREND: {
                'mu': -0.0006,  # Strong negative drift (faster than uptrends)
                'sigma': base_volatility * 1.3,  # Higher volatility
                'mean_reversion_strength': 0.05,  # Low mean reversion (trending)
                'volume_factor': 1.5,  # Even higher volume (panic selling)
                'duration_range': (15, 80)  # Tends to be shorter than uptrends
            },
            MarketRegime.RANGING: {
                'mu': 0.0000,  # No drift
                'sigma': base_volatility * 0.8,  # Lower volatility
                'mean_reversion_strength': 0.2,  # High mean reversion
                'volume_factor': 0.8,  # Lower volume
                'duration_range': (10, 50)  # Variable duration
            },
            MarketRegime.VOLATILITY_EXPANSION: {
                'mu': 0.0000,  # No consistent direction
                'sigma': base_volatility * 2.5,  # Much higher volatility
                'mean_reversion_strength': 0.1,  # Moderate mean reversion
                'volume_factor': 2.0,  # Much higher volume
                'duration_range': (5, 20)  # Usually short-lived
            },
            MarketRegime.VOLATILITY_CONTRACTION: {
                'mu': 0.0001,  # Slight upward bias
                'sigma': base_volatility * 0.5,  # Very low volatility
                'mean_reversion_strength': 0.15,  # Moderate mean reversion
                'volume_factor': 0.6,  # Low volume
                'duration_range': (5, 30)  # Usually precedes a volatility expansion
            }
        }
        
        # Fallback to ranging if regime type is not recognized
        return params.get(regime_type, params[MarketRegime.RANGING])

def generate_regime_sequence(num_samples, regime_distribution=None, seed=None):
    """Generate a sequence of market regimes.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    regime_distribution : dict, optional
        Probability distribution of different regimes
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    list
        List of regime types for each time step
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Default regime distribution if not provided
    if regime_distribution is None:
        regime_distribution = {
            MarketRegime.UPTREND: 0.35,
            MarketRegime.DOWNTREND: 0.25,
            MarketRegime.RANGING: 0.25,
            MarketRegime.VOLATILITY_EXPANSION: 0.1,
            MarketRegime.VOLATILITY_CONTRACTION: 0.05
        }
    
    # Normalize distribution to ensure probabilities sum to 1
    total_prob = sum(regime_distribution.values())
    normalized_distribution = {k: v / total_prob for k, v in regime_distribution.items()}
    
    # Extract regime types and probabilities
    regime_types = list(normalized_distribution.keys())
    probabilities = list(normalized_distribution.values())
    
    # Initialize with a random regime
    current_regime = np.random.choice(regime_types, p=probabilities)
    regime_params = MarketRegime.get_regime_parameters(current_regime)
    
    # Generate regime sequence
    regimes = []
    remaining_duration = np.random.randint(*regime_params['duration_range'])
    
    # Markov transition matrix - probability of transitioning from one regime to another
    # Higher values on the diagonal mean regimes tend to persist
    transition_matrix = {
        MarketRegime.UPTREND: {
            MarketRegime.UPTREND: 0.7,
            MarketRegime.DOWNTREND: 0.1,
            MarketRegime.RANGING: 0.1,
            MarketRegime.VOLATILITY_EXPANSION: 0.05,
            MarketRegime.VOLATILITY_CONTRACTION: 0.05
        },
        MarketRegime.DOWNTREND: {
            MarketRegime.UPTREND: 0.1,
            MarketRegime.DOWNTREND: 0.7,
            MarketRegime.RANGING: 0.1,
            MarketRegime.VOLATILITY_EXPANSION: 0.05,
            MarketRegime.VOLATILITY_CONTRACTION: 0.05
        },
        MarketRegime.RANGING: {
            MarketRegime.UPTREND: 0.15,
            MarketRegime.DOWNTREND: 0.15,
            MarketRegime.RANGING: 0.5,
            MarketRegime.VOLATILITY_EXPANSION: 0.15,
            MarketRegime.VOLATILITY_CONTRACTION: 0.05
        },
        MarketRegime.VOLATILITY_EXPANSION: {
            MarketRegime.UPTREND: 0.3,
            MarketRegime.DOWNTREND: 0.3,
            MarketRegime.RANGING: 0.2,
            MarketRegime.VOLATILITY_EXPANSION: 0.1,
            MarketRegime.VOLATILITY_CONTRACTION: 0.1
        },
        MarketRegime.VOLATILITY_CONTRACTION: {
            MarketRegime.UPTREND: 0.1,
            MarketRegime.DOWNTREND: 0.1,
            MarketRegime.RANGING: 0.1,
            MarketRegime.VOLATILITY_EXPANSION: 0.6,
            MarketRegime.VOLATILITY_CONTRACTION: 0.1
        }
    }
    
    for _ in range(num_samples):
        if remaining_duration <= 0:
            # Time to transition to a new regime
            # Get transition probabilities for current regime
            transition_probs = transition_matrix[current_regime]
            next_regimes = list(transition_probs.keys())
            next_probs = list(transition_probs.values())
            
            # Choose next regime based on transition probabilities
            current_regime = np.random.choice(next_regimes, p=next_probs)
            regime_params = MarketRegime.get_regime_parameters(current_regime)
            remaining_duration = np.random.randint(*regime_params['duration_range'])
        
        regimes.append(current_regime)
        remaining_duration -= 1
    
    logger.info(f"Generated regime sequence with {len(regimes)} elements")
    # Log regime distribution in the generated sequence
    regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
    for regime, count in regime_counts.items():
        logger.info(f"  - {regime}: {count} periods ({count/len(regimes)*100:.1f}%)")
    
    return regimes

def generate_price_process(num_samples, regimes, base_price=10000.0, base_volatility=0.002, seed=None):
    """Generate a realistic price process with regime-switching dynamics.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    regimes : list
        List of regime types for each time step
    base_price : float
        Starting price
    base_volatility : float
        Base volatility level
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (prices, volatilities, volumes) - numpy arrays for each component
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize arrays
    prices = np.zeros(num_samples)
    volatilities = np.zeros(num_samples)
    volumes = np.zeros(num_samples)
    
    # Set initial price
    prices[0] = base_price
    
    # Initialize for GARCH-like volatility process
    alpha = 0.1  # News impact
    beta = 0.85  # Volatility persistence
    gamma = 0.02  # Asymmetry factor (negative returns increase volatility more)
    omega = base_volatility**2 * (1 - alpha - beta)  # Long-run variance
    
    # Initial volatility
    current_volatility = base_volatility
    volatilities[0] = current_volatility
    
    # Initialize volume with log-normal distribution
    volumes[0] = np.random.lognormal(10, 1)
    
    # Support and resistance levels
    support_levels = [base_price * 0.9]
    resistance_levels = [base_price * 1.1]

    # Add price limits to enforce the 5000-120000 range
    min_price = 5000.0  # Minimum allowed price
    max_price = 120000.0  # Maximum allowed price
    
    # Generate price, volatility, and volume processes
    for i in range(1, num_samples):
        # Get regime parameters
        regime = regimes[i]
        params = MarketRegime.get_regime_parameters(regime, base_volatility)
        
        # Update volatility (GARCH-like process)
        if i > 1:
            # Calculate return with safeguard against extreme prices
            if np.isfinite(prices[i-1]) and np.isfinite(prices[i-2]) and prices[i-2] > 0:
                ret = (prices[i-1] / prices[i-2]) - 1
                ret = np.clip(ret, -0.5, 0.5)  # Limit extreme returns
            else:
                ret = 0  # Default to zero if we have bad values
            
            # Asymmetric effect: negative returns increase volatility more
            leverage = 1 + (gamma * (ret < 0))
            
            # GARCH update with safeguards
            current_volatility = np.sqrt(
                omega + 
                alpha * (ret**2) * leverage + 
                beta * volatilities[i-1]**2
            )
            
            # Sanity check on volatility
            if not np.isfinite(current_volatility) or current_volatility <= 0:
                current_volatility = base_volatility
        
        # Scale volatility by regime
        scaled_volatility = current_volatility * params['sigma'] / base_volatility
        scaled_volatility = np.clip(scaled_volatility, base_volatility * 0.1, base_volatility * 10)
        volatilities[i] = scaled_volatility
        
        # Mean-reverting component (if applicable)
        mean_reversion = 0
        if params['mean_reversion_strength'] > 0 and np.isfinite(prices[i-1]):
            # Determine local mean (could be moving average in real implementation)
            local_prices = prices[max(0, i-30):i]
            local_prices = local_prices[np.isfinite(local_prices)]  # Filter out bad values
            
            if len(local_prices) > 0:
                local_mean = np.mean(local_prices)
                
                # Apply price range constraints to mean reversion target
                local_mean = np.clip(local_mean, min_price, max_price)
                
                # Calculate mean reversion effect with safeguard
                if np.isfinite(local_mean):
                    mean_reversion = params['mean_reversion_strength'] * (local_mean - prices[i-1])
                    # Clip to avoid extreme reversion
                    mean_reversion = np.clip(mean_reversion, -0.05, 0.05)
                    
                    # Add stronger mean reversion when price is approaching the min/max bounds
                    if prices[i-1] < min_price * 1.1:  # Within 10% of min price
                        mean_reversion += 0.005  # Add upward pressure
                    elif prices[i-1] > max_price * 0.9:  # Within 10% of max price
                        mean_reversion -= 0.005  # Add downward pressure
        
        # Support and resistance effects
        sr_effect = 0
        
        # Only calculate S/R effects if price is finite and positive
        if np.isfinite(prices[i-1]) and prices[i-1] > 0:
            # Find closest support level below current price
            supports_below = [s for s in support_levels if s < prices[i-1]]
            closest_support = max(supports_below) if supports_below else min_price
            
            # Find closest resistance level above current price
            resistances_above = [r for r in resistance_levels if r > prices[i-1]]
            closest_resistance = min(resistances_above) if resistances_above else max_price
            
            # Distance to closest support/resistance (as percentage of price)
            support_distance = (prices[i-1] - closest_support) / prices[i-1] if closest_support > 0 else 1.0
            resistance_distance = (closest_resistance - prices[i-1]) / prices[i-1] if closest_resistance < float('inf') else 1.0
            
            # Effect becomes stronger as price approaches support/resistance
            if support_distance < 0.03 and support_distance > 0:  # Within 3% of support
                # Stronger support as we get closer
                bounce_probability = 0.7 * (1 - support_distance / 0.03)
                if np.random.random() < bounce_probability:
                    # Bounce effect - pulls price up
                    sr_effect = 0.002 * (1 - support_distance / 0.03)
            
            if resistance_distance < 0.03 and resistance_distance > 0:  # Within 3% of resistance
                # Stronger resistance as we get closer
                bounce_probability = 0.7 * (1 - resistance_distance / 0.03)
                if np.random.random() < bounce_probability:
                    # Resistance effect - pulls price down
                    sr_effect = -0.002 * (1 - resistance_distance / 0.03)
            
            # Occasionally break support/resistance
            if (support_distance < 0.01 and support_distance > 0 and np.random.random() < 0.05) or \
               (resistance_distance < 0.01 and resistance_distance > 0 and np.random.random() < 0.05):
                # Support/resistance breakout
                if support_distance < resistance_distance:
                    # Support break - strong move down
                    sr_effect = -0.01
                    # Remove this support level and add a new one lower
                    if closest_support in support_levels:
                        support_levels.remove(closest_support)
                        # Resistance becomes support when broken
                        resistance_levels.append(closest_support)
                    new_support = closest_support * 0.95
                    # Ensure new support isn't below min_price
                    new_support = max(new_support, min_price)
                    support_levels.append(new_support)
                else:
                    # Resistance break - strong move up
                    sr_effect = 0.01
                    # Remove this resistance level and add a new one higher
                    if closest_resistance in resistance_levels:
                        resistance_levels.remove(closest_resistance)
                        # Resistance becomes support when broken
                        support_levels.append(closest_resistance)
                    new_resistance = closest_resistance * 1.05
                    # Ensure new resistance isn't above max_price
                    new_resistance = min(new_resistance, max_price)
                    resistance_levels.append(new_resistance)
            
            # Occasionally add new support/resistance levels
            if i % 500 == 0:
                # Add new levels near current price
                new_support = prices[i-1] * 0.97
                new_resistance = prices[i-1] * 1.03
                
                # Ensure support and resistance stay within range
                new_support = max(new_support, min_price)
                new_resistance = min(new_resistance, max_price)
                
                support_levels.append(new_support)
                resistance_levels.append(new_resistance)
                # Cleanup old levels (keep a reasonable number)
                support_levels = sorted(support_levels, reverse=True)[:5]
                resistance_levels = sorted(resistance_levels)[:5]
        
        # Combined price change
        drift = params['mu']
        random_component = np.random.normal(0, scaled_volatility)
        
        # Limit extreme movements
        random_component = np.clip(random_component, -0.1, 0.1)
        
        # Final return calculation
        price_return = drift + random_component + mean_reversion + sr_effect
        
        # Apply stronger constraints near boundaries
        if prices[i-1] <= min_price * 1.05:  # Within 5% of minimum
            # Reduce negative returns, increase positive returns
            if price_return < 0:
                price_return *= 0.5  # Dampen downward moves
            else:
                price_return *= 1.2  # Amplify upward moves
        elif prices[i-1] >= max_price * 0.95:  # Within 5% of maximum
            # Reduce positive returns, increase negative returns
            if price_return > 0:
                price_return *= 0.5  # Dampen upward moves
            else:
                price_return *= 1.2  # Amplify downward moves
        
        # Clip the return to prevent extreme moves
        price_return = np.clip(price_return, -0.1, 0.1)
        
        # Update price with safeguard
        prices[i] = prices[i-1] * (1 + price_return)
        
        # Enforce price limits to keep within 5000-120000 range
        prices[i] = np.clip(prices[i], min_price, max_price)
        
        # Generate volume (correlated with volatility and absolute returns)
        base_volume = np.random.lognormal(10, 0.5)
        volume_factor = params['volume_factor']
        # Volume increases with volatility and absolute returns
        volume_multiplier = 1.0 + 2.0 * abs(price_return) + 0.5 * scaled_volatility / base_volatility
        volumes[i] = base_volume * volume_factor * volume_multiplier
        
        # Sanity check on volume
        if not np.isfinite(volumes[i]) or volumes[i] <= 0:
            volumes[i] = volumes[i-1] if np.isfinite(volumes[i-1]) and volumes[i-1] > 0 else base_volume
    
    return prices, volatilities, volumes

def generate_realistic_ohlc(close_prices, volatilities, seed=None):
    """Generate realistic Open, High, Low values based on Close prices and volatility.
    
    Parameters:
    -----------
    close_prices : numpy.ndarray
        Array of close prices
    volatilities : numpy.ndarray
        Array of price volatilities
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (open_prices, high_prices, low_prices) - numpy arrays for OHLC data
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = len(close_prices)
    open_prices = np.zeros(num_samples)
    high_prices = np.zeros(num_samples)
    low_prices = np.zeros(num_samples)
    
    # Make sure all close prices are valid
    valid_prices = np.isfinite(close_prices) & (close_prices > 0)
    if not np.all(valid_prices):
        # Handle invalid prices
        invalid_indices = np.where(~valid_prices)[0]
        logger.warning(f"Found {len(invalid_indices)} invalid close prices, fixing them")
        
        # Find first valid price
        first_valid_idx = np.where(valid_prices)[0]
        if len(first_valid_idx) == 0:
            logger.error("No valid prices found, cannot generate OHLC data")
            # Return dummy data
            return close_prices, close_prices, close_prices
            
        first_valid_price = close_prices[first_valid_idx[0]]
        
        # Fix invalid prices
        for idx in invalid_indices:
            if idx == 0:
                close_prices[idx] = first_valid_price
            else:
                # Use previous valid price
                close_prices[idx] = close_prices[idx-1]
    
    # Set initial values with safeguards
    open_prices[0] = close_prices[0] * (1 - np.random.uniform(0, 0.002))
    high_prices[0] = max(open_prices[0], close_prices[0]) * (1 + np.random.uniform(0, 0.003))
    low_prices[0] = min(open_prices[0], close_prices[0]) * (1 - np.random.uniform(0, 0.003))
    
    # Generate OHLC for remaining periods
    for i in range(1, num_samples):
        if not np.isfinite(close_prices[i]) or close_prices[i] <= 0:
            # Skip invalid prices
            open_prices[i] = close_prices[i]
            high_prices[i] = close_prices[i]
            low_prices[i] = close_prices[i]
            continue
            
        # Open price: Close of previous candle with small noise
        # This creates a gap with small probability (more common in daily timeframes)
        gap_probability = 0.05 if i % 96 == 0 else 0.01  # Higher probability for "daily" candles
        
        if np.random.random() < gap_probability:
            # Create a gap - larger for "daily" candles
            gap_size = np.random.uniform(-0.01, 0.01) * (3.0 if i % 96 == 0 else 1.0)
            open_prices[i] = close_prices[i-1] * (1 + gap_size)
        else:
            # Regular open (close of previous period with tiny noise)
            open_prices[i] = close_prices[i-1] * (1 + np.random.uniform(-0.0005, 0.0005))
        
        # Ensure open price is valid
        if not np.isfinite(open_prices[i]) or open_prices[i] <= 0:
            open_prices[i] = close_prices[i]
        
        # Price direction for this candle
        price_direction = 1 if close_prices[i] >= open_prices[i] else -1
        
        # Range is influenced by volatility, but capped to avoid extreme values
        candle_range_factor = min(2.0 + np.random.exponential(1.0), 5.0)
        range_size = min(volatilities[i] * candle_range_factor, 0.1)
        
        # For bear candles: High comes first, then Low
        # For bull candles: Low comes first, then High
        
        # Limit extension factors to prevent overflow
        max_extension = 0.5
        
        if price_direction > 0:  # Bullish candle
            # Price tends to go lower before moving higher in bull candles
            # Low is below both open and close
            base_price_low = min(open_prices[i], close_prices[i])
            low_extension = min(np.random.uniform(0.2, 0.8) * range_size, max_extension)
            low_prices[i] = base_price_low * (1 - low_extension)
            
            # High is above close
            base_price_high = max(open_prices[i], close_prices[i])
            high_extension = min(np.random.uniform(0.1, 0.7) * range_size, max_extension)
            high_prices[i] = base_price_high * (1 + high_extension)
        else:  # Bearish candle
            # Price tends to go higher before moving lower in bear candles
            # High is above both open and close
            base_price_high = max(open_prices[i], close_prices[i])
            high_extension = min(np.random.uniform(0.2, 0.8) * range_size, max_extension)
            high_prices[i] = base_price_high * (1 + high_extension)
            
            # Low is below close
            base_price_low = min(open_prices[i], close_prices[i])
            low_extension = min(np.random.uniform(0.1, 0.7) * range_size, max_extension)
            low_prices[i] = base_price_low * (1 - low_extension)
        
        # Ensure high is always highest and low is always lowest
        high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
        low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Final sanity check
        if not np.isfinite(high_prices[i]) or high_prices[i] <= 0:
            high_prices[i] = close_prices[i] * 1.001
        
        if not np.isfinite(low_prices[i]) or low_prices[i] <= 0:
            low_prices[i] = close_prices[i] * 0.999
    
    return open_prices, high_prices, low_prices

def generate_synthetic_data(num_samples=525600, output_dir='data/synthetic', config_path=None):
    """Generate synthetic data for training.
    
    Parameters:
    - num_samples: Number of 15-minute intervals to generate (default: 525600 for 15 years)
    - output_dir: Directory to save the HDF5 files
    - config_path: Optional path to a configuration file
    """
    # Load config if provided
    if config_path:
        config = load_config(config_path)
        num_samples = config.get('num_samples', num_samples)
        output_dir = config.get('output_dir', output_dir)
        base_price = config.get('base_price', 10000.0)
        base_volatility = config.get('base_volatility', 0.002)
        regime_distribution = config.get('regime_distribution', None)
        include_indicators = config.get('include_indicators', True)
        train_ratio = config.get('train_ratio', 0.7)
        val_ratio = config.get('val_ratio', 0.15)
        test_ratio = config.get('test_ratio', 0.15)
        shuffle = config.get('shuffle', False)  # Default to chronological order
        seed = config.get('seed', None)
        
        logger.info(f"Generating {num_samples} samples based on configuration")
    else:
        base_price = 10000.0
        base_volatility = 0.002
        regime_distribution = None
        include_indicators = True
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        shuffle = False
        seed = None
        
        logger.info(f"Generating {num_samples} samples with default settings")
    
    # Create a date range
    date_range = pd.date_range(
        start='2010-01-01', 
        periods=num_samples,
        freq='15min'
    )
    
    # Generate market regimes
    logger.info("Generating market regime sequence...")
    regimes = generate_regime_sequence(num_samples, regime_distribution, seed)
    
    # Generate price, volatility, and volume processes
    logger.info("Generating price, volatility, and volume processes...")
    prices, volatilities, volumes = generate_price_process(
        num_samples, regimes, base_price, base_volatility, seed
    )
    
    # Generate realistic OHLC data
    logger.info("Generating realistic OHLC data...")
    open_prices, high_prices, low_prices = generate_realistic_ohlc(prices, volatilities, seed)
    
    # Create 15m dataframe
    df_15m = pd.DataFrame({
        'timestamp': date_range,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volumes
    })
    df_15m.set_index('timestamp', inplace=True)
    
    # Create 4h dataframe (aggregate from 15m)
    df_4h = df_15m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Create 1d dataframe (aggregate from 15m)
    df_1d = df_15m.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    logger.info(f"Generated dataframes: 15m ({len(df_15m)} rows), "
                f"4h ({len(df_4h)} rows), 1d ({len(df_1d)} rows)")
    
    # Add comprehensive set of indicators
    if include_indicators:
        logger.info("Calculating technical indicators...")
        for df in [df_15m, df_4h, df_1d]:
            # Basic OHLCV features (5 features)
            # 'open', 'high', 'low', 'close', 'volume' are already in the DataFrame
            
            # Trend indicators (7 features)
            df['sma_7'] = df['close'].rolling(7).mean()
            df['sma_25'] = df['close'].rolling(25).mean()
            df['sma_99'] = df['close'].rolling(99).mean()
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            # Price position relative to moving averages
            df['price_sma_ratio'] = df['close'] / df['sma_25']
            df['sma_cross_signal'] = ((df['sma_7'] > df['sma_25']).astype(int) * 2 - 1)

            # Momentum indicators (6 features)
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / 
                                    (high_14 - low_14 + np.finfo(float).eps))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # MACD calculation
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Volatility indicators (6 features)
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * std_20
            df['bb_lower'] = df['bb_middle'] - 2 * std_20
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (
                (df['bb_upper'] - df['bb_lower']) + np.finfo(float).eps)
            
            # ATR calculation
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr_14'] = true_range.rolling(14).mean()

            # Volume indicators (4 features)
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On-Balance Volume
            df['price_change'] = df['close'].diff()
            df['price_direction'] = np.where(df['price_change'] > 0, 1, 
                                          np.where(df['price_change'] < 0, -1, 0))
            df['obv'] = (df['volume'] * df['price_direction']).cumsum()
            
            # Chaikin Money Flow
            money_flow_multiplier = ((df['close'] - df['low']) - 
                                    (df['high'] - df['close'])) / (
                                    df['high'] - df['low'] + np.finfo(float).eps)
            money_flow_volume = money_flow_multiplier * df['volume']
            df['cmf_20'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

            # Price-derived features (4 features)
            df['return'] = df['close'].pct_change(fill_method=None)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['open'] - df['close']) / df['close']
            
            # Store volatility from the simulation
            if df is df_15m:
                df['true_volatility'] = volatilities
            elif df is df_4h:
                # Resample volatility to 4h (using max as volatility tends to cluster)
                vol_series = pd.Series(volatilities, index=df_15m.index)
                df['true_volatility'] = vol_series.resample('4h').max()
            elif df is df_1d:
                # Resample volatility to 1d
                vol_series = pd.Series(volatilities, index=df_15m.index)
                df['true_volatility'] = vol_series.resample('1D').max()
            
            # Store regime information
            if df is df_15m:
                # Add regime info
                regimes_series = pd.Series(regimes, index=df_15m.index)
                df['market_regime'] = regimes_series
            elif df is df_4h:
                # Resample regimes to 4h (using mode/most frequent)
                regimes_series = pd.Series(regimes, index=df_15m.index)
                # Since we can't easily aggregate categorical data with pandas,
                # we'll just take the first element of each 4h window
                df['market_regime'] = regimes_series.resample('4h').first()
            elif df is df_1d:
                # Resample regimes to 1d
                regimes_series = pd.Series(regimes, index=df_15m.index)
                df['market_regime'] = regimes_series.resample('1D').first()
            
            # Clean up NaN values that resulted from rolling windows
            df.fillna(0, inplace=True)  # Simply replace NaNs with zeros

    dataset = {
        '15m': df_15m,
        '4h': df_4h,
        '1d': df_1d
    }
    
    # Generate advanced trading signals using multi-timeframe approach
    try:
        logger.info("Calculating advanced multi-timeframe signals...")
        multi_tf_signals = calculate_multi_timeframe_signal(
            dataset,
            primary_tf='15m',
            threshold_pct=0.015,  # 1.5% threshold for significant moves
            lookforward_periods={'15m': 16, '4h': 4, '1d': 1}
        )
        
        # Replace the price_direction in the primary timeframe with the enhanced version
        dataset['15m']['price_direction'] = multi_tf_signals
        
        # Create versions for higher timeframes by downsampling
        # 4-hour timeframe (1 4h candle = 16 15m candles)
        df_4h_indices = list(range(0, len(dataset['15m']), 16))
        dataset['4h']['price_direction'] = multi_tf_signals.iloc[df_4h_indices].values
        
        # Daily timeframe (1 day candle = 96 15m candles)
        df_1d_indices = list(range(0, len(dataset['15m']), 96))
        dataset['1d']['price_direction'] = multi_tf_signals.iloc[df_1d_indices].values
        
        logger.info("Successfully updated price_direction with multi-timeframe signals")
    except Exception as e:
        logger.error(f"Error calculating multi-timeframe signals: {str(e)}")
        logger.info("Falling back to standard price_direction calculation")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data to HDF5 format
    save_dataset_hdf5(dataset, output_dir)
    
    # Create and save train/val/test splits
    create_and_save_splits(dataset, train_ratio, val_ratio, test_ratio, shuffle, output_dir)
    
    logger.info(f"Synthetic data generation complete. Data saved to {output_dir}")
    return dataset

def save_dataset_hdf5(dataset, output_dir, filename="synthetic_dataset.h5"):
    """Save the entire dataset as a single HDF5 file with multiple groups for timeframes."""
    
    # Define HDF5 file path
    hdf5_path = os.path.join(output_dir, filename)
    
    # Save each timeframe as a separate group in the HDF5 file
    with pd.HDFStore(hdf5_path, mode='w') as store:
        for timeframe, df in dataset.items():
            # Save the DataFrame to the HDF5 file, using timeframe as the group name
            store.put(f'/{timeframe}', df, format='table', data_columns=True)
            logger.info(f"Saved {timeframe} data to {hdf5_path}/{timeframe} "
                      f"({len(df)} rows)")
    
    # Create metadata file
    metadata = {
        "creation_date": pd.Timestamp.now().isoformat(),
        "timeframes": list(dataset.keys()),
        "rows_per_timeframe": {tf: len(df) for tf, df in dataset.items()},
        "columns": {tf: list(df.columns) for tf, df in dataset.items()}
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved dataset metadata to {metadata_path}")

def create_and_save_splits(dataset, train_ratio, val_ratio, test_ratio, shuffle, output_dir):
    """Create train/val/test splits and save them as separate HDF5 files."""
    
    split_sizes = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_datasets = {split_name: {} for split_name in split_sizes.keys()}
    
    for timeframe, df in dataset.items():
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Shuffle if requested (usually not recommended for time series)
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=False)
        
        # Calculate split indices
        total_rows = len(data)
        train_idx = int(total_rows * train_ratio)
        val_idx = train_idx + int(total_rows * val_ratio)
        
        # Split the data
        split_datasets["train"][timeframe] = data.iloc[:train_idx]
        split_datasets["val"][timeframe] = data.iloc[train_idx:val_idx]
        split_datasets["test"][timeframe] = data.iloc[val_idx:]
        
        logger.info(
            f"Split {timeframe} data: "
            f"train={len(split_datasets['train'][timeframe])}, "
            f"val={len(split_datasets['val'][timeframe])}, "
            f"test={len(split_datasets['test'][timeframe])}"
        )
    
    # Save each split as a separate HDF5 file
    for split_name, split_data in split_datasets.items():
        split_file = os.path.join(output_dir, f"{split_name}_data.h5")
        
        with pd.HDFStore(split_file, mode='w') as store:
            for timeframe, df in split_data.items():
                store.put(f'/{timeframe}', df, format='table', data_columns=True)
                
        logger.info(f"Saved {split_name} split to {split_file}")

def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for cryptocurrency trading model"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=525600,
        help="Number of 15-minute samples to generate (default: 525600 for ~15 years)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/synthetic",
        help="Directory where the data will be saved (default: data/synthetic)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Generate synthetic data
    if args.config:
        generate_synthetic_data(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            config_path=args.config
        )
    else:
        config = {
            "num_samples": args.num_samples,
            "output_dir": args.output_dir,
            "seed": args.seed
        }
        generate_synthetic_data(**config)
    
    logger.info("Data generation complete.")

if __name__ == "__main__":
    main() 