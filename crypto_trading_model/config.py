"""
Configuration settings for the cryptocurrency trading model.

This module contains default settings for:
1. Paths
2. Data processing
3. Feature engineering
4. Time series models
5. Reinforcement learning models
6. Trading environment
7. Progressive learning workflow
"""

import os
from typing import Dict, Any

# ---------------------------------------------------
# Path Configuration
# ---------------------------------------------------
PATHS = {
    'data': 'data',                            # Raw and processed data
    'models': 'models',                        # Saved models
    'logs': 'logs',                            # Log files
    'results': 'results',                      # Results and visualizations
    'time_series_models': 'models/time_series',  # Time series model files
    'rl_models': 'models/reinforcement',       # Reinforcement learning model files
}

# Ensure all paths exist
for path_name, path in PATHS.items():
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------
# Data Processing Configuration
# ---------------------------------------------------
DATA_SETTINGS = {
    'default_timeframes': ['15m', '1h', '4h', '1d'],  # Default timeframes to process
    'default_symbols': ['BTC/USDT'],                  # Default symbols to process
    'start_date': '2020-01-01',                       # Default start date
    'end_date': None,                                 # Default end date (None = current date)
    'max_rows': 100000,                               # Maximum rows per timeframe
    'ccxt_exchange': 'binance',                       # Default exchange for data fetching
    'fillna_method': 'ffill',                         # Method to fill missing values
    'include_synthetic': True,                        # Whether to generate synthetic data
    'cache_data': True,                               # Whether to cache data to disk
    'data_format': 'hdf5',                            # Format for data storage (csv, hdf5, pickle)
}

# ---------------------------------------------------
# Feature Engineering Configuration
# ---------------------------------------------------
FEATURE_SETTINGS = {
    'trend_indicators': [
        'sma', 'ema', 'macd', 'adx', 'ichimoku', 'parabolic_sar'
    ],
    'momentum_indicators': [
        'rsi', 'stoch', 'stoch_rsi', 'williams_r', 'ultimate_oscillator'
    ],
    'volatility_indicators': [
        'bollinger_bands', 'keltner_channel', 'atr', 'standard_deviation'
    ],
    'volume_indicators': [
        'obv', 'volume_profile', 'vwap', 'cmf', 'mfi'
    ],
    'pattern_recognition': [
        'candlestick_patterns', 'support_resistance', 'fibonacci'
    ],
    # Default moving average windows
    'ma_windows': [5, 10, 20, 50, 100, 200],
    # Default RSI window
    'rsi_window': 14,
    # Default MACD parameters
    'macd_params': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    # Default Bollinger Bands parameters
    'bb_params': {
        'window': 20,
        'std_dev': 2
    }
}

# ---------------------------------------------------
# Time Series Model Configuration
# ---------------------------------------------------
TIME_SERIES_SETTINGS = {
    'sequence_length': 60,            # Number of time steps to look back
    'forecast_horizon': 5,            # Number of steps to predict ahead
    'target_column': 'close',         # Column to predict (usually 'close')
    'model_type': 'lstm',             # Model type: 'lstm', 'bilstm', 'cnn_lstm', 'attention'
    'lstm_units': [64, 64],           # Units in LSTM layers
    'dense_units': [32],              # Units in Dense layers
    'dropout_rate': 0.2,              # Dropout rate
    'learning_rate': 0.001,           # Learning rate for optimizer
    'batch_size': 256,                # Batch size for training
    'epochs': 50,                     # Number of training epochs
    'early_stopping_patience': 10,    # Patience for early stopping
    'validation_split': 0.2,          # Fraction of data to use for validation
    'use_multivariate': True,         # Whether to use multiple features or just price
    'train_test_split': 0.8,          # Train/test split ratio
}

# ---------------------------------------------------
# Reinforcement Learning Configuration
# ---------------------------------------------------
RL_SETTINGS = {
    # General RL settings
    'gamma': 0.99,                    # Discount factor
    'learning_rate': 0.0003,          # Learning rate for RL algorithms
    'batch_size': 64,                 # Batch size for training
    'train_test_split': 0.8,          # Train/test split ratio
    
    # DQN specific settings
    'dqn': {
        'learning_rate': 0.0005,       # Learning rate for DQN
        'gamma': 0.99,                 # Discount factor
        'epsilon_start': 1.0,          # Initial exploration rate
        'epsilon_end': 0.01,           # Final exploration rate
        'epsilon_decay_steps': 50000,  # Number of steps for epsilon decay
        'batch_size': 64,              # Batch size for training
        'update_target_every': 500,    # Update target network every N steps
        'hidden_units': [64, 64],      # Hidden units in each layer
        'buffer_capacity': 100000,     # Experience replay buffer capacity
        'double_dqn': True,            # Whether to use Double DQN
        'total_timesteps': 100000,     # Total timesteps for training
        'log_interval': 1000,          # Interval for logging
        'eval_interval': 10000,        # Interval for evaluation
    },
    
    # PPO specific settings
    'ppo': {
        'learning_rate': 0.0003,       # Learning rate for PPO
        'n_steps': 2048,               # Steps per update
        'batch_size': 64,              # Batch size for training
        'n_epochs': 10,                # Number of epochs when optimizing the surrogate loss
        'gamma': 0.99,                 # Discount factor
        'gae_lambda': 0.95,            # Generalized Advantage Estimation lambda
        'clip_range': 0.2,             # Clipping parameter for PPO
        'clip_range_vf': None,         # Clipping parameter for value function
        'ent_coef': 0.01,              # Entropy coefficient
        'vf_coef': 0.5,                # Value function coefficient
        'max_grad_norm': 0.5,          # Maximum gradient norm
        'total_timesteps': 500000,     # Total timesteps for training
        'log_interval': 5,             # Interval for logging
        'policy_kwargs': {
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
        }
    }
}

# ---------------------------------------------------
# Trading Environment Configuration
# ---------------------------------------------------
TRADING_ENV_SETTINGS = {
    'lookback_window': 20,           # Number of time steps observable by agent
    'initial_balance': 10000,        # Initial account balance
    'commission': 0.001,             # Transaction cost (as fraction)
    'reward_function': 'profit_and_loss',  # Default reward function
    'reward_params': {               # Parameters for reward function
        'scale': 1.0
    },
    'max_position_size': 1.0,        # Maximum position size as fraction of balance
    'normalize_rewards': True,       # Whether to normalize rewards
    'normalize_observations': True,  # Whether to normalize observations
    'action_space': 'discrete',      # Action space type: 'discrete' or 'continuous'
    'discrete_actions': 3,           # Number of discrete actions (buy, sell, hold)
    'include_features': True,        # Whether to include pre-calculated features
    'include_position_info': True,   # Whether to include position info in state
}

# ---------------------------------------------------
# Progressive Learning Configuration
# ---------------------------------------------------
PROGRESSIVE_LEARNING = {
    # Success criteria for each stage
    'time_series_success_threshold': 0.6,  # Minimum R² score to proceed to DQN
    'dqn_success_threshold': 0.1,          # Minimum avg. return to proceed to PPO
    
    # Default stage configurations
    'time_series': {
        'sequence_length': 60,
        'forecast_horizon': 5,
        'target_column': 'close',
        'model_type': 'lstm',
        'epochs': 50,
    },
    
    'dqn': {
        'timesteps': 100000,
        'use_time_series': True,     # Whether to use time series predictions
    },
    
    'ppo': {
        'timesteps': 500000,
        'use_dqn_initialization': True,  # Whether to initialize from DQN
    }
}

def get_config(section: str = None) -> Dict[str, Any]:
    """
    Get configuration settings.
    
    Parameters:
    -----------
    section : str, optional
        Section of config to return. If None, returns all configs.
        
    Returns:
    --------
    Dict[str, Any]
        Configuration dictionary
    """
    all_configs = {
        'paths': PATHS,
        'data': DATA_SETTINGS,
        'features': FEATURE_SETTINGS,
        'time_series': TIME_SERIES_SETTINGS,
        'rl': RL_SETTINGS,
        'environment': TRADING_ENV_SETTINGS,
        'progressive_learning': PROGRESSIVE_LEARNING
    }
    
    if section is None:
        return all_configs
    elif section in all_configs:
        return all_configs[section]
    else:
        raise ValueError(f"Configuration section '{section}' not found.") 