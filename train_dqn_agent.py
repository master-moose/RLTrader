#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a DQN agent for cryptocurrency trading using the LSTM model for state representation.
With FinRL integration for improved GPU utilization.
"""

import os
import argparse
import json
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import psutil
import h5py
import pandas as pd
import sys
import gym
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.data_processors.processor_cryptocurrencies import (
    CryptocurrencyTradingEnv,
)
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.agents.stablebaselines3 import DRLAgent
from finrl.config import INDICATORS
from stable_baselines3.common.logger import configure

from utils.notifications import send_telegram_notification
from utils.logs import setup_logging
from utils.hdf5_utils import load_data_from_hdf5
from utils.time_measurement import timeit
from utils.device_info import get_device_info

# Original imports
from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define trade evaluation constants
GOOD_TRADE_THRESHOLD = 0.001  # 0.1% profit threshold for a good trade
BAD_TRADE_THRESHOLD = -0.001  # -0.1% loss threshold for a bad trade

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a DQN agent for cryptocurrency trading')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                        help='Directory containing the training data')
    parser.add_argument('--lstm_model_path', type=str, required=True,
                        help='Path to the trained LSTM model checkpoint')
    parser.add_argument('--output_dir', type=str, default='models/dqn',
                        help='Directory to save trained models and results')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--episode_length', type=int, default=10000,
                        help='Length of each episode in steps (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for experience replay')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Size of the replay buffer')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.998,
                        help='Exploration rate decay factor')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate for the Q-network')
    parser.add_argument('--target_update', type=int, default=10,
                        help='Frequency of target network updates (episodes)')
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='Number of network updates per environment step')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Environment parameters
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for observations')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for trading')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage')
    parser.add_argument('--reward_scaling', type=float, default=0.001,
                        help='Scaling factor for rewards')
    parser.add_argument('--trade_cooldown', type=int, default=12,
                        help='Number of steps between trades')
    parser.add_argument('--primary_timeframe', type=str, default='1h',
                        help='Primary timeframe for trading')
    parser.add_argument('--use_indicators', action='store_true',
                        help='Whether to use technical indicators')
    parser.add_argument('--use_position_features', action='store_true',
                        help='Whether to include position features in state')
    parser.add_argument('--lookback_window', type=int, default=20,
                        help='Lookback window for LSTM features')
    
    # Parallelization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel environments to run')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda, None for auto-detection)')
    
    # Performance options
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision for faster training')
    
    # Saving and evaluation options
    parser.add_argument('--save_dir', type=str, default='models/dqn',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Frequency to save model checkpoints (episodes)')
    parser.add_argument('--eval_interval', type=int, default=0,
                        help='Frequency to evaluate model (episodes, 0 to disable)')
    
    # FinRL specific options
    parser.add_argument('--use_finrl', action='store_true',
                        help='Use FinRL framework for training')
    parser.add_argument(
        '--finrl_model', 
        type=str, 
        choices=['dqn', 'ppo', 'a2c', 'ddpg', 'td3', 'sac'],
        default='dqn', 
        help='FinRL algorithm to use'
    )
    parser.add_argument('--net_arch', type=str, default='[256,256]',
                        help='Network architecture for FinRL models')
    parser.add_argument('--tensorboard_log', type=str, default='./tensorboard_logs',
                        help='TensorBoard log directory')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total timesteps for FinRL training')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def load_lstm_model(model_path, device):
    """Load the trained LSTM model."""
    try:
        logger.info(f"Loading LSTM model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found at {model_path}")
        
        # Load model configuration
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Loaded model configuration from config.json")
        else:
            logger.warning("No config.json found, using default configuration")
            config = {
                "model": {
                    "hidden_dims": 64,
                    "num_layers": 1,
                    "dropout": 0.7,
                    "bidirectional": True,
                    "attention": False,
                    "num_classes": 3,
                    "use_batch_norm": True
                }
            }
        
        # Create a new model instance with the saved configuration
        model = MultiTimeframeModel(
            input_dims={"15m": 34, "4h": 34, "1d": 34},  # Match dataset feature count
            hidden_dims=config["model"]["hidden_dims"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            bidirectional=config["model"]["bidirectional"],
            attention=config["model"]["attention"],
            num_classes=config["model"]["num_classes"],
            use_batch_norm=config["model"].get("use_batch_norm", True)
        )
        
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle missing/extra keys
        logger.info("Model state dict loaded successfully")
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        logger.info("LSTM model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading LSTM model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def prepare_crypto_data_for_finrl(market_data, primary_timeframe=None):
    """
    Convert crypto data to format that FinRL can use.
    
    Parameters:
    -----------
    market_data : dict
        Dictionary containing market data for different timeframes
    primary_timeframe : str, optional
        Primary timeframe to use, defaults to '15m' if available, 
        otherwise selects the shortest available timeframe
        
    Returns:
    --------
    df : pd.DataFrame
        Processed data in FinRL format
    """
    # Default to 15m if available, otherwise use the first available timeframe
    available_timeframes = list(market_data.keys())
    logger.info(f"Available timeframes: {available_timeframes}")
    
    if primary_timeframe is None:
        # Prefer 15m as base timeframe if available
        if '15m' in available_timeframes:
            primary_timeframe = '15m'
            logger.info(f"Using '15m' as the primary timeframe")
        else:
            # Sort timeframes and use the shortest one as base
            try:
                # Try to sort by timeframe duration
                timeframe_minutes = {}
                for tf in available_timeframes:
                    if tf.endswith('m'):
                        timeframe_minutes[tf] = int(tf[:-1])
                    elif tf.endswith('h'):
                        timeframe_minutes[tf] = int(tf[:-1]) * 60
                    elif tf.endswith('d'):
                        timeframe_minutes[tf] = int(tf[:-1]) * 60 * 24
                
                # Sort by duration and get the shortest timeframe
                primary_timeframe = sorted(timeframe_minutes.items(), key=lambda x: x[1])[0][0]
            except:
                # If sorting fails, just use the first timeframe
                primary_timeframe = available_timeframes[0]
            
            logger.info(f"Using '{primary_timeframe}' as the primary timeframe")
    
    # Check if the requested timeframe exists, otherwise use a fallback
    if primary_timeframe not in market_data:
        logger.warning(f"Requested timeframe '{primary_timeframe}' not found in data.")
        # Use the first available timeframe as fallback
        primary_timeframe = available_timeframes[0]
        logger.info(f"Using '{primary_timeframe}' as the fallback timeframe")
    
    # Get the primary timeframe data
    df = market_data[primary_timeframe].copy()
    
    # Rename columns to match FinRL expectations
    # Assuming the dataframe has price columns like 'open', 'high', 'low', 'close'
    column_map = {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }
    
    df = df.rename(columns={old: new for old, new in column_map.items() if old in df.columns})
    
    # Add required columns for FinRL
    if 'tic' not in df.columns:
        df['tic'] = 'CRYPTO'
    
    # Check if we need to reset the index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Reset the index to get the index as a column if it's not already a column
        df = df.reset_index()
    
    # Make sure we have a date column
    if 'date' not in df.columns:
        # Create a date column from timestamp or index, or create a synthetic one
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
        elif 'index' in df.columns and pd.api.types.is_datetime64_any_dtype(df['index']):
            df['date'] = pd.to_datetime(df['index'])
        else:
            # Create synthetic dates
            logger.info("Creating synthetic dates for the data")
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1h')
    
    # Ensure date is a datetime type
    df['date'] = pd.to_datetime(df['date'])
    
    # Manually add some technical indicators to avoid FinRL preprocessing errors
    logger.info("Adding basic technical indicators manually")
    # Calculate technical indicators on a safe copy to avoid SettingWithCopyWarning
    df_safe = df.copy()
    df_safe['macd'] = df_safe['close'].ewm(span=12).mean() - df_safe['close'].ewm(span=26).mean()
    df_safe['rsi'] = 100 - (100 / (1 + (
        df_safe['close'].diff().clip(lower=0).rolling(window=14).sum() / 
        abs(df_safe['close'].diff().clip(upper=0)).rolling(window=14).sum()
    )))
    df_safe['cci'] = (
        df_safe['close'] - df_safe['close'].rolling(window=20).mean()
    ) / (0.015 * df_safe['close'].rolling(window=20).std())
    df_safe['dx'] = abs(df_safe['close'].diff()) / df_safe['close'] * 100
    
    # Copy calculated indicators back to original dataframe
    for col in ['macd', 'rsi', 'cci', 'dx']:
        df[col] = df_safe[col]
    
    # Set the index to date for FinRL compatibility
    # If the date column already exists as index, this will be a no-op
    if df.index.name != 'date':
        df = df.set_index('date')
    
    # Try to use FinRL's feature engineering
    try:
        # Create FeatureEngineer
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        
        try:
            logger.info("Attempting to preprocess data with FinRL's FeatureEngineer")
            # Reset index but drop the index column to avoid duplicate 'date'
            # Then FinRL will process the date column that already exists
            temp_df = df.reset_index(drop=True)
            processed = fe.preprocess_data(temp_df)
            logger.info(f"Processed data with shape: {processed.shape}")
            
            # Ensure we have a date column
            if 'date' not in processed.columns:
                logger.warning("FinRL preprocessing removed date column, adding it back")
                processed['date'] = df.index.values
                
            return processed
        except Exception as e:
            logger.error(f"Error in FinRL preprocessing: {e}")
            logger.error(traceback.format_exc())
            
            # Return our manually preprocessed dataframe
            # Reset index with drop=True to avoid duplicate 'date' column
            df_result = df.copy()
            # Make sure date column exists after reset
            df_result = df_result.reset_index()
            # Rename duplicated date column if needed
            if 'date' in df_result.columns and df_result.index.name == 'date':
                df_result = df_result.rename_axis(None)
            return df_result
    except Exception as e:
        logger.error(f"Error setting up feature engineering: {e}")
        # Return dataframe with added date and technical indicators
        logger.info("Using basic dataframe with manual indicators")
        # Reset the index and ensure we return a dataframe with a date column
        try:
            # Try first with dropping the index to avoid duplicates
            result_df = df.reset_index(drop=False)
            # If date column already exists as regular column (not just index)
            if 'date' in result_df.columns and result_df.index.name == 'date':
                # Rename the index to avoid conflicts on next reset
                result_df = result_df.rename_axis(None)
            return result_df
        except ValueError:
            # If that fails, try with a different approach
            # Copy to avoid modifying the original during reset
            result_df = df.copy()
            # If we have a date index and a date column, rename the column
            if result_df.index.name == 'date' and 'date' in result_df.columns:
                result_df = result_df.rename(columns={'date': 'date_col'})
            # Then reset index safely
            return result_df.reset_index()

def create_finrl_env(processed_data, args, env_id=0):
    """
    Create a FinRL environment from processed data.
    
    Parameters:
    -----------
    processed_data : pd.DataFrame
        Processed data from prepare_crypto_data_for_finrl
    args : argparse.Namespace
        Command line arguments
    env_id : int
        Environment ID
        
    Returns:
    --------
    env : gym.Env
        FinRL environment
    """
    # Define the stock dimension (number of unique stocks/cryptos)
    stock_dimension = len(processed_data['tic'].unique())
    
    # Define state space dimension
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    if args.use_position_features:
        state_space += 2  # Add position and unrealized PnL
        
    # Define action space dimension
    if args.finrl_model in ['ddpg', 'td3', 'sac']:
        action_space = 1  # Continuous action space for these algorithms
    else:
        action_space = 3  # buy, hold, sell (discrete)
    
    # Define environment parameters
    env_kwargs = {
        "hmax": 100,  # Max number of shares to trade
        "initial_amount": args.initial_balance,
        "buy_cost_pct": args.transaction_fee,
        "sell_cost_pct": args.transaction_fee,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": action_space,
        "reward_scaling": args.reward_scaling,
        "window_size": args.window_size
    }
    
    # Process data for environment
    train_data = processed_data.copy()
    
    # Ensure we have a date column
    if 'date' not in train_data.columns:
        logger.warning("No date column found in processed data, creating one")
        train_data['date'] = pd.date_range(start='2020-01-01', periods=len(train_data), freq='1h')
    
    # Sort by date and tic
    try:
        train_data = train_data.sort_values(['date', 'tic'], ignore_index=True)
    except Exception as e:
        logger.error(f"Error sorting data: {e}")
        # If the sort fails, just reset the index
        train_data = train_data.reset_index(drop=True)
    
    # Create an integer index from date values for FinRL
    try:
        train_data.index = train_data['date'].factorize()[0]
    except Exception as e:
        logger.error(f"Error setting factorized index: {e}")
        # Use simple sequential index if factorize fails
        train_data.index = np.arange(len(train_data))
    
    # Create environment
    try:
        env = CryptocurrencyTradingEnv(df=train_data, **env_kwargs)
        
        # For single process environment
        env_train = DummyVecEnv([lambda: env])
        
        return env_train
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        logger.error(traceback.format_exc())
        raise

def train_dqn_agent(args):
    """
    Train the DQN agent.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Set seeds for reproducibility
    if args.use_finrl:
        set_random_seed(args.seed)
    else:
        set_seeds(args.seed)
    
    # Set device
    use_cuda = torch.cuda.is_available() and (args.device == 'cuda' or args.use_amp)
    device = "cuda" if use_cuda else "cpu"
    logger.info(f"Using device: {device}")
    
    # Advanced GPU optimizations
    if device == "cuda":
        # Enable TF32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} with {gpu_memory:.1f} GB memory")

    # Load data from specified path using h5py
    data_path = args.data_dir
    logger.info(f"Loading data from {data_path}")
    
    # Check which h5 file to use
    h5_file_path = os.path.join(data_path, "synthetic_dataset.h5")
    if not os.path.exists(h5_file_path):
        h5_file_path = os.path.join(data_path, "train_data.h5")
        
    if not os.path.exists(h5_file_path):
        raise ValueError(f"No HDF5 data file found in {data_path}")
    
    # Load data from h5 file
    market_data = {}
    with h5py.File(h5_file_path, 'r') as h5f:
        timeframes = list(h5f.keys())
        logger.info(f"Found timeframes: {timeframes}")
        
        for tf in timeframes:
            # Get the group for this timeframe
            group = h5f[tf]
            
            # Check if the group has a 'table' dataset
            if 'table' not in group:
                logger.error(f"No 'table' dataset found in group {tf}")
                continue
            
            # Get the table dataset
            table = group['table']
            logger.info(f"Found table dataset for {tf} with shape {table.shape} and dtype {table.dtype}")
            
            try:
                # Convert the structured array to a pandas DataFrame
                # The structured array has a field for each column
                data = table[:]  # Read the entire dataset
                
                # Create a DataFrame from the structured array
                df = pd.DataFrame(data)
                
                # Set the index column if it exists
                if 'index' in df.columns:
                    df.set_index('index', inplace=True)
                
                # Convert timestamp to datetime if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                market_data[tf] = df
                logger.info(f"Successfully loaded data for {tf}: {df.shape}")
                
            except Exception as e:
                logger.error(f"Error loading data for timeframe {tf}: {e}")
                logger.error(f"Table info: shape={table.shape}, dtype={table.dtype}")
                raise
    
    # Check if we successfully loaded any data
    if not market_data:
        raise ValueError("Failed to load market data from H5 files")
        
    data_length = len(next(iter(market_data.values())))  # Get length from first timeframe
    logger.info(f"Data loaded, {data_length} samples found")

    # Choose between FinRL and custom implementation
    if args.use_finrl:
        logger.info("Using FinRL framework for training")
        return train_with_finrl(args, market_data, device)
    else:
        logger.info("Using custom DQN implementation for training")
        return train_with_custom_dqn(args, market_data, data_length, device)

def train_with_finrl(args, market_data, device):
    """
    Train a reinforcement learning agent for cryptocurrency trading using FinRL.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    market_data : dict
        Dictionary containing market data for different timeframes
    device : str
        Device to use for training ('cpu' or 'cuda')
        
    Returns:
    --------
    agent : FinRLAgent
        Trained FinRL agent
    """
    logger.info("Using FinRL framework for training")
    
    # Process data for FinRL
    # For FinRL, we primarily use the base timeframe (15m if available)
    # But we'll also incorporate features from higher timeframes later
    processed_data = prepare_crypto_data_for_finrl(market_data, args.primary_timeframe)
    
    # Create environment
    env_train = create_finrl_env(processed_data, args, env_id=0)
    
    # Setup agent parameters
    if args.finrl_model in ['sac', 'td3', 'ddpg']:
        logger.info(f"Using continuous action space for {args.finrl_model}")
        action_noise = True
    else:
        logger.info(f"Using discrete action space for {args.finrl_model}")
        action_noise = False
    
    # Parse net_arch from string
    try:
        net_arch = eval(args.net_arch)  # Convert string to list
    except:
        logger.warning(f"Could not parse net_arch: {args.net_arch}, using default [256, 256]")
        net_arch = [256, 256]
    
    # Create model parameters
    model_params = {
        'learning_rate': args.learning_rate,
        'verbose': 1 if args.verbose else 0,
        'policy': 'MlpPolicy',
        'device': device
    }
    
    # Add specific parameters based on the model
    if args.finrl_model in ['sac', 'td3', 'ddpg']:
        # Continuous action space models might need action noise
        if action_noise:
            model_params['action_noise'] = 'normal'
            
    # Set network architecture if provided
    if net_arch:
        # For SAC and other continuous models, we might need to specify more details
        if args.finrl_model == 'sac':
            model_params['policy_kwargs'] = {
                'net_arch': {
                    'pi': net_arch,  # Policy network
                    'qf': net_arch   # Q-function network
                }
            }
        else:
            # For other models, simpler net_arch specification
            model_params['policy_kwargs'] = {'net_arch': net_arch}

    # Create and train the model
    try:
        logger.info(f"Creating {args.finrl_model.upper()} model with FinRL")
        agent = FinRLAgent(env=env_train)
        
        # Get the model based on the selected algorithm
        model = agent.get_model(args.finrl_model, model_kwargs=model_params)
        
        # Train the model
        logger.info(f"Training {args.finrl_model.upper()} model for {args.total_timesteps} timesteps...")
        trained_model = agent.train_model(model=model, 
                                         tb_log_name=f"{args.finrl_model}",
                                         total_timesteps=args.total_timesteps)
        
        # Save the trained model
        model_save_path = os.path.join(args.save_dir, f"{args.finrl_model}_model")
        trained_model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return trained_model
    except Exception as e:
        logger.error(f"Error during FinRL training: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def train_with_custom_dqn(args, market_data, data_length, device):
    """
    Train using custom DQN implementation.
    This is the original training code.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    market_data : dict
        Market data dictionary with different timeframes
    data_length : int
        Length of data
    device : str
        Device to use (cpu or cuda)
    
    Returns:
    --------
    episode_rewards : list
        List of episode rewards
    episode_balances : list
        List of episode balances
    episode_trade_counts : list
        List of episode trade counts
    """
    # Create multiple environments for parallel training
    logger.info(f"Creating {args.num_workers} environments")
    envs = []
    for i in range(args.num_workers):
        env = TradingEnvironment(
            data_path=os.path.join(args.data_dir, "train_data.h5"),
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            reward_scaling=args.reward_scaling,
            use_position_features=args.use_position_features,
            lookback_window=args.lookback_window,
            trade_cooldown=args.trade_cooldown,
            device=device
        )
        envs.append(env)
    
    # Get state and action dimensions
    state_dim = envs[0]._calculate_state_dim()
    action_dim = 3  # hold, buy, sell
    logger.info(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Create DQN agent
    logger.info("Creating DQN agent")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[512, 256],  # Increase network capacity
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=device,
        verbose=True
    )
    
    # Use torch.compile for faster network computations if available
    if hasattr(torch, 'compile') and device == "cuda":
        logger.info("Using torch.compile to optimize networks")
        agent.policy_net = torch.compile(agent.policy_net, mode="reduce-overhead")
        agent.target_net = torch.compile(agent.target_net, mode="reduce-overhead")
    
    # Initialize metrics
    episode_rewards = []
    episode_balances = []
    episode_trade_counts = []
    
    # Add trade quality tracking
    good_trades_count = 0
    bad_trades_count = 0
    trade_returns = []  # Track returns of each trade for overfitting detection
    
    # Save directory
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Model checkpoints will be saved to {save_dir}")
    
    # Training loop
    logger.info("Starting training...")
    total_updates = 0
    total_experiences = 0
    update_counter = 0
    
    # Pre-allocate tensors for states to avoid repeated allocations
    states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    next_states_tensor = torch.zeros((args.num_workers, state_dim), device=device)
    
    # Pre-allocate replay buffer tensors with dynamic sizing capability
    def get_optimal_batch_size():
        """Dynamically determine optimal batch size based on available memory"""
        if device == "cuda":
            # Get available GPU memory (in bytes)
            free_mem = torch.cuda.mem_get_info()[0]
            # Estimate memory per sample (state_dim * 4 bytes per float32)
            mem_per_sample = state_dim * 4 * 4  # states, actions, rewards, next_states
            # Use up to 40% of free memory for batches
            return min(args.batch_size, max(64, int(free_mem * 0.4 / mem_per_sample)))
        else:
            # On CPU, consider system memory
            free_mem = psutil.virtual_memory().available
            mem_per_sample = state_dim * 4 * 4
            return min(args.batch_size, max(64, int(free_mem * 0.4 / mem_per_sample)))
    
    optimal_batch_size = get_optimal_batch_size()
    logger.info(f"Using optimal batch size: {optimal_batch_size}")
    
    # Helper function to step environments in parallel
    def step_environments_in_parallel(envs, actions, dones):
        """Execute steps across multiple environments in parallel"""
        next_states = []
        rewards = []
        new_dones = []
        infos = []
        
        # Only process active environments
        active_indices = [i for i, done in enumerate(dones) if not done]
        
        # Use threads to parallelize environment steps
        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), len(active_indices))) as executor:
            futures = []
            for i in active_indices:
                futures.append(executor.submit(envs[i].step, actions[i].item()))
            
            # Collect results
            results = [future.result() for future in futures]
            
            # Distribute results back to full lists
            result_idx = 0
            for i in range(len(envs)):
                if i in active_indices:
                    ns, r, d, info = results[result_idx]
                    next_states.append(ns)
                    rewards.append(r)
                    new_dones.append(d)
                    infos.append(info)
                    result_idx += 1
                else:
                    # Keep existing state for done environments
                    next_states.append(None)
                    rewards.append(0)
                    new_dones.append(True)
                    infos.append({})
        
        return next_states, rewards, new_dones, infos
    
    # Enable CUDA graph for faster training
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Determine optimal update frequency
    update_frequency = 16  # Update less frequently to batch more updates
    
    for episode in range(1, args.episodes + 1):
        # Reset all environments with random starting points
        states = []
        for env in envs:
            random_start = np.random.randint(
                args.window_size + 1, 
                data_length - args.episode_length - 100
            )
            env.start_step = random_start
            states.append(env.reset())
            
        # Convert states to tensor for faster processing
        for i, state in enumerate(states):
            if isinstance(state, np.ndarray):
                states_tensor[i] = torch.from_numpy(state).float()
            else:
                states_tensor[i] = state
                
        dones = [False] * args.num_workers
        episode_rewards_per_env = [0] * args.num_workers
        steps_per_env = [0] * args.num_workers
        active_envs = args.num_workers
        
        # Track trade quality for this episode
        episode_good_trades = 0
        episode_bad_trades = 0
        episode_neutral_trades = 0
        last_trade_prices = [None] * args.num_workers
        last_positions = [0] * args.num_workers
        
        # Episode loop
        progress_bar = tqdm(total=args.episode_length, 
                           desc=f"Episode {episode}/{args.episodes}", 
                           disable=False)
        
        step_counter = 0
        while active_envs > 0 and step_counter < args.episode_length:
            # Select actions for all environments at once using vectorized operations
            with torch.no_grad():
                q_values = agent.policy_net(states_tensor)
                # Use inplace operations where possible
                actions = torch.where(
                    torch.rand(args.num_workers, device=device) < agent.epsilon,
                    torch.randint(0, action_dim, (args.num_workers,), device=device),
                    q_values.argmax(dim=1)
                )
            
            # Step all environments in parallel
            next_states, rewards, new_dones, infos = step_environments_in_parallel(
                envs, actions, dones
            )
            
            # Process environment step results
            balances = []
            trade_counts = []
            
            # Pre-process batch data collection to reduce redundant tensor operations
            transitions_to_add = []
            
            for i in range(args.num_workers):
                if not dones[i]:
                    balances.append(infos[i].get('balance', 0))
                    trade_counts.append(infos[i].get('total_trades', 0))
                    
                    # Track trade quality
                    current_position = infos[i].get('position', 0)
                    current_price = infos[i].get('price', 0)
                    
                    # Check if a trade was made
                    if current_position != last_positions[i]:
                        # A position change occurred (trade was made)
                        if last_trade_prices[i] is not None and last_positions[i] != 0:
                            # Calculate trade return only when closing a position
                            if current_position == 0:
                                # Calculate return based on position direction
                                if last_positions[i] > 0:  # Was long, now closed
                                    trade_return = (current_price / last_trade_prices[i]) - 1 - args.transaction_fee
                                else:  # Was short, now closed
                                    trade_return = (last_trade_prices[i] / current_price) - 1 - args.transaction_fee
                                
                                # Classify the trade
                                if trade_return > GOOD_TRADE_THRESHOLD:
                                    episode_good_trades += 1
                                    good_trades_count += 1
                                    logger.debug(f"Environment {i} - GOOD TRADE: Return={trade_return:.4f}, "
                                                f"Entry={last_trade_prices[i]:.2f}, Exit={current_price:.2f}, "
                                                f"Position={last_positions[i]}")
                                elif trade_return < BAD_TRADE_THRESHOLD:
                                    episode_bad_trades += 1
                                    bad_trades_count += 1
                                    logger.debug(f"Environment {i} - BAD TRADE: Return={trade_return:.4f}, "
                                                f"Entry={last_trade_prices[i]:.2f}, Exit={current_price:.2f}, "
                                                f"Position={last_positions[i]}")
                                else:
                                    episode_neutral_trades += 1
                                    logger.debug(f"Environment {i} - NEUTRAL TRADE: Return={trade_return:.4f}, "
                                                f"Entry={last_trade_prices[i]:.2f}, Exit={current_price:.2f}, "
                                                f"Position={last_positions[i]}")
                                
                                trade_returns.append(trade_return)
                                
                        # Update last trade price when opening a new position
                        if current_position != 0:
                            last_trade_prices[i] = current_price
                    
                    # Update last position
                    last_positions[i] = current_position
                    
                    episode_rewards_per_env[i] += rewards[i]
                    steps_per_env[i] += 1
                    
                    # Collect transitions for batch addition to memory
                    transitions_to_add.append((states[i], actions[i].item(), rewards[i], next_states[i], new_dones[i]))
                    
                    if isinstance(next_states[i], np.ndarray):
                        next_states_tensor[i] = torch.from_numpy(next_states[i]).float()
                    else:
                        next_states_tensor[i] = next_states[i]
                else:
                    balances.append(0)
                    trade_counts.append(0)
            
            # Batch add transitions to memory
            for transition in transitions_to_add:
                agent.memory.append(transition)
                total_experiences += 1
                
            # Log balance changes for debugging
            if step_counter % 100 == 0 and active_envs > 0:
                i = next((i for i, d in enumerate(dones) if not d), 0)
                logger.info(f"Environment {i} - Step {step_counter}: Action={actions[i].item()}, "
                           f"Balance={infos[i].get('balance', 0):.2f}, "
                           f"Position={infos[i].get('position', 0)}, "
                           f"Trades={infos[i].get('total_trades', 0)}, "
                           f"Price={infos[i].get('price', 0):.2f}, "
                           f"Good/Bad/Neutral={episode_good_trades}/{episode_bad_trades}/{episode_neutral_trades}")
            
            # Update states and dones
            states = next_states
            dones = new_dones
            states_tensor.copy_(next_states_tensor)
            
            # Count active environments
            active_envs = sum(1 for d in dones if not d)
            
            # Batch updates every few steps for better efficiency
            if step_counter % update_frequency == 0 and len(agent.memory) >= optimal_batch_size:
                # Perform multiple updates in a batch
                for _ in range(args.updates_per_step):  # Simplified update loop
                    # Dynamically adjust batch size based on available replay buffer samples
                    current_batch_size = min(len(agent.memory), optimal_batch_size)
                    
                    # Sample batch indices - ensure we don't sample more than available items
                    indices = np.random.choice(len(agent.memory), current_batch_size, replace=False)
                    batch = [agent.memory[i] for i in indices]
                    
                    # Convert batch to tensors efficiently - use a single loop
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
                    for i, (state, action, reward, next_state, done) in enumerate(batch):
                        batch_states.append(torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state)
                        batch_actions.append(action)
                        batch_rewards.append(reward)
                        batch_next_states.append(torch.from_numpy(next_state).float() if isinstance(next_state, np.ndarray) else next_state)
                        batch_dones.append(done)
                    
                    # Stack tensors at once instead of individual assignments
                    replay_states = torch.stack(batch_states)
                    replay_actions = torch.tensor(batch_actions, device=device)
                    replay_rewards = torch.tensor(batch_rewards, device=device)
                    replay_next_states = torch.stack(batch_next_states)
                    replay_dones = torch.tensor(batch_dones, device=device)
                    
                    # Update networks using vectorized operations
                    with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                        current_q_values = agent.policy_net(replay_states).gather(1, replay_actions.unsqueeze(1))
                        with torch.no_grad():
                            next_q_values = agent.target_net(replay_next_states).max(1)[0]
                            expected_q_values = replay_rewards + (1 - replay_dones.float()) * agent.gamma * next_q_values
                        
                        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
                    
                    # Optimize
                    agent.optimizer.zero_grad()
                    if agent.scaler is not None:
                        agent.scaler.scale(loss).backward()
                        agent.scaler.step(agent.optimizer)
                        agent.scaler.update()
                    else:
                        loss.backward()
                        agent.optimizer.step()
                    
                    total_updates += 1
                    update_counter += 1
            
            # Calculate portfolio values (including unrealized gains/losses)
            portfolio_values = []
            for i, env in enumerate(envs):
                if not dones[i]:
                    # Get current price and calculate portfolio value
                    current_price = env._get_current_price()
                    portfolio_value = env._calculate_portfolio_value(current_price)
                    portfolio_values.append(portfolio_value)
            
            avg_portfolio = sum(portfolio_values) / max(1, len(portfolio_values)) if portfolio_values else 0
            
            # Update progress bar with average metrics
            avg_reward = sum(episode_rewards_per_env) / max(1, active_envs)
            avg_balance = sum(balances) / max(1, active_envs) if balances else 0
            avg_trades = sum(trade_counts) / max(1, active_envs) if trade_counts else 0
            
            gpu_util = ""
            if device == "cuda" and step_counter % 50 == 0:
                # Get GPU utilization percentage
                try:
                    gpu_mem = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory * 100
                    gpu_util = f", GPU:{gpu_mem:.1f}%"
                except Exception:
                    pass
            
            # Initialize current_batch_size for the progress bar
            current_batch_size = optimal_batch_size
            
            progress_bar.set_postfix({
                'reward': f"{avg_reward:.2f}",
                'balance': f"{avg_balance:.2f}",
                'portfolio': f"{avg_portfolio:.2f}",
                'trades': f"{avg_trades:.0f}",
                'active_envs': active_envs,
                'updates': total_updates,
                'info': f"bs:{current_batch_size}{gpu_util}"
            })
            progress_bar.update(1)
            
            step_counter += 1
        
        progress_bar.close()
        
        # Episode summary
        episode_reward = sum(episode_rewards_per_env) / args.num_workers
        episode_profit = sum(env.balance - args.initial_balance for env in envs) / args.num_workers
        episode_trade_count = sum(env.info.get('total_trades', 0) if hasattr(env, 'info') else 0 for env in envs) / args.num_workers
        
        # Calculate trade quality metrics
        total_trades = episode_good_trades + episode_bad_trades + episode_neutral_trades
        good_trade_pct = (episode_good_trades / max(1, total_trades)) * 100
        bad_trade_pct = (episode_bad_trades / max(1, total_trades)) * 100
        
        # Detect overfitting by analyzing trade returns distribution
        if len(trade_returns) > 20:  # Need a minimum number of trades for analysis
            recent_returns = trade_returns[-20:]  # Look at recent trades
            avg_return = sum(recent_returns) / len(recent_returns)
            std_return = np.std(recent_returns) if len(recent_returns) > 1 else 0
            
            # Calculate Sharpe-like ratio (return / volatility)
            sharpe_ratio = avg_return / (std_return + 1e-6)  # Add small constant to avoid division by zero
            
            # Check for overfitting signals
            if avg_return < -0.005 and sharpe_ratio < -0.5:
                logger.warning(f"Potential overfitting detected at episode {episode}: "
                             f"Average return: {avg_return:.4f}, StdDev: {std_return:.4f}, "
                             f"Sharpe: {sharpe_ratio:.2f}")
                
                if args.verbose:
                    logger.info(f"Recent trade returns: {recent_returns}")
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_balances.append(episode_profit)
        episode_trade_counts.append(episode_trade_count)
        
        # Update networks and exploration
        if episode % args.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            logger.info(f"Target network updated at episode {episode}")
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Save model
        if episode % args.save_interval == 0 or episode == args.episodes:
            model_path = os.path.join(save_dir, f"dqn_agent_episode_{episode}.pt")
            state_dict = {
                'policy_net': agent.policy_net.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'episode': episode,
                'epsilon': agent.epsilon,
            }
            torch.save(state_dict, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'episode_balances': episode_balances,
                'episode_trade_counts': episode_trade_counts,
            }
            metrics_path = os.path.join(save_dir, f"metrics_episode_{episode}.pt")
            torch.save(metrics, metrics_path)
        
        # Log episode results
        logger.info(f"Episode {episode}/{args.episodes} - "
                   f"Reward: {episode_reward:.2f}, Profit: {episode_profit:.2f}, "
                   f"Trades: {episode_trade_count:.2f}, Good/Bad: {episode_good_trades}/{episode_bad_trades} "
                   f"({good_trade_pct:.1f}%/{bad_trade_pct:.1f}%), "
                   f"Epsilon: {agent.epsilon:.4f}, "
                   f"Memory: {len(agent.memory)}, Updates: {total_updates}")
        
        # Evaluate model every few episodes
        if args.eval_interval > 0 and episode % args.eval_interval == 0:
            # TODO: Implement evaluation
            pass
    
    # Final trading statistics
    if len(trade_returns) > 0:
        avg_return_per_trade = sum(trade_returns) / len(trade_returns)
        winning_rate = good_trades_count / max(1, good_trades_count + bad_trades_count) * 100
        
        logger.info("Trading Statistics Summary:")
        logger.info(f"Total Trades: {len(trade_returns)}")
        logger.info(f"Good Trades: {good_trades_count} ({winning_rate:.1f}%)")
        logger.info(f"Bad Trades: {bad_trades_count} ({100-winning_rate:.1f}%)")
        logger.info(f"Average Return per Trade: {avg_return_per_trade:.4f}")
        
        if args.verbose:
            # Calculate additional statistics
            max_return = max(trade_returns) if trade_returns else 0
            min_return = min(trade_returns) if trade_returns else 0
            std_return = np.std(trade_returns) if len(trade_returns) > 1 else 0
            
            logger.info(f"Best Trade: {max_return:.4f}")
            logger.info(f"Worst Trade: {min_return:.4f}")
            logger.info(f"Return StdDev: {std_return:.4f}")
            logger.info(f"Sharpe-like Ratio: {(avg_return_per_trade / (std_return + 1e-6)):.2f}")
            
    logger.info("Training completed!")
    return episode_rewards, episode_balances, episode_trade_counts

def plot_training_results(rewards, profits, trades, output_dir):
    """
    Plot and save training results.
    
    Parameters:
    -----------
    rewards : list
        List of episode rewards
    profits : list
        List of episode profits
    trades : list
        List of episode trades
    output_dir : str
        Directory to save plots
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot profits
    ax2.plot(profits)
    ax2.set_title('Episode Profits')
    ax2.set_ylabel('Profit')
    ax2.grid(True)
    
    # Plot trades
    ax3.plot(trades)
    ax3.set_title('Episode Trades')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Trades')
    ax3.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    plt.close()
    
    # Save data as numpy arrays
    np.save(os.path.join(output_dir, 'rewards.npy'), np.array(rewards))
    np.save(os.path.join(output_dir, 'profits.npy'), np.array(profits))
    np.save(os.path.join(output_dir, 'trades.npy'), np.array(trades))

if __name__ == "__main__":
    args = parse_args()
    train_dqn_agent(args) 