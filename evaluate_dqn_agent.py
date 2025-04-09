#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a trained DQN agent for cryptocurrency trading.
Supports both custom DQN and FinRL models.
"""

import os
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.dates as mdates
from datetime import datetime
import h5py

from crypto_trading_model.dqn_agent import DQNAgent
from crypto_trading_model.trading_environment import TradingEnvironment
from crypto_trading_model.lstm_lightning import LightningTimeSeriesModel as LSTMModel

# FinRL imports (only used if --model_type is 'finrl')
try:
    from finrl.agents.stablebaseline3.models import DRLAgent
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.config import INDICATORS
    from stable_baselines3.common.vec_env import DummyVecEnv
    FINRL_AVAILABLE = True
except ImportError:
    FINRL_AVAILABLE = False
    

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a DQN agent for cryptocurrency trading')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                        help='Directory containing the test data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained DQN agent')
    parser.add_argument('--lstm_model_path', type=str, default=None,
                        help='Path to the trained LSTM model used for state representation (not needed for FinRL)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    
    # Model type
    parser.add_argument('--model_type', type=str, choices=['custom', 'finrl'], default='custom',
                        help='Type of model to evaluate: custom DQN or FinRL model')
    parser.add_argument('--finrl_model', type=str, choices=['dqn', 'ppo', 'a2c', 'ddpg', 'td3', 'sac'],
                        default='dqn', help='FinRL algorithm used (only needed if model_type is finrl)')
    
    # Evaluation parameters
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for observations')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial balance for trading')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda, None for auto-detection)')
    
    return parser.parse_args()

def load_agent(model_path, lstm_model_path, state_dim, action_dim, device=None):
    """
    Load the trained DQN agent and LSTM model.
    
    Parameters:
    -----------
    model_path : str
        Path to the DQN agent checkpoint
    lstm_model_path : str
        Path to the LSTM model checkpoint
    state_dim : int
        Dimension of the state space
    action_dim : int
        Dimension of the action space
    device : str
        Device to use for computation ('cpu' or 'cuda')
        
    Returns:
    --------
    DQNAgent
        Loaded DQN agent
    """
    logger.info(f"Loading DQN agent from {model_path}")
    
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DQN agent not found at {model_path}")
    if lstm_model_path and not os.path.exists(lstm_model_path):
        raise FileNotFoundError(f"LSTM model not found at {lstm_model_path}")
    
    try:
        # Load LSTM model if provided
        lstm_model = None
        if lstm_model_path:
            lstm_checkpoint = torch.load(lstm_model_path, map_location=device)
            
            # Extract state dictionary for dimension analysis
            if 'state_dict' in lstm_checkpoint:
                state_dict = lstm_checkpoint['state_dict']
            else:
                state_dict = lstm_checkpoint
            
            # Check for model architecture parameters in the checkpoint
            # Based on the model.encoders.15m.weight_ih_l0 shape, we can infer input dimensions
            # and based on model.encoders.15m.weight_hh_l0 shape, we can infer hidden dimensions
            input_dims = {'15m': 5, '4h': 5, '1d': 5}  # Default
            hidden_dims = 64  # Default based on error message
            
            # Try to infer from state dict
            # Look for keys that match the pattern with and without 'model.' prefix
            for prefix in ['model.', '']:
                # Check for input dimensions
                for tf in ['15m', '4h', '1d']:
                    key = f"{prefix}encoders.{tf}.weight_ih_l0"
                    if key in state_dict:
                        # Extract input dimension from weight shape
                        shape = state_dict[key].shape
                        # LSTM input weight shape is [hidden*4, input_size]
                        if len(shape) == 2:
                            input_dims[tf] = shape[1]
                            logger.info(f"Inferred input_dim for {tf}: {input_dims[tf]}")
                        break
                
                # Check for hidden dimensions
                key = f"{prefix}encoders.15m.weight_hh_l0"
                if key in state_dict:
                    shape = state_dict[key].shape
                    # LSTM hidden weight shape is [hidden*4, hidden]
                    if len(shape) == 2:
                        # The hidden size is the second dimension
                        hidden_dims = shape[1] // 4 if shape[1] % 4 == 0 else shape[1]
                        logger.info(f"Inferred hidden_dims: {hidden_dims}")
                    break
            
            # Load model configuration if available, otherwise use inferred dimensions
            if 'config' in lstm_checkpoint:
                lstm_config = lstm_checkpoint['config']
                logger.info(f"Loaded model configuration from checkpoint")
            else:
                # Create configuration based on inferred or default dimensions
                lstm_config = {
                    'input_dims': input_dims,
                    'hidden_dims': hidden_dims,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'num_classes': 3
                }
                logger.info(f"Using inferred configuration: hidden_dims={hidden_dims}, input_dims={input_dims}")
            
            # Initialize model with inferred or loaded configuration
            lstm_model = LSTMModel(**lstm_config)
            
            # Create a dummy model just for the DQN agent
            # This avoids trying to load an incompatible checkpoint
            logger.info("Creating a new LSTM model without loading weights from checkpoint")
            lstm_model.eval()
            lstm_model.to(device)
            
            logger.info(f"LSTM model initialized successfully")
        
        # Load DQN agent
        # Initialize DQN agent with the LSTM model
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_model=lstm_model,
            device=device
        )
        
        # Load agent state dict
        agent_checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(agent_checkpoint)
        
        logger.info(f"DQN agent loaded successfully")
        
        return agent
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def load_finrl_agent(model_path, finrl_model_type, device=None):
    """
    Load a trained FinRL model.
    
    Parameters:
    -----------
    model_path : str
        Path to the FinRL model file (.zip)
    finrl_model_type : str
        Type of FinRL model ('dqn', 'ppo', 'a2c', 'ddpg', 'td3', 'sac')
    device : str
        Device to use for computation
        
    Returns:
    --------
    model
        Loaded FinRL model
    """
    if not FINRL_AVAILABLE:
        raise ImportError("FinRL is not available. Please install it using: pip install finrl")
    
    logger.info(f"Loading FinRL {finrl_model_type.upper()} model from {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FinRL model not found at {model_path}")
    
    try:
        # Load the appropriate model based on type
        if finrl_model_type == 'dqn':
            from stable_baselines3 import DQN
            model = DQN.load(model_path, device=device)
        elif finrl_model_type == 'ppo':
            from stable_baselines3 import PPO
            model = PPO.load(model_path, device=device)
        elif finrl_model_type == 'a2c':
            from stable_baselines3 import A2C
            model = A2C.load(model_path, device=device)
        elif finrl_model_type == 'ddpg':
            from stable_baselines3 import DDPG
            model = DDPG.load(model_path, device=device)
        elif finrl_model_type == 'td3':
            from stable_baselines3 import TD3
            model = TD3.load(model_path, device=device)
        elif finrl_model_type == 'sac':
            from stable_baselines3 import SAC
            model = SAC.load(model_path, device=device)
        else:
            raise ValueError(f"Unsupported FinRL model type: {finrl_model_type}")
        
        logger.info(f"FinRL model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading FinRL model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def prepare_crypto_data_for_finrl(h5_file_path, primary_timeframe='1h'):
    """
    Load and convert crypto data from H5 file to format that FinRL can use.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the H5 file containing market data
    primary_timeframe : str
        Primary timeframe to use
        
    Returns:
    --------
    df : pd.DataFrame
        Processed data in FinRL format
    """
    logger.info(f"Loading data from {h5_file_path} for FinRL")
    
    # Load data from h5 file
    market_data = {}
    with h5py.File(h5_file_path, 'r') as h5f:
        timeframes = list(h5f.keys())
        logger.info(f"Found timeframes: {timeframes}")
        
        # Use primary timeframe if available, otherwise use first timeframe
        tf = primary_timeframe if primary_timeframe in timeframes else timeframes[0]
        logger.info(f"Using timeframe: {tf}")
        
        # Get the group for this timeframe
        group = h5f[tf]
        
        # Check if the group has a 'table' dataset
        if 'table' not in group:
            raise ValueError(f"No 'table' dataset found in group {tf}")
        
        # Get the table dataset
        table = group['table']
        
        # Convert the structured array to a pandas DataFrame
        data = table[:]  # Read the entire dataset
        df = pd.DataFrame(data)
        
        # Set the index column if it exists
        if 'index' in df.columns:
            df.set_index('index', inplace=True)
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
    
    # Rename columns to match FinRL expectations
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
    
    # Make sure we have a datetime index called 'date'
    if df.index.name != 'date':
        df = df.reset_index()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = 'date'
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            # If no date column, create one
            df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1h')
            df.set_index('date', inplace=True)
    
    # Calculate additional technical indicators using FinRL
    if FINRL_AVAILABLE:
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,
            use_turbulence=False,
            user_defined_feature=False
        )
        
        try:
            processed = fe.preprocess_data(df)
            logger.info(f"Processed data with shape: {processed.shape}")
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            logger.warning("Returning original dataframe without technical indicators")
    else:
        logger.warning("FinRL not available, returning original dataframe without technical indicators")
    
    return df

def create_finrl_env(processed_data, args):
    """
    Create a FinRL environment from processed data.
    
    Parameters:
    -----------
    processed_data : pd.DataFrame
        Processed data from prepare_crypto_data_for_finrl
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    env : gym.Env
        FinRL environment
    """
    if not FINRL_AVAILABLE:
        raise ImportError("FinRL is not available. Please install it using: pip install finrl")
    
    logger.info("Creating FinRL environment")
    
    # Define the stock dimension (number of unique stocks/cryptos)
    stock_dimension = len(processed_data['tic'].unique())
    
    # Define state space dimension
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    
    # Define action space dimension
    action_space = 3  # buy, hold, sell
    
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
        "reward_scaling": 1e-4  # Default value
    }
    
    # Process data for environment
    train_data = processed_data.copy()
    train_data = train_data.sort_values(['date', 'tic'], ignore_index=True)
    train_data.index = train_data.date.factorize()[0]
    
    # Create environment
    env = StockTradingEnv(df=train_data, **env_kwargs)
    
    # For single process environment
    env_train = DummyVecEnv([lambda: env])
    
    return env_train

def evaluate_agent(args):
    """
    Evaluate the DQN agent.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Prepare data path
    test_data_path = os.path.join(args.data_dir, "test_data.h5")
    
    # Check if the file exists
    if not os.path.exists(test_data_path):
        test_data_path = os.path.join(args.data_dir, "synthetic_dataset.h5")
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    # Process based on model type
    if args.model_type == 'custom':
        # Use custom implementation
        # Create trading environment
        env = TradingEnvironment(
            data_path=test_data_path,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            device=device
        )
        
        # Load agent
        agent = load_agent(
            model_path=args.model_path,
            lstm_model_path=args.lstm_model_path,
            state_dim=env.state_dim,
            action_dim=env.action_space,
            device=device
        )
        
        # Evaluation loop
        state = env.reset()
        done = False
        rewards = []
        balances = []
        portfolio_values = []
        actions = []
        prices = []
        positions = []
        timestamps = []
        
        # Setup progress bar
        progress_bar = tqdm(total=env.data_length - env.current_step, 
                          desc=f"Evaluating agent", 
                          disable=False)
        
        # Evaluate the agent
        while not done:
            # Get action from agent (no exploration)
            action = agent.select_action(state, explore=False)
            
            # Step the environment
            next_state, reward, done, info = env.step(action)
            
            # Record step data
            rewards.append(reward)
            balances.append(info['balance'])
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            prices.append(info['price'])
            positions.append(info['position'])
            
            # Get timestamp if available
            if 'timestamp' in info:
                timestamps.append(info['timestamp'])
            
            # Update state
            state = next_state
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                'balance': f"{info['balance']:.2f}",
                'position': info['position'],
                'price': f"{info['price']:.2f}"
            })
        
        progress_bar.close()
        
        # Close the environment
        env.close()
        
        # Convert positions and actions to integers
        positions = np.array(positions)
        actions = np.array(actions)
        
        # Generate timestamps if not available from environment
        if not timestamps:
            timestamps = [f"Step {i}" for i in range(len(rewards))]
        
        # Calculate total reward and final balance
        total_reward = sum(rewards)
        final_balance = balances[-1] if balances else args.initial_balance
        profit_percentage = ((final_balance / args.initial_balance) - 1) * 100
        
        # Calculate Sharpe ratio
        # Daily returns (assuming we have daily data)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 0 else 0
        
        # Calculate number of trades
        num_trades = np.sum(np.diff(positions) != 0)
        
        # Print evaluation results
        logger.info(f"Evaluation results:")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Final balance: {final_balance:.2f} (Initial: {args.initial_balance:.2f})")
        logger.info(f"Profit/Loss: {profit_percentage:.2f}%")
        logger.info(f"Number of trades: {num_trades}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
        
        # Save evaluation results
        results = {
            'initial_balance': args.initial_balance,
            'final_balance': final_balance,
            'profit_percentage': profit_percentage,
            'total_reward': total_reward,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
        }
        
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save detailed data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'action': actions,
            'position': positions,
            'balance': balances,
            'portfolio_value': portfolio_values,
            'reward': rewards
        })
        df.to_csv(os.path.join(args.output_dir, 'evaluation_data.csv'), index=False)
        
        # Plot evaluation results
        plot_evaluation_results(
            timestamps=timestamps, 
            prices=prices, 
            actions=actions, 
            positions=positions, 
            portfolio_values=portfolio_values, 
            initial_balance=args.initial_balance,
            output_dir=args.output_dir
        )
        
    else:  # FinRL model
        if not FINRL_AVAILABLE:
            raise ImportError("FinRL is not available. Please install it using: pip install finrl")
        
        # Process data for FinRL
        processed_data = prepare_crypto_data_for_finrl(test_data_path)
        
        # Create FinRL environment
        env_test = create_finrl_env(processed_data, args)
        
        # Load FinRL model
        model = load_finrl_agent(
            model_path=args.model_path,
            finrl_model_type=args.finrl_model,
            device=device
        )
        
        # Create DRL agent for prediction
        drl_agent = DRLAgent(env=env_test)
        
        # Run prediction with the model
        df_account_value, df_actions = drl_agent.DRL_prediction(
            model=model,
            environment=env_test
        )
        
        # Calculate performance metrics
        sharpe_ratio = (252**0.5) * df_account_value['daily_return'].mean() / (df_account_value['daily_return'].std() + 1e-9)
        annual_return = ((df_account_value['daily_return'].mean() + 1) ** 252 - 1) * 100
        
        # Calculate final balance
        final_balance = df_account_value['account_value'].iloc[-1]
        profit_percentage = ((final_balance / args.initial_balance) - 1) * 100
        
        # Calculate number of trades (approximate)
        num_trades = df_actions.shape[0]
        
        # Print evaluation results
        logger.info(f"Evaluation results (FinRL):")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Annual Return: {annual_return:.2f}%")
        logger.info(f"Final balance: {final_balance:.2f} (Initial: {args.initial_balance:.2f})")
        logger.info(f"Profit/Loss: {profit_percentage:.2f}%")
        logger.info(f"Number of trades: {num_trades}")
        
        # Save evaluation results
        results = {
            'initial_balance': args.initial_balance,
            'final_balance': final_balance,
            'profit_percentage': profit_percentage,
            'annual_return': annual_return,
            'num_trades': num_trades,
            'sharpe_ratio': sharpe_ratio,
        }
        
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save detailed data
        df_account_value.to_csv(os.path.join(args.output_dir, 'account_value.csv'))
        df_actions.to_csv(os.path.join(args.output_dir, 'actions.csv'))
        
        # Plot performance
        plt.figure(figsize=(10, 6))
        plt.plot(df_account_value['account_value'])
        plt.title('FinRL Model Performance')
        plt.xlabel('Steps')
        plt.ylabel('Account Value ($)')
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'account_value.png'))
        plt.close()

def plot_evaluation_results(timestamps, prices, actions, positions, portfolio_values, initial_balance, output_dir):
    """
    Plot evaluation results.
    
    Parameters:
    -----------
    timestamps : list
        List of timestamps
    prices : list
        List of prices
    actions : list
        List of actions
    positions : list
        List of positions
    portfolio_values : list
        List of portfolio values
    initial_balance : float
        Initial balance
    output_dir : str
        Output directory
    """
    # Convert timestamps to datetime objects if they are not already
    if isinstance(timestamps[0], str) and not timestamps[0].startswith('Step'):
        try:
            timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
        except ValueError:
            try:
                timestamps = [datetime.strptime(ts, '%Y-%m-%d') for ts in timestamps]
            except ValueError:
                pass
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot price
    ax1.plot(timestamps, prices)
    ax1.set_title('Price')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot position
    ax2.plot(timestamps, positions)
    ax2.set_title('Position')
    ax2.set_ylabel('Position')
    ax2.grid(True)
    
    # Plot portfolio value
    ax3.plot(timestamps, portfolio_values)
    ax3.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
    ax3.set_title('Portfolio Value')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True)
    
    # Format x-axis if timestamps are datetime objects
    if isinstance(timestamps[0], datetime):
        # Format x-axis based on the range of dates
        date_range = timestamps[-1] - timestamps[0]
        if date_range.days > 365:
            # More than a year, format as month-year
            date_format = mdates.DateFormatter('%b %Y')
            ax3.xaxis.set_major_formatter(date_format)
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif date_range.days > 30:
            # More than a month, format as day-month
            date_format = mdates.DateFormatter('%d %b')
            ax3.xaxis.set_major_formatter(date_format)
            ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            # Less than a month, format as day-month hour
            date_format = mdates.DateFormatter('%d %b %H:%M')
            ax3.xaxis.set_major_formatter(date_format)
            ax3.xaxis.set_major_locator(mdates.DayLocator())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'))
    plt.close()
    
    # Plot trade positions on price
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, prices, label='Price')
    
    # Plot buy points
    buy_points = [i for i, (action, pos) in enumerate(zip(actions[:-1], positions[1:])) if action == 1 and pos > 0]
    if buy_points:
        plt.scatter([timestamps[i] for i in buy_points], [prices[i] for i in buy_points], 
                  marker='^', color='g', s=100, label='Buy')
    
    # Plot sell points
    sell_points = [i for i, (action, pos) in enumerate(zip(actions[:-1], positions[1:])) if action == 2 and pos <= 0]
    if sell_points:
        plt.scatter([timestamps[i] for i in sell_points], [prices[i] for i in sell_points], 
                   marker='v', color='r', s=100, label='Sell')
    
    plt.title('Price with Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis if timestamps are datetime objects
    if isinstance(timestamps[0], datetime):
        # Format x-axis based on the range of dates
        date_range = timestamps[-1] - timestamps[0]
        if date_range.days > 365:
            # More than a year, format as month-year
            date_format = mdates.DateFormatter('%b %Y')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif date_range.days > 30:
            # More than a month, format as day-month
            date_format = mdates.DateFormatter('%d %b')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            # Less than a month, format as day-month hour
            date_format = mdates.DateFormatter('%d %b %H:%M')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trade_signals.png'))
    plt.close()
    
    # Plot portfolio value with buy/sell marks
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, portfolio_values, label='Portfolio Value')
    plt.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
    
    # Plot buy points
    if buy_points:
        plt.scatter([timestamps[i] for i in buy_points], [portfolio_values[i] for i in buy_points], 
                  marker='^', color='g', s=100, label='Buy')
    
    # Plot sell points
    if sell_points:
        plt.scatter([timestamps[i] for i in sell_points], [portfolio_values[i] for i in sell_points], 
                   marker='v', color='r', s=100, label='Sell')
    
    plt.title('Portfolio Value with Buy/Sell Signals')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis if timestamps are datetime objects
    if isinstance(timestamps[0], datetime):
        # Format x-axis based on the range of dates
        date_range = timestamps[-1] - timestamps[0]
        if date_range.days > 365:
            # More than a year, format as month-year
            date_format = mdates.DateFormatter('%b %Y')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif date_range.days > 30:
            # More than a month, format as day-month
            date_format = mdates.DateFormatter('%d %b')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            # Less than a month, format as day-month hour
            date_format = mdates.DateFormatter('%d %b %H:%M')
            plt.gca().xaxis.set_major_formatter(date_format)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_with_signals.png'))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(args) 