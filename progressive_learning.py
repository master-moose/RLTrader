#!/usr/bin/env python
"""
Progressive learning script for crypto trading model.

This script implements a staged learning approach:
1. Time series prediction using LSTM
2. Initial reinforcement learning with DQN
3. Advanced reinforcement learning with PPO

Each stage builds upon the knowledge gained in the previous stage.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gym
from typing import Dict, List, Tuple, Optional, Any
import json

# Import project modules
from crypto_trading_model.models.time_series.lstm_model import TimeSeriesLSTM, prepare_time_series_data
from crypto_trading_model.environment.trading_env import CryptoTradingEnv, create_multi_timeframe_env
from crypto_trading_model.environment.reward_functions import create_reward_function
from crypto_trading_model.utils.performance_metrics import PerformanceMetrics
from crypto_trading_model.utils.visualization import MarketVisualizer, ModelVisualizer
from crypto_trading_model.models.reinforcement.dqn_agent import DQNTradingAgent, create_dqn_agent

# Import project configuration
from crypto_trading_model.config import (
    PATHS, TIME_SERIES_SETTINGS, RL_SETTINGS, 
    TRADING_ENV_SETTINGS, PROGRESSIVE_LEARNING
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('progressive_learning')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Progressive Learning for Crypto Trading Model')
    
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the processed data HDF5 file')
    parser.add_argument('--timeframes', type=str, default='15m,4h,1d',
                        help='Comma-separated list of timeframes to use')
    parser.add_argument('--primary_timeframe', type=str, default='15m',
                        help='Primary timeframe for trading')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output directory for results')
    parser.add_argument('--stage', type=int, default=0,
                        help='Starting stage (0=all, 1=time_series, 2=dqn, 3=ppo)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation after training')
    
    return parser.parse_args()

def load_data(filepath: str, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load data from HDF5 file.
    
    Parameters:
    -----------
    filepath : str
        Path to the HDF5 file
    timeframes : List[str], optional
        List of timeframes to load
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
    """
    logger.info(f"Loading data from: {filepath}")
    
    data_dict = {}
    
    with pd.HDFStore(filepath, mode='r') as store:
        # Get available timeframes in the file
        available_timeframes = [k[1:] for k in store.keys() if k != '/combined']
        logger.info(f"Available timeframes: {available_timeframes}")
        
        # Load specified timeframes or all available timeframes
        for tf in timeframes or available_timeframes:
            if f'/{tf}' in store:
                data_dict[tf] = store[f'/{tf}']
                logger.info(f"  - Loaded {tf} data: {len(data_dict[tf])} rows")
            else:
                logger.warning(f"  - Timeframe {tf} not found in the file")
    
    return data_dict

def split_data_train_test_val(
    data_dict: Dict[str, pd.DataFrame],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split data into training, validation, and testing sets.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
    train_ratio : float
        Ratio of data to use for training
    val_ratio : float
        Ratio of data to use for validation
        
    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Training, validation, and testing dictionaries
    """
    logger.info(f"Splitting data with train ratio: {train_ratio}, validation ratio: {val_ratio}")
    
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for tf, df in data_dict.items():
        # Calculate split points
        train_idx = int(len(df) * train_ratio)
        val_idx = train_idx + int(len(df) * val_ratio)
        
        # Split data
        train_dict[tf] = df.iloc[:train_idx].copy()
        val_dict[tf] = df.iloc[train_idx:val_idx].copy()
        test_dict[tf] = df.iloc[val_idx:].copy()
        
        logger.info(f"  - {tf}: {len(train_dict[tf])} training, {len(val_dict[tf])} validation, {len(test_dict[tf])} testing periods")
    
    return train_dict, val_dict, test_dict

def stage1_time_series_prediction(
    train_dict: Dict[str, pd.DataFrame],
    val_dict: Dict[str, pd.DataFrame],
    test_dict: Dict[str, pd.DataFrame],
    timeframe: str,
    output_dir: str,
    config: Dict = None
) -> Tuple[TimeSeriesLSTM, Dict[str, Any]]:
    """
    Stage 1: Time series prediction using LSTM.
    
    Parameters:
    -----------
    train_dict : Dict[str, pd.DataFrame]
        Training data for each timeframe
    val_dict : Dict[str, pd.DataFrame]
        Validation data for each timeframe
    test_dict : Dict[str, pd.DataFrame]
        Testing data for each timeframe
    timeframe : str
        Timeframe to use for prediction
    output_dir : str
        Directory to save outputs
    config : Dict
        Configuration settings for this stage
        
    Returns:
    --------
    Tuple[TimeSeriesLSTM, Dict[str, Any]]
        Trained model and results
    """
    logger.info(f"Stage 1: Time series prediction using LSTM for timeframe {timeframe}")
    
    # Get data for the specified timeframe
    train_data = train_dict[timeframe]
    val_data = val_dict[timeframe]
    test_data = test_dict[timeframe]
    
    # Create directories for outputs
    model_dir = os.path.join(output_dir, 'models', 'time_series')
    results_dir = os.path.join(output_dir, 'results', 'time_series')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get configuration
    if config is None:
        config = {}
    
    sequence_length = config.get('sequence_length', TIME_SERIES_SETTINGS['sequence_length'])
    forecast_horizon = config.get('forecast_horizon', TIME_SERIES_SETTINGS['forecast_horizon'])
    target_column = config.get('target_column', TIME_SERIES_SETTINGS['target_column'])
    epochs = config.get('epochs', TIME_SERIES_SETTINGS['epochs'])
    batch_size = config.get('batch_size', TIME_SERIES_SETTINGS['batch_size'])
    
    # Prepare data for time series prediction
    logger.info("Preparing data for time series prediction")
    feature_columns = [col for col in train_data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'pattern']]
    
    # Get default feature columns if none provided
    if not feature_columns:
        feature_columns = None  # Use all available features
    
    X_train, y_train, X_val, y_val = prepare_time_series_data(
        train_data, 
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_steps=forecast_horizon,
        train_ratio=0.8  # This is within the training data
    )
    
    # Also prepare test data
    X_test, y_test, _, _ = prepare_time_series_data(
        test_data,
        sequence_length=sequence_length,
        target_column=target_column,
        feature_columns=feature_columns,
        target_steps=forecast_horizon,
        train_ratio=1.0  # Use all test data
    )
    
    logger.info(f"Prepared data shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"Prepared data shapes: X_val {X_val.shape}, y_val {y_val.shape}")
    logger.info(f"Prepared data shapes: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # Get input dimensions
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    # Create and train the LSTM model
    logger.info("Creating and training LSTM model")
    model = TimeSeriesLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        n_outputs=n_outputs,
        lstm_units=config.get('lstm_units', TIME_SERIES_SETTINGS['lstm_units']),
        dense_units=config.get('dense_units', TIME_SERIES_SETTINGS['dense_units']),
        dropout_rate=config.get('dropout_rate', TIME_SERIES_SETTINGS['dropout_rate']),
        learning_rate=config.get('learning_rate', TIME_SERIES_SETTINGS['learning_rate']),
        model_type=config.get('model_type', TIME_SERIES_SETTINGS['model_type'])
    )
    
    # Build the model
    model.build_model()
    
    # Train the model
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model
    logger.info("Evaluating model on test data")
    mse, mae = model.evaluate(X_test, y_test)
    logger.info(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(X_test)
    
    # Prepare results
    results = {
        'history': history.history,
        'test_mse': float(mse),
        'test_mae': float(mae),
        'predictions': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'actual': y_test.tolist() if isinstance(y_test, np.ndarray) else y_test,
        'config': {
            'sequence_length': sequence_length,
            'forecast_horizon': forecast_horizon,
            'target_column': target_column,
            'epochs': epochs,
            'batch_size': batch_size,
            'model_type': model.model_type
        }
    }
    
    # Save results
    results_file = os.path.join(results_dir, f'time_series_{timeframe}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_file = os.path.join(model_dir, f'time_series_{timeframe}_model')
    model.save(model_file)
    logger.info(f"Model saved to {model_file}")
    
    # Visualize results
    visualizer = ModelVisualizer()
    
    # Plot training history
    fig = visualizer.plot_training_history(
        history=history.history,
        title=f'Time Series Training History - {timeframe}'
    )
    fig.savefig(os.path.join(results_dir, f'time_series_{timeframe}_history.png'))
    plt.close(fig)
    
    # Plot predictions vs actual
    fig = visualizer.plot_predictions(
        actual=y_test.reshape(-1) if len(y_test.shape) > 1 else y_test,
        predicted=y_pred.reshape(-1) if len(y_pred.shape) > 1 else y_pred,
        dates=test_data.index[-len(y_test):],
        title=f'Price Predictions vs Actual - {timeframe}'
    )
    fig.savefig(os.path.join(results_dir, f'time_series_{timeframe}_predictions.png'))
    plt.close(fig)
    
    logger.info("Stage 1 completed")
    
    return model, results

def stage2_dqn_training(
    train_dict: Dict[str, pd.DataFrame],
    val_dict: Dict[str, pd.DataFrame],
    test_dict: Dict[str, pd.DataFrame],
    time_series_model: TimeSeriesLSTM,
    primary_timeframe: str,
    output_dir: str,
    config: Dict = None
) -> Tuple[DQNTradingAgent, Dict[str, Any]]:
    """
    Stage 2: DQN reinforcement learning using time series predictions.
    
    Parameters:
    -----------
    train_dict : Dict[str, pd.DataFrame]
        Training data for each timeframe
    val_dict : Dict[str, pd.DataFrame]
        Validation data for each timeframe
    test_dict : Dict[str, pd.DataFrame]
        Testing data for each timeframe
    time_series_model : TimeSeriesLSTM
        Trained time series model from Stage 1
    primary_timeframe : str
        Primary timeframe for trading
    output_dir : str
        Directory to save outputs
    config : Dict
        Configuration settings for this stage
        
    Returns:
    --------
    Tuple[DQNTradingAgent, Dict[str, Any]]
        Trained DQN agent and results
    """
    logger.info(f"Stage 2: DQN reinforcement learning for timeframe {primary_timeframe}")
    
    # Create directories for outputs
    model_dir = os.path.join(output_dir, 'models', 'reinforcement', 'dqn')
    results_dir = os.path.join(output_dir, 'results', 'reinforcement', 'dqn')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get configuration
    if config is None:
        config = {}
    
    # Get DQN specific config
    dqn_config = config.get('dqn', RL_SETTINGS['dqn'])
    
    # Get environment settings
    env_config = config.get('environment', TRADING_ENV_SETTINGS)
    
    # Create training environment
    logger.info("Creating trading environment for DQN training")
    
    # Select appropriate reward function
    reward_function = create_reward_function(
        env_config.get('reward_function', 'profit_and_loss'),
        **env_config.get('reward_params', {})
    )
    
    # Create environment
    train_env = CryptoTradingEnv(
        data=train_dict[primary_timeframe],
        lookback_window=env_config.get('lookback_window', 20),
        reward_function=reward_function,
        commission=env_config.get('commission', 0.001),
        initial_balance=env_config.get('initial_balance', 10000),
        max_position_size=env_config.get('max_position_size', 1.0),
        normalize_rewards=env_config.get('normalize_rewards', True),
        include_features=True
    )
    
    # Create validation environment
    val_env = CryptoTradingEnv(
        data=val_dict[primary_timeframe],
        lookback_window=env_config.get('lookback_window', 20),
        reward_function=reward_function,
        commission=env_config.get('commission', 0.001),
        initial_balance=env_config.get('initial_balance', 10000),
        max_position_size=env_config.get('max_position_size', 1.0),
        normalize_rewards=env_config.get('normalize_rewards', True),
        include_features=True
    )
    
    # Create DQN agent with time series model integration
    logger.info("Creating DQN agent with time series model integration")
    
    # Get DQN hyperparameters
    learning_rate = dqn_config.get('learning_rate', 0.0005)
    gamma = dqn_config.get('gamma', 0.99)
    epsilon_start = dqn_config.get('epsilon_start', 1.0)
    epsilon_end = dqn_config.get('epsilon_end', 0.01)
    epsilon_decay_steps = dqn_config.get('epsilon_decay_steps', 50000)
    batch_size = dqn_config.get('batch_size', 64)
    update_target_every = dqn_config.get('update_target_every', 100)
    hidden_units = dqn_config.get('hidden_units', [64, 64])
    buffer_capacity = dqn_config.get('buffer_capacity', 10000)
    double_dqn = dqn_config.get('double_dqn', True)
    
    # Setup TensorBoard logging
    tensorboard_log = os.path.join(output_dir, 'logs', 'tensorboard', 'dqn')
    
    # Create the DQN agent
    agent = DQNTradingAgent(
        env=train_env,
        time_series_model=time_series_model,  # Pass the trained time series model
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        batch_size=batch_size,
        update_target_every=update_target_every,
        buffer_capacity=buffer_capacity,
        hidden_units=hidden_units,
        double_dqn=double_dqn,
        tensorboard_log=tensorboard_log
    )
    
    # Train the DQN agent
    logger.info("Training DQN agent")
    total_timesteps = dqn_config.get('total_timesteps', 100000)
    log_interval = dqn_config.get('log_interval', 1000)
    eval_interval = dqn_config.get('eval_interval', 10000)
    
    training_history = agent.train(
        total_timesteps=total_timesteps,
        log_interval=log_interval,
        eval_interval=eval_interval,
        n_eval_episodes=5
    )
    
    # Save training history
    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {
            'loss': [float(x) for x in training_history['loss']],
            'reward': [float(x) for x in training_history['reward']],
            'epsilon': [float(x) for x in training_history['epsilon']]
        }
        json.dump(serializable_history, f)
    
    # Plot and save training results
    fig = agent.plot_training_results(save_path=os.path.join(results_dir, 'training_results.png'))
    plt.close(fig)
    
    # Evaluate the trained agent on validation data
    logger.info("Evaluating DQN agent on validation data")
    val_rewards = agent._evaluate(n_episodes=10)
    avg_val_reward = np.mean(val_rewards)
    logger.info(f"Validation: Average reward = {avg_val_reward:.4f}")
    
    # Save the agent
    logger.info("Saving DQN agent")
    agent.save(path=os.path.join(model_dir, 'dqn_agent'))
    
    # Create test trading environment for final evaluation
    test_env = CryptoTradingEnv(
        data=test_dict[primary_timeframe],
        lookback_window=env_config.get('lookback_window', 20),
        reward_function=reward_function,
        commission=env_config.get('commission', 0.001),
        initial_balance=env_config.get('initial_balance', 10000),
        max_position_size=env_config.get('max_position_size', 1.0),
        normalize_rewards=env_config.get('normalize_rewards', True),
        include_features=True
    )
    
    # Comprehensive evaluation on test data
    logger.info("Performing comprehensive evaluation on test data")
    test_rewards = []
    test_returns = []
    all_actions = []
    episode_lengths = []
    
    # Run multiple episodes on test data for evaluation
    n_test_episodes = 5
    for episode in range(n_test_episodes):
        state = test_env.reset()
        done = False
        episode_reward = 0
        actions = []
        step = 0
        
        while not done:
            # Get prediction from time series model if available
            prediction = None
            if time_series_model is not None:
                # TODO: Implement getting predictions from time series model
                prediction = 0.0  # Placeholder
            
            # Select action deterministically (no exploration)
            action = agent.select_action(state, prediction, deterministic=True)
            actions.append(action)
            
            # Take step in environment
            next_state, reward, done, info = test_env.step(action)
            
            # Update statistics
            episode_reward += reward
            state = next_state
            step += 1
        
        # Collect episode statistics
        test_rewards.append(episode_reward)
        test_returns.append(test_env.portfolio_value / test_env.initial_balance - 1.0)
        all_actions.append(actions)
        episode_lengths.append(step)
        
        logger.info(f"Test episode {episode+1}/{n_test_episodes}: "
                   f"Reward = {episode_reward:.4f}, "
                   f"Return = {test_returns[-1]:.4f}")
    
    # Calculate and log performance metrics
    avg_test_reward = np.mean(test_rewards)
    avg_test_return = np.mean(test_returns)
    
    logger.info(f"Test results - Average reward: {avg_test_reward:.4f}, Average return: {avg_test_return:.4f}")
    
    # Create performance metrics
    metrics = PerformanceMetrics()
    
    # Store results
    results = {
        'validation': {
            'rewards': val_rewards,
            'mean_reward': float(avg_val_reward)
        },
        'test': {
            'rewards': [float(r) for r in test_rewards],
            'returns': [float(r) for r in test_returns],
            'mean_reward': float(avg_test_reward),
            'mean_return': float(avg_test_return),
            'episode_lengths': episode_lengths
        },
        'config': {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay_steps': epsilon_decay_steps,
            'batch_size': batch_size,
            'total_timesteps': total_timesteps,
            'reward_function': env_config.get('reward_function', 'profit_and_loss')
        }
    }
    
    # Save results
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f)
    
    logger.info("Stage 2 (DQN training) completed successfully")
    
    return agent, results

def stage3_ppo_training(
    train_dict: Dict[str, pd.DataFrame],
    val_dict: Dict[str, pd.DataFrame],
    test_dict: Dict[str, pd.DataFrame],
    dqn_agent: DQNTradingAgent,
    primary_timeframe: str,
    output_dir: str,
    config: Dict = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Stage 3: Advanced reinforcement learning with PPO, building on DQN knowledge.
    
    Parameters:
    -----------
    train_dict : Dict[str, pd.DataFrame]
        Training data for each timeframe
    val_dict : Dict[str, pd.DataFrame]
        Validation data for each timeframe
    test_dict : Dict[str, pd.DataFrame]
        Testing data for each timeframe
    dqn_agent : DQNTradingAgent
        Trained DQN agent from stage 2
    primary_timeframe : str
        Primary timeframe for trading
    output_dir : str
        Directory to save outputs
    config : Dict
        Configuration settings for this stage
        
    Returns:
    --------
    Tuple[Any, Dict[str, Any]]
        Trained PPO agent and results
    """
    logger.info(f"Stage 3: Advanced PPO reinforcement learning for timeframe {primary_timeframe}")
    
    # Create directories for outputs
    model_dir = os.path.join(output_dir, 'models', 'reinforcement', 'ppo')
    results_dir = os.path.join(output_dir, 'results', 'reinforcement', 'ppo')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Get configuration
    if config is None:
        config = {}
    
    # Get PPO specific config
    ppo_config = config.get('ppo', RL_SETTINGS['ppo'])
    
    # Get environment settings
    env_config = config.get('environment', TRADING_ENV_SETTINGS)
    
    # Create training environment
    logger.info("Creating trading environment for PPO training")
    
    # Select appropriate reward function
    reward_function = create_reward_function(
        env_config.get('reward_function', 'profit_and_loss'),
        **env_config.get('reward_params', {})
    )
    
    # Create environment
    train_env = CryptoTradingEnv(
        data=train_dict[primary_timeframe],
        lookback_window=env_config.get('lookback_window', 20),
        reward_function=reward_function,
        commission=env_config.get('commission', 0.001),
        initial_balance=env_config.get('initial_balance', 10000),
        max_position_size=env_config.get('max_position_size', 1.0),
        normalize_rewards=env_config.get('normalize_rewards', True),
        include_features=True
    )
    
    # Create validation environment
    val_env = CryptoTradingEnv(
        data=val_dict[primary_timeframe],
        lookback_window=env_config.get('lookback_window', 20),
        reward_function=reward_function,
        commission=env_config.get('commission', 0.001),
        initial_balance=env_config.get('initial_balance', 10000),
        max_position_size=env_config.get('max_position_size', 1.0),
        normalize_rewards=env_config.get('normalize_rewards', True),
        include_features=True
    )
    
    # Create PPO agent leveraging knowledge from DQN
    logger.info("Creating PPO agent leveraging DQN knowledge")
    
    # For now, we'll add a placeholder for the PPO implementation
    # This will be implemented in the future with a proper PPO agent that can:
    # 1. Initialize its policy network from the DQN agent's Q-network
    # 2. Use the time series model for predictions
    # 3. Incorporate the DQN agent's learned value function
    
    logger.info("PPO implementation pending...")
    
    # Placeholder for the PPO agent
    ppo_agent = None
    
    # Placeholder for results
    results = {
        'config': config,
        'status': 'pending_implementation'
    }
    
    logger.info("Stage 3 pending implementation. When implemented, PPO will build upon DQN's knowledge.")
    
    return ppo_agent, results

def run_progressive_learning(args):
    """
    Run the progressive learning workflow.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting progressive learning workflow")
    
    # Determine data file path
    if args.data:
        data_filepath = args.data
    else:
        data_filepath = os.path.join(PATHS.get('data', 'crypto_trading_model/data'), 'synthetic_processed.h5')
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('runs', timestamp)
        os.makedirs(output_dir, exist_ok=True)
    
    # Get timeframes
    timeframes = args.timeframes.split(',')
    primary_timeframe = args.primary_timeframe
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    data_dict = load_data(data_filepath, timeframes)
    
    # Step 2: Split data
    logger.info("Step 2: Splitting data")
    train_dict, val_dict, test_dict = split_data_train_test_val(
        data_dict, 
        train_ratio=0.7, 
        val_ratio=0.15
    )
    
    # Extract stage configurations from settings
    stages_config = PROGRESSIVE_LEARNING.get('stages', [])
    stage_configs = {}
    for stage in stages_config:
        stage_type = stage['type']
        if stage_type == 'time_series':
            stage_configs['time_series'] = stage
        elif stage_type == 'reinforcement' and stage['model'] == 'dqn':
            stage_configs['dqn'] = stage
        elif stage_type == 'reinforcement' and stage['model'] == 'ppo':
            stage_configs['ppo'] = stage
    
    # Initialize models and results
    time_series_model = None
    dqn_agent = None
    ppo_agent = None
    
    # Run the stages based on starting stage
    if args.stage <= 1:
        # Stage 1: Time series prediction
        time_series_model, ts_results = stage1_time_series_prediction(
            train_dict=train_dict,
            val_dict=val_dict,
            test_dict=test_dict,
            timeframe=primary_timeframe,
            output_dir=output_dir,
            config=stage_configs.get('time_series')
        )
    
    if args.stage <= 2 and args.stage != 1:
        # Stage 2: DQN training
        dqn_agent, dqn_results = stage2_dqn_training(
            train_dict=train_dict,
            val_dict=val_dict,
            test_dict=test_dict,
            time_series_model=time_series_model,
            primary_timeframe=primary_timeframe,
            output_dir=output_dir,
            config=stage_configs.get('dqn')
        )
    
    if args.stage <= 3 and args.stage != 1 and args.stage != 2:
        # Stage 3: PPO training
        ppo_agent, ppo_results = stage3_ppo_training(
            train_dict=train_dict,
            val_dict=val_dict,
            test_dict=test_dict,
            dqn_agent=dqn_agent,
            primary_timeframe=primary_timeframe,
            output_dir=output_dir,
            config=stage_configs.get('ppo')
        )
    
    logger.info("Progressive learning workflow completed")
    logger.info(f"Results saved to: {output_dir}")

def main():
    """Main function."""
    args = parse_arguments()
    run_progressive_learning(args)

if __name__ == "__main__":
    main() 