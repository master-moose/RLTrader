#!/usr/bin/env python
"""
Progressive learning module for transitioning from time series to RL models.

This module implements the full pipeline:
1. Train a multi-timeframe time series model
2. Use this model as a feature extractor for DQN
3. Transition to PPO with transfer learning
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import gymnasium as gym
from datetime import datetime

# Import time series models
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.models.time_series.trainer import train_time_series_model, TimeSeriesDataset
from torch.utils.data import DataLoader

# Import reinforcement learning components
from crypto_trading_model.models.reinforcement.trading_env import CryptoTradingEnv
from crypto_trading_model.models.reinforcement.dqn_agent import DQNTradingAgent
from crypto_trading_model.models.reinforcement.ppo_agent import PPOTradingAgent, create_ppo_agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('progressive_learning.log')
    ]
)
logger = logging.getLogger('progressive_learning')

def load_data(data_path: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load data for different timeframes.
    
    Parameters:
    -----------
    data_path : str
        Path to data directory
    timeframes : List[str]
        List of timeframes to load
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with data for each timeframe
    """
    data_dict = {}
    
    for tf in timeframes:
        tf_path = os.path.join(data_path, f"{tf}.csv")
        if os.path.exists(tf_path):
            logger.info(f"Loading data for timeframe {tf} from {tf_path}")
            data_dict[tf] = pd.read_csv(tf_path)
        else:
            logger.warning(f"Data file for timeframe {tf} not found at {tf_path}")
    
    return data_dict

def prepare_datasets(
    data_dict: Dict[str, pd.DataFrame],
    timeframes: List[str],
    sequence_length: int,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Prepare datasets for training, validation, and testing.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary with data for each timeframe
    timeframes : List[str]
        List of timeframes
    sequence_length : int
        Sequence length for time series
    batch_size : int
        Batch size for training
    train_ratio : float
        Ratio of data for training
    val_ratio : float
        Ratio of data for validation
        
    Returns:
    --------
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        Dictionaries with data splits and dataloaders
    """
    # Split data into train, validation, and test
    train_data = {}
    val_data = {}
    test_data = {}
    
    for tf in timeframes:
        if tf in data_dict:
            df = data_dict[tf]
            train_size = int(len(df) * train_ratio)
            val_size = int(len(df) * val_ratio)
            
            train_data[tf] = df.iloc[:train_size]
            val_data[tf] = df.iloc[train_size:train_size+val_size]
            test_data[tf] = df.iloc[train_size+val_size:]
            
            logger.info(f"Timeframe {tf}: {len(train_data[tf])} training, {len(val_data[tf])} validation, {len(test_data[tf])} test samples")
    
    # Create datasets and dataloaders for time series model
    feature_dims = {}
    for tf in timeframes:
        if tf in data_dict:
            # Exclude non-feature columns
            feature_cols = [col for col in data_dict[tf].columns if col not in ['timestamp', 'date', 'time']]
            feature_dims[tf] = len(feature_cols)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length=sequence_length)
    val_dataset = TimeSeriesDataset(val_data, sequence_length=sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        'train_data': train_data,
        'val_data': val_data, 
        'test_data': test_data,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'feature_dims': feature_dims
    }

def train_time_series(
    datasets: Dict[str, Any],
    timeframes: List[str],
    model_config: Dict[str, Any],
    output_dir: str
) -> MultiTimeframeModel:
    """
    Train a multi-timeframe time series model.
    
    Parameters:
    -----------
    datasets : Dict[str, Any]
        Dictionary with datasets and dataloaders
    timeframes : List[str]
        List of timeframes
    model_config : Dict[str, Any]
        Model configuration
    output_dir : str
        Output directory for saving model
        
    Returns:
    --------
    MultiTimeframeModel
        Trained model
    """
    logger.info("Stage 1: Training multi-timeframe time series model")
    
    # Extract configuration parameters
    hidden_dims = model_config.get('hidden_dims', 128)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.2)
    bidirectional = model_config.get('bidirectional', True)
    attention = model_config.get('attention', True)
    num_classes = model_config.get('num_classes', 3)  # Buy, sell, hold
    
    # Training parameters
    epochs = model_config.get('epochs', 50)
    learning_rate = model_config.get('learning_rate', 0.001)
    weight_decay = model_config.get('weight_decay', 1e-5)
    patience = model_config.get('patience', 10)
    
    # Create the model
    model = MultiTimeframeModel(
        input_dims=datasets['feature_dims'],
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        attention=attention,
        num_classes=num_classes
    )
    
    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    model.to(device)
    
    history = train_time_series_model(
        model=model,
        train_loader=datasets['train_loader'],
        val_loader=datasets['val_loader'],
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=device
    )
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "time_series_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, "time_series_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_series_training.png"))
    
    return model

def train_dqn_agent(
    datasets: Dict[str, Any],
    time_series_model: MultiTimeframeModel,
    timeframes: List[str],
    dqn_config: Dict[str, Any],
    output_dir: str
) -> DQNTradingAgent:
    """
    Train DQN agent with time series model features.
    
    Parameters:
    -----------
    datasets : Dict[str, Any]
        Dictionary with datasets
    time_series_model : MultiTimeframeModel
        Trained time series model
    timeframes : List[str]
        List of timeframes
    dqn_config : Dict[str, Any]
        DQN configuration
    output_dir : str
        Output directory for saving model
        
    Returns:
    --------
    DQNTradingAgent
        Trained DQN agent
    """
    logger.info("Stage 2: Training DQN agent with time series model features")
    
    # Extract configuration parameters
    window_size = dqn_config.get('window_size', 50)
    initial_balance = dqn_config.get('initial_balance', 10000.0)
    transaction_cost = dqn_config.get('transaction_cost', 0.001)
    position_size = dqn_config.get('position_size', 0.2)
    reward_function = dqn_config.get('reward_function', 'pnl')
    
    # DQN parameters
    learning_rate = dqn_config.get('learning_rate', 0.0005)
    gamma = dqn_config.get('gamma', 0.99)
    epsilon_start = dqn_config.get('epsilon_start', 1.0)
    epsilon_end = dqn_config.get('epsilon_end', 0.05)
    epsilon_decay_steps = dqn_config.get('epsilon_decay_steps', 10000)
    batch_size = dqn_config.get('batch_size', 64)
    buffer_capacity = dqn_config.get('buffer_capacity', 10000)
    update_target_every = dqn_config.get('update_target_every', 100)
    total_timesteps = dqn_config.get('total_timesteps', 100000)
    feature_extraction_mode = dqn_config.get('feature_extraction_mode', 'concat')
    
    # Create the trading environment
    env = CryptoTradingEnv(
        data=datasets['train_data'],
        timeframes=timeframes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        reward_function=reward_function,
        time_series_model=time_series_model
    )
    
    # Create validation environment
    val_env = CryptoTradingEnv(
        data=datasets['val_data'],
        timeframes=timeframes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        reward_function=reward_function,
        time_series_model=time_series_model
    )
    
    # Create the DQN agent
    dqn_agent = DQNTradingAgent(
        env=env,
        time_series_model=time_series_model,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        batch_size=batch_size,
        update_target_every=update_target_every,
        buffer_capacity=buffer_capacity,
        feature_extraction_mode=feature_extraction_mode,
        tensorboard_log=os.path.join(output_dir, 'logs')
    )
    
    # Train the agent
    training_history = dqn_agent.train(
        total_timesteps=total_timesteps,
        eval_interval=5000,
        n_eval_episodes=5
    )
    
    # Save the agent
    agent_path = os.path.join(output_dir, "dqn_agent")
    dqn_agent.save(agent_path)
    logger.info(f"DQN agent saved to {agent_path}")
    
    # Evaluate the agent
    logger.info("Evaluating DQN agent on test data")
    test_env = CryptoTradingEnv(
        data=datasets['test_data'],
        timeframes=timeframes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        reward_function=reward_function,
        time_series_model=time_series_model
    )
    
    # Plot training results
    dqn_agent.plot_training_results(save_path=os.path.join(output_dir, 'dqn_training.png'))
    
    return dqn_agent

def train_ppo_agent(
    datasets: Dict[str, Any],
    dqn_agent: DQNTradingAgent, 
    time_series_model: MultiTimeframeModel,
    timeframes: List[str],
    ppo_config: Dict[str, Any],
    output_dir: str
) -> PPOTradingAgent:
    """
    Train PPO agent with transfer learning from DQN.
    
    Parameters:
    -----------
    datasets : Dict[str, Any]
        Dictionary with datasets
    dqn_agent : DQNTradingAgent
        Trained DQN agent
    time_series_model : MultiTimeframeModel
        Trained time series model
    timeframes : List[str]
        List of timeframes
    ppo_config : Dict[str, Any]
        PPO configuration
    output_dir : str
        Output directory for saving model
        
    Returns:
    --------
    PPOTradingAgent
        Trained PPO agent
    """
    logger.info("Stage 3: Training PPO agent with transfer learning from DQN")
    
    # Extract configuration parameters
    window_size = ppo_config.get('window_size', 50)
    initial_balance = ppo_config.get('initial_balance', 10000.0)
    transaction_cost = ppo_config.get('transaction_cost', 0.001)
    position_size = ppo_config.get('position_size', 0.2)
    reward_function = ppo_config.get('reward_function', 'pnl')
    
    # PPO parameters
    learning_rate = ppo_config.get('learning_rate', 0.0003)
    n_steps = ppo_config.get('n_steps', 2048)
    batch_size = ppo_config.get('batch_size', 64)
    n_epochs = ppo_config.get('n_epochs', 10)
    gamma = ppo_config.get('gamma', 0.99)
    gae_lambda = ppo_config.get('gae_lambda', 0.95)
    clip_range = ppo_config.get('clip_range', 0.2)
    ent_coef = ppo_config.get('ent_coef', 0.01)
    normalize_env = ppo_config.get('normalize_env', True)
    total_timesteps = ppo_config.get('total_timesteps', 100000)
    
    # Create the trading environment
    env = CryptoTradingEnv(
        data=datasets['train_data'],
        timeframes=timeframes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        reward_function=reward_function,
        time_series_model=time_series_model
    )
    
    # Create validation environment
    val_env = CryptoTradingEnv(
        data=datasets['val_data'],
        timeframes=timeframes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        reward_function=reward_function,
        time_series_model=time_series_model
    )
    
    # Create the PPO agent with knowledge from DQN
    ppo_agent = create_ppo_agent(
        env=env,
        eval_env=val_env,
        policy="MlpPolicy",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        normalize_env=normalize_env,
        tensorboard_log=os.path.join(output_dir, 'logs')
    )
    
    # Train the agent
    ppo_history = ppo_agent.train(
        total_timesteps=total_timesteps,
        eval_freq=5000,
        n_eval_episodes=5,
        save_freq=10000
    )
    
    # Save the agent
    agent_path = os.path.join(output_dir, "ppo_agent")
    ppo_agent.save(agent_path)
    logger.info(f"PPO agent saved to {agent_path}")
    
    # Evaluate the agent
    logger.info("Evaluating PPO agent on test data")
    test_env = CryptoTradingEnv(
        data=datasets['test_data'],
        timeframes=timeframes,
        window_size=window_size,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        reward_function=reward_function,
        time_series_model=time_series_model
    )
    
    # Plot training results
    ppo_agent.plot_training_results(save_path=os.path.join(output_dir, 'ppo_training.png'))
    
    return ppo_agent

def run_progressive_learning(
    data_path: str,
    timeframes: List[str] = ["15m", "4h", "1d"],
    output_dir: str = "output",
    config_path: str = None,
    start_stage: int = 1,
    end_stage: int = 3
):
    """
    Run the full progressive learning pipeline.
    
    Parameters:
    -----------
    data_path : str
        Path to data directory
    timeframes : List[str]
        List of timeframes to use
    output_dir : str
        Output directory
    config_path : str
        Path to configuration file
    start_stage : int
        Stage to start from (1=time_series, 2=dqn, 3=ppo)
    end_stage : int
        Stage to end at
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'time_series': {
                'sequence_length': 50,
                'batch_size': 64,
                'hidden_dims': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True,
                'attention': True,
                'num_classes': 3,
                'epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'patience': 10
            },
            'dqn': {
                'window_size': 50,
                'initial_balance': 10000.0,
                'transaction_cost': 0.001,
                'position_size': 0.2,
                'reward_function': 'pnl',
                'learning_rate': 0.0005,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay_steps': 10000,
                'batch_size': 64,
                'buffer_capacity': 10000,
                'update_target_every': 100,
                'total_timesteps': 100000,
                'feature_extraction_mode': 'concat'
            },
            'ppo': {
                'window_size': 50,
                'initial_balance': 10000.0,
                'transaction_cost': 0.001,
                'position_size': 0.2,
                'reward_function': 'pnl',
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'normalize_env': True,
                'total_timesteps': 100000
            }
        }
    
    # Load data
    data_dict = load_data(data_path, timeframes)
    
    # Prepare datasets
    datasets = prepare_datasets(
        data_dict=data_dict,
        timeframes=timeframes,
        sequence_length=config['time_series']['sequence_length'],
        batch_size=config['time_series']['batch_size']
    )
    
    # Stage 1: Train time series model
    time_series_model = None
    if start_stage <= 1 and end_stage >= 1:
        time_series_model = train_time_series(
            datasets=datasets,
            timeframes=timeframes,
            model_config=config['time_series'],
            output_dir=os.path.join(output_dir, 'time_series')
        )
    elif start_stage > 1:
        # Load pre-trained time series model
        model_path = os.path.join(output_dir, 'time_series', 'time_series_model.pt')
        if os.path.exists(model_path):
            logger.info(f"Loading pre-trained time series model from {model_path}")
            time_series_model = MultiTimeframeModel(
                input_dims=datasets['feature_dims'],
                hidden_dims=config['time_series']['hidden_dims'],
                num_layers=config['time_series']['num_layers'],
                dropout=config['time_series']['dropout'],
                bidirectional=config['time_series']['bidirectional'],
                attention=config['time_series']['attention'],
                num_classes=config['time_series']['num_classes']
            )
            time_series_model.load_state_dict(torch.load(model_path))
        else:
            logger.error(f"Pre-trained time series model not found at {model_path}. Cannot continue.")
            return
    
    # Stage 2: Train DQN agent
    dqn_agent = None
    if start_stage <= 2 and end_stage >= 2:
        if time_series_model is not None:
            dqn_agent = train_dqn_agent(
                datasets=datasets,
                time_series_model=time_series_model,
                timeframes=timeframes,
                dqn_config=config['dqn'],
                output_dir=os.path.join(output_dir, 'dqn')
            )
        else:
            logger.error("Time series model is required for DQN training. Cannot continue.")
            return
    elif start_stage > 2:
        # Load pre-trained DQN agent
        agent_path = os.path.join(output_dir, 'dqn', 'dqn_agent')
        if os.path.exists(f"{agent_path}.index"):
            logger.info(f"Loading pre-trained DQN agent from {agent_path}")
            # Create environment
            env = CryptoTradingEnv(
                data=datasets['train_data'],
                timeframes=timeframes,
                window_size=config['dqn']['window_size'],
                initial_balance=config['dqn']['initial_balance'],
                transaction_cost=config['dqn']['transaction_cost'],
                position_size=config['dqn']['position_size'],
                reward_function=config['dqn']['reward_function'],
                time_series_model=time_series_model
            )
            dqn_agent = DQNTradingAgent.load(agent_path, env, time_series_model)
        else:
            logger.error(f"Pre-trained DQN agent not found at {agent_path}. Cannot continue.")
            return
    
    # Stage 3: Train PPO agent
    if start_stage <= 3 and end_stage >= 3:
        if time_series_model is not None and dqn_agent is not None:
            ppo_agent = train_ppo_agent(
                datasets=datasets,
                dqn_agent=dqn_agent,
                time_series_model=time_series_model,
                timeframes=timeframes,
                ppo_config=config['ppo'],
                output_dir=os.path.join(output_dir, 'ppo')
            )
        else:
            logger.error("Time series model and DQN agent are required for PPO training. Cannot continue.")
            return
    
    logger.info("Progressive learning pipeline completed successfully.")

def main():
    """Main function for running the progressive learning pipeline."""
    parser = argparse.ArgumentParser(description="Progressive learning for crypto trading")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--timeframes", type=str, default="15m,4h,1d", help="Comma-separated list of timeframes")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--config_path", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--start_stage", type=int, default=1, help="Stage to start from (1=time_series, 2=dqn, 3=ppo)")
    parser.add_argument("--end_stage", type=int, default=3, help="Stage to end at")
    
    args = parser.parse_args()
    
    # Convert timeframes string to list
    timeframes = args.timeframes.split(',')
    
    run_progressive_learning(
        data_path=args.data_path,
        timeframes=timeframes,
        output_dir=args.output_dir,
        config_path=args.config_path,
        start_stage=args.start_stage,
        end_stage=args.end_stage
    )

if __name__ == "__main__":
    main() 