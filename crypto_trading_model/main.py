#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('crypto_trading.log')
    ]
)
logger = logging.getLogger('crypto_trading_model')

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

def generate_synthetic_data(config_path):
    """Generate synthetic data for training."""
    from synthetic_data.dataset_builder import build_synthetic_dataset, save_dataset, create_train_val_test_split
    
    config = load_config(config_path)
    logger.info("Starting synthetic data generation...")
    
    # Extract configuration parameters
    num_samples = config['data_settings']['num_samples']
    pattern_distribution = config['data_settings']['pattern_distribution']
    include_indicators = config['data_settings']['include_indicators']
    output_dir = config['data_settings']['output_dir']
    timeframes = config['timeframe_settings']['timeframes']
    multi_timeframe = config['timeframe_settings']['multi_timeframe']
    
    # Generate synthetic dataset
    dataset = build_synthetic_dataset(
        num_samples=num_samples,
        pattern_distribution=pattern_distribution,
        include_indicators=include_indicators
    )
    
    # Save the dataset
    save_dataset(dataset, output_dir=output_dir)
    
    # Create train/val/test splits
    train_ratio = config['training_settings']['train_ratio']
    val_ratio = config['training_settings']['val_ratio']
    test_ratio = config['training_settings']['test_ratio']
    shuffle = config['training_settings']['shuffle']
    
    create_train_val_test_split(
        dataset_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=shuffle
    )
    
    logger.info(f"Synthetic data generation complete. Data saved to {output_dir}")

def train_time_series_model(config_path):
    """Train the time series prediction model."""
    from models.time_series.model import MultiTimeframeModel, TimeSeriesTransformer, TimeSeriesForecaster
    from models.time_series.trainer import train_time_series_model, TimeSeriesDataset, plot_training_history
    import torch
    from torch.utils.data import DataLoader
    import pandas as pd
    import numpy as np
    
    config = load_config(config_path)
    logger.info("Starting time series model training...")
    
    # Extract configuration parameters
    data_dir = config['data_settings']['data_dir']
    output_dir = config['data_settings']['output_dir']
    timeframes = config['data_settings']['timeframes']
    seq_length = config['data_settings']['sequence_length']
    forecast_steps = config['data_settings']['forecast_steps']
    
    # Model settings
    model_type = config['model_settings']['model_type']
    hidden_dims = config['model_settings']['hidden_dims']
    num_layers = config['model_settings']['num_layers']
    dropout = config['model_settings']['dropout']
    bidirectional = config['model_settings']['bidirectional']
    attention = config['model_settings']['attention']
    feature_dims = config['model_settings']['feature_dims']
    
    # Training settings
    epochs = config['training_settings']['epochs']
    batch_size = config['training_settings']['batch_size']
    learning_rate = config['training_settings']['learning_rate']
    weight_decay = config['training_settings']['weight_decay']
    patience = config['training_settings']['patience']
    device = config['training_settings']['device']
    
    # Load data
    train_data = {}
    val_data = {}
    
    for timeframe in timeframes:
        train_data[timeframe] = pd.read_csv(f"{data_dir}/train_{timeframe}.csv")
        val_data[timeframe] = pd.read_csv(f"{data_dir}/val_{timeframe}.csv")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_length, forecast_steps)
    val_dataset = TimeSeriesDataset(val_data, seq_length, forecast_steps)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Initialize model based on type
    input_dims = {tf: feature_dims[tf] for tf in timeframes}
    
    if model_type == 'multi_timeframe':
        model = MultiTimeframeModel(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            attention=attention
        )
    elif model_type == 'transformer':
        model = TimeSeriesTransformer(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=config['model_settings'].get('num_heads', 4),
            max_seq_len=seq_length
        )
    elif model_type == 'forecaster':
        model = TimeSeriesForecaster(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=config['data_settings'].get('forecast_horizon', 5),
            bidirectional=bidirectional,
            attention=attention
        )
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)
    
    # Train model
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = train_time_series_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        device=device
    )
    
    # Save model and training history
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(history, f)
    
    # Plot training history
    plot_training_history(history, f"{output_dir}/training_history.png")
    
    logger.info(f"Time series model training complete. Model saved to {output_dir}/model.pt")

def train_reinforcement_agent(config_path):
    """Train a reinforcement learning agent."""
    from models.reinforcement.trading_env import CryptoTradingEnv
    from models.reinforcement.agent import DQNAgent, PPOAgent
    from models.reinforcement.trainer import train_agent
    import pandas as pd
    import numpy as np
    
    config = load_config(config_path)
    logger.info("Starting reinforcement learning agent training...")
    
    # Extract configuration parameters
    data_dir = config['data_settings']['data_dir']
    output_dir = config['data_settings']['output_dir']
    model_type = config['agent_settings']['model_type']
    lookback_window = config['environment_settings']['lookback_window']
    initial_balance = config['environment_settings']['initial_balance']
    transaction_fee = config['environment_settings']['transaction_fee']
    reward_function = config['environment_settings']['reward_function']
    
    # Load data
    train_data = pd.read_csv(f"{data_dir}/train_processed.csv")
    val_data = pd.read_csv(f"{data_dir}/val_processed.csv")
    
    # Create environments
    train_env = CryptoTradingEnv(
        data=train_data,
        lookback_window=lookback_window,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        reward_function=reward_function
    )
    
    val_env = CryptoTradingEnv(
        data=val_data,
        lookback_window=lookback_window,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee,
        reward_function=reward_function
    )
    
    # Initialize agent based on type
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    
    if model_type == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['agent_settings']['hidden_dims'],
            learning_rate=config['training_settings']['learning_rate'],
            gamma=config['training_settings']['gamma'],
            epsilon_start=config['training_settings']['epsilon_start'],
            epsilon_end=config['training_settings']['epsilon_end'],
            epsilon_decay=config['training_settings']['epsilon_decay'],
            buffer_size=config['training_settings']['buffer_size'],
            batch_size=config['training_settings']['batch_size']
        )
    elif model_type == 'ppo':
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['agent_settings']['hidden_dims'],
            actor_lr=config['training_settings']['actor_lr'],
            critic_lr=config['training_settings']['critic_lr'],
            gamma=config['training_settings']['gamma'],
            gae_lambda=config['training_settings']['gae_lambda'],
            clip_ratio=config['training_settings']['clip_ratio'],
            value_coef=config['training_settings']['value_coef'],
            entropy_coef=config['training_settings']['entropy_coef']
        )
    else:
        logger.error(f"Unknown agent type: {model_type}")
        sys.exit(1)
    
    # Train agent
    episodes = config['training_settings']['episodes']
    eval_frequency = config['training_settings']['eval_frequency']
    save_frequency = config['training_settings']['save_frequency']
    
    train_agent(
        agent=agent,
        train_env=train_env,
        val_env=val_env,
        episodes=episodes,
        eval_frequency=eval_frequency,
        save_frequency=save_frequency,
        output_dir=output_dir
    )
    
    logger.info(f"Reinforcement learning agent training complete. Agent saved to {output_dir}")

def create_ensemble_model(config_path):
    """Create an ensemble model from multiple trained models."""
    import torch
    import numpy as np
    import os
    from models.time_series.model import MultiTimeframeModel
    
    config = load_config(config_path)
    logger.info("Creating ensemble model...")
    
    # Extract configuration parameters
    output_dir = config['output_dir']
    ensemble_method = config['ensemble_method']
    models = config['models']
    weights = config['weights']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ensemble configuration
    with open(f"{output_dir}/ensemble_config.json", "w") as f:
        json.dump(config, f)
    
    logger.info(f"Ensemble model configuration saved to {output_dir}/ensemble_config.json")

def run_backtest(config_path):
    """Run backtesting on the trained models."""
    import pandas as pd
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    
    config = load_config(config_path)
    logger.info("Starting backtesting...")
    
    # Extract configuration parameters
    data_path = config['data_settings']['data_path']
    output_dir = config['data_settings']['output_dir']
    model_path = config['data_settings']['model_path']
    
    initial_balance = config['trading_settings']['initial_balance']
    position_size = config['trading_settings']['position_size']
    transaction_cost = config['trading_settings']['transaction_cost']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    test_data = pd.read_csv(f"{data_path}/test_processed.csv")
    
    # Implement backtesting logic here
    # This would depend on the specific model type and trading strategy
    
    # Generate performance metrics and visualizations
    # Save results to output directory
    
    logger.info(f"Backtesting complete. Results saved to {output_dir}")

def run_full_pipeline():
    """Run the complete pipeline from data generation to evaluation."""
    logger.info("Starting full pipeline execution...")
    
    # Generate synthetic data
    generate_synthetic_data("config/synthetic_config.json")
    
    # Train time series model
    train_time_series_model("config/time_series_config.json")
    
    # Train reinforcement learning agent
    train_reinforcement_agent("config/rl_config.json")
    
    # Create ensemble model
    create_ensemble_model("config/ensemble_config.json")
    
    # Run backtesting
    run_backtest("config/backtest_config.json")
    
    logger.info("Full pipeline execution complete.")

def main():
    """Main entry point for the cryptocurrency trading model."""
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Model")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["setup", "synthetic", "time_series", "reinforcement", 
                                "ensemble", "backtest", "full"],
                        help="Mode of operation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.mode == "setup":
        setup_directories()
    elif args.mode == "synthetic":
        if not args.config:
            logger.error("Config file path is required for synthetic data generation")
            sys.exit(1)
        generate_synthetic_data(args.config)
    elif args.mode == "time_series":
        if not args.config:
            logger.error("Config file path is required for time series model training")
            sys.exit(1)
        train_time_series_model(args.config)
    elif args.mode == "reinforcement":
        if not args.config:
            logger.error("Config file path is required for reinforcement learning")
            sys.exit(1)
        train_reinforcement_agent(args.config)
    elif args.mode == "ensemble":
        if not args.config:
            logger.error("Config file path is required for ensemble model creation")
            sys.exit(1)
        create_ensemble_model(args.config)
    elif args.mode == "backtest":
        if not args.config:
            logger.error("Config file path is required for backtesting")
            sys.exit(1)
        run_backtest(args.config)
    elif args.mode == "full":
        run_full_pipeline()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 