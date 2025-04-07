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
        'output/reinforcement/dqn',
        'output/reinforcement/ppo',
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

def run_progressive_learning(config_path):
    """Run the progressive learning pipeline."""
    from crypto_trading_model.progressive_learning import run_progressive_learning
    
    config = load_config(config_path)
    logger.info("Starting progressive learning pipeline...")
    
    # Extract configuration parameters
    data_path = config['data_settings']['data_path']
    timeframes = config['data_settings']['timeframes']
    output_dir = config['data_settings']['output_dir']
    start_stage = config['progressive_learning']['start_stage']
    end_stage = config['progressive_learning']['end_stage']
    
    # Run the progressive learning pipeline
    run_progressive_learning(
        data_path=data_path,
        timeframes=timeframes,
        output_dir=output_dir,
        config_path=config_path,  # Pass the full config for the pipeline
        start_stage=start_stage,
        end_stage=end_stage
    )
    
    logger.info("Progressive learning pipeline completed.")

def generate_synthetic_data(config_path):
    """Generate synthetic data for training."""
    from crypto_trading_model.synthetic_data.dataset_builder import build_synthetic_dataset, save_dataset, create_train_val_test_split
    
    config = load_config(config_path)
    logger.info("Starting synthetic data generation...")
    
    # Extract configuration parameters
    num_samples = config['synthetic_data']['num_samples']
    pattern_distribution = config['synthetic_data']['pattern_distribution']
    include_indicators = config['synthetic_data']['include_indicators']
    output_dir = config['synthetic_data']['output_dir']
    
    logger.info(f"Generating {num_samples} samples with pattern distribution: {pattern_distribution}")
    
    # Generate synthetic dataset
    dataset = build_synthetic_dataset(
        num_samples=num_samples,
        pattern_distribution=pattern_distribution,
        with_indicators=include_indicators
    )
    
    # Save the dataset
    save_dataset(dataset, output_dir=output_dir)
    
    # Create train/val/test splits
    train_ratio = config['synthetic_data']['train_ratio']
    val_ratio = config['synthetic_data']['val_ratio']
    test_ratio = config['synthetic_data']['test_ratio']
    shuffle = config['synthetic_data']['shuffle']
    
    create_train_val_test_split(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=shuffle,
        output_dir=output_dir
    )
    
    logger.info(f"Synthetic data generation complete. Data saved to {output_dir}")

def run_standalone_time_series(config_path):
    """Run standalone time series model training."""
    from crypto_trading_model.models.time_series.trainer import run_time_series_training
    
    config = load_config(config_path)
    logger.info("Starting standalone time series model training...")
    
    # Run the time series training
    run_time_series_training(config)
    
    logger.info("Standalone time series model training completed.")

def run_standalone_dqn(config_path):
    """Run standalone DQN agent training."""
    from crypto_trading_model.models.reinforcement.dqn_agent import run_dqn_training
    
    config = load_config(config_path)
    logger.info("Starting standalone DQN agent training...")
    
    # Run the DQN training
    run_dqn_training(config)
    
    logger.info("Standalone DQN agent training completed.")

def run_standalone_ppo(config_path):
    """Run standalone PPO agent training."""
    from crypto_trading_model.models.reinforcement.ppo_agent import run_ppo_training
    
    config = load_config(config_path)
    logger.info("Starting standalone PPO agent training...")
    
    # Run the PPO training
    run_ppo_training(config)
    
    logger.info("Standalone PPO agent training completed.")

def run_backtest(config_path):
    """Run backtest on trained models."""
    from crypto_trading_model.evaluation.backtest import run_backtest as run_backtest_eval
    
    config = load_config(config_path)
    logger.info("Starting backtest...")
    
    # Run the backtest
    run_backtest_eval(config)
    
    logger.info("Backtest completed.")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Crypto Trading Model')
    
    # Required arguments
    parser.add_argument('--action', required=True,
                       choices=['setup', 'synthetic', 'progressive', 'time_series', 'dqn', 'ppo', 'backtest'],
                       help='Action to perform')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default='crypto_trading_model/config/config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Perform the requested action
    if args.action == 'setup':
        setup_directories()
    elif args.action == 'synthetic':
        generate_synthetic_data(args.config)
    elif args.action == 'progressive':
        run_progressive_learning(args.config)
    elif args.action == 'time_series':
        run_standalone_time_series(args.config)
    elif args.action == 'dqn':
        run_standalone_dqn(args.config)
    elif args.action == 'ppo':
        run_standalone_ppo(args.config)
    elif args.action == 'backtest':
        run_backtest(args.config)
    else:
        logger.error(f"Invalid action: {args.action}")
        sys.exit(1)

if __name__ == '__main__':
    main() 