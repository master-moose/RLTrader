#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated training script for the cryptocurrency trading model using PyTorch Lightning.
This script provides a more streamlined and automated training process with cleaner logs.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'model_training_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('automated_training')

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
        'logs',
        'models/checkpoints',
        'logs/lightning_logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")
    
    logger.debug("Directory setup complete.")

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

def summarize_config(config):
    """Print a summary of the configuration."""
    logger.info("Configuration summary:")
    
    # Data settings
    data_config = config.get('data', {})
    logger.info(f"  Data paths: train={data_config.get('train_data_path')}, val={data_config.get('val_data_path')}")
    logger.info(f"  Timeframes: {data_config.get('timeframes', [])}")
    
    # Model settings
    model_config = config.get('model', {})
    logger.info(f"  Model type: {model_config.get('type', 'unknown')}")
    logger.info(f"  Hidden dimensions: {model_config.get('hidden_dims', 'N/A')}")
    logger.info(f"  Num layers: {model_config.get('num_layers', 'N/A')}")
    
    # Training settings
    training_config = config.get('training', {})
    logger.info(f"  Batch size: {training_config.get('batch_size', 'N/A')}")
    logger.info(f"  Learning rate: {training_config.get('learning_rate', 'N/A')}")
    logger.info(f"  Max epochs: {training_config.get('epochs', 'N/A')}")

def run_lightning_training(config_path, max_epochs=None, patience=None, verbose=False):
    """
    Run the PyTorch Lightning training process.
    
    Args:
        config_path (str): Path to the configuration file
        max_epochs (int, optional): Maximum number of epochs to train
        patience (int, optional): Patience for early stopping
        verbose (bool, optional): Whether to enable verbose logging
    """
    from crypto_trading_model.lstm_lightning import train_lightning_model
    
    # Load the config to extract parameters
    config = load_config(config_path)
    
    # Summarize the configuration
    summarize_config(config)
    
    # Set defaults if not provided
    if max_epochs is None:
        max_epochs = config.get('training', {}).get('epochs', 100)
    
    if patience is None:
        patience = config.get('training', {}).get('patience', 15)
    
    logger.info(f"Starting Lightning training with max_epochs={max_epochs}, patience={patience}")
    
    # Run the Lightning training
    model, trainer = train_lightning_model(
        config_path=config_path,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        verbose=verbose
    )
    
    if model is None:
        logger.error("Training failed or no model was returned")
        return
    
    logger.info(f"Training completed successfully")
    
    # Save final model if needed
    output_dir = config.get('data', {}).get('output_dir', 'output/time_series')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'final_model.pt')
    try:
        import torch
        torch.save(model.state_dict(), model_path)
        logger.info(f"Final model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {str(e)}")

def main():
    """Main entry point for the automated training script."""
    parser = argparse.ArgumentParser(description='Automated Cryptocurrency Trading Model Training')
    parser.add_argument('--config', type=str, default='crypto_trading_model/config/time_series_config.json',
                      help='Path to config file')
    parser.add_argument('--max-epochs', type=int, default=None,
                      help='Maximum number of epochs to train')
    parser.add_argument('--patience', type=int, default=None,
                      help='Early stopping patience')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    # Log system info
    try:
        import torch
        import pytorch_lightning as pl
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch Lightning version: {pl.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("Could not import PyTorch or PyTorch Lightning")
    
    # Run Lightning training
    run_lightning_training(
        config_path=args.config,
        max_epochs=args.max_epochs,
        patience=args.patience,
        verbose=args.verbose
    )
    
    logger.info("Automated training completed")

if __name__ == "__main__":
    main() 