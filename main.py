#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the LSTM-DQN cryptocurrency trading system.
"""

import os
import argparse
import logging
import json
import subprocess
from datetime import datetime

# Import the correct model class
from crypto_trading_model.lstm_lightning import LightningTimeSeriesModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LSTM-DQN Cryptocurrency Trading System')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['train_lstm', 'evaluate_lstm', 'train_dqn', 'evaluate_dqn', 'all'],
                      help='Operation mode (train_lstm, evaluate_lstm, train_dqn, evaluate_dqn, all)')
    
    # Common parameters
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                      help='Directory containing the data')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save models and results')
    
    # LSTM parameters
    parser.add_argument('--lstm_max_epochs', type=int, default=100,
                      help='Maximum epochs for LSTM training')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--lstm_model_path', type=str, default=None,
                      help='Path to the LSTM model checkpoint for evaluation or DQN training')
    
    # DQN parameters
    parser.add_argument('--dqn_episodes', type=int, default=1000,
                      help='Number of episodes for DQN training')
    parser.add_argument('--dqn_model_path', type=str, default=None,
                      help='Path to the DQN model for evaluation')
    
    # Environment parameters
    parser.add_argument('--window_size', type=int, default=20,
                      help='Window size for observations')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                      help='Initial balance for trading')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                      help='Transaction fee as a percentage')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cpu or cuda, None for auto-detection)')
    
    return parser.parse_args()

def train_lstm(args):
    """
    Train the LSTM model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting LSTM model training...")
    
    # Build command
    cmd = [
        "python", "train_improved_lstm.py",
        "--data_dir", args.data_dir,
        "--max_epochs", str(args.lstm_max_epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate)
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Execute command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    logger.info("LSTM model training completed.")

def evaluate_lstm(args):
    """
    Evaluate the LSTM model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting LSTM model evaluation...")
    
    # Check if model path is provided
    if not args.lstm_model_path:
        # Try to find the best model
        checkpoint_dir = os.path.join("models", "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoints:
                # Sort by validation loss
                checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]))
                best_model = os.path.join(checkpoint_dir, checkpoints[0])
                logger.info(f"Found best model: {best_model}")
            else:
                logger.error("No checkpoints found. Please provide --lstm_model_path")
                return
        else:
            logger.error("No checkpoint directory found. Please provide --lstm_model_path")
            return
    else:
        best_model = args.lstm_model_path
    
    # Build command
    cmd = [
        "python", "evaluate_lstm.py",
        "--model_dir", os.path.join(args.output_dir, "lstm_improved"),
        "--model_path", best_model,
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size)
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Execute command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    logger.info("LSTM model evaluation completed.")

def train_dqn(args):
    """
    Train the DQN agent.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting DQN agent training...")
    
    # Check if LSTM model path is provided
    if not args.lstm_model_path:
        # Try to find the best model
        checkpoint_dir = os.path.join("models", "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoints:
                # Sort by validation loss
                checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]))
                lstm_model = os.path.join(checkpoint_dir, checkpoints[0])
                logger.info(f"Found best LSTM model: {lstm_model}")
            else:
                logger.error("No checkpoints found. Please provide --lstm_model_path")
                return
        else:
            logger.error("No checkpoint directory found. Please provide --lstm_model_path")
            return
    else:
        lstm_model = args.lstm_model_path
    
    # Create output directory
    dqn_output_dir = os.path.join(args.output_dir, "dqn")
    os.makedirs(dqn_output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "train_dqn_agent.py",
        "--lstm_model_path", lstm_model,
        "--data_dir", args.data_dir,
        "--episodes", str(args.dqn_episodes),
        "--output_dir", dqn_output_dir,
        "--window_size", str(args.window_size),
        "--initial_balance", str(args.initial_balance),
        "--transaction_fee", str(args.transaction_fee)
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Execute command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    logger.info("DQN agent training completed.")

def evaluate_dqn(args):
    """
    Evaluate the DQN agent.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting DQN agent evaluation...")
    
    # Check if DQN model path is provided
    if not args.dqn_model_path:
        # Try to find the best model
        dqn_model_dir = os.path.join(args.output_dir, "dqn")
        if os.path.exists(dqn_model_dir):
            if os.path.exists(os.path.join(dqn_model_dir, "dqn_agent_best.pt")):
                dqn_model = os.path.join(dqn_model_dir, "dqn_agent_best.pt")
                logger.info(f"Found best DQN model: {dqn_model}")
            elif os.path.exists(os.path.join(dqn_model_dir, "dqn_agent_final.pt")):
                dqn_model = os.path.join(dqn_model_dir, "dqn_agent_final.pt")
                logger.info(f"Found final DQN model: {dqn_model}")
            else:
                logger.error("No DQN models found. Please provide --dqn_model_path")
                return
        else:
            logger.error("No DQN model directory found. Please provide --dqn_model_path")
            return
    else:
        dqn_model = args.dqn_model_path
    
    # Check if LSTM model path is provided
    if not args.lstm_model_path:
        # Try to find the best model
        checkpoint_dir = os.path.join("models", "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoints:
                # Sort by validation loss
                checkpoints.sort(key=lambda x: float(x.split('val_loss=')[1].split('.ckpt')[0]))
                lstm_model = os.path.join(checkpoint_dir, checkpoints[0])
                logger.info(f"Found best LSTM model: {lstm_model}")
            else:
                logger.error("No checkpoints found. Please provide --lstm_model_path")
                return
        else:
            logger.error("No checkpoint directory found. Please provide --lstm_model_path")
            return
    else:
        lstm_model = args.lstm_model_path
    
    # Create output directory
    evaluation_dir = os.path.join("evaluation_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "evaluate_dqn_agent.py",
        "--model_path", dqn_model,
        "--lstm_model_path", lstm_model,
        "--data_dir", args.data_dir,
        "--output_dir", evaluation_dir,
        "--window_size", str(args.window_size),
        "--initial_balance", str(args.initial_balance),
        "--transaction_fee", str(args.transaction_fee)
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    # Execute command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    logger.info("DQN agent evaluation completed.")
    logger.info(f"Results saved to {evaluation_dir}")

def run_all(args):
    """
    Run the entire pipeline from LSTM training to DQN evaluation.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    logger.info("Starting full pipeline...")
    
    # 1. Train LSTM model
    train_lstm(args)
    
    # 2. Evaluate LSTM model
    evaluate_lstm(args)
    
    # 3. Train DQN agent
    train_dqn(args)
    
    # 4. Evaluate DQN agent
    evaluate_dqn(args)
    
    logger.info("Full pipeline completed successfully!")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "lstm_improved"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "dqn"), exist_ok=True)
    
    # Execute selected mode
    if args.mode == 'train_lstm':
        train_lstm(args)
    elif args.mode == 'evaluate_lstm':
        evaluate_lstm(args)
    elif args.mode == 'train_dqn':
        train_dqn(args)
    elif args.mode == 'evaluate_dqn':
        evaluate_dqn(args)
    elif args.mode == 'all':
        run_all(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main() 