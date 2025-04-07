#!/usr/bin/env python
"""
Main entry point for the crypto trading model.

This script allows users to generate synthetic data,
engineer features, and train reinforcement learning models.
"""

import os
import argparse
import logging
import subprocess
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Trading Model')
    
    # Main command
    parser.add_argument('command', choices=['generate_data', 'train', 'all'],
                        help='Command to execute')
    
    # Data generation options
    parser.add_argument('--years', type=int, default=4,
                        help='Number of years of synthetic data to generate')
    parser.add_argument('--timeframes', type=str, default='15m,4h,1d',
                        help='Comma-separated list of timeframes to generate')
    
    # Training options
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Number of timesteps to train the RL agent')
    parser.add_argument('--reward', type=str, default='sharpe',
                        choices=['pnl', 'sharpe', 'sortino', 'calmar'],
                        help='Reward function to use')
    
    # Model options
    parser.add_argument('--policy', type=str, default='MlpPolicy',
                        help='Policy network architecture')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output directory for results')
    
    return parser.parse_args()

def generate_data(args):
    """Generate synthetic data."""
    logger.info("Starting synthetic data generation")
    
    # Prepare command
    cmd = [sys.executable, 'generate_synthetic_data.py']
    
    # Add optional arguments 
    # (these would be implemented in generate_synthetic_data.py to handle these args)
    if args.years:
        cmd.extend(['--years', str(args.years)])
    if args.timeframes:
        cmd.extend(['--timeframes', args.timeframes])
    if args.output:
        cmd.extend(['--output', args.output])
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error("Data generation failed")
        return False
    
    logger.info("Data generation completed successfully")
    return True

def train_model(args):
    """Train reinforcement learning model."""
    logger.info("Starting reinforcement learning model training")
    
    # Prepare command
    cmd = [sys.executable, 'train_rl_model.py']
    
    # Add optional arguments
    # (these would be implemented in train_rl_model.py to handle these args)
    if args.timesteps:
        cmd.extend(['--timesteps', str(args.timesteps)])
    if args.reward:
        cmd.extend(['--reward', args.reward])
    if args.policy:
        cmd.extend(['--policy', args.policy])
    if args.output:
        cmd.extend(['--output', args.output])
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logger.error("Model training failed")
        return False
    
    logger.info("Model training completed successfully")
    return True

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting run at {timestamp}")
    
    # Create output directory with timestamp if not specified
    if not args.output:
        args.output = f"runs/{timestamp}"
        os.makedirs(args.output, exist_ok=True)
        logger.info(f"Created output directory: {args.output}")
    
    # Execute the requested command
    if args.command == 'generate_data':
        generate_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'all':
        success = generate_data(args)
        if success:
            train_model(args)
        else:
            logger.error("Skipping training due to data generation failure")
    
    logger.info("Run completed")

if __name__ == "__main__":
    main() 