#!/usr/bin/env python
"""
Test script to verify that our environment setup works correctly.
This script creates and initializes the trading environment without running the full training.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from train_dqn_agent import (
    BaseStockTradingEnv, 
    create_parallel_finrl_envs, 
    setup_logging,
    parse_args,
    ensure_technical_indicators
)

# Configure logging
logger = setup_logging()

def create_test_data():
    """Create a simple test DataFrame for environment testing"""
    # Create a DataFrame with 100 days of data
    dates = pd.date_range(start='2020-01-01', periods=100)
    data = {
        'date': dates,
        'tic': ['BTC'] * 100,
        'open': np.random.uniform(9000, 11000, 100),
        'high': np.random.uniform(9500, 11500, 100),
        'low': np.random.uniform(8500, 10500, 100),
        'close': np.random.uniform(9000, 11000, 100),
        'volume': np.random.uniform(1000000, 5000000, 100),
        'day': np.arange(100),
    }
    df = pd.DataFrame(data)
    
    # Add technical indicators
    indicators = [
        'macd', 'rsi_14', 'cci_30', 'dx_30', 
        'close_5_sma', 'close_10_sma', 'close_20_sma',
        'volatility_30', 'volume_change'
    ]
    df = ensure_technical_indicators(df, indicators)
    
    return df

def test_single_environment():
    """Test creating and stepping through a single environment"""
    logger.info("Testing single environment creation")
    
    # Create test data
    df = create_test_data()
    
    # Create environment parameters
    env_params = {
        'df': df,
        'stock_dim': 1,
        'hmax': 100,
        'initial_amount': 10000,
        'buy_cost_pct': [0.00075],  # Transaction cost
        'sell_cost_pct': [0.00075],  # Transaction cost
        'state_space': 30,  # Just a large enough value
        'action_space': 3,  # Sell, hold, buy
        'tech_indicator_list': ['macd', 'rsi_14', 'cci_30', 'dx_30'],
        'reward_scaling': 1e-4,
        'print_verbosity': 1
    }
    
    # Create the environment
    try:
        env = BaseStockTradingEnv(**env_params)
        logger.info(f"Successfully created environment with observation space: {env.observation_space}")
        
        # Test reset
        obs = env.reset()
        logger.info(f"Reset observation shape: {obs.shape if hasattr(obs, 'shape') else 'not a numpy array'}")
        
        # Test step
        for i in range(5):
            action = np.random.randint(0, 3)  # Random action: 0=sell, 1=hold, 2=buy
            obs, reward, done, info = env.step(action)
            logger.info(f"Step {i+1} - Action: {action}, Reward: {reward}, Done: {done}")
            logger.info(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else 'not a numpy array'}")
            
            if done:
                logger.info("Environment signaled done")
                break
                
        logger.info("Single environment test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing single environment: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_vectorized_environment(args):
    """Test creating and using a vectorized environment"""
    logger.info("Testing vectorized environment creation")
    
    # Create test data
    df = create_test_data()
    
    # Create the vectorized environment
    try:
        vec_env = create_parallel_finrl_envs(df, args, num_workers=2)
        logger.info(f"Successfully created vectorized environment with {vec_env.num_envs} environments")
        
        # Test reset
        obs = vec_env.reset()
        logger.info(f"Reset observation shape: {obs.shape}")
        
        # Test step
        for i in range(5):
            actions = [np.random.randint(0, 3) for _ in range(vec_env.num_envs)]
            obs, rewards, dones, infos = vec_env.step(actions)
            logger.info(f"Step {i+1} - Actions: {actions}")
            logger.info(f"Rewards: {rewards}")
            logger.info(f"Dones: {dones}")
            logger.info(f"Observation shape: {obs.shape}")
            
            if any(dones):
                logger.info("At least one environment signaled done")
                break
                
        logger.info("Vectorized environment test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing vectorized environment: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main entry point for the test script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test DQN trading environment')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--test_type', type=str, default='both', 
                        choices=['single', 'vectorized', 'both'],
                        help='Which environment type to test')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of parallel environments')
    parser.add_argument('--state_dim', type=int, default=16, help='State dimension')
    
    args = parser.parse_args()
    
    # Run tests
    if args.test_type in ['single', 'both']:
        test_single_environment()
        
    if args.test_type in ['vectorized', 'both']:
        test_vectorized_environment(args)
        
    logger.info("All tests completed")

if __name__ == "__main__":
    main() 