#!/usr/bin/env python
"""
Training script for reinforcement learning-based crypto trading model.

This script trains a PPO (Proximal Policy Optimization) agent using
the synthetic data generated and engineered in previous steps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import gym
from typing import Dict, List, Tuple, Optional

# Import project modules
from crypto_trading_model.models.reinforcement.ppo_agent import PPOTradingAgent, create_ppo_agent
from crypto_trading_model.environment.trading_env import CryptoTradingEnv, create_multi_timeframe_env
from crypto_trading_model.environment.reward_functions import create_reward_function
from crypto_trading_model.utils.performance_metrics import PerformanceMetrics
from crypto_trading_model.utils.visualization import MarketVisualizer, ModelVisualizer

# Import project configuration
from crypto_trading_model.config import PATHS, RL_SETTINGS, TRADING_ENV_SETTINGS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_rl_model')

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

def create_training_environment(
    data_dict: Dict[str, pd.DataFrame],
    primary_timeframe: str = '15m',
    window_size: int = 50,
    reward_function: str = 'sharpe',
    **kwargs
) -> CryptoTradingEnv:
    """
    Create a trading environment for training.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
    primary_timeframe : str
        Primary timeframe to use for trading
    window_size : int
        Number of periods to include in the observation window
    reward_function : str
        Reward function to use
    **kwargs : dict
        Additional arguments for the environment
        
    Returns:
    --------
    CryptoTradingEnv
        Trading environment for training
    """
    logger.info(f"Creating training environment with primary timeframe: {primary_timeframe}")
    
    # Set default parameters
    params = {
        'initial_balance': TRADING_ENV_SETTINGS.get('initial_balance', 10000.0),
        'transaction_fee': TRADING_ENV_SETTINGS.get('transaction_fee', 0.001),
        'slippage': TRADING_ENV_SETTINGS.get('slippage', 0.0005),
        'max_position_size': TRADING_ENV_SETTINGS.get('max_position_size', 0.2),
        'stop_loss': TRADING_ENV_SETTINGS.get('stop_loss', 0.05),
        'take_profit': TRADING_ENV_SETTINGS.get('take_profit', 0.15)
    }
    
    # Update with custom parameters
    params.update(kwargs)
    
    # Create environment
    if len(data_dict) > 1:
        # Multi-timeframe environment
        env = create_multi_timeframe_env(
            data_dict=data_dict,
            primary_timeframe=primary_timeframe,
            reward_function=reward_function,
            window_size=window_size,
            **params
        )
        logger.info(f"Created multi-timeframe environment with {len(data_dict)} timeframes")
    else:
        # Single timeframe environment
        env = CryptoTradingEnv(
            data=data_dict[primary_timeframe],
            reward_function=reward_function,
            window_size=window_size,
            **params
        )
        logger.info(f"Created single timeframe environment with timeframe: {primary_timeframe}")
    
    return env

def split_data_train_test(
    data_dict: Dict[str, pd.DataFrame],
    train_ratio: float = 0.8
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary of DataFrames with data for each timeframe
    train_ratio : float
        Ratio of data to use for training
        
    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Training and testing dictionaries
    """
    logger.info(f"Splitting data with train ratio: {train_ratio}")
    
    train_dict = {}
    test_dict = {}
    
    for tf, df in data_dict.items():
        # Calculate split point
        split_idx = int(len(df) * train_ratio)
        
        # Split data
        train_dict[tf] = df.iloc[:split_idx].copy()
        test_dict[tf] = df.iloc[split_idx:].copy()
        
        logger.info(f"  - {tf}: {len(train_dict[tf])} training periods, {len(test_dict[tf])} testing periods")
    
    return train_dict, test_dict

def train_ppo_agent(
    train_env: CryptoTradingEnv,
    eval_env: Optional[CryptoTradingEnv] = None,
    total_timesteps: int = 1000000,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    **kwargs
) -> PPOTradingAgent:
    """
    Train a PPO agent.
    
    Parameters:
    -----------
    train_env : CryptoTradingEnv
        Training environment
    eval_env : CryptoTradingEnv, optional
        Evaluation environment
    total_timesteps : int
        Total number of timesteps to train
    eval_freq : int
        Frequency of evaluation in timesteps
    save_freq : int
        Frequency of saving checkpoints
    **kwargs : dict
        Additional arguments for the PPO agent
        
    Returns:
    --------
    PPOTradingAgent
        Trained PPO agent
    """
    logger.info(f"Training PPO agent for {total_timesteps} timesteps")
    
    # Create agent
    agent = create_ppo_agent(
        env=train_env,
        eval_env=eval_env,
        **kwargs
    )
    
    # Log training parameters
    logger.info(f"Training parameters:")
    logger.info(f"  - Learning rate: {agent.learning_rate}")
    logger.info(f"  - Batch size: {agent.batch_size}")
    logger.info(f"  - n_steps: {agent.n_steps}")
    logger.info(f"  - n_epochs: {agent.n_epochs}")
    logger.info(f"  - gamma: {agent.gamma}")
    logger.info(f"  - Normalize environment: {agent.normalize_env}")
    
    # Train agent
    train_results = agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        save_freq=save_freq
    )
    
    logger.info(f"Training completed")
    
    return agent

def evaluate_agent(
    agent: PPOTradingAgent,
    eval_env: CryptoTradingEnv,
    n_episodes: int = 5
) -> Dict:
    """
    Evaluate a trained agent.
    
    Parameters:
    -----------
    agent : PPOTradingAgent
        Trained agent
    eval_env : CryptoTradingEnv
        Evaluation environment
    n_episodes : int
        Number of episodes to evaluate
        
    Returns:
    --------
    Dict
        Evaluation results
    """
    logger.info(f"Evaluating agent for {n_episodes} episodes")
    
    episode_rewards = []
    episode_lengths = []
    trades_history = []
    equity_curves = []
    
    for i in range(n_episodes):
        logger.info(f"Episode {i+1}/{n_episodes}")
        
        # Reset environment
        obs = eval_env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Run episode
        while not done:
            # Select action
            action, _ = agent.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = eval_env.step(action)
            
            # Update statistics
            total_reward += reward
            step += 1
            
            # Record trade if executed
            if info['executed_action'] not in ['hold']:
                trades_history.append({
                    'episode': i,
                    'step': step,
                    'action': info['executed_action'],
                    'price': info['price'],
                    'balance': info['balance'],
                    'position': info['position'],
                    'equity': info['equity'],
                    'reward': info['reward']
                })
        
        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        
        # Extract equity curve
        equity_curve = pd.Series([t[1] for t in eval_env.returns], index=[t[0] for t in eval_env.returns])
        equity_curves.append(equity_curve)
        
        logger.info(f"  - Reward: {total_reward:.4f}, Length: {step}, Final Equity: {eval_env.equity:.2f}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades_history)
    
    # Create evaluation results dictionary
    results = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'trades': trades_df,
        'equity_curves': equity_curves
    }
    
    logger.info(f"Evaluation results:")
    logger.info(f"  - Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    logger.info(f"  - Mean episode length: {mean_length:.2f}")
    logger.info(f"  - Number of trades: {len(trades_df)}")
    
    return results

def analyze_performance(
    eval_results: Dict,
    initial_balance: float = 10000.0
) -> Dict:
    """
    Analyze the performance of the agent.
    
    Parameters:
    -----------
    eval_results : Dict
        Evaluation results
    initial_balance : float
        Initial balance
        
    Returns:
    --------
    Dict
        Performance analysis
    """
    logger.info("Analyzing agent performance")
    
    # Create performance metrics calculator
    metrics_calc = PerformanceMetrics()
    
    # Initialize results dictionary
    performance = {}
    
    # Calculate average equity curve
    if eval_results['equity_curves']:
        # Combine equity curves from all episodes
        all_equity = pd.concat(eval_results['equity_curves'], axis=1)
        avg_equity = all_equity.mean(axis=1)
        
        # Calculate return metrics
        returns_data, returns_summary = metrics_calc.calculate_returns(avg_equity, initial_balance)
        
        # Calculate risk metrics
        risk_metrics = metrics_calc.calculate_risk_metrics(returns_data['returns'])
        
        # Store in results
        performance['returns_data'] = returns_data
        performance['returns_summary'] = returns_summary
        performance['risk_metrics'] = risk_metrics
        
        # Log performance summary
        logger.info("Performance summary:")
        for i, row in returns_summary.iterrows():
            logger.info(f"  - {row['Metric']}: {row['Value']}")
        
        logger.info("Risk metrics:")
        for metric, value in risk_metrics.items():
            logger.info(f"  - {metric}: {value:.4f}")
    
    # Analyze trades if available
    if 'trades' in eval_results and not eval_results['trades'].empty:
        trades = eval_results['trades']
        
        # Calculate basic trade statistics
        total_trades = len(trades)
        unique_actions = trades['action'].value_counts().to_dict()
        
        # Store in results
        performance['trade_stats'] = {
            'total_trades': total_trades,
            'unique_actions': unique_actions
        }
        
        logger.info(f"Trade statistics:")
        logger.info(f"  - Total trades: {total_trades}")
        for action, count in unique_actions.items():
            logger.info(f"  - {action}: {count} ({count/total_trades:.2%})")
    
    return performance

def visualize_results(
    agent: PPOTradingAgent,
    eval_results: Dict,
    performance: Dict,
    save_dir: str
):
    """
    Visualize the results of training and evaluation.
    
    Parameters:
    -----------
    agent : PPOTradingAgent
        Trained agent
    eval_results : Dict
        Evaluation results
    performance : Dict
        Performance analysis
    save_dir : str
        Directory to save visualizations
    """
    logger.info(f"Visualizing results in {save_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize training performance
    if agent.train_results is not None:
        # Plot training rewards
        fig = agent.plot_training_results()
        fig.savefig(os.path.join(save_dir, 'training_rewards.png'))
        plt.close(fig)
        logger.info(f"  - Saved training rewards plot")
    
    # Visualize equity curve
    if 'returns_data' in performance:
        # Create visualizer
        visualizer = MarketVisualizer()
        
        # Plot equity curve
        fig = visualizer.plot_price_series(
            data=performance['returns_data'][['equity']],
            title='Equity Curve',
            volume=False
        )
        fig.savefig(os.path.join(save_dir, 'equity_curve.png'))
        plt.close(fig)
        logger.info(f"  - Saved equity curve plot")
        
        # Plot drawdowns
        drawdowns = performance['returns_data']['drawdowns'] * 100  # Convert to percentage
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdowns.index, 0, drawdowns.values, color='red', alpha=0.3)
        ax.set_title('Drawdowns')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        fig.savefig(os.path.join(save_dir, 'drawdowns.png'))
        plt.close(fig)
        logger.info(f"  - Saved drawdowns plot")
    
    # Visualize trade distribution
    if 'trades' in eval_results and not eval_results['trades'].empty:
        trades = eval_results['trades']
        
        # Plot actions distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        trades['action'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Action Distribution')
        ax.set_ylabel('Count')
        ax.grid(True, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'action_distribution.png'))
        plt.close(fig)
        logger.info(f"  - Saved action distribution plot")
        
        # Plot equity progression with trades
        if 'equity' in trades.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by episode and plot equity for each episode
            for episode in trades['episode'].unique():
                ep_trades = trades[trades['episode'] == episode]
                ax.plot(ep_trades['step'], ep_trades['equity'], label=f'Episode {episode+1}')
            
            ax.set_title('Equity Progression')
            ax.set_xlabel('Step')
            ax.set_ylabel('Equity')
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, 'equity_progression.png'))
            plt.close(fig)
            logger.info(f"  - Saved equity progression plot")

def main():
    """Main function to train and evaluate the reinforcement learning model."""
    logger.info("Starting reinforcement learning model training")
    
    # Set parameters
    timeframes = ["15m", "4h", "1d"]
    primary_timeframe = "15m"
    data_filepath = os.path.join(PATHS.get('data', 'crypto_trading_model/data'), 'synthetic_processed.h5')
    
    # Create paths
    model_dir = os.path.join(PATHS.get('rl_models', 'crypto_trading_model/models/reinforcement'), 'ppo')
    results_dir = os.path.join(PATHS.get('results', 'crypto_trading_model/results'), 'ppo')
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load data
    logger.info("Step 1: Loading data")
    data_dict = load_data(data_filepath, timeframes)
    
    # Step 2: Split data into training and testing sets
    logger.info("Step 2: Splitting data")
    train_dict, test_dict = split_data_train_test(data_dict, train_ratio=0.8)
    
    # Step 3: Create trading environments
    logger.info("Step 3: Creating trading environments")
    
    # Training environment
    train_env = create_training_environment(
        data_dict=train_dict,
        primary_timeframe=primary_timeframe,
        window_size=50,
        reward_function='sharpe'
    )
    
    # Evaluation environment
    eval_env = create_training_environment(
        data_dict=test_dict,
        primary_timeframe=primary_timeframe,
        window_size=50,
        reward_function='sharpe'
    )
    
    # Step 4: Train PPO agent
    logger.info("Step 4: Training PPO agent")
    agent = train_ppo_agent(
        train_env=train_env,
        eval_env=eval_env,
        total_timesteps=RL_SETTINGS.get('total_timesteps', 500000),
        eval_freq=RL_SETTINGS.get('eval_freq', 10000),
        save_freq=RL_SETTINGS.get('save_freq', 50000),
        learning_rate=RL_SETTINGS.get('learning_rate', 0.0003),
        batch_size=RL_SETTINGS.get('batch_size', 64),
        policy=RL_SETTINGS.get('policy', "MlpPolicy")
    )
    
    # Step 5: Evaluate the agent
    logger.info("Step 5: Evaluating agent")
    eval_results = evaluate_agent(
        agent=agent,
        eval_env=eval_env,
        n_episodes=5
    )
    
    # Step 6: Analyze performance
    logger.info("Step 6: Analyzing performance")
    performance = analyze_performance(
        eval_results=eval_results,
        initial_balance=TRADING_ENV_SETTINGS.get('initial_balance', 10000.0)
    )
    
    # Step 7: Visualize results
    logger.info("Step 7: Visualizing results")
    visualize_results(
        agent=agent,
        eval_results=eval_results,
        performance=performance,
        save_dir=results_dir
    )
    
    logger.info("Reinforcement learning model training complete!")
    logger.info(f"Model saved to: {model_dir}")
    logger.info(f"Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 