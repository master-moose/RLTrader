import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gym
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

from .trading_env import CryptoTradingEnv, MultiTimeframeTradingEnv
from .policy import MLPPolicy, LSTMPolicy, MultiTimeframePolicy
from .agent import DQNAgent, PPOAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLTrainer:
    """
    Reinforcement Learning Trainer for crypto trading
    
    Handles data preparation, environment setup, agent training, evaluation,
    and result visualization.
    """
    def __init__(self,
                data_path: str,
                output_dir: str,
                timeframes: List[str] = ["15m", "4h", "1d"],
                window_size: int = 50,
                train_test_split: float = 0.8,
                validation_split: float = 0.1,
                seed: int = 42):
        """
        Initialize the RL trainer
        
        Parameters:
        - data_path: Path to data directory containing price data CSVs
        - output_dir: Directory to save models and results
        - timeframes: List of timeframes to use
        - window_size: Size of the observation window
        - train_test_split: Ratio of data for training
        - validation_split: Ratio of training data for validation
        - seed: Random seed
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.timeframes = timeframes
        self.window_size = window_size
        self.train_test_split = train_test_split
        self.validation_split = validation_split
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.data = self._load_data()
        
        # Split data
        self.train_data, self.val_data, self.test_data = self._split_data()
        
        # Initialize environment and agent
        self.train_env = None
        self.val_env = None
        self.test_env = None
        self.agent = None
    
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for each timeframe
        
        Returns:
        - data: Dictionary of DataFrames for each timeframe
        """
        data = {}
        
        for tf in self.timeframes:
            # Load data from CSV file
            file_path = os.path.join(self.data_path, f"price_data_{tf}.csv")
            if os.path.exists(file_path):
                logger.info(f"Loading data for timeframe {tf} from {file_path}")
                df = pd.read_csv(file_path)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                data[tf] = df
            else:
                raise FileNotFoundError(f"Data file not found: {file_path}")
        
        return data
    
    def _split_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Split data into train, validation, and test sets
        
        Returns:
        - train_data, val_data, test_data: Dictionaries of DataFrames for each timeframe
        """
        train_data = {}
        val_data = {}
        test_data = {}
        
        for tf, df in self.data.items():
            # Calculate split indices
            n = len(df)
            train_idx = int(n * self.train_test_split)
            val_idx = int(train_idx * (1 - self.validation_split))
            
            # Split data
            train_data[tf] = df.iloc[:val_idx].copy()
            val_data[tf] = df.iloc[val_idx:train_idx].copy()
            test_data[tf] = df.iloc[train_idx:].copy()
            
            logger.info(f"Timeframe {tf}: "
                      f"Train: {len(train_data[tf])}, "
                      f"Validation: {len(val_data[tf])}, "
                      f"Test: {len(test_data[tf])}")
        
        return train_data, val_data, test_data
    
    def setup_environments(self,
                          env_type: str = "multi",
                          initial_balance: float = 10000.0,
                          transaction_cost: float = 0.001,
                          position_size: float = 0.2,
                          reward_function: str = "pnl",
                          include_position_info: bool = True,
                          observation_type: str = "composed"):
        """
        Set up training, validation, and test environments
        
        Parameters:
        - env_type: Type of environment ('basic' or 'multi')
        - initial_balance: Initial account balance
        - transaction_cost: Transaction cost as a percentage
        - position_size: Size of each position as a percentage of balance
        - reward_function: Type of reward function to use
        - include_position_info: Whether to include position info in state
        - observation_type: Type of observation to return ('flat', 'composed', 'dict')
        """
        # Common parameters
        env_params = {
            'data': self.train_data,
            'timeframes': self.timeframes,
            'window_size': self.window_size,
            'initial_balance': initial_balance,
            'transaction_cost': transaction_cost,
            'position_size': position_size,
            'reward_function': reward_function,
            'include_position_info': include_position_info
        }
        
        # Create environments based on type
        if env_type == "basic":
            # Basic environments
            self.train_env = CryptoTradingEnv(**env_params)
            
            # Update data for validation and test
            val_params = env_params.copy()
            val_params['data'] = self.val_data
            self.val_env = CryptoTradingEnv(**val_params)
            
            test_params = env_params.copy()
            test_params['data'] = self.test_data
            self.test_env = CryptoTradingEnv(**test_params)
            
        elif env_type == "multi":
            # Multi-timeframe environments with additional params
            multi_params = env_params.copy()
            multi_params['observation_type'] = observation_type
            multi_params['include_technical_indicators'] = True
            
            self.train_env = MultiTimeframeTradingEnv(**multi_params)
            
            # Update data for validation and test
            val_params = multi_params.copy()
            val_params['data'] = self.val_data
            self.val_env = MultiTimeframeTradingEnv(**val_params)
            
            test_params = multi_params.copy()
            test_params['data'] = self.test_data
            self.test_env = MultiTimeframeTradingEnv(**test_params)
        
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        logger.info(f"Environments created with {env_type} type")
    
    def setup_agent(self,
                   agent_type: str = "dqn",
                   policy_type: str = "mlp",
                   policy_kwargs: Dict = None,
                   learning_kwargs: Dict = None):
        """
        Set up reinforcement learning agent
        
        Parameters:
        - agent_type: Type of agent ('dqn' or 'ppo')
        - policy_type: Type of policy network ('mlp', 'lstm', or 'multi')
        - policy_kwargs: Additional parameters for policy network
        - learning_kwargs: Additional parameters for agent
        """
        if self.train_env is None:
            raise ValueError("Environment not setup. Call setup_environments() first.")
        
        # Default kwargs
        if policy_kwargs is None:
            policy_kwargs = {}
        
        if learning_kwargs is None:
            learning_kwargs = {}
        
        # Create policy based on type
        if policy_type == "mlp":
            policy = MLPPolicy(
                observation_space=self.train_env.observation_space,
                action_space=self.train_env.action_space,
                **policy_kwargs
            )
        
        elif policy_type == "lstm":
            policy = LSTMPolicy(
                observation_space=self.train_env.observation_space,
                action_space=self.train_env.action_space,
                window_size=self.window_size,
                **policy_kwargs
            )
        
        elif policy_type == "multi":
            # For multi-timeframe policy, observation space must be dict
            if not hasattr(self.train_env, 'observation_space') or not isinstance(self.train_env.observation_space, dict):
                raise ValueError("Multi-timeframe policy requires dictionary observation space. "
                               "Use MultiTimeframeTradingEnv with observation_type='composed' or 'dict'.")
            
            policy = MultiTimeframePolicy(
                observation_space=self.train_env.observation_space,
                action_space=self.train_env.action_space,
                timeframes=self.timeframes,
                **policy_kwargs
            )
        
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Create agent based on type
        if agent_type == "dqn":
            self.agent = DQNAgent(
                env=self.train_env,
                policy_net=policy,
                **learning_kwargs
            )
        
        elif agent_type == "ppo":
            self.agent = PPOAgent(
                env=self.train_env,
                policy_net=policy,
                **learning_kwargs
            )
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        logger.info(f"Agent created with {agent_type} type and {policy_type} policy")
    
    def train(self, **kwargs):
        """
        Train the agent
        
        Parameters:
        - **kwargs: Parameters to pass to agent.train()
        
        Returns:
        - metrics: Training metrics
        """
        if self.agent is None:
            raise ValueError("Agent not setup. Call setup_agent() first.")
        
        logger.info("Starting training...")
        metrics = self.agent.train(**kwargs)
        logger.info("Training completed")
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def evaluate(self, env=None, num_episodes=10):
        """
        Evaluate the agent
        
        Parameters:
        - env: Environment to evaluate on (defaults to test_env)
        - num_episodes: Number of episodes to evaluate
        
        Returns:
        - results: Evaluation results
        """
        if self.agent is None:
            raise ValueError("Agent not setup. Call setup_agent() first.")
        
        if env is None:
            env = self.test_env
        
        # Store original environment
        original_env = self.agent.env
        
        # Set evaluation environment
        self.agent.env = env
        
        # Evaluate agent
        logger.info(f"Evaluating agent on {num_episodes} episodes...")
        returns = []
        trade_histories = []
        
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_return = 0
            
            while not done:
                action = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, info = env.step(action)
                
                state = next_state
                episode_return += reward
            
            returns.append(episode_return)
            
            # Get trade history if available
            if hasattr(env, 'trade_history'):
                trade_histories.append(env.trade_history)
            
            logger.info(f"Episode {episode+1}/{num_episodes}: Return = {episode_return:.2f}")
        
        # Restore original environment
        self.agent.env = original_env
        
        # Compute statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        logger.info(f"Evaluation results: Mean return = {mean_return:.2f}, Std dev = {std_return:.2f}")
        
        # Prepare results
        results = {
            'returns': returns,
            'mean_return': mean_return,
            'std_return': std_return,
            'trade_histories': trade_histories
        }
        
        return results
    
    def save_agent(self, name=None):
        """
        Save agent to disk
        
        Parameters:
        - name: Name for saved model (defaults to timestamp)
        
        Returns:
        - save_path: Path where agent was saved
        """
        if self.agent is None:
            raise ValueError("Agent not setup. Call setup_agent() first.")
        
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_path = os.path.join(self.output_dir, f"agent_{name}")
        self.agent.save(save_path)
        
        logger.info(f"Agent saved to {save_path}")
        
        return save_path
    
    def load_agent(self, path):
        """
        Load agent from disk
        
        Parameters:
        - path: Path to saved agent
        """
        if self.agent is None:
            raise ValueError("Agent not setup. Call setup_agent() first.")
        
        self.agent.load(path)
        
        logger.info(f"Agent loaded from {path}")
    
    def _save_metrics(self, metrics):
        """
        Save training metrics to disk
        
        Parameters:
        - metrics: Training metrics to save
        """
        # Save metrics as JSON
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        
        # Convert numpy arrays to lists
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [v.tolist() for v in value]
            else:
                serializable_metrics[key] = value
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f)
        
        # Plot metrics
        self._plot_metrics(metrics)
    
    def _plot_metrics(self, metrics):
        """
        Plot training metrics
        
        Parameters:
        - metrics: Training metrics to plot
        """
        # Create figure directory
        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot rewards
        if 'episode_rewards' in metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['episode_rewards'])
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig(os.path.join(figures_dir, "episode_rewards.png"))
            plt.close()
        
        # Plot losses
        if 'avg_losses' in metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['avg_losses'])
            plt.title('Average Losses')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(figures_dir, "losses.png"))
            plt.close()
        
        # Plot epsilon (for DQN)
        if 'epsilons' in metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['epsilons'])
            plt.title('Exploration Rate (Epsilon)')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.savefig(os.path.join(figures_dir, "epsilon.png"))
            plt.close()
        
        # Plot evaluation returns
        if 'eval_returns' in metrics:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics['eval_returns'])
            plt.title('Evaluation Returns')
            plt.xlabel('Evaluation')
            plt.ylabel('Return')
            plt.savefig(os.path.join(figures_dir, "eval_returns.png"))
            plt.close()

def run_experiment(config_path: str):
    """
    Run a reinforcement learning experiment from config file
    
    Parameters:
    - config_path: Path to configuration JSON file
    
    Returns:
    - trainer: Trained RL trainer
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Starting experiment with config from {config_path}")
    
    # Extract configurations
    data_config = config.get('data', {})
    env_config = config.get('environment', {})
    agent_config = config.get('agent', {})
    training_config = config.get('training', {})
    
    # Create trainer
    trainer = RLTrainer(
        data_path=data_config.get('data_path', './data'),
        output_dir=data_config.get('output_dir', './output'),
        timeframes=data_config.get('timeframes', ["15m", "4h", "1d"]),
        window_size=data_config.get('window_size', 50),
        train_test_split=data_config.get('train_test_split', 0.8),
        validation_split=data_config.get('validation_split', 0.1),
        seed=data_config.get('seed', 42)
    )
    
    # Setup environments
    trainer.setup_environments(
        env_type=env_config.get('type', 'multi'),
        initial_balance=env_config.get('initial_balance', 10000.0),
        transaction_cost=env_config.get('transaction_cost', 0.001),
        position_size=env_config.get('position_size', 0.2),
        reward_function=env_config.get('reward_function', 'pnl'),
        include_position_info=env_config.get('include_position_info', True),
        observation_type=env_config.get('observation_type', 'composed')
    )
    
    # Setup agent
    trainer.setup_agent(
        agent_type=agent_config.get('type', 'dqn'),
        policy_type=agent_config.get('policy_type', 'mlp'),
        policy_kwargs=agent_config.get('policy_kwargs', {}),
        learning_kwargs=agent_config.get('learning_kwargs', {})
    )
    
    # Train agent
    trainer.train(**training_config)
    
    # Evaluate agent
    eval_results = trainer.evaluate(num_episodes=10)
    
    # Save agent
    trainer.save_agent()
    
    logger.info("Experiment completed")
    
    return trainer, eval_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RL trading experiment')
    parser.add_argument('--config', type=str, default='./config/rl_config.json',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    trainer, results = run_experiment(args.config)
    
    print(f"Evaluation results: Mean return = {results['mean_return']:.2f}") 