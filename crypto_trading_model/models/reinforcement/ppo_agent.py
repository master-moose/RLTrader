"""
PPO (Proximal Policy Optimization) agent for cryptocurrency trading.

This module implements a PPO agent for reinforcement learning-based
cryptocurrency trading using stable-baselines3.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any
import logging
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# Import from parent directory
import sys
sys.path.append('../..')
from config import RL_SETTINGS, PATHS
from environment.trading_env import CryptoTradingEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ppo_agent')

class CustomCallback(BaseCallback):
    """
    Custom callback for monitoring and saving the RL agent.
    """
    
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_path: str = None,
        verbose: int = 1
    ):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        eval_env : gym.Env
            Environment for evaluation
        eval_freq : int
            Frequency of evaluation in timesteps
        n_eval_episodes : int
            Number of episodes for evaluation
        save_path : str
            Path to save best model
        verbose : int
            Verbosity level
        """
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.evaluations_results = []
        self.evaluations_timesteps = []
    
    def _on_step(self) -> bool:
        """
        Called at every step of training.
        
        Returns:
        --------
        bool
            Whether to continue training
        """
        # Evaluate the agent periodically
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate agent
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            # Record results
            self.evaluations_results.append(mean_reward)
            self.evaluations_timesteps.append(self.n_calls)
            
            # Log results
            logger.info(f"Evaluation at timestep {self.n_calls}: Mean reward = {mean_reward:.4f} +/- {std_reward:.4f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                if self.save_path is not None:
                    logger.info(f"New best model with mean reward {mean_reward:.4f}")
                    self.model.save(os.path.join(self.save_path, "best_model"))
        
        return True

class PPOTradingAgent:
    """
    PPO agent for cryptocurrency trading.
    """
    
    def __init__(
        self,
        env: Union[gym.Env, DummyVecEnv],
        policy: str = "MlpPolicy",
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        normalize_env: bool = True,
        tensorboard_log: str = None,
        verbose: int = 1
    ):
        """
        Initialize the PPO agent.
        
        Parameters:
        -----------
        env : gym.Env or DummyVecEnv
            Trading environment
        policy : str
            Policy type
        learning_rate : float
            Learning rate
        n_steps : int
            Number of steps to run for each environment per update
        batch_size : int
            Minibatch size
        n_epochs : int
            Number of epochs when optimizing the surrogate loss
        gamma : float
            Discount factor
        gae_lambda : float
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range : float
            Clipping parameter for PPO
        ent_coef : float
            Entropy coefficient for exploration
        normalize_env : bool
            Whether to normalize observations and rewards
        tensorboard_log : str
            TensorBoard log directory
        verbose : int
            Verbosity level
        """
        # Set parameters
        self.env = env
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.normalize_env = normalize_env
        self.verbose = verbose
        
        # Tensorboard logging
        if tensorboard_log is None:
            self.tensorboard_log = os.path.join(PATHS['logs'], 'tensorboard')
        else:
            self.tensorboard_log = tensorboard_log
        
        # Create log directory
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Vectorize and normalize environment if needed
        if normalize_env:
            if not isinstance(env, VecNormalize):
                if not isinstance(env, DummyVecEnv):
                    self.env = DummyVecEnv([lambda: env])
                self.env = VecNormalize(self.env)
        elif not isinstance(env, DummyVecEnv):
            self.env = DummyVecEnv([lambda: env])
        
        # Create PPO agent
        self.model = PPO(
            policy=self.policy,
            env=self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose
        )
        
        # Training history
        self.train_results = None
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: gym.Env = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 10000
    ) -> Dict:
        """
        Train the PPO agent.
        
        Parameters:
        -----------
        total_timesteps : int
            Number of timesteps to train
        eval_env : gym.Env
            Environment for evaluation
        eval_freq : int
            Frequency of evaluation in timesteps
        n_eval_episodes : int
            Number of episodes for evaluation
        save_freq : int
            Frequency of saving checkpoints
            
        Returns:
        --------
        Dict
            Training results
        """
        # Create model directory if it doesn't exist
        model_dir = os.path.join(PATHS['rl_models'], 'ppo')
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up callbacks
        callbacks = []
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = CustomCallback(
                eval_env=eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                save_path=model_dir,
                verbose=self.verbose
            )
            callbacks.append(eval_callback)
        
        # Checkpoint callback
        if save_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=os.path.join(model_dir, 'checkpoints'),
                name_prefix='ppo_model',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name='ppo'
        )
        
        # Save the final model
        self.model.save(os.path.join(model_dir, 'final_model'))
        
        # Save the vectorized environment for proper normalization during inference
        if isinstance(self.env, VecNormalize):
            self.env.save(os.path.join(model_dir, 'vec_normalize.pkl'))
        
        # Store training results
        if eval_env is not None and len(callbacks) > 0:
            self.train_results = {
                'timesteps': eval_callback.evaluations_timesteps,
                'rewards': eval_callback.evaluations_results
            }
        
        return self.train_results
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict action for a given observation.
        
        Parameters:
        -----------
        observation : np.ndarray
            Observation
        deterministic : bool
            Whether to use deterministic actions
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Actions and states
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str = None):
        """
        Save the agent.
        
        Parameters:
        -----------
        path : str
            Path to save the agent
        """
        if path is None:
            path = os.path.join(PATHS['rl_models'], 'ppo', 'model')
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        
        # Save the vectorized environment for proper normalization during inference
        if isinstance(self.env, VecNormalize):
            env_path = os.path.join(os.path.dirname(path), 'vec_normalize.pkl')
            self.env.save(env_path)
            logger.info(f"Normalized environment saved to {env_path}")
    
    @classmethod
    def load(cls, path: str, env: gym.Env = None, normalize_env: bool = True):
        """
        Load a saved agent.
        
        Parameters:
        -----------
        path : str
            Path to saved agent
        env : gym.Env
            Trading environment
        normalize_env : bool
            Whether to normalize observations and rewards
            
        Returns:
        --------
        PPOTradingAgent
            Loaded agent
        """
        # Create instance without model
        instance = cls(env=env, normalize_env=normalize_env)
        
        # Load vectorized environment if it exists
        env_path = os.path.join(os.path.dirname(path), 'vec_normalize.pkl')
        if os.path.exists(env_path) and normalize_env:
            instance.env = VecNormalize.load(env_path, instance.env)
            instance.env.training = False  # Turn off updates for evaluation
            instance.env.norm_reward = False  # Do not normalize rewards for evaluation
        
        # Load model
        instance.model = PPO.load(path, env=instance.env)
        logger.info(f"Model loaded from {path}")
        
        return instance
    
    def plot_training_results(self, save_path: str = None):
        """
        Plot training results.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        if self.train_results is None:
            raise ValueError("Agent must be trained before plotting results")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.train_results['timesteps'], self.train_results['rewards'])
        
        ax.set_title('Training Performance')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Episode Reward')
        ax.grid(True)
        
        if save_path is not None:
            plt.savefig(save_path)
        
        return fig

# Function to create and configure PPO agent
def create_ppo_agent(
    env: gym.Env,
    eval_env: gym.Env = None,
    policy: str = "MlpPolicy",
    normalize_env: bool = True,
    **kwargs
) -> PPOTradingAgent:
    """
    Create a PPO agent with custom configuration.
    
    Parameters:
    -----------
    env : gym.Env
        Trading environment
    eval_env : gym.Env
        Environment for evaluation
    policy : str
        Policy type
    normalize_env : bool
        Whether to normalize observations and rewards
    **kwargs : dict
        Additional arguments for PPO agent
        
    Returns:
    --------
    PPOTradingAgent
        Configured PPO agent
    """
    # Set default parameters from config
    params = {
        'learning_rate': RL_SETTINGS.get('learning_rate', 0.0003),
        'n_steps': RL_SETTINGS.get('n_steps', 2048),
        'batch_size': RL_SETTINGS.get('batch_size', 64),
        'n_epochs': RL_SETTINGS.get('n_epochs', 10),
        'gamma': RL_SETTINGS.get('discount_factor', 0.99),
        'gae_lambda': RL_SETTINGS.get('gae_lambda', 0.95),
        'clip_range': RL_SETTINGS.get('clip_range', 0.2),
        'ent_coef': RL_SETTINGS.get('ent_coef', 0.01),
    }
    
    # Update with custom parameters
    params.update(kwargs)
    
    # Create agent
    agent = PPOTradingAgent(
        env=env,
        policy=policy,
        normalize_env=normalize_env,
        **params
    )
    
    return agent

# Example usage
if __name__ == "__main__":
    logger.info("PPO Trading Agent Example")
    
    # Create a simple trading environment for testing
    from environment.trading_env import CryptoTradingEnv
    
    # Generate synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 2, len(dates)),
        'high': np.random.normal(101, 2, len(dates)),
        'low': np.random.normal(99, 2, len(dates)),
        'close': np.random.normal(100, 2, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    })
    
    # Add some technical indicators
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['rsi_14'] = np.random.normal(50, 10, len(dates))  # Dummy RSI
    
    # Drop NaN values
    data = data.dropna()
    
    # Create environment
    env = CryptoTradingEnv(
        data=data,
        window_size=50,
        reward_function='sharpe'
    )
    
    # Create a separate environment for evaluation
    eval_env = CryptoTradingEnv(
        data=data,
        window_size=50,
        reward_function='sharpe'
    )
    
    # Create and train PPO agent
    agent = create_ppo_agent(env=env, eval_env=eval_env)
    
    # Train for a small number of steps for demonstration
    train_results = agent.train(
        total_timesteps=10000,  # Small number for quick example
        eval_freq=1000,
        n_eval_episodes=5
    )
    
    # Plot training results
    fig = agent.plot_training_results()
    plt.savefig(os.path.join(PATHS['results'], 'ppo_training.png'))
    plt.close(fig)
    
    # Save the model
    agent.save()
    
    logger.info("PPO agent trained and saved") 