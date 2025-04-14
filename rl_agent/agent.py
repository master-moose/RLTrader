#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reinforcement Learning Agent for Trading.

This module contains the main RL agent class that handles training,
evaluation, prediction, and saving/loading of trading agents.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

import torch
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

from .callbacks import get_callback_list
from .utils import ensure_dir_exists


# Setup logger
logger = logging.getLogger("rl_agent")


class TradingRLAgent:
    """
    Reinforcement Learning Agent for trading environments.
    
    This class provides a unified interface for training, evaluating,
    and using RL agents for trading strategies, supporting different
    algorithms like PPO, A2C, DQN, and SAC.
    """
    
    # Dictionary mapping algorithm names to their classes
    SUPPORTED_ALGORITHMS = {
        "ppo": PPO,
        "a2c": A2C,
        "dqn": DQN, 
        "sac": SAC
    }
    
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        algorithm: str = "ppo",
        policy: str = "MlpPolicy",
        device: str = "auto",
        tensorboard_log: Optional[str] = "./logs/tensorboard/",
        seed: int = 42,
        verbose: int = 1,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RL agent.
        
        Args:
            env: The training environment
            algorithm: Algorithm name (ppo, a2c, dqn, sac)
            policy: Policy network type
            device: Device to run the model on ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
            tensorboard_log: Directory for tensorboard logs
            seed: Random seed
            verbose: Verbosity level (0=no output, 1=info, 2=debug)
            model_kwargs: Additional keyword arguments for model initialization
        """
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Store parameters
        self.env = env
        self.algorithm_name = algorithm.lower()
        self.policy = policy
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.model_kwargs = model_kwargs or {}
        
        # Check if algorithm is supported
        if self.algorithm_name not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm {algorithm} not supported. Choose from {list(self.SUPPORTED_ALGORITHMS.keys())}")
        
        # Setup tensorboard logging if specified
        if tensorboard_log:
            ensure_dir_exists(tensorboard_log)
            self.tensorboard_log = tensorboard_log
            logger.info(f"Tensorboard logs will be saved to {tensorboard_log}")
        else:
            self.tensorboard_log = None
        
        # Create the model
        self._create_model()
        
        # Setup model tracking
        self.training_timesteps = 0
        self.eval_results = []
        self.train_metrics = []
    
    def _create_model(self):
        """Create the RL model with the specified algorithm."""
        algorithm_class = self.SUPPORTED_ALGORITHMS[self.algorithm_name]
        
        # Setup additional arguments based on algorithm
        kwargs = self.model_kwargs.copy()
        
        # Add action noise for continuous action space algorithms
        if self.algorithm_name in ["sac"] and hasattr(self.env, "action_space"):
            action_space = self.env.action_space
            if hasattr(action_space, "shape"):
                action_shape = action_space.shape[0]
                action_noise = NormalActionNoise(
                    mean=np.zeros(action_shape),
                    sigma=0.1 * np.ones(action_shape)
                )
                kwargs["action_noise"] = action_noise
        
        # Initialize the model
        self.model = algorithm_class(
            policy=self.policy,
            env=self.env,
            verbose=self.verbose,
            tensorboard_log=self.tensorboard_log,
            device=self.device,
            seed=self.seed,
            **kwargs
        )
        
        logger.info(f"Created {self.algorithm_name.upper()} model with {self.policy} policy")
    
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[Union[gym.Env, VecEnv]] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_dir: str = "./logs",
        save_freq: int = 10000,
        keep_checkpoints: int = 3,
        resource_check_freq: int = 1000,
        metrics_log_freq: int = 1000,
        early_stopping_patience: int = 0,
        custom_callbacks: Optional[List[Any]] = None,
        reset_num_timesteps: bool = False,
        tb_log_name: Optional[str] = None,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total timesteps to train for
            eval_env: Environment for evaluation (if None, no evaluation)
            eval_freq: How often to evaluate (in timesteps)
            n_eval_episodes: Number of episodes for evaluation
            log_dir: Directory to save logs
            save_freq: How often to save model checkpoints (in timesteps)
            keep_checkpoints: Maximum number of checkpoints to keep
            resource_check_freq: How often to check system resources (in timesteps)
            metrics_log_freq: How often to log trading metrics (in timesteps)
            early_stopping_patience: Number of evaluations without improvement before stopping
                                    (0 means no early stopping)
            custom_callbacks: List of additional callbacks
            reset_num_timesteps: Whether to reset the timestep counter
            tb_log_name: Name for the tensorboard log
            progress_bar: Whether to display a progress bar during training
        
        Returns:
            Dictionary with training metrics
        """
        # Create log directory
        ensure_dir_exists(log_dir)
        
        # Set or update tensorboard log name if not specified
        if tb_log_name is None:
            timestamp = int(time.time())
            tb_log_name = f"{self.algorithm_name}_{timestamp}"
        
        # Create callbacks list
        callbacks = get_callback_list(
            eval_env=eval_env,
            log_dir=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_freq=save_freq,
            keep_checkpoints=keep_checkpoints,
            resource_check_freq=resource_check_freq,
            metrics_log_freq=metrics_log_freq,
            early_stopping_patience=early_stopping_patience,
            custom_callbacks=custom_callbacks
        )
        
        # Start training
        logger.info(f"Starting training for {total_timesteps} timesteps")
        start_time = time.time()
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )
        
        # Update total timesteps
        if reset_num_timesteps:
            self.training_timesteps = total_timesteps
        else:
            self.training_timesteps += total_timesteps
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Get training metrics
        training_metrics = {
            "algorithm": self.algorithm_name,
            "total_timesteps": self.training_timesteps,
            "training_time": training_time,
            "training_time_per_step": training_time / total_timesteps
        }
        
        # Add to training metrics history
        self.train_metrics.append(training_metrics)
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return training_metrics
    
    def evaluate(
        self, 
        eval_env: Union[gym.Env, VecEnv], 
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        warn: bool = True
    ) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
        """
        Evaluate the agent.
        
        Args:
            eval_env: Environment for evaluation
            n_eval_episodes: Number of episodes for evaluation
            deterministic: Whether to use deterministic actions
            render: Whether to render the environment during evaluation
            callback: Callback to call at each step
            reward_threshold: Minimum reward threshold to consider evaluation successful
            return_episode_rewards: Whether to return episode rewards and lengths
            warn: Whether to warn if the environment has no monitor wrapper
        
        Returns:
            Mean reward and standard deviation, or list of rewards and episode lengths
        """
        logger.info(f"Evaluating model for {n_eval_episodes} episodes")
        start_time = time.time()
        
        # Evaluate policy
        results = evaluate_policy(
            model=self.model,
            env=eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            callback=callback,
            reward_threshold=reward_threshold,
            return_episode_rewards=return_episode_rewards,
            warn=warn
        )
        
        evaluation_time = time.time() - start_time
        
        # Process results
        if return_episode_rewards:
            episode_rewards, episode_lengths = results
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
        else:
            mean_reward, std_reward = results
            episode_rewards = None
            episode_lengths = None
        
        # Record evaluation results
        eval_data = {
            "timesteps": self.training_timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "n_eval_episodes": n_eval_episodes,
            "deterministic": deterministic,
            "evaluation_time": evaluation_time
        }
        
        self.eval_results.append(eval_data)
        
        logger.info(f"Evaluation result: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Return raw results
        return results
    
    def predict(
        self, 
        observation: np.ndarray, 
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the agent's prediction for an observation.
        
        Args:
            observation: The input observation
            state: The last states (for recurrent policies)
            episode_start: Whether the episode has started
            deterministic: Whether to use deterministic actions
        
        Returns:
            Predicted action and states
        """
        return self.model.predict(
            observation, 
            state=state, 
            episode_start=episode_start, 
            deterministic=deterministic
        )
    
    def save(self, path: str, include_optimizer: bool = True) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
            include_optimizer: Whether to include optimizer in the saved model
        """
        # Ensure directory exists
        ensure_dir_exists(os.path.dirname(path))
        
        # Save the model
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        
        # Save additional metadata
        metadata_path = f"{path}_metadata.npz"
        np.savez(
            metadata_path,
            algorithm=self.algorithm_name,
            training_timesteps=self.training_timesteps,
            eval_results=self.eval_results,
            train_metrics=self.train_metrics,
            seed=self.seed
        )
        logger.info(f"Model metadata saved to {metadata_path}")
    
    @classmethod
    def load(
        cls,
        path: str,
        env: Union[gym.Env, VecEnv],
        device: str = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        **kwargs
    ) -> "TradingRLAgent":
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            env: The environment to use
            device: Device to load the model on
            custom_objects: Custom objects that were saved with the model
            verbose: Verbosity level
            **kwargs: Additional arguments for model initialization
        
        Returns:
            Loaded TradingRLAgent instance
        """
        # Load metadata if available
        metadata_path = f"{path}_metadata.npz"
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                loaded_metadata = np.load(metadata_path, allow_pickle=True)
                metadata = {key: loaded_metadata[key] for key in loaded_metadata.files}
                logger.info(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        # Determine algorithm
        algorithm = metadata.get("algorithm", "ppo")
        if hasattr(algorithm, "item"):  # Convert numpy string to Python string
            algorithm = algorithm.item()
        
        seed = metadata.get("seed", 42)
        if hasattr(seed, "item"):
            seed = seed.item()
        
        # Create agent instance
        agent = cls(
            env=env,
            algorithm=algorithm,
            device=device,
            seed=seed,
            verbose=verbose,
            **kwargs
        )
        
        # Load model
        algorithm_class = cls.SUPPORTED_ALGORITHMS[algorithm.lower()]
        agent.model = algorithm_class.load(
            path=path,
            env=env,
            device=device,
            custom_objects=custom_objects
        )
        logger.info(f"Model loaded from {path}")
        
        # Set training timesteps
        training_timesteps = metadata.get("training_timesteps", 0)
        if hasattr(training_timesteps, "item"):
            training_timesteps = training_timesteps.item()
        agent.training_timesteps = training_timesteps
        
        # Set evaluation results and training metrics if available
        if "eval_results" in metadata:
            agent.eval_results = metadata["eval_results"].tolist()
        
        if "train_metrics" in metadata:
            agent.train_metrics = metadata["train_metrics"].tolist()
        
        return agent
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dict of model parameters
        """
        return self.model.get_parameters()
    
    def set_parameters(self, load_path_or_dict: Union[str, Dict[str, Any]], exact_match: bool = True) -> None:
        """
        Load parameters from a file or dictionary.
        
        Args:
            load_path_or_dict: Path to the parameters file or dictionary of parameters
            exact_match: Whether the parameter names should match exactly
        """
        self.model.set_parameters(load_path_or_dict, exact_match=exact_match)
    
    def get_env(self) -> Union[gym.Env, VecEnv]:
        """
        Get the training environment.
        
        Returns:
            The training environment
        """
        return self.model.get_env()
    
    def set_env(self, env: Union[gym.Env, VecEnv]) -> None:
        """
        Set the environment.
        
        Args:
            env: The new environment
        """
        self.model.set_env(env)
        self.env = env
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"TradingRLAgent(algorithm={self.algorithm_name}, policy={self.policy}, timesteps={self.training_timesteps})" 