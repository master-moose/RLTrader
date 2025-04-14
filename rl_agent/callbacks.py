#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom callbacks for the RL agent training process.

This module contains custom callbacks for monitoring, logging, and 
controlling the training process for RL agents.
"""

import os
import time
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from collections import deque
import psutil
import torch
import gc
import traceback
import itertools

from .utils import check_resources, ensure_dir_exists
from .env_wrappers import SafeTradingEnvWrapper

# Setup logger
logger = logging.getLogger("rl_agent")


class ResourceMonitorCallback(BaseCallback):
    """
    Callback for monitoring system resources during training.
    
    This callback periodically checks system resources (CPU, RAM, GPU) 
    and logs the information. It can also perform garbage collection
    when memory usage is high.
    """
    
    def __init__(
        self, 
        check_freq: int = 1000, 
        verbose: int = 1,
        warning_threshold: float = 0.9
    ):
        """
        Initialize the resource monitor callback.
        
        Args:
            check_freq: Frequency of resource checks in timesteps
            verbose: Verbosity level
            warning_threshold: Memory usage threshold for warnings (0.0-1.0)
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.warning_threshold = warning_threshold
        self.resource_history = []
        self.last_memory = 0
        self.last_cpu = 0
    
    def _on_step(self) -> bool:
        """
        Check resources periodically during training.
        
        Returns:
            Whether training should continue
        """
        if self.n_calls % self.check_freq == 0:
            # Check system resources
            resource_info = check_resources(
                logger=logger if self.verbose > 0 else None,
                warning_threshold=self.warning_threshold
            )
            
            # Store resource information
            resource_info["timestep"] = self.n_calls
            self.resource_history.append(resource_info)
            
            # Log resource usage
            if self.verbose > 0:
                mem_used = resource_info["memory_percent"]
                cpu_used = resource_info["cpu_percent"]
                logger.info(f"Step {self.n_calls}: Memory: {mem_used:.1f}%, CPU: {cpu_used:.1f}%")
                
                if "gpu_memory_used" in resource_info:
                    gpu_mem = resource_info["gpu_memory_used"]
                    gpu_util = resource_info["gpu_utilization"]
                    logger.info(f"GPU Memory: {gpu_mem:.1f}MB, GPU Utilization: {gpu_util:.1f}%")
            
            # Log to tensorboard if available
            if hasattr(self, 'logger') and self.logger:
                self.logger.record('resources/memory_mb', resource_info["memory_used"] * 1024)
                self.logger.record('resources/cpu_percent', cpu_used)
                if "gpu_memory_used" in resource_info:
                    self.logger.record('resources/gpu_memory_gb', resource_info["gpu_memory_used"] / 1024)
                if "gpu_utilization" in resource_info:
                    self.logger.record('resources/gpu_utilization', resource_info["gpu_utilization"])
            
            # Calculate change since last check
            memory_mb = resource_info["memory_used"] * 1024
            memory_change = memory_mb - self.last_memory
            
            # Log significant changes
            if abs(memory_change) > 500:  # Over 500MB change
                logger.warning(f"Memory usage changed by {memory_change:.1f}MB to {memory_mb:.1f}MB")
            
            # Check for critical memory usage (over 90% of system memory)
            system_memory = psutil.virtual_memory()
            if system_memory.percent > 90:
                logger.warning(f"CRITICAL: System memory usage at {system_memory.percent}%, consider stopping training")
                
                # Try to free some memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Stop if extremely critical (over 95%)
                if system_memory.percent > 95:
                    logger.error("CRITICAL MEMORY SHORTAGE: Stopping training to prevent system crash")
                    return False
            
            # Update last values
            self.last_memory = memory_mb
            self.last_cpu = cpu_used
        
        return True
    
    def get_resource_history(self) -> List[Dict[str, Any]]:
        """Get the resource usage history."""
        return self.resource_history


class TradingMetricsCallback(BaseCallback):
    """
    Callback for tracking trading-specific metrics during training.
    
    This callback collects metrics such as portfolio value, returns,
    Sharpe ratio, and trading actions from the environment.
    """
    
    def __init__(
        self, 
        log_freq: int = 1000, 
        verbose: int = 1,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the trading metrics callback.
        
        Args:
            log_freq: Frequency of metrics logging in timesteps
            verbose: Verbosity level
            log_dir: Directory to save logs
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir
        
        if log_dir:
            ensure_dir_exists(log_dir)
        
        # Metrics tracking
        self.metrics_history = []
        self.current_episode_metrics = {
            "rewards": [],
            "portfolio_values": [],
            "actions": [],
            "positions": [],
            "cash": []
        }
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        
        # Initialize timestamp
        self.last_time = time.time()
    
    def _on_step(self) -> bool:
        """
        Track metrics at each step during training.
        
        Returns:
            Whether training should continue
        """
        # Get environment info
        if hasattr(self.model, "env") and hasattr(self.model.env, "get_attr"):
            # For vectorized environments
            try:
                infos = self.model.env.get_attr("info_buffer")
                dones = self.model.env.get_attr("episode_returns")
                
                # Process only the first environment for simplicity
                if len(infos) > 0 and len(infos[0]) > 0:
                    info = infos[0][-1]  # Latest info from first env
                    
                    # Collect metrics if available
                    if "portfolio_value" in info:
                        self.current_episode_metrics["portfolio_values"].append(info["portfolio_value"])
                    
                    if "reward" in info:
                        self.current_episode_metrics["rewards"].append(info["reward"])
                    
                    if "action" in info:
                        self.current_episode_metrics["actions"].append(info["action"])
                    
                    if "position" in info:
                        self.current_episode_metrics["positions"].append(info["position"])
                    
                    if "cash" in info:
                        self.current_episode_metrics["cash"].append(info["cash"])
                
                # Check for episode completion
                if len(dones) > 0 and dones[0] is not None:
                    self._on_episode_end()
            
            except (AttributeError, KeyError) as e:
                if self.verbose > 0:
                    logger.warning(f"Could not get environment info: {e}")
        
        # Periodically log metrics
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()
        
        return True
    
    def _on_episode_end(self) -> None:
        """Process metrics at the end of an episode."""
        # Calculate episode statistics
        episode_reward = sum(self.current_episode_metrics["rewards"])
        episode_length = len(self.current_episode_metrics["rewards"])
        episode_time = time.time() - self.last_time
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_times.append(episode_time)
        
        # Calculate trading metrics
        portfolio_values = self.current_episode_metrics["portfolio_values"]
        if len(portfolio_values) > 1:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            returns = (final_value - initial_value) / initial_value
            
            # Calculate daily returns if we have enough data
            if len(portfolio_values) > 1:
                daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)  # Annualized
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
            else:
                daily_returns = []
                sharpe = 0
                max_drawdown = 0
            
            # Count trades
            actions = self.current_episode_metrics["actions"]
            trade_count = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
            
            # Create metrics summary
            metrics = {
                "episode": self.episode_count,
                "timestep": self.n_calls,
                "reward": episode_reward,
                "length": episode_length,
                "time": episode_time,
                "portfolio_initial": initial_value,
                "portfolio_final": final_value,
                "returns": returns,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "trade_count": trade_count
            }
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Log metrics
            if self.verbose > 0:
                logger.info(f"Episode {self.episode_count} finished: "
                          f"Reward={episode_reward:.2f}, "
                          f"Length={episode_length}, "
                          f"Return={returns*100:.2f}%, "
                          f"Sharpe={sharpe:.2f}, "
                          f"Trades={trade_count}")
            
            # Save detailed metrics to file
            if self.log_dir:
                self._save_episode_data()
        
        # Reset episode metrics
        self.current_episode_metrics = {
            "rewards": [],
            "portfolio_values": [],
            "actions": [],
            "positions": [],
            "cash": []
        }
        self.episode_count += 1
        self.last_time = time.time()
    
    def _log_metrics(self) -> None:
        """Log metrics to TensorBoard or logger."""
        # Example: Log latest episode reward
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if self.logger:
                self.logger.record('rollout/ep_rew_mean', mean_reward)
                self.logger.dump(step=self.num_timesteps)
            if self.verbose > 1:
                logger.debug(f"Step {self.n_calls}: Mean reward={mean_reward:.2f}")
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate the maximum drawdown from a list of portfolio values."""
        if not portfolio_values:
            return 0.0
        
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown

    def _save_episode_data(self) -> None:
        """
        Save detailed metrics from the last episode to a JSON file.
        """
        if not self.metrics_history:
            return
            
        last_episode_metrics = self.metrics_history[-1]
        
        # Combine episode metrics and last step's info
        episode_data = {
            **last_episode_metrics, 
            "detailed_metrics": self.current_episode_metrics
        }
        
        try:
            # Ensure all values are serializable
            for key, value in episode_data["detailed_metrics"].items():
                if isinstance(value, list) and value and isinstance(value[0], np.generic):
                    episode_data["detailed_metrics"][key] = [v.item() for v in value]
                elif isinstance(value, np.generic):
                     episode_data["detailed_metrics"][key] = value.item()
            
            episode_file = os.path.join(self.log_dir, f"episode_{self.episode_count}.json")
            with open(episode_file, 'w') as f:
                json.dump(episode_data, f, indent=4)
                
            if self.verbose > 1:
                logger.debug(f"Saved episode data to {episode_file}")
        except Exception as e:
            logger.error(f"Failed to save episode data: {e}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the historical metrics data."""
        return self.metrics_history


class BestModelCallback(EvalCallback):
    """
    Callback for evaluating the agent and saving the best model.
    Optionally includes early stopping.
    """
    
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        log_dir: str,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        verbose: int = 1,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
        callback_after_eval: Optional[Callable] = None,
        patience: int = 0
    ):
        """
        Initialize the BestModelCallback.
        
        Args:
            eval_env: Environment for evaluation
            log_dir: Directory to save logs and models
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions for evaluation
            verbose: Verbosity level
            best_model_save_path: Path to save the best model
            log_path: Path to save evaluation logs
            callback_after_eval: Callback to call after evaluation
            patience: Number of evaluations without improvement before early stopping (0=disabled)
        """
        if best_model_save_path is None:
            best_model_save_path = os.path.join(log_dir, "best_model")
        
        if log_path is None:
            log_path = os.path.join(log_dir, "evaluations")
            
        super().__init__(
            eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            deterministic=deterministic,
            n_eval_episodes=n_eval_episodes,
            callback_after_eval=callback_after_eval,
            verbose=verbose
        )
        
        self.patience = patience
        self.no_improvement_count = 0
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """
        Run evaluation and check for early stopping.
        
        Returns:
            Whether training should continue
        """
        continue_training = super()._on_step()
        
        # Check for early stopping if patience is set
        if continue_training and self.patience > 0 and self.last_mean_reward is not None:
            if self.last_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.last_mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
                if self.verbose > 0:
                    logger.info(f"No improvement in evaluation for {self.no_improvement_count} consecutive evaluations.")
                
                if self.no_improvement_count >= self.patience:
                    if self.verbose > 0:
                        logger.info(f"Early stopping triggered after {self.no_improvement_count} evaluations without improvement.")
                    return False  # Stop training
        
        return continue_training


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints during training.
    Manages saving frequency and keeps a limited number of checkpoints.
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        keep_checkpoints: int = 3,
        verbose: int = 1
    ):
        """
        Initialize the CheckpointCallback.
        
        Args:
            save_freq: Frequency of saving checkpoints in timesteps
            save_path: Path to save checkpoints
            name_prefix: Prefix for checkpoint filenames
            keep_checkpoints: Maximum number of checkpoints to keep (0=keep all)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.keep_checkpoints = keep_checkpoints
        
        ensure_dir_exists(self.save_path)
    
    def _on_step(self) -> bool:
        """
        Save a checkpoint periodically during training.
        
        Returns:
            Whether training should continue
        """
        if self.n_calls % self.save_freq == 0:
            # Generate checkpoint filename
            checkpoint_num = self.n_calls // self.save_freq
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps.zip")
            
            # Save the model
            self.model.save(path)
            
            if self.verbose > 0:
                logger.info(f"Saving model checkpoint to {path}")
            
            # Clean up old checkpoints if limit is set
            if self.keep_checkpoints > 0:
                self._cleanup_old_checkpoints()
        
        return True

    def _cleanup_old_checkpoints(self) -> None:
        """Remove the oldest checkpoints if the limit is exceeded."""
        try:
            # Get all checkpoint files
            checkpoints = [f for f in os.listdir(self.save_path) 
                           if f.startswith(self.name_prefix) and f.endswith("_steps.zip")]
            
            # Sort by timestep number (extracted from filename)
            checkpoints.sort(key=lambda f: int(f.split('_')[-2]))
            
            # Remove oldest checkpoints if limit exceeded
            if len(checkpoints) > self.keep_checkpoints:
                num_to_delete = len(checkpoints) - self.keep_checkpoints
                for i in range(num_to_delete):
                    file_to_delete = os.path.join(self.save_path, checkpoints[i])
                    os.remove(file_to_delete)
                    if self.verbose > 1:
                        logger.debug(f"Removed old checkpoint: {file_to_delete}")
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")


def get_callback_list(
    eval_env: Optional[Union[gym.Env, VecEnv]] = None,
    log_dir: str = "./logs",
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    save_freq: int = 10000,
    keep_checkpoints: int = 3,
    resource_check_freq: int = 1000,
    metrics_log_freq: int = 1000,
    early_stopping_patience: int = 0,
    custom_callbacks: Optional[List[BaseCallback]] = None
) -> CallbackList:
    """
    Create a list of callbacks for training.
    
    Args:
        eval_env: Environment for evaluation
        log_dir: Directory for logs and models
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        save_freq: Checkpoint save frequency
        keep_checkpoints: Number of checkpoints to keep
        resource_check_freq: Resource check frequency
        metrics_log_freq: Trading metrics log frequency
        early_stopping_patience: Patience for early stopping
        custom_callbacks: List of additional custom callbacks
    
    Returns:
        CallbackList containing all specified callbacks
    """
    callbacks = []
    
    # Checkpoint callback
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    callbacks.append(CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_path,
        keep_checkpoints=keep_checkpoints,
        verbose=1
    ))
    
    # Resource monitor callback
    callbacks.append(ResourceMonitorCallback(check_freq=resource_check_freq, verbose=1))
    
    # Trading metrics callback
    callbacks.append(TradingMetricsCallback(log_freq=metrics_log_freq, verbose=1, log_dir=log_dir))
    
    # Evaluation callback (includes best model saving and early stopping)
    if eval_env is not None:
        eval_log_path = os.path.join(log_dir, "evaluations")
        best_model_path = os.path.join(log_dir, "best_model")
        callbacks.append(BestModelCallback(
            eval_env=eval_env,
            log_dir=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=best_model_path,
            log_path=eval_log_path,
            patience=early_stopping_patience,
            verbose=1
        ))
        
    # Add custom callbacks if provided
    if custom_callbacks:
        callbacks.extend(custom_callbacks)
        
    return CallbackList(callbacks)

# TensorboardCallback moved from train_dqn.py
class TensorboardCallback(BaseCallback):
    """
    Custom callback for tracking metrics.
    This callback tracks detailed metrics about trading performance and logs them to TensorBoard.
    """
    
    def __init__(self, verbose=0, model_name=None, debug_frequency=250):
        """Initialize the callback with options for logging frequency and model name"""
        super(TensorboardCallback, self).__init__(verbose)
        self.debug_frequency = debug_frequency
        self.model_name = model_name if model_name else "model"
        
        # Initialize metrics
        self.episode_count = 0
        self.trade_count = 0
        self.last_trades = deque(maxlen=100)  # Track last 100 trades for analysis
        self.portfolio_values = []
        
        # Action tracking
        self.action_counts = {"sell": 0, "hold": 0, "buy": 0}
        self.episode_action_counts = {0: 0, 1: 0, 2: 0} # Per episode tracking
        self.consecutive_holds = 0
        self.max_consecutive_holds = 0
        self.hold_action_frequency = 0.0
        
        # Hold metrics tracking
        self.hold_durations = []  # List to track holding periods
        self.current_hold_duration = 0
        self.hold_histogram = {i: 0 for i in range(0, 21)}  # For holding periods 0-20+
        
        # Force sell tracking
        self.forced_sells = 0
        
        # Portfolio metrics
        self.initial_portfolio = None
        self.max_portfolio = 0.0
        self.min_portfolio = float('inf')
        
        # Oscillation metrics
        self.oscillation_counts = 0
        self.actions_sequence = []  # Track sequence of actions
        
        # Extra metrics for hold analysis
        self.hold_penalties = []
        self.average_hold_penalty = 0.0
        self.total_holds = 0
        self.hold_ratio = 0.0
        
        logger.info("TensorboardCallback initialized")
    
    def _on_step(self) -> bool:
        """
        Log metrics on each step.
        This is called at every step of the environment.
        """
        # Skip processing if we don't have the model yet
        if self.model is None:
            return True
        
        # Extract information from the environment
        # Cast from VecEnv wrapper if needed
        if hasattr(self.model.get_env(), 'envs'):
            env = self.model.get_env().envs[0]
        else:
            env = self.model.get_env()
            
        # Extract wrapper env if available
        if hasattr(env, 'env'):
            env = env.env
            
        # Unwrap to get the base environment and any wrapper class
        safe_wrapper = None
        base_env = None
        current_env = env
        
        # Find the SafeTradingEnvWrapper and base environment
        while hasattr(current_env, 'env'):
            if isinstance(current_env, SafeTradingEnvWrapper):
                safe_wrapper = current_env
            current_env = current_env.env
            if not hasattr(current_env, 'env'):
                base_env = current_env
                break
        
        # Get information from SafeTradingEnvWrapper if available
        if safe_wrapper is not None:
            hold_penalty = 0.0
            if hasattr(safe_wrapper, 'consecutive_holds'):
                self.consecutive_holds = safe_wrapper.consecutive_holds
                self.max_consecutive_holds = max(self.max_consecutive_holds, self.consecutive_holds)
                
                # Track hold penalties
                if self.consecutive_holds > 0:
                    # Calculate estimated penalty
                    base_penalty = 0.8  # From the wrapper
                    if self.consecutive_holds > 3:
                        additional = min((self.consecutive_holds - 3) * 0.1, 10.0)
                        hold_penalty = base_penalty + additional
                    else:
                        hold_penalty = base_penalty
                    
                    self.hold_penalties.append(hold_penalty)
                    self.average_hold_penalty = sum(self.hold_penalties) / len(self.hold_penalties)
            
            # Track action distribution
            if hasattr(safe_wrapper, 'action_counts'):
                self.action_counts["sell"] = safe_wrapper.action_counts.get(0, 0)
                self.action_counts["hold"] = safe_wrapper.action_counts.get(1, 0)
                self.action_counts["buy"] = safe_wrapper.action_counts.get(2, 0)
                
                total_actions = sum(self.action_counts.values())
                if total_actions > 0:
                    self.hold_action_frequency = self.action_counts["hold"] / total_actions
                    self.hold_ratio = self.action_counts["hold"] / max(1, self.action_counts["sell"] + self.action_counts["buy"])
                    
            # Track oscillation counts
            if hasattr(safe_wrapper, 'oscillation_count'):
                self.oscillation_counts = safe_wrapper.oscillation_count
                
            # Track action sequence
            if hasattr(safe_wrapper, 'action_history') and len(safe_wrapper.action_history) > 0:
                self.actions_sequence = safe_wrapper.action_history[-20:]  # Last 20 actions
        
        # Extract information from base environment
        if base_env is not None:
            if hasattr(base_env, 'holding_counter'):
                self.current_hold_duration = base_env.holding_counter
            
            # Track forced sells
            if hasattr(base_env, 'forced_sells'):
                self.forced_sells = base_env.forced_sells
                
            # Update hold duration histogram
            if self.current_hold_duration > 0:
                bucket = min(self.current_hold_duration, 20)  # Cap at 20+
                self.hold_histogram[bucket] = self.hold_histogram.get(bucket, 0) + 1
                
            # When not holding (hold_counter is 0), we just completed a holding period
            if self.current_hold_duration == 0 and hasattr(self, 'last_hold_duration') and self.last_hold_duration > 0:
                self.hold_durations.append(self.last_hold_duration)
                self.last_hold_duration = 0
            elif self.current_hold_duration > 0:
                self.last_hold_duration = self.current_hold_duration
        
        # Log to TensorBoard on regular intervals
        if self.num_timesteps % self.debug_frequency == 0 and self.verbose > 0 and self.logger:
            # Log standard metrics
            self.logger.record("environment/trade_count", self.trade_count)
            self.logger.record("environment/episode_count", self.episode_count)
            
            # Portfolio metrics
            if hasattr(base_env, 'portfolio_value'):
                portfolio_value = base_env.portfolio_value
                self.portfolio_values.append(portfolio_value)
                
                # Initialize initial portfolio if not set
                if self.initial_portfolio is None:
                    self.initial_portfolio = portfolio_value
                    
                # Update min/max portfolio values
                self.max_portfolio = max(self.max_portfolio, portfolio_value)
                self.min_portfolio = min(self.min_portfolio, portfolio_value)
                
                # Calculate and log portfolio performance
                portfolio_growth = (portfolio_value - self.initial_portfolio) / self.initial_portfolio if self.initial_portfolio > 0 else 0
                self.logger.record("portfolio/value", portfolio_value)
                self.logger.record("portfolio/growth", portfolio_growth)
                self.logger.record("portfolio/max_value", self.max_portfolio)
            
            # Action distribution
            self.logger.record("actions/sell_count", self.action_counts["sell"])
            self.logger.record("actions/hold_count", self.action_counts["hold"])
            self.logger.record("actions/buy_count", self.action_counts["buy"])
            self.logger.record("actions/hold_ratio", self.hold_ratio)
            self.logger.record("actions/hold_frequency", self.hold_action_frequency)
            
            # Holding metrics
            self.logger.record("holding/consecutive_holds", self.consecutive_holds)
            self.logger.record("holding/max_consecutive_holds", self.max_consecutive_holds)
            self.logger.record("holding/current_duration", self.current_hold_duration)
            self.logger.record("holding/average_hold_penalty", self.average_hold_penalty)
            
            # Force sell and oscillation metrics
            self.logger.record("trading/forced_sells", self.forced_sells)
            self.logger.record("trading/oscillation_count", self.oscillation_counts)
            
            # More detailed holding histogram
            for duration, count in self.hold_histogram.items():
                self.logger.record(f"holding_histogram/duration_{duration}", count)
                
            # Log action sequence pattern (converted to string representation)
            if len(self.actions_sequence) > 0:
                action_pattern = ''.join([str(a) for a in self.actions_sequence[-10:]])
                # Can't log strings directly, so log the pattern as a 'categorical' value
                # Use modulo to keep the value within a reasonable range for TensorBoard
                self.logger.record("actions/recent_pattern", hash(action_pattern) % 1000)
                
                # Check for problematic patterns like long holds using itertools
                hold_sequences = [len(list(g)) for k, g in itertools.groupby(self.actions_sequence) if k == 1]
                if hold_sequences:
                    self.logger.record("actions/longest_hold_sequence", max(hold_sequences))
            
            self.logger.dump(self.num_timesteps)
        
        return True
    
    def _extract_actions_from_envs(self):
        """Extract action counts directly from environments"""
        try:
            action_counts_updated = False
            
            if hasattr(self.training_env, 'envs'):
                for env_idx, env in enumerate(self.training_env.envs):
                    # Unroll wrappers to find the right env or wrapper
                    current_env = env
                    safe_wrapper = None
                    while hasattr(current_env, 'env'):
                        if isinstance(current_env, SafeTradingEnvWrapper):
                            safe_wrapper = current_env
                        current_env = current_env.env
                    base_env = current_env

                    # Try SafeTradingEnvWrapper first
                    if safe_wrapper and hasattr(safe_wrapper, 'action_counts'):
                        for action, count in safe_wrapper.action_counts.items():
                            if action == 0: self.action_counts["sell"] += count
                            if action == 1: self.action_counts["hold"] += count
                            if action == 2: self.action_counts["buy"] += count
                        action_counts_updated = True
                    # Fallback: try to get last action from base env if available
                    elif hasattr(base_env, 'last_action') and base_env.last_action is not None:
                        action = base_env.last_action
                        if action == 0: self.action_counts["sell"] += 1
                        if action == 1: self.action_counts["hold"] += 1
                        if action == 2: self.action_counts["buy"] += 1
                        action_counts_updated = True
            
            # If we couldn't extract any actions, use the action distribution from the logs
            if not action_counts_updated:
                # Get actions directly from episode information
                if hasattr(self, 'model') and hasattr(self.model, 'ep_info_buffer'):
                    for info in self.model.ep_info_buffer:
                        if 'action' in info:
                            action = info['action']
                            if action == 0: self.action_counts["sell"] += 1
                            if action == 1: self.action_counts["hold"] += 1
                            if action == 2: self.action_counts["buy"] += 1
                            action_counts_updated = True
                
                # If still no actions, use a fallback action history from the observation
                if not action_counts_updated and hasattr(self, 'locals') and 'obs' in self.locals:
                    obs = self.locals['obs']
                    if isinstance(obs, np.ndarray) and obs.shape[-1] > 15:  # Assuming augmented observation includes action history
                        # The augmented observation has action history one-hot encoded in positions beyond the original observation
                        # We can try to extract it, but this is implementation-specific
                        logger.warning("Fallback to extracting actions from observation - may not be accurate")
                        self.action_counts = {0: 1, 1: 5, 2: 1}  # Set some reasonable defaults based on logs
            
            if action_counts_updated:
                logger.info(f"Successfully extracted actions: {self.action_counts}")
            else:
                logger.warning("Failed to extract actions from any source")
                
        except Exception as e:
            logger.error(f"Error extracting actions from environments: {e}")
            logger.error(traceback.format_exc())
    
    def on_episode_end(self, episode_rewards, episode_lengths, episode_info=None):
        """Called at the end of an episode"""
        # Log episode action distribution
        total_actions = sum(self.episode_action_counts.values())
        if total_actions > 0:
            episode_action_table = "Episode Action Distribution:\n"
            episode_action_table += "-" * 40 + "\n"
            episode_action_table += "| Action | Count | Percentage |\n"
            episode_action_table += "-" * 40 + "\n"
            
            for action_name, action_id in {"Sell": 0, "Hold": 1, "Buy": 2}.items():
                count = self.episode_action_counts.get(action_id, 0)
                percentage = (count / max(1, total_actions)) * 100
                episode_action_table += f"| {action_name:<6} | {count:>5} | {percentage:>10.1f}% |\n"
            
            episode_action_table += "-" * 40
            logger.info(episode_action_table)
            
            # Reset episode action counts
            self.episode_action_counts = {0: 0, 1: 0, 2: 0}
        
        # Call parent method - ensure proper handling of episode end logic
        if hasattr(super(), 'on_episode_end'):
            super().on_episode_end(episode_rewards, episode_lengths, episode_info)
        
        self.episode_count += 1 # Increment episode count

# ResourceCheckCallback moved from train_dqn.py
# NOTE: This class is duplicated. Removing this one.
# class ResourceCheckCallback(BaseCallback):
#     """
#     Callback for monitoring system resources during training.
#     Helps prevent out-of-memory errors and tracks resource usage.
#     """
#
#     def __init__(self, check_interval=5000, verbose=0):
#         super(ResourceCheckCallback, self).__init__(verbose)
#         self.check_interval = check_interval
#         self.last_memory = 0
#         self.last_cpu = 0
#
#     def _on_step(self) -> bool:
#         """Check system resources periodically during training"""
#         if self.n_calls % self.check_interval == 0:
#             # Get memory usage
#             process = psutil.Process()
#             memory_info = process.memory_info()
#             memory_mb = memory_info.rss / 1024 / 1024
#             cpu_percent = process.cpu_percent()
#
#             # Log to tensorboard if available
#             if hasattr(self, 'logger') and self.logger:
#                 self.logger.record('resources/memory_mb', memory_mb)
#                 self.logger.record('resources/cpu_percent', cpu_percent)
#
#             # Calculate change since last check
#             memory_change = memory_mb - self.last_memory
#             # Unused variable cpu_change removed
#             # cpu_change = cpu_percent - self.last_cpu
#
#             # Log significant changes
#             if abs(memory_change) > 500:  # Over 500MB change
#                 logger.warning(
#                     f"Memory usage changed by {memory_change:.1f}MB "
#                     f"to {memory_mb:.1f}MB"
#                 )
#
#             # Check for critical memory usage (over 90% of system memory)
#             system_memory = psutil.virtual_memory()
#             if system_memory.percent > 90:
#                 logger.warning(
#                     f"CRITICAL: System memory usage at {system_memory.percent}%,"
#                     f" consider stopping training"
#                 )
#
#                 # Try to free some memory
#                 gc.collect()
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
#
#                 # Stop if extremely critical (over 95%)
#                 if system_memory.percent > 95:
#                     logger.error(
#                         "CRITICAL MEMORY SHORTAGE: Stopping training to "
#                         "prevent system crash"
#                     )
#                     return False
#
#             # Update last values
#             self.last_memory = memory_mb
#             self.last_cpu = cpu_percent
#
#             # Log GPU info if available
#             if torch.cuda.is_available():
#                 try:
#                     gpu_memory_allocated = (
#                         torch.cuda.memory_allocated() / (1024**3)
#                     )  # in GB
#                     gpu_memory_cached = (
#                         torch.cuda.memory_reserved() / (1024**3)
#                     )  # in GB
#                     # Default value if we can't get utilization
#                     gpu_utilization = -1
#
#                     # Try to get GPU utilization if possible
#                     try:
#                         import subprocess
#                         result = subprocess.check_output([
#                             'nvidia-smi',
#                             '--query-gpu=utilization.gpu',
#                             '--format=csv,noheader,nounits'
#                         ])
#                         gpu_utilization = float(result.decode('utf-8').strip())
#                     except Exception as gpu_err: # Catch specific exception
#                         logger.debug(f"Could not get GPU utilization: {gpu_err}")
#                         pass
#
#                     gpu_info_str = (
#                         f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated, "
#                         f"{gpu_memory_cached:.2f}GB cached"
#                     )
#                     if gpu_utilization >= 0:
#                         gpu_info_str += f", {gpu_utilization}% utilized"
#                     logger.info(gpu_info_str)
#
#                     # Log to tensorboard
#                     if hasattr(self, 'logger') and self.logger:
#                         self.logger.record(
#                             'resources/gpu_memory_gb', gpu_memory_allocated
#                         )
#                         if gpu_utilization >= 0:
#                             self.logger.record(
#                                 'resources/gpu_utilization', gpu_utilization
#                             )
#                 except Exception as e:
#                     logger.warning(f"Error checking GPU memory: {e}")
#
#         return True 