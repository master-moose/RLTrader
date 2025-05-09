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
import gc

# --- Third-Party Imports --- #
import gymnasium as gym
import psutil
import torch
from stable_baselines3.common.callbacks import (\
    BaseCallback, CallbackList, EvalCallback
)
from stable_baselines3.common.vec_env import VecEnv

# --- Local Module Imports --- #
from .utils import check_resources, ensure_dir_exists

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
                logger.info(f"Step {self.n_calls}: Memory: {mem_used:.1f}%, "
                           f"CPU: {cpu_used:.1f}%")
                
                if "gpu_memory_used" in resource_info:
                    gpu_mem = resource_info["gpu_memory_used"]
                    gpu_util = resource_info["gpu_utilization"]
                    logger.info(f"GPU Memory: {gpu_mem:.1f}MB, "
                               f"GPU Utilization: {gpu_util:.1f}%")
            
            # Log to tensorboard if available
            if hasattr(self, 'logger') and self.logger:
                # Use underscores instead of slashes for Ray Tune compatibility
                self.logger.record('resources_memory_mb', 
                                   resource_info["memory_used"] * 1024)
                self.logger.record('resources_cpu_percent', cpu_used)
                if "gpu_memory_used" in resource_info:
                    self.logger.record('resources_gpu_memory_gb', 
                                       resource_info["gpu_memory_used"] / 1024)
                if "gpu_utilization" in resource_info:
                    self.logger.record('resources_gpu_utilization', 
                                       resource_info["gpu_utilization"])
            
            # Calculate change since last check
            memory_mb = resource_info["memory_used"] * 1024
            memory_change = memory_mb - self.last_memory
            
            # Log significant changes
            if abs(memory_change) > 500:  # Over 500MB change
                logger.warning(f"Memory usage changed by {memory_change:.1f}MB "
                              f"to {memory_mb:.1f}MB")
            
            # Check for critical memory usage (over 90% of system memory)
            system_memory = psutil.virtual_memory()
            if system_memory.percent > 90:
                logger.warning(f"CRITICAL: System memory usage at "
                              f"{system_memory.percent}%, "
                              f"consider stopping training")
                
                # Try to free some memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Stop if extremely critical (over 95%)
                if system_memory.percent > 95:
                    logger.error("CRITICAL MEMORY SHORTAGE: Stopping training "
                                 "to prevent system crash")
                    return False
            
            # Update last values
            self.last_memory = memory_mb
            self.last_cpu = cpu_used
        
        return True
    
    def get_resource_history(self) -> List[Dict[str, Any]]:
        """Get the resource usage history."""
        return self.resource_history


def convert_to_native_types(data):
    """Recursively convert numpy types to native Python types."""
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return convert_to_native_types(data.tolist()) # Convert numpy arrays to lists first
    # Add checks for other potential non-serializable types if necessary
    return data


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
        """Log metrics at the specified frequency using info dict."""
        if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
            if not self.locals['infos']:
                return True  # Should not happen with Monitor wrapper
            
            info = self.locals['infos'][0]
            
            # --- Collect metrics from info dict --- 
            if "portfolio_value" in info:
                self.current_episode_metrics["portfolio_values"].append(
                    info["portfolio_value"]
                )
            
            # Use step reward from locals if available, otherwise check info
            step_reward = self.locals['rewards'][0]
            if 'rewards' in self.locals:
                step_reward = self.locals['rewards'][0]
            else:
                step_reward = info.get("reward", 0.0)
            self.current_episode_metrics["rewards"].append(step_reward)

            # Get action from model prediction if needed, or check info
            # action = self.locals['actions'][0] \n            #     if 'actions' in self.locals else info.get("action", None)
            # self.current_episode_metrics["actions"].append(action) # Action less useful

            if "position" in info:  # Assuming base env or wrapper adds this
                self.current_episode_metrics["positions"].append(info["position"])
            
            if "cash" in info:  # Assuming base env or wrapper adds this
                self.current_episode_metrics["cash"].append(info["cash"])
            
            # Add metrics from SafeTradingEnvWrapper if present
            for key in info:
                if key.startswith('wrapper_'):
                    metric_name = key.replace('wrapper_', '')
                    if metric_name not in self.current_episode_metrics:
                        self.current_episode_metrics[metric_name] = []
                    self.current_episode_metrics[metric_name].append(info[key])

            # Check for episode completion using Monitor info
            if "episode" in info:
                self._on_episode_end()
                # Reset for next episode
                self.current_episode_metrics = { 
                    "portfolio_values": [], "rewards": [], "actions": [], 
                    "positions": [], "cash": [], "consecutive_holds": [],
                    "oscillation_count": [], "current_cooldown": [],
                    "sharpe_ratio": [], "max_drawdown": [], "portfolio_growth_rate": [],
                    "successful_trade_streak": [], "forced_actions": [],
                    "cooldown_violations": []
                }
                self.episode_start_step = self.n_calls
        
        return True
    
    def _on_episode_end(self) -> None:
        """Process metrics at the end of an episode."""
        # Ensure Monitor wrapper info is available
        if 'episode' not in self.locals['infos'][0]:
            logger.warning("Monitor wrapper 'episode' info not found in _on_episode_end. Skipping metric logging.")
            return
            
        ep_info = self.locals['infos'][0]['episode']
        episode_reward = ep_info['r'] # Get reward from Monitor
        episode_length = ep_info['l'] # Get length from Monitor
        # episode_time = ep_info['t'] # Get time from Monitor
        # --- Remove manual calculation ---
        # episode_reward = sum(self.current_episode_metrics["rewards"])
        # episode_length = len(self.current_episode_metrics["rewards"])
        episode_time = time.time() - self.last_time # Keep manual time for now, Monitor time might be different
        
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
                sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) \
                         * np.sqrt(252)  # Annualized
                max_drawdown = self._calculate_max_drawdown(portfolio_values)
            else:
                daily_returns = []
                sharpe = 0
                max_drawdown = 0
            
            # Count trades
            actions = self.current_episode_metrics["actions"]
            trade_count = sum(1 for i in range(1, len(actions)) \
                              if actions[i] != actions[i-1])
            
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
        # Ensure deep copy to avoid modifying original data
        episode_data = {
            **last_episode_metrics,
            # Convert the detailed metrics dict recursively
            "detailed_metrics": convert_to_native_types(self.current_episode_metrics) 
        }
        
        try:
            # Convert the overall episode_data dict as well, just in case
            serializable_episode_data = convert_to_native_types(episode_data)
            
            episode_file = os.path.join(self.log_dir,
                                        f"episode_{self.episode_count}.json")
            with open(episode_file, 'w') as f:
                # Dump the fully converted data
                json.dump(serializable_episode_data, f, indent=4) 

            if self.verbose > 1:
                logger.debug(f"Saved episode data to {episode_file}")
        except Exception as e:
            logger.error(f"Failed to save episode data: {e}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the metrics history."""
        return self.metrics_history


class EpisodeInfoCallback(BaseCallback):
    """
    Callback to log detailed episode information when an episode ends,
    including custom metrics like total trades.
    """
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_count = 0
        # Use a different name to avoid conflict with BaseCallback.logger
        self._info_logger = logging.getLogger("rl_agent")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Check if any environments are done
        if "dones" not in self.locals or self.locals["dones"] is None:
            return True # Should not happen with VecEnv

        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                # Extract info for the finished episode
                info = self.locals["infos"][i].copy() # Use .copy() to avoid modifying original

                # Get metrics from Monitor's 'episode' sub-dict
                ep_info = info.get("episode", {})
                ep_reward = ep_info.get("r", "N/A")
                ep_length = ep_info.get("l", "N/A")

                # Get custom metrics directly from info dict (from TradingEnvironment)
                total_trades = info.get("total_trades", "N/A")
                episode_return = info.get("episode_return", 0.0) # This is a fraction
                sharpe_ratio = info.get("sharpe_ratio_episode", "N/A")
                sharpe_rolling = info.get("sharpe_ratio_rolling", "N/A")
                calmar_ratio = info.get("calmar_ratio", "N/A")
                sortino_ratio = info.get("sortino_ratio", "N/A")

                # Format values for logging
                reward_str = f"{ep_reward:.2f}" if isinstance(ep_reward, (int, float)) else str(ep_reward)
                # --- SAFELY FORMAT RETURN (Handling NaN/Inf) --- #
                return_val = info.get("episode_return", None)
                if isinstance(return_val, (int, float)) and np.isfinite(return_val):
                    return_str = f"{return_val * 100:.2f}%"
                else:
                    return_str = "N/A"
                # --- END SAFE FORMAT --- #
                sharpe_str = f"{sharpe_ratio:.2f}" if isinstance(sharpe_ratio, (int, float)) else str(sharpe_ratio)
                sharpe_rolling_str = f"{sharpe_rolling:.2f}" if isinstance(sharpe_rolling, (int, float)) else str(sharpe_rolling)
                calmar_str = f"{calmar_ratio:.4f}" if isinstance(calmar_ratio, (int, float)) else str(calmar_ratio)
                sortino_str = f"{sortino_ratio:.4f}" if isinstance(sortino_ratio, (int, float)) else str(sortino_ratio)
                trades_str = str(total_trades)
                length_str = str(ep_length)

                # --- Extract and process CUMULATIVE reward components information ---
                cumulative_components_str = ""
                if 'cumulative_reward_components' in info:
                    cumulative_components = info['cumulative_reward_components']
                    components_to_log = {k: v for k, v in cumulative_components.items() }
                    if components_to_log:
                        components_str = ', '.join([f"{k}: {v:.4f}" for k, v in components_to_log.items()])
                        cumulative_components_str = f"\nCum. Rewards: {components_str}"
                # --- End Extract ---

                # Format log message using all tracked ratios
                log_str = (
                    f"Episode {self.episode_count} finished: "
                    f"Reward={reward_str}, Length={length_str}, "
                    f"Return={return_str}, Sharpe={sharpe_str}, "
                    f"Sharpe(rolling)={sharpe_rolling_str}, Calmar={calmar_str}, Sortino={sortino_str}, "
                    f"Trades={trades_str}{cumulative_components_str}"
                )
                self._info_logger.info(log_str)

        return True


class BestModelCallback(EvalCallback):
    """
    Callback for evaluating the agent and saving the best model based on evaluation results.
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
            patience: Number of evaluations without improvement before early 
                      stopping (0=disabled)
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

        # --- Added Debug Log ---
        logger.info(f"BestModelCallback initialized: eval_freq={self.eval_freq}, patience={self.patience}")
        # -----------------------
    
    def _on_step(self) -> bool:
        # --- Added Debug Log ---
        logger.debug(f"BestModelCallback _on_step START: n_calls={self.n_calls}, eval_freq={self.eval_freq}, last_mean_reward={self.last_mean_reward}")
        # -----------------------

        continue_training = super()._on_step()

        # --- Added Debug Log ---
        logger.debug(f"BestModelCallback _on_step AFTER super: continue={continue_training}, last_mean_reward={self.last_mean_reward}, no_improvement_count={self.no_improvement_count}")
        # -----------------------

        # Check for early stopping if patience is set
        if continue_training and self.patience > 0 and \
           self.last_mean_reward is not None and \
           not np.isinf(self.last_mean_reward): # Check if eval has run at least once

            # Note: self.last_mean_reward is updated by super()._on_step() only if eval runs
            # We need to compare with the *previous* best mean reward
            current_eval_reward = self.last_mean_reward # Reward from the *most recent* evaluation

            if current_eval_reward > self.best_mean_reward:
                # --- Added Debug Log ---
                logger.debug(f"Evaluation improved: {current_eval_reward:.4f} > {self.best_mean_reward:.4f}. Resetting patience.")
                # -----------------------
                self.best_mean_reward = current_eval_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                # --- Added Debug Log ---
                logger.debug(f"Evaluation did not improve ({current_eval_reward:.4f} <= {self.best_mean_reward:.4f}). Patience count: {self.no_improvement_count}/{self.patience}")
                # -----------------------

                if self.verbose > 0:
                     # Keep existing info log for clarity
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
            keep_checkpoints: Maximum number of checkpoints to keep 
                              (0=keep all)
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
            # checkpoint_num = self.n_calls // self.save_freq # Unused
            path = os.path.join(self.save_path, 
                                f"{self.name_prefix}_{self.n_calls}_steps.zip")
            
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
                           if f.startswith(self.name_prefix) and \
                           f.endswith("_steps.zip")]
            
            # Sort by timestep number (extracted from filename)
            checkpoints.sort(key=lambda f: int(f.split('_')[-2]))
            
            # Remove oldest checkpoints if limit exceeded
            if len(checkpoints) > self.keep_checkpoints:
                num_to_delete = len(checkpoints) - self.keep_checkpoints
                for i in range(num_to_delete):
                    file_to_delete = os.path.join(self.save_path, 
                                                checkpoints[i])
                    os.remove(file_to_delete)
                    if self.verbose > 1:
                        logger.debug(f"Removed old checkpoint: {file_to_delete}")
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")


# Modified get_callback_list to accept curriculum params
def get_callback_list(
    eval_env: Optional[Union[gym.Env, VecEnv]] = None,
    log_dir: str = "./logs",
    eval_freq: int = 500000,
    n_eval_episodes: int = 5,
    save_freq: int = 500000,
    keep_checkpoints: int = 3,
    resource_check_freq: int = 1000,
    metrics_log_freq: int = 1000,
    early_stopping_patience: int = 0,
    custom_callbacks: Optional[List[BaseCallback]] = None,
    checkpoint_save_path: str = "./checkpoints",
    model_name: str = "rl_model",
    target_transaction_fee: float = 0.001,
    curriculum_duration_fraction: float = 0.5
) -> CallbackList:
    """
    Constructs the list of callbacks for training.

    Args:
        eval_env: Environment for evaluation
        log_dir: Directory for saving logs
        eval_freq: Frequency of evaluation
        n_eval_episodes: Number of episodes for evaluation
        save_freq: Frequency of saving model checkpoints
        keep_checkpoints: Number of checkpoints to keep
        resource_check_freq: Frequency of resource usage checks
        metrics_log_freq: Frequency of trading metrics logging
        early_stopping_patience: Patience for early stopping (0=disable)
        custom_callbacks: List of additional custom callbacks
        checkpoint_save_path: Base path to save checkpoints
        model_name: Name of the model (used for checkpoint naming)
        target_transaction_fee: Target fee for curriculum learning
        curriculum_duration_fraction: Fraction of training for curriculum

    Returns:
        CallbackList containing all configured callbacks.
    """
    callbacks = []
    
    # Checkpoint save path setup
    checkpoint_dir = os.path.join(checkpoint_save_path, model_name)
    ensure_dir_exists(checkpoint_dir)
    
    # Curriculum Callback (if duration > 0)
    # Check if total_timesteps is available in the config/model
    # Note: This requires passing total_timesteps or accessing model.total_timesteps
    # For now, assume it's handled elsewhere or add a parameter if needed.
    if curriculum_duration_fraction > 0:
        callbacks.append(CurriculumCallback(
            target_fee=target_transaction_fee, 
            curriculum_duration_fraction=curriculum_duration_fraction,
            verbose=1
        ))

    # Resource Monitoring Callback
    callbacks.append(ResourceMonitorCallback(
        check_freq=max(1, resource_check_freq), 
        verbose=1)
    )
    
    # Trading Metrics Logging Callback (COMMENTED OUT FOR DEBUGGING)
    # callbacks.append(TradingMetricsCallback(
    #     log_freq=max(1, metrics_log_freq), 
    #     verbose=1, 
    #     log_dir=os.path.join(log_dir, "trading_metrics"))
    # )
    
    # --- Evaluation Callback ---
    # TODO: Replace EvalCallback with BestModelCallback if custom metrics are needed
    # if eval_env is not None:
    #     eval_logger = logging.getLogger("rl_agent.eval_callback")
    #     eval_logger.info(
    #         f"Setting up EvalCallback/BestModelCallback: eval_freq={eval_freq}, "
    #         f"n_eval_episodes={n_eval_episodes}, patience={early_stopping_patience}"
    #     )
        
    #     # Determine path for best model (inside the specific trial's log dir)
    #     best_model_save_path = os.path.join(log_dir, "best_model")
    #     ensure_dir_exists(best_model_save_path)
    #     eval_log_path = os.path.join(log_dir, "eval_logs") # Log eval stats here
    #     ensure_dir_exists(eval_log_path)

    #     # --- Use BestModelCallback --- #
    #     # This callback handles both evaluation and saving the best model based on reward
    #     eval_callback = BestModelCallback(
    #         eval_env=eval_env,
    #         n_eval_episodes=n_eval_episodes,
    #         eval_freq=eval_freq,
    #         log_path=eval_log_path, # Separate log path for eval
    #         best_model_save_path=best_model_save_path, # Save best model here
    #         deterministic=True,
    #         render=False,
    #         verbose=1,
    #         patience=early_stopping_patience # Pass patience here
    #     )
    #     callbacks.append(eval_callback)
    # else:
    #     logger.warning("No validation environment provided. Skipping EvalCallback.")
    
    # --- Checkpoint Callback --- #
    # Save checkpoints periodically regardless of evaluation performance
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, save_freq),
        save_path=checkpoint_dir, # Use the dedicated checkpoint directory
        name_prefix=model_name, # Use model_name for prefix
        keep_checkpoints=keep_checkpoints,
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # --- Add Custom Episode Info Logger ---
    # This will log details AFTER Monitor has processed the episode end
    callbacks.append(EpisodeInfoCallback(verbose=1))

    # Tensorboard Logging Callback (Optional - SB3 logger is usually preferred)
    # callbacks.append(TensorboardCallback(verbose=1, model_name=model_name))

    # Add custom callbacks if provided
    if custom_callbacks:
        callbacks.extend(custom_callbacks)

    return CallbackList(callbacks)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging detailed trading metrics to TensorBoard.
    Relies on metrics being present in the `info` dictionary returned by env.step().
    It expects the standard `Monitor` wrapper info ('r', 'l', 't') and potentially
    custom metrics added by other wrappers (e.g., 'wrapper_*').
    """

    def __init__(self, verbose=0, model_name=None, log_freq=250):
        """
        Initialize the callback.

        Args:
            verbose: Verbosity level.
            model_name: Name of the model for logging clarity.
            log_freq: Log every N steps.
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.model_name = model_name if model_name else "model"
        self.episode_count = 0
        # Keep simple counters/trackers if needed across steps/episodes
        self.total_trades_episode = 0
        self.last_portfolio_value = None

    def _on_step(self) -> bool:
        """Log metrics every log_freq steps using the info dict."""
        if self.log_freq > 0 and self.n_calls % self.log_freq == 0:
            if self.logger is None:
                return True  # No logger configured

            # --- Get info from the first environment (assuming VecEnv) ---
            if 'infos' not in self.locals or not self.locals['infos']:
                logger.warning("No 'infos' dictionary found in locals. "
                             "Cannot log metrics.")
                return True
            
            info = self.locals['infos'][0]  # Get info dict for the first env

            # --- Log standard Monitor wrapper info --- 
            if "episode" in info:
                ep_info = info["episode"]
                self.logger.record("rollout/ep_rew_mean", ep_info['r'])
                self.logger.record("rollout/ep_len_mean", ep_info['l'])
                self.episode_count += 1  # Increment based on Monitor signal
                # Reset episode-specific counters
                self.total_trades_episode = 0
                self.last_portfolio_value = None
            
            self.logger.record("rollout/episode_count", self.episode_count)

            # --- Log metrics directly from the info dictionary --- 
            # Keys should be added by TradingEnvironment or wrappers
            loggable_metrics = [
                # From TradingEnvironment._get_info()
                'price', 'balance', 'shares_held', 'asset_value', 
                'portfolio_value', 'total_trades', 'total_buys', 
                'total_sells', 'total_holds', 'drawdown', 'cash_ratio',
                'returns_mean', 'returns_std', 'sharpe_ratio',
                # From SafeTradingEnvWrapper (prefixed)
                'wrapper_consecutive_holds', 'wrapper_oscillation_count',
                'wrapper_current_cooldown', 'wrapper_sharpe_ratio',
                'wrapper_max_drawdown', 'wrapper_portfolio_growth_rate',
                'wrapper_successful_trade_streak', 'wrapper_forced_actions',
                'wrapper_cooldown_violations'
            ]

            for key in loggable_metrics:
                if key in info:
                    # Use appropriate prefixes for TensorBoard readability
                    if key.startswith('wrapper_'):
                        tb_key = f"wrapper/{key.replace('wrapper_', '')}"
                    elif key in ['price', 'balance', 'shares_held', 
                                 'asset_value', 'portfolio_value', 'cash_ratio']:
                        tb_key = f"account/{key}"
                    elif key.startswith('returns') or key in \
                         ['sharpe_ratio', 'drawdown']:
                        tb_key = f"performance/{key}"
                    else:
                        tb_key = f"trading/{key}"
                    
                    self.logger.record(tb_key, info[key])
                # else: # Optional: Warn if expected key is missing
                #     logger.debug(f"Metric '{key}' not found in info dict "
                #                  f"at step {self.n_calls}")
            
            # --- Calculate and Log Custom/Derived Metrics --- 
            # Example: Trade count for the current episode (if available)
            if 'total_trades' in info:
                self.total_trades_episode = info['total_trades']
            self.logger.record("trading/ep_trade_count", 
                               self.total_trades_episode)

            # Example: Step reward
            if 'rewards' in self.locals:
                step_reward = self.locals['rewards'][0]
                self.logger.record("rollout/step_reward", step_reward)

            self.logger.dump(step=self.num_timesteps)

        return True

    # Remove _extract_actions_from_envs and on_episode_end as we now
    # rely on info dict and Monitor wrapper signals

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

# --- NEW CURRICULUM CALLBACK --- #


class CurriculumCallback(BaseCallback):
    """
    Callback to implement curriculum learning by gradually increasing a parameter.
    Currently designed to schedule the transaction fee.
    
    Args:
        target_fee (float): The final target transaction fee.
        curriculum_duration_fraction (float): Fraction of total timesteps over 
                                          which to increase the fee.
        verbose (int): Verbosity level.
    """
    def __init__(
        self, 
        target_fee: float,
        curriculum_duration_fraction: float = 0.5, 
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.target_fee = target_fee
        self.curriculum_duration_fraction = curriculum_duration_fraction
        self.curriculum_end_step = None
        self.last_logged_fee = -1.0

    def _on_training_start(self) -> None:
        """Calculate the end step for the curriculum."""
        total_timesteps = self.locals['total_timesteps']
        self.curriculum_end_step = int(total_timesteps * \
                                     self.curriculum_duration_fraction)
        if self.verbose > 0:
            logger.info(f"Curriculum Learning: Transaction fee will increase "
                        f"from 0.0 to {self.target_fee:.6f} over "
                        f"{self.curriculum_end_step} steps.")
        # Initialize fee to 0 in the environment
        self.training_env.set_attr('current_transaction_fee', 0.0)

    def _on_step(self) -> bool:
        """
        Update the transaction fee in the environment based on training progress.
        """
        if self.curriculum_end_step is None:
            logger.warning("Curriculum end step not set. "
                         "Training may not have started correctly.")
            return True
        
        current_step = self.num_timesteps
        current_fee_target = 0.0
        
        if current_step < self.curriculum_end_step:
            # Linearly interpolate the fee
            progress = current_step / self.curriculum_end_step
            current_fee_target = self.target_fee * progress
        else:
            # Curriculum finished, use target fee
            current_fee_target = self.target_fee
        
        # Get the current fee from the first environment using get_attr
        try:
            # get_attr returns a list, one item per env
            env_current_fees = self.training_env.get_attr('current_transaction_fee')
            env_current_fee = env_current_fees[0] if env_current_fees else -1.0
        except AttributeError:
             # Handle cases where the attribute might not exist yet (shouldn't happen after _on_training_start)
             env_current_fee = -1.0
             logger.warning("Could not retrieve 'current_transaction_fee' via get_attr.")

        # Check if the fee has actually changed significantly enough to warrant 
        # logging/setting
        if abs(current_fee_target - env_current_fee) > 1e-9:
            self.training_env.set_attr('current_transaction_fee', current_fee_target)
             
            # Log the change periodically or when it changes significantly
            if self.verbose > 1 and \
               abs(current_fee_target - self.last_logged_fee) > self.target_fee / 10:
                logger.info(f"Step {current_step}: Curriculum Fee set to "
                           f"{current_fee_target:.6f}")
                self.last_logged_fee = current_fee_target
                 
        # Set final fee explicitly and log once
        if current_step == self.curriculum_end_step and self.verbose > 0:
            self.training_env.set_attr('current_transaction_fee', self.target_fee)
            logger.info(f"Step {current_step}: Curriculum finished. Fee fixed at "
                        f"{self.target_fee:.6f}")
            self.last_logged_fee = self.target_fee

        return True

# --- END NEW CURRICULUM CALLBACK --- # 