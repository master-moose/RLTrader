#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training script for the LSTM-DQN agent.

This script provides a command-line interface for training and evaluating 
the LSTM-DQN reinforcement learning agent on financial time series data.
"""

# --- Standard Library Imports --- #
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import traceback

# Add parent directory to path *before* attempting local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Third-Party Imports --- #
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import ray
from ray import tune
# Unused imports removed: ASHAScheduler, OptunaSearch, HyperOptSearch

# --- Stable Baselines3 Imports --- # 
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm as BaseRLModel
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy # For PPO/A2C
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize
)
from stable_baselines3.dqn.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines3.common.callbacks import BaseCallback

# --- SB3 Contrib Imports --- #
from sb3_contrib import QRDQN, RecurrentPPO
# QRDQNPolicy removed as it was unused

# --- Local Module Imports --- #
from rl_agent.callbacks import get_callback_list
from rl_agent.data.data_loader import DataLoader
from rl_agent.environment import TradingEnvironment
from rl_agent.models import LSTMFeatureExtractor
from rl_agent.utils import (
    calculate_trading_metrics, check_resources,
    create_evaluation_plots, ensure_dir_exists,
    load_config, save_config, set_seeds, setup_logger,
    setup_sb3_logger
)


# --- Global Settings --- # 

# Ray availability check
RAY_AVAILABLE = False
try:
    # Just check if ray exists, specific tune components imported above
    import ray 
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Initialize logger globally (will be configured later)
logger = logging.getLogger(__name__)


# --- Ray Tune Callback Class --- #

class TuneReportCallback(BaseCallback):
    """Callback to report metrics to Ray Tune, focusing on scheduler needs."""
    def __init__(self):
        super().__init__(verbose=0)
        self.last_reported_timestep = -1  # Track last report
        # Remove episode tracking variables:
        # self.episode_rewards = []
        # self.last_episode_rewards = []

    def _on_init(self) -> None:
        """Ensure the logger is available."""
        # Use self.logger which is automatically set by BaseCallback
        if self.logger is None:
            # Use the logger instance configured elsewhere in the script
            global_logger = logging.getLogger("rl_agent")
            global_logger.warning("SB3 logger not available in TuneReportCallback's logger attribute.")

        # Set up initial logging
        global_logger = logging.getLogger("rl_agent")
        # Modify log message
        global_logger.info("TuneReportCallback initialized - will report metrics when available.")

    def _normalize_and_combine_metrics(self, reward, explained_variance):
        """
        Create a normalized combined score that equally weights reward and explained variance.

        Args:
            reward: The mean reward value, can be very large or small
            explained_variance: The explained variance, typically between -1 and 1

        Returns:
            A combined normalized score between 0 and 1 (higher is better)
        """
        # Handle None or non-numeric values robustly
        if reward is None or not isinstance(reward, (int, float, np.number)):
            # If reward is missing, we cannot calculate a meaningful score
            return 0.5 # Return neutral score if reward is unavailable
        if explained_variance is None or not isinstance(explained_variance, (int, float, np.number)):
            explained_variance = 0.0 # Default variance to 0 if unavailable

        # Use tanh to normalize reward to [-1, 1] range regardless of magnitude
        # Scale factor of 1000 helps with gradual scaling for typical reward ranges
        normalized_reward = np.tanh(reward / 1000.0)

        # Explained variance is typically between -1 and 1 already
        # Clip it just to be safe
        normalized_variance = np.clip(explained_variance, -1.0, 1.0)

        # Calculate combined score (equal weight to both normalized metrics)
        # We add 1 to each to get values in [0, 2] range, then divide by 4 to get [0, 1]
        combined_score = ((normalized_reward + 1.0) + (normalized_variance + 1.0)) / 4.0

        return combined_score

    # Remove _on_rollout_end method entirely
    # def _on_rollout_end(self) -> None:
    #     """
    #     Capture episode rewards at the end of each rollout.
    #     This ensures we have the latest episode rewards.
    #     """
    #     ... (removed code) ...

    def _get_current_reward_metrics(self) -> Optional[float]:
        """
        Get the current reward metrics, prioritizing the SB3 logger.
        Returns None if the primary metric ('rollout/ep_rew_mean') is not available.
        """
        global_logger = logging.getLogger("rl_agent")
        reward_value = None

        # Method 1: Try to get from SB3 logger (Primary Method)
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
                try:
                    reward_value = float(self.model.logger.name_to_value["rollout/ep_rew_mean"])
                    global_logger.debug(f"Using SB3 logger rewards: {reward_value}")
                except (ValueError, TypeError):
                     global_logger.warning("Could not convert SB3 logger reward to float.")
                     reward_value = None # Treat conversion error as unavailable

        # Method 2: Try to get directly from environment if logger failed
        if reward_value is None and hasattr(self.model, "env"):
            try:
                # Try to access VecEnv's episode rewards
                if hasattr(self.model.env, "get_episode_rewards"):
                    rewards = self.model.env.get_episode_rewards()
                    if rewards:
                        # Use the mean of recent episodes if available
                        reward_value = float(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))
                        global_logger.debug(f"Using env episode rewards: {reward_value}")
            except (AttributeError, IndexError, ValueError, TypeError) as e:
                global_logger.debug(f"Could not get rewards from env directly or convert: {e}")
                reward_value = None # Treat error as unavailable

        # If reward_value is still None, it means neither reliable source had it.
        if reward_value is None:
            global_logger.debug("No reward metric found from SB3 logger or env buffer yet.")
            return None # Explicitly return None

        return reward_value # Return the found float value

    def _on_step(self) -> bool:
        """Report metrics to Ray Tune only when key metrics are available."""
        global_logger = logging.getLogger("rl_agent")

        # Try to get the reward value
        reward_value = self._get_current_reward_metrics()

        # --- Only report if reward_value is available ---
        if reward_value is None:
            # Log that we are skipping the report because the key metric is missing
            if self.num_timesteps % 1000 == 0: # Log periodically to avoid spam
                 global_logger.debug(f"Skipping Ray Tune report at step {self.num_timesteps}: rollout/ep_rew_mean not yet available.")
            return True # Continue training

        # --- Proceed with reporting if reward_value is valid ---
        # Check if we're at a new timestep to report
        if self.num_timesteps <= self.last_reported_timestep:
             return True # Already reported for this step

        # Prepare metrics dictionary
        metrics_to_report = {}
        metrics_to_report["timesteps"] = self.num_timesteps
        metrics_to_report["training_iteration"] = self.num_timesteps

        # Add the validated reward
        metrics_to_report["eval/mean_reward"] = float(reward_value)

        # Get variance metrics from logger if available, default to 0.0
        variance_value = 0.0
        if self.logger and hasattr(self.logger, "name_to_value"):
            if "train/explained_variance" in self.logger.name_to_value:
                 try:
                     variance_value = float(self.logger.name_to_value["train/explained_variance"])
                 except (ValueError, TypeError):
                     variance_value = 0.0 # Default on conversion error

        metrics_to_report["eval/explained_variance"] = float(variance_value)

        # Calculate combined score using the valid reward
        combined_score = self._normalize_and_combine_metrics(reward_value, variance_value)
        metrics_to_report["eval/combined_score"] = float(combined_score)

        # Add other useful metrics if available
        if self.logger and hasattr(self.logger, "name_to_value"):
            other_metrics = ["rollout/ep_len_mean", "time/fps"]
            for key in other_metrics:
                if key in self.logger.name_to_value:
                    value = self.logger.name_to_value[key]
                    # Ensure value is scalar (int or float) before reporting
                    if isinstance(value, (int, float, np.number)):
                        try:
                            metrics_to_report[key] = float(value)
                        except (ValueError, TypeError):
                            pass # Skip if cannot convert

        # Log the metrics being reported
        global_logger.debug(f"Reporting metrics: reward={reward_value:.4f}, variance={variance_value:.4f}, combined={combined_score:.4f}")

        # Report to Ray Tune
        try:
            if RAY_AVAILABLE:
                # Use session.report for Ray >= 2.0 (preferred)
                if hasattr(ray, "air") and hasattr(ray.air, "session") \
                   and ray.air.session.is_active():
                    ray.air.session.report(metrics_to_report)
                else:
                    # Fallback for older Ray or direct tune usage without AIR session
                    tune.report(**metrics_to_report)

                self.last_reported_timestep = self.num_timesteps  # Update only on successful report
                global_logger.debug(f"Reported metrics to Ray Tune at timestep {self.num_timesteps}")
            else:
                # Log warning if Ray is somehow unavailable here
                global_logger.warning("Ray is not available during tune.report call.")

        except Exception as e:
            global_logger.error(f"Error during tune.report: {e}", exc_info=True)
            # We'll continue training even if reporting fails

        return True  # Continue training


# --- Ray Tune Trainable Function --- #

def train_rl_agent_tune(config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable function for training an RL agent.
    
    Args:
        config: Dictionary of hyperparameters from Ray Tune
    """
    # Ensure Ray is available
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not found. Install with 'pip install ray[tune]'")
    
    # Create a copy of the config to avoid modifying the original
    train_config = config.copy()
    
    # Set up some defaults if not provided
    train_config.setdefault("model_type", "recurrentppo")
    train_config.setdefault("verbose", 1)
    
    # Generate a consistent name for this trial
    if "trial_id" in config:
        trial_id = config["trial_id"]
    else:
        trial_id = tune.get_trial_id()
    
    model_name = f"{train_config['model_type']}_{trial_id}"
    train_config["model_name"] = model_name
    
    # Set up directories
    log_dir = os.path.join(train_config.get("log_dir", "./logs"), model_name)
    checkpoint_dir = os.path.join(train_config.get("checkpoint_dir", "./checkpoints"), model_name)
    ensure_dir_exists(log_dir)
    ensure_dir_exists(checkpoint_dir)
    train_config["log_dir"] = log_dir
    train_config["checkpoint_dir"] = checkpoint_dir
    
    # Report initial metrics to ensure they exist
    # This provides a default value for combined_score before any training happens
    initial_metrics = {
        "timesteps": 0,
        "training_iteration": 0,
        "eval/mean_reward": 0.0,
        "eval/explained_variance": 0.0,
        "eval/combined_score": 0.5,  # Default middle value
        "early_training": True       # Flag to indicate this is pre-training data
    }
    
    if RAY_AVAILABLE:
        logger.info("Reporting initial metrics to Ray Tune")
        if hasattr(ray, "air") and hasattr(ray.air, "session") and ray.air.session.is_active():
            ray.air.session.report(initial_metrics)
        else:
            tune.report(**initial_metrics)
    
    # Setup logger
    setup_logger(
        log_dir=log_dir,
        log_level=logging.DEBUG if train_config.get("verbose", 1) >= 2 else logging.INFO
    )
    
    # Setup seed - important for reproducible results across trials
    seed = train_config.get("seed")
    if seed is None:
        # If no seed provided, derive one from trial_id to ensure consistency
        seed = abs(hash(trial_id)) % (2**32)
        train_config["seed"] = seed
    
    set_seeds(seed)
    
    # Log the configuration
    logger.info(f"Starting Ray Tune trial with ID: {trial_id}")
    logger.info("Configuration:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")
    
    # Setup SB3 logger
    sb3_logger_instance = setup_sb3_logger(log_dir=log_dir)
    
    # --- Calculate dependent parameters if needed --- #
    
    # Ensure n_steps and batch_size are set appropriately
    # batch_size should typically be n_steps * num_envs or a divisor of that
    num_envs = train_config.get("num_envs", 8)
    n_steps = train_config.get("n_steps", 2048)
    
    # Ensure batch_size is a divisor of n_steps * num_envs
    total_steps = n_steps * num_envs
    if "batch_size" not in train_config:
        train_config["batch_size"] = min(total_steps, 2048)  # Default that works well
        logger.info(f"Setting batch_size to {train_config['batch_size']} based on n_steps={n_steps} and num_envs={num_envs}")
    
    # --- Environment Creation --- #
    logger.info(f"Creating {num_envs} parallel environment(s)...")
    
    # Function to create a single env instance
    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            # Setup logger inside subprocess
            process_log_path = os.path.join(log_dir)
            setup_logger(
                log_dir=process_log_path,
                log_level=logging.DEBUG if train_config.get("verbose", 1) >= 2 else logging.INFO,
                console_level=logging.INFO,  # Avoid console spam
                log_filename=f"rl_agent_{rank}.log"  # Use different log files per process
            )
            
            env_config = train_config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=False)
            
            # Use a different Monitor file per process
            monitor_log_path = os.path.join(log_dir, f'monitor_{rank}.csv')
            os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
            env = Monitor(env, filename=monitor_log_path)
            return env
        return _init
    
    # Create vectorized environment
    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=seed),
        n_envs=num_envs,
        seed=None,  # Seeding handled in make_single_env
        vec_env_cls=vec_env_cls,
        env_kwargs=None
    )
    
    # Apply VecNormalize
    norm_obs_setting = train_config.get("norm_obs", "auto").lower()
    if norm_obs_setting == "auto":
        # Auto-detect if features are already scaled
        features = train_config.get("features", [])
        if isinstance(features, str):
            features = features.split(",")
        
        # Check if features contain _scaled suffix
        has_scaled_features = any("_scaled" in feature for feature in features)
        should_norm_obs = not has_scaled_features
        logger.info(f"Auto-detected {'pre-scaled' if has_scaled_features else 'raw'} features. Setting norm_obs={should_norm_obs}")
    else:
        should_norm_obs = norm_obs_setting == "true"
        logger.info(f"Using explicit norm_obs={should_norm_obs} from config")
    
    train_env = VecNormalize(
        train_env,
        norm_obs=should_norm_obs,
        norm_reward=False,
        clip_obs=10.,
        gamma=train_config["gamma"]
    )
    
    # Create validation environment if specified
    eval_env = None
    if train_config.get("val_data_path"):
        logger.info(f"Creating validation env: {train_config['val_data_path']}")
        eval_env_config = train_config.copy()
        
        eval_env = make_vec_env(
            env_id=make_single_env(rank=0, base_seed=seed),
            n_envs=1,  # Use single env for validation
            seed=None,
            vec_env_cls=DummyVecEnv,
            env_kwargs=None
        )
        
        # Apply VecNormalize to eval_env
        eval_env = VecNormalize(
            eval_env,
            norm_obs=should_norm_obs,
            norm_reward=False,
            clip_obs=10.,
            gamma=train_config["gamma"],
            training=False
        )
    
    # --- Model Creation --- #
    logger.info(f"Creating new {train_config['model_type']} model")
    model = create_model(env=train_env, config=train_config)
    
    # Set logger for the model
    model.set_logger(sb3_logger_instance)
    
    # --- Callbacks --- #
    # Create a list of callbacks, including the Ray Tune reporter
    # --- Disable eval_env for now ---
    eval_env = None 
    # -------------------------------
    callbacks = get_callback_list(
        eval_env=eval_env, # Pass None to disable EvalCallback
        log_dir=log_dir,
        eval_freq=max(train_config.get("eval_freq", 10000), 5000),
        n_eval_episodes=train_config.get("n_eval_episodes", 5),
        save_freq=train_config.get("save_freq", 50000),
        keep_checkpoints=train_config.get("keep_checkpoints", 3),
        resource_check_freq=train_config.get("resource_check_freq", 5000),
        metrics_log_freq=train_config.get("metrics_log_freq", 1000),
        early_stopping_patience=max(20, train_config.get("early_stopping_patience", 10)),
        checkpoint_save_path=checkpoint_dir,
        model_name=train_config["model_type"],
        custom_callbacks=[TuneReportCallback()]
    )
    
    # --- Training --- #
    logger.info(f"Starting training: {train_config['total_timesteps']} timesteps...")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=train_config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=not train_config.get("continue_training", False)
        )
        training_time = time.time() - training_start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        # Report failure to Ray Tune
        if RAY_AVAILABLE:
            tune.report(training_iteration=model.num_timesteps, training_failure=True)
        raise
    finally:
        # Close environments
        if 'train_env' in locals() and hasattr(train_env, 'close'):
            train_env.close()
        if 'eval_env' in locals() and eval_env is not None and hasattr(eval_env, 'close'):
            eval_env.close()
    
    training_time = time.time() - training_start_time
    
    # --- Final Evaluation ---
    logger.info("Performing final evaluation...")
    # If we have an eval env, run a final evaluation
    final_eval_metrics = {}
    if eval_env is not None:
        try:
            mean_reward, _, _, _ = evaluate_model(
                model=model,
                env=eval_env,
                config=train_config,
                n_episodes=train_config.get("n_eval_episodes", 5),
                deterministic=True
            )
            final_eval_metrics["final_eval_mean_reward"] = mean_reward
            logger.info(f"Final evaluation mean reward: {mean_reward:.4f}")
        except Exception as e:
            logger.error(f"Error during final evaluation: {e}")
    
    # --- Final Report ---
    # Prepare metrics to report to Ray Tune
    metrics = {
        "training_time": training_time,
        "timesteps": model.num_timesteps,
        "training_iteration": model.num_timesteps,
        **final_eval_metrics
    }
    
    # Collect metrics from various sources to ensure we have the best values
    reward_value = None
    variance_value = None
    
    # Source 1: Check the model's logger (Primary source now)
    if hasattr(model, "logger") and hasattr(model.logger, "name_to_value"):
        if "rollout/ep_rew_mean" in model.logger.name_to_value:
            try:
                reward_value = float(model.logger.name_to_value["rollout/ep_rew_mean"])
                logger.info(f"Using final rewards from model logger: {reward_value:.4f}")
            except (ValueError, TypeError):
                 reward_value = None # Handle potential non-float values

        if "train/explained_variance" in model.logger.name_to_value:
             try:
                 variance_value = float(model.logger.name_to_value["train/explained_variance"])
                 logger.info(f"Using final explained variance from model logger: {variance_value:.4f}")
             except (ValueError, TypeError):
                 variance_value = None

    # Source 2: Check the environment directly (Fallback)
    if reward_value is None and hasattr(model, "env"):
        try:
            if hasattr(model.env, "get_episode_rewards"):
                rewards = model.env.get_episode_rewards()
                if rewards:
                    reward_value = float(np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards))
                    logger.info(f"Using final rewards from environment: {reward_value:.4f}")
        except (AttributeError, IndexError, ValueError, TypeError) as e:
            logger.warning(f"Could not get final rewards from environment: {e}")
            reward_value = None

    # Fallback values if needed
    if reward_value is None:
        # Use final eval reward if other sources failed
        reward_value = final_eval_metrics.get("final_eval_mean_reward", 0.0)
        logger.warning(f"No reward metric found from logger/env, using final eval: {reward_value:.4f}")

    if variance_value is None:
        variance_value = 0.0 # Default variance if not found
        logger.warning("No explained variance metric found, using 0.0")

    # Add the evaluation metrics (ensure they are floats)
    metrics["eval/mean_reward"] = float(reward_value)
    metrics["eval/explained_variance"] = float(variance_value)

    # Create a callback instance just to use the normalization method
    # Note: Need to instantiate it here as it's not passed directly
    temp_callback = TuneReportCallback()
    combined_score = temp_callback._normalize_and_combine_metrics(reward_value, variance_value)
    metrics["eval/combined_score"] = float(combined_score)

    logger.info(f"Final metrics - reward: {reward_value:.4f}, variance: {variance_value:.4f}, combined: {combined_score:.4f}")
    
    # Final report to Ray Tune
    if RAY_AVAILABLE:
        logger.info("Sending final report to Ray Tune")
        if hasattr(ray, "air") and hasattr(ray.air, "session") and ray.air.session.is_active():
            ray.air.session.report(metrics)
        else:
            tune.report(**metrics)
    
    return model, metrics

# --- Argument Parsing --- #

def parse_args():
    """Parse command line arguments, incorporating args from dqn_old.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate RL agents (DQN, PPO, A2C, SAC, LSTM-DQN, "
            "QRDQN, RecurrentPPO) for trading"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- General Parameters --- #
    general = parser.add_argument_group('General Parameters')
    general.add_argument(
        "--model_type", type=str, default="dqn",
        choices=[
            "dqn", "ppo", "a2c", "sac", "lstm_dqn", "qrdqn", "recurrentppo"
        ],
        help="RL algorithm to use"
    )
    general.add_argument(
        "--load_model", type=str, default=None,
        help="Path to saved model to continue training from"
    )
    general.add_argument(
        "--load_config", type=str, default=None,
        help="Path to config file (overrides defaults, overridden by CLI)"
    )
    general.add_argument(
        "--model_name", type=str, default=None,
        help="Name for model/log folder (auto-generated if None)"
    )
    general.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (None = random)"
    )
    general.add_argument(
        "--verbose", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level: 0=no output, 1=info, 2=debug"
    )
    general.add_argument(
        "--cpu_only", action="store_true",
        help="Force using CPU even if GPU is available"
    )
    general.add_argument(
        "--eval_only", action="store_true",
        help="Only evaluate a trained model (--load_model required)"
    )

    # --- Data Parameters --- #
    data = parser.add_argument_group('Data Parameters')
    data.add_argument(
        "--data_path", type=str, required=True,
        help="Path to training data file (CSV or HDF5)"
    )
    data.add_argument(
        "--val_data_path", type=str, default=None,
        help="Path to validation data file (optional)"
    )
    data.add_argument(
        "--test_data_path", type=str, default=None,
        help="Path to test data file (optional)"
    )
    data.add_argument(
        "--data_key", type=str, default=None,
        help="Key for HDF5 file (e.g., '/15m' for 15-minute data)"
    )
    data.add_argument(
        "--symbol", type=str, default="BTC/USDT",
        help="Trading symbol"
    )
    data.add_argument(
        "--sequence_length", type=int, default=60,
        help="Length of history sequence for features/LSTM (range: 30-120)"
    )
    data.add_argument(
        "--features", type=str,
        default="close,volume,open,high,low,rsi_14,macd,ema_9,ema_21",
        help="Comma-separated list of features/indicators to use"
    )

    # --- Environment Parameters --- #
    env = parser.add_argument_group('Environment Parameters')
    env.add_argument(
        "--initial_balance", type=float, default=10000,
        help="Initial balance for trading environment"
    )
    env.add_argument(
        "--max_position", type=float, default=1.0,
        help="Maximum position size as fraction of balance (range: 0.0-1.0)"
    )
    env.add_argument(
        "--commission", type=float, default=0.001,
        help="Trading commission percentage (0.001 = 0.1%)"
    )
    env.add_argument(
        "--max_steps", type=int, default=100000,
        help="Maximum steps per episode"
    )
    env.add_argument(
        "--episode_length", type=int, default=None,
        help="Length of each episode in days (overrides max_steps if set)"
    )
    env.add_argument(
        "--reward_scaling", type=float, default=1.0,
        help="Scaling factor for environment rewards (range: 0.1-10.0)"
    )
    env.add_argument(
        "--max_holding_steps", type=int, default=8,
        help="Max steps to hold before potential forced action"
    )
    env.add_argument(
        "--take_profit_pct", type=float, default=0.03,
        help="Take profit percentage (range: 0.01-0.05)"
    )
    env.add_argument(
        "--target_cash_ratio", type=str, default="0.3-0.7",
        help="Target cash ratio range for reward shaping (format: 'min-max')"
    )
    
    # --- Reward Component Weights --- #
    rewards = parser.add_argument_group('Reward Parameters')
    rewards.add_argument(
        "--portfolio_change_weight", type=float, default=1.0,
        help="Weight for portfolio value change reward (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--drawdown_penalty_weight", type=float, default=0.5,
        help="Weight for drawdown penalty (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--sharpe_reward_weight", type=float, default=0.5,
        help="Weight for Sharpe ratio reward (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--fee_penalty_weight", type=float, default=2.0,
        help="Weight for transaction fee penalty (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--benchmark_reward_weight", type=float, default=0.5,
        help="Weight for benchmark comparison reward (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--consistency_penalty_weight", type=float, default=0.2,
        help="Weight for trade consistency penalty (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--idle_penalty_weight", type=float, default=0.1,
        help="Weight for idle position penalty (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--profit_bonus_weight", type=float, default=0.5,
        help="Weight for profit bonus (range: 0.0-5.0)"
    )
    rewards.add_argument(
        "--exploration_bonus_weight", type=float, default=0.1,
        help="Weight for exploration bonus (range: 0.0-1.0)"
    )
    rewards.add_argument(
        "--trade_penalty_weight", type=float, default=0.0,
        help="Weight for direct penalty per trade (range: 0.0-10.0)"
    )
    
    # --- Additional Reward Parameters --- #
    rewards.add_argument(
        "--sharpe_window", type=int, default=20,
        help="Window size for Sharpe ratio calculation (range: 10-50)"
    )
    rewards.add_argument(
        "--consistency_threshold", type=int, default=3,
        help="Min consecutive actions before flip is acceptable (range: 2-10)"
    )
    rewards.add_argument(
        "--idle_threshold", type=int, default=5,
        help="Consecutive holds before idle penalty (range: 3-10)"
    )

    # --- Common Training Parameters --- #
    training = parser.add_argument_group('Common Training Parameters')
    training.add_argument(
        "--total_timesteps", type=int, default=1000000,
        help="Total number of training timesteps (range: 100k-10M)"
    )
    training.add_argument(
        "--learning_rate", type=float, default=0.0003,
        help="Learning rate for optimizer (range: 1e-5 to 1e-2)"
    )
    training.add_argument(
        "--batch_size", type=int, default=2048,
        help="Batch size (DQN: 32-128, PPO/A2C/SAC: 256-2048)"
    )
    training.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor for future rewards (range: 0.9-0.999)"
    )
    training.add_argument(
        "--eval_freq", type=int, default=10000,
        help="Evaluation frequency during training (steps)"
    )
    training.add_argument(
        "--n_eval_episodes", type=int, default=5,
        help="Number of episodes for evaluation (range: 1-20)"
    )
    training.add_argument(
        "--save_freq", type=int, default=50000,
        help="Model saving frequency during training (steps)"
    )
    training.add_argument(
        "--keep_checkpoints", type=int, default=3,
        help="Number of model checkpoints to keep"
    )
    training.add_argument(
        "--early_stopping_patience", type=int, default=10,
        help="Patience for early stopping (0=disable)"
    )
    training.add_argument(
        "--num_envs", type=int, default=1,
        help="Number of parallel environments for training"
    )

    # --- DQN/QRDQN Specific Parameters --- #
    dqn = parser.add_argument_group('DQN/QRDQN Specific Parameters')
    dqn.add_argument(
        "--buffer_size", type=int, default=100000,
        help="Replay buffer size (range: 10k-1M)"
    )
    dqn.add_argument(
        "--exploration_fraction", type=float, default=0.1,
        help="Exploration decay fraction (range: 0.05-0.5)"
    )
    dqn.add_argument(
        "--exploration_initial_eps", type=float, default=1.0,
        help="Initial exploration rate epsilon (range: 0.5-1.0)"
    )
    dqn.add_argument(
        "--exploration_final_eps", type=float, default=0.05,
        help="Final exploration rate epsilon (range: 0.01-0.1)"
    )
    dqn.add_argument(
        "--target_update_interval", type=int, default=10000,
        help="Target network update frequency (range: 1k-20k)"
    )
    dqn.add_argument(
        "--gradient_steps", type=int, default=1,
        help="Gradient steps per environment step (range: 1-10)"
    )
    dqn.add_argument(
        "--learning_starts", type=int, default=1000,
        help="Steps before learning starts (range: 100-10k)"
    )
    # Environment exploration bonus parameters
    dqn.add_argument(
        "--exploration_start", type=float, default=1.0,
        help="Starting value for exploration bonus (range: 0.5-1.0)"
    )
    dqn.add_argument(
        "--exploration_end", type=float, default=0.01,
        help="Final value for exploration bonus (range: 0.01-0.1)"
    )
    dqn.add_argument(
        "--exploration_decay_rate", type=float, default=0.0001,
        help="Decay rate for exploration bonus per step (range: 0.00001-0.001)"
    )
    # QRDQN specific
    dqn.add_argument(
        "--n_quantiles", type=int, default=200,
        help="Number of quantiles for QRDQN (range: 50-200)"
    )

    # --- PPO/A2C Specific Parameters --- #
    ppo = parser.add_argument_group('PPO/A2C Specific Parameters')
    ppo.add_argument(
        "--n_steps", type=int, default=2048,
        help="Steps per update for PPO/A2C (range: 128-2048)"
    )
    ppo.add_argument(
        "--ent_coef", type=str, default="0.01",
        help="Entropy coefficient ('auto' for SAC, range: 0.0-0.1)"
    )
    ppo.add_argument(
        "--vf_coef", type=float, default=0.5,
        help="Value function coefficient for PPO/A2C (range: 0.1-1.0)"
    )
    ppo.add_argument(
        "--n_epochs", type=int, default=10,
        help="Number of epochs per update for PPO (range: 3-20)"
    )
    ppo.add_argument(
        "--clip_range", type=float, default=0.2,
        help="PPO clip range (range: 0.1-0.3)"
    )
    ppo.add_argument(
        "--gae_lambda", type=float, default=0.95,
        help="Lambda parameter for GAE in PPO/A2C (range: 0.9-0.99)"
    )
    ppo.add_argument(
        "--max_grad_norm", type=float, default=0.5,
        help="Maximum norm for gradient clipping (range: 0.1-1.0)"
    )

    # --- SAC Specific Parameters --- #
    sac = parser.add_argument_group('SAC Specific Parameters')
    sac.add_argument(
        "--tau", type=float, default=0.005,
        help="Soft update coefficient tau (range: 0.001-0.01)"
    )

    # --- LSTM/Network Architecture --- #
    network = parser.add_argument_group('Network Architecture')
    network.add_argument(
        "--lstm_model_path", type=str, default=None,
        help="Path to saved LSTM model state_dict for feature extraction"
    )
    network.add_argument(
        "--lstm_hidden_size", type=int, default=128,
        help="Hidden size of LSTM layer (range: 32-512)"
    )
    network.add_argument(
        "--n_lstm_layers", type=int, default=1,
        help="Number of LSTM layers for RecurrentPPO (range: 1-3)"
    )
    network.add_argument(
        "--shared_lstm", type=str, default="shared",
        choices=["shared", "seperate", "none"],
        help="LSTM mode for RecurrentPPO (shared, seperate, or none)"
    )
    network.add_argument(
        "--fc_hidden_size", type=int, default=64,
        help="Hidden size of FC layers after LSTM/features (range: 32-256)"
    )

    # --- Resource Management & Logging --- #
    logging_group = parser.add_argument_group(
        'Logging and Resource Management'
    )
    logging_group.add_argument(
        "--resource_check_freq", type=int, default=5000,
        help="Frequency of resource usage checks (steps)"
    )
    logging_group.add_argument(
        "--metrics_log_freq", type=int, default=1000,
        help="Frequency of trading metrics logging (steps)"
    )
    logging_group.add_argument(
        "--log_dir", type=str, default="./logs",
        help="Directory for saving logs"
    )
    logging_group.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints",
        help="Directory for saving model checkpoints"
    )
    logging_group.add_argument(
        "--norm_obs", type=str, default="auto",
        choices=["auto", "true", "false"],
        help="Control observation normalization in VecNormalize. 'auto' uses True for raw features, False for _scaled features."
    )

    args = parser.parse_args()

    # Auto-generate model name if not provided
    if args.model_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.model_name = f"{args.model_type}_{timestamp}"

    # Convert features string into list
    if isinstance(args.features, str):
        args.features = args.features.split(",")

    return args

# --- Helper Functions --- #

def args_to_config(args) -> Dict[str, Any]:
    """Convert argparse arguments to config dictionary."""
    return vars(args)


# Update create_env to use config dict and potentially handle HDF5
def create_env(
    config: Dict[str, Any],  # Use config dict
    data_override: Optional[pd.DataFrame] = None,
    is_eval: bool = False
) -> gym.Env:
    """
    Create a trading environment based on configuration.

    Args:
        config: Configuration dictionary containing env parameters.
        data_override: Optional DataFrame to use instead of loading from path.
        is_eval: Flag indicating if this is for evaluation (affects seeding).

    Returns:
        Trading environment instance.
    """
    # Load data if not provided
    if data_override is not None:
        data = data_override
    else:
        data_path = config["data_path"]
        data_key = config.get("data_key")  # Use .get for optional key
        data_loader = DataLoader(data_path=data_path, data_key=data_key)
        data = data_loader.load_data()

    # Note: Seeding is handled by make_vec_env wrapper now

    # Create environment instance
    # Start with core arguments
    env_kwargs = {
        "data": data,
        "features": config.get("features"),
        "sequence_length": config["sequence_length"],
        "initial_balance": config["initial_balance"],
        "transaction_fee": config["commission"],
        "reward_scaling": config["reward_scaling"],
        # Optional args from TradingEnvironment (using defaults or config)
        "window_size": config.get("window_size", 20),
        "max_position": config.get("max_position", 1.0),
        "max_steps": config.get("max_steps"),
        "random_start": config.get("random_start", True),
    }

    # --- Correct the features format if it's a string ---
    if isinstance(env_kwargs["features"], str):
        logger.debug(f"Splitting features string: {env_kwargs['features']}")
        env_kwargs["features"] = [f.strip() for f in env_kwargs["features"].split(',') if f.strip()]
        logger.debug(f"Converted features to list: {env_kwargs['features']}")
    elif env_kwargs["features"] is None:
        # Handle case where features might not be provided, although TradingEnv requires it
        logger.warning("Features not provided in config, TradingEnvironment might fail.")
        # You might want to raise an error here or provide a default list
        env_kwargs["features"] = [] # Default to empty list or handle appropriately

    # --- Conditionally add new reward parameters ---
    # Define the keys for the new reward parameters
    reward_param_keys = [
        "portfolio_change_weight", "drawdown_penalty_weight",
        "sharpe_reward_weight", "fee_penalty_weight",
        "benchmark_reward_weight", "consistency_penalty_weight",
        "idle_penalty_weight", "profit_bonus_weight",
        "exploration_bonus_weight", "sharpe_window",
        "consistency_threshold", "idle_threshold",
        "exploration_start", "exploration_end",
        "exploration_decay_rate", "trade_penalty_weight"
    ]

    # Add them to env_kwargs ONLY if they exist in the config dict
    for key in reward_param_keys:
        if key in config:
            env_kwargs[key] = config[key]
            logger.debug(
                f"Passing reward param '{key}' = {config[key]} to env."
            )
        # else: # Optional: Log if a parameter is *not* found in config
        #     logger.debug(f"Reward param '{key}' not found in config,"
        #                  " using env default.")

    env = TradingEnvironment(**env_kwargs)

    # Apply wrappers if needed
    # env = TimeLimit(env, max_episode_steps=config["max_steps"])

    return env


# Update create_model to handle more algorithms and LSTM features
def create_model(
    env: gym.Env,  # Can be a VecEnv
    config: Dict[str, Any],
) -> BaseRLModel:
    """
    Create a reinforcement learning model based on configuration.

    Args:
        env: Training environment (potentially vectorized).
        config: Configuration dictionary.

    Returns:
        Reinforcement learning model instance.
    """
    model_type = config["model_type"].lower()
    learning_rate = config["learning_rate"]
    seed = config.get("seed")

    # Determine device setting
    device = "cpu" if config["cpu_only"] else "auto"

    policy_kwargs = {}
    model_kwargs = {
        "policy": None,  # Determined below
        "env": env,
        "learning_rate": learning_rate,
        "gamma": config["gamma"],
        "seed": seed,
        "device": device,
        "verbose": 0,  # Callbacks handle progress output
        "policy_kwargs": policy_kwargs,  # Updated below
        "tensorboard_log": os.path.join(
            config["log_dir"], config["model_name"], "sb3_logs"
        )
    }

    # --- LSTM Feature Extractor Setup --- #
    use_lstm_features = (
        model_type == "lstm_dqn" or config.get("lstm_model_path")
    )
    if use_lstm_features:
        lstm_state_dict = None
        if config.get("lstm_model_path"):
            lstm_state_dict = config["lstm_model_path"]
            logger.info(f"Using LSTM model path: {lstm_state_dict}")

        policy_kwargs["features_extractor_class"] = LSTMFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "lstm_state_dict": lstm_state_dict,
            "features_dim": config.get("lstm_hidden_size", 128)
        }

        # Adjust network architecture if FC layers are specified
        if "fc_hidden_size" in config and config["fc_hidden_size"] > 0:
            fc_size = config["fc_hidden_size"]
            if model_type in ["ppo", "a2c"]:
                # SB3 uses net_arch=[dict(pi=[...], vf=[...])] or shared
                policy_kwargs["net_arch"] = [fc_size, fc_size]  # Shared
            # DQN/SAC net_arch handled below in specific sections

    # --- Algorithm Specific Setup --- #
    if model_type == "dqn" or model_type == "lstm_dqn":
        model_kwargs["policy"] = DqnMlpPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config["learning_starts"]
        model_kwargs["gradient_steps"] = config["gradient_steps"]
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["exploration_fraction"] = config["exploration_fraction"]
        model_kwargs["exploration_initial_eps"] = config["exploration_initial_eps"]
        model_kwargs["exploration_final_eps"] = config["exploration_final_eps"]

        # Set default net_arch for DqnMlpPolicy if not overridden
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model_kwargs["policy_kwargs"] = policy_kwargs

        # Use default ReplayBuffer (PER commented out)
        model_kwargs["replay_buffer_class"] = ReplayBuffer
        model_kwargs["replay_buffer_kwargs"] = {}

        # Remove kwargs not accepted by DQN.__init__
        model_kwargs.pop("tensorboard_log", None)  # Uses learn's tb_log_name

        model = DQN(**model_kwargs)

    elif model_type == "ppo":
        model_kwargs["policy"] = ActorCriticPolicy
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["n_epochs"] = config["n_epochs"]
        model_kwargs["ent_coef"] = float(config["ent_coef"])
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["clip_range"] = config["clip_range"]
        model_kwargs["gae_lambda"] = config["gae_lambda"]
        model_kwargs["max_grad_norm"] = config["max_grad_norm"]
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model = PPO(**model_kwargs)

    elif model_type == "a2c":
        model_kwargs["policy"] = ActorCriticPolicy
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["ent_coef"] = float(config["ent_coef"])
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["gae_lambda"] = config["gae_lambda"]
        model_kwargs["max_grad_norm"] = config["max_grad_norm"]
        model_kwargs["rms_prop_eps"] = config.get("rms_prop_eps", 1e-5)
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model = A2C(**model_kwargs)

    elif model_type == "sac":
        model_kwargs["policy"] = SacMlpPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config["learning_starts"]
        model_kwargs["gradient_steps"] = config["gradient_steps"]
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["tau"] = config["tau"]

        # Handle ent_coef ('auto' or float)
        ent_coef_value = config.get("ent_coef", "auto")
        if isinstance(ent_coef_value, str) \
                and ent_coef_value.lower() == 'auto':
            model_kwargs["ent_coef"] = 'auto'
        else:
            try:
                model_kwargs["ent_coef"] = float(ent_coef_value)
            except ValueError:
                logger.warning(
                    f"Invalid ent_coef '{ent_coef_value}'. Defaulting 'auto'."
                )
                model_kwargs["ent_coef"] = 'auto'

        # Set default net_arch for SacMlpPolicy if not overridden
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        model = SAC(**model_kwargs)

    elif model_type == "qrdqn":
        # Import QRDQNPolicy from sb3_contrib
        from sb3_contrib.qrdqn.policies import QRDQNPolicy
        model_kwargs["policy"] = QRDQNPolicy  # Use QRDQNPolicy
        model_kwargs["buffer_size"] = config["buffer_size"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["learning_starts"] = config["learning_starts"]
        model_kwargs["gradient_steps"] = config["gradient_steps"]
        model_kwargs["target_update_interval"] = config["target_update_interval"]
        model_kwargs["exploration_fraction"] = config["exploration_fraction"]
        model_kwargs["exploration_initial_eps"] = config["exploration_initial_eps"]
        model_kwargs["exploration_final_eps"] = config["exploration_final_eps"]

        # Set default net_arch for QRDQNPolicy
        if "net_arch" not in policy_kwargs:
            fc_size = config.get("fc_hidden_size", 64)
            policy_kwargs["net_arch"] = [fc_size] * 2
        
        # QRDQN specific args - set n_quantiles in policy_kwargs
        policy_kwargs["n_quantiles"] = config.get("n_quantiles", 200)
        model_kwargs["policy_kwargs"] = policy_kwargs

        # Remove incompatible args
        model_kwargs.pop("tensorboard_log", None)
        model = QRDQN(**model_kwargs)

    elif model_type == "recurrentppo":
        # Use the string identifier for the policy
        model_kwargs["policy"] = "MlpLstmPolicy"
        model_kwargs["n_steps"] = config["n_steps"]
        model_kwargs["batch_size"] = config["batch_size"]
        model_kwargs["n_epochs"] = config["n_epochs"]
        model_kwargs["ent_coef"] = float(config["ent_coef"])
        model_kwargs["vf_coef"] = config["vf_coef"]
        model_kwargs["clip_range"] = config["clip_range"]
        model_kwargs["gae_lambda"] = config["gae_lambda"]
        model_kwargs["max_grad_norm"] = config["max_grad_norm"]

        # Configure LSTM within the policy_kwargs
        if "lstm_hidden_size" in config:
            policy_kwargs["lstm_hidden_size"] = config["lstm_hidden_size"]
        if "n_lstm_layers" in config:
            policy_kwargs["n_lstm_layers"] = config["n_lstm_layers"]
        if "shared_lstm" in config:
            # Convert from string param to the proper format
            shared_lstm_mode = config["shared_lstm"]
            # For RecurrentPPO, shared_lstm must be one of:
            # 'shared', 'seperate', or 'none'
            valid_modes = ["shared", "seperate", "none"]
            if shared_lstm_mode not in valid_modes:
                logger.warning(
                    f"Invalid shared_lstm '{shared_lstm_mode}', "
                    f"defaulting to 'shared'"
                )
                shared_lstm_mode = "shared"
            # --- Correctly set boolean flags ---
            if shared_lstm_mode == "shared":
                policy_kwargs["shared_lstm"] = True
                policy_kwargs["enable_critic_lstm"] = False
            elif shared_lstm_mode == "seperate":
                policy_kwargs["shared_lstm"] = False
                policy_kwargs["enable_critic_lstm"] = True
            else: # shared_lstm_mode == "none"
                policy_kwargs["shared_lstm"] = False
                policy_kwargs["enable_critic_lstm"] = False
            # ----------------------------------
        
        # Set policy_kwargs in model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs
        
        logger.info(
            f"RecurrentPPO LSTM: "
            f"hidden={policy_kwargs.get('lstm_hidden_size', 128)}, "
            f"layers={policy_kwargs.get('n_lstm_layers', 1)}, "
            f"shared={policy_kwargs.get('shared_lstm', 'shared')}"
        )

        model = RecurrentPPO(**model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    policy_name = (model_kwargs['policy'] if isinstance(model_kwargs['policy'], str)
                   else model_kwargs['policy'].__name__)
    logger.info(f"Created {model_type.upper()} model with policy {policy_name}")
    return model

# --- Evaluation --- #

# Update evaluate_model to handle vectorized environments
def evaluate_model(
    model: BaseRLModel,
    env: gym.Env,  # VecEnv expected
    config: Dict[str, Any],
    n_episodes: int = 1,
    deterministic: bool = True,
) -> Tuple[float, np.ndarray, List[int], List[float]]:
    """
    Evaluate a model over n_episodes and return metrics.

    Args:
        model: The trained model
        env: The vectorized environment to evaluate on
        config: Configuration dictionary
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions

    Returns:
        Tuple containing:
        - mean reward
        - portfolio values (flattened from all episodes)
        - actions taken (flattened)
        - rewards received (flattened)
    """
    is_vectorized = hasattr(env, 'num_envs')
    if not is_vectorized:
        # This function expects a VecEnv
        logger.warning("evaluate_model expects VecEnv, wrapping in DummyVecEnv")
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs

    # Tracking variables
    all_episode_rewards = []
    all_portfolio_values = []
    all_actions = []
    all_rewards = []

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_portfolio_values = [[] for _ in range(n_envs)]
    current_actions = [[] for _ in range(n_envs)]
    current_rewards_list = [[] for _ in range(n_envs)]

    # Reset environment
    obs = env.reset()
    episodes_completed = 0

    # Main evaluation loop
    while episodes_completed < n_episodes:
        # Get model prediction
        action, _ = model.predict(obs, deterministic=deterministic)

        # Step environment
        obs, rewards, terminated, infos = env.step(action)
        # In new gym API, 'done' is split into 'terminated' and 'truncated'
        # SB3 VecEnv 'infos' usually contains 'TimeLimit.truncated'
        truncated = np.array([info.get('TimeLimit.truncated', False)
                              for info in infos])
        dones = np.logical_or(terminated, truncated)

        # Update tracking for each environment
        for i in range(n_envs):
            current_rewards[i] += rewards[i]
            current_lengths[i] += 1

            portfolio_value = infos[i].get("portfolio_value", 0.0)
            current_portfolio_values[i].append(portfolio_value)
            current_actions[i].append(action[i])
            current_rewards_list[i].append(rewards[i])

            # Handle episode completion
            if dones[i]:
                if episodes_completed < n_episodes:
                    all_episode_rewards.append(current_rewards[i])
                    all_portfolio_values.extend(current_portfolio_values[i])
                    all_actions.extend(current_actions[i])
                    all_rewards.extend(current_rewards_list[i])
                    episodes_completed += 1

                # Reset tracking for this environment
                current_rewards[i] = 0
                current_lengths[i] = 0
                current_portfolio_values[i] = []
                current_actions[i] = []
                current_rewards_list[i] = []

                # Important: Check if env needs manual reset after done
                # VecEnv handles auto-reset, but Monitor might need manual?
                # Generally VecEnv handles this.

    # Calculate mean reward across all completed episodes
    mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0

    return (
        mean_reward,
        np.array(all_portfolio_values),
        all_actions,
        all_rewards
    )


# Implement the missing evaluate function
def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    # Setup logger
    log_path = os.path.join(config["log_dir"],
                            config["model_name"], "evaluation")
    ensure_dir_exists(log_path)
    setup_logger(
        log_dir=log_path,
        log_level=logging.DEBUG if config.get("verbose", 1) >= 2
        else logging.INFO
    )

    # Check if model path exists
    model_path = config["load_model"]
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)

    # Load test data
    if not config.get("test_data_path"):
        logger.error("No test data path provided")
        sys.exit(1)

    # Create test environment with same settings as training
    logger.info(f"Creating test environment from: {config['test_data_path']}")
    test_env_config = config.copy()
    test_env_config["data_path"] = config["test_data_path"]

    base_seed_from_config = config.get("seed")

    # Function to create a single env instance
    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            # --- ADD LOGGER SETUP INSIDE SUBPROCESS ---
            # Use the same log path and verbosity level from the main config
            # Important: Use a unique log filename per process OR ensure FileHandler is process-safe
            # For simplicity here, let's log to the same file (FileHandler is generally process-safe)
            process_log_path = os.path.join(config["log_dir"], config["model_name"])
            setup_logger(
                log_dir=process_log_path,
                log_level=logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO,
                # Keep console level INFO to avoid spamming main console from subprocesses
                console_level=logging.INFO, 
                log_filename="rl_agent.log" # Log to the same main file
            )
            # --- END LOGGER SETUP ---

            env_config = config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=False)
            # Use a different Monitor file per process to avoid conflicts
            monitor_log_path = os.path.join(log_path, f'monitor_{rank}.csv') 
            os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
            env = Monitor(env, filename=monitor_log_path)
            return env
        return _init

    # Create VecEnv for evaluation (can use DummyVecEnv for simplicity)
    test_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=base_seed_from_config),
        n_envs=1,  # Evaluate on a single env instance for clearer metrics
        seed=None,  # Seeding handled in make_single_env
        vec_env_cls=DummyVecEnv,  # Use DummyVecEnv for eval
        env_kwargs=None
    )

    # --- Apply VecNormalize to test_env --- #
    vec_normalize_stats_path = None
    potential_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(potential_stats_path):
        vec_normalize_stats_path = potential_stats_path

    if vec_normalize_stats_path:
        logger.info(f"Loading VecNormalize stats: {vec_normalize_stats_path}")
        test_env = VecNormalize.load(vec_normalize_stats_path, test_env)
        test_env.training = False  # Set to inference mode
        test_env.norm_reward = False  # Do not normalize rewards for eval
    else:
        logger.warning("VecNormalize stats not found. Creating fresh wrapper for evaluation.")
        
        # Determine norm_obs setting based on config - same logic as train function
        norm_obs_setting = config.get("norm_obs", "auto").lower()
        
        if norm_obs_setting == "auto":
            # Auto-detect if features are already scaled
            features = config.get("features", [])
            if isinstance(features, str):
                features = features.split(",")
            
            # Check if features contain _scaled suffix
            has_scaled_features = any("_scaled" in feature for feature in features)
            should_norm_obs = not has_scaled_features
            logger.info(f"Auto-detected {'pre-scaled' if has_scaled_features else 'raw'} features. Setting norm_obs={should_norm_obs}")
        else:
            should_norm_obs = norm_obs_setting == "true"
            logger.info(f"Using explicit norm_obs={should_norm_obs} from config")
            
        # Apply VecNormalize with the determined settings
        test_env = VecNormalize(
            test_env,
            norm_obs=should_norm_obs,
            norm_reward=False,
            clip_obs=10.,
            gamma=config["gamma"],
            training=False
        )

    # Load the model
    model_cls = {
        "dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
        "lstm_dqn": DQN, "qrdqn": QRDQN, "recurrentppo": RecurrentPPO
    }
    model = model_cls[config["model_type"]].load(model_path, env=test_env)

    # Run evaluation
    n_eval_episodes = config.get("n_eval_episodes", 5)
    logger.info(f"Starting evaluation for {n_eval_episodes} episodes")

    mean_reward, portfolio_values, actions, rewards = evaluate_model(
        model=model,
        env=test_env,
        config=config,
        n_episodes=n_eval_episodes,
        deterministic=True
    )

    # Calculate additional metrics
    if len(portfolio_values) > 0:
        final_portfolio_value = portfolio_values[-1]
        initial_portfolio_value = config.get("initial_balance", 10000)
        # Use initial balance from config as start for total return calc
        total_return = (
            (final_portfolio_value / initial_portfolio_value) - 1
            if initial_portfolio_value > 0 else 0
        )
    else:
        final_portfolio_value = 0.0
        initial_portfolio_value = config.get("initial_balance", 10000)
        total_return = 0.0
        logger.warning("No portfolio values recorded during evaluation!")

    metrics = {
        "mean_reward": mean_reward,
        "final_portfolio_value": final_portfolio_value,
        "total_return": total_return,
    }

    if len(portfolio_values) > 10:
        try:
            trading_metrics = calculate_trading_metrics(portfolio_values)
            metrics.update(trading_metrics)
            create_evaluation_plots(
                portfolio_values=portfolio_values,
                actions=actions,
                rewards=rewards,
                save_path=log_path
            )
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {e}")
    else:
        logger.warning(
            f"Only {len(portfolio_values)} values; skipping advanced metrics."
        )

    # Log metrics
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        log_str = f"  {key}: {value:.4f}" if isinstance(value, float) \
            else f"  {key}: {value}"
        logger.info(log_str)

    # Save metrics to JSON file
    metrics_file = os.path.join(log_path, "evaluation_metrics.json")
    try:
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert numpy types for JSON serialization
            serializable_metrics = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in metrics.items()
            }
            json.dump(serializable_metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_file}")
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")

    # Close environment
    test_env.close()

    return metrics

# --- Training --- #

# Update train function
def train(config: Dict[str, Any]) -> Tuple[BaseRLModel, Dict[str, Any]]:
    """
    Train a reinforcement learning agent based on config.
    """
    # Setup logger
    log_path = os.path.join(config["log_dir"], config["model_name"])
    ensure_dir_exists(log_path)
    setup_logger(
        log_dir=log_path,
        log_level=logging.DEBUG if config.get("verbose", 1) >= 2
        else logging.INFO
    )

    # Setup SB3 logger
    sb3_log_path = log_path
    sb3_logger_instance = setup_sb3_logger(log_dir=sb3_log_path)

    # Save configuration
    save_config(config=config, log_dir=log_path, filename="config.json")

    # Set random seeds
    if config.get("seed") is not None:
        set_seeds(config["seed"])

    # Log system information
    logger.info(f"Starting training with model: {config['model_name']}")
    device = ("cpu" if config["cpu_only"]
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training on device: {device}")
    check_resources(logger)

    # --- Environment Creation --- #
    num_envs = config.get("num_envs", 1)
    logger.info(f"Creating {num_envs} parallel environment(s)...")

    # Function to create a single env instance
    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            # --- ADD LOGGER SETUP INSIDE SUBPROCESS ---
            # Use the same log path and verbosity level from the main config
            # Important: Use a unique log filename per process OR ensure FileHandler is process-safe
            # For simplicity here, let's log to the same file (FileHandler is generally process-safe)
            process_log_path = os.path.join(config["log_dir"], config["model_name"])
            setup_logger(
                log_dir=process_log_path,
                log_level=logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO,
                # Keep console level INFO to avoid spamming main console from subprocesses
                console_level=logging.INFO, 
                log_filename="rl_agent.log" # Log to the same main file
            )
            # --- END LOGGER SETUP ---

            env_config = config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=False)
            # Use a different Monitor file per process to avoid conflicts
            monitor_log_path = os.path.join(log_path, f'monitor_{rank}.csv') 
            os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
            env = Monitor(env, filename=monitor_log_path)
            return env
        return _init

    # Create vectorized environment
    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    base_seed_from_config = config.get("seed")
    train_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=base_seed_from_config),
        n_envs=num_envs,
        seed=None,  # Seeding handled in make_single_env
        vec_env_cls=vec_env_cls,
        env_kwargs=None
    )

    # --- Apply VecNormalize --- #
    vec_normalize_stats_path = None
    if config.get("load_model"):
        potential_stats_path = config["load_model"].replace(
            ".zip", "_vecnormalize.pkl"
        )
        if os.path.exists(potential_stats_path):
            vec_normalize_stats_path = potential_stats_path
            logger.info(f"Found VecNorm stats: {vec_normalize_stats_path}")

    if vec_normalize_stats_path:
        logger.info(f"Loading VecNorm stats: {vec_normalize_stats_path}")
        train_env = VecNormalize.load(vec_normalize_stats_path, train_env)
        train_env.training = True  # Ensure it continues training
    else:
        # Determine norm_obs setting based on config
        norm_obs_setting = config.get("norm_obs", "auto").lower()
        
        if norm_obs_setting == "auto":
            # Auto-detect if features are already scaled
            features = config.get("features", [])
            if isinstance(features, str):
                features = features.split(",")
            
            # Check if features contain _scaled suffix
            has_scaled_features = any("_scaled" in feature for feature in features)
            should_norm_obs = not has_scaled_features
            logger.info(f"Auto-detected {'pre-scaled' if has_scaled_features else 'raw'} features. Setting norm_obs={should_norm_obs}")
        else:
            should_norm_obs = norm_obs_setting == "true"
            logger.info(f"Using explicit norm_obs={should_norm_obs} from config")
        
        logger.info(f"Applying VecNorm (norm_obs={should_norm_obs}, norm_reward=False).")
        train_env = VecNormalize(
            train_env,
            norm_obs=should_norm_obs,
            norm_reward=False,
            clip_obs=10.,
            gamma=config["gamma"]
        )

    # Create validation environment if specified
    eval_env = None
    if config.get("val_data_path"):
        logger.info(f"Creating validation env: {config['val_data_path']}")
        eval_env_config = config.copy()
        eval_env_config["data_path"] = config["val_data_path"]

        eval_env = make_vec_env(
            env_id=make_single_env(rank=0, base_seed=base_seed_from_config),
            n_envs=1,  # Use single env for validation
            seed=None,
            vec_env_cls=DummyVecEnv,  # Use DummyVecEnv for validation
            env_kwargs=None
        )

        # Apply VecNormalize to eval_env
        if vec_normalize_stats_path:
            logger.info("Applying loaded VecNorm stats to eval_env.")
            eval_env = VecNormalize.load(vec_normalize_stats_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            logger.info("Applying new VecNorm wrapper to eval_env.")
            
            # Use the same norm_obs setting as determined for training
            norm_obs_setting = config.get("norm_obs", "auto").lower()
            if norm_obs_setting == "auto":
                features = config.get("features", [])
                if isinstance(features, str):
                    features = features.split(",")
                has_scaled_features = any("_scaled" in feature for feature in features)
                should_norm_obs = not has_scaled_features
            else:
                should_norm_obs = norm_obs_setting == "true"
                
            logger.info(f"Using norm_obs={should_norm_obs} for validation environment")
            
            eval_env = VecNormalize(
                eval_env,
                norm_obs=should_norm_obs,
                norm_reward=False,
                clip_obs=10.,
                gamma=config["gamma"],
                training=False
            )

    # --- Model Creation / Loading --- #
    if config.get("load_model"):
        load_path = config["load_model"]
        logger.info(f"Loading model from: {load_path}")
        if not os.path.exists(load_path):
            logger.error(f"Model path not found: {load_path}")
            sys.exit(1)

        model_cls = {
            "dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
            "lstm_dqn": DQN, "qrdqn": QRDQN, "recurrentppo": RecurrentPPO
        }
        model = model_cls[config["model_type"]].load(load_path, env=train_env)

        # Override loaded learning rate if specified
        if 'learning_rate' in config and config['learning_rate'] is not None:
            new_lr = config['learning_rate']
            if hasattr(model.policy, 'optimizer') and model.policy.optimizer:
                logger.info(f"Overriding loaded LR. Setting to: {new_lr}")
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                logger.warning("Could not find optimizer to override LR.")

    else:
        logger.info(f"Creating new {config['model_type']} model")
        model = create_model(env=train_env, config=config)

    # Set logger for the model
    model.set_logger(sb3_logger_instance)

    # --- Callbacks --- #
    checkpoint_save_path = os.path.join(
        config["checkpoint_dir"], config["model_name"]
    )
    ensure_dir_exists(checkpoint_save_path)
    callbacks = get_callback_list(
        eval_env=eval_env,
        log_dir=log_path,
        eval_freq=max(config["eval_freq"], 5000),
        n_eval_episodes=config["n_eval_episodes"],
        save_freq=config["save_freq"],
        keep_checkpoints=config["keep_checkpoints"],
        resource_check_freq=config["resource_check_freq"],
        metrics_log_freq=config["metrics_log_freq"],
        early_stopping_patience=max(20, config["early_stopping_patience"]),
        checkpoint_save_path=checkpoint_save_path,
        model_name=config["model_type"],
        custom_callbacks=[]
    )

    # --- Training --- #
    logger.info(f"Starting training: {config['total_timesteps']} timesteps...")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=1,
            reset_num_timesteps=not config.get("load_model")
        )
    except Exception as e:
        logger.critical(f"Training failed: {e}", exc_info=True)
        error_save_path = os.path.join(log_path, "model_on_error.zip")
        try:
            model.save(error_save_path)
            logger.info(f"Model state saved to {error_save_path}.")
        except Exception as save_e:
            logger.error(f"Could not save model after error: {save_e}")
        sys.exit(1)
    finally:
        # Ensure train_env exists and has a close method before calling it
        if 'train_env' in locals() and hasattr(train_env, 'close'):
            train_env.close()
        # Ensure eval_env exists and has a close method before calling it
        if 'eval_env' in locals() and eval_env is not None and hasattr(eval_env, 'close'):
            eval_env.close()

    training_time = time.time() - training_start_time
    logger.info(f"Training finished in {training_time:.2f} seconds.")

    # Save final model
    final_model_path = os.path.join(log_path, "final_model.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save VecNormalize stats only if train_env is VecNormalize
    if 'train_env' in locals() and isinstance(train_env, VecNormalize):
        stats_path = final_model_path.replace(".zip", "_vecnormalize.pkl")
        train_env.save(stats_path)
        logger.info(f"VecNorm stats saved to {stats_path}")

    metrics = {
        "training_time": training_time,
        "total_timesteps": model.num_timesteps,
        "model_type": config["model_type"],
        "eval/mean_reward": model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0) if hasattr(model.logger, "name_to_value") else 0.0,
        "eval/explained_variance": model.logger.name_to_value.get("train/explained_variance", 0.0) if hasattr(model.logger, "name_to_value") else 0.0
    }
    
    # Add the evaluation metrics if available
    if hasattr(model.logger, "name_to_value"):
        reward_value = model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)
        variance_value = model.logger.name_to_value.get("train/explained_variance", 0.0)
        
        metrics["eval/mean_reward"] = float(reward_value)
        metrics["eval/explained_variance"] = float(variance_value)
        
        # Create a callback instance just to use the normalization method
        callback = TuneReportCallback()
        combined_score = callback._normalize_and_combine_metrics(reward_value, variance_value)
        metrics["eval/combined_score"] = float(combined_score)

    # Final report to Ray Tune (basic completion info)
    if RAY_AVAILABLE:
        # Pass the consolidated metrics dictionary with the correct metric names
        tune.report(**metrics)
    
    return model, metrics

# --- Main Execution --- #

# Updated main function to handle different models and eval mode
def main():
    """Main function: parse args, setup, run train/eval."""
    args = parse_args()
    config = args_to_config(args)

    # --- Config Loading --- #
    if args.load_config is not None:
        if os.path.exists(args.load_config):
            print(f"Loading configuration from {args.load_config}")
            file_config = load_config(args.load_config)
            cli_overrides = {
                k: v for k, v in vars(args).items() if v is not None
            }
            file_config.update(cli_overrides)
            config = file_config
            if 'features' in config and isinstance(config['features'], str):
                config['features'] = config['features'].split(",")
        else:
            print(f"Error: Config file not found at {args.load_config}")
            sys.exit(1)

    # --- Directory Setup --- #
    log_base_dir = config["log_dir"]
    model_name = config["model_name"]
    checkpoint_base_dir = config["checkpoint_dir"]
    ensure_dir_exists(log_base_dir)
    ensure_dir_exists(checkpoint_base_dir)
    ensure_dir_exists(os.path.join(log_base_dir, model_name))
    ensure_dir_exists(os.path.join(checkpoint_base_dir, model_name))

    # --- Mode Selection --- #
    if config.get("eval_only", False):
        # --- Evaluation Mode --- #
        print("Running in Evaluation-Only Mode")
        if config.get("load_model") is None:
            print("Error: Must provide --load_model for evaluation mode.")
            sys.exit(1)
        if config.get("test_data_path") is None:
            print("Error: Must provide --test_data_path for evaluation mode.")
            sys.exit(1)
        evaluate(config)

    else:
        # --- Training Mode --- #
        print(f"Running in Training Mode for model: {config['model_type']}")
        model, train_metrics = train(config)

        # Evaluate on test data if provided after training
        if config.get("test_data_path") is not None:
            print("\nStarting final evaluation on test data...")
            final_model_path = os.path.join(
                log_base_dir, model_name, "final_model.zip"
            )
            eval_config = config.copy()
            eval_config["load_model"] = final_model_path
            eval_config["test_data_path"] = config["test_data_path"]

            test_metrics = evaluate(eval_config)

            print("\n--- Training Summary ---")
            print(f"Training time: {train_metrics['training_time']:.2f} sec")
            print(f"Total steps: {train_metrics['total_timesteps']}")
            print(f"Final model: {final_model_path}")
            print("\n--- Test Set Evaluation Results ---")
            for key, value in test_metrics.items():
                print_str = f"  {key}: {value:.4f}" if isinstance(value, float) \
                    else f"  {key}: {value}"
                print(print_str)

        else:
            final_model_path = os.path.join(
                log_base_dir, model_name, "final_model.zip"
            )
            print("\n--- Training Completed --- ")
            print(f"Training time: {train_metrics['training_time']:.2f} sec")
            print(f"Total steps: {train_metrics['total_timesteps']}")
            print(f"Final model: {final_model_path}")
            print("No test data provided for final evaluation.")


if __name__ == "__main__":
    main() 