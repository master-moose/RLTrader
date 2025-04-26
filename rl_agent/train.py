#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main training script for the LSTM-DQN agent.

This script provides a command-line interface for training and evaluating 
the LSTM-DQN reinforcement learning agent on financial time series data.
"""

# --- Standard Library Imports --- #
import argparse
import copy
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Callable

# --- Third-Party Imports --- #
import gym
# Add support for gymnasium (newer version of gym)
try:
    import gymnasium
except ImportError:
    pass
    
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv, VecEnv,
                                             VecNormalize, sync_envs_normalization)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

# --- Project Imports --- #
# Note: TradingEnvironment is imported below; create_env is defined in this file
from rl_agent.utils import (calculate_max_drawdown, setup_logger, ensure_dir_exists)
# from rl_agent.features import load_model_features, FEATURE_CHOICES # Removed non-existent import

# --- Ray Tune Support --- #
RAY_AVAILABLE = False
try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    pass

def check_ray_session():
    """Helper function to check if a Ray Tune session is available in a way that works with different Ray versions."""
    if not RAY_AVAILABLE:
        return False
    
    try:
        # Modern Ray version uses session.report
        if hasattr(ray, "air") and hasattr(ray.air, "session"):
            return True
        # Older Ray versions check tune.is_session_enabled
        elif hasattr(tune, "is_session_enabled"):
            return tune.is_session_enabled()
        # Very old Ray versions use tune.report directly
        elif hasattr(tune, "report"):
            return True
        return False
    except (AttributeError, ImportError):
        return False

# --- Local Imports --- #
# _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # No longer needed
# sys.path.append(_parent_dir) # No longer needed

def find_project_root(marker='.git'):
    """Find the project root directory by searching upwards for a marker."""
    path = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(path)
        if os.path.exists(os.path.join(path, marker)):
            return path
        if parent == path: # Reached filesystem root
            # Fallback: Assume script is in root or one level down
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(os.path.join(script_dir, 'rl_agent')):
                 return script_dir
            parent_dir = os.path.dirname(script_dir)
            if os.path.exists(os.path.join(parent_dir, 'rl_agent')):
                 return parent_dir
            # If still not found, default to script's directory
            print("Warning: Project root marker not found. Using script directory.", file=sys.stderr)
            return script_dir
        path = parent

project_root = find_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"Debug: Added {project_root} to sys.path") # Optional debug

from rl_agent.callbacks import get_callback_list
from rl_agent.data.data_loader import DataLoader
from rl_agent.environment import TradingEnvironment
from rl_agent.models import LSTMFeatureExtractor
from rl_agent.policies import TcnPolicy, TcnSacPolicy
from rl_agent.utils import (
    calculate_trading_metrics, check_resources,
    create_evaluation_plots, ensure_dir_exists,
    load_config, save_config, set_seeds, setup_logger,
    setup_sb3_logger
)

# --- Global Settings --- #

# Define our own make_vec_env function
def make_vec_env(env_id, n_envs=1, seed=None, vec_env_cls=None, env_kwargs=None):
    """
    Create a wrapped, vectorized environment.
    Equivalent to stable_baselines3's make_vec_env function, implementing it here
    since it might not be available in some SB3 versions.
    
    :param env_id: The environment callable to create (can be a function returning a gym.Env)
    :param n_envs: The number of environments to create
    :param seed: The initial seed for the environment
    :param vec_env_cls: The vectorized env class to use (default: DummyVecEnv)
    :param env_kwargs: Additional keyword arguments for the environment
    :return: The vectorized environment
    """
    if vec_env_cls is None:
        vec_env_cls = DummyVecEnv
    
    env_kwargs = {} if env_kwargs is None else env_kwargs
    
    def make_env(rank):
        def _init():
            if callable(env_id):
                env = env_id()
            else:
                raise ValueError(f"Expected callable env_id, got {type(env_id)}")
            
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            
            return env
        return _init
    
    return vec_env_cls([make_env(i) for i in range(n_envs)])

# Initialize logger globally (will be configured later)
# Use specific logger name instead of __name__ for consistency
logger = logging.getLogger("rl_agent")


# --- Ray Tune Callback Class --- #

class TuneReportCallback(BaseCallback):
    """
    Callback for reporting metrics to Ray Tune during training.
    Tracks and reports episode rewards, returns, lengths, and other metrics.
    """
    
    def __init__(self, n_eval_episodes: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.n_eval_episodes = n_eval_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_returns = []
        self.eval_env_ids = []
        self.episode_count = 0
        self.step_count = 0
        self.start_time = time.time()
        self.num_timesteps = 0
        self.last_fps_log_time = 0
        self.rollout_count = 0
        self.best_mean_reward = -np.inf
        self.best_combined_score = -np.inf
        self.last_combined_score = 0.0  # Initialize here
        self.log_freq = 1 # Log summary table every 1 rollout
        
    def _on_training_start(self) -> None:
        """
        Called at the start of training.
        """
        self.start_time = time.time()
        
    def _calculate_eval_score(self, rewards, max_drawdown, sharpe, avg_trade_duration, win_rate):
        """Calculate a combined evaluation score.
        
        Args:
            rewards: Total episode rewards
            max_drawdown: Maximum drawdown (0-1 scale, lower is better)
            sharpe: Sharpe ratio (higher is better)
            avg_trade_duration: Average trade duration in steps
            win_rate: Proportion of winning trades (0-1)
            
        Returns:
            Combined score from 0-1
        """
        # Normalize values to appropriate ranges for scoring
        norm_drawdown = 1 - max_drawdown  # Convert so higher is better
        norm_sharpe = min(max(sharpe / 3.0, 0), 1)  # Scale with 3.0 being excellent
        
        # Calculate duration score (we want longer trades generally)
        # Scale so 20-step trades get 0.8 score, 50+ gets 1.0
        duration_score = min(avg_trade_duration / 50.0, 1.0)
        
        # Weights for combined score (sum should be 1.0)
        weights = {
            'reward': 0.3,
            'sharpe': 0.3,
            'drawdown': 0.2,
            'win_rate': 0.1,
            'duration': 0.1
        }
        
        # Calculate weighted score
        combined_score = (
            weights['reward'] * rewards +
            weights['sharpe'] * norm_sharpe +
            weights['drawdown'] * norm_drawdown +
            weights['win_rate'] * win_rate +
            weights['duration'] * duration_score
        )
        
        # Normalize to 0-1 range for final score
        # Using a logistic function to map values to 0-1 range
        # Adjust the scale to calibrate sensitivity
        scale = 5.0
        combined_score_normalized = 1 / (1 + np.exp(-scale * (combined_score - 0.5)))
        
        # Clip to ensure we're in the 0-1 range
        combined_score_final = np.clip(combined_score_normalized, 0.0, 1.0)
        return combined_score_final

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        This implementation reports metrics to Ray Tune when episodes complete.
        Now also reports FPS metrics periodically, regardless of episode completion.
        """
        # Determine model type from policy class name
        policy_name = self.model.policy.__class__.__name__.lower()
        if "sac" in policy_name:
            model_type = "sac"
        elif "ppo" in policy_name:
            model_type = "ppo"
        elif "dqn" in policy_name:
            model_type = "dqn"
        elif "a2c" in policy_name:
            model_type = "a2c"
        else:
            model_type = "unknown"
        
        self.step_count += 1
        self.num_timesteps = self.model.num_timesteps
        callback_logger = logging.getLogger("rl_agent.train")
        
        # Check for Ray Tune session in a way that works with different Ray versions
        has_ray_session = check_ray_session()
        
        # Periodically report FPS and step metrics
        # Reduce frequency from 10 to 1000
        if self.step_count % 1000 == 0 and has_ray_session:
            try:
                fps_val = int(self.num_timesteps / (time.time() - self.start_time)) if (time.time() - self.start_time) > 0 else 0
                report_dict = {
                    "timesteps_total": self.num_timesteps, # Use standard Ray Tune key
                    "fps": fps_val, # Use standard key
                    # --- Add combined_score reporting here ---
                    "combined_score": float(self.last_combined_score) 
                    # ------------------------------------------
                }
                # Use the modern check_ray_session function
                # Use the modern reporting API if available
                if hasattr(ray, "air") and hasattr(ray.air, "session"):
                    ray.air.session.report(report_dict)
                # Fallback to tune.report (passing dict)
                elif hasattr(tune, "report"):
                    tune.report(report_dict)
            except Exception as tune_err:
                callback_logger.error(f"Failed to report periodic metrics: {tune_err}")
                
        # Get most recently completed episodes
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            # Episode completed
            if "episode" in info.keys():
                # Log episode metrics locally
                episode_info = info["episode"]
                # Use dictionary access instead of attribute access
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                self.episode_returns.append(episode_info['r'])  # Same as rewards for now
                self.eval_env_ids.append(env_idx)

                # Info logging (local)
                callback_logger.info(
                    f"Episode {self.episode_count}: reward={episode_info['r']:.2f}, "
                    f"length={episode_info['l']}, env_id={env_idx}"
                )

                # Increment episode counter
                self.episode_count += 1
        
        # You would return False to stop training
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout in PPO. Collects metrics.
        """
        # Minimal version to isolate slowdowns
        callback_logger = logging.getLogger("rl_agent.callback")
        if not hasattr(self, 'num_timesteps') or self.num_timesteps == 0: return  # Skip initial call safely

        # --- Minimal Reporting --- #
        # Check if a Ray session is active
        if check_ray_session():
            try:
                # Only report essential metrics infrequently
                if self.rollout_count % 10 == 0: # Report every 10 rollouts
                    now = time.time()
                    start_time = getattr(self, 'start_time', now) # Get start time safely
                    fps = int(self.num_timesteps / (now - start_time)) if (now - start_time) > 0 else 0
                    
                    # Get the last known combined score safely
                    last_score = getattr(self, 'last_combined_score', 0.0)

                    report_dict = {
                        "timesteps_total": self.num_timesteps,
                        "rollout/fps": fps, # Use standard key
                        "combined_score": float(last_score), # Report last known score
                        # Add placeholders for other essential keys if Tune expects them
                        # "eval/mean_reward": 0.0,
                        # "eval/sharpe_ratio": 0.0,
                    }

                    callback_logger.debug(f"Minimal report keys: {list(report_dict.keys())}")

                    # Use modern ray.air.session.report API
                    if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
                            ray.air.session.report(report_dict)
                            callback_logger.debug(f"Reported {len(report_dict)} metrics via ray.air.session.report (minimal).")
                    # Fallback to tune.report (passing dict, not kwargs)
                    elif RAY_AVAILABLE and hasattr(tune, "report"):
                            tune.report(report_dict)
                            callback_logger.debug(f"Reported {len(report_dict)} metrics via tune.report (fallback, minimal).")
                    else:
                        callback_logger.warning("Could not find ray.air.session.report or tune.report to report minimal metrics.")

            except Exception as tune_err:
                callback_logger.error(f"Failed to report minimal metrics to Ray Tune: {tune_err}", exc_info=True)

        self.rollout_count += 1

        # Continue training
        return True

        # --- OLD DETAILED LOGIC (COMMENTED OUT BELOW) --- #
        # [...] # The rest of the original function remains commented out

# --- Ray Tune Trainable Function --- #

def train_rl_agent_tune(config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable function for training an RL agent.
    
    Args:
        config: Dictionary of hyperparameters from Ray Tune
    """
    # --- Enable Anomaly Detection --- #
    torch.autograd.set_detect_anomaly(True)
    # -------------------------------- #

    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not found. Install via pip.")

    train_config = config.copy()
    train_config.setdefault("model_type", "recurrentppo")
    train_config.setdefault("verbose", 1)

    # --- Trial Naming and Paths ---
    try:
        trial_id = tune.get_trial_id()
    except Exception:
        trial_id = f"standalone_{int(time.time())}"
    model_name = f"{train_config['model_type']}_{trial_id}"
    train_config["model_name"] = model_name

    try:
        ray_trial_dir = tune.get_trial_dir()
        default_base_log_dir = ray_trial_dir
    except Exception:
        default_base_log_dir = os.path.abspath("./logs")
    base_log_dir = train_config.get("log_dir", default_base_log_dir)
    log_dir = os.path.join(base_log_dir, model_name)
    base_checkpoint_dir = train_config.get("checkpoint_dir", base_log_dir)
    checkpoint_dir = os.path.join(base_checkpoint_dir, model_name)
    ensure_dir_exists(log_dir)
    ensure_dir_exists(checkpoint_dir)
    train_config["log_dir"] = log_dir
    train_config["checkpoint_dir"] = checkpoint_dir

    # --- Logger Setup --- #
    log_level_to_set = logging.DEBUG if config.get("verbose", 1) >= 2 \
        else logging.INFO
    setup_logger(
        log_dir=log_dir,
        log_level=log_level_to_set,
        console_level=log_level_to_set
    )
    trial_logger = logging.getLogger("rl_agent")

    # --- Initial Reporting and Seeding ---
    # Removed initial_metrics dictionary and the initial tune.report() call
    # as it caused TypeErrors with unexpected keyword arguments.
    # Reporting will be handled by the TuneReportCallback during training.
    # initial_metrics = {
    #     "config": train_config
    # }
    # trial_logger.info("Reporting initial metrics to Ray Tune")
    # tune.report(**initial_metrics)

    seed = train_config.get("seed")
    if seed is None:
        seed = abs(hash(trial_id)) % (2**32)
        train_config["seed"] = seed
    set_seeds(seed)

    trial_logger.info(f"Starting Ray Tune trial ID: {trial_id}")
    trial_logger.info("Full Configuration:")
    for key, value in sorted(train_config.items()):
        trial_logger.info(f"  {key}: {value}")

    sb3_log_dir = os.path.join(log_dir, "sb3_logs")
    ensure_dir_exists(sb3_log_dir)
    sb3_logger_instance = setup_sb3_logger(log_dir=sb3_log_dir)

    # --- Dependent Parameters (e.g., batch_size) ---
    model_type = train_config.get("model_type", "recurrentppo").lower()
    num_envs = train_config.get("num_envs", 8)
    n_steps = train_config.get("n_steps") # Might be None for non-PPO

    # Only calculate PPO-specific batch_size if using a PPO model and n_steps exists
    if 'ppo' in model_type and n_steps is not None:
        total_steps_per_rollout = n_steps * num_envs
        if "batch_size" not in train_config:
            train_config["batch_size"] = min(total_steps_per_rollout, 4096)
            trial_logger.info(
                f"Setting PPO batch_size={train_config['batch_size']} "
                f"(n_steps={n_steps}, num_envs={num_envs})"
            )
        else:
            # Existing checks for PPO batch_size validity
            if train_config["batch_size"] > total_steps_per_rollout:
                trial_logger.warning(
                    f"batch_size ({train_config['batch_size']}) > "
                    f"n_steps*num_envs ({total_steps_per_rollout}). Clipping."
                )
                train_config["batch_size"] = total_steps_per_rollout
            elif total_steps_per_rollout % train_config["batch_size"] != 0:
                trial_logger.warning(
                    f"n_steps*num_envs ({total_steps_per_rollout}) not divisible "
                    f"by batch_size ({train_config['batch_size']})."
                )
    elif "batch_size" not in train_config:
        # For non-PPO models, if batch_size is not in config, let create_model handle it
        trial_logger.info(f"batch_size not in config for {model_type}, using default in create_model")
    else:
        # If batch_size IS in config (e.g., from SAC search space), use it
        trial_logger.info(f"Using batch_size={train_config['batch_size']} from config for {model_type}")

    # --- Environment Creation --- #
    trial_logger.info(f"Creating {num_envs} parallel environment(s)...")

    # Define the environment creation function for Ray Tune workers
    def make_single_env(rank):
        def _init():
            env_config = train_config.copy() # Use the trial's config
            instance_seed = seed
            if instance_seed is not None:
                 instance_seed += rank
            env_config["seed"] = instance_seed # Pass seed to create_env

            # Create the environment using the project's create_env function
            env = create_env(config=env_config, is_eval=False)

            # Wrap with Monitor to capture episode stats
            # Use the trial's log_dir for monitor files
            monitor_log = os.path.join(log_dir, f'monitor_train_rank{rank}.csv')
            os.makedirs(os.path.dirname(monitor_log), exist_ok=True)
            env = Monitor(env, filename=monitor_log)

            # Seeding is handled by create_env based on env_config["seed"]

            return env
        return _init

    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    # Use the make_single_env factory to create training environments
    train_env = make_vec_env(
        env_id=make_single_env(rank=0), # Pass factory with rank 0 and base trial seed
        n_envs=num_envs,
        seed=None, # Seed is handled within the factory
        vec_env_cls=vec_env_cls,
        env_kwargs=None
    )

    # --- VecNormalize Setup --- #
    norm_obs_setting = train_config.get("norm_obs", "auto").lower()
    if norm_obs_setting == "auto":
        features = train_config.get("features", [])
        if isinstance(features, str): features = features.split(",")
        has_scaled_features = any("_scaled" in f for f in features)
        should_norm_obs = not has_scaled_features
        trial_logger.info(
            f"Auto-detected norm_obs={should_norm_obs} "
            f"(scaled features: {has_scaled_features})"
        )
    else:
        should_norm_obs = norm_obs_setting == "true"
        trial_logger.info(f"Explicit norm_obs={should_norm_obs}")

    train_env = VecNormalize(
        train_env,
        norm_obs=should_norm_obs,
        norm_reward=False,
        clip_obs=10.,
        gamma=train_config["gamma"]
    )

    # --- Validation Environment Setup --- #
    eval_env = None
    if train_config.get("val_data_path"):
        trial_logger.info(f"Creating validation env: {train_config['val_data_path']}")
        eval_env_seed_val = seed + num_envs if seed is not None else None
        # Step 1: Create the base DummyVecEnv for evaluation
        raw_eval_env = make_vec_env(
            env_id=make_single_env(rank=0), # Pass factory with rank 0 and eval seed
            n_envs=1, seed=None, vec_env_cls=DummyVecEnv, env_kwargs=None
        )

        # Step 2: Apply VecNormalize wrapper, mirroring train_env setup
        if config.get("load_model"):
            potential_stats = config["load_model"].replace(".zip", "_vecnormalize.pkl")
            if os.path.exists(potential_stats):
                trial_logger.info(f"Loading VecNorm stats: {potential_stats}")
                eval_env = VecNormalize.load(potential_stats, raw_eval_env)
                # Ensure it's set to inference mode
                eval_env.training = False
                eval_env.norm_reward = False
            else:
                trial_logger.warning("VecNorm stats not found. Creating fresh wrapper.")
                norm_obs_setting = config.get("norm_obs", "auto").lower()
                if norm_obs_setting == "auto":
                    features = config.get("features", [])
                    if isinstance(features, str): features = features.split(",")
                    has_scaled = any("_scaled" in f for f in features)
                    should_norm_obs = not has_scaled
                else: should_norm_obs = norm_obs_setting == "true"
                trial_logger.info(f"Eval VecNorm: norm_obs={should_norm_obs}")
                eval_env = VecNormalize(
                    raw_eval_env, # Wrap the raw eval env
                    norm_obs=should_norm_obs,
                    norm_reward=False, # Never normalize rewards for eval
                    clip_obs=10.,
                    gamma=config["gamma"],
                    training=False # Set to False for evaluation
                )
        else:
            trial_logger.warning("No load_model specified, skipping VecNorm for eval_env.")

        trial_logger.info(f"Validation env wrapped with VecNormalize: {eval_env}")

    # --- Model Creation --- #
    trial_logger.info(f"Creating new {train_config['model_type']} model")
    model = create_model(env=train_env, config=train_config)
    model.set_logger(sb3_logger_instance)

    # --- Callbacks Setup --- #
    model_type = train_config.get("model_type", "recurrentppo").lower()
    n_steps = train_config.get("n_steps") # Already fetched earlier, re-get for clarity or use variable
    num_envs = train_config.get("num_envs", 1)
    base_eval_freq = train_config.get("eval_freq", 10000)

    # Calculate effective_eval_freq differently based on model type
    if 'ppo' in model_type and n_steps is not None:
        rollout_steps = n_steps * num_envs
        effective_eval_freq = max(base_eval_freq, rollout_steps)
        trial_logger.info(f"PPO model: Effective eval_freq = {effective_eval_freq} (max({base_eval_freq}, {rollout_steps}))")
    else:
        # For non-PPO models, just use the base eval frequency
        effective_eval_freq = base_eval_freq
        trial_logger.info(f"{model_type.upper()} model: Effective eval_freq = {effective_eval_freq}")

    tune_early_stopping_patience = 0 # Disabled for debugging
    trial_logger.info(f"Tune early_stopping_patience = {tune_early_stopping_patience}")

    callbacks = get_callback_list(
        eval_env=eval_env,
        log_dir=log_dir,
        eval_freq=effective_eval_freq, # Use calculated effective frequency
        n_eval_episodes=train_config.get("n_eval_episodes", 5),
        save_freq=config.get("save_freq", 10000),
        keep_checkpoints=config.get("keep_checkpoints", 3),
        resource_check_freq=config.get("resource_check_freq", 1000),
        metrics_log_freq=config.get("metrics_log_freq", 1000),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        checkpoint_save_path=checkpoint_dir,
        model_name=train_config["model_type"],
        # Conditionally add TuneReportCallback only if running under Ray Tune
        custom_callbacks=[TuneReportCallback()] if check_ray_session() else [],
        curriculum_duration_fraction=0.0
    )

    # --- Training Loop --- #
    trial_logger.info(f"Starting training: {train_config['total_timesteps']} steps")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=train_config["total_timesteps"],
            callback=callbacks,
            log_interval=250,  # Log SB3 stats every 250 updates
            reset_num_timesteps=not train_config.get("continue_training", False)
        )
        training_time = time.time() - training_start_time
        trial_logger.info(f"Training finished in {training_time:.2f}s")
    except Exception as e:
        trial_logger.error(f"Error during model.learn: {e}", exc_info=True)
        # Attempt to report failure metrics
        if check_ray_session():
            try:
                failure_metrics = {
                    "trial_status": "ERROR",
                    "error_message": str(e)[:200], # Limit error length
                    "timesteps_total": getattr(model, 'num_timesteps', 0),
                    # Try to get last known values
                    "eval/mean_reward": 0.0, 
                    "eval/explained_variance": 0.0,
                    "combined_score": 0.0 # Default score on failure
                }
                last_reward = 0.0
                last_variance = None
                last_actor_loss = None
                last_critic_loss = None
                if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
                    log_vals = model.logger.name_to_value
                    last_reward = log_vals.get("rollout/ep_rew_mean", 0.0)
                    last_variance = log_vals.get("train/explained_variance", 0.0)
                    # Try to get losses based on model type
                    model_type = train_config.get("model_type", "ppo")
                    if model_type == "sac":
                        last_actor_loss = log_vals.get("train/actor_loss", log_vals.get("train/policy_loss"))
                        last_critic_loss = log_vals.get("train/critic_loss")
                    elif model_type in ["ppo", "a2c", "tcn_ppo", "recurrentppo"]:
                        last_actor_loss = log_vals.get("train/policy_loss")
                        last_critic_loss = log_vals.get("train/value_loss")
                    
                    try: last_reward = float(last_reward)
                    except (ValueError, TypeError): last_reward = 0.0
                    try: last_variance = float(last_variance) if last_variance is not None else None
                    except (ValueError, TypeError): last_variance = None
                    try: 
                        if last_actor_loss is not None:
                            last_actor_loss = float(last_actor_loss)
                    except (ValueError, TypeError): last_actor_loss = None
                    try: 
                        if last_critic_loss is not None:
                             last_critic_loss = float(last_critic_loss)
                    except (ValueError, TypeError): last_critic_loss = None

                failure_metrics["eval/mean_reward"] = last_reward
                
                # --- FIX: Call the global function, not the callback method ---
                combo_score = normalize_and_combine_metrics(
                    reward=last_reward,
                    explained_variance=last_variance, 
                    sharpe_ratio=None, # Not available here
                    episode_return=None, # Not available here
                    calmar_ratio=None, # Not available here
                    sortino_ratio=None, # Not available here
                    actor_loss=last_actor_loss,
                    critic_loss=last_critic_loss,
                    model_type=train_config.get("model_type", "ppo")
                )
                # -------------------------------------------------------------
                
                if train_config.get("model_type", "ppo") != "sac":
                    failure_metrics["eval/explained_variance"] = last_variance
                failure_metrics["combined_score"] = float(combo_score if combo_score is not None and np.isfinite(combo_score) else 0.0)
                trial_logger.info(f"Reporting failure: {failure_metrics}")

                # Use the updated check_ray_session function
                has_ray_session = check_ray_session()
                if has_ray_session:
                    if hasattr(ray, "air") and hasattr(ray.air, "session"):
                        ray.air.session.report(failure_metrics)
                        trial_logger.info("Reported failure metrics via ray.air.session.report")
                    # Fallback to tune.report (passing dict)
                    elif hasattr(tune, "report"):
                        tune.report(failure_metrics)
                        trial_logger.info("Reported failure metrics via tune.report (fallback)")
                    else:
                        trial_logger.warning("Could not find ray.air.session.report or tune.report to report failure metrics.")
            except Exception as report_err:
                trial_logger.error(f"Failed to report failure: {report_err}")
        raise

    finally:
        # --- Environment Closure ---
        trial_logger.info("Closing environments...")
        if 'train_env' in locals() and hasattr(train_env, 'close'):
            try: train_env.close(); trial_logger.debug("Closed train_env.")
            except Exception as ce: trial_logger.error(f"Closing train_env failed: {ce}")
        if 'eval_env' in locals() and eval_env is not None and hasattr(eval_env, 'close'):
            try: eval_env.close(); trial_logger.debug("Closed eval_env.")
            except Exception as ce: trial_logger.error(f"Closing eval_env failed: {ce}")
        trial_logger.info("Env closure attempted.")

    # --- Final Evaluation --- #
    trial_logger.info("Performing final evaluation...")
    final_eval_metrics = {}
    if eval_env is not None:
        try:
            mean_reward, portfolio_values, actions, rewards = evaluate_model(
                model=model,
                env=eval_env,
                config=train_config,
                n_episodes=train_config.get("n_eval_episodes", 5),
                deterministic=True
            )
            final_eval_metrics["final_eval_mean_reward"] = mean_reward
            trial_logger.info(f"Final eval mean reward: {mean_reward:.4f}")

            if portfolio_values is not None and len(portfolio_values) > 10:
                final_metrics_path = os.path.join(log_dir, "final_eval_metrics")
                ensure_dir_exists(final_metrics_path)
                try:
                    trading_metrics = calculate_trading_metrics(list(portfolio_values))
                    final_eval_metrics.update(trading_metrics)
                    create_evaluation_plots(
                        portfolio_values=list(portfolio_values),
                        actions=actions,
                        rewards=rewards,
                        save_path=os.path.join(final_metrics_path, "final_eval_plots.png")
                    )
                    metrics_file = os.path.join(final_metrics_path, "final_trading_metrics.json")
                    with open(metrics_file, 'w') as f:
                        json.dump({k: float(v) if isinstance(v, np.number) else v
                                   for k, v in trading_metrics.items()}, f, indent=2)
                    trial_logger.info(f"Saved final eval plots/metrics to {final_metrics_path}")
                except Exception as plot_err:
                    trial_logger.warning(f"Final plot/metrics failed: {plot_err}")
        except Exception as e:
            trial_logger.error(f"Final evaluation error: {e}", exc_info=True)
        # Ensure eval_env is closed if it was used
        if hasattr(eval_env, 'close'):
            try: eval_env.close(); trial_logger.debug("Closed eval_env post-eval.")
            except Exception as ce: trial_logger.error(f"Closing eval_env post-eval failed: {ce}")

    # --- Final Reporting --- #
    final_summary_metrics = {
        "training_time_total": training_time,
        "final_timesteps": getattr(model, 'num_timesteps', 0),
        **final_eval_metrics
    }
    # Add final logger metrics
    final_reward = 0.0
    final_variance = 0.0
    if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
        log_vals = model.logger.name_to_value
        final_reward = log_vals.get("rollout/ep_rew_mean", 0.0)
        final_variance = log_vals.get("train/explained_variance", 0.0)
        try: final_reward = float(final_reward)
        except (ValueError, TypeError): final_reward = 0.0
        try: final_variance = float(final_variance)
        except (ValueError, TypeError): final_variance = 0.0

        final_summary_metrics["final_logger_mean_reward"] = final_reward
        final_summary_metrics["final_logger_explained_variance"] = final_variance

        temp_cb = TuneReportCallback()
        final_combined = temp_cb._normalize_and_combine_metrics(
            reward=final_reward, 
            explained_variance=final_variance, 
            sharpe_ratio=0.0, 
            episode_return=0.0, 
            calmar_ratio=0.0, 
            sortino_ratio=0.0,
            actor_loss=None,
            critic_loss=None,
            model_type=train_config.get("model_type", "ppo") # Get model type from config
        )
        final_summary_metrics["combined_score"] = float(final_combined if final_combined is not None and np.isfinite(final_combined) else 0.0)

    trial_logger.info("Final Summary Metrics:")
    for k, v in final_summary_metrics.items():
        trial_logger.info(f"  {k}: {v}")

    # Use the modern check for Ray Tune session
    session_active = False
    if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
        session_active = ray.air.session.is_active()

    if session_active:
        try: 
            # Use modern ray.air.session.report API
            if hasattr(ray, "air") and hasattr(ray.air, "session"):
                ray.air.session.report(final_summary_metrics) 
                trial_logger.info("Reported final metrics via Ray AIR session")
            # Fallback to tune.report (passing dict)
            elif hasattr(tune, "report"):
                tune.report(final_summary_metrics)
                trial_logger.info("Reported final metrics via tune.report (fallback)")
            else:
                 trial_logger.warning("Could not find ray.air.session.report or tune.report to report final metrics.")

        except Exception as re: 
            trial_logger.warning(f"Failed final Ray AIR session report: {re}")

# --- Argument Parsing --- #

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate RL agents for trading"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments...
    # (Assuming arguments are defined as before)
    # ... Rest of parse_args function ...
    # --- General Parameters --- #
    general = parser.add_argument_group('General Parameters')
    general.add_argument(
        "--model_type", type=str, default="recurrentppo", # Changed default
        choices=[
            "dqn", "ppo", "a2c", "sac", "lstm_dqn", "qrdqn", "recurrentppo", "tcn_ppo", "tcn_sac"
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
        default=(
            "open_scaled,high_scaled,low_scaled,close_scaled,volume_scaled,sma_7_scaled,sma_25_scaled,sma_99_scaled,"
            "ema_9_scaled,ema_21_scaled,rsi_14_scaled,macd_scaled,macd_signal_scaled,macd_hist_scaled,"
            "atr_14_scaled,true_volatility_scaled,volume_sma_20_scaled,volume_ratio_scaled,price_sma_ratio_scaled,"
            "sma_cross_signal_scaled,bb_middle_scaled,bb_upper_scaled,bb_lower_scaled,bb_width_scaled,bb_pct_b_scaled"
        ),
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
        help="Max position size as fraction of balance (0.0-1.0)"
    )
    env.add_argument(
        "--commission", type=float, default=0.0, # Updated to 0.0% Maker fee (BNB Discount)
        help="Trading commission percentage (0.0 = 0.0%)"
    )
    env.add_argument(
        "--max_steps", type=int, default=100000,
        help="Maximum steps per episode"
    )
    env.add_argument(
        "--episode_length", type=int, default=None,
        help="Length of episode in days (overrides max_steps)"
    )
    env.add_argument(
        "--reward_scaling", type=float, default=1.0,
        help="Scaling factor for environment rewards (0.1-10.0)"
    )
    env.add_argument(
        "--max_holding_steps", type=int, default=8,
        help="Max steps to hold before potential forced action"
    )
    env.add_argument(
        "--take_profit_pct", type=float, default=0.03,
        help="Take profit percentage (0.01-0.05)"
    )
    env.add_argument(
        "--target_cash_ratio", type=str, default="0.3-0.7",
        help="Target cash ratio range for reward shaping ('min-max')"
    )

    # --- Reward Component Weights --- #
    rewards = parser.add_argument_group('Reward Parameters')
    rewards.add_argument("--portfolio_change_weight", type=float, default=1.0)
    rewards.add_argument("--drawdown_penalty_weight", type=float, default=0.5)
    rewards.add_argument("--sharpe_reward_weight", type=float, default=0.5)
    rewards.add_argument("--fee_penalty_weight", type=float, default=2.0)
    rewards.add_argument("--benchmark_reward_weight", type=float, default=0.5)
    rewards.add_argument("--idle_penalty_weight", type=float, default=0.1)
    rewards.add_argument("--profit_bonus_weight", type=float, default=0.5)
    rewards.add_argument("--exploration_bonus_weight", type=float, default=0.1)
    rewards.add_argument("--trade_penalty_weight", type=float, default=0.0)

    # --- Additional Reward Parameters --- #
    rewards.add_argument("--sharpe_window", type=int, default=20)
    rewards.add_argument("--consistency_threshold", type=int, default=3)
    rewards.add_argument("--idle_threshold", type=int, default=5)

    # --- Common Training Parameters --- #
    training = parser.add_argument_group('Common Training Parameters')
    training.add_argument("--total_timesteps", type=int, default=1000000)
    training.add_argument("--learning_rate", type=float, default=0.0003)
    training.add_argument("--batch_size", type=int, default=2048)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--eval_freq", type=int, default=10000)
    training.add_argument("--n_eval_episodes", type=int, default=5)
    training.add_argument("--save_freq", type=int, default=50000)
    training.add_argument("--keep_checkpoints", type=int, default=3)
    training.add_argument("--early_stopping_patience", type=int, default=10)
    training.add_argument("--num_envs", type=int, default=1)

    # --- DQN/QRDQN Specific Parameters --- #
    dqn = parser.add_argument_group('DQN/QRDQN Specific Parameters')
    dqn.add_argument("--buffer_size", type=int, default=100000)
    dqn.add_argument("--exploration_fraction", type=float, default=0.1)
    dqn.add_argument("--exploration_initial_eps", type=float, default=1.0)
    dqn.add_argument("--exploration_final_eps", type=float, default=0.05)
    dqn.add_argument("--target_update_interval", type=int, default=10000)
    dqn.add_argument("--gradient_steps", type=int, default=1)
    dqn.add_argument("--learning_starts", type=int, default=1000)
    dqn.add_argument("--exploration_start", type=float, default=1.0)
    dqn.add_argument("--exploration_end", type=float, default=0.01)
    dqn.add_argument("--exploration_decay_rate", type=float, default=0.0001)
    dqn.add_argument("--n_quantiles", type=int, default=200)

    # --- PPO/A2C Specific Parameters --- #
    ppo = parser.add_argument_group('PPO/A2C Specific Parameters')
    ppo.add_argument("--n_steps", type=int, default=2048)
    ppo.add_argument("--ent_coef", type=str, default="0.01",
                       help="Entropy coefficient (PPO/A2C: float, SAC: float or 'auto')")
    ppo.add_argument("--vf_coef", type=float, default=0.5)
    ppo.add_argument("--n_epochs", type=int, default=10)
    ppo.add_argument("--clip_range", type=float, default=0.2)
    ppo.add_argument("--gae_lambda", type=float, default=0.95)
    ppo.add_argument("--max_grad_norm", type=float, default=0.5)

    # --- SAC Specific Parameters --- #
    sac = parser.add_argument_group('SAC Specific Parameters')
    sac.add_argument("--tau", type=float, default=0.005)
    sac.add_argument("--use_sde", action="store_true",
                       help="Use State Dependent Exploration (SDE) for SAC")
    sac.add_argument("--sde_sample_freq", type=int, default=-1,
                       help="Sample frequency for SDE (if --use_sde is set)")
    sac.add_argument("--batch_size", type=int, default=256,
                       help="Minibatch size for SAC optimization")
    sac.add_argument("--buffer_size", type=int, default=1000000,
                       help="Replay buffer size for SAC")
    sac.add_argument("--learning_starts", type=int, default=100,
                       help="How many steps of interactions to collect before starting learning for SAC")

    # --- LSTM/Network Architecture --- #
    network = parser.add_argument_group('Network Architecture')
    network.add_argument("--lstm_model_path", type=str, default=None)
    network.add_argument("--lstm_hidden_size", type=int, default=128)
    network.add_argument("--n_lstm_layers", type=int, default=1)
    network.add_argument("--shared_lstm", type=str, default="shared",
                         choices=["shared", "seperate", "none"])
    network.add_argument("--fc_hidden_size", type=int, default=64)

    # --- Resource Management & Logging --- #
    logging_group = parser.add_argument_group('Logging & Resources')
    logging_group.add_argument("--resource_check_freq", type=int, default=5000)
    logging_group.add_argument("--metrics_log_freq", type=int, default=1000)
    logging_group.add_argument("--log_dir", type=str, default="./logs")
    logging_group.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    logging_group.add_argument("--norm_obs", type=str, default="auto",
                               choices=["auto", "true", "false"],
                               help="Control VecNormalize ('auto': based on feats)")
                               
    # --- TCN Parameters --- #
    tcn_group = parser.add_argument_group('TCN Parameters (for tcn_ppo)')
    tcn_group.add_argument("--tcn_num_layers", type=int, default=4,
                         help="Number of TCN layers (4-6 recommended)")
    tcn_group.add_argument("--tcn_num_filters", type=int, default=64,
                         help="Number of filters in each TCN layer")
    tcn_group.add_argument("--tcn_kernel_size", type=int, default=3,
                         help="Kernel size for TCN convolutions")
    tcn_group.add_argument("--tcn_dropout", type=float, default=0.2,
                         help="Dropout rate for TCN layers")

    args = parser.parse_args()

    if args.model_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.model_name = f"{args.model_type}_{timestamp}"

    if isinstance(args.features, str):
        args.features = args.features.split(",")

    return args


# --- Helper Functions --- #

def args_to_config(args) -> Dict[str, Any]:
    """Convert argparse arguments to config dictionary using deepcopy."""
    return copy.deepcopy(vars(args)) # USE DEEPCOPY


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining: (float) Remaining progress.
        :return: (float) current learning rate
        """
        # Simple linear decay to almost zero (add small epsilon if needed)
        # Example: return max(progress_remaining * initial_value, 1e-6)
        return progress_remaining * initial_value

    return func


def create_env(
    config: Dict[str, Any],
    data_override: Optional[pd.DataFrame] = None,
    is_eval: bool = False
) -> gym.Env:
    """
    Create a trading environment based on configuration.

    Args:
        config: Configuration dictionary containing env parameters.
        data_override: Optional DataFrame to use instead of loading from path.
        is_eval: Flag indicating if this is for evaluation.

    Returns:
        Trading environment instance.
    """
    if data_override is not None:
        data = data_override
    else:
        data_path = config["data_path"]
        # Load data for each timeframe
        data_15m = DataLoader(data_path=data_path, data_key="/15m").load_data()
        data_4h = DataLoader(data_path=data_path, data_key="/4h").load_data()
        data_1d = DataLoader(data_path=data_path, data_key="/1d").load_data()

        # Combine data into a single DataFrame
        # Assuming the data can be aligned on a common index like timestamp
        data = data_15m.join(data_4h, rsuffix='_4h').join(data_1d, rsuffix='_1d')

    env_kwargs = {
        "data": data,
        "features": config.get("features"),
        "sequence_length": config["sequence_length"],
        "initial_balance": config["initial_balance"],
        "transaction_fee": config["commission"],
        "reward_scaling": config["reward_scaling"],
        "window_size": config.get("window_size", 20),
        "max_position": config.get("max_position", 1.0),
        "max_steps": config.get("max_steps"),
        "random_start": config.get("random_start", True),
    }

    if isinstance(env_kwargs["features"], str):
        logger.debug(f"Splitting features string: {env_kwargs['features']}")
        env_kwargs["features"] = [f.strip() for f in env_kwargs["features"].split(',') if f.strip()]
        logger.debug(f"Converted features to list: {env_kwargs['features']}")
    elif env_kwargs["features"] is None:
        logger.warning("Features not provided; TradingEnv might fail.")
        env_kwargs["features"] = []

    # Filter out features with suffixes for other timeframes
    # features = [f for f in env_kwargs['features'] if not f.endswith(('_15m', '_4h', '_1d'))] # Removed filtering
    # env_kwargs['features'] = features # Keep all features specified in config

    reward_param_keys = [
        "portfolio_change_weight", "drawdown_penalty_weight", "sharpe_reward_weight",
        "fee_penalty_weight", "benchmark_reward_weight", # Removed "consistency_penalty_weight",
        "idle_penalty_weight", "profit_bonus_weight", "exploration_bonus_weight",
        "sharpe_window", # Removed "consistency_threshold",
        "idle_threshold",
        "exploration_start", "exploration_end", "exploration_decay_rate",
        "trade_penalty_weight"
    ]
    for key in reward_param_keys:
        if key in config:
            env_kwargs[key] = config[key]
            # logger.debug(f"Passing reward param '{key}' = {config[key]} to env.")

    # Explicitly remove fee parameters if they somehow ended up in env_kwargs
    env_kwargs.pop('transaction_fee', None)
    env_kwargs.pop('fee_penalty_weight', None)

    env = TradingEnvironment(**env_kwargs)
    return env


def create_model(
    env: gym.Env,
    config: Dict[str, Any],
) -> BaseAlgorithm: # Changed type hint
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
    device = "cpu" if config["cpu_only"] else "auto"
    policy_kwargs = {}
    tb_log_path = None
    if config.get("log_dir") and config.get("model_name"):
        tb_log_path = os.path.join(config["log_dir"], config["model_name"], "sb3_logs")

    model_kwargs = {
        "policy": None, "env": env, "learning_rate": learning_rate,
        "gamma": config["gamma"], "seed": seed, "device": device,
        "verbose": 0, "policy_kwargs": policy_kwargs, "tensorboard_log": tb_log_path
    }

    use_lstm_features = (model_type == "lstm_dqn" or config.get("lstm_model_path"))
    if use_lstm_features:
        lstm_state_dict = config.get("lstm_model_path")
        if lstm_state_dict: logger.info(f"Using LSTM model path: {lstm_state_dict}")
        policy_kwargs["features_extractor_class"] = LSTMFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "lstm_state_dict": lstm_state_dict,
            "features_dim": config.get("lstm_hidden_size", 128)
        }
        if config.get("fc_hidden_size", 0) > 0:
            fc_size = config["fc_hidden_size"]
            if model_type in ["ppo", "a2c"]:
                policy_kwargs["net_arch"] = [fc_size, fc_size]

    # Default net_arch (can be overridden by specific model type below)
    if "net_arch" not in policy_kwargs and model_type not in ["recurrentppo"]:
        fc_size = config.get("fc_hidden_size", 64)
        policy_kwargs["net_arch"] = [fc_size] * 2 # Default shared [64, 64]

    # --- Algorithm Specific Setup --- #
    if model_type == "dqn" or model_type == "lstm_dqn":
        model_kwargs.update({
            "policy": DqnMlpPolicy,
            "buffer_size": config["buffer_size"],
            "batch_size": config["batch_size"],
            "learning_starts": config["learning_starts"],
            "gradient_steps": config["gradient_steps"],
            "target_update_interval": config["target_update_interval"],
            "exploration_fraction": config["exploration_fraction"],
            "exploration_initial_eps": config["exploration_initial_eps"],
            "exploration_final_eps": config["exploration_final_eps"],
            "replay_buffer_class": ReplayBuffer,
            "replay_buffer_kwargs": {}
        })
        model_kwargs.pop("tensorboard_log", None)
        model = DQN(**model_kwargs)

    elif model_type == "ppo":
        lr = config["learning_rate"]
        lr_schedule = linear_schedule(lr) if isinstance(lr, float) else lr
        model_kwargs.update({
            "policy": ActorCriticPolicy,
            "learning_rate": lr_schedule, # Use schedule
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "n_epochs": config["n_epochs"],
            "ent_coef": float(config["ent_coef"]),
            "vf_coef": config["vf_coef"],
            "clip_range": config["clip_range"],
            "gae_lambda": config["gae_lambda"],
            "max_grad_norm": config["max_grad_norm"]
        })
        model = PPO(**model_kwargs)

    elif model_type == "a2c":
        lr = config["learning_rate"]
        lr_schedule = linear_schedule(lr) if isinstance(lr, float) else lr
        model_kwargs.update({
            "policy": ActorCriticPolicy,
            "learning_rate": lr_schedule, # Use schedule
            "n_steps": config["n_steps"],
            "ent_coef": float(config["ent_coef"]),
            "vf_coef": config["vf_coef"],
            "gae_lambda": config["gae_lambda"],
            "max_grad_norm": config["max_grad_norm"],
            "rms_prop_eps": config.get("rms_prop_eps", 1e-5)
        })
        model = A2C(**model_kwargs)

    elif model_type == "sac":
        lr = config["learning_rate"]
        lr_schedule = linear_schedule(lr) if isinstance(lr, float) else lr
        ent_coef_value = config.get("ent_coef", "auto")
        if isinstance(ent_coef_value, str) and ent_coef_value.lower() == 'auto':
            sac_ent_coef = 'auto'
        else:
            try: sac_ent_coef = float(ent_coef_value)
            except ValueError: sac_ent_coef = 'auto'; logger.warning(f"Invalid SAC ent_coef. Defaulting auto.")

        model_kwargs.update({
            "policy": SacMlpPolicy, # Use the imported policy
            "learning_rate": lr_schedule, # Use schedule
            "buffer_size": config["buffer_size"],
            "batch_size": config["batch_size"],
            "learning_starts": config["learning_starts"],
            "gradient_steps": config["gradient_steps"],
            "target_update_interval": config["target_update_interval"],
            "tau": config["tau"],
            "ent_coef": sac_ent_coef
        })
        model = SAC(**model_kwargs)

    elif model_type == "qrdqn":
        from sb3_contrib.qrdqn.policies import QRDQNPolicy
        policy_kwargs["n_quantiles"] = config.get("n_quantiles", 200)
        model_kwargs.update({
            "policy": QRDQNPolicy,
            "buffer_size": config["buffer_size"],
            "batch_size": config["batch_size"],
            "learning_starts": config["learning_starts"],
            "gradient_steps": config["gradient_steps"],
            "target_update_interval": config["target_update_interval"],
            "exploration_fraction": config["exploration_fraction"],
            "exploration_initial_eps": config["exploration_initial_eps"],
            "exploration_final_eps": config["exploration_final_eps"]
        })
        model_kwargs.pop("tensorboard_log", None)
        model = QRDQN(**model_kwargs)

    elif model_type == "recurrentppo":
        lr = config["learning_rate"]
        lr_schedule = linear_schedule(lr) if isinstance(lr, float) else lr
        policy_kwargs["lstm_hidden_size"] = config.get("lstm_hidden_size", 128)
        policy_kwargs["n_lstm_layers"] = config.get("n_lstm_layers", 1)
        shared_lstm_mode = config.get("shared_lstm", "shared")
        valid_modes = ["shared", "seperate", "none"]
        if shared_lstm_mode not in valid_modes: shared_lstm_mode = "shared"
        if shared_lstm_mode == "shared": policy_kwargs.update({"shared_lstm": True, "enable_critic_lstm": False})
        elif shared_lstm_mode == "seperate": policy_kwargs.update({"shared_lstm": False, "enable_critic_lstm": True})
        else: policy_kwargs.update({"shared_lstm": False, "enable_critic_lstm": False})

        model_kwargs.update({
            "policy": "MlpLstmPolicy",
            "learning_rate": lr_schedule, # Use schedule
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "n_epochs": config["n_epochs"],
            "ent_coef": float(config["ent_coef"]),
            "vf_coef": config["vf_coef"],
            "clip_range": config["clip_range"],
            "gae_lambda": config["gae_lambda"],
            "max_grad_norm": config["max_grad_norm"]
        })
        # Add default net_arch for policy/value heads after LSTM
        fc_size = config.get("fc_hidden_size", 64)
        policy_kwargs["net_arch"] = [fc_size] * 2
        logger.info(f"RecurrentPPO LSTM: hidden={policy_kwargs['lstm_hidden_size']}, layers={policy_kwargs['n_lstm_layers']}, shared={policy_kwargs['shared_lstm']}, critic={policy_kwargs['enable_critic_lstm']}, net_arch={policy_kwargs['net_arch']}")
        model = RecurrentPPO(**model_kwargs)

    elif model_type == "tcn_ppo":
        lr = config["learning_rate"]
        lr_schedule = linear_schedule(lr) if isinstance(lr, float) else lr
        
        # Configure TCN parameters
        sequence_length = config.get("sequence_length", 60)
        tcn_params = {
            "num_layers": config.get("tcn_num_layers", 4),
            "num_filters": config.get("tcn_num_filters", 64),
            "kernel_size": config.get("tcn_kernel_size", 3),
            "dropout": config.get("tcn_dropout", 0.2)
        }
        
        # --- Infer actual features per timestep from environment's observation space ---
        # The 'env' passed here is the VecNormalize-wrapped environment
        # Updated to handle both gym and gymnasium Box spaces 
        if not hasattr(env.observation_space, 'shape'):
            raise ValueError(f"TcnPolicy requires an observation space with a shape attribute, got {type(env.observation_space)}")
    
        obs_shape = env.observation_space.shape
        if len(obs_shape) != 1:
            raise ValueError(f"TcnPolicy requires a 1D observation space shape (features_dim,), got {obs_shape}")
        
        total_obs_dim = obs_shape[0] # This includes features * seq_len + 2 state vars
        # Infer actual features per timestep, accounting for the 2 state variables
        expected_feature_dim = total_obs_dim - 2 
        if expected_feature_dim <= 0 or expected_feature_dim % sequence_length != 0:
            raise ValueError(
                f"Observation dimension ({total_obs_dim}) minus 2 is not cleanly divisible by sequence length ({sequence_length}). "
                f"Cannot infer features per timestep. Check environment observation space or config.")
        
        actual_features_per_timestep = expected_feature_dim // sequence_length 
        logger.info(f"Inferred actual features per timestep for TCN: {actual_features_per_timestep} (Obs dim: {total_obs_dim}, Seq len: {sequence_length})")
        # --- End inference ---

        # Pass TCN parameters and feature info to the policy
        policy_kwargs.update({
            "tcn_params": tcn_params,
            "sequence_length": sequence_length,
            "features_per_timestep": actual_features_per_timestep, # Use the inferred value
            "features_dim": total_obs_dim # <<< ADD THIS LINE >>>
        })
        # --- PATCH: Pass features list to policy_kwargs for TcnPolicy ---
        # This is redundant now as we pass num_features_per_timestep
        # if "features" in config:
        #     policy_kwargs["features"] = config["features"]
        
        model_kwargs.update({
            "policy": TcnPolicy,
            "learning_rate": lr_schedule,
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "n_epochs": config["n_epochs"],
            "ent_coef": float(config["ent_coef"]),
            "vf_coef": config["vf_coef"],
            "clip_range": config["clip_range"],
            "gae_lambda": config["gae_lambda"],
            "max_grad_norm": config["max_grad_norm"]
        })
        
        logger.info(f"TCN-PPO configuration: layers={tcn_params['num_layers']}, filters={tcn_params['num_filters']}, kernel={tcn_params['kernel_size']}, dropout={tcn_params['dropout']}, features/step={actual_features_per_timestep}")
        model = PPO(**model_kwargs)

    elif model_type == "tcn_sac":
        lr = config["learning_rate"]
        lr_schedule = linear_schedule(lr) if isinstance(lr, float) else lr

        # Extract TCN-specific parameters from config
        tcn_params = {
            "num_filters": config.get("tcn_num_filters", 64),
            "num_layers": config.get("tcn_num_layers", 4),
            "kernel_size": config.get("tcn_kernel_size", 3),
            "dropout": config.get("tcn_dropout", 0.2)
        }
        
        # Get sequence length from config
        sequence_length = config.get("sequence_length", 60)
        
        # --- Infer actual features per timestep ---
        # Updated to handle both gym and gymnasium Box spaces
        if not hasattr(env.observation_space, 'shape'):
            raise ValueError(f"TcnSacPolicy requires an observation space with a shape attribute, got {type(env.observation_space)}")
    
        obs_shape = env.observation_space.shape
        if len(obs_shape) != 1:
            raise ValueError(f"TcnSacPolicy requires a 1D observation space shape (features_dim,), got {obs_shape}")
        
        total_obs_dim = obs_shape[0]
        expected_feature_dim = total_obs_dim - 2 # Assuming 2 state vars
        if expected_feature_dim <= 0 or expected_feature_dim % sequence_length != 0:
            raise ValueError(
                f"Observation dimension ({total_obs_dim}) minus 2 is not cleanly divisible by sequence length ({sequence_length}). "
                f"Cannot infer features per timestep for TCN+SAC.")
        
        actual_features_per_timestep = expected_feature_dim // sequence_length 
        logger.info(f"Inferred actual features per timestep for TCN+SAC: {actual_features_per_timestep} (Obs dim: {total_obs_dim}, Seq len: {sequence_length})")
        # --- End inference ---

        # Pass TCN parameters and feature info to the policy
        # Update policy_kwargs with TCN details for TcnSacPolicy
        policy_kwargs.update({
            "tcn_params": tcn_params,
            "sequence_length": sequence_length,
            "features_per_timestep": actual_features_per_timestep,
            # Note: TcnSacPolicy does not take features_dim in its __init__
            # but the TcnExtractor it uses needs it implicitly.
            # The extractor calculates its output `_features_dim` correctly internally.
        })

        # Handle SAC specific ent_coef
        ent_coef_value = config.get("ent_coef", "auto")
        if isinstance(ent_coef_value, str) and ent_coef_value.lower() == 'auto':
            sac_ent_coef = 'auto'
        else:
            try: sac_ent_coef = float(ent_coef_value)
            except ValueError: sac_ent_coef = 'auto'; logger.warning(f"Invalid SAC ent_coef '{ent_coef_value}'. Defaulting auto.")

        # Update model_kwargs with SAC algorithm parameters
        model_kwargs.update({
            "policy": TcnSacPolicy, # Use the new policy
            "learning_rate": lr_schedule,
            "buffer_size": config.get("buffer_size", 1000000), # SAC default buffer size
            "batch_size": config.get("batch_size", 256), # SAC default batch size
            "learning_starts": config.get("learning_starts", 100), # SAC default learning starts
            "gradient_steps": config.get("gradient_steps", 1), # SAC default gradient steps
            "target_update_interval": config.get("target_update_interval", 1), # SAC default target update
            "tau": config.get("tau", 0.005), # SAC default tau
            "ent_coef": sac_ent_coef,
            "use_sde": config.get("use_sde", False), # Optional SAC param
            "sde_sample_freq": config.get("sde_sample_freq", -1) # Optional SAC param
        })

        logger.info(f"TCN-SAC configuration: layers={tcn_params['num_layers']}, filters={tcn_params['num_filters']}, kernel={tcn_params['kernel_size']}, dropout={tcn_params['dropout']}, features/step={actual_features_per_timestep}")
        model = SAC(**model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    policy_name = (model_kwargs['policy'] if isinstance(model_kwargs['policy'], str)
                   else model_kwargs['policy'].__name__)
    logger.info(f"Created {model_type.upper()} model with policy {policy_name}")
    return model


# --- Evaluation --- #

def evaluate_model(
    model: BaseAlgorithm, # Changed type hint
    env: gym.Env, # VecEnv expected
    config: Dict[str, Any], # Unused but kept for consistency
    n_episodes: int = 1,
    deterministic: bool = True,
) -> Tuple[float, np.ndarray, List[int], List[float]]:
    """
    Evaluate a model over n_episodes and return metrics.
    Expects env to be a VecEnv, preferably wrapped with Monitor.
    """
    if not hasattr(env, 'num_envs'):
        logger.warning("evaluate_model expects VecEnv; wrapping in DummyVecEnv")
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    all_episode_rewards, all_portfolio_values, all_actions, all_rewards = [], [], [], []
    current_rewards, current_lengths = np.zeros(n_envs), np.zeros(n_envs, dtype="int")
    current_portfolio_values = [[] for _ in range(n_envs)]
    current_actions, current_rewards_list = [[] for _ in range(n_envs)], [[] for _ in range(n_envs)]
    obs, episodes_completed = env.reset(), 0

    while episodes_completed < n_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, terminated, infos = env.step(action)
        truncated = np.array([info.get('TimeLimit.truncated', False) for info in infos])
        dones = np.logical_or(terminated, truncated)

        for i in range(n_envs):
            current_rewards[i] += rewards[i]
            current_lengths[i] += 1
            portfolio_value = infos[i].get("portfolio_value", 0.0)
            current_portfolio_values[i].append(portfolio_value)
            current_actions[i].append(int(action[i])) # Store integer action
            current_rewards_list[i].append(rewards[i])

            if dones[i]:
                if episodes_completed < n_episodes:
                    all_episode_rewards.append(current_rewards[i])
                    all_portfolio_values.extend(current_portfolio_values[i])
                    all_actions.extend(current_actions[i])
                    all_rewards.extend(current_rewards_list[i])
                    episodes_completed += 1
                current_rewards[i], current_lengths[i] = 0, 0
                current_portfolio_values[i], current_actions[i], current_rewards_list[i] = [], [], []

    mean_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0
    return (mean_reward, np.array(all_portfolio_values), all_actions, all_rewards)


def evaluate(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:  # Add args parameter
    """Evaluate a trained model."""
    eval_logger = logging.getLogger("rl_agent.evaluate")
    eval_logger.info("Starting evaluation...")

    # <<< Use args.load_model directly >>>
    model_path = args.load_model
    if not model_path or not os.path.exists(model_path):
        eval_logger.error(f"Model path not found or invalid: {model_path}")
        raise FileNotFoundError(f"Model path {model_path} required for evaluation.")
    eval_logger.info(f"Loading model from: {model_path}")

    # <<< Use args.test_data_path directly >>>
    test_data_path = args.test_data_path
    if not test_data_path:
        eval_logger.error("Test data path (--test_data_path) is required for evaluation.")
        raise ValueError("Test data path is required for evaluation.")
    eval_logger.info(f"Using test data: {test_data_path}")

    # <<< Use args.data_key directly >>>
    data_key = args.data_key

    # --- Environment Setup --- #
    # Create eval_env_config by extracting relevant keys from the main config
    env_keys = [
        "features", "sequence_length", "initial_balance", "commission",
        "reward_scaling", "window_size", "max_position", "max_steps",
        "random_start",
        # Reward component keys
        "portfolio_change_weight", "drawdown_penalty_weight", "sharpe_reward_weight",
        "fee_penalty_weight", "benchmark_reward_weight", # Removed "consistency_penalty_weight",
        "idle_penalty_weight", "profit_bonus_weight", "exploration_bonus_weight",
        "sharpe_window", # Removed "consistency_threshold",
        "idle_threshold",
        "exploration_start", "exploration_end", "exploration_decay_rate",
        "trade_penalty_weight"
    ]
    eval_env_config = {k: config[k] for k in env_keys if k in config}
    
    # Override data path and key for evaluation
    eval_env_config["data_path"] = test_data_path
    eval_env_config["data_key"] = data_key
    # Use a distinct seed for evaluation if available, otherwise None
    eval_seed = config.get("seed") + 1000 if config.get("seed") is not None else None
    eval_env_config["seed"] = eval_seed # Add seed to the config for create_env

    eval_logger.info(f"Creating evaluation environment with data: {test_data_path}")
    # Create a single environment instance directly for evaluation
    try:
        single_eval_env = create_env(config=eval_env_config, is_eval=True)
        # Wrap with Monitor to capture episode stats
        # Use the trial's log_dir for monitor files
        monitor_log_path = os.path.join(os.path.dirname(model_path), f'monitor_eval.csv')
        ensure_dir_exists(os.path.dirname(monitor_log_path))
        single_eval_env = Monitor(single_eval_env, filename=monitor_log_path)
        # Wrap with DummyVecEnv to make it a VecEnv
        test_env = DummyVecEnv([lambda: single_eval_env])
        eval_logger.info("Evaluation environment created successfully.")
    except Exception as e:
        eval_logger.error(f"Failed to create evaluation environment: {e}", exc_info=True)
        raise

    # Apply VecNormalize if stats file exists
    potential_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(potential_stats_path):
        eval_logger.info(f"Loading VecNorm stats: {potential_stats_path}")
        test_env = VecNormalize.load(potential_stats_path, test_env)
        test_env.training = False; test_env.norm_reward = False
        eval_logger.info("VecNormalize applied for evaluation.")
    else:
        eval_logger.info("No VecNormalize stats file found, using raw environment.")

    # --- Load Model --- #
    model_type_str = config.get("model_type", "ppo").lower()
    # Define the mapping from model type string to class
    model_cls_map = {
        "dqn": DQN,
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "lstm_dqn": DQN, # LSTM handled by feature extractor
        "qrdqn": QRDQN,
        "recurrentppo": RecurrentPPO,
        "tcn_ppo": PPO,  # TCN is handled by TcnPolicy
        "tcn_sac": SAC  # Added for tcn_sac
    }
    if model_type_str not in model_cls_map:
        eval_logger.error(f"Unknown model type '{model_type_str}' specified in config.")
        raise ValueError(f"Unknown model type: {model_type_str}")
    ModelClass = model_cls_map[model_type_str]
    
    try:
        model = ModelClass.load(model_path, env=test_env, device=config.get("device", "auto"))
        eval_logger.info(f"Model {model_path} loaded successfully.")
    except Exception as e:
        eval_logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

    # --- Run Evaluation --- #
    # Prioritize CLI arg for n_eval_episodes, then config, then default
    n_eval = args.n_eval_episodes if args.n_eval_episodes is not None else config.get("n_eval_episodes", 5)
    eval_logger.info(f"Evaluating model for {n_eval} episodes...")
    # Call the existing evaluate_model function
    mean_reward, portfolio_values, actions, rewards = evaluate_model(
        model=model, env=test_env, config=config, n_episodes=n_eval, deterministic=True
    )
    eval_logger.info(f"Evaluation complete. Mean reward: {mean_reward:.4f}")

    # --- Calculate Metrics --- #
    initial_balance = config.get("initial_balance", 10000)
    # Ensure portfolio_values is not empty and get the last value
    # Corrected condition: Check if the numpy array is not empty using .size
    final_value = portfolio_values[-1] if portfolio_values.size > 0 else initial_balance
    total_return = (final_value / initial_balance) - 1 if initial_balance > 0 else 0.0

    metrics = {
        "mean_reward": mean_reward,
        "final_portfolio_value": final_value,
        "total_return": total_return,
        "n_eval_episodes": n_eval
    }

    # Flatten portfolio values for metric calculation
    # portfolio_values is already a flat NumPy array from evaluate_model
    flat_portfolio_values = portfolio_values

    if len(flat_portfolio_values) > 10: # Need enough data points
        try:
            # <<< Add debug print here >>>
            print(f"DEBUG: Type before calculate_trading_metrics: {type(flat_portfolio_values)}")
            print(f"DEBUG: Shape before calculate_trading_metrics: {flat_portfolio_values.shape}")
            print(f"DEBUG: First 10 elements: {flat_portfolio_values[:10]}")
            # <<< End debug print >>>
            trading_metrics = calculate_trading_metrics(flat_portfolio_values)
            metrics.update(trading_metrics)

            # --- Format metrics into a markdown table --- #
            metrics_table = "| Metric                | Value    |\n"
            metrics_table += "| :-------------------- | :------- |\n"
            for key, value in trading_metrics.items():
                try:
                    # Format float values, handle potential non-float values gracefully
                    metrics_table += f"| {key:<21} | {float(value):<8.4f} |\n"
                except (ValueError, TypeError):
                    metrics_table += f"| {key:<21} | {str(value):<8} |\n"
            eval_logger.info(f"Trading Metrics:\n{metrics_table}")
            # -------------------------------------------- #

        except Exception as e:
            eval_logger.warning(f"Could not calculate trading metrics: {e}")

    # --- Generate Plots --- #
    if config.get("generate_plots", True) and len(flat_portfolio_values) > 1:
        plot_dir = os.path.join(os.path.dirname(model_path), "evaluation_plots")
        ensure_dir_exists(plot_dir)
        try:
            create_evaluation_plots(
                portfolio_values=flat_portfolio_values,
                actions=actions, # Pass flattened/combined actions if needed
                rewards=rewards, # Pass flattened/combined rewards if needed
                save_path=plot_dir
            )
            eval_logger.info(f"Evaluation plots saved to: {plot_dir}")
        except Exception as e:
            eval_logger.warning(f"Could not generate evaluation plots: {e}")

    test_env.close()
    eval_logger.info("Evaluation finished.")
    return metrics


# --- Training --- #

def train(config: Dict[str, Any]) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    """
    Train a reinforcement learning agent based on config.
    """
    log_path = os.path.join(config["log_dir"], config["model_name"])
    ensure_dir_exists(log_path)
    # Determine console level based on verbose setting
    console_log_level = logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO
    # Always set file level to DEBUG --> Change to INFO
    file_log_level = logging.INFO # Changed from DEBUG 
    setup_logger(log_dir=log_path, log_level=file_log_level, console_level=console_log_level)
    train_logger = logging.getLogger("rl_agent")
    
    # <<< Ensure environment logger inherits the DEBUG level --> Change to INFO >>>
    logging.getLogger("rl_agent.environment").setLevel(logging.INFO) # Changed from DEBUG
    # <<< End change >>>

    sb3_log_path = os.path.join(log_path, "sb3_logs")
    ensure_dir_exists(sb3_log_path)
    sb3_logger_instance = setup_sb3_logger(log_dir=sb3_log_path)
    save_config(config=config, log_dir=log_path, filename="config.json")
    if config.get("seed") is not None: set_seeds(config["seed"])

    train_logger.info(f"Starting training for model: {config['model_name']}")
    device = ("cpu" if config["cpu_only"] else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_logger.debug(f"Training on device: {device}") # Changed from info
    check_resources(train_logger)

    # --- Environment Creation --- #
    # --- Restore: Allow num_envs from config for non-tune runs ---
    # is_tune_run = RAY_AVAILABLE and check_ray_session()
    # if not is_tune_run:
    #     if config.get("num_envs", 1) != 1:
    #         train_logger.warning(f"Overriding num_envs from config ({config.get('num_envs')}) to 1 for standalone run.")
    #     config["num_envs"] = 1 # Force 1 env for standalone runs
    num_envs = config.get("num_envs", 1) # Get num_envs from config (or default to 1)
    # -----------------------------------------------------------
    train_logger.info(f"Creating {num_envs} parallel environment(s)...")
    base_seed = config.get("seed")
    if base_seed is None:
        base_seed = int(time.time()) # Generate a default seed if missing
        train_logger.warning(f"Seed not found in config, using default seed: {base_seed}")
        config["seed"] = base_seed # Optionally update config dict
        set_seeds(base_seed) # Set seed explicitly if it was missing

    def make_single_train_env(rank: int, base_seed_val: Optional[int]):
        def _init():
            env_config = config.copy()
            instance_seed = base_seed_val
            if instance_seed is not None:
                 instance_seed += rank
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=False)
            monitor_log = os.path.join(log_path, f'monitor_train_{rank}.csv')
            os.makedirs(os.path.dirname(monitor_log), exist_ok=True)
            env = Monitor(env, filename=monitor_log)
            return env
        return _init

    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        env_id=make_single_train_env(rank=0, base_seed_val=base_seed), # Pass the potentially generated base_seed
        n_envs=num_envs, seed=None, vec_env_cls=vec_env_cls, env_kwargs=None
    )

    # --- VecNormalize Setup --- #
    vec_normalize_stats_path = None
    if config.get("load_model"):
        potential_stats = config["load_model"].replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(potential_stats):
            vec_normalize_stats_path = potential_stats
            train_logger.info(f"Found VecNorm stats: {vec_normalize_stats_path}")

    if vec_normalize_stats_path:
        train_logger.info(f"Loading VecNorm stats: {vec_normalize_stats_path}")
        train_env = VecNormalize.load(vec_normalize_stats_path, train_env)
        train_env.training = True
    else:
        norm_obs_setting = config.get("norm_obs", "auto").lower()
        if norm_obs_setting == "auto":
            features = config.get("features", [])
            if isinstance(features, str): features = features.split(",")
            has_scaled = any("_scaled" in f for f in features)
            should_norm_obs = not has_scaled
        else: should_norm_obs = norm_obs_setting == "true"
        train_logger.info(f"Applying VecNorm: norm_obs={should_norm_obs}")
        train_env = VecNormalize(
            train_env, norm_obs=should_norm_obs, norm_reward=False,
            clip_obs=10., gamma=config["gamma"]
        )

    # --- Validation Env Setup --- #
    eval_env = None
    if config.get("val_data_path"):
        train_logger.info(f"Creating validation env: {config['val_data_path']}")
        eval_env_seed_val = base_seed + num_envs if base_seed is not None else None
        # Step 1: Create the base DummyVecEnv for evaluation
        raw_eval_env = make_vec_env(
            env_id=make_single_train_env(rank=0, base_seed_val=eval_env_seed_val),
            n_envs=1, seed=None, vec_env_cls=DummyVecEnv, env_kwargs=None
        )

        # Step 2: Apply VecNormalize wrapper, mirroring train_env setup
        if vec_normalize_stats_path:
            train_logger.info("Applying loaded VecNorm stats to eval_env.")
            eval_env = VecNormalize.load(vec_normalize_stats_path, raw_eval_env)
            # Ensure it's set to inference mode
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            train_logger.info("Applying new VecNorm wrapper to eval_env.")
            # Determine norm_obs setting based on config (same logic as for train_env)
            norm_obs_setting = config.get("norm_obs", "auto").lower()
            if norm_obs_setting == "auto":
                features = config.get("features", [])
                if isinstance(features, str): features = features.split(",")
                has_scaled = any("_scaled" in f for f in features)
                should_norm_obs = not has_scaled
            else: should_norm_obs = norm_obs_setting == "true"
            train_logger.info(f"Eval VecNorm: norm_obs={should_norm_obs}")
            eval_env = VecNormalize(
                raw_eval_env, # Wrap the raw eval env
                norm_obs=should_norm_obs,
                norm_reward=False, # Never normalize rewards for eval
                clip_obs=10.,
                gamma=config["gamma"],
                training=False # Set to False for evaluation
            )
        train_logger.info(f"Validation env wrapped with VecNormalize: {eval_env}")

    # --- Model Creation --- #
    if config.get("load_model"):
        load_path = config["load_model"]
        train_logger.info(f"Loading model from: {load_path}")
        if not os.path.exists(load_path):
            train_logger.error(f"Model path not found: {load_path}"); sys.exit(1)
        model_cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
                     "lstm_dqn": DQN, "qrdqn": QRDQN, "recurrentppo": RecurrentPPO}
        model = model_cls[config["model_type"]].load(load_path, env=train_env)
        new_lr = config.get('learning_rate')
        if new_lr is not None and hasattr(model.policy, 'optimizer') and model.policy.optimizer:
            train_logger.info(f"Overriding loaded LR to: {new_lr}")
            for pg in model.policy.optimizer.param_groups: pg['lr'] = new_lr
    else:
        train_logger.info(f"Creating new {config['model_type']} model")
        model = create_model(env=train_env, config=config)
    model.set_logger(sb3_logger_instance)

    # --- Callbacks --- #
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["model_name"])
    ensure_dir_exists(checkpoint_dir)
    callbacks = get_callback_list(
        eval_env=eval_env, log_dir=log_path,
        eval_freq=max(config.get("eval_freq", 10000), 5000),
        n_eval_episodes=config["n_eval_episodes"],
        save_freq=config.get("save_freq", 10000),
        keep_checkpoints=config.get("keep_checkpoints", 3),
        resource_check_freq=config.get("resource_check_freq", 1000),
        metrics_log_freq=config.get("metrics_log_freq", 1000),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        checkpoint_save_path=checkpoint_dir,
        model_name=config["model_type"],
        # REMOVE TuneReportCallback entirely from standard train function
        custom_callbacks=[],
        curriculum_duration_fraction=0.0
    )

    # --- Training --- #
    train_logger.info(f"Starting training: {config['total_timesteps']} steps")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=250,  # Log SB3 stats every 250 updates
            reset_num_timesteps=not config.get("load_model")
        )
    except Exception as e:
        train_logger.critical(f"Training failed: {e}", exc_info=True)
        error_path = os.path.join(log_path, "model_on_error.zip")
        try: model.save(error_path); train_logger.info(f"Saved error model: {error_path}")
        except Exception as se: train_logger.error(f"Could not save error model: {se}")
        if 'train_env' in locals() and hasattr(train_env, 'close'): train_env.close()
        if 'eval_env' in locals() and eval_env is not None and hasattr(eval_env, 'close'): eval_env.close()
        sys.exit(1)
    finally:
        if 'train_env' in locals() and hasattr(train_env, 'close'): train_env.close()
        if 'eval_env' in locals() and eval_env is not None and hasattr(eval_env, 'close'): eval_env.close()

    training_time = time.time() - training_start_time
    train_logger.info(f"Training finished in {training_time:.2f}s.")

    final_model_path = os.path.join(log_path, "final_model.zip")
    model.save(final_model_path)
    train_logger.info(f"Final model saved: {final_model_path}")
    if isinstance(train_env, VecNormalize):
        stats_path = final_model_path.replace(".zip", "_vecnormalize.pkl")
        train_env.save(stats_path)
        train_logger.info(f"VecNorm stats saved: {stats_path}")

    # --- Final Metrics --- #
    final_metrics = {"training_time": training_time, "total_timesteps": getattr(model, 'num_timesteps', 0),
                     "model_type": config["model_type"], "eval/mean_reward": 0.0,
                     "eval/explained_variance": 0.0, "combined_score": 0.0} # Default renamed score
    if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
        log_vals = model.logger.name_to_value
        final_reward = log_vals.get("rollout/ep_rew_mean", 0.0)
        final_variance = log_vals.get("train/explained_variance", None) # Allow None
        final_sharpe = log_vals.get("trading/sharpe_ratio", None)
        final_calmar = log_vals.get("trading/calmar_ratio", None)
        final_sortino = log_vals.get("trading/sortino_ratio", None)
        final_return = log_vals.get("trading/portfolio_return", None)
        final_max_dd = log_vals.get("trading/max_drawdown", None)
        final_win_rate = log_vals.get("trading/win_rate", None)

        # Extract final losses if available
        final_actor_loss, final_critic_loss = None, None
        model_type = config.get("model_type", "ppo")
        if model_type == "sac":
            final_actor_loss = log_vals.get("train/actor_loss", log_vals.get("train/policy_loss"))
            final_critic_loss = log_vals.get("train/critic_loss")
        elif model_type in ["ppo", "a2c", "tcn_ppo", "recurrentppo"]: # Include recurrentppo
            final_actor_loss = log_vals.get("train/policy_loss")
            final_critic_loss = log_vals.get("train/value_loss")

        try: final_reward = float(final_reward)
        except (ValueError, TypeError, AttributeError): final_reward = 0.0
        try: final_variance = float(final_variance) if final_variance is not None else None
        except (ValueError, TypeError, AttributeError): final_variance = None
        # Convert losses safely
        try: final_actor_loss = float(final_actor_loss) if final_actor_loss is not None else None
        except (ValueError, TypeError, AttributeError): final_actor_loss = None
        try: final_critic_loss = float(final_critic_loss) if final_critic_loss is not None else None
        except (ValueError, TypeError, AttributeError): final_critic_loss = None

        final_metrics["eval/mean_reward"] = final_reward
        if model_type != "sac" and final_variance is not None:
            final_metrics["eval/explained_variance"] = final_variance

        # Call the standalone helper function
        final_combined = normalize_and_combine_metrics(
            reward=final_reward,
            explained_variance=final_variance,
            sharpe_ratio=final_sharpe,
            episode_return=final_return,
            calmar_ratio=final_calmar,
            sortino_ratio=final_sortino,
            actor_loss=final_actor_loss,
            critic_loss=final_critic_loss,
            model_type=model_type,
            max_drawdown=final_max_dd,
            win_rate=final_win_rate
        )
        final_metrics["combined_score"] = float(final_combined if final_combined is not None and np.isfinite(final_combined) else 0.0)

    # Use the modern check for Ray Tune session
    session_active = False
    if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
        session_active = ray.air.session.is_active()
        
    if session_active:
        try: 
            # Use modern ray.air.session.report API
            if hasattr(ray, "air") and hasattr(ray.air, "session"):
                ray.air.session.report(final_metrics) 
                train_logger.info("Reported final metrics via Ray AIR session")
            # Fallback to tune.report (passing dict)
            elif hasattr(tune, "report"):
                tune.report(final_metrics)
                train_logger.info("Reported final metrics via tune.report (fallback)")
            else:
                 train_logger.warning("Could not find ray.air.session.report or tune.report to report final metrics.")

        except Exception as re: 
            train_logger.warning(f"Failed final Ray AIR session report: {re}")

    return model, final_metrics


# --- Helper function for metric combination --- #

def normalize_and_combine_metrics(
    reward: float,
    explained_variance: Optional[float],
    sharpe_ratio: Optional[float],
    episode_return: Optional[float],
    calmar_ratio: Optional[float],
    sortino_ratio: Optional[float],
    actor_loss: Optional[float],
    critic_loss: Optional[float],
    model_type: str,
    max_drawdown: Optional[float] = None, # Added optional max_drawdown
    win_rate: Optional[float] = None, # Added optional win_rate
) -> float:
    """
    Normalizes and combines various performance metrics into a single score.

    Args:
        reward: Mean episode reward.
        explained_variance: Explained variance (if applicable).
        sharpe_ratio: Sharpe ratio.
        episode_return: Total portfolio return for the episode.
        calmar_ratio: Calmar ratio.
        sortino_ratio: Sortino ratio.
        actor_loss: Actor/Policy loss (if applicable).
        critic_loss: Critic/Value loss (if applicable).
        model_type: The type of model ('ppo', 'sac', etc.) used for normalization logic.
        max_drawdown: Maximum drawdown (optional).
        win_rate: Win rate (optional).

    Returns:
        A combined score between 0 and 1.
    """
    callback_logger = logging.getLogger("rl_agent.metrics") # Use a specific logger
    weights = {
        "reward": 0.25,
        "sharpe": 0.20,
        "calmar": 0.10,
        "sortino": 0.10,
        "return": 0.10,
        "variance": 0.05, # Lower weight unless crucial
        "losses": 0.10, # Penalize high losses
        "drawdown": 0.05, # Added drawdown penalty
        "win_rate": 0.05, # Added win rate bonus
    }
    # Ensure weights sum to 1 (adjust if needed)
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
         callback_logger.warning(f"Metric weights sum to {total_weight:.2f}, not 1.0. Normalizing.")
         weights = {k: v / total_weight for k, v in weights.items()}


    score = 0.0

    # --- Normalize and Add Base Reward ---
    # Assuming reward is roughly centered around 0, scaling up slightly
    norm_reward = np.tanh(reward * 0.5) # tanh maps to [-1, 1], centered at 0
    score += weights["reward"] * (norm_reward + 1) / 2 # Scale to [0, 1]

    # --- Normalize and Add Financial Ratios ---
    if sharpe_ratio is not None and np.isfinite(sharpe_ratio):
        # Sharpe: tanh maps roughly [-3, 3] to [-1, 1]. Good sharpe > 1.
        norm_sharpe = np.tanh(sharpe_ratio / 2.0)
        score += weights["sharpe"] * (norm_sharpe + 1) / 2 # Scale to [0, 1]

    if calmar_ratio is not None and np.isfinite(calmar_ratio):
         # Calmar: Higher is better. Tanh maps [0, 5] roughly to [0, 1]
         norm_calmar = np.tanh(calmar_ratio / 2.5)
         score += weights["calmar"] * max(0, norm_calmar) # Ensure non-negative contribution

    if sortino_ratio is not None and np.isfinite(sortino_ratio):
        # Sortino: Higher is better. Tanh maps [0, 6] roughly to [0, 1]
        norm_sortino = np.tanh(sortino_ratio / 3.0)
        score += weights["sortino"] * max(0, norm_sortino)

    if episode_return is not None and np.isfinite(episode_return):
         # Return: Map percentage return (e.g., -0.5 to 0.5) to [0, 1]
         norm_return = np.tanh(episode_return * 2.0) # Scale input sensitivity
         score += weights["return"] * (norm_return + 1) / 2

    # --- Normalize and Add Explained Variance (Non-SAC) ---
    # Only add variance score if it's relevant and provided
    if model_type != "sac" and explained_variance is not None and np.isfinite(explained_variance):
        # Explained variance is typically [0, 1] or slightly negative
        norm_variance = max(0, explained_variance) # Clip negative values
        score += weights["variance"] * norm_variance

    # --- Penalize High Losses ---
    # We want *low* losses. Higher loss = lower score contribution.
    norm_actor_loss, norm_critic_loss = 0.0, 0.0
    if actor_loss is not None and np.isfinite(actor_loss):
        # Assuming losses are non-negative. Lower is better. Use exp(-loss) -> range (0, 1]
        norm_actor_loss = np.exp(-abs(actor_loss) * 0.5) # Adjust multiplier for sensitivity
    if critic_loss is not None and np.isfinite(critic_loss):
        norm_critic_loss = np.exp(-abs(critic_loss) * 0.5)

    # Average the loss scores (if both exist, otherwise use the one available)
    if actor_loss is not None and critic_loss is not None:
        loss_score = (norm_actor_loss + norm_critic_loss) / 2
    elif actor_loss is not None:
        loss_score = norm_actor_loss
    elif critic_loss is not None:
        loss_score = norm_critic_loss
    else:
        loss_score = 0.5 # Neutral score if no losses provided

    score += weights["losses"] * loss_score

    # --- Add Max Drawdown Penalty ---
    if max_drawdown is not None and np.isfinite(max_drawdown):
         # Max drawdown is [0, 1], lower is better. Score = (1 - drawdown)
         drawdown_score = 1.0 - max(0, min(1, max_drawdown)) # Ensure it's in [0, 1]
         score += weights["drawdown"] * drawdown_score

    # --- Add Win Rate Bonus ---
    if win_rate is not None and np.isfinite(win_rate):
         # Win rate is [0, 1], higher is better.
         win_rate_score = max(0, min(1, win_rate)) # Ensure it's in [0, 1]
         score += weights["win_rate"] * win_rate_score

    # --- Final Clipping ---
    final_score = np.clip(score, 0.0, 1.0)
    # callback_logger.debug(f"Combined Score: {final_score:.4f} (R:{norm_reward:.2f}, S:{norm_sharpe:.2f}, P/V:{loss_score:.2f})")
    return float(final_score)


# --- Main Execution --- #

def main():
    print(f"Raw sys.argv: {sys.argv}") # KEEP THIS CHECK
    args = parse_args()
    print(f"Parsed args: {args}") # KEEP THIS CHECK
    print(f"Value of args.eval_only AFTER parse_args: {args.eval_only}") # KEEP THIS CHECK

    # Initialize config from args FIRST
    config = args_to_config(args)
    print(f"Initial config from args: {config}") # Optional debug print

    # --- Config Loading --- #
    if args.load_config is not None:
        # Load config from file if path provided
        config_path = os.path.abspath(os.path.expanduser(args.load_config))
        if os.path.exists(config_path):
            print(f"Loading configuration from: {config_path}") # CORRECTED PRINT
            file_config = load_config(config_path)
            # Update config with file values first
            config.update(file_config)
            print(f"Config updated with values from {config_path}") # CORRECTED PRINT

    # Ensure features are a list (might be redundant now, but safe)
    if 'features' in config and isinstance(config['features'], str):
        config['features'] = config['features'].split(',')
    elif 'features' in config and isinstance(config['features'], list):
         # Ensure items are strings if read from JSON as list
         config['features'] = [str(f) for f in config['features']]

    # Override with CLI arguments
    config.update(vars(args))

    # --- Directory Setup --- #
    log_base_dir, model_name = config["log_dir"], config["model_name"]
    ckpt_base_dir = config["checkpoint_dir"]
    ensure_dir_exists(log_base_dir); ensure_dir_exists(ckpt_base_dir)
    ensure_dir_exists(os.path.join(log_base_dir, model_name))
    ensure_dir_exists(os.path.join(ckpt_base_dir, model_name))

    # <<< REVISED CLI OVERRIDES >>>
    # Apply CLI overrides *after* loading config and *before* mode selection
    print("\nApplying final CLI overrides for run control parameters...")
    if args.total_timesteps is not None:
         config['total_timesteps'] = args.total_timesteps
         print(f"  Overriding total_timesteps -> {config['total_timesteps']}")
    if args.eval_freq is not None:
         config['eval_freq'] = args.eval_freq
         print(f"  Overriding eval_freq -> {config['eval_freq']}")
    # eval_only override removed - will use args.eval_only directly
    print("--- End CLI Overrides ---\n")
    # <<< END REVISED CLI OVERRIDES >>>

    # --- Mode Selection --- #
    # REMOVED: print("\n--- Debugging Mode Selection ---")
    # REMOVED: print(f"Value of config['eval_only'] before check: {config.get('eval_only')}")
    # REMOVED: print("--- End Debugging ---")

    # <<< Add one more check right before the condition >>>
    print(f"Value of args.eval_only JUST BEFORE 'if': {args.eval_only}") # KEEP THIS CHECK

    # <<< Use args.eval_only directly for mode selection >>>
    if args.eval_only:
        print("Running in Evaluation-Only Mode")
        # <<< Pass args to evaluate function >>>
        evaluate(config, args)
    else:
        print("Running in Training Mode")
        # Check if Ray Tune is being used (using the modern check)
        session_active = False
        if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
            session_active = ray.air.session.is_active()
            
        if session_active:
            print("Detected Ray Tune session (via AIR). Running train_rl_agent_tune.")
            # Call the trainable directly when run via Tune
            train_rl_agent_tune(config)
        else:
            print("No Ray Tune session detected. Running standard train.")
            train(config)


if __name__ == "__main__":
    main() 