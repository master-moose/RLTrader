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
import traceback  # Added for potential use
from typing import Any, Dict, List, Optional, Tuple, Callable

# --- Third-Party Imports --- #
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

# --- Stable Baselines3 Imports --- #
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm as BaseRLModel
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy  # For PPO/A2C
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize
)
from stable_baselines3.dqn.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines3.sac.policies import MlpPolicy as SacMlpPolicy
from stable_baselines3.common.callbacks import BaseCallback

# --- SB3 Contrib Imports --- #
from sb3_contrib import QRDQN, RecurrentPPO

# --- Ray Imports (with availability check) --- #
RAY_AVAILABLE = False
try:
    import ray
    from ray import tune
    # Unused imports removed: ASHAScheduler, OptunaSearch, HyperOptSearch
    from ray.tune import CLIReporter  # Keep CLIReporter
    RAY_AVAILABLE = True
except ImportError:
    # Detailed error handling in run_tune_sweep.py
    pass

# --- Local Module Imports --- #
# Add parent directory to path *before* attempting local imports
# Ensure this runs only once or is handled carefully if script is imported
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.append(_parent_dir)

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

# Initialize logger globally (will be configured later)
# Use specific logger name instead of __name__ for consistency
logger = logging.getLogger("rl_agent")


# --- Ray Tune Callback Class --- #

class TuneReportCallback(BaseCallback):
    """Callback to report metrics to Ray Tune, focusing on scheduler needs."""
    def __init__(self):
        super().__init__(verbose=0)
        self.last_explained_variance = 0.0 # Store last known variance

    def _on_init(self) -> None:
        """Ensure the logger is available."""
        callback_logger = logging.getLogger("rl_agent")
        if self.logger is None:
            callback_logger.warning(
                "SB3 logger not available in TuneReportCallback logger attr."
            )
        callback_logger.info(
            "TuneReportCallback initialized - reporting metrics at rollout end."
        )

    def _normalize_and_combine_metrics(self, reward, explained_variance):
        """
        Create a normalized combined score weighting reward and variance.

        Args:
            reward: The mean reward value.
            explained_variance: Explained variance, typically between -1 and 1.

        Returns:
            A combined normalized score between 0 and 1 (higher is better).
        """
        if reward is None or not isinstance(reward, (int, float, np.number)):
            return 0.5  # Neutral score if reward is missing
        if explained_variance is None \
           or not isinstance(explained_variance, (int, float, np.number)):
            explained_variance = 0.0  # Default variance if missing

        # Normalize reward using tanh
        normalized_reward = np.tanh(reward / 1000.0)
        # Clip variance
        normalized_variance = np.clip(explained_variance, -1.0, 1.0)
        # Combine scores: 70% reward, 30% variance, shifted to [0, 1]
        combined_score = (0.7 * normalized_reward + 0.3 * normalized_variance + 1.0) / 2.0
        return combined_score

    def _on_step(self) -> bool:
        """
        Update the last known explained variance after each training step.
        Try fetching directly from the logger's internal dictionary.
        """
        # Try fetching from logger first, as it might be updated later in step
        if self.logger is not None and hasattr(self.logger, 'name_to_value'):
            # Use the name_to_value dictionary with .get() for safety
            logged_variance = self.logger.name_to_value.get("train/explained_variance", None)
            if logged_variance is not None:
                try:
                    self.last_explained_variance = float(logged_variance)
                    # callback_logger.debug(f"Stored variance {self.last_explained_variance:.4f} from logger dict in _on_step")
                    return True # Found it in logger, no need to check locals
                except (ValueError, TypeError):
                    pass # Ignore conversion errors from logger value

        # Fallback: Try to get explained_variance from locals
        if hasattr(self, 'locals') and self.locals:
            possible_keys = ["explained_variance", "train/explained_variance"]
            for key in possible_keys:
                if key in self.locals:
                    try:
                        self.last_explained_variance = float(self.locals[key])
                        # callback_logger.debug(f"Stored variance {self.last_explained_variance:.4f} from locals key '{key}' in _on_step")
                        break 
                    except (ValueError, TypeError, KeyError):
                        pass # Ignore conversion errors from locals value
        return True

    def _on_rollout_end(self) -> None:
        """
        Report metrics at the end of each rollout.
        Uses ep_rew_mean from ep_info_buffer and fetches the most recent
        explained_variance logged by SB3.
        """
        callback_logger = logging.getLogger("rl_agent")
        callback_logger.debug(
            f"Rollout end at step {self.num_timesteps}. Attempting report."
        )
        reward_value = 0.0  # Default reward

        # --- Get Mean Reward from Monitor Buffer ---
        # (Keep existing reward fetching logic)
        if hasattr(self.model, "ep_info_buffer") and \
           len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                reward_value = float(np.mean(ep_rewards))
                callback_logger.debug(
                    f"Mean reward ({len(ep_rewards)} eps): {reward_value:.4f}"
                )
        else:
            callback_logger.debug(
                f"No finished eps in rollout (step {self.num_timesteps}). R=0.0"
            )


        # --- Use the explained variance stored from _on_step --- #
        variance_value = self.last_explained_variance # Use the value stored in _on_step
        callback_logger.debug(f"Using stored variance from _on_step: {variance_value:.4f}")


        # --- Prepare and Report Metrics ---
        combined_score = self._normalize_and_combine_metrics(reward_value, variance_value)

        metrics_to_report = {
            "eval/mean_reward": reward_value,
            "eval/explained_variance": variance_value, # Report the fetched variance
            "eval/combined_score": combined_score,
            "timesteps": self.num_timesteps,
        }
        callback_logger.debug(
             f"Reporting (step {self.num_timesteps}): "
             f"reward={reward_value:.4f}, var={variance_value:.4f}, "
             f"combined={combined_score:.4f}"
         )

        # Report to Ray Tune
        if RAY_AVAILABLE and tune.is_session_enabled():
            try:
                tune.report(**metrics_to_report)
                callback_logger.debug(
                    f"Reported to Ray Tune at step {self.num_timesteps}"
                )
            except Exception as e:
                callback_logger.error(
                    f"Error reporting to Ray Tune at step {self.num_timesteps}: {e}"
                )
        else:
             callback_logger.debug("Ray Tune session not enabled, skipping report.")

# --- Ray Tune Trainable Function --- #

def train_rl_agent_tune(config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable function for training an RL agent.
    
    Args:
        config: Dictionary of hyperparameters from Ray Tune
    """
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

    # --- Logger Setup ---
    log_level_to_set = logging.DEBUG if config.get("verbose", 1) >= 2 \
        else logging.INFO
    setup_logger(
        log_dir=log_dir,
        log_level=log_level_to_set,
        console_level=log_level_to_set
    )
    trial_logger = logging.getLogger("rl_agent")

    # --- Initial Reporting and Seeding ---
    initial_metrics = {
        "timesteps": 0,
        "training_iteration": 0,
        "eval/mean_reward": 0.0,
        "eval/explained_variance": 0.0,
        "eval/combined_score": 0.5,
        "config": train_config
    }
    if RAY_AVAILABLE and tune.is_session_enabled():
        trial_logger.info("Reporting initial metrics to Ray Tune")
        try:
            tune.report(**initial_metrics)
        except Exception as report_err:
            trial_logger.warning(f"Initial report failed: {report_err}")

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
    num_envs = train_config.get("num_envs", 8)
    n_steps = train_config.get("n_steps", 2048)
    total_steps_per_rollout = n_steps * num_envs
    if "batch_size" not in train_config:
        train_config["batch_size"] = min(total_steps_per_rollout, 4096)
        trial_logger.info(
            f"Setting batch_size={train_config['batch_size']} "
            f"(n_steps={n_steps}, num_envs={num_envs})"
        )
    else:
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

    # --- Environment Creation --- #
    trial_logger.info(f"Creating {num_envs} parallel environment(s)...")

    def make_single_env(rank: int, base_seed: Optional[int]):
        def _init():
            env_config = train_config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=False)
            monitor_log_path = os.path.join(log_dir, f'monitor_{rank}.csv')
            ensure_dir_exists(os.path.dirname(monitor_log_path))
            env = Monitor(env, filename=monitor_log_path)
            return env
        return _init

    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=seed),
        n_envs=num_envs,
        seed=None,
        vec_env_cls=vec_env_cls,
        env_kwargs=None
    )

    # --- VecNormalize Setup ---
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

    # --- Validation Environment Setup ---
    eval_env = None
    if train_config.get("val_data_path"):
        trial_logger.info(f"Creating validation env: {train_config['val_data_path']}")
        eval_env = make_vec_env(
            env_id=make_single_env(rank=0, base_seed=seed + num_envs),
            n_envs=1,
            seed=None,
            vec_env_cls=DummyVecEnv,
            env_kwargs=None
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=should_norm_obs,
            norm_reward=False,
            clip_obs=10.,
            gamma=train_config["gamma"],
            training=False
        )

    # --- Model Creation --- #
    trial_logger.info(f"Creating new {train_config['model_type']} model")
    model = create_model(env=train_env, config=train_config)
    model.set_logger(sb3_logger_instance)

    # --- Callbacks Setup ---
    rollout_steps = n_steps * num_envs
    base_eval_freq = train_config.get("eval_freq", 10000)
    effective_eval_freq = max(base_eval_freq, rollout_steps)
    trial_logger.info(f"Effective eval_freq = {effective_eval_freq}")

    tune_early_stopping_patience = 0 # Disabled for debugging
    trial_logger.info(f"Tune early_stopping_patience = {tune_early_stopping_patience}")

    callbacks = get_callback_list(
        eval_env=eval_env,
        log_dir=log_dir,
        eval_freq=effective_eval_freq,
        n_eval_episodes=train_config.get("n_eval_episodes", 5),
        save_freq=train_config.get("save_freq", 50000),
        keep_checkpoints=train_config.get("keep_checkpoints", 3),
        resource_check_freq=train_config.get("resource_check_freq", 5000),
        metrics_log_freq=train_config.get("metrics_log_freq", 1000),
        early_stopping_patience=tune_early_stopping_patience,
        checkpoint_save_path=checkpoint_dir,
        model_name=train_config["model_type"],
        custom_callbacks=[TuneReportCallback()] # Add our reporter
    )

    # --- Training Loop --- #
    trial_logger.info(f"Starting training: {train_config['total_timesteps']} steps")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=train_config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=not train_config.get("continue_training", False)
        )
        training_time = time.time() - training_start_time
        trial_logger.info(f"Training finished in {training_time:.2f}s")
    except Exception as e:
        trial_logger.error(f"Error during model.learn: {e}", exc_info=True)
        # Report failure to Ray Tune
        if RAY_AVAILABLE:
            try:
                failure_metrics = {
                    "training_failure": True,
                    "timesteps": getattr(model, 'num_timesteps', 0)
                }
                # Try to get last metrics
                last_reward = 0.0
                last_variance = 0.0
                if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
                    log_vals = model.logger.name_to_value
                    last_reward = log_vals.get("rollout/ep_rew_mean", 0.0)
                    last_variance = log_vals.get("train/explained_variance", 0.0)
                    try: last_reward = float(last_reward)
                    except (ValueError, TypeError): last_reward = 0.0
                    try: last_variance = float(last_variance)
                    except (ValueError, TypeError): last_variance = 0.0

                temp_cb = TuneReportCallback()
                combo_score = temp_cb._normalize_and_combine_metrics(last_reward, last_variance)
                failure_metrics["eval/mean_reward"] = last_reward
                failure_metrics["eval/explained_variance"] = last_variance
                failure_metrics["eval/combined_score"] = combo_score
                trial_logger.info(f"Reporting failure: {failure_metrics}")

                if hasattr(ray, "air") and hasattr(ray.air, "session") \
                   and ray.air.session.is_active():
                    ray.air.session.report(failure_metrics)
                else:
                    tune.report(**failure_metrics)
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
        final_combined = temp_cb._normalize_and_combine_metrics(final_reward, final_variance)
        final_summary_metrics["eval/combined_score"] = final_combined
        trial_logger.debug(f"Final combined score for summary: {final_combined:.4f}")

    trial_logger.info("Final Summary Metrics:")
    for k, v in final_summary_metrics.items():
        trial_logger.info(f"  {k}: {v}")

    if RAY_AVAILABLE and tune.is_session_enabled():
        try: tune.report(**final_summary_metrics); trial_logger.info("Reported final metrics via tune.report")
        except Exception as re: trial_logger.warning(f"Failed final tune report: {re}")

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
        help="Max position size as fraction of balance (0.0-1.0)"
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
    rewards.add_argument("--consistency_penalty_weight", type=float, default=0.2)
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
    ppo.add_argument("--ent_coef", type=str, default="0.01")
    ppo.add_argument("--vf_coef", type=float, default=0.5)
    ppo.add_argument("--n_epochs", type=int, default=10)
    ppo.add_argument("--clip_range", type=float, default=0.2)
    ppo.add_argument("--gae_lambda", type=float, default=0.95)
    ppo.add_argument("--max_grad_norm", type=float, default=0.5)

    # --- SAC Specific Parameters --- #
    sac = parser.add_argument_group('SAC Specific Parameters')
    sac.add_argument("--tau", type=float, default=0.005)

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

    args = parser.parse_args()

    if args.model_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.model_name = f"{args.model_type}_{timestamp}"

    if isinstance(args.features, str):
        args.features = args.features.split(",")

    return args


# --- Helper Functions --- #

def args_to_config(args) -> Dict[str, Any]:
    """Convert argparse arguments to config dictionary."""
    return vars(args)


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
        data_key = config.get("data_key")
        data_loader = DataLoader(data_path=data_path, data_key=data_key)
        data = data_loader.load_data()

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

    reward_param_keys = [
        "portfolio_change_weight", "drawdown_penalty_weight", "sharpe_reward_weight",
        "fee_penalty_weight", "benchmark_reward_weight", "consistency_penalty_weight",
        "idle_penalty_weight", "profit_bonus_weight", "exploration_bonus_weight",
        "sharpe_window", "consistency_threshold", "idle_threshold",
        "exploration_start", "exploration_end", "exploration_decay_rate",
        "trade_penalty_weight"
    ]
    for key in reward_param_keys:
        if key in config:
            env_kwargs[key] = config[key]
            # logger.debug(f"Passing reward param '{key}' = {config[key]} to env.")

    env = TradingEnvironment(**env_kwargs)
    return env


def create_model(
    env: gym.Env,
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
            "policy": SacMlpPolicy,
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

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    policy_name = (model_kwargs['policy'] if isinstance(model_kwargs['policy'], str)
                   else model_kwargs['policy'].__name__)
    logger.info(f"Created {model_type.upper()} model with policy {policy_name}")
    return model


# --- Evaluation --- #

def evaluate_model(
    model: BaseRLModel,
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


def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    """
    log_path = os.path.join(config["log_dir"], config["model_name"], "evaluation")
    ensure_dir_exists(log_path)
    log_level = logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO
    setup_logger(log_dir=log_path, log_level=log_level)
    eval_logger = logging.getLogger("rl_agent")

    model_path = config["load_model"]
    if not model_path or not os.path.exists(model_path):
        eval_logger.error(f"Model path not found/specified: {model_path}"); sys.exit(1)
    if not config.get("test_data_path"):
        eval_logger.error("No test data path provided (--test_data_path)"); sys.exit(1)

    eval_logger.info(f"Creating test env from: {config['test_data_path']}")
    base_seed = config.get("seed")

    def make_single_eval_env(rank: int, base_seed_val: Optional[int]):
        def _init():
            env_config = config.copy()
            env_config["seed"] = base_seed_val + rank if base_seed_val is not None else None
            env = create_env(config=env_config, is_eval=True)
            monitor_log = os.path.join(log_path, f'monitor_eval_{rank}.csv')
            os.makedirs(os.path.dirname(monitor_log), exist_ok=True)
            env = Monitor(env, filename=monitor_log)
            return env
        return _init

    test_env = make_vec_env(
        env_id=make_single_eval_env(rank=0, base_seed_val=base_seed),
        n_envs=1, seed=None, vec_env_cls=DummyVecEnv, env_kwargs=None
    )

    # --- Apply VecNormalize --- #
    potential_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(potential_stats_path):
        eval_logger.info(f"Loading VecNorm stats: {potential_stats_path}")
        test_env = VecNormalize.load(potential_stats_path, test_env)
        test_env.training = False; test_env.norm_reward = False
    else:
        eval_logger.warning("VecNorm stats not found. Creating fresh wrapper.")
        norm_obs_setting = config.get("norm_obs", "auto").lower()
        if norm_obs_setting == "auto":
            features = config.get("features", [])
            if isinstance(features, str): features = features.split(",")
            has_scaled = any("_scaled" in f for f in features)
            should_norm_obs = not has_scaled
        else: should_norm_obs = norm_obs_setting == "true"
        eval_logger.info(f"Applying VecNorm: norm_obs={should_norm_obs}")
        test_env = VecNormalize(
            test_env, norm_obs=should_norm_obs, norm_reward=False,
            clip_obs=10., gamma=config["gamma"], training=False
        )

    model_cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C, "sac": SAC,
                 "lstm_dqn": DQN, "qrdqn": QRDQN, "recurrentppo": RecurrentPPO}
    model = model_cls[config["model_type"]].load(model_path, env=test_env)

    n_eval = config.get("n_eval_episodes", 5)
    eval_logger.info(f"Starting evaluation for {n_eval} episodes")
    mean_reward, portfolio_values, actions, rewards = evaluate_model(
        model=model, env=test_env, config=config,
        n_episodes=n_eval, deterministic=True
    )

    # Calculate metrics
    initial_balance = config.get("initial_balance", 10000)
    final_value = portfolio_values[-1] if len(portfolio_values) > 0 else initial_balance
    total_return = (final_value / initial_balance) - 1 if initial_balance > 0 else 0.0
    metrics = {"mean_reward": mean_reward, "final_portfolio_value": final_value,
               "total_return": total_return, "n_eval_episodes": n_eval}
    if len(portfolio_values) > 10:
        try:
            trading_metrics = calculate_trading_metrics(portfolio_values)
            metrics.update(trading_metrics)
            plot_path = os.path.join(log_path, "evaluation_plots.png")
            create_evaluation_plots(list(portfolio_values), actions, rewards, save_path=plot_path)
            eval_logger.info(f"Evaluation plots saved: {plot_path}")
        except Exception as e: eval_logger.warning(f"Eval metrics/plots error: {e}")
    else: eval_logger.warning(f"Only {len(portfolio_values)} vals; skipping advanced metrics.")

    eval_logger.info("Evaluation Results:")
    for k, v in metrics.items():
        log_str = f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
        eval_logger.info(log_str)

    metrics_file = os.path.join(log_path, "evaluation_metrics.json")
    try:
        serializable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in metrics.items()}
        with open(metrics_file, 'w', encoding='utf-8') as f: json.dump(serializable, f, indent=4)
        eval_logger.info(f"Metrics saved to {metrics_file}")
    except Exception as e: eval_logger.warning(f"Failed to save metrics: {e}")

    test_env.close()
    return metrics


# --- Training --- #

def train(config: Dict[str, Any]) -> Tuple[BaseRLModel, Dict[str, Any]]:
    """
    Train a reinforcement learning agent based on config.
    """
    log_path = os.path.join(config["log_dir"], config["model_name"])
    ensure_dir_exists(log_path)
    log_level = logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO
    setup_logger(log_dir=log_path, log_level=log_level)
    train_logger = logging.getLogger("rl_agent")

    sb3_log_path = os.path.join(log_path, "sb3_logs")
    ensure_dir_exists(sb3_log_path)
    sb3_logger_instance = setup_sb3_logger(log_dir=sb3_log_path)
    save_config(config=config, log_dir=log_path, filename="config.json")
    if config.get("seed") is not None: set_seeds(config["seed"])

    train_logger.info(f"Starting training for model: {config['model_name']}")
    device = ("cpu" if config["cpu_only"] else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_logger.info(f"Training on device: {device}")
    check_resources(train_logger)

    # --- Environment Creation --- #
    # --- Restore: Allow num_envs from config for non-tune runs ---
    # is_tune_run = RAY_AVAILABLE and tune.is_session_enabled()
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
        eval_env = make_vec_env(
            env_id=make_single_train_env(rank=0, base_seed_val=eval_env_seed_val), # Use corrected seed value
            n_envs=1, seed=None, vec_env_cls=DummyVecEnv, env_kwargs=None
        )

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
        eval_freq=max(config["eval_freq"], 5000),
        n_eval_episodes=config["n_eval_episodes"],
        save_freq=config["save_freq"],
        keep_checkpoints=config["keep_checkpoints"],
        resource_check_freq=config["resource_check_freq"],
        metrics_log_freq=config["metrics_log_freq"],
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_save_path=checkpoint_dir,
        model_name=config["model_type"],
        custom_callbacks=[]
    )

    # --- Training --- #
    train_logger.info(f"Starting training: {config['total_timesteps']} steps")
    training_start_time = time.time()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=1,
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
                     "eval/explained_variance": 0.0, "eval/combined_score": 0.5}
    if hasattr(model, 'logger') and hasattr(model.logger, 'name_to_value'):
        log_vals = model.logger.name_to_value
        final_reward = log_vals.get("rollout/ep_rew_mean", 0.0)
        final_variance = log_vals.get("train/explained_variance", 0.0)
        try: final_reward = float(final_reward)
        except (ValueError, TypeError): final_reward = 0.0
        try: final_variance = float(final_variance)
        except (ValueError, TypeError): final_variance = 0.0

        final_metrics["eval/mean_reward"] = final_reward
        final_metrics["eval/explained_variance"] = final_variance

        temp_cb = TuneReportCallback()
        final_combined = temp_cb._normalize_and_combine_metrics(final_reward, final_variance)
        final_metrics["eval/combined_score"] = float(final_combined)

    if RAY_AVAILABLE and tune.is_session_enabled():
        try: tune.report(**final_metrics); train_logger.info("Reported final metrics via tune.report")
        except Exception as re: train_logger.warning(f"Failed final tune report: {re}")

    return model, final_metrics


# --- Main Execution --- #

def main():
    """Main function: parse args, setup, run train/eval."""
    args = parse_args()
    config = args_to_config(args) # Get config from args FIRST

    # --- Config Loading --- #
    if args.load_config is not None:
        if os.path.exists(args.load_config):
            print(f"Loading configuration from {args.load_config}")
            file_config = load_config(args.load_config)
            # Update the initial config with values from the loaded file
            # This ensures file values overwrite defaults from args
            config.update(file_config)
            # Optional: Re-apply explicit CLI args if needed, but usually file takes precedence
            # cli_overrides = {k: v for k, v in vars(args).items() if # logic to detect non-default CLI args}
            # config.update(cli_overrides)
            print(f"Config updated with values from {args.load_config}")
        else:
            print(f"Error: Config file not found: {args.load_config}"); sys.exit(1)

    # Ensure features are a list (might be redundant now, but safe)
    if 'features' in config and isinstance(config['features'], str):
        config['features'] = config['features'].split(',')
    elif 'features' in config and isinstance(config['features'], list):
         # Ensure items are strings if read from JSON as list
         config['features'] = [str(f) for f in config['features']]

    # --- Directory Setup --- #
    log_base_dir, model_name = config["log_dir"], config["model_name"]
    ckpt_base_dir = config["checkpoint_dir"]
    ensure_dir_exists(log_base_dir); ensure_dir_exists(ckpt_base_dir)
    ensure_dir_exists(os.path.join(log_base_dir, model_name))
    ensure_dir_exists(os.path.join(ckpt_base_dir, model_name))

    # --- Mode Selection --- #
    if config.get("eval_only", False):
        print("Running in Evaluation-Only Mode")
        if config.get("load_model") is None: print("Error: --load_model required for eval"); sys.exit(1)
        if config.get("test_data_path") is None: print("Error: --test_data_path required for eval"); sys.exit(1)
        evaluate(config)
    else:
        print(f"Running Training Mode: {config['model_type']}")
        model, train_metrics = train(config)
        if config.get("test_data_path") is not None:
            print("\nStarting final evaluation on test data...")
            final_model = os.path.join(log_base_dir, model_name, "final_model.zip")
            eval_config = config.copy()
            eval_config["load_model"] = final_model
            eval_config["test_data_path"] = config["test_data_path"]
            test_metrics = evaluate(eval_config)
            print("\n--- Training Summary ---")
            print(f"Time: {train_metrics.get('training_time', 0):.2f}s")
            print(f"Steps: {train_metrics.get('total_timesteps', 0)}")
            print(f"Model: {final_model}")
            print("\n--- Test Set Evaluation Results ---")
            for k, v in test_metrics.items(): print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            final_model = os.path.join(log_base_dir, model_name, "final_model.zip")
            print("\n--- Training Completed ---")
            print(f"Time: {train_metrics.get('training_time', 0):.2f}s")
            print(f"Steps: {train_metrics.get('total_timesteps', 0)}")
            print(f"Model: {final_model}")
            print("No test data provided (--test_data_path).")


if __name__ == "__main__":
    main() 