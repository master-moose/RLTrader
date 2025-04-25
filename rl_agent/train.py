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
import copy  # Added import
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque # Add deque import

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
    # from ray.tune import CLIReporter # Keep CLIReporter # Unused
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
from rl_agent.policies import TcnPolicy


# --- Global Settings --- #

# Initialize logger globally (will be configured later)
# Use specific logger name instead of __name__ for consistency
logger = logging.getLogger("rl_agent")


# --- Ray Tune Callback Class --- #

class TuneReportCallback(BaseCallback):
    """Callback to report metrics to Ray Tune, focusing on scheduler needs."""
    def __init__(self, rollout_buffer_size=100):
        super().__init__(verbose=0)
        # Store last known variance
        self.last_explained_variance = 0.0
        # Buffer to store final metrics from completed episodes during rollout
        self.rollout_metrics_buffer = deque(maxlen=rollout_buffer_size)
        # Removing unused last_... variables for ratios/return
        # self.last_sharpe_ratio = 0.0
        # self.last_episode_return = 0.0
        # self.last_calmar_ratio = 0.0
        # self.last_sortino_ratio = 0.0

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

    def _normalize_and_combine_metrics(
        self, reward, explained_variance, sharpe_ratio, episode_return,
        calmar_ratio, sortino_ratio
    ):
        """
        Create a normalized combined score weighting various performance metrics,
        including bonuses for meeting specific ratio thresholds.

        Args:
            reward: The mean reward value.
            explained_variance: Explained variance (-1 to 1).
            sharpe_ratio: Sharpe ratio for the episode.
            episode_return: Episode return as a fraction.
            calmar_ratio: Calmar ratio for the episode.
            sortino_ratio: Sortino ratio for the episode.

        Returns:
            A combined normalized score between 0 and 1 (higher is better).
        """
        # --- Target Thresholds and Bonuses ---
        TARGET_SHARPE = 1.0
        TARGET_SORTINO = 1.5  # Can adjust this target
        TARGET_CALMAR = 0.5   # Can adjust this target
        METRIC_BONUS = 0.15   # Bonus added for each threshold met

        # --- Handle missing or invalid inputs --- #
        if reward is None or not isinstance(reward, (int, float, np.number)):
            reward = 0.0
        if (explained_variance is None
                or not isinstance(explained_variance, (int, float, np.number))):
            explained_variance = 0.0
        if (sharpe_ratio is None
                or not isinstance(sharpe_ratio, (int, float, np.number))
                or not np.isfinite(sharpe_ratio)):
            # Neutral Sharpe if invalid/infinite
            sharpe_ratio = 0.0
        if (episode_return is None
                or not isinstance(episode_return, (int, float, np.number))
                or not np.isfinite(episode_return)):
            # Neutral return if invalid/infinite
            episode_return = 0.0
        if (calmar_ratio is None
                or not isinstance(calmar_ratio, (int, float, np.number))
                or not np.isfinite(calmar_ratio)):
            # Neutral Calmar if invalid/infinite
            calmar_ratio = 0.0
        if (sortino_ratio is None
                or not isinstance(sortino_ratio, (int, float, np.number))
                or not np.isfinite(sortino_ratio)):
            # Neutral Sortino if invalid/infinite
            sortino_ratio = 0.0

        # --- Normalization (aiming for values roughly in [-1, 1]) --- #
        normalized_reward = np.tanh(reward / 1000.0)
        normalized_variance = np.clip(explained_variance, -1.0, 1.0)
        # Divisor 5 assumes typical range -5 to 5
        normalized_sharpe = np.tanh(sharpe_ratio / 5.0)
        normalized_return = np.tanh(episode_return / 2.0)  # Scale by 2
        # Divisor 2 assumes typical range -2 to 2
        normalized_calmar = np.tanh(calmar_ratio / 2.0)
        # Divisor 3 assumes typical range -3 to 3
        normalized_sortino = np.tanh(sortino_ratio / 3.0)

        # --- Weighted Combination (Base Score) --- #
        # Adjusted weights (ensure they sum reasonably, e.g., to 1.0)
        w_reward = 0.25
        w_variance = 0.10  # Reduced weight for variance
        w_sharpe = 0.15
        w_sortino = 0.25
        w_calmar = 0.25
        # Removing direct return weight, focus on risk-adjusted
        w_return = 0.0

        # Calculate weighted sum
        combined_score_raw = (
            w_reward * normalized_reward
            + w_variance * normalized_variance
            + w_sharpe * normalized_sharpe
            + w_return * normalized_return
            + w_calmar * normalized_calmar    # Added Calmar
            + w_sortino * normalized_sortino  # Added Sortino
        )

        # --- Add Threshold Bonuses ---
        bonus_score = 0.0
        if sharpe_ratio > TARGET_SHARPE:
            bonus_score += METRIC_BONUS
            logging.getLogger("rl_agent").debug(
                f"  Sharpe bonus added ({sharpe_ratio:.2f} > {TARGET_SHARPE})"
            )
        if sortino_ratio > TARGET_SORTINO:
            bonus_score += METRIC_BONUS
            logging.getLogger("rl_agent").debug(
                f"  Sortino bonus added ({sortino_ratio:.2f} > {TARGET_SORTINO})"
            )
        if calmar_ratio > TARGET_CALMAR:
            bonus_score += METRIC_BONUS
            logging.getLogger("rl_agent").debug(
                f"  Calmar bonus added ({calmar_ratio:.2f} > {TARGET_CALMAR})"
            )

        combined_score_with_bonuses = combined_score_raw + bonus_score

        # --- Shift to [0, 1] range --- #
        # The raw score + bonuses might now range approx [-1, 1 + 3*METRIC_BONUS]
        weights = [w_reward, w_variance, w_sharpe,
                   w_sortino, w_calmar, w_return]
        max_possible_score = sum(weights) + 3 * METRIC_BONUS
        min_possible_score = -sum(weights)  # Approx lower bound
        score_range = max_possible_score - min_possible_score

        # Scale to [0, 1] based on the estimated range
        if score_range > 0:
            combined_score_normalized = (
                (combined_score_with_bonuses - min_possible_score) / score_range
            )
        else:
            combined_score_normalized = 0.5

        # Ensure final score is strictly within [0, 1]
        combined_score_final = np.clip(combined_score_normalized, 0.0, 1.0)
        logging.getLogger("rl_agent").debug(
            f"  Combined Score: raw={combined_score_raw:.3f}, "
            f"bonus={bonus_score:.3f}, "
            f"total={combined_score_with_bonuses:.3f}, "
            f"scaled={combined_score_final:.3f}"
        )

        return combined_score_final

    def _on_step(self) -> bool:
        """
        Update the last known metrics (variance, sharpe, return) after each
        training step. Try fetching directly from the logger's internal
        dictionary or locals. Also fetches risk metrics from the environment
        info dictionary.
        """
        # --- Update Explained Variance (existing logic) --- #
        if self.logger is not None and hasattr(self.logger, 'name_to_value'):
            logged_variance = self.logger.name_to_value.get(
                "train/explained_variance", None
            )
            if logged_variance is not None:
                try:
                    self.last_explained_variance = float(logged_variance)
                except (ValueError, TypeError):
                    pass
            # Fallback to locals if not in logger yet
            elif hasattr(self, 'locals') and self.locals:
                possible_keys = ["explained_variance",
                                 "train/explained_variance"]
                for key in possible_keys:
                    if key in self.locals:
                        try:
                            self.last_explained_variance = float(
                                self.locals[key]
                            )
                            break
                        except (ValueError, TypeError, KeyError):
                            pass

        # --- Store FINAL metrics from DONE environments --- #
        if (hasattr(self, 'locals') and self.locals and
                'infos' in self.locals and 'dones' in self.locals):
            infos = self.locals['infos']
            dones = self.locals['dones']
            callback_logger = logging.getLogger("rl_agent.train") # Logger for reports
            for i, done in enumerate(dones):
                if done:
                    # Get final info dictionary
                    final_info = infos[i].get("final_info", infos[i])

                    # Extract single-episode metrics
                    ep_metrics = {
                        'reward': final_info.get('r', 0.0),
                        'length': final_info.get('l', 0),
                        'time': final_info.get('t', 0.0),
                        'sharpe': final_info.get('sharpe_ratio_episode', 0.0),
                        'return': final_info.get('episode_return', 0.0),
                        'calmar': final_info.get('calmar_ratio', 0.0),
                        'sortino': final_info.get('sortino_ratio', 0.0)
                    }

                    # Ensure values are floats and finite, default to 0.0 otherwise
                    for k, v in ep_metrics.items():
                        try:
                            val = float(v)
                            ep_metrics[k] = val if np.isfinite(val) else 0.0
                        except (TypeError, ValueError):
                            ep_metrics[k] = 0.0

                    # --- Calculate Combined Score for this episode --- #
                    # Use last known explained variance as best proxy
                    combined_score = self._normalize_and_combine_metrics(
                        reward=ep_metrics['reward'],
                        explained_variance=self.last_explained_variance,
                        sharpe_ratio=ep_metrics['sharpe'],
                        episode_return=ep_metrics['return'],
                        calmar_ratio=ep_metrics['calmar'],
                        sortino_ratio=ep_metrics['sortino']
                    )

                    # --- Report directly to Ray Tune --- #
                    session_active = False
                    if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
                        session_active = ray.air.session.is_active()

                    if session_active:
                        try:
                            metrics_to_report = {
                                "episode_reward_mean": ep_metrics['reward'], # Report single ep reward
                                "episode_len_mean": ep_metrics['length'],
                                "timesteps_total": self.num_timesteps,
                                # Use eval prefix for consistency with scheduler/search alg
                                "eval/mean_reward": ep_metrics['reward'],
                                "eval/explained_variance": self.last_explained_variance, # Last known variance
                                "eval/combined_score": combined_score,
                                "eval/sharpe_ratio": ep_metrics['sharpe'],
                                "eval/sortino_ratio": ep_metrics['sortino'],
                                "eval/calmar_ratio": ep_metrics['calmar'],
                                "eval/mean_return_pct": ep_metrics['return'] * 100,
                            }
                            ray.air.session.report(metrics_to_report)
                            callback_logger.debug(
                                f"Reported episode end metrics at step "
                                f"{self.num_timesteps}: Reward={ep_metrics['reward']:.2f}, "
                                f"Score={combined_score:.3f}"
                            )
                        except Exception as e:
                            callback_logger.error(
                                f"Error reporting episode metrics at step "
                                f"{self.num_timesteps}: {e}"
                            )
                    # --- End Reporting ---

                    # REMOVED: Don't buffer metrics anymore
                    # self.rollout_metrics_buffer.append(metrics)
                    # logging.getLogger("rl_agent").debug(f"  Stored final metrics from env {i}: {metrics}")

        return True

    def _on_rollout_end(self) -> None:
        """
        Log summary statistics at the end of each rollout.
        Reporting to Ray Tune is now handled in _on_step when an episode ends.
        """
        callback_logger = logging.getLogger("rl_agent.train")
        # --- Get Mean Reward from Monitor Buffer --- #
        reward_value = 0.0
        num_completed_eps = 0
        if (hasattr(self.model, "ep_info_buffer") and
                len(self.model.ep_info_buffer) > 0):
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                reward_value = float(np.mean(ep_rewards))
                num_completed_eps = len(ep_rewards)
                # callback_logger.debug(
                #     f"Mean reward ({len(ep_rewards)} eps): {reward_value:.4f}"
                # )

        # REMOVED: Averaging logic for rollout_metrics_buffer

        # Use last known explained variance for logging consistency
        variance_value = self.last_explained_variance

        # REMOVED: Combined score calculation is not needed here for reporting

        # --- Log as Markdown Table (Shows Monitor/Rollout View) --- #
        log_table = "| Metric (Rollout View) | Value     |\n"
        log_table += "| :---------------------- | :-------- |\n"
        log_table += f"| Timesteps               | {self.num_timesteps:<9} |\n"
        log_table += f"| Completed Eps (Monitor) | {num_completed_eps:<9} |\n"
        log_table += f"| Mean Reward (Monitor)   | {reward_value:<9.3f} |\n"
        # Financial metrics are no longer averaged here
        log_table += f"| Mean Return (%)         | N/A       |\n" # Indicate N/A
        log_table += f"| Mean Sharpe             | N/A       |\n"
        log_table += f"| Mean Sortino            | N/A       |\n"
        log_table += f"| Mean Calmar             | N/A       |\n"
        log_table += f"| Expl. Variance (Last) | {variance_value:<9.3f} |\n"
        # Combined score is reported per-episode now
        log_table += f"| Combined Score          | N/A       |\n"
        callback_logger.info(f"Rollout End Log:\n{log_table}")

        # REMOVED: Ray Tune reporting section

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

    def make_single_env(rank: int, base_seed: Optional[int], is_eval_flag: bool = False):
        def _init():
            env_config = train_config.copy()
            instance_seed = base_seed + rank if base_seed is not None else None
            env_config["seed"] = instance_seed
            env = create_env(config=env_config, is_eval=is_eval_flag)
            log_suffix = f'monitor_eval_{rank}.csv' if is_eval_flag else f'monitor_{rank}.csv'
            monitor_log_path = os.path.join(log_dir, log_suffix)
            ensure_dir_exists(os.path.dirname(monitor_log_path))
            env = Monitor(env, filename=monitor_log_path)
            return env
        return _init

    vec_env_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    train_env = make_vec_env(
        env_id=make_single_env(rank=0, base_seed=seed, is_eval_flag=False),
        n_envs=num_envs,
        seed=None,
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
            env_id=make_single_env(rank=0, base_seed=eval_env_seed_val, is_eval_flag=True),
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
    rollout_steps = n_steps * num_envs
    base_eval_freq = train_config.get("eval_freq", 10000)
    effective_eval_freq = max(base_eval_freq, rollout_steps)
    trial_logger.info(f"Effective eval_freq = {effective_eval_freq}")

    tune_early_stopping_patience = 0 # Disabled for debugging
    trial_logger.info(f"Tune early_stopping_patience = {tune_early_stopping_patience}")

    callbacks = get_callback_list(
        eval_env=eval_env,
        log_dir=log_dir,
        eval_freq=max(config.get("eval_freq", 10000), 5000),
        n_eval_episodes=train_config.get("n_eval_episodes", 5),
        save_freq=config.get("save_freq", 10000),
        keep_checkpoints=config.get("keep_checkpoints", 3),
        resource_check_freq=config.get("resource_check_freq", 1000),
        metrics_log_freq=config.get("metrics_log_freq", 1000),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        checkpoint_save_path=checkpoint_dir,
        model_name=train_config["model_type"],
        custom_callbacks=[TuneReportCallback()],
        curriculum_duration_fraction=0.0
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
                combo_score = temp_cb._normalize_and_combine_metrics(last_reward, last_variance, 0.0, 0.0, 0.0, 0.0)
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
        final_combined = temp_cb._normalize_and_combine_metrics(final_reward, final_variance, 0.0, 0.0, 0.0, 0.0)
        final_summary_metrics["eval/combined_score"] = float(final_combined)

    trial_logger.info("Final Summary Metrics:")
    for k, v in final_summary_metrics.items():
        trial_logger.info(f"  {k}: {v}")

    # Use the modern check for Ray Tune session
    session_active = False
    if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
        session_active = ray.air.session.is_active()

    if session_active:
        try: 
            # Use ray.air.session.report for newer versions
            ray.air.session.report(final_summary_metrics) 
            trial_logger.info("Reported final metrics via Ray AIR session")
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
            "dqn", "ppo", "a2c", "sac", "lstm_dqn", "qrdqn", "recurrentppo", "tcn_ppo"
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
    features = [f for f in env_kwargs['features'] if not f.endswith(('_15m', '_4h', '_1d'))]
    env_kwargs['features'] = features

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

    elif model_type == "tcn_ppo":
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
        
        # --- Infer actual features per timestep from environment's observation space ---
        # The 'env' passed here is the VecNormalize-wrapped environment
        if not isinstance(env.observation_space, gym.spaces.Box):
             raise ValueError(f"TcnPolicy requires a Box observation space, got {type(env.observation_space)}")
    
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
        # Wrap with Monitor for logging episode returns/lengths
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
        "tcn_ppo": PPO  # TCN is handled by TcnPolicy
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

def train(config: Dict[str, Any]) -> Tuple[BaseRLModel, Dict[str, Any]]:
    """
    Train a reinforcement learning agent based on config.
    """
    log_path = os.path.join(config["log_dir"], config["model_name"])
    ensure_dir_exists(log_path)
    # Determine console level based on verbose setting
    console_log_level = logging.DEBUG if config.get("verbose", 1) >= 2 else logging.INFO
    # Always set file level to DEBUG
    file_log_level = logging.DEBUG 
    setup_logger(log_dir=log_path, log_level=file_log_level, console_level=console_log_level)
    train_logger = logging.getLogger("rl_agent")
    
    # <<< Ensure environment logger inherits the DEBUG level >>>
    logging.getLogger("rl_agent.environment").setLevel(logging.DEBUG)
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
        custom_callbacks=[TuneReportCallback()],
        curriculum_duration_fraction=0.0
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
        final_combined = temp_cb._normalize_and_combine_metrics(final_reward, final_variance, 0.0, 0.0, 0.0, 0.0)
        final_metrics["eval/combined_score"] = float(final_combined)

    # Use the modern check for Ray Tune session
    session_active = False
    if RAY_AVAILABLE and hasattr(ray, "air") and hasattr(ray.air, "session"):
        session_active = ray.air.session.is_active()
        
    if session_active:
        try: 
            # Use ray.air.session.report for newer versions
            ray.air.session.report(final_metrics) 
            train_logger.info("Reported final metrics via Ray AIR session")
        except Exception as re: 
            train_logger.warning(f"Failed final Ray AIR session report: {re}")

    return model, final_metrics


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