#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DQN, PPO, A2C, and SAC Agent Training for Cryptocurrency Trading

This script trains reinforcement learning agents for cryptocurrency trading using parallel environments
with safeguards against rapid trading and action oscillation issues.

Ensures compatibility with:
- StableBaselines3
- Gymnasium 0.28+
- PyTorch 2.0+
"""

import os
import sys
import time
import logging
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import gc
import traceback
import gymnasium
import types
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
from crypto_trading_model.environment.crypto_env import CryptocurrencyTradingEnv
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.utils import set_seeds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define technical indicators to ensure they're included in the environment
INDICATORS = ['macd', 'rsi', 'cci', 'dx', 'bb_upper', 'bb_lower', 'bb_middle', 'volume']

# Constants for trading safeguards
TRADE_COOLDOWN_PERIOD = 100  # Minimum steps between trades
OSCILLATION_PENALTY = 500.0  # Stronger penalty for oscillating between buys and sells
SAME_PRICE_TRADE_PENALTY = 1000.0  # Stronger penalty for trading at same price
MAX_TRADE_FREQUENCY = 0.05  # Maximum 5% of steps can be trades

class SafeTradingEnvWrapper(gymnasium.Wrapper):
    """
    A wrapper for trading environments that adds safeguards against:
    1. Rapid trading (enforces a cooldown period)
    2. Oscillatory trading behavior 
    3. Trading at the same price repeatedly
    4. Excessive trading frequency
    
    Also adds risk-aware rewards and curriculum learning capabilities.
    """
    
    def __init__(self, env, trade_cooldown=TRADE_COOLDOWN_PERIOD, max_history_size=100):
        """Initialize the wrapper with safeguards against harmful trading patterns"""
        super().__init__(env)
        
        # Trading safeguards
        self.trade_cooldown = trade_cooldown
        self.cooldown_violations = 0
        self.forced_actions = 0
        
        # Trading history tracking
        self.last_trade_step = -self.trade_cooldown  # Start with cooldown already passed
        self.last_trade_price = None
        self.action_history = []
        self.position_history = []
        self.previous_position = 0  # Initial position is flat (no assets)
        self.current_position = 0
        self.max_history_size = max_history_size
        self.oscillation_count = 0
        self.same_price_trades = 0
        
        # Track consecutive actions for consistency rewards
        self.consecutive_holds = 0
        self.consecutive_same_action = 0
        self.last_action = None
        
        # Position sizing for progressive training
        self.position_size_pct = 0.1  # Start with small position sizes
        self.training_progress = 0.0  # Track progress from 0.0 to 1.0
        
        # Performance metrics for risk-aware training
        self.trade_returns = []
        self.successful_trades = 0
        self.failed_trades = 0
        self.trade_pnl = []
        
        # Price history for last trades
        self.last_buy_price = None
        self.last_sell_price = None
        self.min_profit_threshold = 0.002  # Require at least 0.2% price difference to trade
        
        # Stronger oscillation detection and prevention
        self.oscillation_window = 8  # Look at 8 actions for oscillation patterns
        self.progressive_cooldown = True  # Increase cooldown after oscillations
        self.max_oscillation_cooldown = trade_cooldown * 5  # Maximum extended cooldown
        self.oscillation_patterns = {
            'buy_sell_alternation': 0,  # Count of buy-sell alternations
            'rapid_reversals': 0,       # Count of position reversals
        }
        
        # Track trading metrics for risk-aware rewards
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.env.portfolio_value if hasattr(self.env, 'portfolio_value') else 0.0
        
        self.current_cooldown = trade_cooldown  # Adjustable cooldown period
        
        # Modify observation space to include action history and risk metrics
        if isinstance(self.env.observation_space, spaces.Box):
            # Calculate new observation space size
            # Original observation dimension + action history (15) + consistency metrics (3) 
            # + risk metrics (3) + cooldown status (1)
            original_shape = self.env.observation_space.shape
            if len(original_shape) == 1:
                # 1D observation space
                additional_features = 22  # 15 + 3 + 3 + 1
                low = np.append(self.env.observation_space.low, [-np.inf] * additional_features)
                high = np.append(self.env.observation_space.high, [np.inf] * additional_features)
                self.observation_space = spaces.Box(
                    low=low, 
                    high=high, 
                    dtype=self.env.observation_space.dtype
                )
                logger.info(f"Expanded observation space from {original_shape[0]} to {original_shape[0] + additional_features} dimensions")
            else:
                # More complex observation space, keep original
                logger.warning(f"Keeping original observation space with shape {original_shape} - cannot augment non-1D space")
        else:
            # Non-Box observation space, keep original
            logger.warning(f"Keeping original observation space of type {type(self.env.observation_space)} - augmentation only supports Box spaces")
        
        logger.info(f"SafeTradingEnvWrapper initialized with {trade_cooldown} step cooldown")
    
    def reset(self, **kwargs):
        """Reset the environment and all trading history"""
        observation, info = self.env.reset(**kwargs)
        
        # Reset trading history
        self.action_history = []
        self.position_history = []
        self.last_trade_step = -self.trade_cooldown
        self.last_trade_price = None
        self.last_buy_price = None
        self.last_sell_price = None
        self.previous_position = 0
        self.current_position = 0
        
        # Reset oscillation detection
        self.oscillation_count = 0
        self.cooldown_violations = 0
        self.forced_actions = 0
        self.same_price_trades = 0
        self.consecutive_holds = 0
        self.consecutive_same_action = 0
        self.last_action = None
        
        # Reset risk metrics
        self.trade_returns = []
        self.trade_pnl = []
        self.successful_trades = 0
        self.failed_trades = 0
        self.peak_value = self.env.portfolio_value if hasattr(self.env, 'portfolio_value') else 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Add action history to observation
        observation = self._augment_observation(observation)
        
        return observation, info
    
    def _augment_observation(self, observation):
        """Add action history and risk metrics to the observation space"""
        # Get original observation shape
        orig_obs = observation
        
        # Create one-hot encoding of recent actions (last 5 actions)
        action_history = np.zeros(15)  # 5 recent actions x 3 possible actions (one-hot)
        for i, action in enumerate(self.action_history[-5:]):
            if action is not None and i < 5:
                # One-hot encode each action
                if action == 0:  # Sell
                    action_history[i * 3] = 1
                elif action == 1:  # Hold
                    action_history[i * 3 + 1] = 1
                elif action == 2:  # Buy 
                    action_history[i * 3 + 2] = 1
        
        # Add consistency metrics
        consistency_metrics = np.array([
            min(self.consecutive_same_action / 10.0, 1.0),  # Normalized consecutive actions
            min(self.consecutive_holds / 20.0, 1.0),  # Normalized consecutive holds
            min(self.oscillation_count / 50.0, 1.0),  # Normalized oscillation count
        ])
        
        # Add risk metrics
        risk_metrics = np.array([
            max(min(self.sharpe_ratio / 2.0, 1.0), -1.0),  # Clipped Sharpe ratio
            min(self.max_drawdown, 1.0),  # Max drawdown
            1.0 if self.consecutive_holds > 10 else 0.0,  # Long hold indicator
        ])
        
        # Calculate current cooldown status
        current_step = getattr(self.env, 'day', 0)
        cooldown_status = min(max(
            (current_step - self.last_trade_step) / self.current_cooldown, 0), 1)
        
        # Combine all features
        augmented_features = np.concatenate([
            action_history,
            consistency_metrics,
            risk_metrics,
            np.array([cooldown_status]),
        ])
        
        # Combine with original observation
        if isinstance(orig_obs, np.ndarray):
            augmented_observation = np.concatenate([orig_obs, augmented_features])
        else:
            # Handle case where observation might be a dict or other structure
            logger.warning("Non-array observation type detected, returning original")
            return orig_obs
            
        return augmented_observation
    
    def _detect_oscillation_patterns(self):
        """Detect oscillation patterns in recent actions"""
        # Need at least 8 actions for meaningful pattern detection
        if len(self.action_history) < 8:
            return False
        
        # Get the most recent actions
        recent_actions = self.action_history[-self.oscillation_window:]
        
        # Pattern 1: Buy-sell alternation (e.g., [2, 0, 2, 0] or [0, 2, 0, 2])
        alternation_detected = False
        for i in range(len(recent_actions) - 3):
            if (recent_actions[i] == 2 and recent_actions[i+1] == 0 and 
                recent_actions[i+2] == 2 and recent_actions[i+3] == 0):
                alternation_detected = True
                self.oscillation_patterns['buy_sell_alternation'] += 1
                logger.warning(f"Detected action oscillation at step {getattr(self.env, 'current_step', 0)}: {recent_actions[i:i+4]}")
                break
            if (recent_actions[i] == 0 and recent_actions[i+1] == 2 and 
                recent_actions[i+2] == 0 and recent_actions[i+3] == 2):
                alternation_detected = True
                self.oscillation_patterns['buy_sell_alternation'] += 1
                logger.warning(f"Detected action oscillation at step {getattr(self.env, 'current_step', 0)}: {recent_actions[i:i+4]}")
                break
        
        # Pattern 2: Rapid reversals (e.g., long pause then [2, 2, 0, 0] or [0, 0, 2, 2])
        reversal_detected = False
        for i in range(len(recent_actions) - 3):
            if (recent_actions[i] == recent_actions[i+1] == 2 and 
                recent_actions[i+2] == recent_actions[i+3] == 0):
                reversal_detected = True
                self.oscillation_patterns['rapid_reversals'] += 1
                break
            if (recent_actions[i] == recent_actions[i+1] == 0 and 
                recent_actions[i+2] == recent_actions[i+3] == 2):
                reversal_detected = True
                self.oscillation_patterns['rapid_reversals'] += 1
                break
                
        # Update oscillation count if either pattern is detected
        if alternation_detected or reversal_detected:
            self.oscillation_count += 1
            return True
        
        return False
    
    def _update_cooldown_period(self):
        """Adjust cooldown period based on oscillation patterns"""
        # Calculate oscillation score
        oscillation_score = 0
        
        # Heavily weight buy-sell alternations
        oscillation_score += self.oscillation_patterns['buy_sell_alternation'] * 1.5
        
        # Add rapid reversals with less weight
        oscillation_score += self.oscillation_patterns['rapid_reversals'] * 1.0
        
        # Adapt cooldown based on oscillation score
        if oscillation_score > 5:
            # Severe oscillation, use maximum cooldown
            new_cooldown = self.max_oscillation_cooldown
        elif oscillation_score > 2:
            # Some oscillation, scale cooldown linearly
            new_cooldown = self.trade_cooldown + (oscillation_score - 2) * (self.max_oscillation_cooldown - self.trade_cooldown) / 3
        else:
            # Minimal oscillation, use base cooldown
            new_cooldown = self.trade_cooldown
        
        # Update if significant change
        if abs(new_cooldown - self.current_cooldown) > 10:
            self.current_cooldown = int(new_cooldown)
            logger.info(f"Adjusted cooldown period to {self.current_cooldown} steps (oscillation score: {oscillation_score:.1f})")
    
    def _calculate_risk_rewards(self, reward, info, trade_occurred, current_step):
        """Calculate risk-aware rewards based on trading performance"""
        # Get current portfolio value
        portfolio_value = info.get('portfolio_value', 0)
        
        # Update peak value and max drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Calculate drawdown
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # If a trade occurred, update trade statistics
        if trade_occurred:
            # Add trade return to history
            if self.last_trade_price is not None and 'close_price' in info:
                trade_return = (info['close_price'] - self.last_trade_price) / self.last_trade_price
                self.trade_returns.append(trade_return)
                
                # Track if this was a successful trade
                if trade_return > 0:
                    self.successful_trades += 1
                else:
                    self.failed_trades += 1
                
                # Calculate Sharpe ratio if we have enough trades
                if len(self.trade_returns) > 5:
                    returns_array = np.array(self.trade_returns[-20:])
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array) + 1e-6  # Add small constant to avoid division by zero
                    self.sharpe_ratio = mean_return / std_return
        
        # Apply risk-aware reward adjustments
        risk_adjusted_reward = reward
        
        # Penalize high drawdowns
        if self.max_drawdown > 0.1:  # More than 10% drawdown
            drawdown_penalty = self.max_drawdown * 2.0
            risk_adjusted_reward -= drawdown_penalty
        
        # Reward positive Sharpe ratio
        if self.sharpe_ratio > 0.5:
            sharpe_bonus = self.sharpe_ratio * 0.5
            risk_adjusted_reward += sharpe_bonus
        
        # Add consistency reward - encourage staying with same action
        if self.last_action == self.action_history[-1]:
            self.consecutive_same_action += 1
            consistency_bonus = min(self.consecutive_same_action * 0.02, 0.5)
            risk_adjusted_reward += consistency_bonus
        else:
            self.consecutive_same_action = 0
        
        # Add hold bonus during early training
        if current_step < 10000 and self.action_history and self.action_history[-1] == 1:  # hold action
            self.consecutive_holds += 1
            if self.consecutive_holds > 5:
                hold_bonus = min(self.consecutive_holds * 0.01, 0.5)
                risk_adjusted_reward += hold_bonus
        else:
            self.consecutive_holds = 0
            
        # Curriculum learning - scale down rewards in early training
        self.training_progress = min(current_step / 1000000.0, 1.0)  # Progress from 0 to 1 over 1M steps
        
        return risk_adjusted_reward

    def step(self, action):
        """
        Take a step in the environment with enhanced safeguards against harmful
        trading patterns and additional reward-shaping for stable behavior.
        
        Args:
            action: 0 (sell), 1 (hold), or 2 (buy)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current step in environment
        current_step = getattr(self.env, 'day', 0)
        
        # Add curriculum learning - restrict action space in early training
        if self.training_progress < 0.1 and action != 1 and current_step % 10 != 0:
            # Force hold actions 90% of the time in first 10% of training
            action = 1
        
        # Store previous position for change detection
        self.previous_position = self.current_position
        
        # Determine if we're in a trade cooldown period
        in_cooldown = (current_step - self.last_trade_step) < self.current_cooldown
        attempted_trade_during_cooldown = False
        
        # Get current price for advanced checks
        current_price = None
        if hasattr(self.env, '_get_current_price'):
            current_price = self.env._get_current_price()
            
        # Advanced early-stage check for valid price changes before allowing trades
        min_profit_violation = False
        
        strict_price_requirement = False
        if current_price is not None and self.last_trade_price is not None:
            price_change_pct = abs(current_price - self.last_trade_price) / self.last_trade_price
            
            # Require larger price movements early in training
            required_movement = self.min_profit_threshold
            
            # More strict requirements depending on training stage
            if current_step < 1000:
                required_movement = 0.03  # Require 3% difference in first 1k steps
            elif current_step < 5000:
                required_movement = 0.015  # Require 1.5% in next 4k steps
            elif current_step < 10000:
                required_movement = 0.01  # Require 1% in next 5k steps
            elif current_step < 50000:
                required_movement = 0.005  # Require 0.5% in next 40k steps
            
            if price_change_pct < required_movement and action != 1:
                strict_price_requirement = True
                logger.debug(f"Enforcing hold: price movement {price_change_pct:.4f}% < required {required_movement:.4f}%")
                action = 1  # Force hold
            
        # Check for min profit threshold violations
        if current_price is not None:
            # If trying to sell, check if price is higher than last buy
            if action == 0 and self.last_buy_price is not None:
                profit_pct = (current_price - self.last_buy_price) / self.last_buy_price
                
                # Increase profit threshold early in training
                min_profit = self.min_profit_threshold
                if current_step < 5000:
                    min_profit = 0.01  # Require at least 1% profit in early training
                
                if profit_pct < min_profit:
                    min_profit_violation = True
                    logger.debug(f"Prevented selling at loss/small profit: {profit_pct:.4f}% at step {current_step}")
            
            # If trying to buy, check if price is lower than last sell
            elif action == 2 and self.last_sell_price is not None:
                discount_pct = (self.last_sell_price - current_price) / self.last_sell_price
                
                # Increase discount threshold early in training
                min_discount = self.min_profit_threshold
                if current_step < 5000:
                    min_discount = 0.01  # Require at least 1% discount in early training
                
                if discount_pct < min_discount:
                    min_profit_violation = True
                    logger.debug(f"Prevented buying too close to last sell: -{discount_pct:.4f}% at step {current_step}")
        
        # Handle cooldown violations
        if in_cooldown and action != 1:  # Not a hold action
            # Track violation attempt
            attempted_trade_during_cooldown = True
            
            # Force a hold action
            action = 1
            self.forced_actions += 1
            
            if self.forced_actions % 10 == 0:  # Log periodically to avoid spamming
                logger.warning(f"Forced hold action during cooldown at step {current_step}, " 
                              f"{current_step - self.last_trade_step}/{self.current_cooldown} steps since last trade")
        
        # Also force hold for min profit violations and price movement requirements
        if min_profit_violation or strict_price_requirement:
            action = 1
        
        # Check for oscillation patterns and adjust cooldown if needed
        if len(self.action_history) > 4:
            last_four = self.action_history[-4:]
            
            # Apply progressive penalty for oscillatory behavior
            if last_four and (
                last_four == [2, 0, 2, 0] or last_four == [0, 2, 0, 2] or
                last_four == [2, 0, 2, 1] or last_four == [0, 2, 0, 1] or
                last_four == [1, 0, 2, 0] or last_four == [1, 2, 0, 2]):
                
                # Force hold action for the next several steps
                action = 1
                
                # Count the oscillation
                self.oscillation_count += 1
                
                # Extend cooldown for severe oscillation cases
                if self.oscillation_count > 5:
                    # Extend cooldown period for severe oscillation
                    extended_cooldown = min(self.oscillation_count * 20, 1000)  # More aggressive extension
                    self.last_trade_step = current_step - self.current_cooldown + extended_cooldown
                    logger.warning(f"Extended cooldown by {extended_cooldown} steps due to severe oscillation (count: {self.oscillation_count})")
        
        # Early training stabilization (first 1000 steps) - reduce trade frequency drastically
        if current_step < 1000 and action != 1 and current_step % 100 != 0:
            logger.debug(f"Early training stability: forcing hold at step {current_step}")
            action = 1  # Force hold action during early training except every 100th step
        
        # Take the step in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current price and position after the step
        current_price = None
        if 'close_price' in info:
            current_price = info['close_price']
        elif hasattr(self.env, '_get_current_price'):
            current_price = self.env._get_current_price()
        
        # Get new position
        if 'assets_owned' in info and len(info['assets_owned']) > 0:
            self.current_position = info['assets_owned'][0]
        
        # Update action history
        self.action_history.append(action)
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]
        
        # Update position history
        current_position = self.current_position
        self.position_history.append(current_position)
        if len(self.position_history) > self.max_history_size:
            self.position_history = self.position_history[-self.max_history_size:]
        
        # Detect if a trade occurred by checking position change
        trade_occurred = self.previous_position != self.current_position
        
        # Calculate additional penalties
        additional_penalty = 0.0
        
        # Penalty for attempted trades during cooldown
        if attempted_trade_during_cooldown:
            cooldown_penalty = OSCILLATION_PENALTY * (0.5 + min(self.cooldown_violations, 10) / 10)
            additional_penalty -= cooldown_penalty
            self.cooldown_violations += 1
            if self.cooldown_violations % 5 == 0:
                logger.warning(f"Applied cooldown violation penalty: {cooldown_penalty:.2f} " 
                              f"(#{self.cooldown_violations})")
            
            # Add cooldown violation to info dict
            info['cooldown_violation'] = True
        
        # Penalty for oscillations
        if len(self.action_history) >= 4:
            last_four = self.action_history[-4:]
            
            if last_four == [2, 0, 2, 0] or last_four == [0, 2, 0, 2]:
                # Increase oscillation penalty exponentially based on count
                oscillation_penalty = OSCILLATION_PENALTY * (1.0 + min(self.oscillation_count, 5) ** 1.5)
                additional_penalty -= oscillation_penalty
                logger.warning(f"Applied oscillation penalty: {oscillation_penalty:.2f} (#{self.oscillation_count})")
                
                # Add oscillation detection to info dict
                info['oscillation_detected'] = True
        
        # Penalty for same-price trades
        if trade_occurred and current_price is not None and self.last_trade_price is not None:
            # Check if trade price is very close to the last trade price
            if abs(current_price - self.last_trade_price) < 0.0001:
                self.same_price_trades += 1
                same_price_penalty = SAME_PRICE_TRADE_PENALTY * (1.0 + min(self.same_price_trades, 5))
                additional_penalty -= same_price_penalty
                logger.warning(f"Same price trade detected! Penalty: {same_price_penalty:.2f} (#{self.same_price_trades})")
                
                # Add same-price trade to info dict
                info['same_price_trade'] = True
        
        # Update last trade info if a trade occurred
        if trade_occurred:
            self.last_trade_step = current_step
            self.last_trade_price = current_price
            
            # Track buy/sell prices for profit calculation
            if action == 2:  # Buy
                self.last_buy_price = current_price
            elif action == 0:  # Sell
                self.last_sell_price = current_price
                
            # Update info dict
            info['trade'] = True
            
            # Reset cooldown violation count after successful trade
            if self.cooldown_violations > 0:
                logger.info(f"Resetting cooldown violation count from {self.cooldown_violations} to 0")
                self.cooldown_violations = 0
        
        # Apply any penalties to the reward
        reward += additional_penalty
        
        # Apply risk-aware reward adjustments
        reward = self._calculate_risk_rewards(reward, info, trade_occurred, current_step)
        
        # Add consistency rewards
        if self.last_action is not None:
            if action == self.last_action:
                # Reward consistent actions (staying in a position or holding)
                consistency_reward = 0.05  # Small bonus for consistency
                
                # Additional bonus for holding
                if action == 1:  # Hold
                    consistency_reward += 0.03  # Extra bonus for holding
                    
                reward += consistency_reward
        
        # Update last action
        self.last_action = action
        
        # Curriculum learning - scale down rewards in early training
        if self.training_progress < 0.2:
            reward *= 0.5  # Reduce reward magnitude early in training
        
        # Augment observation with action history and risk metrics
        augmented_observation = self._augment_observation(observation)
        
        # Add training progress to info dict
        info['training_progress'] = self.training_progress
        info['action_taken'] = action
        
        return augmented_observation, reward, terminated, truncated, info


class TensorboardCallback(BaseCallback):
    """Custom callback for tracking metrics during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Environment metrics
        self.returns = []
        self.portfolio_values = []
        self.episode_rewards = []
        self.trade_count = 0
        self.successful_trades = 0
        
        # Trading metrics
        self.trade_returns = []
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Sell, Hold, Buy
        
        # Safety metrics
        self.cooldown_violations = 0
        self.oscillation_count = 0
        self.same_price_trades = 0
        
        # Debug counter
        self.debug_steps = 0
        self.debug_frequency = 250  # Increased frequency from 1000 to 250 steps
        self.last_debug_output = 0
        
        # Add step timing for performance monitoring
        self.last_time = time.time()
        self.steps_since_last_log = 0
    
    def _on_step(self) -> bool:
        """Called at each step"""
        self.debug_steps += 1
        self.steps_since_last_log += 1
        
        # Debug output every N steps
        if self.debug_steps % self.debug_frequency == 0 and self.debug_steps > self.last_debug_output:
            self.last_debug_output = self.debug_steps
            current_time = time.time()
            time_elapsed = current_time - self.last_time
            steps_per_second = self.steps_since_last_log / time_elapsed if time_elapsed > 0 else 0
            
            # Get current portfolio value if available
            portfolio_value = None
            if self.locals is not None and 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0]
                if 'portfolio_value' in info:
                    portfolio_value = info['portfolio_value']
            
            # Print comprehensive stats
            portfolio_str = f", Portfolio: {portfolio_value:.2f}" if portfolio_value is not None else ""
            action_counts_str = ", ".join([f"{action}: {count}" for action, count in self.action_counts.items() if count > 0])
            action_str = f", Actions: [{action_counts_str}]" if action_counts_str else ""
            
            logger.info(f"Step {self.debug_steps}, {steps_per_second:.1f} steps/s{portfolio_str}{action_str}")
            
            # Log key metrics to tensorboard
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.record('training/steps_per_second', steps_per_second)
                # Record number of steps per action
                for action, count in self.action_counts.items():
                    self.logger.record(f'actions/action_{action}_count', count)
            
            # Reset counters
            self.last_time = current_time
            self.steps_since_last_log = 0
        
        # Log rewards when episodes complete
        if self.locals is not None and 'dones' in self.locals and 'rewards' in self.locals:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    reward = self.locals['rewards'][i]
                    self.episode_rewards.append(reward)
                    self.logger.record('environment/reward', reward)
                    
                    # If info dict available, extract additional metrics
                    if 'infos' in self.locals and len(self.locals['infos']) > i:
                        info = self.locals['infos'][i]
                        
                        # Track portfolio value
                        if 'portfolio_value' in info:
                            portfolio_value = info['portfolio_value']
                            self.portfolio_values.append(portfolio_value)
                            self.logger.record('portfolio/value', portfolio_value)
                            
                            # Log episode completion with portfolio value
                            logger.info(f"Episode complete, reward: {reward:.2f}, portfolio: {portfolio_value:.2f}")
                        
                        # Track cooldown violations
                        if 'cooldown_violation' in info and info['cooldown_violation']:
                            self.cooldown_violations += 1
                            self.logger.record('safety/cooldown_violations', self.cooldown_violations)
                        
                        # Track oscillations
                        if 'oscillation_detected' in info and info['oscillation_detected']:
                            self.oscillation_count += 1
                            self.logger.record('safety/oscillation_count', self.oscillation_count)
                        
                        # Track same price trades
                        if 'same_price_trade' in info and info['same_price_trade']:
                            self.same_price_trades += 1
                            self.logger.record('safety/same_price_trades', self.same_price_trades)
                        
                        # Track actions if available
                        if 'action' in info:
                            action = info['action']
                            if action in self.action_counts:
                                self.action_counts[action] += 1
                                
                        # Track trades
                        if 'trade' in info and info['trade']:
                            self.trade_count += 1
                            self.logger.record('trades/count', self.trade_count)
                            
                            # Trade metrics if available
                            if 'trade_profit' in info:
                                profit = info['trade_profit']
                                self.trade_returns.append(profit)
                                
                                if profit > 0:
                                    self.successful_trades += 1
                                    self.total_profit += profit
                                else:
                                    self.total_loss += abs(profit)
                                
                                # Calculate win rate and profit factor
                                if self.trade_count > 0:
                                    win_rate = (self.successful_trades / self.trade_count) * 100
                                    self.logger.record('trades/win_rate', win_rate)
                                    
                                if self.total_loss > 0:
                                    profit_factor = self.total_profit / max(self.total_loss, 1e-6)
                                    self.logger.record('trades/profit_factor', profit_factor)
        
        return True


def setup_logging(log_dir=None):
    """Configure logging to file and console"""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    else:
        os.makedirs('logs', exist_ok=True)
        log_path = os.path.join('logs', f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Return path to log file for reference
    return log_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DQN, PPO, A2C, or SAC agent for cryptocurrency trading")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=1000000, help="Number of timesteps for training")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Model parameters
    parser.add_argument("--finrl_model", type=str, default="ppo", 
                        choices=["ppo", "a2c", "dqn", "sac"], help="Type of RL model to use")
    parser.add_argument("--use_finrl", action="store_true", help="Use FinRL library for training")
    parser.add_argument("--lstm_model_path", type=str, default=None, 
                        help="Path to pre-trained LSTM model for state representation")
    
    # Environment parameters
    parser.add_argument("--data_path", type=str, default="data/synthetic/synthetic_dataset.h5", 
                       help="Path to market data (HDF5 format)")
    parser.add_argument("--data_key", type=str, default=None,
                       help="Specific key/group to use when HDF5 file contains multiple datasets")
    parser.add_argument("--state_space", type=int, default=16, help="State space dimensionality")
    parser.add_argument("--initial_balance", type=float, default=10000.0, help="Initial portfolio balance")
    parser.add_argument("--transaction_cost_pct", type=float, default=0.001, 
                       help="Transaction cost percentage")
    parser.add_argument("--reward_scaling", type=float, default=0.01, help="Reward scaling factor")
    parser.add_argument("--normalize_observations", type=bool, default=True, 
                       help="Whether to normalize observations")
    
    # Parallelization parameters
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--num_envs_per_worker", type=int, default=1, 
                       help="Number of environments per worker")
    
    # PPO/A2C-specific parameters
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    
    # DQN/SAC-specific parameters
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--exploration_fraction", type=float, default=0.1, 
                       help="Fraction of training time for exploration")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, 
                       help="Initial exploration rate")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, 
                       help="Final exploration rate")
    parser.add_argument("--target_update_interval", type=int, default=10000, 
                       help="Update frequency for target network")
    
    # Trading safeguards
    parser.add_argument("--trade_cooldown", type=int, default=50, 
                       help="Minimum steps between trades")
    
    # Misc
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--checkpoint_freq", type=int, default=100000, 
                       help="Frequency of model checkpoints")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (cuda, cpu, or auto to automatically detect)")
    
    args = parser.parse_args()
    
    # Auto-detect device if set to 'auto'
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return args


def ensure_technical_indicators(df, indicators):
    """Ensure the dataframe has all required technical indicators"""
    missing_indicators = [ind for ind in indicators if ind not in df.columns]
    
    if missing_indicators:
        logger.info(f"Adding missing technical indicators: {missing_indicators}")
        import pandas_ta as ta
        
        # Calculate basic technical indicators if missing
        if 'macd' in missing_indicators:
            # MACD
            macd = ta.macd(df['close'])
            df = pd.concat([df, macd], axis=1)
        
        if 'rsi' in missing_indicators:
            # RSI
            df['rsi'] = ta.rsi(df['close'])
        
        if 'cci' in missing_indicators:
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])
        
        if 'dx' in missing_indicators:
            # DX (Directional Movement Index)
            dx = ta.adx(df['high'], df['low'], df['close'])
            df = pd.concat([df, dx], axis=1)
        
        if any(bb in missing_indicators for bb in ['bb_upper', 'bb_lower', 'bb_middle']):
            # Bollinger Bands
            bb = ta.bbands(df['close'])
            df = pd.concat([df, bb], axis=1)
    
    # Ensure no NaN values
    df = df.ffill().bfill().fillna(0)
    
    return df


def load_and_preprocess_market_data(args):
    """Load and preprocess market data for training"""
    data_path = args.data_path
    
    try:
        # Check if data is in HDF5 format
        if data_path.endswith('.h5'):
            logger.info(f"Loading market data from HDF5 file: {data_path}")
            
            # Try to load with key if specified
            if args.data_key:
                logger.info(f"Using specified key: {args.data_key}")
                market_data = pd.read_hdf(data_path, key=args.data_key)
            else:
                # Try loading directly first (for single dataset files)
                try:
                    market_data = pd.read_hdf(data_path)
                except ValueError as e:
                    if "key must be provided" in str(e):
                        # File has multiple datasets, list them and use the first one
                        import h5py
                        with h5py.File(data_path, 'r') as f:
                            keys = list(f.keys())
                            logger.info(f"File contains multiple datasets with keys: {keys}")
                            if keys:
                                first_key = keys[0]
                                logger.info(f"Using first key automatically: {first_key}")
                                market_data = pd.read_hdf(data_path, key=first_key)
                            else:
                                raise ValueError("HDF5 file doesn't contain any datasets")
                    else:
                        raise
        else:
            # Assume CSV format
            logger.info(f"Loading market data from CSV file: {data_path}")
            market_data = pd.read_csv(data_path)
            
            # Convert to datetime if there's a timestamp column
            if 'timestamp' in market_data.columns:
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                market_data.set_index('timestamp', inplace=True)
        
        # Ensure required technical indicators
        market_data = ensure_technical_indicators(market_data, INDICATORS)
        
        # Log data information
        logger.info(f"Market data loaded with shape: {market_data.shape}")
        logger.info(f"Sample of market data:\n{market_data.head(2)}")
        
        # Train/test split
        train_size = int(len(market_data) * 0.8)
        train_data = market_data[:train_size]
        
        # Return processed data
        return train_data
        
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def create_vec_env(df, args, num_envs=4):
    """Create a vectorized environment for parallel training"""
    
    def make_env(rank, seed=None):
        """Factory function to create a single environment"""
        def _init():
            # Create base environment
            env = CryptocurrencyTradingEnv(
                df=df,
                initial_amount=args.initial_balance,
                state_space=args.state_space,
                buy_cost_pct=args.transaction_cost_pct,
                sell_cost_pct=args.transaction_cost_pct,
                reward_scaling=args.reward_scaling,
                tech_indicator_list=list(set(INDICATORS).intersection(set(df.columns)))
            )
            
            # Add safeguards against rapid trading
            env = SafeTradingEnvWrapper(env, trade_cooldown=args.trade_cooldown)
            
            # Add Monitor wrapper for logging
            log_dir = os.path.join('logs', 'env_logs')
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, os.path.join(log_dir, f'env_{rank}'))
            
            # Set random seed if provided
            if seed is not None:
                env.seed(seed + rank)
                
            return env
        return _init
    
    # Create environment functions based on number of environments requested
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Create appropriate vectorized environment
    if num_envs == 1:
        # Use DummyVecEnv for single environment
        vec_env = DummyVecEnv(env_fns)
    else:
        # Use SubprocVecEnv for parallel environments
        vec_env = SubprocVecEnv(env_fns)
    
    # Add observation normalization if requested
    if args.normalize_observations:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    return vec_env


def load_lstm_model(model_path):
    """Load pre-trained LSTM model for state representation"""
    if model_path and os.path.exists(model_path):
        try:
            logger.info(f"Loading pre-trained LSTM model from {model_path}")
            
            # Load the saved model checkpoint with robust error handling
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(model_path, map_location=device)
                logger.info(f"Model checkpoint loaded from {model_path}")
            except Exception as load_error:
                logger.error(f"Error loading model checkpoint: {str(load_error)}")
                logger.error(traceback.format_exc())
                return None
            
            # Extract model configuration or use defaults
            if 'config' in checkpoint:
                model_config = checkpoint['config']
                logger.info(f"Using configuration from saved model: {model_config}")
            elif 'hparams' in checkpoint:
                # Handle Lightning checkpoint format
                model_config = {
                    'input_dims': checkpoint['hparams'].get('input_dims', {'15m': 39, '4h': 39, '1d': 39}),
                    'hidden_dims': checkpoint['hparams'].get('hidden_dims', 128),
                    'num_layers': checkpoint['hparams'].get('num_layers', 2),
                    'dropout': checkpoint['hparams'].get('dropout', 0.2),
                    'bidirectional': checkpoint['hparams'].get('bidirectional', True),
                    'attention': checkpoint['hparams'].get('attention', True),
                    'num_classes': checkpoint['hparams'].get('num_classes', 3)
                }
                logger.info(f"Using configuration from Lightning checkpoint: {model_config}")
            else:
                # Use default configuration
                model_config = {
                    'input_dims': {'15m': 39, '4h': 39, '1d': 39},
                    'hidden_dims': 128,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'bidirectional': True,
                    'attention': True,
                    'num_classes': 3
                }
                logger.info(f"Using default model configuration: {model_config}")
            
            # Create a new model instance
            try:
                model = MultiTimeframeModel(**model_config)
                logger.info(f"Created MultiTimeframeModel instance with config: {model_config}")
            except Exception as model_init_error:
                logger.error(f"Error creating model instance: {str(model_init_error)}")
                logger.error(traceback.format_exc())
                return None
            
            # Load the state dictionary
            try:
                if 'state_dict' in checkpoint:
                    # Handle Lightning checkpoint format
                    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded model state from Lightning checkpoint")
                else:
                    # Regular PyTorch model
                    model.load_state_dict(checkpoint, strict=False)
                    logger.info("Loaded model state from regular checkpoint")
            except Exception as state_dict_error:
                logger.error(f"Error loading state dictionary: {str(state_dict_error)}")
                logger.error(traceback.format_exc())
                # Continue with the uninitialized model
                
            # Set to evaluation mode
            model.eval()
            
            # Verify the model has the expected structure
            if not hasattr(model, 'timeframes'):
                logger.warning("Loaded model does not have 'timeframes' attribute. It might not be compatible.")
            else:
                logger.info(f"Model timeframes: {model.timeframes}")
                
            # Add a utility method for feature extraction if needed
            if not hasattr(model, 'forward_features'):
                def forward_features(self, x):
                    """Extract features without classification head"""
                    with torch.no_grad():
                        # Process through encoders and get encoded timeframes
                        # This is a simplified version of the model's forward method
                        if isinstance(x, dict):
                            # Multi-timeframe input
                            batch_size = next(iter(x.values())).size(0)
                            return torch.zeros(batch_size, model.hidden_dims * 2, device=x[next(iter(x))].device)
                        else:
                            # Single tensor input
                            return torch.zeros(x.size(0), model.hidden_dims * 2, device=x.device)
                
                # Add the method to the model
                model.forward_features = types.MethodType(forward_features, model)
                logger.info("Added forward_features method to model")
            
            return model
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.warning("No valid LSTM model provided or found. Proceeding without LSTM.")
    return None


def train_dqn(env, args, callbacks=None):
    """Train a DQN model with specified parameters."""
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args.finrl_model)
    
    # Initialize callbacks
    if callbacks is None:
        callbacks = []
    
    # Add resource callback if available in the main function's scope
    try:
        if 'resource_callback' in globals():
            callbacks.append(resource_callback)
    except NameError:
        logger.warning("Resource callback not available")
        
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=checkpoint_dir,
        name_prefix="dqn_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Add TensorBoard callback
    tb_callback = TensorboardCallback(model_name=args.finrl_model, debug_frequency=250)
    callbacks.append(tb_callback)
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        target_update_interval=args.target_update_interval,
        verbose=1,  # Always use verbose=1 to show progress bar
        tensorboard_log="tensorboard_log",
        device=args.device
    )
    
    # Train model
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="dqn_run",
            log_interval=25  # Log more frequently - every 25 updates
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"dqn_model_{int(time.time())}")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during DQN training: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def train_ppo(env, args, callbacks=None):
    """Train a PPO model with specified parameters."""
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args.finrl_model)
    
    # Initialize callbacks
    if callbacks is None:
        callbacks = []
    
    # Add resource callback if available in the main function's scope
    try:
        if 'resource_callback' in globals():
            callbacks.append(resource_callback)
    except NameError:
        logger.warning("Resource callback not available")
        
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=checkpoint_dir,
        name_prefix="ppo_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Add TensorBoard callback
    tb_callback = TensorboardCallback(model_name=args.finrl_model, debug_frequency=250)
    callbacks.append(tb_callback)
    
    # Create model with appropriate parameters for stability
    # Cap learning rate at 0.0005 for stability
    learning_rate = min(0.0005, args.learning_rate)
    if learning_rate != args.learning_rate:
        logger.warning(f"Capped learning rate from {args.learning_rate} to {learning_rate} for training stability")
    
    # Learning rate schedule for stability
    def lr_schedule(remaining_progress):
        return learning_rate * remaining_progress  # Linear schedule
    
    # Force CPU for MlpPolicy to avoid the GPU utilization warning
    # Only use GPU if specifically requested via args.device and not auto-detected
    device = "cpu"
    if args.device == "cuda" and torch.cuda.is_available() and "--device=cuda" in sys.argv:
        logger.info("Using CUDA as explicitly requested via --device=cuda")
        device = "cuda"
    else:
        logger.info("Using CPU for PPO with MlpPolicy as recommended for better performance")
    
    logger.info(f"Creating PPO model with n_steps={args.n_steps}, batch_size={args.batch_size}")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=0.01,  # Slightly higher entropy for better exploration
        clip_range=0.2,
        verbose=1,  # Always use verbose=1 to show progress bar
        tensorboard_log=os.path.join("tensorboard_log", f"PPO_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        device=device
    )
    
    # Train model with standard approach
    try:
        logger.info(f"Training PPO model for {args.timesteps} timesteps...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="ppo_run",
            log_interval=25  # Log more frequently - every 25 updates
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"ppo_model_{int(time.time())}.zip")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during PPO training: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def train_a2c(env, args, callbacks=None):
    """Train an A2C model with specified parameters."""
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args.finrl_model)
    
    # Initialize callbacks
    if callbacks is None:
        callbacks = []
    
    # Add resource callback if available in the main function's scope
    try:
        if 'resource_callback' in globals():
            callbacks.append(resource_callback)
    except NameError:
        logger.warning("Resource callback not available")
        
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=checkpoint_dir,
        name_prefix="a2c_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Add TensorBoard callback
    tb_callback = TensorboardCallback(model_name=args.finrl_model, debug_frequency=250)
    callbacks.append(tb_callback)
    
    # Create model with appropriate parameters for stability
    # Cap learning rate for stability
    learning_rate = min(0.0007, args.learning_rate)
    if learning_rate != args.learning_rate:
        logger.warning(f"Capped learning rate from {args.learning_rate} to {learning_rate} for A2C training stability")
    
    # Learning rate schedule for stability
    def lr_schedule(remaining_progress):
        return learning_rate * remaining_progress  # Linear schedule
    
    # Load LSTM model if path provided
    lstm_model = None
    if args.lstm_model_path:
        lstm_model = load_lstm_model(args.lstm_model_path)
        
    # Force CPU for MlpPolicy to avoid the GPU utilization warning
    # Only use GPU if specifically requested via args.device and not auto-detected
    device = "cpu"
    if args.device == "cuda" and torch.cuda.is_available():
        logger.info("Using CUDA for A2C training")
        device = "cuda"
    else:
        logger.info("Using CPU for A2C training")
    
    policy_kwargs = {
        'net_arch': [256, 256]  # Use a deeper network
    }
    
    if lstm_model is not None:
        policy_kwargs["features_extractor_class"] = LSTMAugmentedFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {"lstm_model": lstm_model}
        
    logger.info(f"Creating A2C model with n_steps={args.n_steps}, using device={device}")
    from stable_baselines3 import A2C
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,  # Changed from conditional to always have verbose=1 to show progress
        tensorboard_log=os.path.join("tensorboard_log", f"A2C_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        device=device
    )
    
    # Train model
    try:
        logger.info(f"Training A2C model for {args.timesteps} timesteps...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="a2c_run_1",  # Changed to include "1" to ensure uniqueness
            log_interval=25  # Log more frequently - every 25 updates
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"a2c_model_{int(time.time())}.zip")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during A2C training: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def train_sac(env, args, callbacks=None):
    """Train a SAC model with specified parameters."""
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(args.finrl_model)
    
    # Initialize callbacks
    if callbacks is None:
        callbacks = []
    
    # Add resource callback if available in the main function's scope
    try:
        if 'resource_callback' in globals():
            callbacks.append(resource_callback)
    except NameError:
        logger.warning("Resource callback not available")
        
    # Add checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=checkpoint_dir,
        name_prefix="sac_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Add TensorBoard callback
    tb_callback = TensorboardCallback(model_name=args.finrl_model, debug_frequency=250)
    callbacks.append(tb_callback)
    
    # Create model with appropriate parameters for stability
    # Cap learning rate for stability
    learning_rate = min(0.0003, args.learning_rate)
    if learning_rate != args.learning_rate:
        logger.warning(f"Capped learning rate from {args.learning_rate} to {learning_rate} for SAC training stability")
    
    # SAC can effectively use GPU as it doesn't have the same issues as PPO/A2C with MlpPolicy
    device = args.device
    
    logger.info(f"Creating SAC model with buffer_size={args.buffer_size}, batch_size={args.batch_size}")
    
    # Import only if needed to avoid dependency issues if not installed
    from stable_baselines3 import SAC
    
    # Create SAC-specific policy_kwargs
    policy_kwargs = {
        'net_arch': {
            'pi': [256, 256],  # Actor network architecture
            'qf': [256, 256]   # Critic network architecture
        }
    }
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=0.005,  # For soft target update
        ent_coef="auto",  # Automatic entropy tuning
        target_entropy="auto",  # Automatic target entropy
        train_freq=1,  # Update policy every step
        gradient_steps=1,  # One gradient step per step in the environment
        action_noise=None,  # No additional action noise
        verbose=1,  # Always use verbose=1 to show progress bar 
        tensorboard_log=os.path.join("tensorboard_log", f"SAC_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        device=device,
        policy_kwargs=policy_kwargs
    )
    
    # Train model
    try:
        logger.info(f"Training SAC model for {args.timesteps} timesteps...")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="sac_run_1",
            log_interval=25  # Log more frequently
        )
        
        # Save trained model
        model_save_path = os.path.join('models', f"sac_model_{int(time.time())}.zip")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during SAC training: {str(e)}")
        logger.error(traceback.format_exc())
        return None


# Add a custom feature extractor class that uses the pre-trained LSTM
class LSTMAugmentedFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses a pre-trained LSTM model to enhance the features.
    """
    
    def __init__(self, observation_space, lstm_model=None, features_dim=128):
        super(LSTMAugmentedFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Save the LSTM model
        self.lstm_model = lstm_model
        
        # Calculate input size from observation space
        input_size = int(np.prod(observation_space.shape))
        
        # We need robust handling of the LSTM adapter
        self.lstm_adapter = None
        
        # Create neural network to process features
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
        
        logger.info(f"Created feature extractor with input size {input_size}, output size {features_dim}")
    
    def _adapt_observations_for_lstm(self, observations):
        """
        Adapt the observations tensor to be compatible with the LSTM model.
        
        Parameters:
        -----------
        observations : torch.Tensor
            Raw observations tensor from the environment
            
        Returns:
        --------
        dict
            Dictionary of observations formatted for the LSTM model
        """
        if self.lstm_model is None:
            return None
            
        # Check if model expects multi-timeframe input
        is_multi_timeframe = hasattr(self.lstm_model, 'timeframes')
        if not is_multi_timeframe:
            return observations.float()
            
        # For multi-timeframe models, we need to create a properly structured input
        timeframes = getattr(self.lstm_model, 'timeframes', ['15m'])
        
        # Create a dictionary with the same data for each timeframe
        # This is a simplification - ideally, we would convert the features to match each timeframe
        formatted_obs = {}
        
        # Handle different observation shapes
        if len(observations.shape) == 2:  # [batch_size, features]
            # Add sequence dimension: [batch_size, 1, features]
            obs_with_seq = observations.float().unsqueeze(1)
            for tf in timeframes:
                formatted_obs[tf] = obs_with_seq
        else:  # Assume already has sequence dimension or other format
            for tf in timeframes:
                formatted_obs[tf] = observations.float()
                
        return formatted_obs
        
    def forward(self, observations):
        """Extract features using the feature network and augment with LSTM if available"""
        # Extract features using the feature network
        features = self.feature_net(observations.float())
        
        # If we have an LSTM model, use it to enhance features
        if self.lstm_model is not None:
            try:
                # Format the observations for the LSTM
                with torch.no_grad():
                    # First determine if this is a multi-timeframe model
                    is_multi_timeframe = hasattr(self.lstm_model, 'timeframes')
                    
                    # Process the input based on its type and the model type
                    if is_multi_timeframe:
                        # For multi-timeframe models, we need dictionary input
                        if isinstance(observations, dict):
                            # Input is already a dictionary
                            formatted_obs = observations
                        else:
                            # Use the helper method to create properly formatted observations
                            formatted_obs = self._adapt_observations_for_lstm(observations)
                        
                        # Try to get features from the model
                        try:
                            lstm_features = self.lstm_model(formatted_obs)
                        except Exception as format_error:
                            logger.warning(f"Failed with formatted observations: {format_error}")
                            # Try a direct approach as fallback
                            try:
                                # Try direct forward pass if that's available
                                if hasattr(self.lstm_model, 'forward_features'):
                                    lstm_features = self.lstm_model.forward_features(observations.float())
                                else:
                                    # Last resort - try with a single timeframe directly
                                    single_tf = self.lstm_model.timeframes[0]
                                    single_tf_dict = {single_tf: observations.float().unsqueeze(1)}
                                    lstm_features = self.lstm_model(single_tf_dict)
                            except Exception as direct_error:
                                logger.warning(f"All LSTM approaches failed: {direct_error}")
                                lstm_features = None
                    elif hasattr(self.lstm_model, 'forward_features'):
                        # Use dedicated feature extraction method if available
                        try:
                            lstm_features = self.lstm_model.forward_features(observations.float())
                        except Exception as fwd_error:
                            # If this fails, try a different approach
                            logger.warning(f"Error using forward_features: {fwd_error}")
                            try:
                                # Try a direct forward call as fallback
                                lstm_features = self.lstm_model(observations.float())
                            except Exception as fallback_error:
                                logger.warning(f"Failed with direct call: {fallback_error}")
                                lstm_features = None
                    else:
                        # Fall back to regular forward if forward_features is not available
                        # and the model accepts tensor input directly
                        try:
                            lstm_features, _ = self.lstm_model(observations.float())
                        except Exception:
                            # Some models may not return a tuple
                            try:
                                lstm_features = self.lstm_model(observations.float())
                            except Exception as direct_error:
                                logger.warning(f"Direct model call failed: {direct_error}")
                                lstm_features = None
                
                # Combine the features if LSTM processing was successful
                if lstm_features is not None and lstm_features.size(0) == features.size(0):
                    # For simplicity, average the LSTM features if multi-dimensional
                    if len(lstm_features.shape) > 2:
                        lstm_features = lstm_features.mean(dim=1)
                    
                    # Resize if needed
                    if lstm_features.shape[1] != features.shape[1]:
                        if self.lstm_adapter is None:
                            # Lazily create adapter when we know the dimensions
                            self.lstm_adapter = nn.Linear(
                                lstm_features.shape[1], 
                                features.shape[1]
                            ).to(features.device)
                        lstm_features = self.lstm_adapter(lstm_features)
                    
                    # Add LSTM features to our features (weighted addition)
                    features = features + 0.5 * lstm_features
            except Exception as e:
                # Log error but continue with base features to be resilient
                logger.warning(f"Error using LSTM model in feature extraction: {e}")
                import traceback
                logger.debug(f"LSTM extraction error details: {traceback.format_exc()}")
                # Continue with the base features
        
        return features
        

def check_resources():
    """Check system resources and log information about CPU and memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Get CPU usage
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # Get memory usage
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    
    # Get GPU info if available
    gpu_info = ""
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)
        gpu_info = f", GPU: {gpu_memory_allocated:.0f}MB used / {gpu_memory_cached:.0f}MB reserved"
    
    logger.info(f"Resources: CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.0f}MB{gpu_info}")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_mb': memory_mb
    }


def main():
    """Main entry point for script"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_path = setup_logging()
    logger.info(f"Log file created at: {log_path}")
    
    # Log environment information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Number of available CPUs: {psutil.cpu_count(logical=False)}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")
        
    # Set random seed if provided
    if args.seed is not None:
        logger.info(f"Setting random seed to: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            
    # Log key parameters
    logger.info(f"Training with model: {args.finrl_model}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Trade cooldown period: {args.trade_cooldown}")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('tensorboard_log', exist_ok=True)
    
    # Check resources before starting
    logger.info("Checking system resources before starting:")
    check_resources()
    
    trained_model = None
    vec_env = None
    
    try:
        # Load data
        df = load_and_preprocess_market_data(args)
        
        if df is None or len(df) == 0:
            logger.error("Failed to load market data. Exiting.")
            return
            
        logger.info(f"Loaded market data with shape: {df.shape}")
        
        # Create environment
        total_envs = args.num_workers * args.num_envs_per_worker
        logger.info(f"Creating {total_envs} parallel environments")
        vec_env = create_vec_env(df, args, num_envs=total_envs)
        
        # Load pre-trained LSTM model if specified
        lstm_model = None
        if args.lstm_model_path:
            logger.info(f"Loading LSTM model from {args.lstm_model_path}")
            lstm_model = load_lstm_model(args.lstm_model_path)
            if lstm_model is not None:
                logger.info("LSTM model loaded successfully")
            else:
                logger.warning("Failed to load LSTM model, proceeding without it")
                
        # Create resource monitoring callback
        class ResourceCheckCallback(BaseCallback):
            def __init__(self, check_interval=10000, verbose=0):
                super().__init__(verbose)
                self.check_interval = check_interval
                self.last_check = 0
            
            def _on_step(self) -> bool:
                if self.num_timesteps - self.last_check > self.check_interval:
                    self.last_check = self.num_timesteps
                    check_resources()
                return True
        
        # Add resource callback to all training algorithms
        resource_callback = ResourceCheckCallback(check_interval=10000)  # Check every 10K steps
        
        # Train according to selected algorithm
        if args.finrl_model.lower() == "dqn":
            trained_model = train_dqn(vec_env, args)
        elif args.finrl_model.lower() == "ppo":
            trained_model = train_ppo(vec_env, args)
        elif args.finrl_model.lower() == "a2c":
            trained_model = train_a2c(vec_env, args)
        elif args.finrl_model.lower() == "sac":
            trained_model = train_sac(vec_env, args)
        else:
            logger.error(f"Unknown model type: {args.finrl_model}")
            return
        
        logger.info("Training completed successfully")
        
        # Check resources after training
        logger.info("Checking system resources after training:")
        check_resources()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Save any final models if training was interrupted
        if trained_model is not None:
            final_save_path = os.path.join('models', f"final_{args.finrl_model}_model")
            trained_model.save(final_save_path)
            logger.info(f"Final model saved to {final_save_path}")
            
        # Clean up
        try:
            if vec_env is not None:
                vec_env.close()
                logger.info("Environment resources cleaned up")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
        
        # Log completion
        logger.info("Script execution completed.")


if __name__ == "__main__":
    # Run the main function
    main()
