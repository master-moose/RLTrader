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

# Constants for trading safeguards - ADJUSTING FOR MORE AGGRESSIVE TRADING
TRADE_COOLDOWN_PERIOD = 15  # Reduced from 25 to 15 - allows even more frequent trading
OSCILLATION_PENALTY = 120.0  # Reduced from 250.0 to 120.0 - much less penalty for oscillation
SAME_PRICE_TRADE_PENALTY = 250.0  # Reduced from 500.0 to 250.0
MAX_TRADE_FREQUENCY = 0.20  # Increased from 0.15 to 0.20 - allow up to 20% of steps to be trades

# Class for LSTM feature extraction
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses a pre-trained LSTM model for state representation.
    
    The LSTM model is loaded from a saved state dict and used to extract features
    from the environment observations before passing them to the RL policy.
    """
    
    def __init__(self, observation_space, lstm_state_dict=None, features_dim=64):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: The observation space of the environment
            lstm_state_dict: The state dict of the pre-trained LSTM model
            features_dim: The dimension of the features to extract
        """
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Initialize the LSTM model
        self.lstm_model = None
        # Don't set features_dim directly as it's already set by the parent class
        # self.features_dim = features_dim  # This line is causing the error
        
        if lstm_state_dict is not None:
            # Create LSTM model architecture - this should match the saved model
            self.lstm_model = torch.nn.LSTM(
                input_size=observation_space.shape[0], 
                hidden_size=features_dim,
                num_layers=1,
                batch_first=True
            )
            
            # Try to load the state dict
            try:
                # Use partial loading if the state dicts don't match exactly
                if hasattr(lstm_state_dict, 'items'):  # It's a dictionary
                    # Extract LSTM weights from the state dict
                    filtered_state_dict = {}
                    for key, value in lstm_state_dict.items():
                        if key.startswith('lstm') or key.startswith('encoder') or key.startswith('rnn'):
                            filtered_state_dict[key] = value
                    
                    # Log what we're loading
                    logger.info(f"Loading LSTM weights: {list(filtered_state_dict.keys())}")
                    
                    # Load the filtered state dict
                    if filtered_state_dict:
                        # Try to load with strict=False to allow partial loading
                        self.lstm_model.load_state_dict(filtered_state_dict, strict=False)
                        logger.info("LSTM weights loaded successfully")
                    else:
                        logger.warning("No LSTM weights found in the state dict")
                else:
                    logger.warning("LSTM state dict is not a dictionary")
            except Exception as e:
                logger.error(f"Error loading LSTM weights: {e}")
                logger.error(traceback.format_exc())
        
        # If LSTM model is not available, use a simple linear layer
        if self.lstm_model is None:
            logger.warning("Using linear layer for feature extraction instead of LSTM")
            self.linear = torch.nn.Linear(observation_space.shape[0], features_dim)
            
        # Output layer to ensure proper dimension
        self.output_layer = torch.nn.Linear(features_dim, features_dim)
        
    def forward(self, observations):
        """
        Extract features from observations using the LSTM model.
        
        Args:
            observations: The observations from the environment
            
        Returns:
            torch.Tensor: The extracted features
        """
        # Add sequence dimension if not present
        if len(observations.shape) == 2:
            # For batched observations without sequence dimension
            # Shape from [batch_size, features] to [batch_size, 1, features]
            observations = observations.unsqueeze(1)
        
        if self.lstm_model is not None:
            # Use LSTM for feature extraction
            try:
                lstm_out, _ = self.lstm_model(observations)
                # Take the last time step output
                features = lstm_out[:, -1, :]
            except Exception as e:
                logger.error(f"Error in LSTM forward pass: {e}")
                # Fallback to linear layer
                features = self.output_layer(observations[:, -1, :])
        else:
            # Use linear layer
            features = self.linear(observations[:, -1, :])
        
        return self.output_layer(features)

class SafeTradingEnvWrapper(gymnasium.Wrapper):
    """
    A wrapper for trading environments that adds safeguards against:
    1. Rapid trading (enforces a cooldown period)
    2. Action oscillation (penalizes rapid changes between buy and sell)
    3. Implements proper risk management strategies
    """
    
    def __init__(self, env, trade_cooldown=TRADE_COOLDOWN_PERIOD, max_history_size=100, max_risk_per_trade=0.02):
        """Initialize the wrapper with safeguards against harmful trading patterns"""
        super().__init__(env)
        
        # Trading safeguards
        self.trade_cooldown = trade_cooldown
        # Reduce cooldown as training progresses
        self.min_cooldown = max(3, trade_cooldown // 5)  # Minimum cooldown period reduced from 4 to 5 divisor
        
        # Risk management parameters
        self.max_risk_per_trade = max_risk_per_trade  # Maximum risk per trade (% of portfolio)
        self.target_risk_reward_ratio = 2.0  # Target risk-reward ratio (reward should be 2x the risk)
        self.risk_adjusted_position_sizing = True  # Use risk-adjusted position sizing
        self.cumulative_risk = 0.0  # Track cumulative risk across open positions
        self.max_cumulative_risk = 0.06  # Maximum 6% portfolio risk across all positions
        
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
        self.min_profit_threshold = 0.001  # Reduced from 0.002 to 0.001 - requires only 0.1% difference
        
        # Stronger oscillation detection and prevention
        self.oscillation_window = 8  # Look at 8 actions for oscillation patterns
        self.progressive_cooldown = True  # Increase cooldown after oscillations
        self.max_oscillation_cooldown = trade_cooldown * 2  # Reduced from 3x to 2x
        self.oscillation_patterns = {
            'buy_sell_alternation': 0,  # Count of buy-sell alternations
            'rapid_reversals': 0,       # Count of position reversals
        }
        
        # Track trading metrics for risk-aware rewards
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.peak_value = self.env.portfolio_value if hasattr(self.env, 'portfolio_value') else 0.0
        
        self.current_cooldown = trade_cooldown  # Adjustable cooldown period
        
        # Track metrics for analyzing agent behavior
        self.hold_duration = 0
        self.successful_trade_streak = 0
        self.max_successful_streak = 0
        self.profitable_trades = 0
        self.total_trades = 0
        self.win_rate = 0.0
        
        # Add a dynamic profit threshold that decreases over time
        self.initial_profit_threshold = 0.001  # Reduced from 0.002 to 0.001
        self.min_profit_threshold = 0.0003  # Reduced from 0.0005 to 0.0003
        
        # Portfolio growth tracking for better rewards
        self.starting_portfolio = self.env.portfolio_value if hasattr(self.env, 'portfolio_value') else 0.0
        self.highest_portfolio = self.starting_portfolio
        self.portfolio_growth_rate = 0.0
        
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
        
        # Reset portfolio growth tracking
        self.starting_portfolio = self.env.portfolio_value if hasattr(self.env, 'portfolio_value') else 0.0
        self.highest_portfolio = self.starting_portfolio
        self.portfolio_growth_rate = 0.0
        
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
        
        # Maintain highest portfolio value for growth rewards
        if portfolio_value > self.highest_portfolio:
            # Calculate growth rate for rewards
            prev_highest = self.highest_portfolio
            self.highest_portfolio = portfolio_value
            self.portfolio_growth_rate = (self.highest_portfolio - prev_highest) / max(prev_highest, 1.0)
        
        # Calculate drawdown - but make it less impactful
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
        
        # Add portfolio growth reward (significant boost)
        if portfolio_value > self.starting_portfolio:
            growth_pct = (portfolio_value - self.starting_portfolio) / self.starting_portfolio
            # Log detailed growth information for debugging at less frequent intervals
            if current_step % 100000 == 0 or (trade_occurred and current_step % 10000 == 0):
                logger.info(f"Portfolio growth: {growth_pct:.4f} (starting: {self.starting_portfolio:.2f}, current: {portfolio_value:.2f})")
            growth_reward = min(growth_pct * 3.0, 3.0)  # Increased from 2.0 to 3.0
            risk_adjusted_reward += growth_reward
            
            # Extra reward for achieving new highs
            if self.portfolio_growth_rate > 0:
                new_high_reward = min(self.portfolio_growth_rate * 5.0, 1.0)
                risk_adjusted_reward += new_high_reward
        
        # Add drawdown penalty - but make it much less severe
        if self.max_drawdown > 0.15:  # Only penalize significant drawdowns (>15%)
            drawdown_penalty = self.max_drawdown * 2.0  # Less severe penalty (reduced from 5.0)
            risk_adjusted_reward -= drawdown_penalty
        
        # Add hold bonus during early training, but also add holding penalty for extended periods
        if self.action_history and self.action_history[-1] == 1:  # hold action
            self.consecutive_holds += 1
            self.hold_duration += 1
            
            # Small bonus for short-term holds early in training
            if current_step < 5000 and self.consecutive_holds <= 10:  # Reduced from 10000 to 5000
                hold_bonus = min(self.consecutive_holds * 0.01, 0.3)
                risk_adjusted_reward += hold_bonus
            # Add penalty for excessive holding after early training - make this more aggressive
            elif current_step >= 5000 and self.consecutive_holds > 20:  # Reduced from 30 to 20
                # Gradually increasing penalty for excessive holding
                hold_penalty = min((self.consecutive_holds - 20) * 0.01, 1.0)  # Increased from 0.005 to 0.01
                risk_adjusted_reward -= hold_penalty
                if self.consecutive_holds % 50 == 0:
                    logger.warning(f"Excessive holding penalty: -{hold_penalty:.4f} after {self.consecutive_holds} consecutive holds")
        else:
            self.consecutive_holds = 0
        
        # Add profit streak bonus - reward successful trade streaks more generously
        if trade_occurred and 'trade_profit' in info and info['trade_profit'] > 0:
            self.successful_trade_streak += 1
            streak_bonus = min(self.successful_trade_streak * 0.10, 2.0)  # Doubled bonus and max cap
            risk_adjusted_reward += streak_bonus
            self.max_successful_streak = max(self.max_successful_streak, self.successful_trade_streak)
            logger.debug(f"Trade streak bonus: +{streak_bonus:.4f} (streak: {self.successful_trade_streak})")
        else:
            self.successful_trade_streak = 0
        
        return risk_adjusted_reward

    def _calculate_position_size(self, action, current_price, stop_loss_price):
        """
        Calculate appropriate position size based on risk parameters
        """
        if action == 1:  # Hold action
            return 0.0
            
        # Get current portfolio value
        portfolio_value = self.env.portfolio_value if hasattr(self.env, 'portfolio_value') else 10000.0
        
        # Calculate risk amount in currency units
        risk_amount = portfolio_value * self.max_risk_per_trade
        
        # Calculate distance to stop loss
        if stop_loss_price is None or current_price is None:
            # Default to 2% stop loss if prices not available
            price_distance = current_price * 0.02 if current_price is not None else 0.02
        else:
            price_distance = abs(current_price - stop_loss_price)
        
        # Avoid division by zero
        if price_distance <= 0:
            price_distance = current_price * 0.01 if current_price is not None else 0.01
            
        # Calculate position size based on risk
        position_size = risk_amount / price_distance
        
        # Check if adding this position would exceed our cumulative risk tolerance
        if self.risk_adjusted_position_sizing and (self.cumulative_risk + self.max_risk_per_trade) > self.max_cumulative_risk:
            # Scale down position if it would exceed max cumulative risk
            available_risk = max(0, self.max_cumulative_risk - self.cumulative_risk)
            position_size = position_size * (available_risk / self.max_risk_per_trade)
            
        return position_size
        
    def _calculate_risk_reward_ratio(self, current_price, entry_price, target_price, stop_loss_price):
        """
        Calculate the risk-reward ratio for a potential trade
        """
        if None in (current_price, entry_price, target_price, stop_loss_price):
            return 0.0
            
        risk = abs(entry_price - stop_loss_price)
        reward = abs(target_price - entry_price)
        
        # Avoid division by zero
        if risk <= 0:
            return 0.0
            
        return reward / risk

    def step(self, action):
        """
        Override step method to add trading safeguards and risk management
        """
        # Get current step and price information
        current_step = getattr(self.env, 'current_step', 0)
        current_price = None
        
        if hasattr(self.env, '_get_current_price'):
            current_price = self.env._get_current_price()
            
        # Calculate potential stop loss and target prices based on current market conditions
        # Simple example: 2% stop loss, 4% target (2:1 reward-risk ratio)
        stop_loss_price = current_price * 0.98 if current_price is not None else None
        target_price = current_price * 1.04 if current_price is not None else None
        
        # Check risk-reward ratio before allowing a trade
        if action != 1 and current_price is not None:  # Not a hold action
            # Calculate risk-reward ratio for this potential trade
            risk_reward = self._calculate_risk_reward_ratio(
                current_price, 
                current_price, 
                target_price, 
                stop_loss_price
            )
            
            # Only allow trades with favorable risk-reward ratio
            if risk_reward < self.target_risk_reward_ratio:
                logger.debug(f"Risk-reward ratio {risk_reward:.2f} below target {self.target_risk_reward_ratio:.2f}, forcing hold")
                action = 1  # Force hold if risk-reward is unfavorable
                
        # Check if we're in a cooldown period
        in_cooldown = (current_step - self.last_trade_step) < self.current_cooldown
        attempted_trade_during_cooldown = False
        
        if in_cooldown and action != 1:  # Not a hold action during cooldown
            attempted_trade_during_cooldown = True
            action = 1  # Force hold action
            self.forced_actions += 1
            
            if self.forced_actions % 10 == 0:  # Log periodically to avoid spamming
                logger.warning(f"Forced hold action during cooldown at step {current_step}, " 
                              f"{current_step - self.last_trade_step}/{self.min_cooldown} steps since last trade")
    
        # Store previous position for change detection
        self.previous_position = self.current_position
        
        # Determine if we're in a trade cooldown period - use adjusted cooldown
        in_cooldown = (current_step - self.last_trade_step) < self.min_cooldown
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
            
            # Require larger price movements early in training but be more lenient later
            required_movement = self.min_profit_threshold
            
            # More strict requirements depending on training stage, but make strict period shorter
            if current_step < 300:  # Reduced from 500 to 300
                required_movement = 0.01  # Reduced from 0.02 to 0.01
            elif current_step < 1000:  # Reduced from 2000 to 1000
                required_movement = 0.005  # Reduced from 0.01 to 0.005
            elif current_step < 3000:  # Reduced from 5000 to 3000
                required_movement = 0.003  # Reduced from 0.005 to 0.003
            
            if price_change_pct < required_movement and action != 1:
                strict_price_requirement = True
                logger.debug(f"Enforcing hold: price movement {price_change_pct:.4f}% < required {required_movement:.4f}%")
                action = 1  # Force hold
            
        # Check for min profit threshold violations - but make this less strict
        if current_price is not None:
            # If trying to sell, check if price is higher than last buy
            if action == 0 and self.last_buy_price is not None:
                profit_pct = (current_price - self.last_buy_price) / self.last_buy_price
                
                # Increase profit threshold early in training - but be more lenient
                min_profit = self.min_profit_threshold
                if current_step < 2000:  # Reduced from 5000 to 2000
                    min_profit = 0.005  # Reduced from 0.01 to 0.005
                
                if profit_pct < min_profit:
                    min_profit_violation = True
                    logger.debug(f"Prevented selling at loss/small profit: {profit_pct:.4f}% at step {current_step}")
            
            # If trying to buy, check if price is lower than last sell
            elif action == 2 and self.last_sell_price is not None:
                discount_pct = (self.last_sell_price - current_price) / self.last_sell_price
                
                # Increase discount threshold early in training - but be more lenient
                min_discount = self.min_profit_threshold
                if current_step < 2000:  # Reduced from 5000 to 2000
                    min_discount = 0.005  # Reduced from 0.01 to 0.005
                
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
                              f"{current_step - self.last_trade_step}/{self.min_cooldown} steps since last trade")
        
        # Also force hold for min profit violations and price movement requirements
        if min_profit_violation or strict_price_requirement:
            action = 1
        
        # Check for oscillation patterns and adjust cooldown if needed - but make penalties less severe
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
                
                # Extend cooldown for severe oscillation cases - but make it less aggressive
                if self.oscillation_count > 5:
                    # Extend cooldown period for severe oscillation
                    extended_cooldown = min(self.oscillation_count * 15, 500)  # Reduced from 20 to 15 and 1000 to 500
                    self.last_trade_step = current_step - self.min_cooldown + extended_cooldown
                    logger.warning(f"Extended cooldown by {extended_cooldown} steps due to severe oscillation (count: {self.oscillation_count})")
        
        # Early training stabilization - make this period much shorter
        if current_step < 100 and action != 1 and current_step % 20 != 0:  # Reduced from 300 to 100 and from 50 to 20
            logger.debug(f"Early training stability: forcing hold at step {current_step}")
            action = 1  # Force hold action during early training except every 20th step
        
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
        
        # Log when trades occur
        if trade_occurred:
            # Get portfolio value from environment or info
            portfolio_value = None
            if hasattr(self.env, 'portfolio_value'):
                portfolio_value = self.env.portfolio_value
            elif 'portfolio_value' in info:
                portfolio_value = info['portfolio_value']
                
            if portfolio_value is not None:
                logger.debug(f"TRADE DETECTED at step {current_step}: Position changed from {self.previous_position} to {self.current_position}")
                logger.debug(f"Portfolio value: {portfolio_value:.2f}, Price: {current_price if current_price is not None else 'unknown'}")
            else:
                logger.debug(f"TRADE DETECTED at step {current_step}: Position changed from {self.previous_position} to {self.current_position}")
                logger.debug(f"Price: {current_price if current_price is not None else 'unknown'}")
                
            # If this is an actual trade, update risk management metrics
            # For simplicity, assume each trade uses max_risk_per_trade amount of risk
            portfolio_value = getattr(self.env, 'portfolio_value', None)
            if action == 0:  # Sell (close position)
                self.cumulative_risk = max(0, self.cumulative_risk - self.max_risk_per_trade)
            elif action == 2:  # Buy (open position)
                self.cumulative_risk += self.max_risk_per_trade
                
            # Only log risk management metrics when there's a trade or periodically
            if trade_occurred and (current_step % 10000 == 0 or self.cumulative_risk % 0.01 < 0.001):
                portfolio_value_str = f"{portfolio_value:.2f}" if portfolio_value is not None else "unknown"
                logger.info(f"Risk management: Cumulative risk now {self.cumulative_risk:.1%}, " +
                           f"Portfolio value: {portfolio_value_str}")
        
        # Log action distribution less frequently
        if current_step % 100000 == 0:
            action_counts = {}
            for a in self.action_history[-1000:]:
                if a is not None:
                    if a not in action_counts:
                        action_counts[a] = 0
                    action_counts[a] += 1
            logger.info(f"Action distribution (last 1000 steps): {action_counts}")
        
        # Calculate additional penalties - but make them less severe
        additional_penalty = 0.0
        
        # Penalty for attempted trades during cooldown - reduce penalty
        if attempted_trade_during_cooldown:
            cooldown_penalty = OSCILLATION_PENALTY * (0.3 + min(self.cooldown_violations, 10) / 15)  # Reduced from 0.5 to 0.3 and 10 to 15
            additional_penalty -= cooldown_penalty
            self.cooldown_violations += 1
            if self.cooldown_violations % 5 == 0:
                logger.warning(f"Applied cooldown violation penalty: {cooldown_penalty:.2f} " 
                              f"(#{self.cooldown_violations})")
            
            # Add cooldown violation to info dict
            info['cooldown_violation'] = True
        
        # Penalty for oscillations - reduce penalty
        if len(self.action_history) >= 4:
            last_four = self.action_history[-4:]
            
            if last_four == [2, 0, 2, 0] or last_four == [0, 2, 0, 2]:
                # Increase oscillation penalty exponentially based on count - but make it less severe
                oscillation_penalty = OSCILLATION_PENALTY * (0.7 + min(self.oscillation_count, 5) ** 1.3)  # Reduced from 1.0 to 0.7 and 1.5 to 1.3
                additional_penalty -= oscillation_penalty
                logger.warning(f"Applied oscillation penalty: {oscillation_penalty:.2f} (#{self.oscillation_count})")
                
                # Add oscillation detection to info dict
                info['oscillation_detected'] = True
        
        # Penalty for same-price trades - reduce penalty 
        if trade_occurred and current_price is not None and self.last_trade_price is not None:
            # Check if trade price is very close to the last trade price
            if abs(current_price - self.last_trade_price) < 0.0001:
                self.same_price_trades += 1
                same_price_penalty = SAME_PRICE_TRADE_PENALTY * (0.7 + min(self.same_price_trades, 5))  # Reduced from 1.0 to 0.7
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
                
                # Additional bonus for holding - reduce this to encourage more trading
                if action == 1:  # Hold
                    consistency_reward += 0.01  # Reduced from 0.03 to 0.01
                    
                reward += consistency_reward
        
        # Update last action
        self.last_action = action
        
        # Curriculum learning - scale down rewards in early training, but for a shorter period
        if self.training_progress < 0.1:  # Reduced from 0.2 to 0.1
            reward *= 0.7  # Less reduction, changed from 0.5 to 0.7
        
        # Augment observation with action history and risk metrics
        augmented_observation = self._augment_observation(observation)
        
        # Add training progress to info dict
        info['training_progress'] = self.training_progress
        info['action_taken'] = action
        
        # Add portfolio metrics to info
        if hasattr(self.env, 'portfolio_value'):
            info['portfolio_growth_pct'] = ((self.env.portfolio_value / self.starting_portfolio) - 1.0) * 100
            info['highest_portfolio'] = self.highest_portfolio
        
        return augmented_observation, reward, terminated, truncated, info


class TensorboardCallback(BaseCallback):
    """Custom callback for tracking metrics during training"""
    
    def __init__(self, verbose=0, model_name=None, debug_frequency=250):
        super().__init__(verbose)
        # Save model name for logging
        self.model_name = model_name
        
        # Environment metrics
        self.returns = []
        self.portfolio_values = []
        self.episode_rewards = []
        self.trade_count = 0
        self.successful_trades = 0
        
        # Enhanced portfolio tracking
        self.starting_portfolio = None
        self.highest_portfolio = 0.0
        self.growth_rates = []
        self.drawdowns = []
        
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
        self.debug_frequency = debug_frequency  # Use the parameter value
        self.last_debug_output = 0
        
        # Add step timing for performance monitoring
        self.last_time = time.time()
        self.steps_since_last_log = 0
        
        logger.info(f"TensorboardCallback initialized for model {model_name} with debug frequency {debug_frequency}")
    
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
            reward = None
            action_counts = {}
            cooldown_violations = 0
            oscillation_count = 0
            trade_count = 0
            
            if self.locals is not None:
                # Get rewards if available
                if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
                    reward = self.locals['rewards'][0]
                
                # Get info from environment
                if 'infos' in self.locals and len(self.locals['infos']) > 0:
                    info = self.locals['infos'][0]
                    if 'portfolio_value' in info:
                        portfolio_value = info['portfolio_value']
                        
                        # Initialize starting portfolio on first collection
                        if self.starting_portfolio is None:
                            self.starting_portfolio = portfolio_value
                            
                        # Calculate growth from starting value
                        if self.starting_portfolio > 0:
                            growth_pct = ((portfolio_value / self.starting_portfolio) - 1.0) * 100
                            self.growth_rates.append(growth_pct)
                            
                            # Track highest portfolio value
                            if portfolio_value > self.highest_portfolio:
                                self.highest_portfolio = portfolio_value
                                
                            # Calculate drawdown from peak
                            if self.highest_portfolio > 0:
                                drawdown_pct = ((self.highest_portfolio - portfolio_value) / self.highest_portfolio) * 100
                                self.drawdowns.append(drawdown_pct)
                                
                    if 'cooldown_violation' in info:
                        cooldown_violations = info.get('cooldown_violation', False)
                    if 'oscillation_detected' in info:
                        oscillation_count = info.get('oscillation_detected', False)
                    if 'action' in info:
                        action = info['action']
                        if action in self.action_counts:
                            self.action_counts[action] += 1
                    if 'trade' in info:
                        trade_count = info.get('trade', False)
                        
                    # Track additional metrics if available
                    if 'portfolio_growth_pct' in info:
                        portfolio_growth = info['portfolio_growth_pct']
                        if hasattr(self, 'logger') and self.logger is not None:
                            self.logger.record('portfolio/growth_pct', portfolio_growth)
                            
                    if 'highest_portfolio' in info:
                        highest = info['highest_portfolio']
                        if hasattr(self, 'logger') and self.logger is not None:
                            self.logger.record('portfolio/highest_value', highest)
            
            # Print comprehensive stats
            portfolio_str = f", Portfolio: {portfolio_value:.2f}" if portfolio_value is not None else ""
            reward_str = f", Reward: {reward:.4f}" if reward is not None else ""
            action_counts_str = ", ".join([f"{action}: {count}" for action, count in self.action_counts.items() if count > 0])
            action_str = f", Actions: [{action_counts_str}]" if action_counts_str else ""
            
            # Add portfolio growth reporting
            growth_str = ""
            if self.starting_portfolio is not None and portfolio_value is not None and self.starting_portfolio > 0:
                growth_pct = ((portfolio_value / self.starting_portfolio) - 1.0) * 100
                growth_str = f", Growth: {growth_pct:.2f}%"
                
                # Add drawdown info if we have peak data
                if self.highest_portfolio > 0:
                    drawdown = ((self.highest_portfolio - portfolio_value) / self.highest_portfolio) * 100
                    growth_str += f", Drawdown: {drawdown:.2f}%"
            
            # Show exploration rate
            exploration_str = ""
            if hasattr(self.model, 'exploration') and hasattr(self.model.exploration, 'value'):
                exploration_str = f", Exploration: {self.model.exploration.value:.4f}"
            
            metrics_str = f"Trade count: {self.trade_count}, Cooldown violations: {self.cooldown_violations}, Oscillations: {self.oscillation_count}"
            
            # Only log basic metrics to console every 5th interval to reduce verbosity
            if self.debug_steps % (self.debug_frequency * 5) == 0:
                logger.info(f"Step {self.debug_steps}, {steps_per_second:.1f} steps/s{portfolio_str}{reward_str}{action_str}{exploration_str}{growth_str}")
                logger.info(f"Training metrics: {metrics_str}")
                
                # Show exploration/exploitation stats if available
                if hasattr(self.model, 'exploration') and hasattr(self.model.exploration, 'value'):
                    logger.info(f"Exploration rate: {self.model.exploration.value:.4f}")
            
            # Log key metrics to tensorboard
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.record('training/steps_per_second', steps_per_second)
                # Record number of steps per action
                for action, count in self.action_counts.items():
                    self.logger.record(f'actions/action_{action}_count', count)
                
                # Record portfolio metrics if available
                if portfolio_value is not None:
                    self.logger.record('portfolio/current_value', portfolio_value)
                    
                    # Record growth metrics
                    if self.starting_portfolio is not None and self.starting_portfolio > 0:
                        growth_pct = ((portfolio_value / self.starting_portfolio) - 1.0) * 100
                        self.logger.record('portfolio/growth_percent', growth_pct)
                        
                        # Record highest value and drawdown
                        self.logger.record('portfolio/highest_value', self.highest_portfolio)
                        if self.highest_portfolio > 0:
                            drawdown = ((self.highest_portfolio - portfolio_value) / self.highest_portfolio) * 100
                            self.logger.record('portfolio/drawdown_percent', drawdown)
            
            # Reset counter for steps since last log
            self.steps_since_last_log = 0
            self.last_time = current_time
        
        # Log rewards when episodes complete
        if self.locals is not None and 'dones' in self.locals and 'rewards' in self.locals:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    reward = self.locals['rewards'][i]
                    self.episode_rewards.append(reward)
                    self.logger.record('rewards/episodic_reward', reward)
                    
                    # If info dict available, extract additional metrics
                    if 'infos' in self.locals and len(self.locals['infos']) > i:
                        info = self.locals['infos'][i]
                        
                        # Track portfolio value
                        if 'portfolio_value' in info:
                            portfolio_value = info['portfolio_value']
                            self.portfolio_values.append(portfolio_value)
                            self.logger.record('portfolio/value', portfolio_value)
                            
                            # Track episode ending portfolio growth
                            if self.starting_portfolio is not None and self.starting_portfolio > 0:
                                final_growth_pct = ((portfolio_value / self.starting_portfolio) - 1.0) * 100
                                self.logger.record('portfolio/episode_end_growth_pct', final_growth_pct)
                            
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
                                
                                # Calculate action distribution
                                total_actions = sum(self.action_counts.values())
                                if total_actions > 0:
                                    distribution = self.action_counts[action] / total_actions * 100
                                    self.logger.record(f'actions/distribution_{action}_percent', distribution)
                        
                        if 'trade' in info and info['trade']:
                            self.trade_count += 1
                            self.logger.record('trades/count', self.trade_count)
                            
                            # Trade metrics if available
                            if 'trade_profit' in info:
                                trade_profit = info['trade_profit']
                                if trade_profit > 0:
                                    self.successful_trades += 1
                                    self.total_profit += trade_profit
                                else:
                                    self.total_loss += abs(trade_profit)
                                    
                                self.logger.record('trades/profit', trade_profit)
                                self.logger.record('trades/successful', self.successful_trades)
                                self.logger.record('trades/total_profit', self.total_profit)
                                self.logger.record('trades/total_loss', self.total_loss)
                                
                                # Calculate win rate and profit factor
                                if self.trade_count > 0:
                                    win_rate = (self.successful_trades / self.trade_count) * 100
                                    self.logger.record('trades/win_rate', win_rate)
                                    
                                if self.total_loss > 0:
                                    profit_factor = self.total_profit / max(self.total_loss, 1e-6)
                                    self.logger.record('trades/profit_factor', profit_factor)
                                    
                                # Calculate average profit per trade
                                if self.trade_count > 0:
                                    avg_profit = (self.total_profit - self.total_loss) / self.trade_count
                                    self.logger.record('trades/avg_profit', avg_profit)
        
        return True


class ResourceCheckCallback(BaseCallback):
    """
    Callback for monitoring system resources during training.
    Helps prevent out-of-memory errors and tracks resource usage.
    """
    
    def __init__(self, check_interval=5000, verbose=0):
        super(ResourceCheckCallback, self).__init__(verbose)
        self.check_interval = check_interval
        self.last_memory = 0
        self.last_cpu = 0
        
    def _on_step(self) -> bool:
        """Check system resources periodically during training"""
        if self.n_calls % self.check_interval == 0:
            # Get memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            # Log to tensorboard if available
            if hasattr(self, 'logger') and self.logger:
                self.logger.record('resources/memory_mb', memory_mb)
                self.logger.record('resources/cpu_percent', cpu_percent)
            
            # Calculate change since last check
            memory_change = memory_mb - self.last_memory
            cpu_change = cpu_percent - self.last_cpu
            
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
            self.last_cpu = cpu_percent
            
            # Log GPU info if available
            if torch.cuda.is_available():
                try:
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # in GB
                    gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # in GB
                    gpu_utilization = -1  # Default value if we can't get utilization
                    
                    # Try to get GPU utilization if possible
                    try:
                        import subprocess
                        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                        gpu_utilization = float(result.decode('utf-8').strip())
                    except:
                        pass
                    
                    logger.info(f"GPU Memory: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached" + 
                               (f", {gpu_utilization}% utilized" if gpu_utilization >= 0 else ""))
                    
                    # Log to tensorboard
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.record('resources/gpu_memory_gb', gpu_memory_allocated)
                        if gpu_utilization >= 0:
                            self.logger.record('resources/gpu_utilization', gpu_utilization)
                except Exception as e:
                    logger.warning(f"Error checking GPU memory: {e}")
        
        return True


def check_resources():
    """Check and log system resources"""
    # Memory usage
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.percent}% used, {memory.available / (1024**3):.1f}GB available")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU: {cpu_percent}% used")
    
    # GPU info if available
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # in GB
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # in GB
            logger.info(f"GPU: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached")
        except Exception as e:
            logger.warning(f"Error checking GPU resources: {e}")


def train_dqn(env, args, callbacks=None):
    """
    Train a DQN agent for cryptocurrency trading
    
    Args:
        env: Training environment
        args: Command line arguments
        callbacks: List of callbacks for training
        
    Returns:
        Trained DQN model
    """
    logger.info("Setting up DQN model with parameters from command line")
    
    # Get parameters from args
    buffer_size = args.buffer_size
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    gamma = args.gamma
    exploration_fraction = args.exploration_fraction
    exploration_initial_eps = args.exploration_initial_eps
    exploration_final_eps = args.exploration_final_eps
    target_update_interval = args.target_update_interval
    
    # Create tensorboard callback
    tb_callback = TensorboardCallback(verbose=1, model_name="DQN", debug_frequency=10000)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Combine all callbacks
    all_callbacks = [tb_callback, checkpoint_callback]
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Create the model with parameters from args
    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        learning_starts=1000,
        batch_size=batch_size,
        tau=1.0,
        gamma=gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log="./logs/dqn/"
    )
    
    # Load model from checkpoint if specified
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        model = DQN.load(args.load_model, env=env)
    
    # Train the model
    total_timesteps = args.timesteps if hasattr(args, 'timesteps') else 1000000
    logger.info(f"Training DQN for {total_timesteps} timesteps")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            log_interval=10,
            tb_log_name="dqn",
            progress_bar=True
        )
        
        # Save the final model
        model_save_path = os.path.join("models", f"dqn_model_final")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    
    return model


def train_ppo(env, args, callbacks=None):
    """
    Train a PPO agent for cryptocurrency trading
    
    Args:
        env: Training environment
        args: Command line arguments
        callbacks: List of callbacks for training
        
    Returns:
        Trained PPO model
    """
    logger.info("Setting up PPO model with parameters from command line")
    
    # Get parameters from args
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    gamma = args.gamma
    n_steps = args.n_steps
    n_epochs = args.n_epochs
    ent_coef = args.ent_coef
    clip_range = args.clip_range
    
    # Create tensorboard callback
    tb_callback = TensorboardCallback(verbose=1, model_name="PPO", debug_frequency=10000)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_model",
        save_vecnormalize=True,
    )
    
    # Combine all callbacks
    all_callbacks = [tb_callback, checkpoint_callback]
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Create the model with parameters from args
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        clip_range_vf=None,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/ppo/"
    )
    
    # Load model from checkpoint if specified
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        model = PPO.load(args.load_model, env=env)
    
    # Train the model
    total_timesteps = args.timesteps if hasattr(args, 'timesteps') else 1000000
    logger.info(f"Training PPO for {total_timesteps} timesteps")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            log_interval=10,
            tb_log_name="ppo",
            progress_bar=True
        )
        
        # Save the final model
        model_save_path = os.path.join("models", f"ppo_model_final")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    
    return model


def train_a2c(env, args, callbacks=None):
    """
    Train an A2C agent for cryptocurrency trading
    
    Args:
        env: Training environment
        args: Command line arguments
        callbacks: List of callbacks for training
        
    Returns:
        Trained A2C model
    """
    logger.info("Setting up A2C model with parameters from command line")
    
    # Get parameters from args
    learning_rate = args.learning_rate
    gamma = args.gamma
    n_steps = args.n_steps
    ent_coef = args.ent_coef
    
    # Increase entropy coefficient to encourage exploration
    ent_coef = max(ent_coef, 0.05)  # Ensure minimum entropy for exploration
    logger.info(f"Using entropy coefficient: {ent_coef} to encourage action exploration")
    
    # Create tensorboard callback
    tb_callback = TensorboardCallback(verbose=1, model_name="A2C", debug_frequency=10000)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="a2c_model",
        save_vecnormalize=True,
    )
    
    # Combine all callbacks
    all_callbacks = [tb_callback, checkpoint_callback]
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Check if we have an LSTM model to use for feature extraction
    lstm_state_dict = None
    policy_kwargs = {}
    
    if args.lstm_model_path:
        logger.info(f"Using LSTM model from {args.lstm_model_path} for feature extraction")
        try:
            # Load the LSTM model
            import torch
            lstm_state_dict = torch.load(args.lstm_model_path, map_location=torch.device('cpu'))
            
            # Set up policy kwargs to use our custom feature extractor
            policy_kwargs = {
                "features_extractor_class": LSTMFeatureExtractor,
                "features_extractor_kwargs": {
                    "lstm_state_dict": lstm_state_dict,
                    "features_dim": 64  # Match with the hidden size in the LSTM model
                }
            }
            logger.info("Successfully configured LSTM feature extractor")
        except Exception as e:
            logger.error(f"Error setting up LSTM feature extractor: {e}")
            logger.error(traceback.format_exc())
            # Continue without LSTM feature extraction
            policy_kwargs = {}
    
    # Create the model with parameters from args
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        verbose=1,
        tensorboard_log="./logs/a2c/",
        policy_kwargs=policy_kwargs
    )
    
    # Load model from checkpoint if specified
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        try:
            model = A2C.load(args.load_model, env=env)
        except AssertionError as e:
            if "No data found in the saved file" in str(e) and args.lstm_model_path:
                logger.warning("Failed to load model as A2C model, proceeding with a new A2C model")
                # Create a new model since the load failed
                model = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    gamma=gamma,
                    gae_lambda=0.95,
                    ent_coef=ent_coef,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    rms_prop_eps=1e-5,
                    verbose=1,
                    tensorboard_log="./logs/a2c/",
                    policy_kwargs=policy_kwargs
                )
            else:
                # Re-raise the exception if it's not the specific one we're handling
                raise
    
    # Load LSTM model if specified
    if args.lstm_model_path:
        logger.info(f"LSTM model loaded from {args.lstm_model_path}")
        # The LSTM is now integrated into the policy via feature extractor
    
    # Train the model
    total_timesteps = args.timesteps if hasattr(args, 'timesteps') else 1000000
    logger.info(f"Training A2C for {total_timesteps} timesteps")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            log_interval=10,
            tb_log_name="a2c",
            progress_bar=True
        )
        
        # Save the final model
        model_save_path = os.path.join("models", f"a2c_model_final")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    
    return model


def train_sac(env, args, callbacks=None):
    """
    Train a SAC agent for cryptocurrency trading
    
    Args:
        env: Training environment
        args: Command line arguments
        callbacks: List of callbacks for training
        
    Returns:
        Trained SAC model
    """
    logger.info("Setting up SAC model with parameters from command line")
    
    # Get parameters from args
    learning_rate = args.learning_rate
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    gamma = args.gamma
    target_update_interval = args.target_update_interval
    
    # Create tensorboard callback
    tb_callback = TensorboardCallback(verbose=1, model_name="SAC", debug_frequency=10000)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="sac_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Combine all callbacks
    all_callbacks = [tb_callback, checkpoint_callback]
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Create the model with parameters from args
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        tau=0.005,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef="auto",
        target_update_interval=target_update_interval,
        target_entropy="auto",
        verbose=1,
        tensorboard_log="./logs/sac/"
    )
    
    # Load model from checkpoint if specified
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        model = SAC.load(args.load_model, env=env)
    
    # Train the model
    total_timesteps = args.timesteps if hasattr(args, 'timesteps') else 1000000
    logger.info(f"Training SAC for {total_timesteps} timesteps")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            log_interval=10,
            tb_log_name="sac",
            progress_bar=True
        )
        
        # Save the final model
        model_save_path = os.path.join("models", f"sac_model_final")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    
    return model


def main():
    """Main function to run the training process"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agents for cryptocurrency trading")
    
    # Model selection and parameters
    parser.add_argument("--finrl_model", type=str, default="dqn", choices=["dqn", "ppo", "a2c", "sac"],
                        help="RL algorithm to use (default: dqn)")
    parser.add_argument("--timesteps", type=int, default=1000000, 
                        help="Number of timesteps to train (default: 1,000,000)")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a saved model to continue training from")
    
    # Add parameter for LSTM model
    parser.add_argument("--lstm_model_path", type=str, default=None,
                        help="Path to a saved LSTM model to use for state representation")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/crypto_data.csv",
                        help="Path to the data file (default: data/crypto_data.csv)")
    parser.add_argument("--data_key", type=str, default=None,
                        help="Key for HDF5 file (default: '/15m' for synthetic data)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Trading symbol (default: BTC/USDT)")
    
    # Environment parameters
    parser.add_argument("--initial_balance", type=float, default=10000,
                        help="Initial balance in USD (default: 10000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Trading commission (default: 0.001 or 0.1%)")
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="Maximum steps per episode (default: 20000)")
    
    # Additional training parameters
    parser.add_argument("--learning_rate", type=float, default=0.0003, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed")
    
    # PPO/A2C-specific parameters
    parser.add_argument("--n_steps", type=int, default=2048, 
                        help="Number of steps per update for PPO/A2C")
    parser.add_argument("--ent_coef", type=float, default=0.01, 
                        help="Entropy coefficient for PPO/A2C")
    parser.add_argument("--n_epochs", type=int, default=10, 
                        help="Number of epochs per update for PPO")
    parser.add_argument("--clip_range", type=float, default=0.2, 
                        help="PPO clip range")
    
    # DQN/SAC-specific parameters
    parser.add_argument("--buffer_size", type=int, default=100000, 
                        help="Replay buffer size for DQN/SAC")
    parser.add_argument("--exploration_fraction", type=float, default=0.1, 
                        help="Fraction of training time for exploration in DQN")
    parser.add_argument("--exploration_initial_eps", type=float, default=1.0, 
                        help="Initial exploration rate for DQN")
    parser.add_argument("--exploration_final_eps", type=float, default=0.05, 
                        help="Final exploration rate for DQN")
    parser.add_argument("--target_update_interval", type=int, default=10000, 
                        help="Update frequency for target network in DQN/SAC")
    
    # Trading safeguards
    parser.add_argument("--trade_cooldown", type=int, default=TRADE_COOLDOWN_PERIOD, 
                        help=f"Minimum steps between trades (default: {TRADE_COOLDOWN_PERIOD})")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create logs and models directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    
    logger.info(f"Starting training with {args.finrl_model.upper()} agent")
    logger.info(f"Training for {args.timesteps} timesteps")
    
    # Set seeds for reproducibility
    if args.seed is not None:
        set_seeds(args.seed)
    else:
        set_seeds(42)
    
    # Load data
    try:
        logger.info(f"Loading data from {args.data_path}")
        
        # Check file extension to determine loading method
        file_ext = os.path.splitext(args.data_path)[1].lower()
        
        if file_ext == '.h5' or file_ext == '.hdf5':
            # Load HDF5 file
            if args.data_key:
                logger.info(f"Loading HDF5 with key: {args.data_key}")
                data = pd.read_hdf(args.data_path, key=args.data_key)
            else:
                # Default to '15m' timeframe if no key specified for synthetic data
                try:
                    # Try to get a list of keys from the file
                    with pd.HDFStore(args.data_path, mode='r') as store:
                        keys = store.keys()
                    
                    # If there are multiple keys and no specific key provided
                    if len(keys) > 1:
                        # Default to '15m' timeframe
                        default_key = '/15m'
                        if default_key in keys:
                            logger.info(f"Multiple datasets found. Defaulting to '{default_key}' timeframe")
                            data = pd.read_hdf(args.data_path, key=default_key)
                        else:
                            # If '15m' isn't available, use the first key
                            selected_key = keys[0]
                            logger.info(f"Multiple datasets found, '15m' not available. Using {selected_key}")
                            data = pd.read_hdf(args.data_path, key=selected_key)
                    else:
                        # Only one dataset in the file
                        logger.info("Loading the only HDF5 dataset available")
                        data = pd.read_hdf(args.data_path)
                except Exception as e:
                    logger.error(f"Error examining HDF5 structure: {e}")
                    logger.error("Please specify a data_key (e.g., --data_key /15m)")
                    return
        elif file_ext == '.csv':
            # Load CSV file
            data = pd.read_csv(args.data_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Use .csv, .h5, or .hdf5")
            
        logger.info(f"Data loaded successfully: {len(data)} rows")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Check system resources before training
    logger.info("Checking system resources before training:")
    check_resources()
    
    # Create base environment
    logger.info("Creating base environment")
    base_env = CryptocurrencyTradingEnv(
        df=data,
        initial_balance=args.initial_balance,
        transaction_fee=args.commission,
        indicators=INDICATORS,
        symbol=args.symbol
    )
    
    # Log initial portfolio value
    logger.info(f"Environment initialized with initial balance: {args.initial_balance}")
    if hasattr(base_env, 'portfolio_value'):
        logger.info(f"Initial portfolio value in environment: {base_env.portfolio_value}")
    else:
        logger.warning("Environment does not have portfolio_value attribute")
        
    # Wrap with safety features
    logger.info("Applying SafeTradingEnvWrapper")
    safe_env = SafeTradingEnvWrapper(
        env=base_env,
        trade_cooldown=args.trade_cooldown,  # Use value from args
        max_history_size=100
    )
    
    # Wrap with TimeLimit
    logger.info(f"Applying TimeLimit of {args.max_steps} steps")
    time_limit_env = TimeLimit(safe_env, max_episode_steps=args.max_steps)
    
    # Wrap with Monitor for logging
    logger.info("Applying Monitor wrapper")
    monitor_env = Monitor(time_limit_env, "logs/monitor/")
    
    # Create vectorized environment
    logger.info("Creating vectorized environment")
    env_kwargs = {
        "df": data,
        "initial_balance": args.initial_balance,
        "transaction_fee": args.commission,
        "indicators": INDICATORS,
        "symbol": args.symbol
    }
    
    # Define function to create a wrapped environment
    def make_env():
        base_env = CryptocurrencyTradingEnv(**env_kwargs)
        # Verify initial balance
        logger.info(f"Vector env initialized with balance: {base_env.portfolio_value if hasattr(base_env, 'portfolio_value') else 'unknown'}")
        safe_env = SafeTradingEnvWrapper(base_env, trade_cooldown=args.trade_cooldown)  # Use value from args
        time_limit_env = TimeLimit(safe_env, max_episode_steps=args.max_steps)
        return Monitor(time_limit_env, "logs/monitor/")
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env])
    
    # Normalize observations and rewards
    logger.info("Applying VecNormalize")
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards
        clip_obs=10.,  # Clip observation
        clip_reward=10.,  # Clip reward
        gamma=0.99,  # Discount factor
        epsilon=1e-8  # Small constant to avoid division by zero
    )
    
    # Create resource monitoring callback
    resource_callback = ResourceCheckCallback(check_interval=10000)  # Check every 10K steps
    
    # Define model path for saving and loading
    models_path = "models"
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    # Initialize trained model as None
    trained_model = None
    
    try:
        # Train according to selected algorithm
        if args.finrl_model.lower() == "dqn":
            logger.info("Training DQN agent")
            trained_model = train_dqn(vec_env, args, [resource_callback])  # Pass resource_callback
        elif args.finrl_model.lower() == "ppo":
            logger.info("Training PPO agent")
            trained_model = train_ppo(vec_env, args, [resource_callback])  # Pass resource_callback
        elif args.finrl_model.lower() == "sac":
            logger.info("Training SAC agent")
            trained_model = train_sac(vec_env, args, [resource_callback])  # Pass resource_callback
        else:  # Default to A2C
            logger.info("Training A2C agent")
            trained_model = train_a2c(vec_env, args, [resource_callback])  # Pass resource_callback
        
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
