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
from stable_baselines3.dqn.policies import DQNPolicy  # Add explicit import for DQNPolicy
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from collections import deque  # Fix: Import deque from collections module

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

# Constants for trading safeguards - ADJUSTING FOR MORE BALANCED TRADING
TRADE_COOLDOWN_PERIOD = 1  # Minimum value of 1 to prevent division by zero, allows frequent trading
OSCILLATION_PENALTY = 2.0  # Further reduced from 5.0 to 2.0 - much less penalty for oscillation
SAME_PRICE_TRADE_PENALTY = 5.0  # Further reduced from 10.0 to 5.0
MAX_TRADE_FREQUENCY = 0.95  # Increased from 0.80 to 0.95 - allow up to 95% of steps to be trades

# Class for LSTM feature extraction
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses an LSTM model to process observations before 
    passing them to the policy network.
    
    This allows utilization of pre-trained LSTM models for feature extraction
    in reinforcement learning.
    """
    
    def __init__(self, observation_space, lstm_state_dict=None, features_dim=64):
        """
        Initialize the LSTM feature extractor.
        
        Args:
            observation_space: The observation space of the environment
            lstm_state_dict: The state dictionary of a pre-trained LSTM model
            features_dim: The output dimension of the LSTM features
        """
        # Call the parent constructor with the correct features_dim
        super().__init__(observation_space, features_dim=features_dim)
        
        # Create LSTM model architecture - this should match the saved model
        input_dim = observation_space.shape[0]
        self.lstm_model = torch.nn.LSTM(
            input_size=input_dim, 
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Try to load the state dict
        if lstm_state_dict is not None:
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
        
        # Set to eval mode since we're not training the LSTM
        self.lstm_model.eval()
        
    def forward(self, observations):
        """
        Process observations through the LSTM feature extractor.
        
        Args:
            observations: Tensor of observations from the environment
            
        Returns:
            Tensor of processed features
        """
        # Ensure observations are of the right shape
        # LSTM expects [batch_size, sequence_length, input_size]
        
        with torch.no_grad():  # No need to track gradients in feature extraction
            # Process as a single timestep sequence [batch_size, 1, input_size]
            # Add sequence dimension
            obs_seq = observations.unsqueeze(1)
            
            # Forward pass through LSTM
            lstm_out, _ = self.lstm_model(obs_seq)
            
            # Extract the output for the last timestep [batch_size, features_dim]
            features = lstm_out[:, -1, :]
            
        return features

# Add a custom policy for DQN that discourages holding
class AntiHoldPolicy(DQNPolicy):  # Change from stable_baselines3.dqn.policies.DQNPolicy to just DQNPolicy
    """
    A custom DQN policy that discourages the hold action by artificially
    reducing its Q-value during action selection.
    """
    
    def __init__(self, *args, hold_action_bias=-3.0, **kwargs):
        """Initialize the anti-hold policy with a bias against the hold action"""
        super().__init__(*args, **kwargs)
        self.hold_action_bias = hold_action_bias
        self.hold_action = 1  # The action index for hold
    
    def _predict(self, obs, deterministic=True):
        """
        Overrides the parent class _predict method to apply a bias against holding.
        This method reduces the Q-value of the hold action to discourage the agent from holding.
        """
        q_values = self.q_net(obs)
        
        # Apply a negative bias to the hold action's Q-value
        # Use a stronger bias (-3.0 instead of -1.0) to more strongly discourage holding
        q_values[:, self.hold_action] += self.hold_action_bias
        
        # Get the actions using the modified Q-values
        actions = q_values.argmax(dim=1).reshape(-1)
        return actions, q_values

class SafeTradingEnvWrapper(gymnasium.Wrapper):
    """
    A wrapper for trading environments that adds safeguards against:
    1. Rapid trading (enforces a cooldown period)
    2. Action oscillation (detects and prevents buy-sell-buy patterns)
    3. Risk management (enforces position sizing based on risk)
    """
    
    def __init__(self, env, trade_cooldown=TRADE_COOLDOWN_PERIOD, max_history_size=100, max_risk_per_trade=0.02, take_profit_pct=0.03):
        """Initialize the wrapper with safeguards against harmful trading patterns"""
        super().__init__(env)
        
        # Trading safeguards - reduce cooldown significantly to allow more trading
        self.trade_cooldown = 1  # Set to minimum value to encourage more trading
        if trade_cooldown <= 0:
            logger.warning(f"Specified trade_cooldown was {trade_cooldown}, setting to minimum of 1 to prevent division by zero")
        
        # Reduce cooldown as training progresses
        self.min_cooldown = 1  # Minimum cooldown period reduced from 3 to 1
        
        # Risk management parameters - relax constraints to allow more trading
        self.max_risk_per_trade = max_risk_per_trade * 1.5  # Increase maximum risk per trade
        self.max_history_size = max_history_size  # Initialize max_history_size with the parameter value
        self.target_risk_reward_ratio = 0.3  # Further reduced from 0.5 to 0.3 - much less strict
        self.risk_adjusted_position_sizing = True  # Use risk-adjusted position sizing
        self.cumulative_risk = 0.0  # Track cumulative risk across open positions
        self.max_cumulative_risk = 0.3  # Increased from 0.25 to 0.3 - allow more risk
        self.risk_per_position = {}  # Track risk per position
        
        # Take profit parameters
        self.take_profit_pct = take_profit_pct
        
        # Trading history tracking
        self.last_trade_step = -self.trade_cooldown  # Start with cooldown already passed
        self.last_trade_price = None
        self.action_history = []
        self.position_history = []
        
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
        self.min_profit_threshold = 0.0001  # Reduced from 0.0005 to 0.0001 - requires only 0.01% difference
        
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
        
        # Get the portfolio value from the environment
        if hasattr(self.env, 'portfolio_value') and self.env.portfolio_value > 0:
            self.peak_value = self.env.portfolio_value
            # Use the initial portfolio value for tracking growth across training
            self.starting_portfolio = self.env.portfolio_value
            self.highest_portfolio = self.starting_portfolio
            logger.info(f"SafeTradingEnvWrapper initialized with portfolio value: {self.starting_portfolio:.2f}")
        else:
            self.peak_value = 10000.0  # Default if env doesn't provide portfolio value
            self.starting_portfolio = 10000.0
            self.highest_portfolio = self.starting_portfolio
            logger.warning("Environment does not provide portfolio value, using default 10000.0")
        
        self.portfolio_growth_rate = 0.0
        
        self.current_cooldown = max(1, trade_cooldown)  # Ensure cooldown is at least 1
        
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
        
        # Adjust observation space to include the new features
        if isinstance(self.env.observation_space, spaces.Box):
            # Calculate new observation space size
            # Original observation dimension + action history (15) + consistency metrics (3) 
            # + risk metrics (3) + cash metrics (2) + cooldown status (1)
            original_shape = self.env.observation_space.shape
            if len(original_shape) == 1:
                # 1D observation space
                additional_features = 24  # 15 + 3 + 3 + 2 + 1
                low = np.append(self.env.observation_space.low, [-np.inf] * additional_features)
                high = np.append(self.env.observation_space.high, [np.inf] * additional_features)
                self.observation_space = spaces.Box(
                    low=low, 
                    high=high, 
                    dtype=self.env.observation_space.dtype
                )
                logger.info(f"Expanded observation space from {original_shape[0]} to {original_shape[0] + additional_features} dimensions")
        else:
            # Non-Box observation space, keep original
            logger.warning(f"Keeping original observation space of type {type(self.env.observation_space)} - augmentation only supports Box spaces")
        
        logger.info(f"SafeTradingEnvWrapper initialized with {trade_cooldown} step cooldown")
        
        # Keep track of action distributions
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.total_actions = 0
        self.action_window = []
        self.window_size = 1000  # Track last 1000 actions
    
    def reset(self, **kwargs):
        """Reset the environment and all tracking variables"""
        observation, info = self.env.reset(**kwargs)
        
        # Reset all tracking variables
        self.action_history = []
        self.position_history = []
        self.last_trade_step = -self.trade_cooldown
        self.last_trade_price = None
        self.current_position = 0.0
        self.previous_position = 0.0
        self.forced_actions = 0
        self.cooldown_violations = 0
        self.oscillation_patterns = {
            'buy_sell_alternation': 0,
            'rapid_reversals': 0,
        }
        self.oscillation_count = 0
        self.consecutive_same_action = 0
        self.same_price_trades = 0
        self.consecutive_holds = 0
        self.hold_duration = 0
        self.successful_trade_streak = 0
        self.max_successful_streak = 0
        self.max_drawdown = 0.0
        self.last_buy_price = None
        self.last_sell_price = None
        self.last_take_profit_price = None
        
        # Get the initial portfolio value from the environment
        if hasattr(self.env, 'portfolio_value'):
            # Important: Only reset peak_value, but keep starting_portfolio from initial env creation
            # to properly track long-term growth across episodes
            self.peak_value = self.env.portfolio_value
            
            # Log the starting portfolio value for this episode
            logger.debug(f"New episode starting with portfolio value: {self.env.portfolio_value:.2f}")
        else:
            self.peak_value = 0.0
            
        # Reset performance metrics for the new episode
        self.portfolio_growth_rate = 0.0
        self.sharpe_ratio = 0.0
        
        # CRITICAL: Do NOT reset the starting_portfolio and highest_portfolio values
        # so that we can properly track growth across episodes
        # This allows the agent to learn to increase portfolio value over time
        # If we reset these values, the growth is always measured from the beginning of each episode
        
        # Add action history to observation
        observation = self._augment_observation(observation)
        
        self.cumulative_risk = 0.0  # Reset cumulative risk on environment reset
        self.risk_per_position = {}  # Reset risk per position tracking
        
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
        
        # Add cash balance metrics if available in the info dictionary
        cash_metrics = np.zeros(2)
        if hasattr(self.env, '_get_info'):
            info = self.env._get_info()
            cash_ratio = info.get('cash_ratio', None)
            if cash_ratio is not None:
                # Cash ratio (normalized between 0 and 1)
                cash_metrics[0] = min(max(0, cash_ratio), 1.0)
                
                # Cash ratio warning signal - stronger when too far from optimal range (0.3-0.7)
                if cash_ratio < 0.2:  # Too little cash
                    cash_metrics[1] = (0.2 - cash_ratio) * 5.0  # Warning increases as cash decreases
                elif cash_ratio > 0.8:  # Too much cash
                    cash_metrics[1] = (cash_ratio - 0.8) * 5.0  # Warning increases as cash gets too high
                else:
                    cash_metrics[1] = 0.0  # No warning in good range
        
        # Calculate current cooldown status
        current_step = getattr(self.env, 'day', 0)
        # Add safety check to prevent division by zero
        if self.current_cooldown <= 0:
            self.current_cooldown = 1  # Ensure cooldown is at least 1 to prevent division by zero
            logger.warning(f"Cooldown period was 0, setting to 1 to prevent division by zero")
        
        cooldown_status = min(max(
            (current_step - self.last_trade_step) / self.current_cooldown, 0), 1)
        
        # Combine all features
        augmented_features = np.concatenate([
            action_history,
            consistency_metrics,
            risk_metrics,
            cash_metrics,  # Add cash metrics to augmented features
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
        
        # Ensure new_cooldown is at least 1 to prevent division by zero
        new_cooldown = max(1, new_cooldown)
        
        # Update if significant change
        if abs(new_cooldown - self.current_cooldown) > 10:
            self.current_cooldown = int(new_cooldown)
            logger.info(f"Adjusted cooldown period to {self.current_cooldown} steps (oscillation score: {oscillation_score:.1f})")
    
    def _calculate_risk_rewards(self, reward, info, trade_occurred, current_step):
        """Calculate risk-aware rewards based on trading performance"""
        # Get current portfolio value
        portfolio_value = info.get('portfolio_value', 0)
        
        # Get cash balance information
        cash_balance = info.get('cash_balance', 0)
        cash_ratio = info.get('cash_ratio', 1.0)
        
        # Initialize adjusted reward
        risk_adjusted_reward = reward
        
        # Track peak portfolio value for drawdown calculation
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        # Track highest portfolio value across episodes
        if portfolio_value > self.highest_portfolio:
            prev_highest = self.highest_portfolio
            self.highest_portfolio = portfolio_value
            self.portfolio_growth_rate = (self.highest_portfolio - prev_highest) / max(prev_highest, 1.0)
        
        # Calculate drawdown - but make it less impactful
        if self.peak_value > 0:
            self.max_drawdown = min(self.max_drawdown, (self.peak_value - portfolio_value) / self.peak_value)
        
        # Calculate trade return metrics if a trade occurred
        if trade_occurred:
            # Add trade return to history
            if self.last_trade_price is not None and 'close_price' in info:
                trade_return = (info['close_price'] - self.last_trade_price) / self.last_trade_price
                self.trade_returns.append(trade_return)
                
                # Track if this was a successful trade
                if trade_return > 0:
                    self.successful_trades += 1
                    
                    # Calculate Sharpe ratio
                    if len(self.trade_returns) > 5:
                        returns_array = np.array(self.trade_returns[-20:])
                        mean_return = np.mean(returns_array)
                        std_return = np.std(returns_array) + 1e-6  # Add small constant to avoid division by zero
                        self.sharpe_ratio = mean_return / std_return
        
        # Apply risk-aware reward adjustments
        risk_adjusted_reward = reward
        
        # Add portfolio growth reward (significant boost but less than before)
        if portfolio_value > self.starting_portfolio:
            growth_pct = (portfolio_value - self.starting_portfolio) / self.starting_portfolio
            # Log detailed growth information for debugging at less frequent intervals
            if current_step % 10000 == 0 or (trade_occurred and current_step % 10000 == 0):
                logger.info(f"Portfolio growth: {growth_pct:.4f} (starting: {self.starting_portfolio:.2f}, current: {portfolio_value:.2f})")
            growth_reward = min(growth_pct * 3.0, 3.0)  # Increased from 2.0 to 3.0 to reward growth more
            risk_adjusted_reward += growth_reward
            
            # Extra reward for achieving new highs
            if self.portfolio_growth_rate > 0:
                new_high_reward = min(self.portfolio_growth_rate * 5.0, 1.0)  # Increased from 3.0 to 5.0 and max from 0.5 to 1.0
                risk_adjusted_reward += new_high_reward
        
        # Add drawdown penalty - but make it much less severe
        if self.max_drawdown > 0.50:  # Only penalize significant drawdowns (increased from 0.25 to 0.50)
            drawdown_penalty = self.max_drawdown * 0.5  # Less severe penalty (reduced from 1.0 to 0.5)
            risk_adjusted_reward -= drawdown_penalty
        
        # Add portfolio balance rewards/penalties
        # Encourage a balanced portfolio allocation
        if cash_ratio < 0.2:  # Too much in assets
            # Apply stronger penalties for very low cash positions
            balance_penalty = min((0.2 - cash_ratio) * 3.0, 1.0)
            risk_adjusted_reward -= balance_penalty
            logger.debug(f"Low cash ratio penalty: -{balance_penalty:.4f} (cash ratio: {cash_ratio:.2f})")
        elif cash_ratio > 0.9 and current_step > 2000:  # Too much in cash, but only after initial training
            # Gentle nudge to deploy capital after initial training period
            opportunity_cost = min((cash_ratio - 0.9) * 0.5, 0.5)
            risk_adjusted_reward -= opportunity_cost
            logger.debug(f"High cash ratio penalty: -{opportunity_cost:.4f} (cash ratio: {cash_ratio:.2f})")
        elif cash_ratio >= 0.3 and cash_ratio <= 0.7:
            # Reward for maintaining a balanced portfolio
            balance_bonus = 0.1  # Small bonus for good balance
            risk_adjusted_reward += balance_bonus
            logger.debug(f"Balanced portfolio bonus: +{balance_bonus:.4f} (cash ratio: {cash_ratio:.2f})")
        
        # Give bonus for trades, especially selling
        if trade_occurred:
            # Base trade bonus - increased to strongly encourage any trade
            trade_bonus = 0.5  # Increased from 0.25 to 0.5
            risk_adjusted_reward += trade_bonus
            logger.debug(f"Applied trade bonus: +{trade_bonus:.2f} at step {current_step}")
            
            # Extra bonus for selling (to encourage profit taking)
            if self.action_history and self.action_history[-1] == 0:  # sell action
                sell_bonus = 0.75  # Increased from 0.5 to 0.75
                risk_adjusted_reward += sell_bonus
                logger.debug(f"Applied sell bonus: +{sell_bonus:.2f} at step {current_step}")
                
            # Add bonus for buying (to encourage more trading activity)
            elif self.action_history and self.action_history[-1] == 2:  # buy action
                buy_bonus = 0.5  # New bonus for buy actions
                risk_adjusted_reward += buy_bonus
                logger.debug(f"Applied buy bonus: +{buy_bonus:.2f} at step {current_step}")
        
        # Add hold penalty to discourage excessive holding
        if self.action_history and self.action_history[-1] == 1:  # hold action
            self.consecutive_holds += 1
            self.hold_duration += 1
            
            # Apply larger constant hold penalty
            hold_penalty = 0.8  # Increased from 0.5 to 0.8 - even higher penalty for any hold
            risk_adjusted_reward -= hold_penalty
            
            # Add penalty for excessive holding after early training - lower threshold and higher penalty
            if self.consecutive_holds > 3:  # Reduced from 5 to 3
                # Gradually increasing penalty for excessive holding (more aggressive)
                hold_penalty = min((self.consecutive_holds - 3) * 0.1, 10.0)  # Much more aggressive penalty 
                risk_adjusted_reward -= hold_penalty
                if self.consecutive_holds % 5 == 0:  # Reduced from 10 to 5
                    logger.warning(f"Excessive holding penalty: -{hold_penalty:.4f} after {self.consecutive_holds} consecutive holds")
        else:
            self.consecutive_holds = 0
        
        # Reward for taking actions (not just holding)
        if self.action_history and self.action_history[-1] != 1:  # not a hold action
            # Give reward for exploring buying and selling
            if current_step < 30000:  # Throughout a longer early training period
                action_bonus = 0.5  # Increased from 0.2 to 0.5
                risk_adjusted_reward += action_bonus
                if current_step % 5000 == 0:
                    logger.info(f"Applying action exploration bonus: +{action_bonus:.2f} at step {current_step}")
        
        # Add profit streak bonus - more generous to encourage successful trading patterns
        if trade_occurred and 'trade_profit' in info and info['trade_profit'] > 0:
            self.successful_trade_streak += 1
            streak_bonus = min(self.successful_trade_streak * 0.1, 2.0)  # Increased from 0.05 to 0.1 and from 1.0 to 2.0
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
        
        # Check risk-reward ratio before allowing a trade - significantly relax this constraint
        # Only perform this check early in training
        if current_step < 5000 and action != 1 and current_price is not None:  # Not a hold action in early training
            # Calculate risk-reward ratio for this potential trade
            risk_reward = self._calculate_risk_reward_ratio(
                current_price, 
                current_price, 
                target_price, 
                stop_loss_price
            )
            
            # Only block trades with extremely unfavorable risk-reward ratio
            if risk_reward < self.target_risk_reward_ratio * 0.05:  # Further reduced threshold
                logger.debug(f"Risk-reward ratio {risk_reward:.2f} far below target {self.target_risk_reward_ratio:.2f}, forcing hold")
                action = 1  # Force hold if risk-reward is extremely unfavorable
                
        # Completely disable cooldown after initial training
        if current_step > 30:  # Very early training
            self.current_cooldown = 0  # Remove cooldown completely to encourage trading
        
        # Store previous position for change detection
        self.previous_position = self.current_position
        
        # Disable oscillation checks after early training to allow more action freedom
        if current_step > 10000 and len(self.action_history) > 4:
            # Only apply oscillation controls in the most extreme cases
            last_four = self.action_history[-4:]
            
            # Apply progressive penalty for oscillatory behavior - but only for very clear oscillation
            # that repeats multiple times
            if (last_four == [2, 0, 2, 0] and self.action_history[-8:-4] == [2, 0, 2, 0]) or \
               (last_four == [0, 2, 0, 2] and self.action_history[-8:-4] == [0, 2, 0, 2]):
                
                # Force hold action only for severe repeated oscillation
                action = 1
                
                # Count the oscillation
                self.oscillation_count += 1
                
                # Log the oscillation
                logger.warning(f"Forced hold due to severe repeated oscillation at step {current_step}")
        
        # Take the step in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current price and position after the step
        current_price = None
        if 'close_price' in info:
            current_price = info['close_price']
        elif hasattr(self.env, '_get_current_price'):
            current_price = self.env._get_current_price()
        
        # Check if take profit was triggered in the environment
        take_profit_triggered = False
        if 'take_profit_price' in info and info['take_profit_price'] is not None:
            self.last_take_profit_price = info['take_profit_price']
            
            # Check if a take profit sell was just triggered
            if 'take_profit_sells' in info and hasattr(self, 'last_take_profit_sells'):
                if info['take_profit_sells'] > self.last_take_profit_sells:
                    take_profit_triggered = True
                    logger.info(f"SafeTradingEnvWrapper detected take profit sell at step {current_step}")
                self.last_take_profit_sells = info['take_profit_sells']
            else:
                self.last_take_profit_sells = info.get('take_profit_sells', 0)
        
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
            position_id = str(current_step)  # Use step as position identifier
            
            if action == 0:  # Sell (close position)
                # Reduce risk by removing all risk associated with closed positions
                # Instead of blindly subtracting max_risk_per_trade
                for pos_id in list(self.risk_per_position.keys()):
                    # For simplicity, close all positions on a sell
                    self.cumulative_risk -= self.risk_per_position[pos_id]
                    del self.risk_per_position[pos_id]
                
                # Ensure we don't go below zero
                self.cumulative_risk = max(0.0, self.cumulative_risk)
                
            elif action == 2:  # Buy (open position)
                # Add risk for this new position, but respect the maximum
                new_risk = min(self.max_risk_per_trade, 
                              self.max_cumulative_risk - self.cumulative_risk)
                
                if new_risk > 0:
                    self.risk_per_position[position_id] = new_risk
                    self.cumulative_risk += new_risk
                else:
                    # Log that we're at maximum risk
                    logger.warning(f"At maximum risk {self.cumulative_risk:.1%}, cannot take more risk")
                
            # Only log risk management metrics when there's a trade or periodically
            if (current_step % 100000 == 0 or abs(self.cumulative_risk % 0.02) < 0.001):
                position_count = len(self.risk_per_position)
                portfolio_value_str = f"{portfolio_value:.2f}" if portfolio_value is not None else "unknown"
                logger.info(f"Risk management: Cumulative risk now {self.cumulative_risk:.1%}, " +
                           f"Portfolio value: {portfolio_value_str}, Positions: {position_count}")
        
        # Record the action that was actually taken
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.total_actions += 1
        
        # Track action in sliding window
        self.action_window.append(action)
        if len(self.action_window) > self.window_size:
            old_action = self.action_window.pop(0)
        
        # Log action distribution more frequently to monitor hold behavior
        if self.total_actions % 500 == 0:  # Reduced from 1000 to 500
            window_counts = {}
            for a in self.action_window:
                window_counts[a] = window_counts.get(a, 0) + 1
                
            # Calculate portfolio growth if available
            portfolio_growth = 0
            if hasattr(self.env, 'portfolio_value') and self.starting_portfolio > 0:
                portfolio_growth = (self.env.portfolio_value - self.starting_portfolio) / self.starting_portfolio
                
            logger.info(f"Action distribution (last {len(self.action_window)} steps): {window_counts}")
            logger.info(f"Portfolio growth: {portfolio_growth:.4f} (starting: {self.starting_portfolio:.2f}, current: {self.env.portfolio_value:.2f})")
            
            # Detect and warn about persistent hold pattern with a lower threshold
            if window_counts.get(1, 0) > 0.7 * len(self.action_window):  # Reduced from 0.9 to 0.7
                logger.warning(f"PERSISTENT HOLD PATTERN DETECTED: {window_counts.get(1, 0)}/{len(self.action_window)} actions are hold")
                
        # Calculate risk-adjusted rewards after action is taken
        risk_adjusted_reward = self._calculate_risk_rewards(reward, info, trade_occurred, current_step)
        
        return observation, risk_adjusted_reward, terminated, truncated, info


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
        if base_env is not None and hasattr(base_env, 'holding_counter'):
            self.current_hold_duration = base_env.holding_counter
            
            # Track forced sells
            if hasattr(base_env, 'forced_sells'):
                self.forced_sells = base_env.forced_sells
                
            # Update hold duration histogram
            if self.current_hold_duration > 0:
                bucket = min(self.current_hold_duration, 20)  # Cap at 20+
                self.hold_histogram[bucket] = self.hold_histogram.get(bucket, 0) + 1
                
            # When not holding (hold_counter is 0), we just completed a holding period
            if self.current_hold_duration == 0 and len(self.hold_durations) < self.current_hold_duration:
                self.hold_durations.append(self.current_hold_duration)
        
        # Log to TensorBoard on regular intervals
        if self.num_timesteps % self.debug_frequency == 0 and self.verbose > 0:
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
                self.logger.record(f"actions/recent_pattern", hash(action_pattern) % 1000)
                
                # Check for problematic patterns like long holds
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
                    # Try to access action history from various places
                    
                    # First, check if this is a SafeTradingEnvWrapper with action_history
                    if hasattr(env, 'action_history') and env.action_history:
                        # Only take the last 1000 actions to match the log output
                        recent_actions = env.action_history[-1000:]
                        for action in recent_actions:
                            if action is not None and action in self.action_counts:
                                self.action_counts[action] += 1
                                action_counts_updated = True
                    
                    # Also try to access the unwrapped env if it's a wrapper
                    unwrapped = env
                    while hasattr(unwrapped, 'env'):
                        unwrapped = unwrapped.env
                        if hasattr(unwrapped, 'action_history') and unwrapped.action_history:
                            recent_actions = unwrapped.action_history[-1000:]
                            for action in recent_actions:
                                if action is not None and action in self.action_counts:
                                    self.action_counts[action] += 1
                                    action_counts_updated = True
                    
                    # Try to directly access the last_action from various nested environments
                    if hasattr(env, 'last_action') and env.last_action is not None:
                        action = env.last_action
                        if action in self.action_counts:
                            self.action_counts[action] += 1
                            action_counts_updated = True
                    
                    if hasattr(unwrapped, 'last_action') and unwrapped.last_action is not None:
                        action = unwrapped.last_action
                        if action in self.action_counts:
                            self.action_counts[action] += 1
                            action_counts_updated = True
            
            # If we couldn't extract any actions, use the action distribution from the logs
            if not action_counts_updated:
                # Get actions directly from episode information
                if hasattr(self, 'model') and hasattr(self.model, 'ep_info_buffer'):
                    for info in self.model.ep_info_buffer:
                        if 'action' in info:
                            action = info['action']
                            if action in self.action_counts:
                                self.action_counts[action] += 1
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
                episode_action_table += f"| {action_name} | {count} | {percentage:.1f}% |\n"
            
            episode_action_table += "-" * 40
            logger.info(episode_action_table)
            
            # Reset episode action counts
            self.episode_action_counts = {0: 0, 1: 0, 2: 0}
        
        return super().on_episode_end(episode_rewards, episode_lengths, episode_info)


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
    """Check system resources and print their status"""
    # Memory usage
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.percent}% used, {memory.available / (1024**3):.1f}GB available")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU: {cpu_percent}% used")
    
    # GPU usage (if available)
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # in GB
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # in GB
            logger.info(f"GPU: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_cached:.2f}GB cached")
        except Exception as e:
            logger.warning(f"Error checking GPU resources: {e}")


def update_observation_spaces_recursively(env, target_dim, logger, visited=None):
    """
    Recursively update observation spaces in a potentially nested environment.
    
    This function traverses through all layers of a nested VecEnv environment
    and updates the observation space and running mean/std dimension to match
    the target dimension.
    
    Args:
        env: The environment to update
        target_dim: The target dimension for the observation space
        logger: Logger for debugging information
        visited: Set of already visited objects to avoid infinite recursion
    """
    if env is None:
        return
        
    # Initialize visited set if this is the first call
    if visited is None:
        visited = set()
    
    # Use object id to avoid infinite recursion
    obj_id = id(env)
    if obj_id in visited:
        return
    
    # Add to visited set
    visited.add(obj_id)
        
    # Update this environment's observation space if it exists
    if hasattr(env, 'observation_space'):
        current_shape = env.observation_space.shape
        if len(current_shape) == 1 and current_shape[0] != target_dim:
            logger.info(f"Updating observation space from {current_shape} to {(target_dim,)}")
            env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(target_dim,), 
                dtype=np.float32
            )
    
    # Update running mean/std if it exists
    if hasattr(env, 'obs_rms') and hasattr(env.obs_rms, 'mean'):
        if len(env.obs_rms.mean) != target_dim:
            logger.info(f"Updating obs_rms from dim {len(env.obs_rms.mean)} to {target_dim}")
            env.obs_rms.mean = np.zeros(target_dim)
            env.obs_rms.var = np.ones(target_dim)
            env.obs_rms.count = 0
    
    # Check for common VecEnv attributes that might contain nested environments
    for attr_name in ['venv', 'env', 'envs']:
        if hasattr(env, attr_name):
            attr = getattr(env, attr_name)
            
            # Handle different types of nested environments
            if isinstance(attr, list):
                # For environment lists (e.g., SubprocVecEnv)
                for nested_env in attr:
                    update_observation_spaces_recursively(nested_env, target_dim, logger, visited)
            else:
                # For single nested environment
                update_observation_spaces_recursively(attr, target_dim, logger, visited)
    
    # NOTE: Removed 'unwrapped' from the attribute list because it can cause recursion issues


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
    
    # Override exploration parameters to dramatically increase exploration
    # This encourages the agent to try more diverse actions including sell actions
    exploration_fraction = 0.6  # Significantly increased from 0.4 to 0.6, extends exploration period
    exploration_initial_eps = 1.0  # Start with 100% random actions
    exploration_final_eps = 0.2  # Increased from 0.1 to 0.2 - maintain higher exploration even after training
    
    # Get other parameters from args if not overridden
    exploration_fraction = args.exploration_fraction if hasattr(args, 'exploration_fraction') and args.exploration_fraction > 0.1 else exploration_fraction
    exploration_initial_eps = args.exploration_initial_eps if hasattr(args, 'exploration_initial_eps') else exploration_initial_eps
    exploration_final_eps = args.exploration_final_eps if hasattr(args, 'exploration_final_eps') else exploration_final_eps
    target_update_interval = args.target_update_interval
    
    # Log the exploration parameters for clarity
    logger.info(f"Using exploration parameters: initial_eps={exploration_initial_eps}, "
               f"final_eps={exploration_final_eps}, fraction={exploration_fraction}")
    
    # Create tensorboard callback with more frequent debugging
    tb_callback = TensorboardCallback(verbose=1, model_name="DQN", debug_frequency=1000)  # More frequent debugging
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Add resource check callback
    resource_callback = ResourceCheckCallback(check_interval=10000, verbose=1)
    
    # Combine all callbacks
    all_callbacks = [tb_callback, checkpoint_callback, resource_callback]
    if callbacks:
        all_callbacks.extend(callbacks)
    
    # Create the model with enhanced parameters for better exploration
    # Use custom policy that discourages holding
    model = DQN(
        policy=AntiHoldPolicy,  # Use our custom policy that biases against holding
        env=env,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        learning_starts=10000,  # Increased from 5000 to collect more random samples
        batch_size=batch_size,
        tau=1.0,
        gamma=gamma,
        train_freq=1,  # Reduced from 4 to update more frequently
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs={"net_arch": [256, 256], "hold_action_bias": -4.0},  # Increase penalty from -2.0 to -4.0
        verbose=1,
        tensorboard_log="./logs/dqn/"
    )
    
    # Log detailed model configuration
    logger.info(f"DQN model configured with exploration_fraction={exploration_fraction}, "
               f"initial_eps={exploration_initial_eps}, final_eps={exploration_final_eps}, "
               f"buffer_size={buffer_size}, learning_rate={learning_rate}, batch_size={batch_size}")
    
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
    
    # Verify environment observation space dimensions
    obs_space = env.observation_space
    logger.info(f"Environment observation space: {obs_space.shape}")
    
    # Get parameters from args
    learning_rate = args.learning_rate
    gamma = args.gamma
    ent_coef = args.ent_coef
    
    # Set n_steps to represent 1 year of 15-min candles (96 candles per day * 365 days)
    n_steps = args.n_steps if hasattr(args, 'n_steps') else 35040
    
    # Get number of environments
    num_envs = env.num_envs if hasattr(env, 'num_envs') else 1
    
    # Note: A2C doesn't use batch_size directly like DQN. Instead, its effective batch size
    # is determined by n_steps * num_envs (the number of environments running in parallel)
    batch_size = args.batch_size  # Still store for logging purposes
    
    # Log the updated parameters
    logger.info(f"Using n_steps={n_steps} (1 year of 15-min candles: 96 candles/day * 365 days)")
    # Note that this isn't directly passed to A2C but useful to track
    logger.info(f"Using batch size: {batch_size} for A2C training")
    logger.info(f"Training with {num_envs} parallel environments")
    
    # Calculate effective batch size considering the number of environments
    # This is the actual batch size A2C will use internally
    effective_batch_size = n_steps * num_envs
    logger.info(f"Effective batch size (n_steps × num_envs): {effective_batch_size}")
    
    # Increase entropy coefficient significantly to encourage exploration
    ent_coef = max(ent_coef, 0.20)  # Increased from 0.05 to 0.20 for much more exploration
    logger.info(f"Using higher entropy coefficient: {ent_coef} to strongly encourage action exploration")
    
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
    features_dim = 64  # Default feature dimension
    
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
                    "features_dim": features_dim  # Match with the hidden size in the LSTM model
                }
            }
            logger.info("Successfully configured LSTM feature extractor")
            
            # Update all observation spaces in the environment to match the LSTM feature dimension
            logger.info(f"Updating all observation spaces to match feature dimension {features_dim}")
            update_observation_spaces_recursively(env, features_dim, logger)
            
            # Apply the patch to ensure SafeTradingEnvWrapper augmentation works with the new dimension
            logger.info(f"Applying observation augmentation patch for dimension {features_dim}")
            patch_observation_augmentation(env, features_dim)
            
            # Explicitly patch VecNormalize to handle the new dimensions properly
            logger.info(f"Explicitly patching VecNormalize for dimension {features_dim}")
            patch_vec_normalize(env, features_dim, logger)
            
            logger.info("Completed updating observation spaces in all environment wrappers")
            
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
    
    # Log key parameters to ensure they're being used
    logger.info(f"A2C Model initialized with: n_steps={n_steps} (effective batch size: {n_steps * num_envs})")
    logger.info(f"Environment initialized with: initial_balance={args.initial_balance}, max_steps={args.max_steps}")
    
    # Load model from checkpoint if specified
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        try:
            model = A2C.load(args.load_model, env=env)
            # Update key parameters even for loaded models
            model.n_steps = n_steps
            logger.info(f"Updated loaded model with n_steps={n_steps}")
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
    
    # Load LSTM model if specified - already handled earlier
    if args.lstm_model_path:
        logger.info(f"LSTM model loaded from {args.lstm_model_path}")
    
    # Train the model
    total_timesteps = args.timesteps if hasattr(args, 'timesteps') else 1000000
    logger.info(f"Training A2C for {total_timesteps} timesteps")
    
    try:
        # Log the actual parameters before training starts
        logger.info(f"Starting A2C training with n_steps={n_steps} (effective batch size: {n_steps * num_envs}), total_timesteps={total_timesteps}")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            log_interval=10,
            tb_log_name="a2c",
            progress_bar=True
        )
        
        # Save the final model
        model_save_path = os.path.join("models", f"final_a2c_model")
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
    parser.add_argument("--episode_length", type=int, default=None,
                        help="Length of each episode in days (if None, defaults to 365 days or full dataset)")
    
    # Additional training parameters
    parser.add_argument("--learning_rate", type=float, default=0.0003, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="Batch size for training (default: 2048)")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed")
    
    # PPO/A2C-specific parameters
    parser.add_argument("--n_steps", type=int, default=35040, 
                        help="Number of steps per update for PPO/A2C (default: 35040, representing 1 year of 15-min data)")
    parser.add_argument("--ent_coef", type=float, default=0.01, 
                        help="Entropy coefficient for PPO/A2C")
    parser.add_argument("--n_epochs", type=int, default=10, 
                        help="Number of epochs per update for PPO")
    parser.add_argument("--clip_range", type=float, default=0.2, 
                        help="PPO clip range")
    
    # Parallelization parameters
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments for training (default: 1)")
    
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
    parser.add_argument("--max_holding_steps", type=int, default=8,
                        help="Maximum number of steps to hold before forcing a sell (default: 8)")
    parser.add_argument("--take_profit_pct", type=float, default=0.03,
                        help="Take profit percentage for automatic selling (default: 0.03 or 3%)")
    parser.add_argument("--target_cash_ratio", type=str, default="0.3-0.7",
                        help="Target cash ratio range (default: '0.3-0.7', format: 'min-max')")
    
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
    
    # Get max holding steps parameter from args
    max_holding_steps = args.max_holding_steps
    
    # Get take profit percentage from args
    take_profit_pct = args.take_profit_pct
    
    # Parse target cash ratio
    try:
        min_cash_ratio, max_cash_ratio = map(float, args.target_cash_ratio.split('-'))
        if not (0 <= min_cash_ratio <= max_cash_ratio <= 1):
            logger.warning(f"Invalid target cash ratio range: {args.target_cash_ratio}, using default: 0.3-0.7")
            min_cash_ratio, max_cash_ratio = 0.3, 0.7
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse target cash ratio: {args.target_cash_ratio}, using default: 0.3-0.7")
        min_cash_ratio, max_cash_ratio = 0.3, 0.7
    
    logger.info(f"Creating trading environment with cooldown period: {args.trade_cooldown}")
    logger.info(f"Max holding steps: {max_holding_steps}, Take profit: {take_profit_pct*100:.1f}%")
    logger.info(f"Target cash ratio range: {min_cash_ratio:.2f}-{max_cash_ratio:.2f}")
    
    # Calculate candles per day for 15-minute data
    CANDLES_PER_DAY = 96  # 24 hours * 4 candles per hour for 15-minute data

    base_env = CryptocurrencyTradingEnv(
        df=data,
        initial_amount=args.initial_balance,
        buy_cost_pct=args.commission,
        sell_cost_pct=args.commission,
        state_space=16,
        tech_indicator_list=INDICATORS,
        episode_length=args.episode_length,
        randomize_start=True,
        candles_per_day=CANDLES_PER_DAY,  # Add parameter to correctly interpret 15-min data
        max_holding_steps=max_holding_steps,  # Use argument from command line
        take_profit_pct=take_profit_pct  # Use argument from command line
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
        max_history_size=100,
        take_profit_pct=take_profit_pct  # Add take profit percentage to wrapper
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
        "initial_amount": args.initial_balance, 
        "buy_cost_pct": args.commission,
        "sell_cost_pct": args.commission,
        "tech_indicator_list": INDICATORS,
        "episode_length": args.episode_length,
        "candles_per_day": CANDLES_PER_DAY,  # Add parameter to correctly interpret 15-min data
        "max_holding_steps": max_holding_steps,  # Add max holding steps
        "take_profit_pct": take_profit_pct  # Add take profit percentage
    }
    
    # First create a sample environment to determine the observation space
    logger.info("Creating sample environment to determine observation space")
    sample_env = CryptocurrencyTradingEnv(**env_kwargs)
    sample_wrapped = SafeTradingEnvWrapper(
        sample_env,
        trade_cooldown=args.trade_cooldown,
        max_history_size=100,
        max_risk_per_trade=0.02,
        take_profit_pct=take_profit_pct
    )
    
    # Get the observation space from the sample environment
    base_obs_dim = sample_env.observation_space.shape[0]
    wrapped_obs_dim = sample_wrapped.observation_space.shape[0]
    
    logger.info(f"Base observation space dimension: {base_obs_dim}")
    logger.info(f"Wrapped observation space dimension: {wrapped_obs_dim}")
    
    # Close the sample environments to free resources
    try:
        sample_wrapped.close()
        sample_env.close()
        logger.info("Sample environments closed successfully")
    except Exception as e:
        logger.warning(f"Error closing sample environments: {e}")
    
    # Define function to create a wrapped environment with consistent observation space
    def make_env():
        base_env = CryptocurrencyTradingEnv(**env_kwargs)
        
        # Ensure the base environment observation space matches the expected dimension
        if base_env.observation_space.shape[0] != base_obs_dim:
            logger.warning(f"Base environment observation dimension mismatch: expected {base_obs_dim}, got {base_env.observation_space.shape[0]}")
            # Fix by recreating the environment
            base_env = CryptocurrencyTradingEnv(**env_kwargs)
        
        # Verify initial balance
        logger.info(f"Vector env initialized with balance: {base_env.portfolio_value if hasattr(base_env, 'portfolio_value') else 'unknown'}")
        
        safe_env = SafeTradingEnvWrapper(
            base_env, 
            trade_cooldown=args.trade_cooldown,
            max_history_size=100,  # Ensure this is consistent across all environments
            max_risk_per_trade=0.02,  # Add missing parameter
            take_profit_pct=take_profit_pct
        )
        
        # Verify observation space dimension is consistent
        if safe_env.observation_space.shape[0] != wrapped_obs_dim:
            logger.error(f"Observation dimension mismatch: expected {wrapped_obs_dim}, got {safe_env.observation_space.shape[0]}")
            # This should not happen if SafeTradingEnvWrapper behaves consistently
        
        time_limit_env = TimeLimit(safe_env, max_episode_steps=args.max_steps)
        return Monitor(time_limit_env, "logs/monitor/")
    
    # Create vectorized environment with the specified number of environments
    num_envs = args.num_envs
    logger.info(f"Creating vectorized environment with {num_envs} parallel environments")
    
    # Create a list of environment creation functions
    env_fns = [make_env for _ in range(num_envs)]
    
    # Use SubprocVecEnv for true parallelism when num_envs > 1
    if num_envs > 1:
        try:
            vec_env = SubprocVecEnv(env_fns)
            logger.info(f"Using SubprocVecEnv with {num_envs} workers for parallel processing")
        except Exception as e:
            logger.warning(f"Error creating SubprocVecEnv: {e}. Falling back to DummyVecEnv.")
            vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
        logger.info("Using DummyVecEnv with a single environment")
    
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


def patch_observation_augmentation(env, target_dim):
    """
    Add a patch to ensure SafeTradingEnvWrapper can handle different observation dimensions
    when the LSTM feature extractor is used.
    
    Args:
        env: The environment to patch
        target_dim: The target dimension for observations
    """
    if env is None:
        return
    
    # Check if this is a SafeTradingEnvWrapper
    if isinstance(env, SafeTradingEnvWrapper):
        # Monkey patch the _augment_observation method to handle the new dimensions
        original_augment_observation = env._augment_observation
        
        def patched_augment_observation(observation):
            # If observation dimension matches target_dim, return as is
            if isinstance(observation, np.ndarray) and len(observation) == target_dim:
                return observation
            
            # Otherwise, call the original method
            return original_augment_observation(observation)
        
        # Replace the method
        env._augment_observation = patched_augment_observation
        logger.info(f"Patched SafeTradingEnvWrapper._augment_observation to handle {target_dim}-dimensional observations")
    
    # Recursively patch child environments
    if hasattr(env, 'env'):
        patch_observation_augmentation(env.env, target_dim)
    elif hasattr(env, 'venv'):
        patch_observation_augmentation(env.venv, target_dim)
    elif hasattr(env, 'envs') and isinstance(env.envs, list):
        for nested_env in env.envs:
            patch_observation_augmentation(nested_env, target_dim)


def patch_vec_normalize(env, target_dim, logger):
    """
    Explicitly patch VecNormalize to ensure it properly handles the target observation dimension.
    This is needed because VecNormalize has unique handling of observation normalization.
    
    Args:
        env: The environment containing VecNormalize
        target_dim: Target observation dimension
        logger: Logger for debug information
    """
    if env is None:
        return
    
    # Check if this is a VecNormalize environment
    if hasattr(env, 'obs_rms') and hasattr(env.obs_rms, 'mean'):
        logger.info(f"Found VecNormalize, checking observation dimensions")
        
        # Check if dimensions need updating
        if len(env.obs_rms.mean) != target_dim:
            logger.info(f"Updating VecNormalize obs_rms from dim {len(env.obs_rms.mean)} to {target_dim}")
            # Create new running mean and variance with correct dimension
            env.obs_rms.mean = np.zeros(target_dim, dtype=np.float64)
            env.obs_rms.var = np.ones(target_dim, dtype=np.float64)
            env.obs_rms.count = 0  # Reset the count to indicate new statistics
    
    # Continue checking child environments
    if hasattr(env, 'venv'):
        patch_vec_normalize(env.venv, target_dim, logger)
    elif hasattr(env, 'env'):
        patch_vec_normalize(env.env, target_dim, logger)


if __name__ == "__main__":
    # Run the main function
    main()
