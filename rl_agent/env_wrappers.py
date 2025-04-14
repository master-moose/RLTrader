#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment wrappers for RL agents.

This module contains wrappers that enhance environment functionality
with features like trading safeguards, risk management, and observation
space augmentation.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

# Import necessary constants
from .config import (
    logger,
    TRADE_COOLDOWN_PERIOD,
    HOLD_ACTION,
    OSCILLATION_PENALTY,
    SAME_PRICE_TRADE_PENALTY,
    MAX_TRADE_FREQUENCY
)


class SafeTradingEnvWrapper(gymnasium.Wrapper):
    """
    A wrapper for trading environments that adds safeguards against:
    1. Rapid trading (enforces a cooldown period)
    2. Action oscillation (detects and prevents buy-sell-buy patterns)
    3. Risk management (enforces position sizing based on risk)
    """

    def __init__(self, env, trade_cooldown=TRADE_COOLDOWN_PERIOD, max_history_size=100,
                 max_risk_per_trade=0.02, take_profit_pct=0.03):
        """Initialize the wrapper with safeguards against harmful trading patterns"""
        super().__init__(env)

        # Trading safeguards - set a more balanced cooldown
        self.trade_cooldown = 3  # Increased from 1 to 3 to prevent rapid oscillation
        if trade_cooldown <= 0:
            logger.warning(
                f"Specified trade_cooldown was {trade_cooldown}, "
                f"setting to minimum of 3 to prevent division by zero"
            )

        # Increase minimum cooldown to prevent rapid trades
        self.min_cooldown = 3  # Increased from 1 to 3

        # Initialize step counter and cooldown tracking
        self.current_step = 0
        self.in_cooldown = False
        self.cooldown_steps = 0
        self.has_active_positions = False # Track if there are open positions

        # Risk management parameters - relax constraints
        self.max_risk_per_trade = max_risk_per_trade * 1.5
        self.max_history_size = max_history_size
        self.target_risk_reward_ratio = 0.3  # Less strict
        self.risk_adjusted_position_sizing = True
        self.cumulative_risk = 0.0
        self.max_cumulative_risk = 0.3  # Allow more risk
        self.risk_per_position = {}

        # Take profit parameters
        self.take_profit_pct = take_profit_pct

        # Trading history tracking
        self.last_trade_step = -self.trade_cooldown  # Start cooldown passed
        self.last_trade_price = None
        self.action_history = []
        self.position_history = []
        self.current_position = 0.0
        self.previous_position = 0.0
        self.forced_actions = 0
        self.cooldown_violations = 0
        self.trade_returns = []
        self.same_price_trades = 0
        self.successful_trade_streak = 0
        self.max_successful_streak = 0
        self.max_drawdown = 0.0 # Initialize max_drawdown
        self.last_buy_price = None
        self.last_sell_price = None
        self.last_take_profit_price = None # Added from reset logic

        # Attributes from dqn_old.py __init__
        self.consecutive_holds = 0
        self.consecutive_same_action = 0
        self.last_action = None
        self.position_size_pct = 0.1  # Start with small position sizes
        self.training_progress = 0.0  # Track progress from 0.0 to 1.0
        self.successful_trades = 0
        self.failed_trades = 0
        self.trade_pnl = []
        self.oscillation_window = 8  # Look at 8 actions for oscillation patterns
        self.progressive_cooldown = True  # Increase cooldown after oscillations
        self.max_oscillation_cooldown = 10  # Set to a fixed value
        self.oscillation_patterns = {
            'buy_sell_alternation': 0,
            'rapid_reversals': 0,
        }
        self.oscillation_count = 0 # Added from reset logic
        self.sharpe_ratio = 0.0 # Added from reset logic
        self.last_take_profit_sells = 0 # Internal tracking for take profit logic

        # Get the portfolio value from the environment
        if hasattr(self.env, 'portfolio_value') and self.env.portfolio_value > 0:
            self.peak_value = self.env.portfolio_value
            # Use initial portfolio value for tracking growth
            self.starting_portfolio = self.env.portfolio_value
            self.highest_portfolio = self.starting_portfolio
            logger.info(
                f"SafeTradingEnvWrapper initialized with portfolio value: "
                f"{self.starting_portfolio:.2f}"
            )
        else:
            self.peak_value = 10000.0  # Default if no portfolio value
            self.starting_portfolio = 10000.0
            self.highest_portfolio = self.starting_portfolio
            logger.warning(
                "Environment does not provide portfolio value, "
                "using default 10000.0"
            )

        self.portfolio_growth_rate = 0.0

        self.current_cooldown = max(3, trade_cooldown)  # Ensure cooldown >= 3

        # Track metrics for analyzing agent behavior
        self.hold_duration = 0
        self.profitable_trades = 0
        self.total_trades = 0
        self.win_rate = 0.0
        self.trade_count = 0 # Added from step logic

        # Add a dynamic profit threshold that decreases over time
        self.initial_profit_threshold = 0.001  # Reduced
        self.min_profit_threshold = 0.0003

        # Adjust observation space to include the new features
        if isinstance(self.env.observation_space, spaces.Box):
            original_shape = self.env.observation_space.shape
            if len(original_shape) == 1:
                additional_features = 24  # 15 + 3 + 3 + 2 + 1
                low = np.append(
                    self.env.observation_space.low, [-np.inf] * additional_features
                )
                high = np.append(
                    self.env.observation_space.high, [np.inf] * additional_features
                )
                self.observation_space = spaces.Box(
                    low=low,
                    high=high,
                    dtype=self.env.observation_space.dtype
                )
                logger.info(
                    f"Expanded observation space from {original_shape[0]} to "
                    f"{original_shape[0] + additional_features} dimensions"
                )
        else:
            logger.warning(
                f"Keeping original observation space of type "
                f"{type(self.env.observation_space)} - augmentation only "
                f"supports Box spaces"
            )

        logger.info(
            f"SafeTradingEnvWrapper initialized with {trade_cooldown} step cooldown"
        )

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
        self.last_action = None # Reset last action
        self.trade_returns = [] # Reset trade returns
        self.trade_pnl = [] # Reset PnL tracking
        self.successful_trades = 0
        self.failed_trades = 0
        self.last_take_profit_sells = 0
        self.trade_count = 0

        # Get the initial portfolio value from the environment
        if hasattr(self.env, 'portfolio_value'):
            self.peak_value = self.env.portfolio_value
            # Log the starting portfolio value for this episode
            logger.debug(
                f"New episode starting with portfolio value: "
                f"{self.env.portfolio_value:.2f}"
            )
        else:
            self.peak_value = 0.0

        # Reset performance metrics for the new episode
        self.portfolio_growth_rate = 0.0
        self.sharpe_ratio = 0.0

        # Add action history to observation
        observation = self._augment_observation(observation)

        self.cumulative_risk = 0.0  # Reset cumulative risk
        self.risk_per_position = {}  # Reset risk per position

        return observation, info

    def _augment_observation(self, observation):
        """Add action history and risk metrics to the observation space"""
        orig_obs = observation
        action_history_enc = np.zeros(15) # One-hot encoding for last 5 actions
        for i, action in enumerate(self.action_history[-5:]):
            if action is not None and i < 5:
                action_idx = min(int(action), 2)
                offset = i * 3
                action_history_enc[offset + action_idx] = 1.0

        consistency_metrics = np.array([
            min(self.consecutive_same_action / 10.0, 1.0),
            min(self.consecutive_holds / 20.0, 1.0),
            min(self.oscillation_count / 50.0, 1.0),
        ])

        risk_metrics = np.array([
            max(min(self.sharpe_ratio / 2.0, 1.0), -1.0),
            min(self.max_drawdown, 1.0),
            1.0 if self.consecutive_holds > 10 else 0.0,
        ])

        cash_metrics = np.zeros(2)
        cash_ratio = getattr(self.env, 'cash_ratio', None)
        if cash_ratio is not None and 0 <= cash_ratio <= 1:
            cash_metrics[0] = min(max(0, cash_ratio), 1.0)
            if cash_ratio < 0.2:
                cash_metrics[1] = (0.2 - cash_ratio) * 5.0
            elif cash_ratio > 0.8:
                cash_metrics[1] = (cash_ratio - 0.8) * 5.0
            else:
                cash_metrics[1] = 0.0

        current_step = getattr(self.env, 'day', 0)
        if self.current_cooldown <= 0:
            self.current_cooldown = 1
            logger.warning("Cooldown period was 0, setting to 1")
        cooldown_status = min(max((current_step - self.last_trade_step) / self.current_cooldown, 0), 1)

        augmented_features = np.concatenate([
            action_history_enc,
            consistency_metrics,
            risk_metrics,
            cash_metrics,
            np.array([cooldown_status]),
        ])

        if isinstance(orig_obs, np.ndarray):
            augmented_observation = np.concatenate([orig_obs, augmented_features])
        else:
            logger.warning("Non-array observation type, returning original")
            return orig_obs

        return augmented_observation

    # --- Methods copied from dqn_old.py --- #

    def _detect_oscillation_patterns(self):
        """Detect oscillation patterns in recent actions"""
        if len(self.action_history) < self.oscillation_window:
            return False

        recent_actions = self.action_history[-self.oscillation_window:]
        alternation_detected = False
        for i in range(len(recent_actions) - 3):
            if (recent_actions[i] == 2 and recent_actions[i+1] == 0 and
                    recent_actions[i+2] == 2 and recent_actions[i+3] == 0):
                alternation_detected = True
                self.oscillation_patterns['buy_sell_alternation'] += 1
                logger.warning(f"Detected oscillation: {recent_actions[i:i+4]} at step {getattr(self.env, 'current_step', 0)}")
                break
            if (recent_actions[i] == 0 and recent_actions[i+1] == 2 and
                    recent_actions[i+2] == 0 and recent_actions[i+3] == 2):
                alternation_detected = True
                self.oscillation_patterns['buy_sell_alternation'] += 1
                logger.warning(f"Detected oscillation: {recent_actions[i:i+4]} at step {getattr(self.env, 'current_step', 0)}")
                break

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

        if alternation_detected or reversal_detected:
            self.oscillation_count += 1
            return True
        return False

    def _update_cooldown_period(self):
        """Adjust cooldown period based on oscillation patterns"""
        oscillation_score = 0
        oscillation_score += self.oscillation_patterns['buy_sell_alternation'] * 1.0
        oscillation_score += self.oscillation_patterns['rapid_reversals'] * 0.7

        if oscillation_score > 5:
            new_cooldown = self.max_oscillation_cooldown
        elif oscillation_score > 2:
            new_cooldown = self.min_cooldown + (oscillation_score - 2) * (self.max_oscillation_cooldown - self.min_cooldown) / 5
        else:
            new_cooldown = self.min_cooldown

        new_cooldown = max(self.min_cooldown, new_cooldown)

        if self.current_cooldown > 0:
            max_change = 2
            if new_cooldown > self.current_cooldown:
                new_cooldown = min(new_cooldown, self.current_cooldown + max_change)
            else:
                new_cooldown = max(new_cooldown, self.current_cooldown - max_change)

        if abs(new_cooldown - self.current_cooldown) >= 1:
            self.current_cooldown = int(new_cooldown)
            logger.info(f"Adjusted cooldown to {self.current_cooldown} steps (score: {oscillation_score:.1f})")

    def _calculate_risk_rewards(self, reward, info, trade_occurred, current_step):
        """Calculate risk-aware rewards based on trading performance"""
        portfolio_value = info.get('portfolio_value', 0)
        cash_ratio = info.get('cash_ratio', 1.0)
        risk_adjusted_reward = reward

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        if portfolio_value > self.highest_portfolio:
            prev_highest = self.highest_portfolio
            self.highest_portfolio = portfolio_value
            self.portfolio_growth_rate = (self.highest_portfolio - prev_highest) / max(prev_highest, 1.0)
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown) # Correctly update max_drawdown

        if trade_occurred:
            if self.last_trade_price is not None and 'close_price' in info:
                trade_return = (info['close_price'] - self.last_trade_price) / self.last_trade_price
                self.trade_returns.append(trade_return)
                if trade_return > 0:
                    self.successful_trades += 1
                    self.successful_trade_streak += 1
                else:
                    self.successful_trade_streak = 0
                if len(self.trade_returns) >= 5:
                    try:
                        returns_array = np.array(self.trade_returns[-20:])
                        mean_return = np.mean(returns_array)
                        std_return = np.std(returns_array) + 1e-6
                        self.sharpe_ratio = mean_return / std_return * np.sqrt(252) # Annualized
                    except Exception as e:
                        logger.warning(f"Error calculating Sharpe: {e}")

        if portfolio_value > self.starting_portfolio:
            growth_pct = (portfolio_value - self.starting_portfolio) / self.starting_portfolio
            growth_reward = min(growth_pct * 1.5, 2.0)
            risk_adjusted_reward += growth_reward
            if self.portfolio_growth_rate > 0:
                new_high_reward = min(self.portfolio_growth_rate * 2.0, 0.5)
                risk_adjusted_reward += new_high_reward

        if self.max_drawdown > 0.50:
            drawdown_penalty = self.max_drawdown * 0.3
            risk_adjusted_reward -= drawdown_penalty

        if cash_ratio < 0.1 and current_step > 1000:
            balance_penalty = min((0.1 - cash_ratio) * 2.0, 0.5)
            risk_adjusted_reward -= balance_penalty
        elif cash_ratio > 0.9 and current_step > 2000:
            opportunity_cost = min((cash_ratio - 0.9) * 0.3, 0.3)
            risk_adjusted_reward -= opportunity_cost
        elif 0.3 <= cash_ratio <= 0.7:
            balance_bonus = 0.05
            risk_adjusted_reward += balance_bonus

        if trade_occurred:
            trade_bonus = 0.2
            risk_adjusted_reward += trade_bonus
            if self.action_history and self.action_history[-1] == 0: # sell
                sell_bonus = 0.3
                risk_adjusted_reward += sell_bonus
            elif self.action_history and self.action_history[-1] == 2: # buy
                buy_bonus = 0.2
                risk_adjusted_reward += buy_bonus

        if self.action_history and self.action_history[-1] == HOLD_ACTION:
            self.consecutive_holds += 1
            hold_penalty = 0.3
            risk_adjusted_reward -= hold_penalty
            if self.consecutive_holds > 5:
                add_penalty = min((self.consecutive_holds - 5) * 0.05, 1.0)
                risk_adjusted_reward -= add_penalty
        else:
            self.consecutive_holds = 0

        if self.action_history and self.action_history[-1] != HOLD_ACTION:
            if current_step < 30000:
                action_bonus = 0.1
                risk_adjusted_reward += action_bonus

        if trade_occurred and 'trade_profit' in info and info['trade_profit'] > 0:
            self.successful_trade_streak += 1
            streak_bonus = min(self.successful_trade_streak * 0.05, 0.5)
            risk_adjusted_reward += streak_bonus
            self.max_successful_streak = max(self.max_successful_streak, self.successful_trade_streak)
        elif trade_occurred: # Reset streak on non-profitable trade
            self.successful_trade_streak = 0

        return risk_adjusted_reward

    def _calculate_position_size(self, action, current_price, stop_loss_price):
        """Calculate appropriate position size based on risk parameters"""
        if action == HOLD_ACTION:
            return 0.0
        portfolio_value = getattr(self.env, 'portfolio_value', 10000.0)
        risk_amount = portfolio_value * self.max_risk_per_trade

        if stop_loss_price is None or current_price is None:
            price_distance = current_price * 0.02 if current_price is not None else 0.02
        else:
            price_distance = abs(current_price - stop_loss_price)
        if price_distance <= 0:
            price_distance = current_price * 0.01 if current_price is not None else 0.01

        position_size = risk_amount / price_distance

        if self.risk_adjusted_position_sizing and (self.cumulative_risk + self.max_risk_per_trade) > self.max_cumulative_risk:
            available_risk = max(0, self.max_cumulative_risk - self.cumulative_risk)
            position_size *= (available_risk / self.max_risk_per_trade)
        return position_size

    def _calculate_risk_reward_ratio(self, current_price, entry_price, target_price, stop_loss_price):
        """Calculate the risk-reward ratio for a potential trade"""
        if None in (current_price, entry_price, target_price, stop_loss_price):
            return 0.0
        risk = abs(entry_price - stop_loss_price)
        reward = abs(target_price - entry_price)
        if risk <= 0:
            return 0.0
        return reward / risk

    def _check_stop_loss_and_take_profit(self, action):
         # Placeholder - logic handled by underlying env in dqn_old.py
        pass

    # --- Overridden step method with merged logic --- #

    def step(self, action):
        """Take a step in the environment with safeguards"""
        current_price = getattr(self.env, 'current_price', None)
        self.current_step = getattr(self.env, 'current_step', self.current_step + 1)
        self.total_actions += 1
        self.last_action = action # Store last action

        steps_since_last_trade = self.current_step - self.last_trade_step
        self.in_cooldown = steps_since_last_trade < self.current_cooldown

        force_hold = False
        if len(self.action_history) >= 4:
            recent_actions = self.action_history[-4:]
            if ((recent_actions == [2, 0, 2, 0] or recent_actions == [0, 2, 0, 2]) and
                    ((action == 2 and recent_actions[-1] == 0) or (action == 0 and recent_actions[-1] == 2))):
                logger.warning(f"Oscillation pattern detected: {recent_actions} + [{action}] at step {self.current_step}")
                force_hold = True
                self.oscillation_count += 1

        if self.current_step > 10:
            oscillation_detected = self._detect_oscillation_patterns()
            if oscillation_detected and self.progressive_cooldown:
                self._update_cooldown_period()

        allowed_action = action
        if force_hold:
            allowed_action = HOLD_ACTION
            logger.warning(f"Forcing hold due to oscillation at step {self.current_step}")
            self.forced_actions += 1
        elif self.in_cooldown:
            allowed_action = HOLD_ACTION
            if action != HOLD_ACTION: # Log cooldown violation only if a trade was attempted
                 self.cooldown_violations += 1
                 logger.debug(f"Action {action} blocked by cooldown ({steps_since_last_trade}/{self.current_cooldown} steps)")

        # Logic to check stop loss/take profit is assumed to be within env.step or handled by `_check_stop_loss_and_take_profit` if needed
        # self._check_stop_loss_and_take_profit(allowed_action) # Call placeholder if needed

        # Take step with allowed action
        observation, reward, terminated, truncated, info = self.env.step(allowed_action)

        # Augment observation AFTER stepping the base env
        augmented_observation = self._augment_observation(observation)

        # Update action history with the ORIGINAL intended action
        self.action_history.append(action)
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]

        # Update consecutive counters based on ORIGINAL action
        if len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 0
        if action == HOLD_ACTION:
            self.consecutive_holds += 1
        # Removed else block that reset holds - _calculate_risk_rewards handles this

        # Track position changes
        self.previous_position = self.current_position
        self.current_position = info.get('position', getattr(self.env, 'position', 0.0))
        self.position_history.append(self.current_position)
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)

        trade_occurred = self.previous_position != self.current_position
        if trade_occurred:
            self.last_trade_step = self.current_step
            self.last_trade_price = current_price
            self.trade_count += 1
            portfolio_value = info.get('portfolio_value', None)
            position_id = str(self.current_step)
            if action == 0: # Sell
                closed_risk = sum(self.risk_per_position.values())
                self.cumulative_risk -= closed_risk # More accurate risk reduction
                self.risk_per_position.clear()
            elif action == 2: # Buy
                new_risk = min(self.max_risk_per_trade, max(0, self.max_cumulative_risk - self.cumulative_risk))
                if new_risk > 0:
                    self.risk_per_position[position_id] = new_risk
                    self.cumulative_risk += new_risk
                else:
                     logger.warning(f"At max risk {self.cumulative_risk:.1%}, cannot add more.")
            self.cumulative_risk = max(0, self.cumulative_risk) # Ensure risk is non-negative

        # Record the allowed action taken
        self.action_counts[allowed_action] = self.action_counts.get(allowed_action, 0) + 1
        self.action_window.append(allowed_action)
        if len(self.action_window) > self.window_size:
            self.action_window.pop(0)

        # Log distributions periodically
        if self.total_actions % 500 == 0:
            window_counts = {a: self.action_window.count(a) / len(self.action_window) for a in set(self.action_window)}
            portfolio_value = info.get('portfolio_value', self.starting_portfolio)
            portfolio_growth = (portfolio_value - self.starting_portfolio) / self.starting_portfolio if self.starting_portfolio > 0 else 0
            logger.info(f"Action dist (last {len(self.action_window)}): {window_counts}")
            logger.info(f"Portfolio growth: {portfolio_growth:.4f}")
            if window_counts.get(HOLD_ACTION, 0) > 0.7:
                logger.warning(f"Persistent hold: {window_counts.get(HOLD_ACTION, 0)*100:.1f}% hold actions")

        # Calculate risk-adjusted reward using the original reward and updated state
        risk_adjusted_reward = self._calculate_risk_rewards(reward, info, trade_occurred, self.current_step)

        # --- Add Wrapper Info ---
        # Add relevant metrics calculated by this wrapper to the info dictionary
        # Use prefixes to avoid collisions with base env info keys
        info['wrapper_consecutive_holds'] = self.consecutive_holds
        info['wrapper_oscillation_count'] = self.oscillation_count
        info['wrapper_current_cooldown'] = self.current_cooldown
        info['wrapper_sharpe_ratio'] = self.sharpe_ratio # Calculated in risk rewards
        info['wrapper_max_drawdown'] = self.max_drawdown # Calculated in risk rewards
        info['wrapper_portfolio_growth_rate'] = self.portfolio_growth_rate # Calculated in risk rewards
        info['wrapper_successful_trade_streak'] = self.successful_trade_streak
        info['wrapper_forced_actions'] = self.forced_actions
        info['wrapper_cooldown_violations'] = self.cooldown_violations

        return augmented_observation, risk_adjusted_reward, terminated, truncated, info

    # --- Existing set_target_dimension method --- #

    def set_target_dimension(self, target_dim):
        """
        Set the target dimension for observations.
        Needed for LSTM feature extraction.
        """
        self._target_dim = target_dim
        logger.info(f"Set target observation dimension to {target_dim} in SafeTradingEnvWrapper")
        return self