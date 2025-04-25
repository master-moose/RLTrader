"""
Trading Environment for reinforcement learning.

This module implements a cryptocurrency trading environment
compatible with the OpenAI Gym interface, supporting bidirectional trading.
"""

from gymnasium import Env, spaces
import numpy as np
import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
import logging
from rl_agent.utils import calculate_trading_metrics

# Get module logger
logger = logging.getLogger("rl_agent.environment")

# Define a small threshold for floating point comparisons
ZERO_THRESHOLD = 1e-9


class TradingEnvironment(Env):
    """
    A cryptocurrency trading environment for reinforcement learning supporting
    bidirectional trading (long and short positions).

    This environment simulates trading a single cryptocurrency
    with features like transaction fees, portfolio tracking,
    and customizable observation space.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str] = None,
        sequence_length: int = 60,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.00075,
        reward_scaling: float = 1.0,
        window_size: int = 20,
        max_position: float = 1.0,  # Max position size as fraction of *balance*
        max_steps: Optional[int] = None,
        random_start: bool = True,
        render_mode: Optional[str] = None,
        # Add exploration parameters with defaults
        exploration_start: float = 1.0,
        exploration_end: float = 0.01,
        exploration_decay_rate: float = 0.0001,
        exploration_bonus_weight: float = 0.1,
        # Add reward component weights with defaults
        portfolio_change_weight: float = 1.0,
        drawdown_penalty_weight: float = 0.5,
        sharpe_reward_weight: float = 0.5,
        fee_penalty_weight: float = 2.0,
        benchmark_reward_weight: float = 0.5,
        # consistency_penalty_weight: float = 0.2, # Removed for now
        idle_penalty_weight: float = 0.1,
        profit_bonus_weight: float = 0.5,
        # Add additional reward parameters
        sharpe_window: int = 20,
        # consistency_threshold: int = 3, # Removed for now
        idle_threshold: int = 5,
        trade_penalty_weight: float = 0.0,
    ):
        """
        Initialize the trading environment.

        Args:
            data: DataFrame with OHLCV data and technical indicators
            features: List of column names to use as features
            sequence_length: Length of price history sequences
            initial_balance: Initial cash balance
            transaction_fee: Fee per transaction as a fraction
            reward_scaling: Scaling factor for the reward
            window_size: Size of the trading time window for rendering
            max_position: Maximum position size as a fraction of balance
                          (used for both long and short entries)
            max_steps: Maximum number of steps in an episode
            random_start: Start from a random position in the data
            render_mode: Gymnasium render mode ('human', 'rgb_array', None)

            # Exploration parameters
            exploration_start: Starting value of exploration bonus
            exploration_end: Ending value of exploration bonus
            exploration_decay_rate: Rate at which exploration bonus decays
            exploration_bonus_weight: Weight of exploration bonus in reward

            # Reward component weights
            portfolio_change_weight: Weight for portfolio value change reward
            drawdown_penalty_weight: Weight for drawdown penalty
            sharpe_reward_weight: Weight for Sharpe ratio reward
            fee_penalty_weight: Weight for transaction fee penalty
            benchmark_reward_weight: Weight for benchmark comparison reward
            # consistency_penalty_weight: Weight for trade consistency penalty
            idle_penalty_weight: Weight for idle position penalty (when flat)
            profit_bonus_weight: Weight for profit bonus (on closing trades)

            # Additional reward parameters
            sharpe_window: Window size for Sharpe ratio calculation
            # consistency_threshold: Min consecutive actions before flip ok
            idle_threshold: Num consecutive holds (flat) before idle penalty
            trade_penalty_weight: Weight for penalty on entering trades
        """
        super(TradingEnvironment, self).__init__()

        # Store parameters
        self.data = data.copy()
        self.features = features or ["close", "volume", "open", "high", "low"]
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        self.max_position = max_position
        self.sequence_length = sequence_length
        self.random_start = random_start
        self.render_mode = render_mode

        # Store exploration parameters
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.exploration_decay_rate = exploration_decay_rate
        self.exploration_bonus_weight = exploration_bonus_weight

        # Store reward component weights
        self.portfolio_change_weight = portfolio_change_weight
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.sharpe_reward_weight = sharpe_reward_weight
        self.fee_penalty_weight = fee_penalty_weight
        self.benchmark_reward_weight = benchmark_reward_weight
        # self.consistency_penalty_weight = consistency_penalty_weight
        self.idle_penalty_weight = idle_penalty_weight
        self.profit_bonus_weight = profit_bonus_weight

        # Store additional reward parameters
        self.sharpe_window = sharpe_window
        # self.consistency_threshold = consistency_threshold
        self.idle_threshold = idle_threshold
        self.trade_penalty_weight = trade_penalty_weight

        # Data preprocessing
        for feature in self.features:
            if feature not in self.data.columns:
                raise ValueError(
                    f"Feature '{feature}' not found in data columns: "
                    f"{data.columns.tolist()}"
                )

        # --- Define action and observation spaces ---
        # Action space: 0: Hold, 1: Go Long, 2: Go Short, 3: Close Position
        self.action_space = spaces.Discrete(4)

        # Observation space: features + position_type + normalized_entry_price
        # position_type: -1 (Short), 0 (Flat), 1 (Long)
        # normalized_entry_price: (current_price / entry_price) - 1 if pos != 0 else 0
        feature_space_dim = len(self.features) * self.sequence_length
        # Add 2 dimensions for position type and normalized entry price
        observation_dim = feature_space_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )
        # --- End Define action and observation spaces ---

        # Set maximum number of steps
        if max_steps is None:
            self.max_steps = len(self.data) - self.sequence_length - 1
        else:
            self.max_steps = min(max_steps,
                                 len(self.data) - self.sequence_length - 1)

        logger.debug(
            f"Created TradingEnvironment with {len(self.data)} data points, "
            f"{len(self.features)} features, sequence length "
            f"{self.sequence_length}, and Obs Dim {observation_dim}"
        )

        # Initialize state
        self.reset()

    def reset(self, seed=None):
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed

        Returns:
            Initial observation and info dictionary
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset the position in the data
        if self.random_start:
            # Ensure start leaves room for sequence_length and max_steps
            max_start_index = len(self.data) - self.max_steps - 1
            min_start_index = self.sequence_length
            if max_start_index <= min_start_index:
                 # Handle cases where data is too short for max_steps
                 self.current_step = min_start_index
                 logger.warning(
                     f"Data length ({len(self.data)}) too short for "
                     f"sequence_length ({self.sequence_length}) and "
                     f"max_steps ({self.max_steps}). Starting at {self.current_step}." # noqa E501
                 )
            else:
                self.current_step = np.random.randint(min_start_index, max_start_index) # noqa E501
            logger.debug(f"Starting from random position: {self.current_step}")
        else:
            self.current_step = self.sequence_length
            logger.debug(f"Starting from beginning: {self.current_step}")

        # Reset the account state
        self.balance = self.initial_balance
        self.shares_held = 0.0  # Magnitude of position (always non-negative)
        self.position_type = 0  # -1: Short, 0: Flat, 1: Long
        self.entry_price = None  # Price at which current position was entered
        self.asset_value = 0.0   # Current market value of held shares (if long)
        # Portfolio value calculation depends on position type (see _update_portfolio_value)
        self.portfolio_value = self.balance
        self.max_portfolio_value = self.portfolio_value

        # Reset trade history
        self.trades = []
        self.portfolio_values = [self.portfolio_value]
        self.rewards = []

        # Reset metrics
        self.total_trades = 0  # Total trades (long entries + short entries)
        self.total_longs = 0
        self.total_shorts = 0
        self.total_closes = 0
        self.total_holds = 0
        self.max_drawdown = 0.0  # Reset max drawdown

        # Initialize new state variables
        self.step_returns = []  # For Sharpe ratio calculation
        self.episode_start_price = self.data['close'].iloc[self.current_step]
        self.consecutive_holds = 0 # Track consecutive holds while FLAT
        # self.consecutive_buys = 0 # Removed
        # self.consecutive_sells = 0 # Removed
        self.last_action = 0  # Assume initial action is Hold (0)
        self.exploration_bonus_value = self.exploration_start
        self.total_fees_paid = 0.0
        self.failed_trades = 0 # General counter for failed actions
        # Add cumulative reward component trackers
        self.cumulative_rewards = {
            'portfolio_change': 0.0,
            'drawdown_penalty': 0.0,
            'sharpe_reward': 0.0,
            'fee_penalty': 0.0,
            'benchmark_reward': 0.0,
            # 'consistency_penalty': 0.0, # Removed
            'idle_penalty': 0.0,
            'profit_bonus': 0.0,
            'exploration_bonus': 0.0,
            'trade_penalty': 0.0,
        }

        # Steps taken within the current episode
        self.episode_step = 0

        # Get initial observation and info
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to take (0: Hold, 1: Go Long, 2: Go Short, 3: Close)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        prev_portfolio_value = self.portfolio_value
        prev_fees_paid = self.total_fees_paid
        prev_position_type = self.position_type # Track previous position

        # Execute the action
        self._take_action(action)

        # --- Update portfolio and check drawdown BEFORE incrementing step ---
        self._update_portfolio_value() # Uses self.current_step price

        # Update max portfolio value based on value at step t (for metrics/rewards)
        self.max_portfolio_value = max(self.max_portfolio_value,
                                       self.portfolio_value)

        # Calculate drawdown based on peak value at step t (for metrics/rewards)
        current_drawdown_from_peak = 0.0
        # Avoid division by zero if max_portfolio_value is zero or negative
        if self.max_portfolio_value > ZERO_THRESHOLD:
            current_drawdown_from_peak = max(0.0, (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value) # noqa E501 Ensure drawdown isn't negative
        self.max_drawdown = max(self.max_drawdown, current_drawdown_from_peak) # Track max drawdown from peak

        # Check for early stopping based on drawdown from INITIAL BALANCE at step t
        initial_balance_drawdown_terminated = False
        # Using 0.50 (50%) of initial balance as the termination threshold
        termination_portfolio_threshold = self.initial_balance * 0.50
        if self.portfolio_value < termination_portfolio_threshold:
            initial_balance_drawdown_terminated = True

        # --------------------------------------------------------------------

        # Store portfolio value history for step t
        self.portfolio_values.append(self.portfolio_value)

        # Calculate step return for Sharpe ratio (t vs t-1)
        if abs(prev_portfolio_value) > ZERO_THRESHOLD:
            step_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value # noqa E501
            self.step_returns.append(step_return)
        else:
            # Avoid division by zero, append 0 return
            self.step_returns.append(0.0)

        # Calculate reward based on change from t-1 to t
        fee_paid_this_step = self.total_fees_paid - prev_fees_paid
        # Pass previous position type to reward calculation if needed
        reward_info = self._calculate_reward(action, prev_portfolio_value,
                                             fee_paid_this_step,
                                             prev_position_type)
        reward = reward_info['total_reward']
        self.rewards.append(reward)

        # --- Update consecutive holds (only when flat) ---
        if self.position_type == 0 and action == 0: # If flat and held
             self.consecutive_holds += 1
        else:
             self.consecutive_holds = 0 # Reset if not flat or didn't hold flat

        # --- Now move to the next time step (t+1) ---
        self.current_step += 1
        self.episode_step += 1

        # --- Check episode end conditions based on step t+1 ---
        is_end_of_data = self.current_step >= len(self.data) - 1
        is_max_steps_reached = (self.max_steps is not None and
                                self.episode_step >= self.max_steps)

        # Determine termination/truncation
        # Terminate if drawdown from initial balance > 50% OR end of data reached
        terminated = initial_balance_drawdown_terminated or is_end_of_data
        # Truncate ONLY if max steps reached AND not already terminated
        truncated = is_max_steps_reached and not terminated
        # ---------------------------------------------------

        # --- Log Termination/Truncation Reason ---
        termination_reason = "None"
        if terminated:
            if initial_balance_drawdown_terminated:
                # Calculate drawdown from initial for logging clarity
                drawdown_pct_from_initial = 0.0
                if self.initial_balance > ZERO_THRESHOLD:
                    drawdown_pct_from_initial = max(0.0, (self.initial_balance - self.portfolio_value) / self.initial_balance) # noqa E501
                termination_reason = f"Drawdown > 50% of Initial Balance ({drawdown_pct_from_initial:.2%})" # noqa E501
            elif is_end_of_data:
                termination_reason = "End of Data"
        elif truncated:
            termination_reason = f"Max Steps Reached ({self.max_steps})"

        # --- EPISODE END CHECK ---
        episode_ended = terminated or truncated
        if episode_ended:
            # Ensure portfolio is updated with the final price if not end of data # noqa E501
            if not is_end_of_data:
                self._update_portfolio_value()

            # If episode ends with an open position, close it automatically
            # to get final PnL reflected in the balance.
            # This simulates closing out at the end of the test period.
            if self.position_type != 0:
                 logger.info(f"[EPISODE END] Auto-closing {('Long' if self.position_type == 1 else 'Short')} position at step {self.current_step-1}.") # noqa E501
                 self._close_position(self.data['close'].iloc[self.current_step - 1]) # noqa E501 Close using price of the *last valid step*
                 self._update_portfolio_value() # Update PV one last time after closing # noqa E501


            final_info = self._get_info() # Get info *after* potential auto-close # noqa E501
            logger.info(
                (
                    f"[EPISODE END] Step {self.current_step-1} | EpStep {self.episode_step}\n" # noqa E501
                    f"Terminated: {terminated} | Truncated: {truncated} | Reason: {termination_reason}\n" # noqa E501
                    f"Final Portfolio Value: {self.portfolio_value:.2f}\n"
                    f"Profit: {self.portfolio_value - self.initial_balance:.2f}\n" # noqa E501
                    f"Episode Return: {final_info.get('episode_return', 0.0):.4f}\n" # noqa E501
                    f"Sharpe (episode): {final_info.get('sharpe_ratio_episode', 0.0):.4f}\n" # noqa E501
                    f"Sharpe (rolling): {final_info.get('sharpe_ratio_rolling', 0.0):.4f}\n" # noqa E501
                    f"Calmar: {final_info.get('calmar_ratio', 0.0):.4f}\n"
                    f"Sortino: {final_info.get('sortino_ratio', 0.0):.4f}\n"
                    f"Total Trades: {self.total_trades} | Longs: {self.total_longs} | Shorts: {self.total_shorts} | Closes: {self.total_closes}\n" # noqa E501
                    f"Max Drawdown: {self.max_drawdown:.2%}"
                )
            )
        # --- End Log ---

        # Decay exploration bonus for the next step
        if self.exploration_decay_rate > 0 and self.exploration_bonus_value > self.exploration_end: # noqa E501
            self.exploration_bonus_value -= self.exploration_decay_rate

        # Get new observation and info for the *next* step
        observation = self._get_observation() # Based on self.current_step (t+1)
        info = self._get_info() # Based on self.current_step (t+1)

        # Update last action taken
        self.last_action = action

        # Gymnasium expects 5 return values: obs, reward, terminated, truncated, info # noqa E501
        return observation, reward, terminated, truncated, info

    def _take_action(self, action):
        """
        Execute the trading action (Hold, Go Long, Go Short, Close Position).

        Args:
            action: Action to take (0, 1, 2, 3)
        """
        current_price = self.data['close'].iloc[self.current_step]
        action_taken = False # Flag to track if a trade action occurred

        # --- Action Logic ---
        if action == 0:  # Hold / Stay Flat
            if self.position_type == 0: # Only count holds if flat
                 self.total_holds += 1
                 # logger.debug(f"Step {self.current_step}: Held Flat.")
            else:
                 # logger.debug(f"Step {self.current_step}: Held {'Long' if self.position_type == 1 else 'Short'} Position.") # noqa E501
                 pass # No change if holding an existing position

        elif action == 1:  # Go Long
            if self.position_type == 0: # Can only go long if flat
                invest_amount = self.balance * self.max_position
                # Optional: Add min trade value check if needed
                # min_trade_value = self.initial_balance * 0.005
                # if invest_amount >= min_trade_value:
                if invest_amount > ZERO_THRESHOLD and current_price > ZERO_THRESHOLD: # noqa E501
                    fee_multiplier = 1 + self.transaction_fee
                    shares_to_buy = invest_amount / (current_price * fee_multiplier) # noqa E501
                    if shares_to_buy > ZERO_THRESHOLD:
                        cost = shares_to_buy * current_price
                        fee = cost * self.transaction_fee
                        self.balance -= (cost + fee)
                        self.balance = max(0, self.balance) # Ensure balance >= 0
                        self.shares_held = shares_to_buy
                        self.position_type = 1 # Set position to Long
                        self.entry_price = current_price
                        self.total_trades += 1
                        self.total_longs += 1
                        action_taken = True
                        trade_info = {
                            'step': self.current_step, 'type': 'long_entry',
                            'price': current_price, 'shares': self.shares_held,
                            'cost': cost, 'fee': fee,
                            'balance_after': self.balance
                        }
                        self.trades.append(trade_info)
                        logger.debug(
                            f"Step {self.current_step}: Entered LONG {self.shares_held:.6f} @ {current_price:.2f} (Cost: {cost:.2f}, Fee: {fee:.2f}) -> Bal: {self.balance:.2f}" # noqa E501
                        )
                    else:
                        self.failed_trades += 1
                        logger.debug(f"Step {self.current_step}: Attempted Long, but calculated shares {shares_to_buy:.8f} <= threshold.") # noqa E501
                else:
                    self.failed_trades += 1
                    logger.debug(f"Step {self.current_step}: Attempted Long, but insufficient balance ({self.balance:.2f}) or zero price.") # noqa E501
                # else: # Min trade value check
                #     self.failed_trades += 1
                #     logger.debug(f"Step {self.current_step}: Attempted Long. Invest amount {invest_amount:.2f} < min trade value {min_trade_value:.2f}.") # noqa E501
            else: # Already in a position
                 self.failed_trades += 1
                 logger.debug(f"Step {self.current_step}: Attempted Long, but already in a {'Long' if self.position_type == 1 else 'Short'} position.") # noqa E501

        elif action == 2:  # Go Short
            if self.position_type == 0: # Can only go short if flat
                # Short amount based on max_position fraction of *current balance*
                # (Simulating margin implicitly)
                short_value_target = self.balance * self.max_position
                # Optional: Add min trade value check if needed
                if short_value_target > ZERO_THRESHOLD and current_price > ZERO_THRESHOLD: # noqa E501
                    shares_to_short = short_value_target / current_price
                    if shares_to_short > ZERO_THRESHOLD:
                        proceeds = shares_to_short * current_price
                        fee = proceeds * self.transaction_fee
                        self.balance += (proceeds - fee) # Add proceeds minus fee
                        self.shares_held = shares_to_short # Store magnitude
                        self.position_type = -1 # Set position to Short
                        self.entry_price = current_price
                        self.total_trades += 1
                        self.total_shorts += 1
                        action_taken = True
                        trade_info = {
                            'step': self.current_step, 'type': 'short_entry',
                            'price': current_price, 'shares': self.shares_held,
                            'proceeds': proceeds, 'fee': fee,
                            'balance_after': self.balance
                        }
                        self.trades.append(trade_info)
                        logger.debug(
                            f"Step {self.current_step}: Entered SHORT {self.shares_held:.6f} @ {current_price:.2f} (Proceeds: {proceeds:.2f}, Fee: {fee:.2f}) -> Bal: {self.balance:.2f}" # noqa E501
                        )
                    else:
                        self.failed_trades += 1
                        logger.debug(f"Step {self.current_step}: Attempted Short, but calculated shares {shares_to_short:.8f} <= threshold.") # noqa E501
                else:
                    self.failed_trades += 1
                    logger.debug(f"Step {self.current_step}: Attempted Short, but insufficient target value ({short_value_target:.2f}) or zero price.") # noqa E501
            else: # Already in a position
                 self.failed_trades += 1
                 logger.debug(f"Step {self.current_step}: Attempted Short, but already in a {'Long' if self.position_type == 1 else 'Short'} position.") # noqa E501

        elif action == 3:  # Close Position
            if self.position_type != 0: # Can only close if Long or Short
                 closed_pnl = self._close_position(current_price)
                 if closed_pnl is not None: # Check if close was successful
                      action_taken = True
                      self.total_closes += 1
                      logger.debug(f"Step {self.current_step}: Closed Position. PnL: {closed_pnl:.2f} -> Bal: {self.balance:.2f}") # noqa E501
                 else:
                      # _close_position logs the failure reason
                      self.failed_trades += 1
            else: # Already flat
                 self.failed_trades += 1
                 logger.debug(f"Step {self.current_step}: Attempted Close, but already Flat.") # noqa E501

        # If a trade action was taken (entry or close), update fees
        if action_taken:
            # Fee calculation is handled within the specific action logic (buy/sell/close) # noqa E501
            # self.total_fees_paid is updated internally
            pass

        # No need to update self.last_action here, done in step()

    def _close_position(self, closing_price):
        """
        Helper function to close the current Long or Short position.
        Updates balance, shares_held, position_type, entry_price, total_fees_paid.
        Records the closing trade in self.trades.

        Args:
            closing_price: The price at which the position is closed.

        Returns:
            The realized Profit or Loss (PnL) from the closed trade,
            or None if the close failed.
        """
        if self.position_type == 0 or self.shares_held <= ZERO_THRESHOLD:
            logger.warning(f"Step {self.current_step}: Attempted _close_position when already flat or zero shares.") # noqa E501
            return None # Should not happen if called correctly

        initial_balance = self.balance # For PnL calculation later
        shares_to_transact = self.shares_held
        pnl = 0.0
        fee = 0.0
        trade_type = ''

        if self.position_type == 1: # Closing a Long position (Sell)
            trade_type = 'long_exit'
            sell_value = shares_to_transact * closing_price
            fee = sell_value * self.transaction_fee
            self.balance += sell_value - fee
            pnl = (closing_price - self.entry_price) * shares_to_transact - fee
            logger.debug(f"Closing Long: {shares_to_transact:.6f} sold @ {closing_price:.2f}, Entry: {self.entry_price:.2f}, Fee: {fee:.2f}") # noqa E501

        elif self.position_type == -1: # Closing a Short position (Buy)
            trade_type = 'short_exit'
            buy_cost = shares_to_transact * closing_price
            fee_multiplier = 1 + self.transaction_fee
            # Cost including fee to check if affordable
            total_cost = buy_cost * fee_multiplier
            fee = buy_cost * self.transaction_fee

            # Check if we have enough balance to buy back the short position
            if self.balance < total_cost - ZERO_THRESHOLD: # Allow for small float errors # noqa E501
                 logger.warning(
                     f"Step {self.current_step}: Insufficient balance ({self.balance:.2f}) "
                     f"to close short position requiring {total_cost:.2f}. "
                     f"Cannot close."
                 )
                 # Should this trigger termination? Maybe higher drawdown penalty?
                 # For now, fail the close action.
                 return None # Indicate failure

            self.balance -= total_cost # Deduct cost + fee
            # PnL for short = (Entry Price - Closing Price) * Shares - Fee
            pnl = (self.entry_price - closing_price) * shares_to_transact - fee
            logger.debug(f"Closing Short: {shares_to_transact:.6f} bought @ {closing_price:.2f}, Entry: {self.entry_price:.2f}, Fee: {fee:.2f}") # noqa E501


        # Record the trade
        trade_info = {
            'step': self.current_step, 'type': trade_type,
            'price': closing_price, 'shares': shares_to_transact,
            'entry_price': self.entry_price, 'pnl': pnl, 'fee': fee,
            'balance_after': self.balance
        }
        self.trades.append(trade_info)

        # Update state
        self.total_fees_paid += fee
        self.shares_held = 0.0
        self.position_type = 0
        self.entry_price = None

        return pnl # Return the calculated PnL for this specific trade

    def _update_portfolio_value(self):
        """
        Update the portfolio value based on current balance, position type,
        shares held, entry price, and current asset price.
        """
        current_price = self.data['close'].iloc[self.current_step]
        self.asset_value = 0.0 # Reset asset value

        if self.position_type == 1:  # Long Position
            self.asset_value = self.shares_held * current_price
            self.portfolio_value = self.balance + self.asset_value
        elif self.position_type == -1:  # Short Position
            # Portfolio value = Cash Balance + Unrealized PnL from Short
            # Unrealized PnL = (Entry Price - Current Price) * Shares Held
            # Note: Fee to close is not included here, only realized on close
            unrealized_pnl = (self.entry_price - current_price) * self.shares_held if self.entry_price is not None else 0 # noqa E501
            self.portfolio_value = self.balance + unrealized_pnl
            # Asset value for shorts could be considered negative liability
            self.asset_value = -self.shares_held * current_price
        else:  # Flat Position
            self.portfolio_value = self.balance
            self.asset_value = 0.0 # Explicitly zero

        # --- DETAILED LOGGING FOR PORTFOLIO VALUE --- #
        # logger.debug(
        #     f"_update_portfolio_value (Step {self.current_step}): "
        #     f"PosType={self.position_type}, Price={current_price:.4f}, "
        #     f"Shares={self.shares_held:.8f}, Bal={self.balance:.4f}, "
        #     f"Entry={self.entry_price}, AssetVal={self.asset_value:.4f}, "
        #     f"PortVal={self.portfolio_value:.4f}"
        # )
        # --- Check for non-finite values immediately after calculation ---
        if not np.isfinite(self.portfolio_value):
            logger.error(
                f"_update_portfolio_value: NON-FINITE PORTFOLIO VALUE DETECTED! " # noqa E501
                f"(Step {self.current_step})"
                f" PosType={self.position_type}, Price={current_price:.4f}, "
                f"Shares={self.shares_held:.8f}, Bal={self.balance:.4f}, "
                f"Entry={self.entry_price}"
                f" -> AssetVal={self.asset_value}, PortVal={self.portfolio_value}" # noqa E501
            )
            # Attempt to recover or terminate? For now, log error.
            # Consider setting portfolio value to a very low number or initial balance? # noqa E501
            # self.portfolio_value = self.initial_balance # Emergency reset?

    def _calculate_reward(self, action, prev_portfolio_value,
                          fee_paid_this_step, prev_position_type):
        """
        Calculate the reward based on multiple components.

        Args:
            action: The action taken (0: Hold, 1: Long, 2: Short, 3: Close)
            prev_portfolio_value: Portfolio value before the action & update
            fee_paid_this_step: Transaction fee incurred in this step
            prev_position_type: Position type (-1, 0, 1) before the action

        Returns:
            A dictionary containing the breakdown of reward components and
            the total scaled reward.
        """
        reward_components = {
            'portfolio_change': 0.0, 'drawdown_penalty': 0.0,
            'sharpe_reward': 0.0, 'fee_penalty': 0.0,
            'benchmark_reward': 0.0, # 'consistency_penalty': 0.0, # Removed
            'idle_penalty': 0.0, 'profit_bonus': 0.0,
            'exploration_bonus': 0.0, 'trade_penalty': 0.0,
            'raw_total': 0.0, 'total_reward': 0.0
        }

        current_price = self.data['close'].iloc[self.current_step]

        # 1. Portfolio Value Change Reward (Percentage Change)
        portfolio_change_pct = 0.0
        if abs(prev_portfolio_value) > ZERO_THRESHOLD:
            portfolio_change_pct = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value # noqa E501
        reward_components['portfolio_change'] = portfolio_change_pct * self.portfolio_change_weight # noqa E501

        # 2. Drawdown Penalty
        current_drawdown = 0.0
        if self.max_portfolio_value > ZERO_THRESHOLD:
            # Ensure drawdown >= 0
            current_drawdown = max(0.0, (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value) # noqa E501
        reward_components['drawdown_penalty'] = -current_drawdown * self.drawdown_penalty_weight # noqa E501

        # 3. Sharpe Ratio Reward (Rolling Window)
        sharpe_ratio_rolling = 0.0
        if len(self.step_returns) >= self.sharpe_window:
            window_returns = np.array(self.step_returns[-self.sharpe_window:])
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            if std_return > ZERO_THRESHOLD:
                sharpe_ratio_rolling = mean_return / std_return
        reward_components['sharpe_reward'] = sharpe_ratio_rolling * self.sharpe_reward_weight # noqa E501

        # 4. Fee Penalty
        # Penalize based on the portion of portfolio value spent on fees this step # noqa E501
        fee_penalty_normalized = 0.0
        if abs(prev_portfolio_value) > ZERO_THRESHOLD:
             fee_penalty_normalized = fee_paid_this_step / prev_portfolio_value # noqa E501
        reward_components['fee_penalty'] = -abs(fee_penalty_normalized) * self.fee_penalty_weight # noqa E501

        # 5. Benchmark Comparison Reward (Optional - Based on Buy & Hold)
        # ... (kept commented out)

        # 6. Direct Trade *Entry* Penalty
        trade_penalty = 0.0
        # Apply penalty if entering a new long (1) or short (2) position
        if action in [1, 2]:
            trade_penalty = -1.0 # Base penalty for entering a trade
        calculated_trade_penalty = trade_penalty * self.trade_penalty_weight
        reward_components['trade_penalty'] = calculated_trade_penalty

        # 7. Idle Penalty (Only when Flat)
        idle_penalty = 0.0
        # Apply penalty if FLAT (pos_type=0) and Held (action=0)
        # for more than idle_threshold steps
        if self.position_type == 0 and action == 0 and self.consecutive_holds > self.idle_threshold: # noqa E501
            idle_penalty = -1.0 # Base penalty for being idle too long
        reward_components['idle_penalty'] = idle_penalty * self.idle_penalty_weight # noqa E501

        # 8. Profit/Closing Bonus (Only when Closing a position)
        profit_bonus = 0.0
        # Check if the action was Close (3) AND the position actually changed from Long/Short to Flat # noqa E501
        if action == 3 and prev_position_type != 0 and self.position_type == 0:
            # PnL was calculated in _close_position
            # Find the PnL from the last trade entry in self.trades
            last_trade = self.trades[-1] if self.trades else None
            if last_trade and last_trade['type'] in ['long_exit', 'short_exit']: # noqa E501
                trade_pnl = last_trade.get('pnl', 0.0)
                # Normalize PnL by entry value?
                entry_value = 0
                if last_trade['type'] == 'long_exit':
                     entry_value = last_trade.get('shares', 0) * last_trade.get('entry_price', 0) # noqa E501
                elif last_trade['type'] == 'short_exit':
                     # Value at entry for short is also based on entry price
                     entry_value = last_trade.get('shares', 0) * last_trade.get('entry_price', 0) # noqa E501

                if entry_value > ZERO_THRESHOLD:
                     pnl_pct = trade_pnl / entry_value
                     if pnl_pct > 0: # Only reward profitable closes
                          # Scale bonus by profit percentage
                          profit_bonus = pnl_pct * 2 # Adjust multiplier as needed # noqa E501
                     # else: Add penalty for losing closes? Optional.
                     #     profit_bonus = pnl_pct * some_penalty_multiplier

        reward_components['profit_bonus'] = profit_bonus * self.profit_bonus_weight # noqa E501

        # 9. Exploration Bonus (Optional)
        # ... (kept commented out)

        # Sum all active reward components
        components_to_sum = [
            k for k in reward_components
            if k not in ['raw_total', 'total_reward']
        ]
        raw_total = sum(reward_components[key] for key in components_to_sum)
        reward_components['raw_total'] = raw_total

        # Apply reward scaling for the final reward
        reward_components['total_reward'] = raw_total * self.reward_scaling

        # --- Update cumulative reward sums ---
        for key in self.cumulative_rewards:
            if key in reward_components:
                # Ensure value is finite before adding
                comp_value = reward_components[key]
                if np.isfinite(comp_value):
                     self.cumulative_rewards[key] += comp_value
                else:
                     logger.warning(f"Non-finite reward component '{key}': {comp_value}. Skipping cumulative update.") # noqa E501

        # --- Logging ---
        # Log only if reward is non-trivial or components are non-zero
        if abs(reward_components['total_reward']) > 1e-6 or any(abs(v) > 1e-6 for k, v in reward_components.items() if k not in ['raw_total', 'total_reward']): # noqa E501
             component_str = ', '.join([f"{k}: {v:.4f}" for k, v in reward_components.items() # noqa E501
                                       if k not in ['raw_total', 'total_reward'] and abs(v) > 1e-6]) # noqa E501
             logger.debug(
                 f"Step {self.current_step} (EpStep {self.episode_step}) Act {action} -> Reward: {reward_components['total_reward']:.4f} | Components: {component_str}" # noqa E501
             )

        return reward_components

    def _get_observation(self):
        """
        Get the current observation, including historical features,
        position type, and normalized entry price.

        Returns:
            Numpy array containing the observation
        """
        # Check if current_step is valid
        if self.current_step < 0 or self.current_step >= len(self.data):
             logger.error(f"Invalid current_step ({self.current_step}) in _get_observation. Data length: {len(self.data)}") # noqa E501
             # Return a zero observation or handle error appropriately
             # This might happen if step() increments current_step beyond bounds before obs is needed # noqa E501
             # Let's assume step checks handle this, but add safety.
             # If called at the very end, use the last valid step?
             safe_step = min(self.current_step, len(self.data) - 1)
             # Fallback: return zeros matching shape
             # return np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
             safe_step = self.current_step

        # Get historical feature data (sequence_length prior steps including current) # noqa E501
        end_idx = safe_step
        start_idx = max(0, end_idx - self.sequence_length + 1)

        # Ensure we have enough data points relative to sequence length
        actual_len = end_idx - start_idx + 1
        if actual_len < 1: # Should not happen if safe_step is valid
             logger.error(f"Observation calculation error: start_idx {start_idx} > end_idx {end_idx}") # noqa E501
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        historical_data = self.data.iloc[start_idx:end_idx+1]

        # Ensure we have exactly sequence_length rows for feature data
        if len(historical_data) < self.sequence_length:
            # Pad with the first available row if needed
            num_padding = self.sequence_length - len(historical_data)
            if not historical_data.empty:
                padding_df = pd.DataFrame([historical_data.iloc[0]] * num_padding, index=range(num_padding)) # noqa E501
                historical_data = pd.concat([padding_df, historical_data.reset_index(drop=True)], ignore_index=True) # noqa E501
            else: # If historical_data is somehow empty (e.g., start_idx problem) # noqa E501
                # Pad with zeros or a default row structure
                logger.warning(f"Historical data empty in _get_observation at step {safe_step}. Padding with zeros.") # noqa E501
                # Create a zero DataFrame matching feature columns
                zero_data = {feat: [0.0] * self.sequence_length for feat in self.features} # noqa E501
                historical_data = pd.DataFrame(zero_data)


        # Extract features and flatten
        feature_data = []
        for feature in self.features:
             # Ensure feature exists and handle potential NaNs
             if feature in historical_data.columns:
                  values = historical_data[feature].fillna(0).values # Fill NaNs with 0 # noqa E501
                  feature_data.extend(values)
             else:
                  logger.warning(f"Feature '{feature}' not found in historical data for observation. Appending zeros.") # noqa E501
                  feature_data.extend([0.0] * self.sequence_length)

        # --- Add Position Information ---
        # 1. Position Type (-1, 0, 1)
        position_type_feature = float(self.position_type)

        # 2. Normalized Entry Price
        # (current_price / entry_price) - 1 if in position, else 0
        normalized_entry_price = 0.0
        if self.position_type != 0 and self.entry_price is not None and self.entry_price > ZERO_THRESHOLD: # noqa E501
            current_price_obs = self.data['close'].iloc[safe_step]
            if current_price_obs > ZERO_THRESHOLD:
                normalized_entry_price = (current_price_obs / self.entry_price) - 1.0 # noqa E501
        # --- End Position Information ---

        # Combine features and position info
        observation_list = feature_data + [position_type_feature, normalized_entry_price] # noqa E501
        observation = np.array(observation_list, dtype=np.float32)

        # --- Validate Observation Shape ---
        expected_shape = self.observation_space.shape
        if observation.shape != expected_shape:
            logger.error(
                f"Observation shape mismatch! Expected {expected_shape}, "
                f"got {observation.shape}. Step: {safe_step}. "
                f"FeatDataLen: {len(feature_data)}"
            )
            # Attempt to reshape or pad/truncate? Or return zeros?
            # Returning zeros is safer to avoid downstream errors.
            observation = np.zeros(expected_shape, dtype=np.float32)
        elif not np.all(np.isfinite(observation)):
             logger.error(
                 f"Non-finite values found in observation at step {safe_step}! " # noqa E501
                 f"Observation: {observation}"
             )
             # Replace NaNs/Infs with 0
             observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0) # noqa E501


        return observation

    def _get_info(self):
        """
        Get additional information about the environment state.

        Returns:
            Dictionary containing environment info
        """
        # Use safe_step in case called at the very end
        safe_step = min(self.current_step, len(self.data) - 1)
        current_price = self.data['close'].iloc[safe_step]

        # Base info
        info = {
            'step': safe_step,
            'episode_step': self.episode_step,
            'timestamp': self.data.index[safe_step],
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,       # Magnitude
            'position_type': self.position_type,  # -1, 0, 1
            'entry_price': self.entry_price,      # Price of entry
            'asset_value': self.asset_value,      # Value if long, neg liability if short # noqa E501
            'portfolio_value': self.portfolio_value,
            'initial_balance': self.initial_balance,
            # Cash ratio needs careful definition with shorting.
            # Let's define it as balance / portfolio_value when PV > 0
            'cash_ratio': self.balance / self.portfolio_value
                          if self.portfolio_value > ZERO_THRESHOLD else 1.0,
            'total_trades': self.total_trades,    # Entries (long + short)
            'total_longs': self.total_longs,
            'total_shorts': self.total_shorts,
            'total_closes': self.total_closes,
            'total_holds': self.total_holds,      # Holds while flat
            'max_drawdown': self.max_drawdown,
            'total_fees_paid': self.total_fees_paid,
            'failed_trades': self.failed_trades,  # Failed action attempts
            'last_action': self.last_action,
            'consecutive_holds': self.consecutive_holds, # Holds while flat
            'exploration_bonus_value': self.exploration_bonus_value,
        }

        # --- Calculate overall episode returns and metrics ---
        if len(self.portfolio_values) > 1:
            # Use robust calculation for episode return
            initial_val = self.initial_balance
            current_val = self.portfolio_value
            episode_return = 0.0
            if np.isfinite(initial_val) and np.isfinite(current_val) and abs(initial_val) > ZERO_THRESHOLD: # noqa E501
                 try:
                     calculated_return = (current_val - initial_val) / initial_val # noqa E501
                     if np.isfinite(calculated_return):
                         episode_return = calculated_return
                     else:
                         logger.warning(f"_get_info: Calculated episode_return non-finite ({calculated_return}). Forcing to 0.") # noqa E501
                 except Exception as e:
                     logger.error(f"_get_info: Error calculating episode_return: {e}", exc_info=True) # noqa E501
            elif abs(initial_val) <= ZERO_THRESHOLD:
                 logger.warning(f"_get_info: Initial balance near zero ({initial_val}). Return set to 0.") # noqa E501
            elif not np.isfinite(initial_val) or not np.isfinite(current_val):
                 logger.warning(f"_get_info: Non-finite values for return calc (Initial: {initial_val}, Current: {current_val}). Return set to 0.") # noqa E501

            info['episode_return'] = episode_return

            # Calculate overall Sharpe from portfolio_values
            # Ensure portfolio_values contains finite numbers
            finite_pv = [pv for pv in self.portfolio_values if np.isfinite(pv)]
            if len(finite_pv) > 1:
                 pv_array = np.array(finite_pv)
                 # --- REVISED RETURN CALCULATION ---
                 # Calculate returns using the standard formula: (p[t] - p[t-1]) / p[t-1]
                 # Only where the denominator p[t-1] is valid (not near zero)
                 valid_denom_mask = np.abs(pv_array[:-1]) > ZERO_THRESHOLD
                 
                 if np.any(valid_denom_mask): # Check if there are any valid denominators
                     # Select corresponding slices for numerator (diff) and denominator
                     p_t = pv_array[1:][valid_denom_mask]    # Values at t where p[t-1] is valid
                     p_t_minus_1 = pv_array[:-1][valid_denom_mask] # Valid denominator values at t-1
                     
                     # Ensure we have values after masking
                     if p_t.size > 0 and p_t_minus_1.size > 0 and p_t.shape == p_t_minus_1.shape:
                         # Calculate differences
                         diffs = p_t - p_t_minus_1
                         
                         # Denominator already checked, but add safety clamp
                         safe_denominators = np.where(np.abs(p_t_minus_1) > 1e-9, p_t_minus_1, 1e-9)
                         
                         portfolio_returns = diffs / safe_denominators
                         
                         # --- Sharpe Calculation (using calculated returns) ---
                         if len(portfolio_returns) > 1:
                              mean_return = np.mean(portfolio_returns)
                              std_return = np.std(portfolio_returns)
                              if std_return > ZERO_THRESHOLD:
                                   info['sharpe_ratio_episode'] = mean_return / std_return
                              else: info['sharpe_ratio_episode'] = 0.0
                         else: info['sharpe_ratio_episode'] = 0.0
                         # --- End Sharpe Calculation ---
                         
                     else: # Shape mismatch or empty after mask (should not happen with np.any check, but safety)
                         logger.warning(f"_get_info: Shape mismatch or empty array after masking for return calc. p_t: {p_t.shape}, p_t-1: {p_t_minus_1.shape}. Setting Sharpe=0.")
                         info['sharpe_ratio_episode'] = 0.0
                 else:
                     # No valid denominators found
                     info['sharpe_ratio_episode'] = 0.0
                 # --- END REVISED RETURN CALCULATION ---
            else:
                 # Not enough finite portfolio values for calculation
                 info['sharpe_ratio_episode'] = 0.0
        else:
            info['episode_return'] = 0.0
            info['sharpe_ratio_episode'] = 0.0
            info['sharpe_ratio_rolling'] = 0.0

        # Benchmark (Buy & Hold) Performance
        benchmark_portfolio_value = self.initial_balance # Default
        benchmark_return = 0.0
        if abs(self.episode_start_price) > ZERO_THRESHOLD:
            initial_units = self.initial_balance / self.episode_start_price
            benchmark_portfolio_value = initial_units * current_price
            if abs(self.initial_balance) > ZERO_THRESHOLD:
                 benchmark_return = (benchmark_portfolio_value - self.initial_balance) / self.initial_balance # noqa E501

        info['benchmark_portfolio_value'] = benchmark_portfolio_value
        info['benchmark_return'] = benchmark_return

        # Add cumulative reward components to info
        info['cumulative_reward_components'] = self.cumulative_rewards.copy()

        # Calculate and add Calmar/Sortino Ratios at episode end
        info['calmar_ratio'] = 0.0
        info['sortino_ratio'] = 0.0
        # Use finite portfolio values for metrics
        finite_pv_metrics = [pv for pv in self.portfolio_values if np.isfinite(pv)] # noqa E501
        if len(finite_pv_metrics) > 2:
            try:
                portfolio_values_np = np.array(finite_pv_metrics)
                # Ensure no NaNs/Infs passed to calculate_trading_metrics
                if np.all(np.isfinite(portfolio_values_np)):
                     trading_metrics = calculate_trading_metrics(portfolio_values_np) # noqa E501
                     if trading_metrics:
                         info['calmar_ratio'] = trading_metrics.get('calmar_ratio', 0.0) # noqa E501
                         info['sortino_ratio'] = trading_metrics.get('sortino_ratio', 0.0) # noqa E501
                else:
                     logger.warning("_get_info: Non-finite values remain in portfolio_values_np for metrics calculation.") # noqa E501

            except Exception as e:
                logger.error(f"_get_info: Error calculating trading metrics: {e}", exc_info=True) # noqa E501

        return info

    def render(self):
        """
        Render the environment (human mode or rgb_array).
        """
        if self.render_mode == 'human':
            print(
                f"Step: {self.current_step}, "
                f"Portfolio: {self.portfolio_value:.2f}, "
                f"Balance: {self.balance:.2f}, "
                f"Position: {'Long' if self.position_type == 1 else ('Short' if self.position_type == -1 else 'Flat')}, " # noqa E501
                f"Shares: {self.shares_held:.4f}"
            )
            return None

        elif self.render_mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted size
            start = max(0, self.current_step - self.window_size)
            end = self.current_step + 1
            plot_data = self.data.iloc[start:end]

            # Plot price
            ax.plot(plot_data.index, plot_data['close'], label='Close Price', color='grey', alpha=0.8) # noqa E501

            # --- Mark Trades ---
            long_entries = [t for t in self.trades if t['type'] == 'long_entry' and start <= t['step'] < end] # noqa E501
            long_exits = [t for t in self.trades if t['type'] == 'long_exit' and start <= t['step'] < end] # noqa E501
            short_entries = [t for t in self.trades if t['type'] == 'short_entry' and start <= t['step'] < end] # noqa E501
            short_exits = [t for t in self.trades if t['type'] == 'short_exit' and start <= t['step'] < end] # noqa E501

            if long_entries:
                 entry_steps = [t['step'] for t in long_entries]
                 entry_prices = [t['price'] for t in long_entries]
                 indices = plot_data.index.searchsorted(self.data.index[entry_steps]) # noqa E501
                 ax.scatter(plot_data.index[indices], entry_prices, marker='^', color='lime', s=100, label='Long Entry', zorder=5) # noqa E501
            if long_exits:
                 exit_steps = [t['step'] for t in long_exits]
                 exit_prices = [t['price'] for t in long_exits]
                 indices = plot_data.index.searchsorted(self.data.index[exit_steps]) # noqa E501
                 ax.scatter(plot_data.index[indices], exit_prices, marker='^', color='darkgreen', s=100, label='Long Exit', zorder=5) # noqa E501

            if short_entries:
                 entry_steps = [t['step'] for t in short_entries]
                 entry_prices = [t['price'] for t in short_entries]
                 indices = plot_data.index.searchsorted(self.data.index[entry_steps]) # noqa E501
                 ax.scatter(plot_data.index[indices], entry_prices, marker='v', color='red', s=100, label='Short Entry', zorder=5) # noqa E501
            if short_exits:
                 exit_steps = [t['step'] for t in short_exits]
                 exit_prices = [t['price'] for t in short_exits]
                 indices = plot_data.index.searchsorted(self.data.index[exit_steps]) # noqa E501
                 ax.scatter(plot_data.index[indices], exit_prices, marker='v', color='darkred', s=100, label='Short Exit', zorder=5) # noqa E501
            # --- End Mark Trades ---

            # Add portfolio value overlay? Optional
            ax2 = ax.twinx()
            portfolio_plot_values = self.portfolio_values[max(0, len(self.portfolio_values) - (end-start)):] # noqa E501
            # Ensure portfolio values align with price plot length
            if len(portfolio_plot_values) == len(plot_data.index):
                 ax2.plot(plot_data.index, portfolio_plot_values, label='Portfolio Value', color='cyan', alpha=0.5, linestyle='--') # noqa E501
                 ax2.set_ylabel("Portfolio Value", color='cyan')
                 ax2.tick_params(axis='y', labelcolor='cyan')


            ax.set_title(f"Trading Environment - Step {self.current_step}")
            ax.set_ylabel("Price")
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
            ax.grid(True, alpha=0.3)

            # Draw figure and convert to RGB array
            fig.canvas.draw()
            rgb_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)

            return rgb_array
        else:
            # If render_mode is None or unsupported
            return None

    def close(self):
        """Clean up resources."""
        plt.close('all') # Close all matplotlib figures

    def seed(self, seed=None):
        """Set random seed for environment and action space."""
        super().reset(seed=seed) # Call parent seed method
        # Seed the action space's random number generator if needed
        # self.action_space.seed(seed) # Only if using Samplable space
        return [seed] 