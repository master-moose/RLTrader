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
# <<< Added CuPy import >>>
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
# <<< End Added CuPy import >>>
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

        # <<< Added: Pre-clean Data >>>
        logger.info("Pre-cleaning data (fillna(0), replace inf)...")
        # Ensure 'close' is cleaned if not already a feature
        cols_to_clean = self.features + ['close'] if 'close' not in self.features else self.features
        # Make sure we only clean columns that actually exist
        cols_to_clean = [col for col in cols_to_clean if col in self.data.columns]
        initial_nans = self.data[cols_to_clean].isnull().sum().sum()
        initial_infs = np.isinf(self.data[cols_to_clean].values).sum()

        if initial_nans > 0:
             logger.warning(f"Found {initial_nans} NaN values in feature/close columns before cleaning.")
             self.data[cols_to_clean] = self.data[cols_to_clean].fillna(0)
        if initial_infs > 0:
             logger.warning(f"Found {initial_infs} Inf values in feature/close columns before cleaning.")
             self.data[cols_to_clean] = self.data[cols_to_clean].replace([np.inf, -np.inf], 0)

        # Verify cleaning (optional check)
        final_nans = self.data[cols_to_clean].isnull().sum().sum()
        final_infs = np.isinf(self.data[cols_to_clean].values).sum()
        if final_nans > 0 or final_infs > 0:
            logger.error(f"Data cleaning failed! NaNs: {final_nans}, Infs: {final_infs}")
        else:
            logger.info("Data pre-cleaning complete.")
        # <<< End Added: Pre-clean Data >>>


        # <<< Added: Convert features to NumPy/CuPy array >>>
        logger.info("Converting features to NumPy array...")
        self.feature_array = self.data[self.features].values.astype(np.float32)
        logger.info(
            f"NumPy feature array created with shape: {self.feature_array.shape}"
        )

        self.feature_array_gpu = None
        # Use the module-level CUPY_AVAILABLE check
        if CUPY_AVAILABLE:
            try:
                logger.info(
                    "CuPy available. Transferring feature array to GPU..."
                )
                self.feature_array_gpu = cp.asarray(self.feature_array)
                logger.info("Feature array transferred to GPU.")
            except Exception as e:
                logger.error(
                    f"Failed to transfer feature array to CuPy: {e}. "
                    f"Falling back to NumPy.",
                    exc_info=True
                )
        else:
            logger.info(
                "CuPy not available or disabled. "
                "Using NumPy for feature operations."
            )
        # <<< End Added: Convert features to NumPy/CuPy array >>>

        # --- Define action and observation spaces ---
        # Action space: Continuous value between -1 (max short/close)
        # and 1 (max long/close)
        # Interpretation thresholds will be used in _take_action
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

        # Observation space: features + pos_type + norm_entry_price
        # position_type: -1 (Short), 0 (Flat), 1 (Long)
        # normalized_entry_price: (curr / entry) - 1 if pos != 0 else 0
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
                    f"max_steps ({self.max_steps}). "
                    f"Starting at {self.current_step}."
                )
            else:
                self.current_step = np.random.randint(
                    min_start_index,
                    max_start_index
                )
            logger.debug(f"Starting from random position: {self.current_step}")
        else:
            self.current_step = self.sequence_length
            logger.debug(f"Starting from beginning: {self.current_step}")

        # Reset the account state
        self.balance = self.initial_balance
        self.shares_held = 0.0  # Magnitude of position (always non-negative)
        self.position_type = 0  # -1: Short, 0: Flat, 1: Long
        self.entry_price = None  # Price at which current position was entered
        self.asset_value = 0.0   # Curr mkt val of held shares (if long)
        # Portfolio value calculation depends on pos type
        self.portfolio_value = self.balance
        self.max_portfolio_value = self.portfolio_value

        # <<< ADDED: Initialize last valid price >>>
        # Try to get the first valid price, fallback to 0 if data starts
        # with NaN/inf
        first_valid_price = self.data['close'].iloc[self.current_step]
        if not np.isfinite(first_valid_price):
            logger.warning(
                f"Initial price at step {self.current_step} is non-finite "
                f"({first_valid_price}). Searching for first valid price..."
            )
            valid_prices = self.data['close'].iloc[self.current_step:].dropna()
            first_valid_price = (
                valid_prices.iloc[0] if not valid_prices.empty else 0.0
            )
            if not np.isfinite(first_valid_price):
                logger.error(
                    "Could not find any valid starting price in data!"
                )
                first_valid_price = 0.0  # Ultimate fallback
        self.last_valid_price = first_valid_price
        # <<< END ADDED >>>

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
        self.consecutive_holds = 0  # Track consecutive holds while FLAT
        # self.consecutive_buys = 0 # Removed
        # self.consecutive_sells = 0 # Removed
        self.last_action = 0.0  # Init last action as float for continuous
        self.exploration_bonus_value = self.exploration_start
        self.failed_trades = 0  # General counter for failed actions
        # Add cumulative reward component trackers
        self.cumulative_rewards = {
            'portfolio_change': 0.0,
            'drawdown_penalty': 0.0,
            'sharpe_reward': 0.0,
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
        info = self._get_info(self.last_valid_price)

        return observation, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to take (0: Hold, 1: Go Long, 2: Go Short, 3: Close)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.episode_step += 1
        prev_portfolio_value = self.portfolio_value
        prev_position_type = self.position_type

        # --- Get and Validate Current Price ---
        current_price = self.data['close'].iloc[self.current_step]
        if not np.isfinite(current_price):
            logger.warning(
                f"Step {self.current_step}: Non-finite current price "
                f"({current_price}) detected. Using last valid price: "
                f"{self.last_valid_price}"
            )
            validated_price = self.last_valid_price
            # Do NOT update last_valid_price if current is invalid
        else:
            validated_price = current_price
            # Update last valid price if current one is good
            self.last_valid_price = validated_price
        # --- End Price Validation ---

        # Take action using the validated price
        interpreted_action_code = self._take_action(
            action[0], validated_price
        )
        self.last_action = interpreted_action_code

        # Update portfolio value using the validated price
        self._update_portfolio_value(validated_price)

        # Calculate reward using the validated price
        reward = self._calculate_reward(
            interpreted_action_code,
            prev_portfolio_value,
            prev_position_type,
            validated_price  # Pass validated price
        )
        self.rewards.append(reward)
        self.portfolio_values.append(self.portfolio_value)

        # --- Update consecutive holds (only when flat) ---
        # Use interpreted action code for hold logic
        if self.position_type == 0 and interpreted_action_code == 0:
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0  # Reset if not flat or didn't hold flat

        # --- Now move to the next time step (t+1) ---
        self.current_step += 1
        self.episode_step += 1

        # --- Check episode end conditions based on step t+1 ---
        is_end_of_data = self.current_step >= len(self.data) - 1
        is_max_steps_reached = (self.max_steps is not None and
                                self.episode_step >= self.max_steps)

        # Determine termination/truncation
        # Terminate if drawdown > 50% OR end of data reached
        terminated = False
        # Truncate ONLY if max steps reached AND not already terminated
        truncated = is_max_steps_reached and not terminated
        # ---------------------------------------------------

        # --- Log Termination/Truncation Reason ---
        termination_reason = "None"
        if terminated:
            termination_reason = "End of Data"
        elif truncated:
            termination_reason = f"Max Steps Reached ({self.max_steps})"

        # --- EPISODE END CHECK --- #
        episode_ended = terminated or truncated
        if episode_ended:
            # Ensure portfolio is updated with the final price if not end of data # noqa E501
            if not is_end_of_data:
                # Pass the last known valid price
                self._update_portfolio_value(self.last_valid_price)

            # If episode ends with an open position, close it automatically
            # to get final PnL reflected in the balance.
            # This simulates closing out at the end of the test period.
            if self.position_type != 0:
                logger.info(
                    f"[EPISODE END] Auto-closing "
                    f"{('Long' if self.position_type == 1 else 'Short')}"
                    f" position at step {self.current_step-1}."
                )
                # Close using price of the *last valid step*
                self._close_position(
                    self.data['close'].iloc[self.current_step - 1]
                )
                # Update PV one last time after closing, using last valid price
                self._update_portfolio_value(self.last_valid_price)

            # Get info *after* potential auto-close
            final_info = self._get_info(self.last_valid_price)

            # --- Format Episode End Log as Markdown Table --- #
            log_table = "| Metric              | Value     |\n"
            log_table += "| :------------------ | :-------- |\n"
            log_table += f"| End Step            | {self.current_step-1:<9} |\n"
            log_table += f"| Episode Steps       | {self.episode_step:<9} |\n"
            log_table += f"| Terminated          | {terminated!s:<9} |\n"
            log_table += f"| Truncated           | {truncated!s:<9} |\n"
            log_table += f"| Reason              | {termination_reason:<9} |\n"
            log_table += f"| Final Port. Value   | {self.portfolio_value:<9.2f} |\n"
            profit = self.portfolio_value - self.initial_balance
            log_table += f"| Profit              | {profit:<9.2f} |\n"
            ret_pct = final_info.get('episode_return', 0.0) * 100
            log_table += f"| Return (%)          | {ret_pct:<9.2f} |\n"
            sharpe_ep = final_info.get('sharpe_ratio_episode', 0.0)
            log_table += f"| Sharpe (Ep)       | {sharpe_ep:<9.4f} |\n"
            calmar_ep = final_info.get('calmar_ratio', 0.0)
            log_table += f"| Calmar              | {calmar_ep:<9.4f} |\n"
            sortino_ep = final_info.get('sortino_ratio', 0.0)
            log_table += f"| Sortino             | {sortino_ep:<9.4f} |\n"
            max_dd_pct = self.max_drawdown * 100
            log_table += f"| Max Drawdown (%)    | {max_dd_pct:<9.2f} |\n"
            log_table += f"| Total Trades        | {self.total_trades:<9} |\n"
            log_table += f"| Longs               | {self.total_longs:<9} |\n"
            log_table += f"| Shorts              | {self.total_shorts:<9} |\n"
            log_table += f"| Closes              | {self.total_closes:<9} |\n"
            logger.info(f"[EPISODE END SUMMARY]\n{log_table}")
            # --- End Log Table ---

        # Decay exploration bonus for the next step
        if (self.exploration_decay_rate > 0 and
                self.exploration_bonus_value > self.exploration_end):
            self.exploration_bonus_value -= self.exploration_decay_rate

        # --- PRE-OBSERVATION NAN CHECK ---
        if not np.isfinite(self.portfolio_value):
             logger.error(
                 f"[NAN_CHECK] NaN portfolio_value ({self.portfolio_value}) "
                 f"BEFORE observation calculation at step {self.current_step}."
             )
        if not np.isfinite(self.balance):
             logger.error(
                 f"[NAN_CHECK] NaN balance ({self.balance}) BEFORE "
                 f"observation calculation at step {self.current_step}."
             )
        # -------------------------------

        # Get new observation and info for the *next* step
        # Based on self.current_step (t+1)
        observation = self._get_observation()
        # Based on self.current_step (t+1)
        info = self._get_info(self.last_valid_price)

        # Gymnasium expects 5 return values
        return observation, reward, terminated, truncated, info

    def _take_action(self, action: float, current_price: float):
        """
        Interpret the agent's action and execute the corresponding trade.

        Args:
            action: Continuous action value from the agent (-1 to 1).
            current_price: The validated current price for calculations.

        Returns:
            Integer action code: 0=Hold, 1=Buy/Cover, -1=Sell/Short, 3=Close
        """
        # --- Action Interpretation Thresholds ---
        # Define thresholds to map continuous action to discrete decisions
        # Ex: [-1.0, -0.5): Sell/Short | [-0.5, 0.5]: Hold | (0.5, 1.0]: Buy/Cover
        sell_short_threshold = -0.5
        buy_cover_threshold = 0.5
        # --- End Thresholds ---

        interpreted_action_code = 0  # Default: Hold
        # fee_paid = 0.0 # REMOVED
        trade_successful = True # Flag to track if intended action succeeded

        # Price is passed in and assumed validated by step method
        if not np.isfinite(current_price) or current_price <= ZERO_THRESHOLD:
             logger.error(
                 f"Step {self.current_step}: Invalid price ({current_price:.2f})"
                 f" received in _take_action. Cannot execute trade."
             )
             self.failed_trades += 1
             # return 0, 0.0 # REMOVED fee return
             return 0 # Return Hold

        # --- Decision Logic ---
        if action < sell_short_threshold:
            # Action: Sell/Short
            interpreted_action_code = -1 # Intention: Go Short
            if self.position_type == 0:  # Can only go short if flat
                short_value_target = self.balance * self.max_position
                if (short_value_target > ZERO_THRESHOLD and
                        current_price > ZERO_THRESHOLD):
                    shares_to_short = short_value_target / current_price
                    if shares_to_short > ZERO_THRESHOLD:
                        proceeds = shares_to_short * current_price
                        # fee = proceeds * self.transaction_fee # REMOVED
                        # self.balance += (proceeds - fee) # REMOVED fee
                        self.balance += proceeds  # Add net proceeds
                        self.shares_held = shares_to_short
                        self.position_type = -1
                        self.entry_price = current_price
                        self.total_trades += 1
                        self.total_shorts += 1
                        trade_successful = True
                        # fee_paid = fee # REMOVED
                        trade_info = {
                            'step': self.current_step, 'type': 'short_entry',
                            'price': current_price, 'shares': self.shares_held,
                            'proceeds': proceeds,  # 'fee': fee, # REMOVED
                            'balance_after': self.balance
                        }
                        self.trades.append(trade_info)
                        logger.debug(
                            f"Step {self.current_step}: [Act: {action:.2f} < "
                            f"{sell_short_threshold}] -> Entered SHORT "
                            f"{self.shares_held:.6f} @ {current_price:.2f} "
                            f"(Proceeds: {proceeds:.2f}) -> "  # Removed Fee log
                            f"Bal: {self.balance:.2f}"
                        )
                    else:
                        self.failed_trades += 1
                        logger.debug(
                            f"Step {self.current_step}: [Act: {action:.2f}] "
                            f"Attempted Short, but calculated shares "
                            f"{shares_to_short:.8f} <= threshold."
                        )
                else:
                    self.failed_trades += 1
                    logger.debug(
                        f"Step {self.current_step}: [Act: {action:.2f}] "
                        f"Attempted Short, but insufficient target value "
                        f"({short_value_target:.2f}) or zero price."
                    )
            else:  # Already in a position, treat as Hold
                interpreted_action_code = 0 # Override intention: Hold
                logger.debug(
                    f"Step {self.current_step}: [Act: {action:.2f}] Wanted "
                    f"Short, but already in "
                    f"{'Long' if self.position_type == 1 else 'Short'}. "
                    f"Holding."
                )

        elif action > buy_cover_threshold:
            # Action: Buy/Cover
            interpreted_action_code = 1  # Intention: Go Long
            if self.position_type == 0:  # Can only go long if flat
                invest_amount = self.balance * self.max_position
                if (invest_amount > ZERO_THRESHOLD and
                        current_price > ZERO_THRESHOLD):
                    shares_to_buy = invest_amount / current_price  # Simplified

                    if shares_to_buy > ZERO_THRESHOLD:
                        cost = shares_to_buy * current_price
                        self.balance -= cost
                        self.balance = max(0, self.balance)  # Ensure >= 0
                        self.shares_held = shares_to_buy
                        self.position_type = 1
                        self.entry_price = current_price
                        self.total_trades += 1
                        self.total_longs += 1
                        trade_successful = True
                        trade_info = {
                            'step': self.current_step, 'type': 'long_entry',
                            'price': current_price, 'shares': self.shares_held,
                            'cost': cost,
                            'balance_after': self.balance
                        }
                        self.trades.append(trade_info)
                        logger.debug(
                            f"Step {self.current_step}: [Act: {action:.2f} > "
                            f"{buy_cover_threshold}] -> Entered LONG "
                            f"{self.shares_held:.6f} @ {current_price:.2f} "
                            f"(Cost: {cost:.2f}) -> "
                            f"Bal: {self.balance:.2f}"
                        )
                    else:
                        self.failed_trades += 1
                        logger.debug(
                            f"Step {self.current_step}: [Act: {action:.2f}] "
                            f"Attempted Long, but calculated shares "
                            f"{shares_to_buy:.8f} <= threshold."
                        )
                else:
                    self.failed_trades += 1
                    logger.debug(
                        f"Step {self.current_step}: [Act: {action:.2f}] "
                        f"Attempted Long, but insufficient balance "
                        f"({self.balance:.2f}) or zero price."
                    )
            else:  # Already in a position, treat as Hold
                interpreted_action_code = 0  # Override intention: Hold
                logger.debug(
                    f"Step {self.current_step}: [Act: {action:.2f}] Wanted "
                    f"Long, but already in "
                    f"{'Long' if self.position_type == 1 else 'Short'}. "
                    f"Holding."
                )

        elif abs(action) < 0.1:  # Zone around zero: Try Close or Hold Flat
            if self.position_type != 0:  # If in position, try Close
                interpreted_action_code = 3  # Intention: Close
                # Unpack the returned value (only PnL now)
                profit_loss = self._close_position(current_price)
                if profit_loss is not None:  # Check if close was successful using PnL
                    trade_successful = True
                    self.total_closes += 1
                    logger.debug(
                        f"Step {self.current_step}: [Act: {action:.2f} in "
                        f"+/-0.1] -> Closed Position. "
                        # Use the unpacked profit_loss variable for formatting
                        f"PnL: {profit_loss:.2f} -> Bal: {self.balance:.2f}"
                    )
                else:  # Close failed (e.g., insufficient funds for short close)
                    self.failed_trades += 1
                    interpreted_action_code = 0  # Treat failed close as Hold
                    logger.debug(
                        f"Step {self.current_step}: [Act: {action:.2f}] "
                        f"Attempted Close, but failed. Holding position."
                    )
            else:  # Already flat, Hold Flat
                interpreted_action_code = 0  # Intention: Hold Flat
                self.total_holds += 1
                logger.debug(
                    f"Step {self.current_step}: [Act: {action:.2f} in +/-0.1]"
                    f" -> Held Flat."
                )

        else:  # Dead zone between close and trade thresholds: Hold Position
            interpreted_action_code = 0  # Intention: Hold
            pos_desc = ('Long' if self.position_type == 1 else
                        'Short' if self.position_type == -1 else 'Flat')
            logger.debug(
                f"Step {self.current_step}: [Act: {action:.2f} in dead zone] "
                f"-> Holding {pos_desc} Position."
            )

        return interpreted_action_code

    def _close_position(self, closing_price: float):
        """
        Close the current position (Long or Short).

        Args:
            closing_price: The validated price at which to close.

        Returns:
            Profit/Loss of the closed trade, or None if close fails.
        """
        profit_loss = None

        # --- Price Validation ---
        # Price passed should already be validated by the 'step' method
        # But we add a check here as a failsafe
        if not np.isfinite(closing_price) or closing_price <= ZERO_THRESHOLD:
             logger.error(
                 f"Step {self.current_step}: Invalid closing price "
                 f"({closing_price:.2f}) passed to _close_position. "
                 f"Cannot close."
             )
             return None
        # --- End Validation ---

        if self.position_type == 1:  # Closing Long
            trade_type = 'long_exit'
            sell_value = self.shares_held * closing_price
            self.balance += sell_value
            profit_loss = (closing_price - self.entry_price) * self.shares_held
            logger.debug(
                f"Closing Long: {self.shares_held:.6f} sold @ "
            )

        elif self.position_type == -1:  # Closing a Short position (Buy)
            trade_type = 'short_exit'
            buy_cost = self.shares_held * closing_price
            total_cost = buy_cost

            # Check if we have enough balance to buy back the short position
            # Allow for float errors
            if self.balance < total_cost - ZERO_THRESHOLD:
                logger.warning(
                    f"Step {self.current_step}: Insufficient balance "
                    f"({self.balance:.2f}) to close short position requiring "
                    f"{total_cost:.2f}. Cannot close."
                )
                return None  # Indicate failure

            self.balance -= buy_cost
            profit_loss = (self.entry_price - closing_price) * self.shares_held
            logger.debug(
                f"Closing Short: {self.shares_held:.6f} bought @ "
            )

        # Record the trade
        trade_info = {
            'step': self.current_step, 'type': trade_type,
            'price': closing_price, 'shares': self.shares_held,
            'entry_price': self.entry_price,
            'pnl': profit_loss,
            'balance_after': self.balance
        }
        self.trades.append(trade_info)

        # Update state
        self.shares_held = 0.0
        self.position_type = 0
        self.entry_price = None

        return profit_loss

    def _update_portfolio_value(self, current_price: float):
        """
        Update the total portfolio value based on the current price and
        position.

        Args:
            current_price: The validated current price.
        """
        # --- Price Validation ---
        # Price passed should already be validated by the 'step' method
        # Add a check here as a failsafe
        if not np.isfinite(current_price):
            logger.error(
                f"Step {self.current_step}: Invalid price ({current_price:.2f}) "
                f"received in _update_portfolio_value. "
                f"Using last valid: {self.last_valid_price}"
            )
            current_price = self.last_valid_price  # Use last valid as fallback
            if not np.isfinite(current_price):  # If last valid is also bad
                logger.error(
                    f"Step {self.current_step}: Last valid price "
                    f"({self.last_valid_price}) also invalid! "
                    f"Portfolio calculation will likely fail."
                )
                current_price = 0.0  # Prevent crash, but PV will be wrong
        # --- End Validation ---

        self.asset_value = 0.0  # Reset asset value

        if self.position_type == 1:  # Long Position
            self.asset_value = self.shares_held * current_price
            self.portfolio_value = self.balance + self.asset_value
        elif self.position_type == -1:  # Short Position
            # Portfolio value = Cash Balance + Unrealized PnL from Short
            # Unrealized PnL = (Entry Price - Current Price) * Shares Held
            # Note: Fee to close is not included here
            unrealized_pnl = 0
            if self.entry_price is not None:
                unrealized_pnl = (self.entry_price - current_price) * self.shares_held # noqa E501
            self.portfolio_value = self.balance + unrealized_pnl
            # Asset value for shorts could be considered negative liability
            self.asset_value = -self.shares_held * current_price
        else:  # Flat Position
            self.portfolio_value = self.balance
            self.asset_value = 0.0  # Explicitly zero

        # --- Check for non-finite values immediately after calculation --- #
        if not np.isfinite(self.portfolio_value) or not np.isfinite(self.balance):
            logger.error(
                f"_update_portfolio_value: NON-FINITE PORTFOLIO VALUE "
                f"OR BALANCE DETECTED! (Step {self.current_step})\n"
                f" PosType={self.position_type}, Price={current_price}, "
                f"Shares={self.shares_held}, Bal={self.balance}, "
                f"Entry={self.entry_price}\n"
                f" -> AssetVal={self.asset_value}, "
                f"PortVal={self.portfolio_value}"
            )
            # Attempt to recover by setting to a large negative value
            # to penalize heavily or use last known good value.
            # Using initial balance might be too forgiving. Use 0 for now.
            self.portfolio_value = 0.0  # Emergency reset?
            self.balance = 0.0  # Reset balance too if it became non-finite

    def _calculate_reward(self, interpreted_action_code, prev_portfolio_value,
                          prev_position_type, current_price):
        """
        Calculate the reward for the current step.

        Args:
            interpreted_action_code: 0=Hold, 1=Buy/Cover, -1=Sell/Short
            prev_portfolio_value: Portfolio value from the previous step
            prev_position_type: Position type from the previous step
            current_price: Validated current price

        Returns:
            Reward for the current step
        """
        total_reward = 0.0
        reward_components = {} # Store individual component values

        # Price is passed in and assumed validated by step method
        # Add check anyway as failsafe for benchmark calc
        if not np.isfinite(current_price):
            logger.warning(
                f"Step {self.current_step}: Invalid price ({current_price:.2f}) "
                f"in _calculate_reward. Benchmark reward might be inaccurate."
            )
            # Proceed, components using price will likely be 0

        # --- Reward Component 1: Portfolio Value Change ---
        portfolio_change_pct = 0.0
        if (abs(prev_portfolio_value) > ZERO_THRESHOLD and
                np.isfinite(prev_portfolio_value) and
                np.isfinite(self.portfolio_value)):
            portfolio_change_pct = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        elif not np.isfinite(prev_portfolio_value) or not np.isfinite(self.portfolio_value):
            logger.warning(
                f"Step {self.current_step}: Non-finite portfolio value "
                f"detected in reward calc (prev={prev_portfolio_value}, "
                f"curr={self.portfolio_value}). Setting change to -1."
            )
            portfolio_change_pct = -1.0 # Penalize heavily for non-finite
        # Assign the component AFTER calculating the percentage
        reward_components['portfolio_change'] = (
            portfolio_change_pct * self.portfolio_change_weight
        )

        # --- Reward Component 2: Drawdown Penalty ---
        current_drawdown = 0.0
        if self.max_portfolio_value > ZERO_THRESHOLD:
            current_drawdown = max(
                0.0,
                (self.max_portfolio_value - self.portfolio_value) /
                self.max_portfolio_value
            )
        reward_components['drawdown_penalty'] = (
            -current_drawdown * self.drawdown_penalty_weight
        )

        # --- Reward Component 3: Sharpe Ratio Reward ---
        sharpe_ratio_rolling = 0.0
        if len(self.step_returns) >= self.sharpe_window:
            window_returns = np.array(self.step_returns[-self.sharpe_window:])
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            if std_return > ZERO_THRESHOLD:
                sharpe_ratio_rolling = mean_return / std_return
        reward_components['sharpe_reward'] = (
            sharpe_ratio_rolling * self.sharpe_reward_weight
        )

        # --- Reward Component 5: Benchmark Comparison ---
        benchmark_reward = 0.0
        # Get benchmark price change (e.g., previous close to current close)
        # Use last_valid_price for previous step if current_step > 0
        prev_step_index = self.current_step - 1
        if prev_step_index >= 0:
            prev_price = self.data['close'].iloc[prev_step_index]
            # Use last_valid_price if the historical price is invalid
            if not np.isfinite(prev_price):
                logger.warning(
                    f"Step {self.current_step}: Invalid historical price at "
                    f"{prev_step_index} for benchmark. Using last valid: "
                    f"{self.last_valid_price}"
                )
                prev_price = self.last_valid_price  # Fallback needed?
        else:
            prev_price = self.episode_start_price  # Use validated start price

        # Check validity of prices for benchmark calculation
        if (np.isfinite(prev_price) and prev_price > ZERO_THRESHOLD and
                np.isfinite(current_price)):
            benchmark_return = (current_price / prev_price) - 1.0
            # Access portfolio_change component AFTER it has been assigned
            # agent_step_return = reward_components['portfolio_change'] # Unused
            benchmark_reward = benchmark_return * self.benchmark_reward_weight

        reward_components['benchmark_reward'] = benchmark_reward

        # --- Reward Component 6: Direct Trade Entry Penalty ---
        trade_penalty = 0.0
        if interpreted_action_code in [1, 2]:
            trade_penalty = -1.0  # Base penalty for entering a trade
        calculated_trade_penalty = trade_penalty * self.trade_penalty_weight
        reward_components['trade_penalty'] = calculated_trade_penalty

        # --- Reward Component 7: Idle Penalty (Only when Flat) ---
        idle_penalty = 0.0
        if (self.position_type == 0 and interpreted_action_code == 0 and
                self.consecutive_holds > self.idle_threshold):
            idle_penalty = -1.0  # Base penalty for being idle too long
        reward_components['idle_penalty'] = idle_penalty * self.idle_penalty_weight

        # --- Reward Component 8: Profit/Closing Bonus (Only when Closing) ---
        profit_bonus = 0.0
        # Check if action was close, prev pos wasn't flat, current pos is flat
        if (interpreted_action_code == 3 and prev_position_type != 0 and
                self.position_type == 0):
            # PnL was calculated in _close_position
            # Find the PnL from the last trade entry in self.trades
            last_trade = self.trades[-1] if self.trades else None
            if last_trade and last_trade['type'] in ['long_exit', 'short_exit']:
                trade_pnl = last_trade.get('pnl', 0.0)
                # Normalize PnL by entry value?
                entry_value = 0
                if last_trade['type'] == 'long_exit':
                    entry_value = last_trade.get('shares', 0) * last_trade.get('entry_price', 0)  # noqa E501
                elif last_trade['type'] == 'short_exit':
                    # Value at entry for short based on entry price
                    entry_value = last_trade.get('shares', 0) * last_trade.get('entry_price', 0)  # noqa E501

                if entry_value > ZERO_THRESHOLD:
                    pnl_pct = trade_pnl / entry_value
                    if pnl_pct > 0:  # Only reward profitable closes
                        # Scale bonus by profit percentage
                        profit_bonus = pnl_pct * 2  # Adjust multiplier
                    # else: Add penalty for losing closes? Optional.
                    #     profit_bonus = pnl_pct * some_penalty_multiplier

        reward_components['profit_bonus'] = (
            profit_bonus * self.profit_bonus_weight
        )

        # --- Reward Component 9: Exploration Bonus (Optional) ---
        # ... (kept commented out)

        # Sum all active reward components
        components_to_sum = [
            k for k in reward_components
            if k not in ['raw_total', 'total_reward']
        ]
        raw_total = sum(reward_components[key] for key in components_to_sum)
        reward_components['raw_total'] = raw_total

        # Apply reward scaling for the final reward
        total_reward = raw_total * self.reward_scaling

        # --- Update cumulative reward sums ---
        for key in self.cumulative_rewards:
            if key in reward_components:
                # Ensure value is finite before adding
                comp_value = reward_components[key]
                if np.isfinite(comp_value):
                    self.cumulative_rewards[key] += comp_value
                else:
                    logger.warning(
                        f"Non-finite reward component '{key}': {comp_value}. "
                        f"Skipping cumulative update."
                    )

        # --- Logging ---
        # Log only if reward is non-trivial or components are non-zero
        if (abs(total_reward) > 1e-6 or
                any(abs(v) > 1e-6 for k, v in reward_components.items()
                    if k not in ['raw_total', 'total_reward'])):
            component_str = ', '.join([
                f"{k}: {v:.4f}" for k, v in reward_components.items()
                if k not in ['raw_total', 'total_reward'] and abs(v) > 1e-6
            ])
            logger.debug(
                f"Step {self.current_step} (EpStep {self.episode_step}) "
                f"ContAct {self.last_action:.2f} "
                f"(Interpreted: {interpreted_action_code}) -> "
                f"Reward: {total_reward:.4f}"
                f" | Components: {component_str}"
            )

        return total_reward

    def _get_observation(self):
        """
        Get the current observation, including historical features,
        position type, and normalized entry price.
        Uses pre-computed NumPy/CuPy array for features.

        Returns:
            Numpy array containing the observation (on CPU)
        """
        # Check if current_step is valid
        if self.current_step < 0 or self.current_step >= len(self.data):
            logger.error(
                f"Invalid current_step ({self.current_step}) in "
                f"_get_observation. Data length: {len(self.data)}"
            )
            # Fallback: return zeros matching shape
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        safe_step = self.current_step

        # Get historical feature data indices
        end_idx = safe_step
        start_idx = max(0, end_idx - self.sequence_length + 1)
        actual_len = end_idx - start_idx + 1

        # --- Optimized Feature Extraction using NumPy or CuPy ---
        xp = np # Default to numpy
        feature_source = self.feature_array

        if CUPY_AVAILABLE and self.feature_array_gpu is not None:
            xp = cp
            feature_source = self.feature_array_gpu
            # logger.debug("Using CuPy for observation features") # Optional debug
        # else: logger.debug("Using NumPy for observation features") # Optional debug

        try:
            # Slice the feature array (NumPy or CuPy)
            historical_features_sliced = feature_source[start_idx:end_idx+1]

            # Handle padding if needed (at the start of the data)
            if actual_len < self.sequence_length:
                num_padding = self.sequence_length - actual_len
                # Get the first row of the slice for padding
                padding_row = historical_features_sliced[0:1]
                # Repeat the first row `num_padding` times
                padding = xp.repeat(padding_row, num_padding, axis=0)
                # Concatenate padding and the slice
                historical_features_padded = xp.concatenate((padding, historical_features_sliced), axis=0)
            else:
                historical_features_padded = historical_features_sliced

            # Flatten the features (using NumPy/CuPy flatten)
            feature_vector = historical_features_padded.flatten()

        except IndexError as e:
             logger.error(f"Error slicing feature array at step {safe_step} (Indices {start_idx}:{end_idx+1}): {e}")
             # Fallback to zeros if slicing fails
             feature_vector = xp.zeros(len(self.features) * self.sequence_length, dtype=np.float32)
        except Exception as e:
             logger.error(f"Unexpected error during feature extraction (step {safe_step}): {e}", exc_info=True)
             feature_vector = xp.zeros(len(self.features) * self.sequence_length, dtype=np.float32)
        # --- End Optimized Feature Extraction ---

        # --- Position Information (calculated on CPU for simplicity) ---
        position_type_feature = float(self.position_type)
        normalized_entry_price = 0.0
        current_price_obs = self.last_valid_price # Use validated price from step

        entry_price_valid = (self.entry_price is not None and
                             np.isfinite(self.entry_price) and
                             abs(self.entry_price) > ZERO_THRESHOLD)

        if self.position_type != 0 and entry_price_valid:
            if np.isfinite(current_price_obs) and abs(current_price_obs) > ZERO_THRESHOLD:
                normalized_entry_price = (current_price_obs / self.entry_price) - 1.0
                if not np.isfinite(normalized_entry_price):
                    logger.warning(
                        f"Step {self.current_step}: normalized_entry_price became non-finite. Setting to 0."
                    )
                    normalized_entry_price = 0.0
            # else: logger.warning(...) # Optional logging if current price is invalid
        # elif self.position_type != 0: logger.warning(...) # Optional logging if entry price invalid

        position_info = np.array([position_type_feature, normalized_entry_price], dtype=np.float32)
        # --- End Position Information ---

        # --- Combine features and position info --- 
        # If features are on GPU, move position_info to GPU for concatenation
        if xp == cp:
            try:
                position_info_gpu = cp.asarray(position_info)
                observation_gpu = cp.concatenate((feature_vector, position_info_gpu))
                # IMPORTANT: Convert back to NumPy CPU array for Stable Baselines
                observation = cp.asnumpy(observation_gpu)
            except Exception as e:
                 logger.error(f"Error during CuPy concatenation/conversion (step {safe_step}): {e}. Falling back.", exc_info=True)
                 # Fallback: Move feature_vector to CPU and concatenate there
                 feature_vector_cpu = cp.asnumpy(feature_vector)
                 observation = np.concatenate((feature_vector_cpu, position_info))                 
        else:
            # If using NumPy, directly concatenate
            observation = np.concatenate((feature_vector, position_info))
        # --- End Combination ---

        # Final check for non-finite values (should be less likely now)
        if not np.all(np.isfinite(observation)):
            logger.error(
                f"Step {self.current_step}: Non-finite values detected in final observation AFTER processing! Clipping."
            )
            observation = np.nan_to_num(observation, nan=0.0, posinf=1e9, neginf=-1e9)

        return observation.astype(np.float32) # Ensure correct dtype

    def _get_info(self, current_price: float):
        """
        Get additional information about the environment state.

        Args:
            current_price: The validated current price.

        Returns:
            Dictionary containing environment info
        """
        # Use safe_step in case called at the very end
        safe_step = min(self.current_step, len(self.data) - 1)
        # current_price is passed in, already validated
        if not np.isfinite(current_price):
            logger.warning(
                f"Step {self.current_step}: Invalid price ({current_price:.2f}) "
                f"received in _get_info. "
                f"Using last valid: {self.last_valid_price}"
            )
            current_price = self.last_valid_price  # Use last valid as fallback

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
            'asset_value': self.asset_value,      # Curr val (long) / neg liab (short)
            'portfolio_value': self.portfolio_value,
            'initial_balance': self.initial_balance,
            # Cash ratio: balance / portfolio_value when PV > 0
            'cash_ratio': self.balance / self.portfolio_value
            if self.portfolio_value > ZERO_THRESHOLD else 1.0,
            'total_trades': self.total_trades,    # Entries (long + short)
            'total_longs': self.total_longs,
            'total_shorts': self.total_shorts,
            'total_closes': self.total_closes,
            'total_holds': self.total_holds,      # Holds while flat
            'max_drawdown': self.max_drawdown,
            'failed_trades': self.failed_trades,  # Failed action attempts
            'last_action': self.last_action,
            'consecutive_holds': self.consecutive_holds,  # Holds while flat
            'exploration_bonus_value': self.exploration_bonus_value,
        }

        # --- Calculate overall episode returns and metrics --- #
        if len(self.portfolio_values) > 1:
            # Use robust calculation for episode return
            initial_val = self.initial_balance
            current_val = self.portfolio_value
            episode_return = 0.0
            if (np.isfinite(initial_val) and np.isfinite(current_val) and
                    abs(initial_val) > ZERO_THRESHOLD):
                try:
                    calculated_return = (current_val - initial_val)
                    if abs(initial_val) > ZERO_THRESHOLD:
                        calculated_return /= initial_val
                    else:
                        calculated_return = 0.0 # Avoid division by zero

                    if np.isfinite(calculated_return):
                        episode_return = calculated_return
                    else:
                        logger.warning(
                            f"_get_info: Calculated episode_return non-finite "
                            f"({calculated_return}). Forcing to 0."
                        )
                except Exception as e:
                    logger.error(
                        f"_get_info: Error calculating episode_return: {e}",
                        exc_info=True
                    )
            elif abs(initial_val) <= ZERO_THRESHOLD:
                logger.warning(
                    f"_get_info: Initial balance near zero ({initial_val}). "
                    f"Return set to 0."
                )
            elif not np.isfinite(initial_val) or not np.isfinite(current_val):
                logger.warning(
                    f"_get_info: Non-finite values for return calc "
                    f"(Initial: {initial_val}, Current: {current_val}). "
                    f"Return set to 0."
                )

            info['episode_return'] = episode_return

            # Calculate overall Sharpe from portfolio_values
            # Ensure portfolio_values contains finite numbers
            finite_pv = [pv for pv in self.portfolio_values if np.isfinite(pv)]
            if len(finite_pv) > 1:
                pv_array = np.array(finite_pv)
                # --- REVISED RETURN CALCULATION ---
                # Calculate returns: (p[t] - p[t-1]) / p[t-1]
                # Only where the denominator p[t-1] is valid (not near zero)
                valid_denom_mask = np.abs(pv_array[:-1]) > ZERO_THRESHOLD

                if np.any(valid_denom_mask):
                    # Select corresponding slices for num and denom
                    p_t = pv_array[1:][valid_denom_mask]
                    p_t_minus_1 = pv_array[:-1][valid_denom_mask]

                    # Ensure we have values after masking
                    if (p_t.size > 0 and p_t_minus_1.size > 0 and
                            p_t.shape == p_t_minus_1.shape):
                        # Calculate differences
                        diffs = p_t - p_t_minus_1

                        # Denom checked, but add safety clamp
                        safe_denominators = np.where(
                            np.abs(p_t_minus_1) > 1e-9, p_t_minus_1, 1e-9
                        )

                        portfolio_returns = diffs / safe_denominators

                        # --- Sharpe Calculation ---
                        if len(portfolio_returns) > 1:
                            mean_return = np.mean(portfolio_returns)
                            std_return = np.std(portfolio_returns)
                            if std_return > ZERO_THRESHOLD:
                                info['sharpe_ratio_episode'] = mean_return / std_return # noqa E501
                            else:
                                info['sharpe_ratio_episode'] = 0.0
                        else:
                            info['sharpe_ratio_episode'] = 0.0
                        # --- End Sharpe Calculation ---

                    else:
                        logger.warning(
                            f"_get_info: Shape mismatch or empty array after "
                            f"masking for return calc. p_t: {p_t.shape}, "
                            f"p_t-1: {p_t_minus_1.shape}. Setting Sharpe=0."
                        )
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
            # Ensure rolling is also 0 if no history
            info['sharpe_ratio_rolling'] = 0.0

        # Benchmark (Buy & Hold) Performance
        benchmark_portfolio_value = self.initial_balance  # Default
        benchmark_return = 0.0
        if abs(self.episode_start_price) > ZERO_THRESHOLD:
            initial_units = self.initial_balance / self.episode_start_price
            benchmark_portfolio_value = initial_units * current_price
            if abs(self.initial_balance) > ZERO_THRESHOLD:
                benchmark_return = (benchmark_portfolio_value - self.initial_balance) / self.initial_balance  # noqa E501

        info['benchmark_portfolio_value'] = benchmark_portfolio_value
        info['benchmark_return'] = benchmark_return

        # Add cumulative reward components to info
        info['cumulative_reward_components'] = self.cumulative_rewards.copy()

        # Calculate and add Calmar/Sortino Ratios at episode end
        info['calmar_ratio'] = 0.0
        info['sortino_ratio'] = 0.0
        # Use finite portfolio values for metrics
        finite_pv_metrics = [pv for pv in self.portfolio_values
                             if np.isfinite(pv)]
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
                    logger.warning(
                        "_get_info: Non-finite values remain in "
                        "portfolio_values_np for metrics calculation."
                    )

            except Exception as e:
                logger.error(
                    f"_get_info: Error calculating trading metrics: {e}",
                    exc_info=True
                )

        return info