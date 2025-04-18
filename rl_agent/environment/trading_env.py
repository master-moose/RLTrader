"""
Trading Environment for reinforcement learning.

This module implements a cryptocurrency trading environment
compatible with the OpenAI Gym interface.
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
ZERO_THRESHOLD = 1e-9 # noqa E221

class TradingEnvironment(Env):
    """
    A cryptocurrency trading environment for reinforcement learning.
    
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
        max_position: float = 1.0,
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
        consistency_penalty_weight: float = 0.2,
        idle_penalty_weight: float = 0.1,
        profit_bonus_weight: float = 0.5,
        # Add additional reward parameters
        sharpe_window: int = 20,
        consistency_threshold: int = 3,
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
            consistency_penalty_weight: Weight for trade consistency penalty
            idle_penalty_weight: Weight for idle position penalty
            profit_bonus_weight: Weight for profit bonus
            
            # Additional reward parameters
            sharpe_window: Window size for Sharpe ratio calculation
            consistency_threshold: Min consecutive actions before flip ok
            idle_threshold: Num consecutive holds before idle penalty
            trade_penalty_weight: Weight for trade penalty
        """ # noqa E501
        super(TradingEnvironment, self).__init__()
        
        # Store parameters
        self.data = data.copy()
        self.features = features or ["close", "volume", "open", "high", "low"]
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee  # Store the original target fee
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
        self.consistency_penalty_weight = consistency_penalty_weight
        self.idle_penalty_weight = idle_penalty_weight
        self.profit_bonus_weight = profit_bonus_weight
        
        # Store additional reward parameters
        self.sharpe_window = sharpe_window
        self.consistency_threshold = consistency_threshold
        self.idle_threshold = idle_threshold
        self.trade_penalty_weight = trade_penalty_weight
        
        # Data preprocessing
        for feature in self.features:
            if feature not in self.data.columns:
                raise ValueError(
                    f"Feature '{feature}' not found in data columns: "
                    f"{data.columns.tolist()}"
                )
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy
        
        # Observation space: features only, no account info
        feature_space_dim = len(self.features) * self.sequence_length
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_space_dim,),  # Only features, not account info
            dtype=np.float32
        )
        
        # Set maximum number of steps
        if max_steps is None:
            self.max_steps = len(self.data) - self.sequence_length - 1
        else:
            self.max_steps = min(max_steps,
                                 len(self.data) - self.sequence_length - 1)
        
        logger.debug(
            f"Created TradingEnvironment with {len(self.data)} data points, "
            f"{len(self.features)} features, and sequence length "
            f"{self.sequence_length}"
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
            self.current_step = np.random.randint(
                self.sequence_length, len(self.data) - self.max_steps
            )
            logger.debug(f"Starting from random position: {self.current_step}")
        else:
            self.current_step = self.sequence_length
            logger.debug(f"Starting from beginning: {self.current_step}")
        
        # Reset the account balance
        self.balance = self.initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.portfolio_value = self.balance + self.asset_value
        self.max_portfolio_value = self.portfolio_value
        
        # Reset trade history
        self.trades = [] # noqa E221
        self.buy_prices = [] # noqa E221
        self.sell_prices = [] # noqa E221
        self.portfolio_values = [self.portfolio_value] # noqa E221
        self.rewards = [] # noqa E221
        self.last_buy_price = None
        
        # Reset metrics
        self.total_trades = 0 # noqa E221
        self.total_buys = 0 # noqa E221
        self.total_sells = 0 # noqa E221
        self.total_holds = 0
        self.max_drawdown = 0.0  # Reset max drawdown

        # --- Initialize new state variables --- <<< MOVED BEFORE _get_info()
        self.step_returns = []  # For Sharpe ratio calculation
        self.episode_start_price = self.data['close'].iloc[self.current_step]
        self.consecutive_holds = 0
        self.consecutive_buys = 0 # Track consecutive buys for consistency penalty
        self.consecutive_sells = 0 # Track consecutive sells for consistency penalty
        self.last_action = 1 # Assume initial action is Hold (or None?) - Let's use 1 (Hold)
        self.exploration_bonus_value = self.exploration_start # Reset exploration bonus
        self.total_fees_paid = 0.0 # Reset total fees
        self.failed_buys = 0 # Initialize failed buy counter
        self.failed_sells = 0 # Initialize failed sell counter
        # --- Add cumulative reward component trackers ---
        self.cumulative_rewards = {
            'portfolio_change': 0.0,
            'drawdown_penalty': 0.0,
            'sharpe_reward': 0.0,
            'fee_penalty': 0.0,
            'benchmark_reward': 0.0, # Ensure all potential keys are present
            'consistency_penalty': 0.0, # Ensure all potential keys are present
            'idle_penalty': 0.0,
            'profit_bonus': 0.0,
            'exploration_bonus': 0.0, # Ensure all potential keys are present
            'trade_penalty': 0.0, # ADDED
        }
        # --- End cumulative reward component trackers ---

        # Steps taken within the current episode
        self.episode_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        # Now safe to call _get_info() as attributes are initialized
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: Sell, 1: Hold, 2: Buy)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """ # noqa E501
        # --- Log state at the very beginning of the step ---
        # logger.info(
        #     f"STEP ENTRY (Step {self.current_step}): Balance={self.balance:.2f}, "
        #     f"Shares={self.shares_held:.6f}"
        # ) # Silenced
        # --------------------------------------------------
        # Record previous portfolio value
        prev_portfolio_value = self.portfolio_value
        prev_fees_paid = self.total_fees_paid # Store fees before action
        
        # Execute the action
        self._take_action(action)

        # --- Update portfolio and check drawdown BEFORE incrementing step ---
        # Log state BEFORE update
        # logger.info(
        #     f"PRE-UPDATE (Step {self.current_step}): Price={self.data['close'].iloc[self.current_step]:.2f}, "
        #     f"Balance={self.balance:.2f}, Shares={self.shares_held:.6f}, PV={self.portfolio_value:.2f}, "
        #     f"MaxPV={self.max_portfolio_value:.2f}"
        # ) # Silenced
        # Update portfolio value using price at the current step (t)
        self._update_portfolio_value() # Uses self.current_step
        # Log state AFTER update
        # logger.info(
        #     f"POST-UPDATE (Step {self.current_step}): Price={self.data['close'].iloc[self.current_step]:.2f}, "
        #     f"Balance={self.balance:.2f}, Shares={self.shares_held:.6f}, AssetValue={self.asset_value:.2f}, "
        #     f"PV={self.portfolio_value:.2f}"
        # ) # Silenced

        # Update max portfolio value based on value at step t
        self.max_portfolio_value = max(self.max_portfolio_value, 
                                       self.portfolio_value)

        # Calculate drawdown based on value at step t
        current_drawdown = 0.0 # Use a different name to avoid confusion
        if self.max_portfolio_value > ZERO_THRESHOLD: # Use threshold
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) \
                               / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        # Log drawdown calculation
        # logger.info(
        #     f"DRAWDOWN CALC (Step {self.current_step}): PV={self.portfolio_value:.2f}, MaxPV={self.max_portfolio_value:.2f}, "
        #     f"Drawdown={current_drawdown:.4f}, MaxDrawdown={self.max_drawdown:.4f}"
        # ) # Silenced

        # Check for early stopping based on drawdown at step t
        drawdown_terminated = False # Flag specific to drawdown termination
        # Using 0.50 as the new termination threshold
        if self.max_drawdown > 0.50:
            drawdown_terminated = True
            # Log termination reason later in the step

        # --------------------------------------------------------------------

        # Store portfolio value history for step t
        self.portfolio_values.append(self.portfolio_value)

        # Calculate step return for Sharpe ratio (t vs t-1)
        if abs(prev_portfolio_value) > ZERO_THRESHOLD: # Avoid division by zero/near-zero
            step_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.step_returns.append(step_return)
        else:
            self.step_returns.append(0.0)

        # Calculate reward based on change from t-1 to t
        fee_paid_this_step = self.total_fees_paid - prev_fees_paid
        reward_info = self._calculate_reward(action, prev_portfolio_value, fee_paid_this_step)
        reward = reward_info['total_reward'] # Extract the final reward
        self.rewards.append(reward)

        # --- Now move to the next time step (t+1) ---
        self.current_step += 1
        self.episode_step += 1 # Increment episode step counter

        # --- Check episode end conditions based on step t+1 ---
        is_end_of_data = self.current_step >= len(self.data) - 1
        is_max_steps_reached = self.max_steps is not None and \
                               self.episode_step >= self.max_steps
        # Update terminated/truncated flags 
        terminated = drawdown_terminated or is_end_of_data # Terminate if drawdown OR end of data
        truncated = is_max_steps_reached and not terminated # Truncate if max steps reached AND not already terminated
        # ---------------------------------------------------

        # --- Log Termination/Truncation Reason ---
        termination_reason = "None"
        if terminated:
            if drawdown_terminated:
                termination_reason = f"Drawdown > 50% ({self.max_drawdown:.2%})"
            elif is_end_of_data:
                termination_reason = "End of Data"
        elif truncated:
            termination_reason = f"Max Steps Reached ({self.max_steps})"

        # --- EPISODE SUMMARY LOG ---
        if terminated or truncated:
            info = self._get_info()
            logger.info(
                (
                    f"[EPISODE END] Step {self.current_step} | EpStep {self.episode_step}\n"
                    f"Terminated: {terminated} | Truncated: {truncated} | Reason: {termination_reason}\n"
                    f"Final Portfolio Value: {self.portfolio_value:.2f}\n"
                    f"Profit: {self.portfolio_value - self.initial_balance:.2f}\n"
                    f"Episode Return: {info.get('episode_return', 0.0):.4f}\n"
                    f"Sharpe (episode): {info.get('sharpe_ratio_episode', 0.0):.4f}\n"
                    f"Sharpe (rolling): {info.get('sharpe_ratio_rolling', 0.0):.4f}\n"
                    f"Calmar: {info.get('calmar_ratio', 0.0):.4f}\n"
                    f"Sortino: {info.get('sortino_ratio', 0.0):.4f}\n"
                    f"Total Trades: {self.total_trades} | Buys: {self.total_buys} | Sells: {self.total_sells}\n"
                    f"Max Drawdown: {self.max_drawdown:.2%}"
                )
            )
        # --- End Log ---

        # Decay exploration bonus for the next step
        if self.exploration_decay_rate > 0 and self.exploration_bonus_value > self.exploration_end:
            self.exploration_bonus_value -= self.exploration_decay_rate # noqa E128
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        # Gymnasium expects 5 return values: obs, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info
    
    def _take_action(self, action):
        """
        Execute the trading action.
        
        Args:
            action: Action to take (0: Sell, 1: Hold, 2: Buy)
        """
        current_price = self.data['close'].iloc[self.current_step]
        fee_paid_this_step = 0.0 # Track fee for this specific action
        
        # Record action counts (basic) - Adjust Buy/Trade count logic below
        # self.total_trades += (action != 1)  # Count non-hold actions as trades
        if action == 0:
            self.total_sells += 1
        elif action == 1:
            self.total_holds += 1
        elif action == 2:
            # Defer Buy/Trade count increment until success confirmed
            pass # Handled below
        
        # Update consecutive action counts
        if action == self.last_action:
            if action == 0: # Sell
                self.consecutive_sells += 1
            elif action == 1: # Hold
                self.consecutive_holds += 1
            elif action == 2: # Buy
                self.consecutive_buys += 1
        else:
            # Reset counters when action changes
            if action == 0: # Sell
                self.consecutive_sells = 1
                self.consecutive_holds = 0
                self.consecutive_buys = 0
            elif action == 1: # Hold
                self.consecutive_holds = 1
                self.consecutive_sells = 0
                self.consecutive_buys = 0
            elif action == 2: # Buy
                self.consecutive_buys = 1
                self.consecutive_holds = 0
                self.consecutive_sells = 0

        # Update last action
        self.last_action = action

        if action == 0:  # Sell
            if self.shares_held > ZERO_THRESHOLD: # Check if shares held > 0
                self.total_trades += 1 
                # Calculate transaction fee
                sell_amount = self.shares_held * current_price
                fee = sell_amount * self.transaction_fee # Use fixed fee
                fee_paid_this_step = fee
                self.total_fees_paid += fee
                
                # Update balance and shares
                self.balance += sell_amount - fee
                # --- Log intermediate state --- #
                logger.debug(
                    f"_take_action (Sell): Post-update Balance={self.balance:.4f}, "
                    f"Shares=0, Fee={fee:.4f}"
                )
                if not np.isfinite(self.balance):
                    logger.error(
                        f"_take_action (Sell): Balance became non-finite! "
                        f"{self.balance}"
                    )
                # --- End log --- #
                
                # Log sell
                sell_info = {
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'shares': self.shares_held,
                    'amount': sell_amount,
                    'fee': fee,
                    'balance_after': self.balance
                }
                self.trades.append(sell_info)
                self.sell_prices.append(current_price)
                
                logger.debug(
                    f"Step {self.current_step}: Sold {self.shares_held:.6f} shares @ "
                    f"{current_price:.2f} (Amt: {sell_amount:.2f}, Fee: {fee:.2f}) -> "
                    f"Bal: {self.balance:.2f}"
                )
                
                self.shares_held = 0
                self.asset_value = 0
            else:
                # Log failed sell attempt
                self.failed_sells += 1
                logger.debug(
                    f"Step {self.current_step}: Attempted Sell, but no shares held."
                )
                # We didn't increment total_sells or total_trades, so no need to decrement.

        elif action == 2:  # Buy
            if self.balance > ZERO_THRESHOLD: # Check if balance > 0
                # <<< ADDED CHECK: Prevent buying if already holding shares >>>
                if self.shares_held > ZERO_THRESHOLD:
                    self.failed_buys += 1 # Increment failed buy counter
                    logger.debug(
                        f"Step {self.current_step}: Attempted Buy, but already holding "
                        f"{self.shares_held:.6f} shares. Holding."
                    )
                    return # Exit the function, action becomes Hold # Keep return to prevent execution
                # <<< END ADDED CHECK >>>

                # Calculate amount to invest based on max_position
                invest_amount = self.balance * self.max_position
                
                # <<< ADDED CHECK: Minimum Trade Value >>>
                min_trade_value = self.initial_balance * 0.005 # Minimum 0.5% of initial balance
                if invest_amount < min_trade_value:
                    self.failed_buys += 1
                    logger.debug(
                        f"Step {self.current_step}: Attempted Buy. Invest amount "
                        f"{invest_amount:.2f} < min trade value {min_trade_value:.2f}. "
                        f"Holding."
                    )
                    return # Treat as Hold
                # <<< END ADDED CHECK >>>

                # Calculate shares we can buy considering the fee
                # Avoid division by zero/small price
                if current_price * (1 + self.transaction_fee) > ZERO_THRESHOLD:
                    shares_to_buy = invest_amount / (current_price * (1 + self.transaction_fee))
                else:
                    shares_to_buy = 0 # Cannot buy if price is zero

                # Ensure we can buy a meaningful amount of shares
                if shares_to_buy > ZERO_THRESHOLD: # Use threshold
                    # --- Buy Succeeded ---
                    # Increment trade/buy counts here
                    self.total_buys += 1
                    self.total_trades += 1

                    # Buy shares
                    self.shares_held = shares_to_buy
                    buy_cost = self.shares_held * current_price
                    fee = buy_cost * self.transaction_fee
                    fee_paid_this_step = fee
                    self.total_fees_paid += fee
                    
                    # Update balance and asset value
                    self.balance -= (buy_cost + fee)
                    # Ensure balance doesn't go negative due to float precision
                    self.balance = max(0, self.balance) 
                    self.asset_value = self.shares_held * current_price
                    # --- Log intermediate state --- #
                    logger.debug(
                        f"_take_action (Buy): Post-update Balance={self.balance:.4f}, "
                        f"Shares={self.shares_held:.8f}, AssetVal={self.asset_value:.4f}, "
                        f"Fee={fee:.4f}"
                    )
                    if not np.isfinite(self.balance) or not np.isfinite(self.shares_held) or not np.isfinite(self.asset_value):
                        logger.error(
                            f"_take_action (Buy): State became non-finite! "
                            f"Bal={self.balance}, Shares={self.shares_held}, "
                            f"AssetVal={self.asset_value}"
                        )
                    # --- End log --- #
                    
                    # Record buy price
                    self.last_buy_price = current_price
                    self.buy_prices.append(current_price)
                    
                    # Log buy
                    buy_info = {
                        'step': self.current_step,
                        'type': 'buy',
                        'price': current_price,
                        'shares': self.shares_held,
                        'cost': buy_cost,
                        'fee': fee,
                        'balance_after': self.balance
                    }
                    self.trades.append(buy_info)
                    
                    logger.debug(
                        f"Step {self.current_step}: Bought {self.shares_held:.6f} shares @ "
                        f"{current_price:.2f} (Cost: {buy_cost:.2f}, Fee: {fee:.2f}) -> "
                        f"Bal: {self.balance:.2f}"
                    )
                else:
                    # --- Buy Failed (Insufficient funds for meaningful amount OR price too low) ---
                    self.failed_buys += 1 # Increment failed buy counter
                    logger.debug(
                        f"Step {self.current_step}: Attempted Buy. Bal: {self.balance:.2f}, "
                        f"Price: {current_price:.2f}, MaxPos: {self.max_position:.2f}. "
                        f"Calculated shares {shares_to_buy:.8f} <= threshold. Holding."
                    )
            else:
                # --- Buy Failed (Zero balance) ---
                self.failed_buys += 1 # Increment failed buy counter
                logger.debug(
                    f"Step {self.current_step}: Attempted Buy, but balance is "
                    f"{self.balance:.2f}. Holding."
                )
    
    def _update_portfolio_value(self):
        """Update the portfolio value based on current balance and asset prices."""
        current_price = self.data['close'].iloc[self.current_step]
        self.asset_value = self.shares_held * current_price
        self.portfolio_value = self.balance + self.asset_value
        # --- DETAILED LOGGING FOR PORTFOLIO VALUE --- #
        logger.debug(
            f"_update_portfolio_value (Step {self.current_step}): "
            f"Price={current_price:.4f}, Shares={self.shares_held:.8f}, "
            f"Balance={self.balance:.4f} -> AssetVal={self.asset_value:.4f}, "
            f"PortVal={self.portfolio_value:.4f}"
        )
        # Check for non-finite values immediately after calculation
        if not np.isfinite(self.asset_value) or not np.isfinite(self.portfolio_value):
            logger.error(
                f"_update_portfolio_value: NON-FINITE VALUE DETECTED! "
                f"(Step {self.current_step})"
                f" Price={current_price:.4f}, Shares={self.shares_held:.8f}, "
                f"Balance={self.balance:.4f}"
                f" -> AssetVal={self.asset_value}, PortVal={self.portfolio_value}"
            )
        # --- END DETAILED LOGGING --- #
    
    def _calculate_reward(self, action, prev_portfolio_value, fee_paid_this_step):
        """
        Calculate the reward based on multiple components.
        
        Args:
            action: The action taken (0: Sell, 1: Hold, 2: Buy)
            prev_portfolio_value: Portfolio value before the action
            fee_paid_this_step: Transaction fee incurred in this step
            
        Returns:
            A dictionary containing the breakdown of reward components and the total scaled reward.
        """ # noqa E501
        # Initialize reward components dictionary
        reward_components = {
            'portfolio_change': 0.0,
            'drawdown_penalty': 0.0,
            'sharpe_reward': 0.0,
            'fee_penalty': 0.0,
            'benchmark_reward': 0.0,
            # 'consistency_penalty': 0.0, # REMOVED
            'idle_penalty': 0.0,
            'profit_bonus': 0.0,
            'exploration_bonus': 0.0,
            'trade_penalty': 0.0, # ADDED
            'raw_total': 0.0, # Sum before scaling
            'total_reward': 0.0 # Final scaled reward
        }

        current_price = self.data['close'].iloc[self.current_step]

        # --- RESTORED REWARD COMPONENTS (Except Exploration Bonus - Keep commented) --- #
        # 1. Portfolio Value Change Reward (Use Percentage Change)
        portfolio_change_pct = 0.0
        if prev_portfolio_value > ZERO_THRESHOLD: # Avoid division by zero
            portfolio_change_pct = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Scale by 100 to make it more comparable to other penalties/bonuses? Optional.
        reward_components['portfolio_change'] = portfolio_change_pct * self.portfolio_change_weight

        # 2. Drawdown Penalty (Re-enabled)
        current_drawdown = 0.0
        if self.max_portfolio_value > ZERO_THRESHOLD:
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        reward_components['drawdown_penalty'] = -current_drawdown * self.drawdown_penalty_weight

        # 3. Sharpe Ratio Reward (Re-enabled - using rolling window)
        sharpe_ratio_rolling = 0.0
        if len(self.step_returns) >= self.sharpe_window:
            window_returns = np.array(self.step_returns[-self.sharpe_window:])
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)
            if std_return > ZERO_THRESHOLD:
                # Simple Sharpe (no risk-free rate)
                sharpe_ratio_rolling = mean_return / std_return
        reward_components['sharpe_reward'] = sharpe_ratio_rolling * self.sharpe_reward_weight

        # 4. Fee Penalty (Re-enabled)
        reward_components['fee_penalty'] = -fee_paid_this_step * self.fee_penalty_weight

        # 5. Benchmark Comparison Reward (Keep commented for now)
        # ... (code omitted)

        # 6. Consistency Penalty (DISABLED)
        # consistency_penalty = 0.0
        # if action != self.last_action: # Action changed
        #      if action == 0 and self.consecutive_buys < self.consistency_threshold: 
        #          consistency_penalty = -1.0
        #      elif action == 2 and self.consecutive_sells < self.consistency_threshold: 
        #          consistency_penalty = -1.0
        # reward_components['consistency_penalty'] = consistency_penalty * self.consistency_penalty_weight

        # --- NEW: 6. Direct Trade Penalty --- #
        trade_penalty = 0.0
        # Apply penalty if a buy (2) or sell (0) was the action
        if action in [0, 2]: # REMOVED: and fee_paid_this_step > 1e-9
            trade_penalty = -1.0
        
        # <<< REMOVE DEBUG LOG >>>
        calculated_trade_penalty = trade_penalty * self.trade_penalty_weight
        # logger.debug(f"_calculate_reward: Trade Penalty Calc - Action={action}, BasePenalty={trade_penalty}, Weight={self.trade_penalty_weight:.6f}, Result={calculated_trade_penalty:.6f}")
        # <<< END REMOVE DEBUG LOG >>>
        
        reward_components['trade_penalty'] = calculated_trade_penalty # Use the calculated value
        # --- END NEW --- #

        # 7. Idle Penalty (Re-enabled)
        idle_penalty = 0.0
        if action == 1 and self.consecutive_holds > self.idle_threshold:
             idle_penalty = -1.0 
        reward_components['idle_penalty'] = idle_penalty * self.idle_penalty_weight

        # 8. Profit/Selling Bonus (Re-enabled)
        profit_bonus = 0.0
        if action == 0 and self.last_buy_price is not None: 
            sell_profit_pct = (current_price - self.last_buy_price) / self.last_buy_price
            if sell_profit_pct > 0: 
                profit_bonus = sell_profit_pct * 2 # Reduced multiplier from 10 to 2
                self.last_buy_price = None 
        reward_components['profit_bonus'] = profit_bonus * self.profit_bonus_weight

        # 9. Exploration Bonus (Keep commented out for now)
        # reward_components['exploration_bonus'] = self.exploration_bonus_value * self.exploration_bonus_weight

        # --- END RESTORED COMPONENTS --- #

        # Sum all active reward components (excluding exploration bonus for now)
        components_to_sum = [
            k for k in reward_components 
            if k not in ['raw_total', 'total_reward'] # Exclude invalid penalty
        ]
        raw_total = sum(reward_components[key] for key in components_to_sum)
        
        reward_components['raw_total'] = raw_total

        # Apply reward scaling for the final reward
        reward_components['total_reward'] = raw_total * self.reward_scaling

        # --- Update cumulative reward sums ---
        for key in self.cumulative_rewards:
            if key in reward_components:
                self.cumulative_rewards[key] += reward_components[key]
        # --- End update ---

        # Always log for debugging purposes for now
        component_str = ', '.join([f"{k}: {v:.4f}" for k, v in reward_components.items()
                                  if k not in ['raw_total', 'total_reward'] and abs(v) > 1e-6]) # Only log non-zero components
        logger.debug(
            f"Step {self.current_step} (EpStep {self.episode_step}) Action {action} -> Reward: {reward_components['total_reward']:.4f} | Components: {component_str}"
        )

        return reward_components
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Numpy array containing the observation
        """
        # Get historical feature data (sequence_length prior steps)
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.sequence_length + 1)
        
        # Make sure we have at least sequence_length entries
        if end_idx - start_idx + 1 < self.sequence_length:
            start_idx = max(0, end_idx - self.sequence_length + 1)
        
        historical_data = self.data.iloc[start_idx:end_idx+1]
        
        # Ensure we have exactly sequence_length rows for feature data
        if len(historical_data) < self.sequence_length:
            # Pad with the first row if needed
            padding = pd.DataFrame([historical_data.iloc[0]] * 
                                 (self.sequence_length - len(historical_data)))
            historical_data = pd.concat([padding, historical_data])
        
        # Extract features and flatten
        feature_data = []
        for feature in self.features:
            feature_data.extend(historical_data[feature].values)
        
        # Combine features and account info
        observation = np.array(feature_data, dtype=np.float32)

        # --- DEBUG PRINTS FOR SHAPE DIAGNOSIS ---
        # logger.debug(f"[OBS DEBUG] Features used: {self.features}")
        # logger.debug(f"[OBS DEBUG] feature_data length: {len(feature_data)} (should be sequence_length * num_features)")
        # logger.debug(f"[OBS DEBUG] Final observation shape: {observation.shape}, dtype: {observation.dtype}")
        # logger.debug(f"[OBS DEBUG] First 10 obs values: {observation[:10]}")
        # --- END DEBUG PRINTS ---

        return observation
    
    def _get_info(self):
        """
        Get additional information about the environment state.
        
        Returns:
            Dictionary containing environment info
        """ # noqa E501
        current_price = self.data['close'].iloc[self.current_step]
        
        # Base info
        info = {
            'step': self.current_step,
            'episode_step': self.episode_step,
            'timestamp': self.data.index[self.current_step],
            'price': current_price,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'asset_value': self.asset_value,
            'portfolio_value': self.portfolio_value,
            'initial_balance': self.initial_balance,
            'cash_ratio': self.balance / self.portfolio_value 
                          if self.portfolio_value > 0 else 1.0,
            'total_trades': self.total_trades,
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'total_holds': self.total_holds,
            'max_drawdown': self.max_drawdown,
            'total_fees_paid': self.total_fees_paid, # Added total fees
            'failed_buys': self.failed_buys, # Added failed buy counter
            'failed_sells': self.failed_sells, # Added failed sell counter
            'last_action': self.last_action, # Added last action
            'consecutive_holds': self.consecutive_holds, # Added consecutive holds
            'consecutive_buys': self.consecutive_buys,
            'consecutive_sells': self.consecutive_sells,
            'exploration_bonus_value': self.exploration_bonus_value, # Current bonus value
        }
        
        # Calculate overall episode returns and Sharpe if possible
        # Note: self.step_returns holds individual step returns
        if len(self.portfolio_values) > 1:
            # --- Debugging Return Calculation ---
            initial_val_debug = self.initial_balance
            current_val_debug = self.portfolio_value
            # logger.debug(f"_get_info: Calculating episode_return. Initial: {initial_val_debug}, Current: {current_val_debug}") # Original Debug
            # logger.debug(f"_get_info: [DEBUG CHECK] Calculating episode_return. Initial: {initial_val_debug}, Current: {current_val_debug}") # Now DEBUG only

            # --- ENHANCED ROBUSTNESS CHECK --- #
            episode_return = 0.0 # Default value
            inputs_valid = True

            # 1. Check if inputs are finite
            if not np.isfinite(initial_val_debug):
                logger.warning(f"_get_info: Initial balance ({initial_val_debug}) is non-finite. Setting return to 0.")
                inputs_valid = False
            if not np.isfinite(current_val_debug):
                logger.warning(f"_get_info: Current portfolio value ({current_val_debug}) is non-finite. Setting return to 0.")
                inputs_valid = False

            # 2. Check if initial balance is near zero (if inputs were finite)
            if inputs_valid and abs(initial_val_debug) <= ZERO_THRESHOLD:
                logger.warning(f"_get_info: Initial balance ({initial_val_debug}) is zero or near-zero. Setting return to 0.")
                inputs_valid = False

            # 3. Proceed with calculation only if all checks passed
            if inputs_valid:
                try:
                    calculated_return = (current_val_debug - initial_val_debug) / initial_val_debug
                    # 4. Check if the *result* is finite
                    if not np.isfinite(calculated_return):
                        logger.warning(f"_get_info: Calculated episode_return is non-finite ({calculated_return}). Forcing to 0.0. Initial: {initial_val_debug}, Current: {current_val_debug}")
                        # episode_return remains 0.0 (default)
                    else:
                        episode_return = calculated_return # Use the valid, finite result
                        logger.debug(f"_get_info: Successfully calculated episode_return={episode_return:.4f} (Initial: {initial_val_debug:.2f}, Current: {current_val_debug:.2f})") # Log success
                except Exception as e:
                    logger.error(f"_get_info: Error during episode_return division: {e}. Initial: {initial_val_debug}, Current: {current_val_debug}", exc_info=True)
                    # episode_return remains 0.0 (default)
            # --- END ENHANCED CHECK --- #

            # if initial_val_debug <= ZERO_THRESHOLD: # OLD CHECK - REMOVED
            #     logger.warning(f"_get_info: Initial balance ({initial_val_debug}) is zero or less. Setting return to 0.")
            #     episode_return = 0.0
            # else:
            #     try:
            #         calculated_return = (current_val_debug - initial_val_debug) / initial_val_debug
            #         # Explicitly check for non-finite results AFTER calculation
            #         if not np.isfinite(calculated_return):
            #             logger.warning(f"_get_info: Calculated episode_return is non-finite ({calculated_return}). Forcing to 0.0. Initial: {initial_val_debug}, Current: {current_val_debug}")
            #             episode_return = 0.0 # Force to 0 if NaN or Inf
            #         else:
            #             episode_return = calculated_return # Use the valid, finite result
            #     except Exception as e:
            #         logger.error(f"_get_info: Error calculating episode_return: {e}. Initial: {initial_val_debug}, Current: {current_val_debug}", exc_info=True)
            #         episode_return = 0.0 # Default on error
            # --- End Debugging --- # OLD BLOCK REMOVED
            info['episode_return'] = episode_return
            
            # Calculate overall Sharpe from portfolio_values (less noisy than step_returns)
            portfolio_returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            if len(portfolio_returns) > 1:
                mean_portfolio_return = np.mean(portfolio_returns)
                std_portfolio_return = np.std(portfolio_returns)
                # <<< REMOVE DEBUG LOG >>>
                # logger.debug(f"_get_info: Sharpe Calculation Inputs - MeanReturn={mean_portfolio_return:.6f}, StdReturn={std_portfolio_return:.6f}")
                # <<< END REMOVE DEBUG LOG >>>
                if std_portfolio_return > 1e-9:
                    # Optional annualization (sqrt(252) for daily)
                    # info['sharpe_ratio_episode'] = (mean_portfolio_return / std_portfolio_return) * np.sqrt(252)
                    info['sharpe_ratio_episode'] = mean_portfolio_return / std_portfolio_return # Simple non-annualized
                else:
                    info['sharpe_ratio_episode'] = 0.0
            else:
                info['sharpe_ratio_episode'] = 0.0
                 
            # Rolling Sharpe (based on step returns used in reward)
            # This reflects the Sharpe value used in the reward calculation
            if len(self.step_returns) >= self.sharpe_window:
                window_returns = np.array(self.step_returns[-self.sharpe_window:])
                mean_return = np.mean(window_returns)
                std_return = np.std(window_returns)
                if std_return > 1e-9:
                    sharpe_ratio_rolling = mean_return / std_return
                    info['sharpe_ratio_rolling'] = sharpe_ratio_rolling
                else:
                    info['sharpe_ratio_rolling'] = 0.0
            else:
                info['sharpe_ratio_rolling'] = 0.0 # Not enough data yet
        else:
            info['episode_return'] = 0.0
            info['sharpe_ratio_episode'] = 0.0
            info['sharpe_ratio_rolling'] = 0.0
        
        # Benchmark (Buy & Hold) Performance
        # Avoid division by zero if start price is zero
        if abs(self.episode_start_price) > ZERO_THRESHOLD:
            initial_investment_units = self.initial_balance / self.episode_start_price
            benchmark_portfolio_value = initial_investment_units * current_price
            benchmark_return = (benchmark_portfolio_value - self.initial_balance) / self.initial_balance if self.initial_balance > ZERO_THRESHOLD else 0.0
        else:
            benchmark_portfolio_value = self.initial_balance # If start price is 0, value remains initial balance
            benchmark_return = 0.0 # Return is zero

        info['benchmark_portfolio_value'] = benchmark_portfolio_value
        info['benchmark_return'] = benchmark_return

        # --- Add cumulative reward components to info ---
        info['cumulative_reward_components'] = self.cumulative_rewards.copy()
        # --- End add cumulative ---

        # --- Calculate and add Calmar/Sortino Ratios at episode end ---
        # Default values
        info['calmar_ratio'] = 0.0
        info['sortino_ratio'] = 0.0

        # Check if enough data points exist for metric calculation
        # Using > 2 because metrics like std dev need at least 2 points,
        # and calculate_trading_metrics itself has internal checks.
        if len(self.portfolio_values) > 2:
            try:
                # Ensure portfolio_values is a numpy array
                portfolio_values_np = np.array(self.portfolio_values)
                # Call the utility function
                trading_metrics = calculate_trading_metrics(portfolio_values_np)

                # Add calculated metrics to info, checking if they exist in the result
                if trading_metrics:
                    info['calmar_ratio'] = trading_metrics.get('calmar_ratio', 0.0)
                    info['sortino_ratio'] = trading_metrics.get('sortino_ratio', 0.0)
                    # Optionally add other metrics if needed later
                    # info['max_drawdown_metric'] = trading_metrics.get('max_drawdown', 0.0)
                    # info['sharpe_ratio_metric'] = trading_metrics.get('sharpe_ratio', 0.0)

            except Exception as e:
                logger.error(f"_get_info: Error calculating trading metrics: {e}", exc_info=True)
                # Keep default values (0.0) on error

        # --- End Calmar/Sortino Calculation ---

        # Reward components are added in the step method after _calculate_reward is called
        # if 'reward_components' not in info: # Ensure it exists even at step 0
        #    info['reward_components'] = {} # Initialize empty
        
        return info
    
    def render(self):
        """
        Render the environment (human mode or rgb_array).
        """
        if self.render_mode == 'human':
            # Implement human-readable rendering (e.g., print status)
            print(f"Step: {self.current_step}, "
                  f"Portfolio: {self.portfolio_value:.2f}, "
                  f"Balance: {self.balance:.2f}, "
                  f"Shares: {self.shares_held:.4f}")
            # Potentially add plotting for human mode
            return None  # Return None for human mode

        elif self.render_mode == 'rgb_array':
            # --- Basic RGB Array Rendering --- 
            # Create a simple plot and return as numpy array
            fig, ax = plt.subplots(figsize=(8, 4))
            start = max(0, self.current_step - self.window_size)
            end = self.current_step + 1
            
            # Plot price
            ax.plot(
                self.data.index[start:end],
                self.data['close'].iloc[start:end],
                label='Close Price', color='blue'
            )
            
            # Mark trades
            buy_steps = [t['step'] for t in self.trades if t['type'] == 'buy']
            sell_steps = [t['step'] for t in self.trades if t['type'] == 'sell']
            
            buy_indices = [i for i, step in 
                           enumerate(self.data.index[start:end]) 
                           if step in buy_steps]
            sell_indices = [i for i, step in 
                            enumerate(self.data.index[start:end]) 
                            if step in sell_steps]
            
            if buy_indices:
                ax.scatter(
                     self.data.index[start:end][buy_indices],
                     self.data['close'].iloc[start:end][buy_indices],
                     marker='^', color='green', s=100, label='Buy'
                 )
            if sell_indices:
                ax.scatter(
                    self.data.index[start:end][sell_indices],
                    self.data['close'].iloc[start:end][sell_indices],
                    marker='v', color='red', s=100, label='Sell'
                )
            
            ax.set_title(f"Trading Environment - Step {self.current_step}")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            
            # Draw figure and convert to RGB array
            fig.canvas.draw()
            rgb_array = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)  # Close the plot to avoid display
            
            return rgb_array
        else:
            # If render_mode is None or unsupported
            return None  # Or raise an error if preferred

    def close(self):
        """Clean up resources."""
        plt.close()
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed] 