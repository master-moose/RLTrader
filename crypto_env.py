import gymnasium
from gymnasium import spaces
import numpy as np
import logging
from collections import deque

# Set up logging
logger = logging.getLogger(__name__)

class CryptoTradingEnv(gymnasium.Env):
    """
    A reinforcement learning environment for cryptocurrency trading.
    """
    
    def __init__(
        self,
        df,
        window_size=100,
        initial_balance=10000,
        transaction_fee=0.00075,  # Changed to transaction fee of 0.075%
        reward_scaling=0.01,
        use_position=True,
        oscillation_penalty=10.0,
        normalization_method="zscore",
        position_size=0.25,  # Default position size as 25% of portfolio
        max_position_size=0.5,  # Maximum position size as 50% of portfolio
        stop_loss_pct=0.05,  # Default 5% stop loss
        take_profit_pct=0.1,  # Default 10% take profit
        risk_per_trade=0.02,  # Risk 2% of portfolio per trade
        max_drawdown_pct=0.2  # Maximum allowed drawdown (20%)
    ):
        """
        Initialize the crypto trading environment.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data for a cryptocurrency
        window_size : int
            The number of previous time steps to include in the state
        initial_balance : float
            Starting balance in quote currency (e.g., USD)
        transaction_fee : float
            Trading fee as a decimal (e.g., 0.00075 for 0.075%)
        reward_scaling : float
            Scaling factor for the reward function
        use_position : bool
            Whether to include position information in the state
        oscillation_penalty : float
            Penalty for oscillating between buy and sell actions
        normalization_method : str
            Method to normalize input features ("zscore", "minmax", or None)
        position_size : float
            Default position size as a fraction of portfolio (0.1 = 10%)
        max_position_size : float
            Maximum position size as a fraction of portfolio (0.5 = 50%)
        stop_loss_pct : float
            Default stop loss percentage (0.05 = 5%)
        take_profit_pct : float
            Default take profit percentage (0.1 = 10%)
        risk_per_trade : float
            Maximum risk per trade as a fraction of portfolio (0.02 = 2%)
        max_drawdown_pct : float
            Maximum allowed drawdown as a fraction of portfolio (0.2 = 20%)
        """
        super(CryptoTradingEnv, self).__init__()
        
        # Validate inputs
        if df is None or len(df) == 0:
            raise ValueError("DataFrame cannot be None or empty")
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns")
            
        # Data validation and preprocessing
        # Remove rows with NaN, negative or zero values in key columns
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if df[col].isna().any() or (df[col] <= 0).any():
                logger.warning(f"Found NaN, negative or zero values in {col}. Fixing...")
                # For price columns, forward fill NaN and ensure no negative/zero values
                if col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = np.maximum(df[col], 0.00001)  # Set minimum positive value
                # For volume, replace NaN with 0 and negative with absolute value
                else:
                    df[col] = df[col].fillna(0)
                    df[col] = np.abs(df[col])
        
        self.df = df
        self.window_size = max(1, window_size)
        self.initial_balance = np.float32(max(100, initial_balance))  # Enforce reasonable minimum
        self.transaction_fee = np.float32(max(0, min(0.1, transaction_fee)))  # Limit between 0 and 10%
        self.reward_scaling = np.float32(max(0.0001, min(1.0, reward_scaling)))  # Between 0.01% and 100%
        self.use_position = use_position
        self.oscillation_penalty = np.float32(max(0, oscillation_penalty))
        self.normalization_method = normalization_method
        
        # Risk management parameters
        self.position_size = np.float32(max(0.01, min(1.0, position_size)))  # Between 1% and 100%
        self.max_position_size = np.float32(max(0.1, min(1.0, max_position_size)))  # Between 10% and 100%
        self.stop_loss_pct = np.float32(max(0.01, min(0.5, stop_loss_pct)))  # Between 1% and 50%
        self.take_profit_pct = np.float32(max(0.01, min(1.0, take_profit_pct)))  # Between 1% and 100%
        self.risk_per_trade = np.float32(max(0.001, min(0.1, risk_per_trade)))  # Between 0.1% and 10%
        self.max_drawdown_pct = np.float32(max(0.05, min(0.5, max_drawdown_pct)))  # Between 5% and 50%
        
        # Define limits to prevent numerical instability
        self.MAX_PORTFOLIO_VALUE = self.initial_balance * 100  # Max 100x initial capital
        self.MAX_ASSET_UNITS = 1e6  # Maximum crypto units that can be owned
        self.MAX_REWARD = 1000.0  # Clip rewards to prevent extreme values
        self.MIN_PRICE = 0.00001  # Minimum price to prevent division by zero
        
        # Set up action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: SELL, 1: HOLD, 2: BUY
        
        # Calculate the observation space size
        num_price_features = 5  # OHLCV
        num_technical_indicators = 0  # Will be set based on enabled indicators
        
        # Technical indicators to use (set to empty dictionary to disable)
        self.tech_indicators = {
            'sma': [7, 25, 99],   # Simple Moving Average periods
            'ema': [7, 25, 99],   # Exponential Moving Average periods
            'rsi': [14],          # Relative Strength Index periods
            'macd': True,         # MACD (uses default settings)
            'bbands': True,       # Bollinger Bands (uses default settings)
            'volume_features': True,  # Volume-based features
        }
        
        # Count technical indicators
        if 'sma' in self.tech_indicators:
            num_technical_indicators += len(self.tech_indicators['sma'])
        if 'ema' in self.tech_indicators:
            num_technical_indicators += len(self.tech_indicators['ema'])
        if 'rsi' in self.tech_indicators:
            num_technical_indicators += len(self.tech_indicators['rsi'])
        if self.tech_indicators.get('macd', False):
            num_technical_indicators += 3  # Signal, MACD, Histogram
        if self.tech_indicators.get('bbands', False):
            num_technical_indicators += 3  # Upper, Middle, Lower
        if self.tech_indicators.get('volume_features', False):
            num_technical_indicators += 3  # Volume indicators
            
        # Add account state features
        num_account_features = 2  # Balance and portfolio value
        if self.use_position:
            num_account_features += 1  # Add position feature
            
        total_features = (num_price_features + num_technical_indicators) * window_size + num_account_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32
        )
        
        # Initialize indicators if required
        if num_technical_indicators > 0:
            self._initialize_indicators()
        
        # Additional tracking variables
        self.current_step = 0
        self.price_history = deque(maxlen=1000)  # Store recent prices for reference
        self.portfolio_history = deque(maxlen=1000)  # Store portfolio values
        self.action_history = deque(maxlen=100)  # Store recent actions
        self.trade_counts = {'buy': 0, 'sell': 0}
        self.last_action = None
        self.stable_counter = 0  # Count consecutive steps with stable portfolio value
        
        # Risk management tracking
        self.positions = []  # List to track open positions
        self.highest_portfolio_value = self.initial_balance  # For drawdown calculation
        self.current_drawdown_pct = 0  # Current drawdown percentage
        self.max_drawdown_reached = False  # Flag if max drawdown has been reached
        
        # Normalization parameters
        self.scaler = None
        self.feature_means = None
        self.feature_stds = None
        self.feature_mins = None
        self.feature_maxs = None
        
        # Market statistics for logging
        self.market_change_pct = 0
        self.agent_change_pct = 0
        
        # Fix long lines
        self.oscillation_tracking = {
            'actions': deque(maxlen=4),
            'penalties': 0
        }
        
        # Safety parameters to prevent numerical instability
        self.max_portfolio_value = 1e9  # 1 billion max portfolio value
        self.max_asset_units = 1e6  # 1 million max asset units
        self.min_price = 1e-6  # Minimum price to prevent division by zero
        
        # Validate and preprocess the data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        lowercase_columns = [col.lower() for col in df.columns]
        
        for col in required_columns:
            if col not in lowercase_columns:
                raise ValueError(
                    f"DataFrame must contain {col} column (case insensitive)"
                )
        
        logger.info(
            f"Initialized environment with {total_features} features, "
            f"{num_technical_indicators} technical indicators, "
            f"transaction fee {self.transaction_fee*100:.3f}%, "
            f"position size {self.position_size*100:.1f}%"
        )
    
    def _get_current_price(self):
        """
        Safely get the current price from the dataframe.
        """
        if self.current_step >= len(self.df):
            # If we've reached the end of data, use the last available price
            price = self.df.iloc[-1]['close']
        else:
            price = self.df.iloc[self.current_step]['close']
            
        # Safety check for numerical stability
        if not np.isfinite(price) or price <= 0:
            logger.warning(f"Invalid price value detected: {price}, using last valid price or 1.0")
            # Try to use last valid price from history, or default to 1.0
            if len(self.price_history) > 0:
                price = self.price_history[-1]
            else:
                price = 1.0
                
        # Ensure minimum price to prevent division by zero
        price = max(self.MIN_PRICE, price)
        
        return np.float32(price)
    
    def _calculate_portfolio_value(self):
        """
        Calculate the current portfolio value with improved numerical stability.
        """
        # Get current price with safety checks
        price = self._get_current_price()
        
        # Calculate asset value with safety limits
        asset_value = 0
        if self.assets_owned[0] > 0:
            # Apply a safety limit to prevent overflow
            clipped_assets = min(self.assets_owned[0], self.MAX_ASSET_UNITS)  # Limit asset units
            
            # If assets dramatically exceed limit, log warning
            if self.assets_owned[0] > self.MAX_ASSET_UNITS:
                logger.warning(f"Assets owned exceeded limit: {self.assets_owned[0]}, clipping to {self.MAX_ASSET_UNITS}")
                # Actually clip the stored assets to prevent future issues
                self.assets_owned[0] = np.float32(self.MAX_ASSET_UNITS)
                
            asset_value = clipped_assets * price
            
            # Prevent extreme values
            if asset_value > self.MAX_PORTFOLIO_VALUE:
                logger.warning(f"Asset value too large: {asset_value}, clipping to {self.MAX_PORTFOLIO_VALUE}")
                asset_value = self.MAX_PORTFOLIO_VALUE
        
        # Apply same clipping to balance
        if self.balance[0] > self.MAX_PORTFOLIO_VALUE:
            logger.warning(f"Balance too large: {self.balance[0]}, clipping to {self.MAX_PORTFOLIO_VALUE}")
            self.balance[0] = np.float32(self.MAX_PORTFOLIO_VALUE)
            
        # Calculate total portfolio value
        portfolio_value = self.balance[0] + asset_value
        
        # Ensure portfolio value is finite and reasonable
        if not np.isfinite(portfolio_value) or portfolio_value < 0:
            logger.warning(f"Invalid portfolio value detected: {portfolio_value}, resetting to initial amount")
            portfolio_value = self.initial_balance
            # Also reset balance and assets to recover from numerical error
            self.balance[0] = np.float32(self.initial_balance)
            self.assets_owned[0] = np.float32(0)
        
        # Apply maximum portfolio value limit
        if portfolio_value > self.MAX_PORTFOLIO_VALUE:
            logger.warning(f"Portfolio value exceeded maximum: {portfolio_value}, clipping to {self.MAX_PORTFOLIO_VALUE}")
            # Calculate what portion should be balance vs assets
            if asset_value > 0:
                ratio = asset_value / portfolio_value
                asset_portion = self.MAX_PORTFOLIO_VALUE * ratio
                balance_portion = self.MAX_PORTFOLIO_VALUE - asset_portion
                
                # Recalculate assets owned based on new asset portion
                if price > 0:
                    self.assets_owned[0] = np.float32(asset_portion / price)
                    
                # Update balance
                self.balance[0] = np.float32(balance_portion)
                
                # Recalculate portfolio value
                portfolio_value = self.MAX_PORTFOLIO_VALUE
            else:
                # If no assets, just clip balance
                self.balance[0] = np.float32(self.MAX_PORTFOLIO_VALUE)
                portfolio_value = self.MAX_PORTFOLIO_VALUE
        
        # Update highest portfolio value for drawdown calculation
        if portfolio_value > self.highest_portfolio_value:
            self.highest_portfolio_value = portfolio_value
        
        # Calculate current drawdown
        if self.highest_portfolio_value > 0:
            self.current_drawdown_pct = (self.highest_portfolio_value - portfolio_value) / self.highest_portfolio_value
            # Check if max drawdown has been reached
            if self.current_drawdown_pct > self.max_drawdown_pct:
                if not self.max_drawdown_reached:
                    logger.warning(f"Maximum drawdown of {self.max_drawdown_pct*100:.1f}% reached: current {self.current_drawdown_pct*100:.1f}%")
                    self.max_drawdown_reached = True
                
        return np.float32(portfolio_value)
    
    def _check_stop_loss_take_profit(self):
        """
        Check if any positions have hit their stop loss or take profit levels.
        Returns True if any positions were closed.
        """
        if not self.positions:
            return False
        
        current_price = self._get_current_price()
        positions_to_remove = []
        position_closed = False
        
        for idx, position in enumerate(self.positions):
            entry_price = position['entry_price']
            position_type = position['type']  # 'long' or 'short'
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            position_size = position['size']
            
            # Check stop loss and take profit for long positions
            if position_type == 'long':
                # Check if price fell below stop loss
                if current_price <= stop_loss:
                    logger.info(f"Stop loss triggered at {current_price:.4f} (entry: {entry_price:.4f}, stop: {stop_loss:.4f})")
                    # Close position at stop loss price
                    self._close_position(idx, stop_loss)
                    positions_to_remove.append(idx)
                    position_closed = True
                # Check if price rose above take profit
                elif current_price >= take_profit:
                    logger.info(f"Take profit triggered at {current_price:.4f} (entry: {entry_price:.4f}, target: {take_profit:.4f})")
                    # Close position at take profit price
                    self._close_position(idx, take_profit)
                    positions_to_remove.append(idx)
                    position_closed = True
        
        # Remove closed positions (in reverse order to avoid index issues)
        for idx in sorted(positions_to_remove, reverse=True):
            self.positions.pop(idx)
            
        return position_closed
    
    def _close_position(self, position_idx, price):
        """
        Close a specific position at the given price.
        """
        position = self.positions[position_idx]
        position_type = position['type']
        position_size = position['size']
        entry_price = position['entry_price']
        
        if position_type == 'long':
            # Calculate profit/loss
            pnl = (price - entry_price) / entry_price
            # Calculate sale amount
            sale_amount = position_size * price
            # Apply transaction fee
            fee_amount = sale_amount * self.transaction_fee
            # Add to balance
            net_sale_amount = sale_amount - fee_amount
            self.balance[0] += np.float32(net_sale_amount)
            # Log trade
            logger.debug(f"Closed LONG position at {price:.4f}: P&L {pnl*100:.2f}%, Amount {net_sale_amount:.2f}")
            # Reduce assets owned
            self.assets_owned[0] -= np.float32(position_size)
            self.assets_owned[0] = max(0, self.assets_owned[0])  # Ensure non-negative
            # Record trade
            self.trade_counts['sell'] += 1
    
    def _calculate_position_size(self, portfolio_value, price):
        """
        Calculate appropriate position size based on risk management rules.
        """
        # Default position size as a percentage of portfolio
        default_size = portfolio_value * self.position_size
        
        # Calculate risk-based position size
        risk_amount = portfolio_value * self.risk_per_trade
        # Stop loss is self.stop_loss_pct below entry
        stop_loss_distance = price * self.stop_loss_pct
        
        # Position size based on risk (risk / stop loss distance)
        if stop_loss_distance > 0:
            risk_based_size = risk_amount / stop_loss_distance
            # Convert to number of units
            risk_based_units = risk_based_size / price
        else:
            risk_based_units = default_size / price
        
        # Calculate units based on default position size
        default_units = default_size / price
        
        # Use the smaller of the two
        units = min(risk_based_units, default_units)
        
        # Apply maximum position size limit
        max_units = (portfolio_value * self.max_position_size) / price
        units = min(units, max_units)
        
        # Ensure position is not too small (at least 1% of portfolio)
        min_units = (portfolio_value * 0.01) / price
        units = max(units, min_units)
        
        # Apply numerical stability limits
        units = min(units, self.MAX_ASSET_UNITS)
        units = max(units, 0)
        
        return np.float32(units)
    
    def _calculate_reward(self, action):
        """
        Calculate the reward for the current step with improved numerical stability.
        
        Parameters:
        -----------
        action : int
            The action taken (0: SELL, 1: HOLD, 2: BUY)
            
        Returns:
        --------
        reward : float
            The reward for the current step
        """
        # Calculate portfolio value change
        current_value = self._calculate_portfolio_value()
        prev_value = self.portfolio_value
        
        # If this is the first step or there was a value reset, use current value as previous
        if prev_value is None or prev_value <= 0:
            prev_value = current_value
            
        # Calculate portfolio change as a percentage (more numerically stable than absolute)
        if prev_value > 0:
            pct_change = (current_value - prev_value) / prev_value
        else:
            pct_change = 0
            
        # Clip to reasonable range to prevent extremely large rewards
        pct_change = np.clip(pct_change, -0.5, 0.5)  # Limit to ±50% per step
        
        # Base reward is the percentage change scaled by the reward scaling factor
        reward = pct_change * self.reward_scaling
        
        # Record the action for oscillation detection
        self.action_history.append(action)
        
        # Penalize action oscillation (frequent switching between buy and sell)
        # Only apply if we have at least 4 actions in history
        if len(self.action_history) >= 4:
            # Check last 4 actions for buy-sell-buy-sell or sell-buy-sell-buy pattern
            last_four = list(self.action_history)[-4:]
            
            # Buy-sell-buy-sell or sell-buy-sell-buy pattern
            if (last_four == [2, 0, 2, 0] or last_four == [0, 2, 0, 2]):
                oscillation_penalty = self.oscillation_penalty
                reward -= oscillation_penalty
                logger.warning(f"Detected action oscillation at step {self.current_step}: {last_four}")
        
        # Penalize extreme portfolio values
        if current_value > self.initial_balance * 10:
            # Encourage profit-taking for unusually high returns
            profit_taking_penalty = min(0.05, (current_value / (self.initial_balance * 10)) * 0.01) 
            reward -= profit_taking_penalty
        
        # Penalize excessive drawdown
        if self.current_drawdown_pct > self.max_drawdown_pct * 0.8:  # Approaching max drawdown
            drawdown_penalty = 0.1 * (self.current_drawdown_pct / self.max_drawdown_pct)
            reward -= drawdown_penalty
            
        # Prevent extreme reward values
        reward = np.clip(reward, -self.MAX_REWARD, self.MAX_REWARD)
        
        # Convert to numpy float32 for consistent precision
        return np.float32(reward)
    
    def _take_action(self, action):
        """
        Execute the trading action with improved numerical stability and risk management.
        
        Parameters:
        -----------
        action : int
            The action to take (0: SELL, 1: HOLD, 2: BUY)
        """
        # First check if any existing positions hit stop loss or take profit
        positions_closed = self._check_stop_loss_take_profit()
        
        # Skip further actions if max drawdown reached (only allow closing positions)
        if self.max_drawdown_reached and action == 2:  # Trying to buy during max drawdown
            logger.warning("Maximum drawdown reached - preventing new BUY positions")
            action = 1  # Convert to HOLD
        
        # Get current price with safety checks
        price = self._get_current_price()
        
        # Record price for history
        self.price_history.append(price)
        
        # SELL action
        if action == 0:
            if self.assets_owned[0] > 0:
                # Calculate how much to sell - all assets for now
                assets_to_sell = self.assets_owned[0]
                
                # Calculate sale amount
                sale_amount = assets_to_sell * price
                
                # Apply transaction fee
                fee_amount = sale_amount * self.transaction_fee
                # Ensure fee is reasonable
                fee_amount = min(fee_amount, sale_amount * 0.5)  # Cap at 50% of sale
                
                # Add to balance (after fee)
                self.balance[0] += np.float32(sale_amount - fee_amount)
                
                # Update assets owned
                self.assets_owned[0] -= assets_to_sell
                self.assets_owned[0] = max(0, self.assets_owned[0])  # Ensure non-negative
                
                # Close all positions
                self.positions = []
                
                # Record trade
                self.trade_counts['sell'] += 1
                logger.debug(f"SELL at {price:.4f}: {sale_amount:.2f} - {fee_amount:.2f} fee")
        
        # BUY action
        elif action == 2:
            if self.balance[0] > 0:
                # Calculate portfolio value for position sizing
                portfolio_value = self._calculate_portfolio_value()
                
                # Calculate appropriate position size based on risk management
                units_to_buy = self._calculate_position_size(portfolio_value, price)
                
                # Calculate purchase amount
                purchase_amount = min(units_to_buy * price, self.balance[0])
                
                # Apply transaction fee
                fee_amount = purchase_amount * self.transaction_fee
                # Ensure fee is reasonable
                fee_amount = min(fee_amount, purchase_amount * 0.5)  # Cap at 50% of purchase
                
                # Recalculate assets bought after fee
                assets_bought = (purchase_amount - fee_amount) / price
                
                # Apply safety limits to prevent overflow
                assets_bought = min(assets_bought, self.MAX_ASSET_UNITS)
                
                # Update assets and balance
                self.assets_owned[0] += np.float32(assets_bought)
                self.balance[0] -= np.float32(purchase_amount)
                self.balance[0] = max(0, self.balance[0])  # Ensure non-negative
                
                # Create a new position with stop loss and take profit
                stop_loss_price = price * (1 - self.stop_loss_pct)
                take_profit_price = price * (1 + self.take_profit_pct)
                
                # Add to positions list
                self.positions.append({
                    'type': 'long',
                    'entry_price': price,
                    'size': assets_bought,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'entry_time': self.current_step
                })
                
                # Record trade
                self.trade_counts['buy'] += 1
                logger.debug(f"BUY at {price:.4f}: {assets_bought:.6f} units for {purchase_amount:.2f} - {fee_amount:.2f} fee")
                logger.debug(f"SL: {stop_loss_price:.4f} ({self.stop_loss_pct*100:.1f}%), TP: {take_profit_price:.4f} ({self.take_profit_pct*100:.1f}%)")
        
        # Record last action
        self.last_action = action
        
        # Calculate portfolio value after action
        new_portfolio_value = self._calculate_portfolio_value()
        
        # Stability check - detect if portfolio value is same for too many steps
        if self.portfolio_value is not None and abs(new_portfolio_value - self.portfolio_value) < 0.0001:
            self.stable_counter += 1
            if self.stable_counter > 100:  # If stable for 100 steps, log a warning
                logger.warning(f"Portfolio value stable for {self.stable_counter} steps at {new_portfolio_value:.2f}")
                if self.stable_counter > 500 and self.assets_owned[0] > 0:
                    # Force a small price change to break stability
                    logger.warning("Forcing price change to break stability")
                    # Adjust the effective price slightly
                    self.assets_owned[0] = np.float32(self.assets_owned[0] * 1.001)
                    self.stable_counter = 0
        else:
            self.stable_counter = 0
            
        # Update portfolio value
        self.portfolio_value = new_portfolio_value
        
        # Record portfolio value for history
        self.portfolio_history.append(new_portfolio_value)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state with improved numerical stability.
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.balance = np.array([self.initial_balance], dtype=np.float32)
        self.assets_owned = np.array([0], dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self.current_step = 0
        self.action_history = deque(maxlen=100)
        self.price_history = deque(maxlen=1000)
        self.portfolio_history = deque(maxlen=1000)
        self.last_action = None
        self.trade_counts = {'buy': 0, 'sell': 0}
        self.stable_counter = 0
        
        # Reset risk management tracking
        self.positions = []
        self.highest_portfolio_value = self.initial_balance
        self.current_drawdown_pct = 0
        self.max_drawdown_reached = False
        
        # Market change tracking for final performance comparison
        self.start_price = self._get_current_price()
        self.market_change_pct = 0
        self.agent_change_pct = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Reset indicators
        self._reset_indicators()
        
        info = {
            'portfolio_value': float(self.portfolio_value),
            'balance': float(self.balance[0]),
            'assets_owned': float(self.assets_owned[0]),
            'step': self.current_step,
            'drawdown_pct': float(self.current_drawdown_pct),
            'open_positions': len(self.positions)
        }
        
        return observation, info 

    def _reset_indicators(self):
        """
        Reset technical indicators - called when environment is reset.
        """
        # Will be implemented when the indicators are needed
        pass
    
    def _initialize_indicators(self):
        """
        Initialize technical indicators.
        """
        # Will be implemented when the indicators are needed
        pass
    
    def _get_observation(self):
        """
        Get the current state observation.
        
        Returns:
            observation (np.ndarray): The current state observation
        """
        # Calculate observation space size
        num_price_features = 5  # OHLCV
        num_technical_indicators = 0
        
        # Count technical indicators
        if 'sma' in self.tech_indicators:
            num_technical_indicators += len(self.tech_indicators['sma'])
        if 'ema' in self.tech_indicators:
            num_technical_indicators += len(self.tech_indicators['ema'])
        if 'rsi' in self.tech_indicators:
            num_technical_indicators += len(self.tech_indicators['rsi'])
        if self.tech_indicators.get('macd', False):
            num_technical_indicators += 3  # Signal, MACD, Histogram
        if self.tech_indicators.get('bbands', False):
            num_technical_indicators += 3  # Upper, Middle, Lower
        if self.tech_indicators.get('volume_features', False):
            num_technical_indicators += 3  # Volume indicators
            
        # Add account state features
        num_account_features = 2  # Balance and portfolio value
        if self.use_position:
            num_account_features += 1  # Add position feature
            
        # Check if we've reached the end of data
        if self.current_step >= len(self.df):
            # Return zeros if we're past the end of data
            total_features = (num_price_features + num_technical_indicators) * self.window_size + num_account_features
            return np.zeros(total_features, dtype=np.float32)
            
        # Get price data for the current window
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Handle case where we don't have enough history yet
        padding = max(0, self.window_size - (end_idx - start_idx))
        
        # Get window data with safety checks
        window_data = []
        for i in range(start_idx, end_idx):
            if i < 0 or i >= len(self.df):
                # Padding with zeros for out-of-bounds indices
                window_data.append(np.zeros(num_price_features + num_technical_indicators))
            else:
                # Get current row
                row = self.df.iloc[i]
                
                # Basic OHLCV data
                price_data = [
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ]
                
                # Technical indicators would be added here
                tech_data = []
                
                # Combine data
                window_data.append(np.array(price_data + tech_data, dtype=np.float32))
        
        # Add padding if needed
        if padding > 0:
            for _ in range(padding):
                window_data.insert(0, np.zeros(num_price_features + num_technical_indicators))
        
        # Flatten window data
        flattened_window = np.array(window_data).flatten()
        
        # Add account features
        portfolio_value = self._calculate_portfolio_value()
        
        if self.use_position:
            # Include position information (1 if holding assets, 0 otherwise)
            position = 1.0 if self.assets_owned[0] > 0 else 0.0
            account_features = np.array([
                self.balance[0],
                portfolio_value,
                position
            ], dtype=np.float32)
        else:
            account_features = np.array([
                self.balance[0],
                portfolio_value
            ], dtype=np.float32)
        
        # Combine window and account features
        observation = np.concatenate([flattened_window, account_features])
        
        # Apply normalization if specified
        if self.normalization_method == "zscore":
            # Apply z-score normalization to price data (not account features)
            # This preserves the structure of the features while making them more suitable for neural networks
            
            # Extract price data
            price_data = observation[:-num_account_features]
            
            # Skip normalization if we don't have enough data
            if len(price_data) > 1 and np.std(price_data) > 0:
                # Compute mean and std on the fly
                mean = np.mean(price_data)
                std = np.std(price_data)
                
                # Apply normalization
                normalized_price_data = (price_data - mean) / (std + 1e-8)  # Add small epsilon to prevent division by zero
                
                # Combine normalized price data with account features
                observation = np.concatenate([normalized_price_data, account_features])
        
        # Ensure observation is the correct shape
        if observation.shape[0] != self.observation_space.shape[0]:
            logger.warning(f"Observation shape mismatch: {observation.shape} vs {self.observation_space.shape}")
            # Pad or truncate to correct size
            if observation.shape[0] < self.observation_space.shape[0]:
                # Pad with zeros
                padding = np.zeros(self.observation_space.shape[0] - observation.shape[0], dtype=np.float32)
                observation = np.concatenate([observation, padding])
            else:
                # Truncate
                observation = observation[:self.observation_space.shape[0]]
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Parameters:
        -----------
        action : int
            The action to take (0: SELL, 1: HOLD, 2: BUY)
            
        Returns:
        --------
        observation : numpy.ndarray
            The next state observation
        reward : float
            The reward for the action
        terminated : bool
            Whether the episode is done
        truncated : bool
            Whether the episode was truncated
        info : dict
            Additional information
        """
        # Execute the action
        self._take_action(action)
        
        # Move to next time step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Calculate market performance metrics
        if done and len(self.df) > 0:
            end_price = self._get_current_price()
            if self.start_price > 0:
                self.market_change_pct = (end_price - self.start_price) / self.start_price
            
            if self.initial_balance > 0:
                self.agent_change_pct = (self.portfolio_value - self.initial_balance) / self.initial_balance
                
                # Log performance comparison
                logger.info(f"Episode finished: Market change: {self.market_change_pct*100:.2f}%, "
                           f"Agent change: {self.agent_change_pct*100:.2f}%")
                logger.info(f"Trades: buy={self.trade_counts['buy']}, sell={self.trade_counts['sell']}")
        
        # Create info dictionary
        info = {
            'portfolio_value': float(self.portfolio_value),
            'balance': float(self.balance[0]),
            'assets_owned': float(self.assets_owned[0]),
            'step': self.current_step,
            'price': float(self._get_current_price()),
            'trade_count': sum(self.trade_counts.values()),
            'drawdown_pct': float(self.current_drawdown_pct),
            'open_positions': len(self.positions)
        }
        
        if done:
            info['market_change_pct'] = float(self.market_change_pct)
            info['agent_change_pct'] = float(self.agent_change_pct)
            
        return observation, reward, done, False, info 