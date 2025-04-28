from __future__ import annotations
import torch
import gymnasium as gym # Import Gymnasium
from gymnasium import spaces
from pathlib import Path
# Use correct relative imports now that files are copied
from .builder import ENVIRONMENTS
# from .custom import Environments # Remove unused custom base class import
import pandas as pd
# Assuming functions are exported via __init__.py in trademaster_utils
from .trademaster_utils import get_attr
import numpy as np
import os # Import os for getpid

# import sys # Unused
# from pathlib import Path # Unused
# import pickle # Unused
import os.path as osp # Keep osp if used by get_attr or similar internally
# import glob # Unused
import json
import logging


__all__ = ["HighFrequencyTradingEnvironment",
           "HighFrequencyTradingTrainingEnvironment"]


@ENVIRONMENTS.register_module()
class HighFrequencyTradingEnvironment(gym.Env): # Inherit from gymnasium.Env (via alias)
    # Add metadata for compatibility if needed
    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(self, **kwargs):
        # super(HighFrequencyTradingEnvironment, self).__init__() # Call gym.Env constructor if needed
        # Gymnasium doesn't require explicit super init for basic Env
        pass

        self.dataset = get_attr(kwargs, "dataset", None)
        self.task = get_attr(kwargs, "task", "train")
        self.test_dynamic = int(get_attr(kwargs, "test_dynamic", "-1"))
        self.task_index = int(get_attr(kwargs, "task_index", "-1"))
        self.work_dir = get_attr(kwargs, "work_dir", "")

        # --- Data Loading ---
        # Remove reliance on specific csv paths from dataset config
        # self.df_path = None
        # if self.task.startswith("train"):
        #     raise Exception(...)
        # elif self.task.startswith("valid"):
        #     self.df_path = get_attr(self.dataset, "valid_path", None)
        # else:
        #     self.df_path = get_attr(self.dataset, "test_path", None)

        # Get data directory from kwargs or dataset
        # (assuming dataset has a data_path attribute)
        data_dir = Path(get_attr(kwargs, "data_dir",
                                 get_attr(self.dataset, "data_path", "data")))
        # e.g., "BTCUSDT"
        symbol_filename_part = get_attr(kwargs, "symbol_filename_part", "")
        if not symbol_filename_part:
            # Attempt to derive from symbol if provided in dataset
            symbol = get_attr(self.dataset, "symbol", "")
            if symbol:
                # Fix syntax: Use raw string or escape backslash
                symbol_filename_part = symbol.replace('/', '_').replace(':', '')
        if not symbol_filename_part:
            raise ValueError(
                "Could not determine symbol filename part for data loading.")

        # Load HDF5 data from the specified directory
        # file_pattern = f"*{symbol_filename_part}_lob_*.h5" # Original pattern
        # --- Corrected pattern based on conversion script output ---
        file_pattern = f"{symbol_filename_part}_lob.h5"
        # -------------------------------------------------------
        hdf_files = sorted(data_dir.glob(file_pattern))

        if not hdf_files:
            raise FileNotFoundError(
                f"No HDF5 files matching pattern '{file_pattern}' in {data_dir}")

        logging.info(f"Loading data from HDF5 files: {hdf_files}")
        all_data = []
        for file_path in hdf_files:
            try:
                with pd.HDFStore(file_path, mode='r') as store:
                    # Use HDF_KEY from collector or default
                    hdf_key = get_attr(kwargs, "hdf_key", "lob_data")
                    if f'/{hdf_key}' in store.keys():
                        df_temp = store[hdf_key]
                        all_data.append(df_temp)
                    else:
                        logging.warning(
                            f"Key '{hdf_key}' not found in {file_path}")
            except Exception as e:
                logging.error(f"Error loading data from {file_path}: {e}")

        if not all_data:
            raise ValueError("Failed to load any data from HDF5 files.")

        self.df = pd.concat(all_data, ignore_index=True)
        # Ensure data is sorted by timestamp (use LOB timestamp if available)
        ts_col = 'timestamp' # Use the actual column name from the HDF5 file
        if ts_col not in self.df.columns:
             # Add fallback or error handling if needed
             raise KeyError(f"Required timestamp column '{ts_col}' not found in HDF5 data.")

        self.df = self.df.sort_values(by=ts_col).reset_index(drop=True)

        logging.info(f"Loaded and processed data: {len(self.df)} rows")

        # --- Parameter setup (moved after df loading) ---
        self.transaction_cost_pct = get_attr(
            self.dataset, "transaction_cost_pct", 0.00005
        )
        # Remove tech indicators, use lob_depth instead
        # self.tech_indicator_list = get_attr(
        #     self.dataset, "tech_indicator_list", [])
        # Number of LOB levels to use
        self.lob_depth = get_attr(self.dataset, "lob_depth",
                                  get_attr(kwargs, "lob_depth", 10))
        self.stack_length = get_attr(self.dataset, "backward_num_timestamp", 1)
        self.max_holding_number = get_attr(
            self.dataset, "max_holding_number", 0.01)

        # Removed section loading dynamic csv test data
        # if self.task.startswith("test_dynamic"):
        #     ...
        # else:
        #     ...

        # --- Observation and Action Space ---
        self.action_space = spaces.Discrete( # Use spaces directly (imported from gymnasium)
            get_attr(self.dataset, "num_action", 11))
        self.action_dim = self.action_space.n

        # Define observation space based on LOB data
        lob_features_count = self.lob_depth * 4
        inventory_features_count = 1
        obs_shape = (lob_features_count + inventory_features_count,)
        self.observation_space = spaces.Box( # Use spaces directly
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        self.state_dim = self.observation_space.shape[0]

        # Ensure df is loaded before calling this
        # DP demonstration might need adaptation if it used tech indicators
        # --- Disable computationally expensive DP calculation for now ---
        logging.info("Skipping DP demonstration calculation...")
        # self.demonstration_action = self.making_multi_level_dp_demonstration(
        #     max_punish=get_attr(self.dataset, "max_punish", 1e12)
        # )
        # Initialize with zeros instead
        num_steps = len(self.df)
        self.demonstration_action = np.zeros(num_steps, dtype=int)
        logging.info("Initialized demonstration_action with zeros.")
        # -------------------------------------------------------------

        # reset
        self.terminal = False
        self.day = self.stack_length
        self.data = self.df.iloc[self.day - self.stack_length: self.day]
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.previous_position = 0
        self.position = 0
        self.reward_history = [0]
        self.test_id = 'agent'
        self.position_history = []

    def sell_value(self, price_information, position):
        """Calculates the cash received from selling a position
           by walking the bid side of the LOB.
        Args:
            price_information: A dict/Series representing the current LOB row.
            position: The amount to sell.
        Returns:
            Tuple[float, float]: (cash_received_after_commission, actual_amount_sold)
        """
        original_position = position
        value = 0
        actual_changed_position = 0

        if position <= 0:
            return 0, 0

        for i in range(self.lob_depth):
            try:
                bid_price = price_information.get(f'bid_price_{i}', 0)
                bid_size = price_information.get(f'bid_size_{i}', 0)

                if pd.isna(bid_price) or pd.isna(bid_size) or \
                   bid_price <= 0 or bid_size <= 0:
                    continue # Skip invalid or empty levels

                sell_amount = min(position, bid_size)
                value += sell_amount * bid_price
                position -= sell_amount
                actual_changed_position += sell_amount

                if position <= 1e-9: # Use tolerance for float comparison
                    break # Sold entire target position

            except KeyError as e:
                logging.error(f"Error accessing bid level {i}: {e}")
                break # Stop processing if expected columns are missing

        # if position > 1e-9: # This means not all could be sold
        #    logging.warning(f"Could not sell entire position. Remaining: {position}")

        commission = self.transaction_cost_pct * value
        self.comission_fee_history.append(commission)

        return value * (1 - self.transaction_cost_pct), actual_changed_position

    def buy_value(self, price_information, position):
        """Calculates the cash needed to buy a position
           by walking the ask side of the LOB.
        Args:
            price_information: A dict/Series representing the current LOB row.
            position: The amount to buy.
        Returns:
            Tuple[float, float]: (cash_needed_with_commission, actual_amount_bought)
        """
        original_position = position
        value = 0
        actual_changed_position = 0

        if position <= 0:
            return 0, 0

        for i in range(self.lob_depth):
            try:
                ask_price = price_information.get(f'ask_price_{i}', 0)
                ask_size = price_information.get(f'ask_size_{i}', 0)

                if pd.isna(ask_price) or pd.isna(ask_size) or \
                   ask_price <= 0 or ask_size <= 0:
                    continue # Skip invalid or empty levels

                buy_amount = min(position, ask_size)
                value += buy_amount * ask_price
                position -= buy_amount
                actual_changed_position += buy_amount

                if position <= 1e-9: # Use tolerance
                    break # Bought entire target position

            except KeyError as e:
                logging.error(f"Error accessing ask level {i}: {e}")
                break

        # if position > 1e-9:
        #     logging.warning(f"Could not buy entire position. Remaining: {position}")

        commission = self.transaction_cost_pct * value
        self.comission_fee_history.append(commission)

        return value * (1 + self.transaction_cost_pct), actual_changed_position

    def calculate_value(self, price_information, position):
        """Calculates the current value of a given position based on the
           best bid price available.
        """
        if position <= 0:
            return 0
        # Use best bid price (bid_price_0)
        try:
            bid_price_0 = price_information.get('bid_price_0', 0)
            if pd.isna(bid_price_0) or bid_price_0 <= 0:
                # If no valid best bid, value is uncertain. Using 0.
                return 0
            return bid_price_0 * position
        except KeyError:
            logging.error("Missing 'bid_price_0' for calculate_value.")
            return 0

    def calculate_avaliable_action(self, price_information):
        """Calculates the range of possible discrete actions based on
           available liquidity within the first few LOB levels.
        """
        # Calculate max buy/sell based on available LOB levels (e.g., first 4)
        num_levels_check = min(4, self.lob_depth) # Check up to 4 levels or lob_depth
        buy_size_max = 0
        sell_size_max = 0

        for i in range(num_levels_check):
            try:
                ask_size = price_information.get(f'ask_size_{i}', 0)
                bid_size = price_information.get(f'bid_size_{i}', 0)
                if pd.notna(ask_size) and ask_size > 0:
                    buy_size_max += ask_size
                if pd.notna(bid_size) and bid_size > 0:
                    sell_size_max += bid_size
            except KeyError:
                 logging.warning(f"Missing size column for level {i} in calculate_avaliable_action.")
                 # Continue checking other levels

        position_upper = self.position + buy_size_max
        position_lower = self.position - sell_size_max
        position_lower = max(position_lower, 0)
        position_upper = min(position_upper, self.max_holding_number)

        if self.max_holding_number <= 0:
            scale_factor = 0
        else:
            scale_factor = (self.action_dim - 1) / self.max_holding_number

        current_action = self.position * scale_factor
        # Ensure upper bound calculation is integer and clamped
        action_upper = int(position_upper * scale_factor)
        action_upper = min(action_upper, self.action_dim - 1)

        if position_lower <= 0:
            action_lower = 0
        else:
            # Ensure lower bound calculation is integer and handles edge cases
            action_lower_float = position_lower * scale_factor
            # Use ceiling logic
            action_lower = int(np.ceil(action_lower_float))
            action_lower = min(action_lower, action_upper)
            action_lower = max(action_lower, 0)

        avaliable_discriminator = []
        for i in range(self.action_dim):
            # Ensure comparison is robust (action_lower <= i <= action_upper)
            if action_lower <= i <= action_upper:
                avaliable_discriminator.append(1)
            else:
                avaliable_discriminator.append(0)
        avaliable_discriminator = torch.tensor(avaliable_discriminator)
        return avaliable_discriminator

    def _get_lob_state(self):
        """Extracts the LOB state for the observation.
        Returns a flattened numpy array:
        [ask1_p_norm, ask1_v_log, bid1_p_norm, bid1_v_log, ...,
         ask_depth_p_norm, ..., bid_depth_v_log, inventory_norm]
        Prices are normalized by mid-price, Volumes are log-transformed.
        Inventory is normalized by max_holding_number.
        Handles missing levels with padding (prices=mid, volumes=0).
        """
        state_lob_data = self.data.iloc[-1] # Get latest row

        # Calculate mid-price for normalization from level 0 prices
        try:
            bid_price_0 = state_lob_data['bid_price_0']
            ask_price_0 = state_lob_data['ask_price_0']
            if pd.isna(bid_price_0) or pd.isna(ask_price_0) or \
               ask_price_0 <= 0 or bid_price_0 <= 0:
                mid_price = 0 # Treat as invalid
            else:
                mid_price = (bid_price_0 + ask_price_0) / 2.0
        except KeyError:
             logging.error("Missing bid_price_0 or ask_price_0 for mid-price calculation.")
             mid_price = 0

        lob_state_list = []
        if mid_price <= 0:
            logging.warning(
                f"Day {self.day}: Mid-price is {mid_price}, "
                f"using zero LOB state padding.")
            # Pad with mid-price=1.0, volume=0.0 for all levels
            lob_state_list = [1.0, 0.0, 1.0, 0.0] * self.lob_depth
        else:
            for i in range(self.lob_depth):
                try:
                    # Ask level i
                    ask_price_i = state_lob_data.get(f'ask_price_{i}', mid_price)
                    ask_size_i = state_lob_data.get(f'ask_size_{i}', 0)
                    if pd.isna(ask_price_i) or ask_price_i <= 0:
                        ask_price_norm = 1.0 # Pad with mid
                    else:
                        ask_price_norm = ask_price_i / mid_price
                    ask_vol_log = np.log1p(ask_size_i if pd.notna(ask_size_i) else 0)

                    # Bid level i
                    bid_price_i = state_lob_data.get(f'bid_price_{i}', mid_price)
                    bid_size_i = state_lob_data.get(f'bid_size_{i}', 0)
                    if pd.isna(bid_price_i) or bid_price_i <= 0:
                        bid_price_norm = 1.0 # Pad with mid
                    else:
                        bid_price_norm = bid_price_i / mid_price
                    bid_vol_log = np.log1p(bid_size_i if pd.notna(bid_size_i) else 0)

                except KeyError as e:
                    # This shouldn't happen with .get, but handle defensively
                    logging.error(f"Missing expected LOB column: {e}")
                    ask_price_norm = 1.0
                    ask_vol_log = 0.0
                    bid_price_norm = 1.0
                    bid_vol_log = 0.0

                lob_state_list.extend(
                    [ask_price_norm, ask_vol_log,
                     bid_price_norm, bid_vol_log])

        lob_state = np.array(lob_state_list, dtype=np.float32)

        # Normalize inventory (same as before)
        norm_inventory = 0
        if self.max_holding_number > 0:
            norm_inventory = self.position / self.max_holding_number

        # Combine LOB state and inventory
        full_state = np.concatenate((
            lob_state, np.array([norm_inventory], dtype=np.float32)))

        # Ensure state shape matches observation space (same as before)
        if full_state.shape[0] != self.state_dim:
            # Recalculate expected shape based on current logic
            expected_dim = (self.lob_depth * 4) + 1
            if full_state.shape[0] != expected_dim:
                raise ValueError(
                    f"Constructed state shape {full_state.shape} "
                    f"!= expected {expected_dim}")
            else:
                 # If the dims match expected calculation but not self.state_dim,
                 # update self.state_dim - this indicates an issue during init
                 logging.warning(f"Observation space dim mismatch. Expected {expected_dim}, got {self.state_dim}. Updating self.state_dim.")
                 self.state_dim = expected_dim
                 # It might be better to fix the initial calculation in __init__
                 # self.observation_space needs update too?

        return full_state

    # Add Gymnasium-compatible reset method to the base class
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Reset internal state if any (specific to base class, likely none needed)
        # ...

        # Return a default observation and empty info dictionary
        # The actual observation logic is usually in the subclass (TrainingEnv)
        # Ensure the shape matches the defined observation_space
        # If observation_space is not defined yet, this might need adjustment
        # or be handled post-initialization.
        dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        info = {}
        return dummy_obs, info

    def step(self, action):
        # Base step logic (can raise NotImplementedError or provide basic functionality)
        # This is often overridden by the training environment subclass.
        raise NotImplementedError("Base step method not implemented. Use HighFrequencyTradingTrainingEnvironment.")

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        current_balance = 0
        # Calculate required money properly
        min_balance = 0
        for money in true_money:
            current_balance += money
            balance_list.append(current_balance)
            if current_balance < min_balance:
                min_balance = current_balance
        required_money = -min_balance if min_balance < 0 else 0

        commission_fee = np.sum(self.comission_fee_history)

        # Avoid division by zero for return margin
        if required_money > 0:
            return_margin = final_balance / required_money
        else:
            return_margin = np.inf if final_balance > 0 else 0

        return (
            return_margin,
            final_balance,
            required_money,
            commission_fee,
        )

    def save_asset_memoey(self):
        # Use required_money calculated at the end of the episode
        initial_asset = getattr(self, 'required_money', 0)
        asset_list = [initial_asset]
        current_asset = initial_asset
        # Ensure reward history starts from step 1's reward
        # The initial 0 reward seems to be a placeholder
        relevant_rewards = self.reward_history[1:]  # Skip initial 0 reward
        for reward in relevant_rewards:
            current_asset += reward
            asset_list.append(current_asset)

        # asset_list now has initial_asset + cumulative rewards
        # length should be len(relevant_rewards) + 1
        # df should align rewards with the state *after* the reward was earned
        if len(asset_list) > 1:
            df_value = pd.DataFrame({
                 # Assets at end of step 1, 2, ...
                 'total assets': asset_list[1:],
                 # Reward for step 1, 2, ...
                 'daily_return': relevant_rewards
             })
            # Index might need adjustment depending on how it's used
            # df_value.index = range(1, len(relevant_rewards) + 1)
        else:
            df_value = pd.DataFrame(columns=['total assets', 'daily_return'])

        return df_value

    def get_daily_return_rate(self, price_list: list):
        return_rate_list = []
        if len(price_list) < 2:
            return []
        for i in range(len(price_list) - 1):
            # Avoid division by zero
            if price_list[i] != 0:
                return_rate = (price_list[i + 1] / price_list[i]) - 1
                return_rate_list.append(return_rate)
            else:
                return_rate_list.append(
                    np.inf if price_list[i+1] != 0 else 0)
        return return_rate_list

    def evaualte(self, df):
        if df.empty:
            return 0, 0, 0, 0, 0, 0  # Handle empty df
        daily_return = df["daily_return"]
        neg_ret_lst = daily_return[daily_return < 0]
        assets = df["total assets"]
        # Infer initial from first step
        initial_assets = assets.iloc[0] - daily_return.iloc[0] \
            if len(assets) > 0 else 0
        # Avoid zero/negative initial
        initial_assets = initial_assets if initial_assets > 1e-10 else 1e-10
        final_assets = assets.iloc[-1] if len(assets) > 0 else initial_assets
        tr = final_assets / initial_assets - 1

        return_rate_list = self.get_daily_return_rate(assets.values)
        if not return_rate_list:
            return tr, 0, 0, 0, 0, 0  # Handle no returns

        mean_return = np.mean(return_rate_list)
        std_dev = np.std(return_rate_list)
        sharpe_ratio = mean_return * (31536000 ** 0.5) / (std_dev + 1e-10)
        vol = std_dev
        mdd = 0
        peak = initial_assets  # Start peak at initial inferred asset level
        for value in assets:
            if value > peak:
                peak = value
            # Avoid division by zero if peak is 0
            # (shouldn't happen with initial_assets > 0)
            dd = (peak - value) / peak if peak != 0 else 0
            if dd > mdd:
                mdd = dd
        cr = tr / (mdd + 1e-10)  # Use total return 'tr' here
        downside_std = np.std(neg_ret_lst) if not neg_ret_lst.empty else 0
        # Use mean_return for Sortino numerator consistent with Sharpe
        sor = mean_return * (31536000 ** 0.5) / (downside_std + 1e-10)
        # Original Sortino calculation was different, using mean_return now:
        # downside_std_orig = np.nan_to_num(np.std(neg_ret_lst), nan=0.0)
        # sqrt_len_return = np.sqrt(len(daily_return))
        # sor_orig = np.sum(daily_return) / downside_std_orig / sqrt_len_return
        return tr, sharpe_ratio, vol, mdd, cr, sor

    # Remove redundant definition of get_final_return_rate

    def get_final_return(self):
        return np.sum(self.reward_history[1:])  # Sum relevant rewards

    def check_sell_needed(self, sell_list, buy_list):
        if len(sell_list) != len(buy_list):
            raise Exception("the dimension is not correct")
        else:
            in_out_list = []
            for i in range(len(sell_list)):
                if sell_list[i] != 0 and buy_list[i] != 0:
                    raise Exception(
                        "there is time when money both come in and out")
                # This elif is redundant because of the first if
                # elif buy_list[i] != 0 and sell_list[i] != 0:
                #     raise Exception(
                #         "there is time when money both come in and out")
                else:
                    in_out_list.append(sell_list[i] - buy_list[i])
            balance_list = []
            current_balance = 0
            for flow in in_out_list:
                current_balance += flow
                balance_list.append(current_balance)
            # print("the money we require is", -min(balance_list))
        return balance_list

    def making_multi_level_dp_demonstration(self, max_punish=1e12):
        action_list = []

        # Define internal helpers using LOB flat format
        def sell_value_dp(price_information, position):
            value = 0
            actual_changed_position = 0
            if position <= 0: return -max_punish # Penalize selling nothing

            for i in range(self.lob_depth):
                try:
                    bid_price = price_information.get(f'bid_price_{i}', 0)
                    bid_size = price_information.get(f'bid_size_{i}', 0)
                    if pd.isna(bid_price) or pd.isna(bid_size) or \
                       bid_price <= 0 or bid_size <= 0:
                        continue

                    sell_amount = min(position, bid_size)
                    value += sell_amount * bid_price
                    position -= sell_amount
                    actual_changed_position += sell_amount

                    if position <= 1e-9: break
                except KeyError: break

            if position > 1e-9: # Penalize if not fully sold
                value = -max_punish

            if value > -max_punish / 2:
                return value * (1 - self.transaction_cost_pct)
            else:
                return value

        def buy_value_dp(price_information, position):
            value = 0
            actual_changed_position = 0
            if position <= 0: return max_punish # Penalize buying nothing?

            for i in range(self.lob_depth):
                try:
                    ask_price = price_information.get(f'ask_price_{i}', 0)
                    ask_size = price_information.get(f'ask_size_{i}', 0)
                    if pd.isna(ask_price) or pd.isna(ask_size) or \
                       ask_price <= 0 or ask_size <= 0:
                        continue

                    buy_amount = min(position, ask_size)
                    value += buy_amount * ask_price
                    position -= buy_amount
                    actual_changed_position += buy_amount

                    if position <= 1e-9: break
                except KeyError: break

            if position > 1e-9: # Penalize if not fully bought
                value = max_punish

            if value < max_punish / 2:
                return value * (1 + self.transaction_cost_pct)
            else:
                return value

        # --- DP calculation (no changes needed here, uses helpers) ---
        if self.action_dim <= 1:
            return []
        scale_factor = self.action_dim - 1
        num_steps = len(self.df)
        if num_steps == 0:
            return []

        dp = [[-np.inf] * self.action_dim for _ in range(num_steps)]
        price_information = self.df.iloc[0]
        for j in range(self.action_dim):
            pos_j = j / scale_factor * self.max_holding_number
            if j == 0:
                dp[0][j] = 0
            else:
                dp[0][j] = -buy_value_dp(price_information, pos_j)

        for i in range(1, num_steps):
            price_information = self.df.iloc[i]
            for j in range(self.action_dim):
                max_prev_value = -np.inf
                for k in range(self.action_dim):
                    if dp[i - 1][k] == -np.inf:
                        continue
                    pos_k = k / scale_factor * self.max_holding_number
                    pos_j = j / scale_factor * self.max_holding_number
                    position_changed = pos_j - pos_k
                    transition_pnl = 0
                    if position_changed > 0:
                        transition_pnl = -buy_value_dp(
                            price_information, position_changed)
                    elif position_changed < 0:
                        transition_pnl = sell_value_dp(
                            price_information, -position_changed)
                    current_value = dp[i - 1][k] + transition_pnl
                    if current_value > max_prev_value:
                        max_prev_value = current_value
                dp[i][j] = max_prev_value

        # --- Backtracking (no changes needed here, uses helpers) ---
        action_list_reversed = []
        last_action = 0
        max_final_val = -np.inf
        if num_steps > 1:
            price_information = self.df.iloc[num_steps - 1]
            for k in range(self.action_dim):
                if dp[num_steps - 2][k] == -np.inf:
                    continue
                pos_k = k / scale_factor * self.max_holding_number
                position_changed = 0 - pos_k
                transition_pnl = 0
                if position_changed < 0:
                    transition_pnl = sell_value_dp(
                        price_information, -position_changed)
                current_final_value = dp[num_steps - 2][k] + transition_pnl
                if current_final_value > max_final_val:
                    max_final_val = current_final_value
                    last_action = k
            if max_final_val > -np.inf:
                action_list_reversed.append(last_action)
                last_value = dp[num_steps - 2][last_action]
                for i in range(num_steps - 2, 0, -1):
                    price_information = self.df.iloc[i]
                    current_action = 0
                    found_prev = False
                    for k in range(self.action_dim):
                        if dp[i - 1][k] == -np.inf:
                            continue
                        pos_k = k / scale_factor * self.max_holding_number
                        pos_last = last_action / scale_factor \
                            * self.max_holding_number
                        position_changed = pos_last - pos_k
                        transition_pnl = 0
                        if position_changed > 0:
                            transition_pnl = -buy_value_dp(
                                price_information, position_changed)
                        elif position_changed < 0:
                            transition_pnl = sell_value_dp(
                                price_information, -position_changed)
                        if abs((dp[i - 1][k] + transition_pnl)
                                 - last_value) < 1e-9:
                            current_action = k
                            found_prev = True
                            break
                    if not found_prev:
                        print(f"Warning: DP backtrace failed step {i}")
                        break
                    action_list_reversed.append(current_action)
                    last_action = current_action
                    last_value = dp[i - 1][last_action]

        action_list = action_list_reversed[::-1]
        if len(action_list) < num_steps:
            action_list.extend([0] * (num_steps - len(action_list)))
        return action_list

    # Add stub methods if they don't exist
    def render(self, mode='human'):
        # HFT environments usually don't have a visual representation
        pass

    def close(self):
        # Clean up any resources if needed
        pass

    # seed method is usually handled by wrappers or SB3 itself
    # def seed(self, seed=None):
    #     pass


# @ENVIRONMENTS.register_module() # Commented out: Need builder.py
# Uncomment decorator now that builder/custom should be available
@ENVIRONMENTS.register_module()
class HighFrequencyTradingTrainingEnvironment(HighFrequencyTradingEnvironment):
    def __init__(self, **kwargs):
        # Initialize base class first to load data and setup basic params
        super().__init__(**kwargs)  # CALL SUPER __init__

        # Get training-specific parameters
        self.episode_length = get_attr(self.dataset, "episode_length", 14400)
        # Remove forced episode termination
        # self.max_steps_per_episode = 1000  # Force episodes to end after 1000 steps
        self.current_episode_steps = 0
        
        self.i = 0  # Initialize episode start index

        # The base class __init__ already loaded the df, set up spaces,
        # calculated demonstration, etc. We don't need to repeat that here.
        # Remove redundant/incorrect setup from the original TrainingEnv __init__:
        # self.df_path = None
        # ... (CSV loading logic removed) ...
        # self.transaction_cost_pct = ... (already set in super)
        # self.tech_indicator_list = ... (already removed in super)
        # self.stack_length = ... (already set in super)
        # self.max_holding_number = ... (already set in super)
        # ... (dynamic test path logic removed) ...
        # self.action_space = ... (already set in super)
        # self.observation_space = ... (already set in super based on LOB)
        # self.demonstration_action = ... (already set in super)

        # Reset internal state specific to training episodes
        # (already done in base __init__, reset will handle episode start)
        # self.terminal = False
        # self.day = self.stack_length
        # ... (rest of internal state vars initialized in base or reset) ...

        logging.info("HighFrequencyTradingTrainingEnvironment initialized.")
        # No return needed in __init__

    # Update signature to match Gymnasium standard
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Handle seeding
        super().reset(seed=seed) # Call parent Env reset if needed
        # Seed the numpy random generator used for start index sampling
        if seed is not None:
            np.random.seed(seed)
        
        # Reset step counter for new episode
        self.current_episode_steps = 0
        logging.info(f"Reset called: Starting new episode from day {self.day}")
        
        # --- Implement random episode start --- 
        max_start_index = len(self.df) - self.episode_length - self.stack_length
        if max_start_index <= 0:
            self.i = 0 # Default to start if dataset is too short
            if len(self.df) < self.stack_length + self.episode_length:
                 logging.warning(f"Dataset length ({len(self.df)}) is less than stack+episode length ({self.stack_length + self.episode_length}). Episode may terminate early.")
        else:
            # Randomly sample a starting index
            self.i = np.random.randint(0, max_start_index + 1)
        # -------------------------------------

        # Original logic for bounds check (might be redundant now but safe)
        # max_df_index = len(self.df) - 1
        # if self.i + self.stack_length > max_df_index + 1:
        #      self.i = max(0, max_df_index + 1 - self.stack_length)
        #      logging.warning(
        #          f"Adjusted random start index {self.i}")

        self.terminal = False
        # Set day relative to the randomly chosen episode start index 'i'
        self.day = self.i + self.stack_length
        # print(f"[{os.getpid()}] TrainingEnv reset: Set day = {self.day}")

        # Slice data for the initial state
        # Ensure slicing is within bounds, especially if dataset was short
        start_slice = max(0, self.day - self.stack_length)
        end_slice = max(start_slice, self.day)
        if end_slice > len(self.df):
             logging.error(f"Reset slicing error: end_slice {end_slice} > df length {len(self.df)}. Start index {self.i}")
             # Handle error appropriately, maybe reset i to 0 and retry slice
             self.i = 0
             self.day = self.i + self.stack_length
             start_slice = max(0, self.day - self.stack_length)
             end_slice = max(start_slice, self.day)
        if end_slice > len(self.df):
            raise IndexError("Cannot start episode, data too short for stack_length.")

        self.data = self.df.iloc[start_slice:end_slice]
        # print(f"[{os.getpid()}] TrainingEnv reset: Sliced self.data (shape {self.data.shape}) for day {self.day}")

        # Ensure self.data is not empty after slicing
        if self.data.empty:
            raise RuntimeError(f"Failed to get valid data slice during reset. Start: {start_slice}, End: {end_slice}, Day: {self.day}, i: {self.i}")

        # Reset internal state variables (same as base reset)
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = 0
        self.position = 0
        self.previous_position = 0
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.position_history = []  # Reset position history

        # Calculate initial available actions and DP info (DP is currently zeros)
        # Get info from last row of initial data slice
        price_information = self.data.iloc[-1]
        avaliable_discriminator = self.calculate_avaliable_action(
            price_information)
        DP_distribution = [0] * self.action_dim # Use self.action_dim
        # Index relative to overall df for demonstration
        dp_day_index = self.day - 1 
        if dp_day_index >= 0 and \
           dp_day_index < len(self.demonstration_action):
            action_index = self.demonstration_action[dp_day_index]
            if 0 <= action_index < len(DP_distribution):
                DP_distribution[action_index] = 1
            # No warning needed as DP is zeros anyway
        # else:
        #     logging.warning(
        #         f"Day index {dp_day_index} out of bounds for "
        #         f"demonstration_action during training reset.")
        DP_distribution = np.array(DP_distribution)

        # Calculate the initial state using the LOB helper
        try:
            self.state = self._get_lob_state()
            # print(f"[{os.getpid()}] TrainingEnv reset: Got LOB state (shape {self.state.shape})")
        except Exception as e:
            logging.error(f"Error getting LOB state during reset: {e}", exc_info=True)
            raise

        # Construct the info dictionary (must be returned now)
        info = {
            # Optionally include initial available actions, etc.
            # "available_action": avaliable_discriminator,
            # "DP_action": DP_distribution
        }

        # print(f"[{os.getpid()}] TrainingEnv reset finished. Returning state.")
        # Return observation and info tuple
        return self.state, info

    def step(self, action):
        # Increment step counter
        self.current_episode_steps += 1
        
        # --- Calculate target position (same as base class) ---
        if self.action_dim <= 1:
            normlized_action = 0
        else:
            normlized_action = action / (self.action_dim - 1)
        target_position = self.max_holding_number * normlized_action

        # --- Advance day counter --- 
        self.day += 1

        # --- Check termination conditions ---
        end_of_data = self.day >= len(self.df) # Simpler condition
        end_of_episode = self.day >= self.i + self.stack_length + self.episode_length # Correct condition
        
        # Remove forced episode termination
        # force_truncate = self.current_episode_steps >= self.max_steps_per_episode
        
        # Gymnasium standard: terminated=True if natural end, truncated=True if time limit
        terminated = end_of_data
        truncated = end_of_episode and not terminated
        # if force_truncate:
        #     logging.info(f"FORCED episode truncation after {self.current_episode_steps} steps")
        
        # Old flag kept for potential internal use
        self.terminal = terminated or truncated
        
        # --- Add log message when end of entire dataset is reached ---
        if terminated and not truncated: # Only log if natural end
             logging.info(
                 f"Episode {self.i}: Reached end of entire dataset at day index {self.day-1}. Terminating.")
        elif truncated: # Log truncation separately
             logging.info(
                 f"Episode {self.i}: Reached episode length limit at day index {self.day-1}. Truncating.")
        # -------------------------------------------------------------

        # --- Get previous state info ---
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]

        # --- Get current state info ---        
        if self.day >= len(self.df.index.unique()): # Re-check bounds using day *after* increment
            # Use previous info if we are truly at the end after increment
            current_price_information = previous_price_information            
        else:
            # Slice data for the *current* day to get its LOB info
            self.data = self.df.iloc[self.day - self.stack_length : self.day] # Should use updated day
            current_price_information = self.data.iloc[-1]
            # The state calculation happens *after* execution

        # === Execution Logic (same as base class) ===
        # We can reuse the base class implementation for sell/buy/calculate_value
        cash = 0
        needed_cash = 0
        actual_position_change = 0
        executed_position = self.previous_position

        if previous_position >= target_position:  # Sell or Hold
            sell_size = previous_position - target_position
            if sell_size > 0:
                cash, actual_position_change = self.sell_value(
                    previous_price_information, sell_size
                )
                self.sell_money_memory.append(cash)
                self.needed_money_memory.append(0)
                executed_position = \
                    self.previous_position - actual_position_change
            else:  # Holding
                self.sell_money_memory.append(0)
                self.needed_money_memory.append(0)
                actual_position_change = 0  # Explicitly 0 change
        elif previous_position < target_position:  # Buy
            buy_size = target_position - previous_position
            if buy_size > 0:
                needed_cash, actual_position_change = self.buy_value(
                    previous_price_information, buy_size
                )
                self.needed_money_memory.append(needed_cash)
                self.sell_money_memory.append(0)
                executed_position = \
                    self.previous_position + actual_position_change
            # Note: Need else case? If buy_size is 0, no execution needed.
            # Assuming actual_position_change is 0 if buy_size is 0.

        # === Update position and calculate reward (same as base class) ===
        self.position = executed_position
        previous_value = self.calculate_value(
            previous_price_information, self.previous_position
        )
        current_value = self.calculate_value(
            current_price_information, self.position
        )
        # Using the simpler reward calculation from base class
        self.reward = (current_value + cash) - (previous_value + needed_cash)
        self.reward_history.append(self.reward)
        # print(f"[{os.getpid()}] TrainingEnv step: Reward calculated: {self.reward}")

        # --- Update internal state for next step ---
        # For the next step's calculation
        self.previous_position = self.position
        self.position_history.append(self.position)

        # --- Calculate NEXT state using LOB data ---
        if not self.terminal: # Only get next state if not terminal
            # print(f"[{os.getpid()}] TrainingEnv step: Getting next state for day {self.day}")
            try:
                self.state = self._get_lob_state()
                # print(f"[{os.getpid()}] TrainingEnv step: Next state shape: {self.state.shape}")
            except Exception as e:
                logging.error(f"Error getting LOB state during step {self.day}: {e}", exc_info=True)
                # Handle error: maybe return last valid state or zeros?
                # Or re-raise the exception
                raise
        else:
            # If terminal, the state from the previous step is usually returned
            # Or sometimes a zero state, depending on library/convention
             # print(f"[{os.getpid()}] TrainingEnv step: Terminal state reached. Not calculating next state.")
            # Ensure self.state exists and has the correct shape if needed by wrapper
             if not hasattr(self, 'state') or self.state is None:
                 self.state = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)


        # --- Calculate info dictionary (same as base class) ---
        avaliable_discriminator = self.calculate_avaliable_action(current_price_information)
        DP_distribution = [0] * 11
        dp_day_index = self.day - 1  # Index relative to overall df
        if dp_day_index >= 0 and \
           dp_day_index < len(self.demonstration_action):
            action_index = self.demonstration_action[dp_day_index]
            if 0 <= action_index < len(DP_distribution):
                DP_distribution[action_index] = 1
            else:
                logging.warning(
                    f"DP action index {action_index} out of bounds "
                    f"during training step.")
        else:
            # Don't warn if just past end of demonstration data
            pass
            # logging.warning(f"Day index {dp_day_index} out of bounds for "
            # f"demonstration_action during training step.")
        DP_distribution = np.array(DP_distribution)
        info = {
            "previous_action": action,
            "avaliable_action": avaliable_discriminator,
            "DP_action": DP_distribution, 
        }

        # Log episode completion more visibly
        if terminated or truncated:
            logging.info(f"Episode ended: terminated={terminated}, truncated={truncated}, steps={self.current_episode_steps}, day={self.day}")
        
        # More verbose debug
        if self.current_episode_steps % 250 == 0 or terminated or truncated:
            logging.info(f"Step {self.current_episode_steps}: day={self.day}, reward={self.reward:.4f}, terminated={terminated}, truncated={truncated}")
        
        # --- Final Return ---
        # Return observation, reward, terminated, truncated, info
        # print(f"[{os.getpid()}] TrainingEnv step finished. Returning state, reward: {self.reward}, terminated: {terminated}, truncated: {truncated}")
        logging.debug(f"[{os.getpid()}] Step Return: Day={self.day}, R={self.reward:.4f}, Term={terminated}, Trunc={truncated}, StateType={type(self.state)}, StateShape={self.state.shape if isinstance(self.state, np.ndarray) else 'N/A'}")
        return self.state, self.reward, terminated, truncated, info
