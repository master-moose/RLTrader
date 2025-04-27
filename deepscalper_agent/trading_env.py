"""
Custom Gymnasium environment for simulating a market-making trading strategy
based on Limit Order Book (LOB) data, inspired by the DeepScalper paper.
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingEnv(gym.Env):
    """A Gymnasium environment for simulating LOB-based market making.

    Args:
        data_dir (str or Path): Directory containing the HDF5 LOB data files.
        symbol (str): Trading symbol (e.g., 'BTC/USDT'), used for file naming convention.
        initial_cash (float): Starting cash balance for the agent.
        commission_pct (float): Transaction fee percentage.
        lob_depth (int): Number of LOB levels to include in the observation.
        max_inventory (float): Maximum absolute inventory allowed.
        # Add other parameters like reward shaping components, episode length etc.
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        data_dir: str | Path = 'data/lob_data',
        symbol: str = 'BTC/USDT',
        initial_cash: float = 10000.0,
        commission_pct: float = 0.001, # Example commission (0.1%)
        lob_depth: int = 10, # Use top 10 levels for state
        max_inventory: float = 1.0, # Max 1 BTC long or short
        max_steps: Optional[int] = None # Max steps per episode
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.symbol_filename = symbol.replace('/', '')
        self.initial_cash = initial_cash
        self.commission_pct = commission_pct
        self.lob_depth = lob_depth
        self.max_inventory = max_inventory
        self.max_steps = max_steps

        # --- Load Data ---
        self.data: pd.DataFrame = self._load_data()
        if self.data.empty:
            raise ValueError(f"No LOB data found in {self.data_dir} for symbol {symbol}")
        self.n_steps = len(self.data)
        if self.max_steps is None:
            self.max_steps = self.n_steps - 1 # Default to full dataset length

        # --- Define Spaces ---
        # Action Space: Placeholder - needs refinement based on strategy
        # Example: Discrete action (e.g., 0: hold, 1: post bid/ask near spread, 2: post wider)
        self.action_space = spaces.Discrete(3) # TODO: Refine action space

        # Observation Space: Placeholder - depends heavily on state representation
        # Example: Box space including LOB levels, inventory, cash
        # Shape: (lob_depth * 4 [bid_price, bid_vol, ask_price, ask_vol] + 2 [inventory, cash])
        obs_shape = (self.lob_depth * 4 + 1,) # Simplified: +1 for inventory
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        # TODO: Refine observation space (normalize?, include cash?, time features?)

        # --- Internal State ---
        self._current_step = 0
        self._cash = self.initial_cash
        self._inventory = 0.0 # Base asset (e.g., BTC)
        self._total_value = self.initial_cash
        self._last_observation = None
        self._active_orders = {'buy': None, 'sell': None} # Example: {price, size}

        logging.info(f"TradingEnv initialized with {self.n_steps} data points.")

    def _load_data(self) -> pd.DataFrame:
        """Loads LOB data from HDF5 files in the specified directory."""
        all_data = []
        file_pattern = f"*{self.symbol_filename}_lob_*.h5"
        hdf_files = sorted(self.data_dir.glob(file_pattern))

        if not hdf_files:
            logging.warning(f"No HDF5 files found matching pattern '{file_pattern}' in {self.data_dir}")
            return pd.DataFrame()

        logging.info(f"Found data files: {hdf_files}")
        for file_path in hdf_files:
            try:
                with pd.HDFStore(file_path, mode='r') as store:
                    # Assuming data is stored under key 'lob_data' (matches collector)
                    if '/lob_data' in store.keys():
                        df = store['lob_data']
                        # Deserialize JSON strings back to lists
                        df['bids'] = df['bids'].apply(json.loads)
                        df['asks'] = df['asks'].apply(json.loads)
                        all_data.append(df)
                    else:
                        logging.warning(f"Key 'lob_data' not found in {file_path}")
            except Exception as e:
                logging.error(f"Error loading data from {file_path}: {e}")

        if not all_data:
            return pd.DataFrame()

        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.sort_values(by='timestamp_utc').reset_index(drop=True)
        logging.info(f"Loaded and concatenated data: {full_df.shape[0]} rows")
        return full_df

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation array from the current state."""
        # TODO: Implement actual observation construction based on self.data[self._current_step]
        # and self._inventory, self._cash.
        # This needs to match self.observation_space shape and type.
        # Example placeholder:
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        obs[-1] = self._inventory # Put inventory in the last slot
        return obs

    def _get_reward(self, previous_value: float) -> float:
        """Calculates the reward for the current step."""
        # TODO: Implement reward calculation (e.g., change in portfolio value)
        return self._total_value - previous_value

    def _calculate_current_value(self) -> float:
        """Calculates the current portfolio value (cash + inventory value)."""
        # Need a way to price inventory - e.g., mid-price at current step
        # TODO: Implement inventory valuation
        current_row = self.data.iloc[self._current_step]
        bids = current_row['bids']
        asks = current_row['asks']
        if not bids or not asks:
             mid_price = 0 # Handle case with empty book side
        else:
             mid_price = (bids[0][0] + asks[0][0]) / 2 # Price of top bid/ask

        inventory_value = self._inventory * mid_price
        return self._cash + inventory_value

    def _simulate_trades(self, action: Any):
        """Simulates order placement, cancellation, and fills based on the action
           and the LOB state at the *next* time step."""
        # This is the most complex part. Needs careful logic:
        # 1. Based on `action`, decide which orders to place/cancel.
        # 2. Look at self.data[self._current_step + 1] LOB state.
        # 3. Check if active buy orders cross the *next* step's asks.
        # 4. Check if active sell orders cross the *next* step's bids.
        # 5. Update self._cash and self._inventory based on fills, subtracting commission.
        # 6. Update self._active_orders.
        pass # TODO: Implement trade simulation logic

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""
        terminated = False
        truncated = False
        info = {}

        # Store previous portfolio value for reward calculation
        previous_value = self._total_value

        # --- Simulate market interactions based on action ---
        # (This might involve placing/canceling orders, checking for fills)
        self._simulate_trades(action)

        # --- Advance time ---
        self._current_step += 1

        # --- Update total value ---
        self._total_value = self._calculate_current_value()

        # --- Calculate reward ---
        reward = self._get_reward(previous_value)

        # --- Get next observation ---
        observation = self._get_observation()
        self._last_observation = observation # Store for rendering

        # --- Check for termination/truncation conditions ---
        if self._total_value <= 0: # Ruin
            terminated = True
            logging.warning(f"Episode terminated at step {self._current_step}: Agent ruined.")

        if self._current_step >= self.n_steps - 1 or self._current_step >= self.max_steps:
            truncated = True # End of data or max steps reached
            logging.info(f"Episode truncated at step {self._current_step}.")

        # Optional: Add info dictionary content
        info['step'] = self._current_step
        info['cash'] = self._cash
        info['inventory'] = self._inventory
        info['total_value'] = self._total_value
        # info['active_orders'] = self._active_orders

        return observation, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        self._current_step = 0
        self._cash = self.initial_cash
        self._inventory = 0.0
        self._total_value = self.initial_cash
        self._active_orders = {'buy': None, 'sell': None}

        observation = self._get_observation()
        self._last_observation = observation
        info = {
            'step': self._current_step,
            'cash': self._cash,
            'inventory': self._inventory,
            'total_value': self._total_value
        }

        logging.info("Environment reset.")
        return observation, info

    def render(self, mode="human") -> Optional[str]:
        """Renders the environment.

        (Currently prints basic info or returns it as a string).
        """
        output = (
            f"Step: {self._current_step}/{self.n_steps}\n"
            f"Cash: {self._cash:.2f}\n"
            f"Inventory: {self._inventory:.6f}\n"
            f"Total Value: {self._total_value:.2f}\n"
            # f"Observation: {self._last_observation}\n"
            # f"Active Orders: {self._active_orders}\n"
        )
        if mode == "human":
            print(output)
            return None
        elif mode == "ansi":
            return output
        else:
            super().render(mode=mode) # Raises error for unsupported modes

    def close(self):
        """Perform any necessary cleanup."""
        logging.info("Closing TradingEnv.")
        # No explicit resources to close in this basic version
        pass

# Example Usage (for testing the environment structure)
if __name__ == '__main__':
    print("Testing TradingEnv structure...")

    # Create dummy data for testing if real data isn't present
    dummy_data_dir = Path('data/dummy_lob')
    dummy_data_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = dummy_data_dir / 'binance_BTCUSDT_lob_20240101.h5'

    if not dummy_file.exists():
        print("Creating dummy HDF5 data for testing...")
        dummy_steps = 100
        dummy_bids = [[50000.0 - i*0.5, 0.1+i*0.01] for i in range(10)]
        dummy_asks = [[50000.5 + i*0.5, 0.1+i*0.01] for i in range(10)]
        timestamps = pd.to_datetime(np.arange(dummy_steps), unit='s',
                                    origin=pd.Timestamp('2024-01-01'))
        dummy_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'lob_timestamp_ms': [ts.timestamp() * 1000 for ts in timestamps],
            'lob_nonce': np.arange(dummy_steps),
            'bids': [json.dumps(dummy_bids)] * dummy_steps,
            'asks': [json.dumps(dummy_asks)] * dummy_steps
        })
        with pd.HDFStore(dummy_file, mode='w') as store:
            store.put('lob_data', dummy_df, format='table', data_columns=['timestamp_utc'])
        print(f"Dummy data created at {dummy_file}")

    try:
        env = TradingEnv(data_dir=dummy_data_dir, symbol='BTC/USDT')
        print("Environment created successfully.")
        obs, info = env.reset()
        print(f"Reset successful. Initial observation shape: {obs.shape}, Info: {info}")
        # Test a step
        action = env.action_space.sample() # Sample a random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step successful. Action: {action}, Next Obs Shape: {obs.shape}, Reward: {reward:.4f}, Term: {terminated}, Trunc: {truncated}, Info: {info}")
        env.render(mode="human")
        env.close()
        print("Environment closed.")

    except Exception as e:
        print(f"Error during environment testing: {e}")
        import traceback
        traceback.print_exc()

    # Clean up dummy data
    # print(f"Cleaning up dummy data file: {dummy_file}")
    # dummy_file.unlink()
    # try:
    #     dummy_data_dir.rmdir()
    # except OSError: # Directory might not be empty if run failed
    #     pass 