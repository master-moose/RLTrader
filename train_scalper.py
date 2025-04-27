# train_scalper.py
# import os # Unused
import logging
from pathlib import Path
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Type
import gym # Needed for custom policy/extractor typing

from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env # Unused
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecCheckNan, VecFrameStack
)
# Import Monitor wrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# No longer need BaseModel import
from stable_baselines3.dqn.policies import DQNPolicy

# Use the Training Environment
from deepscalper_agent.trading_env \
    import HighFrequencyTradingTrainingEnvironment

# --- Configuration ---
# Data and Environment
# Match these with your lob_collector.py output and environment settings
DATA_DIRECTORY = Path("data/lob_data")
# e.g., "binance_BTCUSDT"
SYMBOL_FILENAME_PART = "binance_BTCUSDT"
HDF_KEY = "lob_data"  # Key used in HDF5 files
LOB_DEPTH = 10
MAX_HOLDING = 0.01
ACTION_DIM = 11
# Example, adjust if needed
TRANSACTION_COST_PCT = 0.00005
# Number of steps per episode during training (can be long for DQN)
EPISODE_LENGTH = 14400
# Timesteps to stack for LSTM input
N_STACK = 10  # New: Number of frames to stack for LSTM

# Training Settings
TOTAL_TIMESTEPS = 1_000_000  # Adjust as needed
# Number of parallel environments (adjust based on CPU cores)
# DQN might need more memory/CPU per env, consider lowering if needed
N_ENVS = 4
SEED = 42  # For reproducibility
LEARNING_RATE = 1e-4  # DQN often uses smaller LR than PPO
BUFFER_SIZE = 100_000  # New: Replay buffer size for DQN
LEARNING_STARTS = 50_000  # New: Steps before learning starts
BATCH_SIZE = 32 * N_ENVS  # DQN default is 32 per env
TAU = 1.0  # New: Target network update rate (1.0 = hard update)
GAMMA = 0.99  # New: Discount factor (common default)
TARGET_UPDATE_INTERVAL = 1000  # New: Steps between target net updates
TRAIN_FREQ = 4  # New: Update model every N steps
GRADIENT_STEPS = 1  # New: How many gradient steps per update
# Fraction of training for exploration ramp-down
EXPLORATION_FRACTION = 0.1
# Final epsilon value for exploration
EXPLORATION_FINAL_EPS = 0.05
# Remove PPO-specific param: N_STEPS = 2048 // N_ENVS
DEVICE = "cuda"  # Use "cuda" if GPU is available, otherwise "cpu"
# Custom Policy/Network parameters
LSTM_HIDDEN_SIZE = 64  # Size of LSTM hidden layer
FEATURES_DIM = 64  # Output size of feature extractor

# Logging and Saving
LOG_DIR = Path("logs/dqn_lstm_scalper")  # Updated log dir name
SAVE_DIR = Path("models/dqn_lstm_scalper")  # Updated save dir name
# Save checkpoint every N steps (adjust based on buffer size/update freq)
# Save less frequently initially? Depends on training speed.
SAVE_FREQ = 50_000
MODEL_SAVE_NAME = "dqn_lstm_scalper_final"  # Updated model name

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check CUDA availability
IS_CUDA_AVAILABLE = torch.cuda.is_available()
logging.info(f"CUDA Available: {IS_CUDA_AVAILABLE}")
if DEVICE == "cuda" and not IS_CUDA_AVAILABLE:
    logging.warning("DEVICE is set to 'cuda' but CUDA is not available. "
                    "Training will use CPU.")
    # Optionally force DEVICE to cpu if CUDA is requested but unavailable
    # DEVICE = "cpu"


# --- Custom LSTM Network and Policy ---
class LstmFeatureExtractor(BaseFeaturesExtractor):
    """LSTM Feature Extractor for DQN.

    Takes stacked frames (batch, sequence, features) and extracts features.
    :param observation_space: The observation space (after VecFrameStack).
    :param features_dim: Number of features extracted.
    :param n_stack: Number of frames stacked by VecFrameStack.
    :param lstm_hidden_size: Size of the LSTM hidden state.
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = FEATURES_DIM,
                 n_stack: int = N_STACK,
                 lstm_hidden_size: int = LSTM_HIDDEN_SIZE):
        # Ensure features_dim matches policy head input size
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack

        # Calculate feature size per step (original observation dim)
        # Input shape is (batch_size, n_stack * features_per_step)
        assert observation_space.shape[0] % n_stack == 0, \
            "Observation space dim must be divisible by n_stack"
        self.features_per_step = observation_space.shape[0] // n_stack

        self.lstm = nn.LSTM(self.features_per_step, lstm_hidden_size,
                            batch_first=True)
        # Output layer to map LSTM output to features_dim
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, n_stack * features_per_step)
        batch_size = observations.shape[0]
        # Reshape to (batch_size, n_stack, features_per_step) for LSTM
        obs_reshaped = observations.reshape(batch_size, self.n_stack,
                                            self.features_per_step)

        # LSTM output: (output, (h_n, c_n))
        # We only need the output of the last time step
        lstm_out, _ = self.lstm(obs_reshaped)
        # Output shape: (batch_size, n_stack, lstm_hidden_size)

        # Take output of the last time step (batch_size, lstm_hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        features = self.relu(self.linear(last_time_step_out))
        return features


class LstmDqnPolicy(DQNPolicy):
    """DQN Policy with LSTM Feature Extractor."""
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule,
                 # Default MLP layers in Q net head
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor]
                 = LstmFeatureExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):

        # Default features_extractor_kwargs if none provided
        # Pass n_stack from config to the extractor
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(features_dim=FEATURES_DIM,
                                             lstm_hidden_size=LSTM_HIDDEN_SIZE,
                                             n_stack=N_STACK)
        # Ensure n_stack is passed if kwargs are partially provided
        elif 'n_stack' not in features_extractor_kwargs:
            features_extractor_kwargs['n_stack'] = N_STACK

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

# -----------------------------------------

def create_env(rank, seed=0):
    """Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) number of parallel environments
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # Note: The env __init__ uses a dataset dict. We mimic that here.
        env = HighFrequencyTradingTrainingEnvironment(
            data_dir=DATA_DIRECTORY,
            symbol_filename_part=SYMBOL_FILENAME_PART,
            hdf_key=HDF_KEY,
            dataset={
                "lob_depth": LOB_DEPTH,
                "max_holding_number": MAX_HOLDING,
                "num_action": ACTION_DIM,
                "transaction_cost_pct": TRANSACTION_COST_PCT,
                # backward_num_timestamp needs to be >= 1 for env slicing
                # but the actual sequence length is handled by VecFrameStack
                "backward_num_timestamp": 1,  # Keep this >= 1
                "episode_length": EPISODE_LENGTH
                # Add other required dataset params if any
            }
            # Seed is handled by SB3 wrapper normally, but env might use it?
        )
        # Important: Seed the env for reproducibility (handled by wrapper below)
        # env.seed(seed + rank)
        # Wrap the environment with Monitor
        log_file = LOG_DIR / f"monitor_{rank}.csv" # Optional: Log stats per env
        env = Monitor(env, filename=str(log_file) if N_ENVS > 1 else None)
        return env
    # set_global_seeds(seed)
    return _init


if __name__ == "__main__":
    # Updated log message
    logging.info("Starting Deep Scalper (DQN-LSTM) Training...")
    logging.info(f"Using {N_ENVS} parallel environments.")

    # Create the vectorized environment
    if N_ENVS > 1:
        env = SubprocVecEnv([create_env(i, SEED) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([create_env(0, SEED)])

    # Optional: Wrap with VecCheckNan to detect invalid numbers
    env = VecCheckNan(env, raise_exception=True)

    # --- Wrap with Frame Stacker for LSTM ---
    logging.info(f"Wrapping environment with VecFrameStack (n_stack={N_STACK})")
    env = VecFrameStack(env, n_stack=N_STACK)
    # Note: VecFrameStack changes the observation space shape

    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        # Freq is per env, adjust total steps
        save_freq=SAVE_FREQ // N_ENVS,
        save_path=str(SAVE_DIR),
        name_prefix="dqn_lstm_scalper_ckpt"  # Updated checkpoint name
    )

    # --- Model Definition ---
    # Use Custom LstmDqnPolicy
    model = DQN(
        LstmDqnPolicy,  # Use Custom Policy Class
        env,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        # Can be tuple (freq, unit) e.g., (4, "step")
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        seed=SEED,
        device=DEVICE,
        # Double DQN is enabled by default in SB3 DQN
        # Custom policy parameters are now handled within LstmDqnPolicy,
        # ensuring n_stack is passed.
        # policy_kwargs=dict(...)
    )

    logging.info(f"Model Policy: {model.policy}")
    # Log the adjusted observation space after FrameStack
    # Shape should be (n_envs, n_stack * features_per_step)
    logging.info(
        f"Observation Space (Post-FrameStack): {env.observation_space}")
    logging.info(f"Training on device: {model.device}")
    logging.info(f"Training for {TOTAL_TIMESTEPS} timesteps...")

    # --- Training ---
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            # Log stats less frequently for DQN
            log_interval=10
        )
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        # Consider saving the model even if training fails early
        # final_model_path = SAVE_DIR / f"{MODEL_SAVE_NAME}_interrupted"
        # model.save(final_model_path)
        # logging.info(f"Interrupted model saved to {final_model_path}")
    finally:
        # --- Save Final Model ---
        final_model_path = SAVE_DIR / MODEL_SAVE_NAME
        model.save(final_model_path)
        logging.info(
            f"Training finished. Final model saved to {final_model_path}") 