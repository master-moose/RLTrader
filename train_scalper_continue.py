#!/usr/bin/env python
# train_scalper_continue.py - Continue training from a checkpoint
import logging
from pathlib import Path
import torch
import argparse
import os

from stable_baselines3 import DQN

# --- Ensure the classes are defined the same way as in train_scalper.py ---
# Import all required components from the original training script
from train_scalper import (
    # Classes and functions
    LstmFeatureExtractor, LstmDqnPolicy, create_env, StepLogCallback,
    # Configuration constants we may want to override
    DATA_DIRECTORY, SYMBOL_FILENAME_PART, HDF_KEY, LOB_DEPTH, MAX_HOLDING, 
    ACTION_DIM, TRANSACTION_COST_PCT, EPISODE_LENGTH, N_STACK, N_ENVS, 
    SEED, DEVICE, TOTAL_TIMESTEPS, LOG_DIR, SAVE_DIR, SAVE_FREQ,
)

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecFrameStack

# --- Configuration for continued training ---
def parse_args():
    parser = argparse.ArgumentParser(description="Continue training the DQN-LSTM scalper model from a checkpoint")
    
    # Model loading parameters
    parser.add_argument("--load_model", type=str, required=True, 
                        help="Path to the model file to load. Either the zipped checkpoint or final model.")
    parser.add_argument("--tb_logs", type=str, default=None,
                        help="Path to the tensorboard logs folder from previous training.")
    
    # New training parameters
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to the data directory. Defaults to the one from train_scalper.py")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol filename part. Defaults to the one from train_scalper.py.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="Total timesteps for continued training. Default: 1M")
    parser.add_argument("--n_envs", type=int, default=None,
                        help="Number of parallel environments. Default: same as train_scalper.py")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Default: increments from the original seed")
    parser.add_argument("--save_freq", type=int, default=None,
                        help="Frequency of saving checkpoints. Default: same as train_scalper.py")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the model. Default: same as train_scalper.py")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for logs. Default: same as train_scalper.py with '_continued'")
    
    # Return parsed arguments
    return parser.parse_args()


def setup_continued_training():
    """Configure training based on command line arguments and original settings."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup paths
    model_path = Path(args.load_model)
    
    # Validate model path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set up paths with defaults from original training if not specified
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIRECTORY
    symbol = args.symbol if args.symbol else SYMBOL_FILENAME_PART
    total_timesteps = args.total_timesteps
    n_envs = args.n_envs if args.n_envs is not None else N_ENVS
    
    # Use a seed that's different from the original by default
    seed = args.seed if args.seed is not None else (SEED + 1000)
    
    # Set up save and log directories with defaults
    save_dir = Path(args.save_dir) if args.save_dir else SAVE_DIR
    log_dir = Path(args.log_dir) if args.log_dir else LOG_DIR.with_name(f"{LOG_DIR.name}_continued")
    save_freq = args.save_freq if args.save_freq is not None else SAVE_FREQ
    
    # Create directories
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Return all the configured values
    return {
        'model_path': model_path,
        'data_dir': data_dir,
        'symbol': symbol,
        'total_timesteps': total_timesteps,
        'n_envs': n_envs,
        'seed': seed,
        'save_dir': save_dir,
        'log_dir': log_dir,
        'save_freq': save_freq,
        'tb_logs': args.tb_logs,
    }


def continue_training():
    """Main function to continue training from a checkpoint."""
    # Get configuration
    config = setup_continued_training()
    
    logging.info("=== Continuing DQN-LSTM Scalper Training ===")
    logging.info(f"Loading model from: {config['model_path']}")
    logging.info(f"Data directory: {config['data_dir']}")
    logging.info(f"Symbol: {config['symbol']}")
    logging.info(f"Using {config['n_envs']} parallel environments")
    logging.info(f"Training for {config['total_timesteps']} additional timesteps")
    logging.info(f"Save directory: {config['save_dir']}")
    logging.info(f"Log directory: {config['log_dir']}")
    
    # Create the vectorized environment
    if config['n_envs'] > 1:
        env = SubprocVecEnv([create_env(i, config['seed']) for i in range(config['n_envs'])])
    else:
        env = DummyVecEnv([create_env(0, config['seed'])])

    # Optional: Wrap with VecCheckNan to detect invalid numbers
    env = VecCheckNan(env, raise_exception=True)
    
    # Wrap with Frame Stacker for LSTM
    logging.info(f"Wrapping environment with VecFrameStack (n_stack={N_STACK})")
    env = VecFrameStack(env, n_stack=N_STACK)
    
    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        # Freq is per env, adjust total steps
        save_freq=config['save_freq'] // config['n_envs'],
        save_path=str(config['save_dir']),
        name_prefix="dqn_lstm_scalper_continued_ckpt"
    )
    
    # Custom step logging callback
    step_log_callback = StepLogCallback(log_freq=5000)  # Log every 5000 steps
    
    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, step_log_callback])
    
    # Load the model
    try:
        # Load model with custom policy class
        model = DQN.load(
            path=str(config['model_path']),
            env=env,
            custom_objects={
                "policy_class": LstmDqnPolicy,
                "features_extractor_class": LstmFeatureExtractor,
            },
            device=DEVICE,
            # Optionally set the tensorboard_log path
            tensorboard_log=str(config['log_dir']),
        )
        logging.info(f"Successfully loaded model from {config['model_path']}")
        logging.info(f"Model policy: {model.policy}")
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        return
    
    # Log observation space after FrameStack
    logging.info(f"Observation Space (Post-FrameStack): {env.observation_space}")
    logging.info(f"Training on device: {model.device}")
    
    # --- Start Training ---
    logging.info(f"[{os.getpid()}] Starting continued training for {config['total_timesteps']} timesteps...")
    try:
        model.learn(
            total_timesteps=config['total_timesteps'],
            callback=callback_list,
            log_interval=10,  # Log training metrics every 10 updates
            tb_log_name="dqn_lstm_scalper_continued",
            reset_num_timesteps=False,  # Continue counting timesteps from where we left off
        )
    except Exception as e:
        logging.error(f"Error during continued training: {e}", exc_info=True)
    finally:
        # --- Save Final Model ---
        final_model_path = config['save_dir'] / "dqn_lstm_scalper_continued_final"
        model.save(final_model_path)
        logging.info(f"Training finished. Final model saved to {final_model_path}")


if __name__ == "__main__":
    continue_training() 