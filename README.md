# Crypto Trading RL Agent

This project implements and trains Reinforcement Learning (RL) agents (DQN, PPO, A2C, SAC, etc.) using LSTM or TCN features for automated cryptocurrency trading, specifically focusing on BTC/USDT. It includes components for fetching real market data, generating synthetic data, processing features, training models, and evaluating trading strategies within a custom Gym environment.

## Features

-   **Data Fetching:** Connects to Binance via CCXT to download historical OHLCV data (`fetch_binance_data.py`).
-   **Data Processing:** Processes raw data, likely involving feature engineering (`process_historic_data.py`).
-   **Synthetic Data Generation:** Creates realistic, regime-based synthetic market data for robust training (`generate_data.py`, `generate_10y_data.py`).
-   **LSTM Model Training:** Trains an LSTM model, potentially for feature extraction or price prediction (`train_improved_lstm.py`).
-   **RL Agent Training:** Trains various RL agents (DQN, PPO, A2C, SAC, QRDQN, RecurrentPPO) using Stable Baselines3 and SB3 Contrib (`train_dqn.py` -> `rl_agent/train.py`).
    -   Supports custom LSTM and TCN feature extractors.
    -   Integrates with Ray Tune for hyperparameter optimization.
-   **Trading Environment:** Custom `gymnasium.Env` simulating crypto trading with configurable parameters (`rl_agent/environment/trading_env.py`).
    -   Handles transaction fees, portfolio management, and risk metrics.
    -   Complex reward function with weighted components (profit, drawdown, Sharpe, fees, consistency, etc.).
-   **Evaluation:** Scripts for evaluating model performance on historical and synthetic data (`evaluate_*.py`).
-   **Progressive Learning:** Potential framework for progressive learning strategies (`progressive_learning.py`).

## Project Structure

```
.
├── .git/
├── .github/                 # CI/CD workflows (if any)
├── .vscode/                 # VSCode settings (if any)
├── checkpoints/             # Saved model checkpoints (e.g., during RL training)
├── data/
│   ├── raw/                 # Raw data fetched from exchanges (e.g., btc_usdt_15m.csv)
│   ├── processed/           # Processed data (e.g., features added, splits)
│   └── synthetic/           # Generated synthetic datasets (e.g., synthetic_dataset.h5)
│   └── synthetic_10y/       # Generated 10-year synthetic dataset
├── logs/                    # Log files for various processes
├── models/
│   ├── lstm_improved/       # Trained LSTM models
│   └── rl/                  # Trained RL agent models (DQN, PPO, etc.)
├── output/
│   ├── backtest/            # Backtesting results
│   ├── ensemble/            # Ensemble model outputs (if any)
│   ├── reinforcement/       # RL agent evaluation outputs (plots, metrics)
│   └── time_series/         # LSTM model outputs (if any)
├── rl_agent/                # Core RL agent package
│   ├── callbacks.py         # Custom SB3 callbacks (logging, checkpoints)
│   ├── data/
│   │   └── data_loader.py   # Loads data for RL environment
│   │   └── normalize_features.py # Script for feature normalization
│   ├── environment/
│   │   └── trading_env.py   # Custom Gym trading environment
│   ├── models.py            # Custom SB3 policy networks/feature extractors (LSTM)
│   ├── policies.py          # Custom SB3 policies (TCN)
│   ├── train.py             # Main RL agent training and evaluation logic
│   └── utils.py             # Utility functions (logging, metrics, config)
├── tests/                   # Unit and integration tests (if any)
├── .cursorignore
├── .gitignore
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── fetch_binance_data.py    # Script to fetch raw data from Binance
├── process_historic_data.py # Script to process raw data
├── generate_data.py         # Script to generate synthetic data with regimes
├── generate_10y_data.py     # Convenience script for 10-year data generation
├── train_improved_lstm.py   # Script to train the LSTM model
├── train_dqn.py             # Entry point script for training RL agents
├── data_generation_config.json # Config for synthetic data generation
├── train_config.json        # Default config for RL training
├── evaluate_*.py            # Scripts for evaluating models
├── progressive_learning.py  # Script for progressive learning experiments
├── rl_agent/data/normalize_features.py # Utility script for feature normalization (outside package?)
└── ...                      # Other utility/testing scripts
```

## Prerequisites

-   Python 3.8+
-   Required Python packages (see `requirements.txt`)
-   An environment supporting PyTorch (CPU or GPU)

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  Install required packages:
    ```bash
    pip install -r requirements.txt
    # Potentially install PyTorch separately based on your system/CUDA version
    # See: https://pytorch.org/get-started/locally/
    ```

## Usage

### 1. Data Preparation

**a) Fetch Real Data (Optional):**

```bash
python fetch_binance_data.py
```

-   Fetches BTC/USDT 15m data from Binance.
-   Saves to `data/raw/btc_usdt_15m.csv`.

**b) Process Data (Implement as needed):**

```bash
python process_historic_data.py --input_file data/raw/btc_usdt_15m.csv --output_dir data/processed
```

-   Modify `process_historic_data.py` to add desired features.
-   Takes raw data, adds features, and saves processed splits (train/val/test) in HDF5 format to `data/processed/`.

**c) Generate Synthetic Data (Optional):**

-   Configure parameters in `data_generation_config.json`.
-   Run generation:
    ```bash
    # Generate standard amount (e.g., 1 year)
    python generate_data.py --config data_generation_config.json --output_dir data/synthetic --num_samples 35040 # Adjust num_samples as needed

    # Or generate 10 years of data
    python generate_10y_data.py --config data_generation_config.json --output_dir data/synthetic_10y
    ```
-   Generates OHLCV data with different market regimes.
-   Saves splits (train/val/test) in HDF5 format to the specified output directory.

**d) Normalize Features:**

-   After processing or generating data, normalize the features. This step is crucial for many models, especially neural networks.
-   The script `rl_agent/data/normalize_features.py` handles normalization for both CSV and multi-key HDF5 files.
-   Available methods: `minmax`, `zscore`, `robust`.
-   Given the potential outliers and volatility in cryptocurrency data, the `robust` method (which uses median and interquartile range) is often preferred as it's less sensitive to extreme values.
-   Essential columns like OHLCV can be preserved in their original form using `--preserve_columns`.

```bash
# Example: Normalize features in processed data using the robust method
python rl_agent/data/normalize_features.py \
    --input_dir data/processed \
    --output_dir data/processed_normalized \
    --method robust \
    --file_pattern "*.h5" \
    --preserve_columns "open,high,low,close,volume" \
    --suffix "_normalized" \
    --force # Optional: Overwrite existing files
```
- Remember to use the normalized data directory (e.g., `data/processed_normalized`) as the `--data_dir` for subsequent training steps.

### 2. Train LSTM Model (Optional)

-   Configure LSTM training in `train_config.json` (or relevant config).
-   Run LSTM training:
    ```bash
    python train_improved_lstm.py --data_dir data/processed --output_dir models/lstm_improved --config train_config.json
    ```
-   Uses data from `--data_dir` (e.g., `data/processed` or `data/synthetic`).
-   Saves the trained LSTM model to `--output_dir`.

### 3. Train RL Agent

-   Configure RL agent parameters, environment settings, and feature extractor options in `train_config.json` or create a separate config file.
-   Run training using the `train_dqn.py` entry point, which passes arguments to `rl_agent/train.py`:
    ```bash
    # Example: Train PPO with LSTM extractor on processed real data
    python train_dqn.py --agent PPO --config train_config.json --data_dir data/processed --total_timesteps 1000000 --use_lstm_extractor --save_path models/rl/ppo_lstm_real --tb_log_name PPO_LSTM_Real

    # Example: Train DQN with default MLP features on synthetic data
    python train_dqn.py --agent DQN --config train_config.json --data_dir data/synthetic --total_timesteps 500000 --save_path models/rl/dqn_mlp_synthetic --tb_log_name DQN_MLP_Synthetic

    # Example: Train PPO using *normalized* processed data
    python train_dqn.py --agent PPO --config train_config.json --data_dir data/processed_normalized --total_timesteps 1000000 --save_path models/rl/ppo_normalized --tb_log_name PPO_Normalized
    ```
-   **Key Arguments (via `train_dqn.py` -> `rl_agent/train.py`):**
    -   `--agent`: RL algorithm (e.g., `DQN`, `PPO`, `A2C`, `SAC`, `QRDQN`, `RecurrentPPO`).
    -   `--config`: Path to JSON configuration file. Overrides default args.
    -   `--data_dir`: Directory containing the training data (e.g., `data/processed`, `data/synthetic`).
    -   `--total_timesteps`: Total number of training steps.
    -   `--save_path`: Directory to save the trained model and environment stats.
    -   `--tb_log_name`: Name for the TensorBoard log directory.
    -   `--use_lstm_extractor`: Use the custom LSTM feature extractor. Requires `--lstm_model_path` if loading a pre-trained LSTM.
    -   `--use_tcn_policy`: Use the custom TCN policy.
    -   `--n_envs`: Number of parallel environments for VecEnv.
    -   `--eval_freq`: Evaluate the model every N steps.
    -   `--load_model`: Path to a pre-trained RL model to continue training or evaluate.
    -   `--eval_only`: Run evaluation only using `--load_model`.
    -   Many other arguments for hyperparameters, environment settings, callbacks, etc. (see `rl_agent/train.py` `parse_args`).

### 4. Evaluate RL Agent

-   Use the `--eval_only` flag with a trained model:
    ```bash
    python train_dqn.py --agent PPO --config train_config.json --data_dir data/processed --load_model models/rl/ppo_lstm_real/best_model.zip --eval_only --n_eval_episodes 10
    ```
-   Evaluation results (metrics, plots) are typically saved in the `output/reinforcement` directory or specified via config/args.
-   Dedicated evaluation scripts (`evaluate_*.py`) might offer more specific analysis.

## Trading Environment (`rl_agent/environment/trading_env.py`)

-   **State:** Based on `sequence_length` steps of historical `features` (e.g., OHLCV, indicators). Portfolio state (balance, shares held) is *not* part of the observation by default but used internally.
-   **Actions:**
    -   0: Sell (all shares)
    -   1: Hold
    -   2: Buy (using `max_position` fraction of balance)
-   **Reward Function:** A complex combination designed to promote profitable and stable trading:
    -   `portfolio_change_weight`: Based on the change in total portfolio value.
    -   `drawdown_penalty_weight`: Penalizes decreases from the peak portfolio value.
    -   `sharpe_reward_weight`: Rewards higher risk-adjusted returns (calculated over `sharpe_window`).
    -   `fee_penalty_weight`: Penalizes transaction costs (`transaction_fee`).
    -   `benchmark_reward_weight`: Compares performance against holding the asset.
    -   `consistency_penalty_weight`: Penalizes rapidly flipping between buy/sell actions (threshold: `consistency_threshold`).
    -   `idle_penalty_weight`: Penalizes holding for too long without action (threshold: `idle_threshold`).
    -   `profit_bonus_weight`: Adds a bonus for profitable trades.
    -   `trade_penalty_weight`: Small penalty per trade executed.
    -   `exploration_bonus_weight`: Adds a decaying bonus to encourage exploration early on.
-   **Risk Profile:** Primarily controlled by `max_position` (limits capital per trade) and the reward function weights (especially `drawdown_penalty_weight`, `sharpe_reward_weight`). Transaction fees also influence behavior.

## Configuration

-   `data_generation_config.json`: Parameters for synthetic data generation (regimes, volatility, price process).
-   `train_config.json`: Default parameters for RL training, environment, models, and hyperparameters. Specific settings can be overridden by command-line arguments or custom config files passed via `--config`.

## Notes

-   Ensure data directories (`data/`, `models/`, `output/`, `logs/`) exist or are created (some scripts handle this).
-   Training RL agents can be computationally intensive and time-consuming.
-   Hyperparameter tuning (using Ray Tune or manually) is crucial for optimal performance.
-   Monitor training progress using TensorBoard (`tensorboard --logdir logs/tensorboard/`). 