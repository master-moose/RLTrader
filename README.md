# Crypto Trading RL Agent

This project implements and trains Reinforcement Learning (RL) agents (DQN, PPO, A2C, SAC, etc.) using LSTM or TCN features for automated cryptocurrency trading, specifically focusing on BTC/USDT. It includes components for fetching real market data, generating synthetic data, processing features, training models, and evaluating trading strategies within a custom Gym environment.

## Features

-   **Data Fetching:** Connects to Binance via CCXT to download historical OHLCV data (`fetch_binance_data.py`).
-   **Data Processing:** Processes raw data, likely involving feature engineering (`process_historic_data.py`).
-   **Synthetic Data Generation:** Creates realistic, regime-based synthetic market data for robust training (`generate_data.py`, `generate_10y_data.py`).
-   **LSTM Model Training:** Trains an LSTM model, initially used for feature extraction or price prediction, though currently not used as I'm focusing on TCN features over LSTM. (`train_improved_lstm.py`).
-   **RL Agent Training:** Trains various RL agents (DQN, PPO, A2C, SAC, QRDQN, RecurrentPPO) using Stable Baselines3 and SB3 Contrib (`train_dqn.py` -> `rl_agent/train.py`).
    -   Supports custom LSTM and TCN feature extractors.
    -   Integrates with Ray Tune for hyperparameter optimization.
-   **Trading Environment:** Custom `gymnasium.Env` simulating crypto trading with configurable parameters (`rl_agent/environment/trading_env.py`).
    -   **Supports bidirectional trading (Long and Short positions).**
    -   Handles transaction fees, portfolio management, and risk metrics.
    -   Complex reward function with weighted components (profit, drawdown, Sharpe, fees, etc.).
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
│   │   └── trading_env.py   # Custom Gym trading environment (Bidirectional)
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

-   Python 3.10+
-   Required Python packages (see `requirements.txt`)
-   An environment supporting PyTorch (CPU or GPU)
-   CUDA 12.4+ (12.8 is recommended for optimal performance)

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

-   **Purpose:** Synthetic data generation was added to test the performance of the RL agent in a more controlled environment, and to have a large amount of data to train on. Though historical data is preferred, using the synthetic data allows for more granular control over the market conditions - as well as providing a more stable environment for training.
-   **Scripts:** `generate_data.py` is the core script, while `generate_10y_data.py` is a convenience wrapper to generate a specific long duration.

-   **Regime-Based Generation:** The generation process aims for realism by simulating distinct market regimes:
    -   `Uptrend:` Characterized by positive price drift, normal volatility, and lower mean reversion.
    -   `Downtrend:` Negative drift, often higher volatility than uptrends.
    -   `Ranging:` No significant drift, lower volatility, and higher mean reversion.
    -   `Volatility Expansion:` High volatility, high volume, often short-lived bursts.
    -   `Volatility Contraction:` Very low volatility, low volume, often preceding expansion phases.
    -   The script generates a sequence of these regimes based on defined probabilities and durations, using a Markov transition matrix to model realistic shifts between regimes (e.g., volatility contraction is more likely to transition to volatility expansion).

-   **Simulation Process:**
    1.  **Regime Sequence:** A sequence of regimes matching the desired `num_samples` is generated.
    2.  **Price Process:** A price path is simulated, typically using a process like Geometric Brownian Motion (GBM) or an Ornstein-Uhlenbeck (OU) process. The parameters of this process (drift `mu`, volatility `sigma`, mean reversion strength) are dynamically adjusted at each step based on the *current* market regime from the generated sequence.
    3.  **OHLCV Generation:** Realistic Open, High, Low prices are derived from the simulated closing price path and regime-dependent volatility. Volume is also simulated based on the current regime's `volume_factor`.
    4.  **Feature Calculation:** Standard technical indicators (RSI, MACD, Bollinger Bands, etc. - *verify which ones are actually included in `generate_data.py`*) are calculated on the generated OHLCV data.
    5.  **Labeling (Optional but typical):** Price direction labels for supervised learning (like for the LSTM) might be generated based on future price movements.

-   **Configuration:**
    -   Primary configuration is done via a JSON file (default: `data_generation_config.json`). This file typically specifies:
        -   `base_price`: Initial starting price.
        -   `base_volatility`: Baseline volatility level.
        -   `regime_distribution`: Probabilities for initially selecting each regime.
        -   Overrides for individual regime parameters (drift, volatility, mean reversion, volume factor, duration range).
        -   Transition matrix probabilities between regimes.
    -   **Command-line arguments:**
        -   `--num_samples`: Total number of time steps (e.g., 15-minute intervals) to generate.
        -   `--output_dir`: Directory to save the generated data (e.g., `data/synthetic/`).
        -   `--config`: Path to the configuration JSON file.
        -   `--seed`: Random seed for reproducibility.

-   **Output:** The script saves the generated data, typically split into training, validation, and test sets, in HDF5 format (`*.h5`) within the specified output directory. Each HDF5 file may contain multiple keys if different timeframes are generated or processed.

-   **Running Generation:**
    ```bash
    # Generate a specific number of samples (e.g., ~1 year of 15m data)
    python generate_data.py --config data_generation_config.json --output_dir data/synthetic --num_samples 35040 --seed 42

    # Generate 10 years of data using the convenience script
    python generate_10y_data.py --config data_generation_config.json --output_dir data/synthetic_10y --seed 42
    ```

**d) Normalize Features:**

-   After processing or generating data, normalize the features.
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

-   **State Observation:** Based on `sequence_length` steps of historical `features` (e.g., OHLCV, indicators), plus the current position type (-1 for Short, 0 for Flat, 1 for Long) and the entry price normalized relative to the current price. Portfolio state (balance, absolute shares held) is used internally but not directly part of the agent's observation by default.
-   **Actions:**
    -   0: Hold (Maintain current position: Long, Short, or Flat)
    -   1: Go Long (Enter a long position if currently Flat)
    -   2: Go Short (Enter a short position if currently Flat)
    -   3: Close Position (Exit Long or Short position to become Flat)
-   **Reward Function:** A complex combination designed to promote profitable and stable trading, considering both long and short scenarios:
    -   `portfolio_change_weight`: Based on the percentage change in total portfolio value (handles unrealized PnL for shorts correctly).
    -   `drawdown_penalty_weight`: Penalizes decreases from the peak portfolio value.
    -   `sharpe_reward_weight`: Rewards higher risk-adjusted returns (calculated over `sharpe_window`).
    -   `fee_penalty_weight`: Penalizes transaction costs (`transaction_fee`) incurred on entering or closing positions.
    -   `benchmark_reward_weight`: Compares performance against holding the asset (less relevant for bidirectional).
    -   `idle_penalty_weight`: Penalizes holding a *Flat* position for too long without action (threshold: `idle_threshold`).
    -   `profit_bonus_weight`: Rewards profitable closing trades (both long and short).
    -   `trade_penalty_weight`: Applies a small penalty for *entering* a new trade (long or short).
-   **Episode Termination:** Ends if drawdown exceeds a threshold (e.g., 50%), maximum steps are reached, or the end of the data is encountered. Open positions are automatically closed at the end of an episode.

## Configuration

-   `data_generation_config.json`: Parameters for synthetic data generation (regimes, volatility, price process).
-   `train_config.json`: Default parameters for RL training, environment, models, and hyperparameters. Specific settings can be overridden by command-line arguments or custom config files passed via `--config`.



## Current Workflow

-   Pull binance data
-   Process data
-   Normalize features
-   Run Ray sweep to find best hyperparameters
-   Take best performing hyperparameters and train RL agent
-   Evaluate RL agent
-   Rinse and repeat until model performance is satisfactory (I have yet to achieve satisfactory results)


## Notes

-   Ensure data directories (`data/`, `models/`, `output/`, `logs/`) exist or are created (some scripts handle this).
-   Hyperparameter tuning (using Ray Tune or manually) is crucial for optimal performance.
-   Monitor training progress using TensorBoard (`tensorboard --logdir logs/tensorboard/`). 
