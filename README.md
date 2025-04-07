# Cryptocurrency Trading Model

An advanced cryptocurrency trading model leveraging multi-timeframe analysis, synthetic data generation, and ensemble methods combining time series models with reinforcement learning.

## Overview

This project implements a comprehensive cryptocurrency trading model with the following key features:

- **Multi-Timeframe Analysis**: Processes data across multiple timeframes (15m, 4h, 1d) to capture both short-term and long-term patterns
- **Advanced Feature Engineering**: Calculates technical indicators and creates domain-specific features
- **Synthetic Data Generation**: Generates realistic market patterns for training robust models
- **Deep Learning Models**: Implements LSTMs, transformers, and specialized architectures for time series analysis
- **Reinforcement Learning**: Trains agents that can adapt to changing market conditions
- **Ensemble Methods**: Combines predictions from multiple models for improved performance
- **Comprehensive Evaluation**: Includes backtesting and performance metrics visualization

## Project Structure

```
crypto_trading_model/
├── data_processing/
│   ├── feature_engineering.py  # Technical indicators and feature creation
│   ├── data_alignment.py       # Aligns data across timeframes
│
├── synthetic_data/
│   ├── pattern_generator.py       # Generates price patterns
│   ├── indicator_engineering.py   # Creates realistic indicators
│   ├── multi_timeframe_sync.py    # Ensures consistency across timeframes
│   ├── dataset_builder.py         # Builds complete synthetic datasets
│
├── models/
│   ├── time_series/
│   │   ├── model.py         # Time series model architectures
│   │   └── trainer.py       # Training pipeline for time series models
│   │
│   ├── reinforcement/
│   │   ├── trading_env.py   # Trading environment
│   │   ├── policy.py        # Policy networks
│   │   ├── agent.py         # RL agents (DQN, PPO)
│   │   └── trainer.py       # Training framework for RL
│
├── config/
│   ├── rl_config.json          # Reinforcement learning configuration
│   ├── time_series_config.json # Time series model configuration
│   ├── synthetic_config.json   # Synthetic data generation configuration
│   ├── backtest_config.json    # Backtesting configuration
│   └── ensemble_config.json    # Ensemble model configuration
│
├── main.py         # Main entry point
└── requirements.txt # Dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/user/crypto_trading_model.git
cd crypto_trading_model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
python main.py --mode setup
```

## Usage

### Generate Synthetic Data

Generate synthetic training data with realistic market patterns:

```bash
python main.py --mode synthetic --config config/synthetic_config.json
```

### Train Time Series Models

Train deep learning models for price prediction:

```bash
python main.py --mode time_series --config config/time_series_config.json
```

### Train Reinforcement Learning Agents

Train RL agents for adaptive trading decisions:

```bash
python main.py --mode reinforcement --config config/rl_config.json
```

### Create Ensemble Model

Combine multiple models for improved predictions:

```bash
python main.py --mode ensemble --config config/ensemble_config.json
```

### Run Backtesting

Evaluate model performance on historical data:

```bash
python main.py --mode backtest --config config/backtest_config.json
```

### Run Full Pipeline

Execute the entire pipeline from data generation to evaluation:

```bash
python main.py --mode full
```

## Configuration Files

The behavior of each component can be customized via JSON configuration files:

- `rl_config.json`: Parameters for reinforcement learning environments and agents
- `time_series_config.json`: Settings for time series models and training
- `synthetic_config.json`: Controls synthetic data generation process
- `backtest_config.json`: Defines backtesting parameters
- `ensemble_config.json`: Specifies ensemble model composition and weighting

## Dependencies

Major dependencies include:

- PyTorch
- Pandas
- NumPy
- TA-Lib (for technical indicators)
- Gymnasium (for reinforcement learning)

For a complete list, see `requirements.txt`.

## License

MIT License 