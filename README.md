# LSTM-DQN Trading Agent

This project implements a cryptocurrency trading agent using a Deep Q-Network (DQN) enhanced with a Long Short-Term Memory (LSTM) network for feature extraction.

The agent is built using Python, PyTorch, Gymnasium (formerly OpenAI Gym), and Stable-Baselines3.

## Project Structure

```
.
├── rl_agent/
│   ├── __init__.py
│   ├── callbacks.py       # Custom callbacks for training (monitoring, logging, saving)
│   ├── config.py          # Configuration settings and constants
│   ├── data/              # Data loading and preprocessing
│   │   └── data_loader.py
│   ├── environment.py     # Trading environment implementation
│   ├── env_wrappers.py    # Environment wrappers (safeguards, risk management)
│   ├── models/              # Model definitions
│   │   ├── __init__.py
│   │   └── lstm_dqn.py      # LSTM-DQN specific model components (if needed)
│   ├── train.py           # Training and evaluation logic
│   └── utils.py           # Utility functions (logging, resource checks, plots)
├── data/                  # Directory for storing trading data (e.g., CSV files)
├── logs/                  # Directory for storing logs, TensorBoard data, and saved models
├── checkpoints/           # Directory for storing model checkpoints
├── train_dqn.py           # Main script to start training or evaluation
├── requirements.txt       # Python dependencies (to be added)
└── README.md              # This file
```

## Features

*   **LSTM Feature Extraction**: Uses an LSTM network to process time-series features before feeding them to the Q-network.
*   **DQN Agent**: Implements the Deep Q-Network algorithm for learning trading policies.
*   **Customizable Environment**: Includes a `TradingEnvironment` based on Gymnasium.
*   **Environment Wrappers**: Implements safeguards like trade cooldowns, oscillation prevention, and basic risk management.
*   **Callbacks**: Provides custom callbacks for:
    *   Resource monitoring (CPU, RAM, GPU)
    *   Trading metrics logging (portfolio value, returns, Sharpe ratio)
    *   Saving best models during training
    *   Saving periodic checkpoints
    *   TensorBoard integration
*   **Configuration Management**: Allows training parameters to be specified via command-line arguments or a configuration file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd lstm-dqn-trading-agent
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    *A `requirements.txt` file should be created. For now, manually install necessary packages:*
    ```bash
    pip install torch torchvision torchaudio gymnasium stable-baselines3[extra] pandas numpy psutil matplotlib seaborn gputil
    ```
    *(Note: Ensure you install the correct PyTorch version for your system, potentially with CUDA support if you have a compatible GPU)*

4.  **Prepare Data**: Place your training, validation (optional), and testing (optional) data CSV files in the `data/` directory.

## Usage

Use the `train_dqn.py` script to train or evaluate the agent.

**Training:**

```bash
python train_dqn.py --data_path data/your_training_data.csv \
                    --val_data_path data/your_validation_data.csv \
                    --model_type lstm_dqn \
                    --features close,volume,rsi,macd \
                    --sequence_length 60 \
                    --lstm_hidden_size 128 \
                    --fc_hidden_size 64 \
                    --learning_rate 0.0001 \
                    --total_timesteps 200000 \
                    --eval_freq 10000 \
                    --save_freq 20000 \
                    --log_dir ./logs \
                    --checkpoint_dir ./checkpoints \
                    --model_name lstm_dqn_v1
```

**Evaluation:**

To evaluate a pre-trained model:

```bash
python train_dqn.py --eval_only \
                    --resume_from logs/lstm_dqn_v1/best_model.zip \
                    --test_data_path data/your_test_data.csv \
                    --model_type lstm_dqn \
                    --features close,volume,rsi,macd \
                    --sequence_length 60 \
                    --log_dir ./logs \
                    --model_name lstm_dqn_v1
```

**Command-line Arguments:**

Run `python train_dqn.py --help` to see all available arguments for customizing data, environment, model, and training parameters.

## TODO

*   Add `requirements.txt`.
*   Implement detailed risk management rules in `SafeTradingEnvWrapper`.
*   Refine reward function in `TradingEnvironment`.
*   Add comprehensive unit and integration tests.
*   Implement other RL algorithms (PPO, A2C, SAC) for comparison.

## Model Architecture

The LSTM-DQN model consists of:

1. **Feature Extraction**: LSTM layers process the time series data to extract temporal features
2. **Q-Network**: Fully connected layers map the extracted features to Q-values for each action
3. **Action Selection**: Actions are selected based on the Q-values (with exploration during training)

## Performance Metrics

The framework calculates and tracks various performance metrics:

- **Portfolio Value**: Total value of portfolio (cash + assets)
- **Total Return**: Percentage return on initial investment
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest drop from peak to trough
- **Win Rate**: Percentage of profitable trades
- **Trade Frequency**: Number of trades per time period

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stable-Baselines3 for the reinforcement learning framework
- PyTorch for the deep learning framework
- Thanks to the reinforcement learning and algorithmic trading communities for inspiration and ideas 