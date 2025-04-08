# Cryptocurrency Trading with LSTM-DQN

This project implements a cryptocurrency trading system using deep reinforcement learning. The system consists of two main components:

1. **LSTM Model**: A deep learning model that learns patterns in cryptocurrency price data and predicts future price movements.
2. **DQN Agent**: A reinforcement learning agent that uses the LSTM model for state representation and learns optimal trading strategies.

## Project Structure

```
.
├── crypto_trading_model/          # Core model implementation
│   ├── lstm_lightning.py          # LSTM model implementation using PyTorch Lightning
│   ├── dqn_agent.py               # Deep Q-Network agent implementation
│   └── trading_environment.py     # Trading environment for reinforcement learning
├── train_improved_lstm.py         # Script to train the LSTM model
├── evaluate_lstm.py               # Script to evaluate the trained LSTM model
├── train_dqn_agent.py             # Script to train the DQN agent
├── evaluate_dqn_agent.py          # Script to evaluate the trained DQN agent
├── data/                          # Directory for storing datasets
│   └── synthetic/                 # Synthetic data for testing
├── models/                        # Directory for storing trained models
│   ├── lstm_improved/             # LSTM model checkpoints
│   └── dqn/                       # DQN agent checkpoints
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-lstm-dqn.git
cd crypto-trading-lstm-dqn
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the LSTM Model

First, you need to train the LSTM model to understand patterns in cryptocurrency price data:

```bash
python train_improved_lstm.py --data_dir data/synthetic --max_epochs 100
```

Key arguments:
- `--data_dir`: Directory containing training data (HDF5 format)
- `--max_epochs`: Maximum number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--regenerate_labels`: Flag to regenerate labels (only use when changing labeling strategy)

### 2. Evaluate the LSTM Model

Evaluate the performance of the trained LSTM model:

```bash
python evaluate_lstm.py --model_dir models/lstm_improved --model_path models/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt --data_dir data/synthetic
```

Key arguments:
- `--model_dir`: Directory containing the model configuration
- `--model_path`: Path to the specific checkpoint to evaluate
- `--data_dir`: Directory containing test data
- `--batch_size`: Batch size for evaluation

### 3. Train the DQN Agent

Train the DQN agent using the trained LSTM model for state representation:

```bash
python train_dqn_agent.py --lstm_model_path models/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt --data_dir data/synthetic --episodes 1000
```

Key arguments:
- `--lstm_model_path`: Path to the trained LSTM model checkpoint
- `--data_dir`: Directory containing training data
- `--episodes`: Number of episodes to train
- `--output_dir`: Directory to save the trained DQN agent

### 4. Evaluate the DQN Agent

Evaluate the performance of the trained DQN agent:

```bash
python evaluate_dqn_agent.py --model_path models/dqn/dqn_agent_best.pt --lstm_model_path models/checkpoints/lstm-epoch=XX-val_loss=X.XXXX.ckpt --data_dir data/synthetic
```

Key arguments:
- `--model_path`: Path to the trained DQN agent
- `--lstm_model_path`: Path to the trained LSTM model checkpoint
- `--data_dir`: Directory containing test data
- `--output_dir`: Directory to save evaluation results

## Model Architecture

### LSTM Model

The LSTM model is designed to predict price movements in cryptocurrency markets. It consists of:

- Stacked LSTM layers for sequence modeling
- Dropout for regularization
- Dense layers for final classification
- Multi-timeframe feature extraction

The model is trained to predict three classes: Buy (1), Hold (0), and Sell (-1).

### DQN Agent

The Deep Q-Network (DQN) agent learns optimal trading strategies based on the LSTM model's state representation. Key features include:

- Experience replay buffer for stable learning
- Double DQN architecture to reduce overestimation bias
- Epsilon-greedy exploration strategy
- Target network updates for stable learning

### Trading Environment

The trading environment simulates cryptocurrency trading. It includes:

- Multi-timeframe market data loading
- Transaction fees and realistic portfolio value calculation
- Reward function based on profit and loss
- Support for different trading actions (buy, hold, sell)

## Performance Metrics

The system evaluates performance using the following metrics:

- **For LSTM Model**: Accuracy, Precision, Recall, F1 Score
- **For DQN Agent**: Total profit, Number of trades, Return over initial investment

## Customization

You can customize various aspects of the system:

- **LSTM Model**: Change the architecture, hyperparameters, or labeling strategy
- **DQN Agent**: Modify exploration strategy, reward scaling, or replay buffer size
- **Trading Environment**: Adjust transaction fees, initial balance, or reward computation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch and PyTorch Lightning for deep learning implementation
- Libraries used: NumPy, Pandas, Matplotlib, h5py 