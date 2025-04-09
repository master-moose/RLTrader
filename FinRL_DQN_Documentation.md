# FinRL Integration for Cryptocurrency Trading

This document describes how to use the FinRL framework for reinforcement learning-based cryptocurrency trading with our custom DQN agent.

## Overview

FinRL is a deep reinforcement learning framework designed specifically for quantitative finance. It provides optimized implementations of various RL algorithms with GPU acceleration, making it an excellent choice for training complex trading agents.

We have integrated FinRL with our existing cryptocurrency trading pipeline to provide better GPU utilization and access to state-of-the-art DRL algorithms.

## Installation

To use FinRL, install the required dependencies:

```bash
pip install -r requirements.txt
```

The requirements.txt file includes all necessary dependencies including FinRL and its requirements.

## Training a FinRL Agent

To train a FinRL trading agent, use the `train_dqn_agent.py` script with the `--use_finrl` flag:

```bash
python train_dqn_agent.py --use_finrl --lstm_model_path models/lstm/model.pt
```

### Command Line Arguments

In addition to the standard training arguments, FinRL integration adds the following options:

- `--use_finrl`: Use FinRL framework instead of custom DQN implementation
- `--finrl_model`: Algorithm to use. Choices: [dqn, ppo, a2c, ddpg, td3, sac]. Default: dqn
- `--net_arch`: Network architecture for FinRL models. Format: "[256,256]" (as string)
- `--tensorboard_log`: Directory for TensorBoard logs. Default: "./tensorboard_logs"
- `--total_timesteps`: Total steps for FinRL training. Default: 1000000

Example with full parameters:

```bash
python train_dqn_agent.py \
  --use_finrl \
  --finrl_model td3 \
  --lstm_model_path models/lstm/model.pt \
  --data_dir data/synthetic \
  --output_dir models/td3 \
  --net_arch "[512,256]" \
  --learning_rate 0.0001 \
  --gamma 0.99 \
  --total_timesteps 2000000 \
  --device cuda
```

## Available Algorithms

FinRL supports multiple reinforcement learning algorithms:

1. **DQN** (Deep Q-Network): Value-based method good for discrete action spaces (buy, sell, hold)
2. **PPO** (Proximal Policy Optimization): Policy gradient method with good sample efficiency
3. **A2C** (Advantage Actor-Critic): Policy gradient method with parallel training
4. **DDPG** (Deep Deterministic Policy Gradient): For continuous action spaces
5. **TD3** (Twin Delayed DDPG): Improvement over DDPG with reduced overestimation bias
6. **SAC** (Soft Actor-Critic): Off-policy method with entropy regularization

## Evaluating FinRL Models

Trained FinRL models can be evaluated using the `evaluate_dqn_agent.py` script:

```bash
python evaluate_dqn_agent.py \
  --model_type finrl \
  --finrl_model td3 \
  --model_path models/td3/finrl_td3_final.zip \
  --data_dir data/synthetic \
  --output_dir evaluation_results
```

The evaluation will produce detailed metrics and visualizations of the agent's performance.

## Configuration

FinRL parameters can be configured in the `crypto_trading_model/config/rl_config.json` file under the `finrl` section:

```json
"finrl": {
    "use_finrl": false,
    "algorithm": "dqn",
    "net_arch": [256, 256],
    "total_timesteps": 1000000,
    "learning_rate": 1e-4,
    "batch_size": 256,
    "buffer_size": 100000,
    "gamma": 0.99,
    "reward_scaling": 1e-3,
    "train_test_split": 0.8,
    "tensorboard_log": "./tensorboard_logs",
    "technical_indicators": [
        "macd", 
        "rsi_30", 
        "cci_30", 
        "dx_30",
        "close_30_sma",
        "close_60_sma"
    ],
    "n_envs": 4,
    "device": "auto",
    "verbose": 1
}
```

## GPU Utilization

FinRL makes efficient use of GPU resources by:

1. Using optimized tensor operations
2. Employing batch processing for training data
3. Implementing parallel environment sampling
4. Utilizing advanced optimization techniques like automatic mixed precision (AMP)

To monitor GPU usage during training, run:

```bash
nvidia-smi -l 1
```

## Troubleshooting

If you encounter issues with FinRL:

1. **Memory Errors**: Reduce batch size or buffer size
2. **CUDA Out of Memory**: Use `--device cpu` to train on CPU instead of GPU
3. **Import Errors**: Verify installation of all dependencies with `pip install -r requirements.txt`
4. **File Not Found Errors**: Check paths to data and model files

## Performance Comparison

In general, FinRL models may provide better performance than the custom implementation in terms of:

1. **Training Speed**: Typically 2-4x faster on the same hardware
2. **GPU Utilization**: More efficient use of GPU resources
3. **Sample Efficiency**: FinRL algorithms often require fewer samples for training
4. **Convergence**: May converge to better policies more consistently

## References

- [FinRL GitHub Repository](https://github.com/AI4Finance-Foundation/FinRL)
- [FinRL Documentation](https://finrl.readthedocs.io/) 