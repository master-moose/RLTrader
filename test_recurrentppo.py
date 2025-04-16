#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test script for RecurrentPPO creation with seperate LSTM"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_agent.train import create_model

# Create a simple environment for testing
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(10, 10), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
    def reset(self, seed=None):
        obs = np.zeros((10, 10), dtype=np.float32)
        return obs, {}
    
    def step(self, action):
        obs = np.zeros((10, 10), dtype=np.float32)
        return obs, 0.0, False, False, {}

# Create a vectorized env
env = DummyVecEnv([lambda: DummyEnv()])

# Test configuration that includes the "seperate" LSTM option
config = {
    'model_type': 'recurrentppo',
    'shared_lstm': 'seperate',  # This is the key parameter we're testing
    'lstm_hidden_size': 64,
    'n_lstm_layers': 1,
    'log_dir': './logs',
    'model_name': 'test',
    'seed': 42,
    'cpu_only': True,
    'learning_rate': 0.0003,
    'batch_size': 64,
    'gamma': 0.99,
    'n_steps': 128,
    'ent_coef': '0.01',
    'vf_coef': 0.5,
    'clip_range': 0.2,
    'gae_lambda': 0.95,
    'max_grad_norm': 0.5,
    'n_epochs': 10
}

if __name__ == "__main__":
    print("Creating RecurrentPPO model with 'seperate' LSTM...")
    model = create_model(env, config)
    print("Model created successfully!")
    print(f"Model type: {type(model).__name__}")
    print("LSTM config successful!") 