#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) agent for cryptocurrency trading.
Uses pre-trained LSTM model for state representation.
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    Takes the state representation from LSTM and outputs Q-values for each action.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-Network.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space (number of possible actions)
        hidden_dim : int
            Size of hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network agent for cryptocurrency trading.
    Uses a pre-trained LSTM model for state representation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_model_path: str,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100000,
        update_target_every: int = 10,
        device: str = None
    ):
        """
        Initialize the DQN agent.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space (number of possible actions)
        lstm_model_path : str
            Path to pre-trained LSTM model
        hidden_dim : int
            Size of hidden layers in Q-Network
        learning_rate : float
            Learning rate for optimizer
        gamma : float
            Discount factor for future rewards
        epsilon_start : float
            Initial exploration rate
        epsilon_end : float
            Final exploration rate
        epsilon_decay : float
            Decay rate for exploration
        batch_size : int
            Mini-batch size for training
        buffer_size : int
            Size of replay buffer
        update_target_every : int
            Number of steps between updates to target network
        device : str
            Device to use for training ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_model_path = lstm_model_path
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_target_every = update_target_every
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is used for evaluation only
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Initialize step counter
        self.steps = 0
        
        # Load LSTM model for state representation
        self._load_lstm_model()
        
        # Metrics for tracking
        self.losses = []
        self.rewards = []
        self.epsilons = []
    
    def _load_lstm_model(self):
        """Load the pre-trained LSTM model for state representation"""
        try:
            # Placeholder for LSTM model loading
            # This will be implemented based on your specific LSTM structure
            logger.info(f"Loading LSTM model from {self.lstm_model_path}")
            
            # Check if the file exists
            if not os.path.exists(self.lstm_model_path):
                raise FileNotFoundError(f"LSTM model file not found at {self.lstm_model_path}")
            
            # Load the model
            checkpoint = torch.load(self.lstm_model_path, map_location=self.device)
            
            # TODO: Initialize and load LSTM model based on your specific implementation
            # self.lstm_model = ...
            
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_state_representation(self, market_data):
        """
        Extract state representation from market data using the pre-trained LSTM model.
        
        Parameters:
        -----------
        market_data : dict
            Dictionary containing market data for different timeframes
            
        Returns:
        --------
        torch.Tensor
            State representation tensor
        """
        # TODO: Implement based on your specific LSTM model architecture
        # For now, return a dummy tensor
        return torch.randn(self.state_dim).to(self.device)
    
    def select_action(self, state, evaluation=False):
        """
        Select an action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : torch.Tensor
            Current state representation
        evaluation : bool
            Whether to use greedy policy (True) or epsilon-greedy (False)
            
        Returns:
        --------
        int
            Selected action index
        """
        # During evaluation, use greedy policy
        if evaluation or random.random() > self.epsilon:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state)
                self.q_network.train()
                return q_values.argmax().item()
        # During training, use epsilon-greedy policy
        else:
            return random.randint(0, self.action_dim - 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Parameters:
        -----------
        state : torch.Tensor
            Current state representation
        action : int
            Selected action
        reward : float
            Reward received
        next_state : torch.Tensor
            Next state representation
        done : bool
            Whether the episode is done
        """
        self.replay_buffer.append(Experience(state, action, reward, next_state, done))
    
    def update(self):
        """Update the Q-Network using experience replay"""
        # Check if enough experiences are available
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences
        experiences = random.sample(self.replay_buffer, self.batch_size)
        
        # Convert batch of experiences to tensors
        states = torch.stack([e.state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using Double DQN
        # 1. Get actions from current policy network
        next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
        # 2. Get Q-values from target network for those actions
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Track metrics
        self.losses.append(loss.item())
        
        # Update target network periodically
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilons.append(self.epsilon)
        
        # Increment step counter
        self.steps += 1
    
    def save(self, path):
        """
        Save the agent's state.
        
        Parameters:
        -----------
        path : str
            Path to save the agent
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'losses': self.losses,
            'rewards': self.rewards,
            'epsilons': self.epsilons
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path):
        """
        Load the agent's state.
        
        Parameters:
        -----------
        path : str
            Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.losses = checkpoint['losses']
        self.rewards = checkpoint['rewards'] 
        self.epsilons = checkpoint['epsilons']
        logger.info(f"Agent loaded from {path}")
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plots
        """
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(3, 1, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        
        # Plot rewards
        plt.subplot(3, 1, 2)
        plt.plot(self.rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        
        # Plot epsilon
        plt.subplot(3, 1, 3)
        plt.plot(self.epsilons)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Update Steps')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics plot saved to {save_path}")
        
        plt.show() 