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
        
        # Add a flattening layer to handle potentially multidimensional input
        self.network = nn.Sequential(
            nn.Flatten(),  # Flatten any multi-dimensional input
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
        lstm_model,
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
        lstm_model : LightningTimeSeriesModel
            Pre-trained LSTM model for state representation
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
        self.lstm_model = lstm_model
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
        
        # Set LSTM model to evaluation mode
        if self.lstm_model:
            self.lstm_model.eval()
            self.lstm_model.to(self.device)
            logger.info("LSTM model set to evaluation mode")
        
        # Metrics for tracking
        self.losses = []
        self.rewards = []
        self.epsilons = []
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.
        
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
        # Validate inputs to ensure they're tensors with expected dimensions
        if not isinstance(state, torch.Tensor) or state.dim() != 1:
            logger.warning(f"Invalid state shape: {state.shape if hasattr(state, 'shape') else 'not a tensor'}")
            return
            
        if not isinstance(next_state, torch.Tensor) or next_state.dim() != 1:
            logger.warning(f"Invalid next_state shape: {next_state.shape if hasattr(next_state, 'shape') else 'not a tensor'}")
            return
        
        # For debugging purposes
        if len(self.replay_buffer) > 0 and len(self.replay_buffer) % 100 == 0:
            # Get sample dimensions from buffer
            sample_state = self.replay_buffer[0].state
            if state.shape[0] != sample_state.shape[0]:
                logger.info(f"State dimension change detected: {sample_state.shape[0]} -> {state.shape[0]}")
        
        self.replay_buffer.append(Experience(state, action, reward, next_state, done))
    
    def select_action(self, state, explore=True):
        """
        Select an action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : torch.Tensor
            Current state representation
        explore : bool
            Whether to use exploration (epsilon-greedy) or exploitation (greedy)
            
        Returns:
        --------
        int
            Selected action index
        """
        # During evaluation or exploitation, use greedy policy
        if not explore or random.random() > self.epsilon:
            with torch.no_grad():
                self.q_network.eval()
                q_values = self.q_network(state)
                self.q_network.train()
                return q_values.argmax().item()
        # During training with exploration, use epsilon-greedy policy
        else:
            return random.randint(0, self.action_dim - 1)
    
    def update(self):
        """
        Update the Q-Network using experience replay.
        
        Returns:
        --------
        float
            Loss value
        """
        # Check if enough experiences are available
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample a batch of experiences
        experiences = random.sample(self.replay_buffer, self.batch_size)
        
        # Check for state dimension consistency and find max dimension
        state_dims = [exp.state.shape[0] for exp in experiences]
        max_dim = max(state_dims)
        
        # Pad states if necessary to ensure consistent dimensions
        padded_states = []
        padded_next_states = []
        
        for exp in experiences:
            # Handle state
            if exp.state.shape[0] < max_dim:
                # Pad state with zeros
                padded_state = torch.zeros(max_dim, dtype=torch.float32, device=self.device)
                padded_state[:exp.state.shape[0]] = exp.state
                padded_states.append(padded_state)
            else:
                padded_states.append(exp.state)
            
            # Handle next_state
            if exp.next_state.shape[0] < max_dim:
                # Pad next_state with zeros
                padded_next_state = torch.zeros(max_dim, dtype=torch.float32, device=self.device)
                padded_next_state[:exp.next_state.shape[0]] = exp.next_state
                padded_next_states.append(padded_next_state)
            else:
                padded_next_states.append(exp.next_state)
        
        # Stack tensors
        states = torch.stack(padded_states)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float).to(self.device)
        next_states = torch.stack(padded_next_states)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using Double DQN method
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()
        
        # Track metrics
        self.losses.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
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
    
    def load_state_dict(self, checkpoint):
        """
        Load the agent's state from a checkpoint.
        
        Parameters:
        -----------
        checkpoint : dict
            Checkpoint dictionary containing agent's state
        """
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.losses = checkpoint.get('losses', [])
        self.rewards = checkpoint.get('rewards', [])
        self.epsilons = checkpoint.get('epsilons', [])
        logger.info("Agent state loaded successfully")
    
    def plot_metrics(self, save_path=None):
        """
        Plot training metrics.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plots, or None to display
        """
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(3, 1, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot rewards
        plt.subplot(3, 1, 2)
        plt.plot(self.rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot epsilon
        plt.subplot(3, 1, 3)
        plt.plot(self.epsilons)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show() 