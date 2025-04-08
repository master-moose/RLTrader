#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Q-Network (DQN) agent for cryptocurrency trading.
Uses pre-trained LSTM model for state representation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Union
import random
from collections import deque
import logging
from torch.amp import GradScaler, autocast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    """
    Neural network for Q-value approximation.
    Takes the state representation from LSTM and outputs Q-values for each action.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128]
    ):
        """
        Initialize the Q-Network.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space (number of possible actions)
        hidden_dims : List[int]
            List of hidden layer sizes
        """
        super(QNetwork, self).__init__()
        
        # Build network layers dynamically
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),  # Add dropout for regularization
                nn.BatchNorm1d(hidden_dim)  # Add batch normalization
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Handle 1D state tensors by adding batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
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
        hidden_dims: List[int] = [256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DQN agent.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Dimension of the action space (number of possible actions)
        hidden_dims : List[int]
            List of hidden layer sizes
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
        buffer_size : int
            Size of replay buffer
        batch_size : int
            Mini-batch size for training
        target_update : int
            Number of steps between updates to target network
        device : str
            Device to use for training ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        # Initialize AMP GradScaler if using CUDA
        if str(device).startswith("cuda"):
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Initialize replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        # Initialize exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize training parameters
        self.gamma = gamma
        self.target_update = target_update
        self.steps_done = 0
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def select_action(self, state: Union[np.ndarray, torch.Tensor], training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : Union[np.ndarray, torch.Tensor]
            Current state representation
        training : bool
            Whether to use exploration (epsilon-greedy) or exploitation (greedy)
            
        Returns:
        --------
        int
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            # Convert state to tensor if it's not already
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            elif state.device != self.device:
                state = state.to(self.device)
            
            # Set network to evaluation mode for inference to handle batch norm with single samples
            self.policy_net.eval()
            q_values = self.policy_net(state)
            
            # If in training mode, set the network back to train mode
            if training:
                self.policy_net.train()
                
            return q_values.argmax().item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store a transition in the replay buffer.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state
        done : bool
            Whether the episode is done
        """
        # Convert tensors to numpy arrays if needed
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        if torch.is_tensor(next_state):
            next_state = next_state.cpu().numpy()
            
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self) -> float:
        """
        Update the Q-Network using experience replay.
        
        Returns:
        --------
        float
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Process batch data, ensuring proper conversion to tensors
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        
        for state, action, reward, next_state, done in batch:
            # Convert state to numpy if it's a tensor
            if torch.is_tensor(state):
                state = state.cpu().numpy()
            if torch.is_tensor(next_state):
                next_state = next_state.cpu().numpy()
                
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        # Convert to tensors and move to device
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(np.array(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(self.device)
        
        # Use AMP context manager if available
        if self.scaler is not None:
            # Get the device type as a string (e.g., 'cuda', 'cpu')
            device_type = 'cuda' if str(self.device).startswith('cuda') else 'cpu'
            
            with autocast(device_type=device_type):
                # Compute current Q values
                current_q_values = self.policy_net(state_batch).gather(
                    1, action_batch.unsqueeze(1)
                )
                
                # Compute target Q values
                with torch.no_grad():
                    # Always use eval mode for target_net
                    self.target_net.eval()
                    next_q_values = self.target_net(next_state_batch).max(1)[0]
                    target_q_values = (reward_batch + 
                                     (1 - done_batch) * self.gamma * next_q_values)
                
                # Compute loss
                loss = nn.MSELoss()(
                    current_q_values.squeeze(),
                    target_q_values
                )
            
            # Optimizer step with scaler
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training without AMP
            # Compute current Q values
            current_q_values = self.policy_net(state_batch).gather(
                1, action_batch.unsqueeze(1)
            )
            
            # Compute target Q values
            with torch.no_grad():
                # Always use eval mode for target_net
                self.target_net.eval()
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                target_q_values = (reward_batch + 
                                 (1 - done_batch) * self.gamma * next_q_values)
            
            # Compute loss and update
            loss = nn.MSELoss()(
                current_q_values.squeeze(),
                target_q_values
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.logger.info(
                f"Target network updated at step {self.steps_done}"
            )
        
        return loss.detach().item()
    
    def save(self, path: str) -> None:
        """
        Save the agent's state.
        
        Parameters:
        -----------
        path : str
            Path to save the agent
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent's state from a checkpoint.
        
        Parameters:
        -----------
        path : str
            Path to load the agent
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        if self.scaler and checkpoint.get('scaler'):
            self.scaler.load_state_dict(checkpoint['scaler'])
        self.logger.info(f"Model loaded from {path}")