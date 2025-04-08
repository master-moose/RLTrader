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
from torch.amp import GradScaler  # Import from torch.amp instead of torch.cuda.amp
from torch.amp import autocast  # Use the new location for autocast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def get_compatible_device(device_str=None):
    """
    Get a compatible device for PyTorch operations.
    Checks CUDA compatibility and falls back to CPU if needed.
    
    Parameters:
    -----------
    device_str : str, optional
        Device string to use if provided
        
    Returns:
    --------
    torch.device
        Compatible device for PyTorch operations
    """
    if device_str is not None:
        requested_device = torch.device(device_str)
        if requested_device.type == 'cuda':
            # Check if the requested CUDA device is compatible
            if torch.cuda.is_available():
                # Get the current CUDA device capabilities
                current_capability = torch.cuda.get_device_capability()
                logger.info(f"CUDA device capability: {current_capability}")
                
                # Check if the device is compatible with PyTorch
                try:
                    # Try a simple CUDA operation to test compatibility
                    test_tensor = torch.tensor([1.0], device=requested_device)
                    logger.info(f"Using requested CUDA device: {torch.cuda.get_device_name(0)}")
                    return requested_device
                except RuntimeError as e:
                    logger.warning(f"Requested CUDA device not compatible: {str(e)}")
                    logger.warning("Falling back to CPU")
                    return torch.device("cpu")
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU")
                return torch.device("cpu")
        return requested_device
    
    # If no device specified, try CUDA first, then fall back to CPU
    if torch.cuda.is_available():
        try:
            # Try a simple CUDA operation to test compatibility
            test_tensor = torch.tensor([1.0], device="cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            return torch.device("cuda")
        except RuntimeError as e:
            logger.warning(f"CUDA device not compatible: {str(e)}")
            logger.warning("Falling back to CPU")
            return torch.device("cpu")
    
    logger.info("Using CPU device")
    return torch.device("cpu")

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
        
        # Create a deeper network with dropout for better regularization
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),  # Add intermediate layer with fewer neurons
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure input is 2D for batched processing if it's not already
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension if missing
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
        device: str = None,
        use_amp: bool = True  # New parameter to enable/disable AMP
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
        use_amp : bool
            Whether to use Automatic Mixed Precision for faster training
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
        self.use_amp = use_amp
        
        # Determine device using the compatibility function
        self.device = get_compatible_device(device)
            
        # Initialize AMP GradScaler if we're using CUDA
        self.scaler = None
        # First store the adjusted learning rate but don't try to create optimizer yet
        if self.use_amp and str(self.device).startswith("cuda"):
            # Adjust learning rate for AMP stability up front
            if self.learning_rate > 0.0001:
                logger.info(f"Reducing learning rate for AMP stability: {self.learning_rate} -> {self.learning_rate * 0.6}")
                self.learning_rate *= 0.6

        logger.info(f"Using device: {self.device}")
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is used for evaluation only
        
        # Initialize optimizer - now that q_network exists
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Now initialize the GradScaler AFTER networks and optimizer exist
        if self.use_amp and str(self.device).startswith("cuda"):
            try:
                # Use the new API location (torch.amp instead of torch.cuda.amp)
                self.scaler = GradScaler()
                logger.info("Using Automatic Mixed Precision (AMP)")
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            except Exception as e:
                logger.warning(f"Failed to initialize AMP GradScaler: {str(e)}")
                logger.warning("Falling back to full precision training")
                self.scaler = None
        elif self.use_amp:
            logger.warning("AMP requested but not using CUDA device - falling back to full precision")
        
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
                # Ensure state has proper dimensions
                if state.dim() == 1:
                    state = state.unsqueeze(0)  # Add batch dimension
                q_values = self.q_network(state)
                self.q_network.train()
                return q_values.argmax().item()
        # During training with exploration, use epsilon-greedy policy
        else:
            return random.randint(0, self.action_dim - 1)
    
    def store_transitions_batch(self, states, actions, rewards, next_states, dones):
        """
        Store a batch of transitions in the replay buffer at once.
        
        Parameters:
        -----------
        states : torch.Tensor
            Batch of current states 
        actions : torch.Tensor
            Batch of selected actions
        rewards : torch.Tensor
            Batch of rewards received
        next_states : torch.Tensor
            Batch of next states
        dones : torch.Tensor
            Batch of episode termination flags
        """
        # Add experiences to buffer in batch
        batch_size = states.shape[0]
        for i in range(batch_size):
            self.replay_buffer.append(Experience(
                states[i],
                actions[i].item() if torch.is_tensor(actions[i]) else actions[i],
                rewards[i].item() if torch.is_tensor(rewards[i]) else rewards[i],
                next_states[i],
                bool(dones[i].item()) if torch.is_tensor(dones[i]) else bool(dones[i])
            ))
    
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
        
        # Stack tensors - this already creates batch dimension (dimension 0)
        states = torch.stack(padded_states)
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(self.device)
        
        # Get rewards and check for numerical issues
        rewards_list = []
        for exp in experiences:
            # Apply additional safety clipping to handle extreme reward values
            if not np.isfinite(exp.reward):
                rewards_list.append(0.0)  # Replace NaN/Inf with 0
                logger.warning(f"Non-finite reward detected: {exp.reward}, replacing with 0")
            else:
                # Still apply clipping as a safety measure
                reward = np.clip(exp.reward, -10.0, 10.0)
                rewards_list.append(reward)
        
        rewards = torch.tensor(rewards_list, dtype=torch.float).to(self.device)
        next_states = torch.stack(padded_next_states)
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float).to(self.device)

        # AMP training path
        if self.scaler is not None:
            # Zero gradients for next round
            self.optimizer.zero_grad()
            
            # Use mixed precision for forward pass - using the new API
            with autocast(device_type='cuda', dtype=torch.float16):
                # Pre-normalize inputs for better numerical stability
                states = states / (states.abs().max().detach() + 1e-8) if states.abs().max() > 1.0 else states
                next_states = next_states / (next_states.abs().max().detach() + 1e-8) if next_states.abs().max() > 1.0 else next_states
                
                # Compute current Q values - states already has batch dimension
                q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute next Q values using Double DQN method - next_states already has batch dimension
                with torch.no_grad():
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
                
                # Apply reward and Q-value scaling for better numerical stability
                rewards_scaled = rewards.clamp(-10.0, 10.0)  # Ensure rewards are in reasonable range
                next_q_values_scaled = next_q_values.clamp(-100.0, 100.0)  # Prevent extreme Q-values
                
                # Compute target Q values with clamped inputs
                target_q_values = rewards_scaled + (1 - dones) * self.gamma * next_q_values_scaled
                
                # Additional safety check for NaN/Inf before computing loss
                if torch.isnan(target_q_values).any() or torch.isinf(target_q_values).any():
                    logger.warning("NaN or Inf detected in target Q-values, skipping update")
                    return 0.0
                
                # Use Huber loss instead of MSE for better stability with outliers
                loss = nn.SmoothL1Loss()(q_values, target_q_values.detach())
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss detected: {loss.item()}, skipping update")
                return 0.0
            
            # Use scaler to automatically scale gradients for mixed precision
            self.scaler.scale(loss).backward()
            
            # Apply gradient clipping to prevent exploding gradients (on scaled gradients)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)  # Reduced from 1.0 for stability
            
            # Perform scaled optimizer step and update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        else:
            # Original full-precision path
            # Zero gradients for next round
            self.optimizer.zero_grad()
            
            # Compute current Q values - states already has batch dimension
            q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute next Q values using Double DQN method - next_states already has batch dimension
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Check for NaN values in target Q-values
            if torch.isnan(target_q_values).any() or torch.isinf(target_q_values).any():
                logger.warning("NaN or Inf detected in target Q-values, skipping update")
                return 0.0
            
            # Compute loss
            loss = nn.MSELoss()(q_values, target_q_values.detach())
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss detected: {loss.item()}, skipping update")
                return 0.0
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()
        
        # Store loss for tracking
        self.losses.append(loss.detach().item())
        
        return loss.detach().item()
    
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
            'epsilons': self.epsilons,
            'use_amp': self.use_amp,
            'scaler': self.scaler.state_dict() if self.scaler is not None else None
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
        
        # Load AMP state if available
        self.use_amp = checkpoint.get('use_amp', False)
        if self.use_amp and str(self.device).startswith("cuda"):
            self.scaler = GradScaler()
            if checkpoint.get('scaler') is not None:
                self.scaler.load_state_dict(checkpoint['scaler'])
        
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