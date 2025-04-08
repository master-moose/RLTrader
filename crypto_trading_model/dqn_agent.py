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
from torch.amp import GradScaler
import torch.nn.functional as F

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
        
        # First layer is a special case - use 1D convolution for feature extraction
        # if state dimension is large enough
        if state_dim > 50:  # Only use 1D conv for higher dimensional states
            # 1D Convolution for feature extraction - good for sequence data
            self.feature_extractor = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            )
            prev_dim = hidden_dims[0]
            hidden_dims = hidden_dims[1:]  # Skip first hidden dim since we used it
        else:
            self.feature_extractor = None
        
        # Build the rest of the network
        for i, hidden_dim in enumerate(hidden_dims):
            # Use LayerNorm instead of BatchNorm for better stability
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Handle 1D state tensors by adding batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Apply feature extraction if available
        if self.feature_extractor is not None:
            state = self.feature_extractor(state)
            
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True
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
        verbose : bool
            Whether to log target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer with weight decay (L2 regularization) - helps generalization
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            eps=1e-5,  # Increase epsilon for better numerical stability 
            weight_decay=1e-5,  # Small weight decay for regularization
            amsgrad=True  # Use AMSGrad variant for more stable updates
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=10000,
            eta_min=learning_rate/10
        )
        
        # Initialize AMP GradScaler if using CUDA
        if str(device).startswith("cuda"):
            self.scaler = GradScaler()
            # Enable TF32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.scaler = None
        
        # Initialize replay buffer with efficient data structure
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
        
        # Initialize verbose mode
        self.verbose = verbose
        
        # Pre-allocate tensors for faster training - dynamically sized
        max_batch = min(batch_size, 8192)  # Reasonable upper limit
        self.state_tensor = torch.zeros((max_batch, state_dim), device=device)
        self.action_tensor = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.reward_tensor = torch.zeros(max_batch, device=device)
        self.next_state_tensor = torch.zeros((max_batch, state_dim), device=device)
        self.done_tensor = torch.zeros(max_batch, dtype=torch.bool, device=device)
        
        # Implement Prioritized Experience Replay (PER) if buffer size is large enough
        self.use_per = buffer_size >= 10000
        if self.use_per:
            self.priorities = np.zeros(buffer_size)
            self.alpha = 0.6  # Priority exponent
            self.beta = 0.4   # Importance sampling exponent
            self.beta_increment = 0.001  # Beta increment per sampling
            self.epsilon_per = 1e-6  # Small constant to avoid zero priority
        
        # Track metrics
        self.loss_history = []
        self.q_value_history = []
        
        # Enable jit scripting for target network evaluation if available
        try:
            self.target_net_opt = torch.jit.script(self.target_net)
            self.use_jit = True
            if self.verbose:
                self.logger.info("Using JIT optimization for target network")
        except:
            self.target_net_opt = self.target_net
            self.use_jit = False
            if self.verbose:
                self.logger.info("JIT optimization not available, using standard execution")
    
    def select_action(
        self, 
        state: Union[np.ndarray, torch.Tensor], 
        training: bool = True
    ) -> int:
        """
        Select an action using epsilon-greedy policy with exploration noise.
        
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
        # Exploration with probability epsilon during training
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            # Convert state to tensor if it's not already
            if not torch.is_tensor(state):
                state = torch.FloatTensor(state).to(self.device)
            elif state.device != self.device:
                state = state.to(self.device)
            
            # Set network to evaluation mode for inference
            self.policy_net.eval()
            
            # Add noise to q-values for better exploration (only during training)
            q_values = self.policy_net(state)
            if training:
                # Add small Gaussian noise to q-values for exploration
                noise = torch.randn_like(q_values) * 0.01
                q_values = q_values + noise
            
            # If in training mode, set the network back to train mode
            if training:
                self.policy_net.train()
                
            return q_values.argmax().item()
    
    def sample_batch(self):
        """Sample a batch from replay buffer with optional prioritization"""
        if not self.use_per or len(self.memory) > self.priorities.shape[0]:
            # Standard uniform sampling
            return random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # Update beta value for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.memory), 
            min(len(self.memory), self.batch_size), 
            replace=False, 
            p=probs
        )
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get samples
        batch = [self.memory[i] for i in indices]
        
        return batch, indices, torch.FloatTensor(weights).to(self.device)
    
    def update_priorities(self, indices, td_errors):
        """Update priorities in the replay buffer"""
        if not self.use_per:
            return
            
        # Update priorities based on TD error
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) + self.epsilon_per
    
    def update(self) -> float:
        """
        Update the policy network using a batch of experiences from the replay buffer.
        Returns the loss value.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample a batch of experiences with prioritization if enabled
        if self.use_per:
            batch, indices, weights = self.sample_batch()
        else:
            batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
            weights = torch.ones(len(batch), device=self.device)
        
        # Convert batch to tensors efficiently
        current_batch_size = len(batch)
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            self.state_tensor[i] = (
                torch.from_numpy(state).float() 
                if isinstance(state, np.ndarray) else state
            )
            self.action_tensor[i] = action
            self.reward_tensor[i] = reward
            self.next_state_tensor[i] = (
                torch.from_numpy(next_state).float() 
                if isinstance(next_state, np.ndarray) else next_state
            )
            self.done_tensor[i] = done
        
        # Compute current Q values using policy network
        # Use mixed precision for faster computation
        with torch.amp.autocast('cuda'):
            # Get current Q values
            current_q_values = self.policy_net(
                self.state_tensor[:current_batch_size]
            ).gather(
                1, self.action_tensor[:current_batch_size].unsqueeze(1)
            )
            
            # Compute next Q values using target network with Double DQN
            with torch.no_grad():
                # Get actions from policy network (Double DQN)
                next_actions = self.policy_net(
                    self.next_state_tensor[:current_batch_size]
                ).argmax(dim=1, keepdim=True)
                
                # Get Q-values from target network for those actions
                next_q_values = self.target_net_opt(
                    self.next_state_tensor[:current_batch_size]
                ).gather(1, next_actions)
                
                # Compute target Q values
                target_q_values = (
                    self.reward_tensor[:current_batch_size].unsqueeze(1) + 
                    (1 - self.done_tensor[:current_batch_size].float().unsqueeze(1)) * 
                    self.gamma * next_q_values
                )
            
            # Compute loss with prioritized weights if using PER
            td_errors = target_q_values - current_q_values
            loss = (weights.unsqueeze(1) * F.smooth_l1_loss(
                current_q_values, target_q_values, reduction='none'
            )).mean()
        
        # Update priorities in replay buffer if using PER
        if self.use_per:
            self.update_priorities(
                indices, 
                td_errors.abs().detach().cpu().numpy().flatten()
            )
        
        # Optimize
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.use_jit:
                # Update JIT compiled version
                self.target_net_opt = torch.jit.script(self.target_net)
            if self.verbose:
                self.logger.info(
                    f"Target network updated at step {self.steps_done}"
                )
        
        # Track metrics
        self.loss_history.append(loss.detach().item())
        with torch.no_grad():
            self.q_value_history.append(
                current_q_values.mean().detach().item()
            )
        
        return loss.detach().item()
    
    def save(self, path: str):
        """Save the model"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history
        }
        torch.save(checkpoint, path)
        
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint['loss_history']
        self.q_value_history = checkpoint['q_value_history']
        
        # Update JIT compiled version if available
        if self.use_jit:
            self.target_net_opt = torch.jit.script(self.target_net)