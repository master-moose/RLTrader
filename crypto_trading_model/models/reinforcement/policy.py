import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import gym

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasePolicy(nn.Module):
    """
    Base policy class for reinforcement learning agents
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Forward pass"""
        raise NotImplementedError
    
    def act(self, obs, deterministic=False):
        """Return action given observation"""
        raise NotImplementedError
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions given observations"""
        raise NotImplementedError
    
    def save(self, path):
        """Save policy to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load policy from path"""
        self.load_state_dict(torch.load(path))


class MLPPolicy(BasePolicy):
    """
    Policy network using Multi-Layer Perceptron
    """
    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 hidden_sizes: List[int] = [128, 64],
                 activation_fn: nn.Module = nn.ReLU,
                 use_batch_norm: bool = False):
        """
        Initialize MLP policy
        
        Parameters:
        - observation_space: Observation space
        - action_space: Action space
        - hidden_sizes: List of hidden layer sizes
        - activation_fn: Activation function
        - use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.observation_dim = observation_space.shape[0]
        self.action_dim = action_space.n
        
        # Build MLP policy network
        layers = []
        prev_size = self.observation_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(activation_fn())
            prev_size = size
        
        # Output layer for action probabilities
        layers.append(nn.Linear(prev_size, self.action_dim))
        
        self.policy_net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through policy network
        
        Parameters:
        - x: Input tensor of shape (batch_size, observation_dim)
        
        Returns:
        - action_probs: Action probabilities
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Get action logits
        logits = self.policy_net(x)
        
        # Convert to probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs
    
    def act(self, obs, deterministic=False):
        """
        Return action given observation
        
        Parameters:
        - obs: Observation
        - deterministic: Whether to use deterministic actions
        
        Returns:
        - action: Selected action
        """
        with torch.no_grad():
            action_probs = self.forward(obs)
            
            if deterministic:
                # Take most probable action
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from probability distribution
                action = torch.multinomial(action_probs, num_samples=1).item()
            
            return action
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions given observations
        
        Parameters:
        - obs: Observations
        - actions: Actions to evaluate
        
        Returns:
        - log_probs: Log probabilities of actions
        - entropy: Policy entropy
        """
        action_probs = self.forward(obs)
        
        # Convert actions to tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.LongTensor(actions)
        
        # Get log probabilities of selected actions
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        
        # Calculate entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        
        return log_probs, entropy


class LSTMPolicy(BasePolicy):
    """
    Policy network using LSTM for sequential data
    """
    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 window_size: int = 50,
                 feature_dim: int = None):
        """
        Initialize LSTM policy
        
        Parameters:
        - observation_space: Observation space
        - action_space: Action space
        - hidden_size: Size of LSTM hidden states
        - num_layers: Number of LSTM layers
        - dropout: Dropout probability
        - bidirectional: Whether to use bidirectional LSTM
        - window_size: Size of input sequence
        - feature_dim: Dimension of input features (per timestep)
        """
        super().__init__()
        
        # Determine feature dimensions
        if feature_dim is None:
            if len(observation_space.shape) == 1:
                # Flat observation: reshape to (window_size, feature_dim)
                total_dim = observation_space.shape[0]
                self.feature_dim = total_dim // window_size
                self.window_size = window_size
            else:
                # Already has sequence shape
                self.window_size = observation_space.shape[0]
                self.feature_dim = observation_space.shape[1]
        else:
            self.feature_dim = feature_dim
            self.window_size = window_size
        
        self.action_dim = action_space.n
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer for action probabilities
        lstm_output_dim = hidden_size * self.num_directions
        self.policy_head = nn.Linear(lstm_output_dim, self.action_dim)
        
        # Initialize hidden state and cell state
        self.hidden = None
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state and cell state
        
        Parameters:
        - batch_size: Batch size
        - device: Device to create tensors on
        """
        self.hidden = (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass through LSTM policy network
        
        Parameters:
        - x: Input tensor of shape (batch_size, window_size, feature_dim) or (batch_size, window_size*feature_dim)
        - hidden: Initial hidden state or None
        
        Returns:
        - action_probs: Action probabilities
        - hidden: Updated hidden state
        """
        # Convert to tensor if numpy array
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        batch_size = x.size(0)
        
        # Reshape if needed
        if len(x.shape) == 2:
            # Reshape from (batch_size, window_size*feature_dim) to (batch_size, window_size, feature_dim)
            x = x.view(batch_size, self.window_size, self.feature_dim)
        
        # Initialize hidden state if needed
        if hidden is None:
            if self.hidden is None:
                self.init_hidden(batch_size, x.device)
            hidden = self.hidden
        
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Only use output from the last time step
        if self.bidirectional:
            # For bidirectional LSTM, concatenate forward and backward outputs
            last_time_step = lstm_out[:, -1, :]
        else:
            # For unidirectional LSTM, just take the last output
            last_time_step = lstm_out[:, -1, :]
        
        # Get action logits
        logits = self.policy_head(last_time_step)
        
        # Convert to probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        # Update hidden state
        self.hidden = hidden
        
        return action_probs, hidden
    
    def act(self, obs, deterministic=False):
        """
        Return action given observation
        
        Parameters:
        - obs: Observation
        - deterministic: Whether to use deterministic actions
        
        Returns:
        - action: Selected action
        """
        # Add batch dimension if needed
        if len(obs.shape) == 1 or (len(obs.shape) == 2 and obs.shape[0] != 1):
            obs = obs.reshape(1, -1)
        
        with torch.no_grad():
            action_probs, _ = self.forward(obs)
            
            if deterministic:
                # Take most probable action
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from probability distribution
                action = torch.multinomial(action_probs, num_samples=1).item()
            
            return action
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions given observations
        
        Parameters:
        - obs: Observations
        - actions: Actions to evaluate
        
        Returns:
        - log_probs: Log probabilities of actions
        - entropy: Policy entropy
        """
        action_probs, _ = self.forward(obs)
        
        # Convert actions to tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.LongTensor(actions)
        
        # Get log probabilities of selected actions
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        
        # Calculate entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        
        return log_probs, entropy
    
    def reset_hidden(self):
        """Reset hidden state"""
        self.hidden = None


class MultiTimeframePolicy(BasePolicy):
    """
    Policy network for multi-timeframe data
    
    Processes multiple timeframes with separate LSTMs and combines their outputs
    """
    def __init__(self,
                 observation_space: Dict[str, gym.spaces.Box],
                 action_space: gym.spaces.Discrete,
                 timeframes: List[str],
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 use_attention: bool = True):
        """
        Initialize multi-timeframe policy
        
        Parameters:
        - observation_space: Dictionary of observation spaces for each timeframe
        - action_space: Action space
        - timeframes: List of timeframes
        - hidden_size: Size of LSTM hidden states
        - num_layers: Number of LSTM layers
        - dropout: Dropout probability
        - use_attention: Whether to use attention to combine timeframe outputs
        """
        super().__init__()
        
        self.timeframes = timeframes
        self.action_dim = action_space.n
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Create LSTM for each timeframe
        self.lstms = nn.ModuleDict()
        for tf in timeframes:
            # Get observation space for this timeframe
            obs_space = observation_space[tf]
            window_size = obs_space.shape[0]
            feature_dim = obs_space.shape[1]
            
            # Create LSTM
            self.lstms[tf] = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Attention layer for combining outputs
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Tanh()
            )
        
        # Output layer for combined features
        combined_size = hidden_size
        self.policy_head = nn.Linear(combined_size, self.action_dim)
        
        # Initialize hidden states
        self.hidden_states = {tf: None for tf in timeframes}
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden state and cell state for all LSTMs
        
        Parameters:
        - batch_size: Batch size
        - device: Device to create tensors on
        """
        for tf in self.timeframes:
            self.hidden_states[tf] = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            )
    
    def forward(self, x, hidden=None):
        """
        Forward pass through multi-timeframe policy network
        
        Parameters:
        - x: Dictionary of input tensors for each timeframe
        - hidden: Dictionary of initial hidden states or None
        
        Returns:
        - action_probs: Action probabilities
        - hidden: Updated hidden states
        """
        batch_size = None
        device = None
        
        # Get representations from each timeframe
        tf_outputs = []
        new_hidden_states = {}
        
        for tf in self.timeframes:
            # Get input for this timeframe
            tf_input = x[tf]
            
            # Convert to tensor if numpy array
            if isinstance(tf_input, np.ndarray):
                tf_input = torch.FloatTensor(tf_input)
            
            # Get batch size and device
            if batch_size is None:
                batch_size = tf_input.size(0)
                device = tf_input.device
            
            # Get hidden state for this timeframe
            tf_hidden = None
            if hidden is not None:
                tf_hidden = hidden[tf]
            elif self.hidden_states[tf] is not None:
                tf_hidden = self.hidden_states[tf]
            
            # Forward pass through LSTM
            lstm_out, new_hidden = self.lstms[tf](tf_input, tf_hidden)
            
            # Only use output from the last time step
            last_time_step = lstm_out[:, -1, :]
            
            # Store outputs and hidden states
            tf_outputs.append(last_time_step)
            new_hidden_states[tf] = new_hidden
        
        # Combine outputs from all timeframes
        if self.use_attention:
            # Use attention to weight outputs
            attention_scores = [self.attention(output) for output in tf_outputs]
            attention_weights = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
            
            # Apply attention weights
            weighted_outputs = torch.stack([
                tf_outputs[i] * attention_weights[:, i].unsqueeze(1)
                for i in range(len(tf_outputs))
            ], dim=1)
            
            # Sum weighted outputs
            combined_output = torch.sum(weighted_outputs, dim=1)
        else:
            # Simple average of outputs
            combined_output = torch.mean(torch.stack(tf_outputs), dim=0)
        
        # Get action logits
        logits = self.policy_head(combined_output)
        
        # Convert to probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        # Update hidden states
        self.hidden_states = new_hidden_states
        
        return action_probs, new_hidden_states
    
    def act(self, obs, deterministic=False):
        """
        Return action given observation
        
        Parameters:
        - obs: Dictionary of observations for each timeframe
        - deterministic: Whether to use deterministic actions
        
        Returns:
        - action: Selected action
        """
        # Add batch dimension if needed
        batch_obs = {}
        for tf in self.timeframes:
            tf_obs = obs[tf]
            if len(tf_obs.shape) == 2:  # (window_size, feature_dim)
                tf_obs = tf_obs.reshape(1, tf_obs.shape[0], tf_obs.shape[1])
            batch_obs[tf] = tf_obs
        
        with torch.no_grad():
            action_probs, _ = self.forward(batch_obs)
            
            if deterministic:
                # Take most probable action
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from probability distribution
                action = torch.multinomial(action_probs, num_samples=1).item()
            
            return action
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions given observations
        
        Parameters:
        - obs: Dictionary of observations for each timeframe
        - actions: Actions to evaluate
        
        Returns:
        - log_probs: Log probabilities of actions
        - entropy: Policy entropy
        """
        action_probs, _ = self.forward(obs)
        
        # Convert actions to tensor if needed
        if isinstance(actions, np.ndarray):
            actions = torch.LongTensor(actions)
        
        # Get log probabilities of selected actions
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1))
        
        # Calculate entropy
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1)
        
        return log_probs, entropy
    
    def reset_hidden(self):
        """Reset all hidden states"""
        for tf in self.timeframes:
            self.hidden_states[tf] = None 