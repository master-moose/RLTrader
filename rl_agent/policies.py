import torch
import torch.nn as nn
import numpy as np
import gym
from typing import Dict, List, Tuple, Type, Union, Optional, Any

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


class TCN(nn.Module):
    """
    Temporal Convolutional Network with dilated causal convolutions.
    """
    def __init__(
        self, 
        input_channels: int, 
        num_filters: int, 
        num_layers: int, 
        kernel_size: int = 3, 
        dropout: float = 0.2
    ):
        super(TCN, self).__init__()
        layers = []
        # Calculate padding needed to maintain sequence length
        padding = (kernel_size - 1) * 2**(num_layers-1)
        
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_filters
            padding_size = (kernel_size - 1) * dilation_size  # Causal padding
            
            # Add padding layer for causal convolution
            layers.append(nn.ConstantPad1d((padding_size, 0), 0))
            
            # Conv1d layer with appropriate dilation
            layers.append(
                nn.Conv1d(
                    in_channels, 
                    num_filters, 
                    kernel_size, 
                    dilation=dilation_size
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = num_filters  # Output dimension per timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TcnPolicy(ActorCriticPolicy):
    """
    Policy using a Temporal Convolutional Network for feature extraction.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args,
        # TCN specific parameters
        tcn_params: Optional[Dict[str, Any]] = None,
        sequence_length: int = 60,
        features_per_timestep: Optional[int] = None,
        **kwargs
    ):
        # Initialize default TCN parameters if not provided
        self.tcn_params = {
            "num_filters": 64,
            "num_layers": 4,
            "kernel_size": 3,
            "dropout": 0.2
        }
        
        # Override defaults with provided parameters
        if tcn_params is not None:
            self.tcn_params.update(tcn_params)
            
        # Store sequence information
        self.sequence_length = sequence_length
        
        # If features_per_timestep is not provided, try to infer from observation space
        if features_per_timestep is None:
            if isinstance(observation_space, gym.spaces.Box):
                # Assume flattened sequence: features_dim = features_per_timestep * sequence_length
                features_dim = np.prod(observation_space.shape)
                self.features_per_timestep = features_dim // sequence_length
            else:
                # Default to a reasonable value if can't infer
                self.features_per_timestep = 32
                print(f"Warning: Could not infer features_per_timestep from {observation_space}. Using default: {self.features_per_timestep}")
        else:
            self.features_per_timestep = features_per_timestep
            
        # Initialize the parent class
        super(TcnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # Create the TCN after parent initialization (which sets self.features_dim)
        self.tcn = TCN(
            input_channels=self.features_per_timestep,
            num_filters=self.tcn_params["num_filters"],
            num_layers=self.tcn_params["num_layers"],
            kernel_size=self.tcn_params["kernel_size"],
            dropout=self.tcn_params["dropout"]
        )
        
        # Replace the mlp_extractor with our TCN-based feature extractor
        self.mlp_extractor = self._build_mlp_extractor()

    def _build_mlp_extractor(self):
        """Build a TCN-based feature extractor."""
        class TcnExtractor(nn.Module):
            def __init__(self, tcn, features_per_timestep, sequence_length, features_dim):
                super(TcnExtractor, self).__init__()
                self.tcn = tcn
                self.features_per_timestep = features_per_timestep
                self.sequence_length = sequence_length
                self.features_dim = features_dim
                
                # Output size will be (batch_size, num_filters, sequence_length)
                # We need to project this to the appropriate size for actor/critic
                self.output_dim = tcn.output_dim * sequence_length
                
                # Create separate heads for policy and value
                self.policy_net = nn.Sequential(
                    nn.Linear(self.output_dim, 64),
                    nn.ReLU()
                )
                self.value_net = nn.Sequential(
                    nn.Linear(self.output_dim, 64),
                    nn.ReLU()
                )
                
            def forward(self, features):
                batch_size = features.shape[0]
                
                # Reshape from (batch_size, features_dim) to 
                # (batch_size, features_per_timestep, sequence_length)
                x = features.view(batch_size, self.features_per_timestep, self.sequence_length)
                
                # Apply TCN - output: (batch_size, num_filters, sequence_length)
                x = self.tcn(x)
                
                # Flatten for the MLP heads
                x_flat = x.reshape(batch_size, -1)
                
                return self.policy_net(x_flat), self.value_net(x_flat)
                
        return TcnExtractor(
            self.tcn, 
            self.features_per_timestep, 
            self.sequence_length, 
            self.features_dim
        )

    def forward(self, obs, deterministic=False):
        """Forward pass through the network."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get values from the value head
        values = self.value_net(latent_vf)
        
        # Get distribution parameters from the policy head
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        return distribution, values

    def _get_action_dist_from_latent(self, latent_pi):
        """Get the action distribution from the latent features."""
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=mean_actions) 