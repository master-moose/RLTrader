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
        # Ensure shape is (batch_size, channels, sequence_length)
        if len(x.shape) != 3:
            print(f"Warning: Expected 3D input (batch_size, channels, sequence_length), got {x.shape}")
            if len(x.shape) == 2:
                # If 2D, assume (batch_size, features) and reshape to (batch_size, 1, features)
                x = x.unsqueeze(1)
                print(f"Reshaped to {x.shape}")
            elif len(x.shape) > 3:
                # If more than 3D, try to flatten extra dimensions
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1, x.shape[-1])
                print(f"Reshaped to {x.shape}")
        
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
                # Calculate features_per_timestep by dividing total features by sequence_length
                features_dim = int(np.prod(observation_space.shape))
                self.features_per_timestep = features_dim // sequence_length
                print(f"Inferring features_per_timestep from observation space: {self.features_per_timestep} (total features: {features_dim}, sequence_length: {sequence_length})")
            else:
                # Default to a reasonable value if can't infer
                self.features_per_timestep = 32
                print(f"Warning: Could not infer features_per_timestep from {observation_space}. Using default: {self.features_per_timestep}")
        else:
            self.features_per_timestep = features_per_timestep
            print(f"Using provided features_per_timestep: {self.features_per_timestep}")
        
        # Store these parameters for later use in _build
        self._tcn = None
        
        # Initialize the parent class - IMPORTANT: This will call _build,
        # which will in turn call our overridden _build_mlp_extractor
        super(TcnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
    
    def _build(self, lr_schedule):
        """
        Completely override the _build method from ActorCriticPolicy to ensure
        proper initialization sequence for the TCN-based policy.
        """
        # 1. Create the underlying TCN first
        self._tcn = TCN(
            input_channels=self.features_per_timestep,
            num_filters=self.tcn_params["num_filters"],
            num_layers=self.tcn_params["num_layers"],
            kernel_size=self.tcn_params["kernel_size"],
            dropout=self.tcn_params["dropout"]
        )
        
        # 2. Create our custom mlp_extractor using the TCN
        self.mlp_extractor = self._build_mlp_extractor()
        
        # 3. Add required properties for compatibility with parent class
        self.mlp_extractor.latent_dim_pi = 64  # Output dim of policy net in TcnExtractor
        self.mlp_extractor.latent_dim_vf = 64  # Output dim of value net in TcnExtractor
        
        # 4. Create action net (for policy output) - handle different action space types
        # Get the type name as a string to handle both gym and gymnasium
        action_space_type = self.action_space.__class__.__name__
        
        if action_space_type == "Discrete":
            # For discrete actions (Discrete from gym or gymnasium)
            action_net_output_dim = self.action_space.n
        elif action_space_type == "Box":
            # For continuous actions (Box from gym or gymnasium)
            action_net_output_dim = int(np.prod(self.action_space.shape))
        elif action_space_type == "MultiDiscrete":
            # For multi-discrete actions
            action_net_output_dim = sum(self.action_space.nvec)
        elif action_space_type == "MultiBinary":
            # For multi-binary actions
            action_net_output_dim = self.action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
        
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_net_output_dim)
        
        # 5. Create value net (for value function) - same as in ActorCriticPolicy
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # 6. Initialize weights - similar to ActorCriticPolicy
        # Initialize policy weights
        module_gains = {
            self.mlp_extractor: 1.0,
            self.action_net: 0.01,
            self.value_net: 1.0,
        }
        for module, gain in module_gains.items():
            module.apply(lambda m: self.init_weights(m, gain=gain))
            
        # 7. Set up optimizer
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def tcn(self):
        """Safe access to TCN with error handling."""
        if self._tcn is None:
            raise ValueError("TCN not initialized properly. Check if tcn_params is provided correctly.")
        return self._tcn

    def _build_mlp_extractor(self):
        """Build a TCN-based feature extractor."""
        if self._tcn is None:
            raise ValueError("TCN must be initialized before building the MLP extractor.")
            
        class TcnExtractor(nn.Module):
            def __init__(self_extractor, tcn, features_per_timestep, sequence_length, features_dim):
                super(TcnExtractor, self_extractor).__init__()
                self_extractor.tcn = tcn
                self_extractor.features_per_timestep = features_per_timestep
                self_extractor.sequence_length = sequence_length
                self_extractor.features_dim = features_dim
                
                # Output size will be (batch_size, num_filters, sequence_length)
                # We need to project this to the appropriate size for actor/critic
                self_extractor.output_dim = tcn.output_dim * sequence_length
                
                # Create separate heads for policy and value
                self_extractor.policy_net = nn.Sequential(
                    nn.Linear(self_extractor.output_dim, 64),
                    nn.ReLU()
                )
                self_extractor.value_net = nn.Sequential(
                    nn.Linear(self_extractor.output_dim, 64),
                    nn.ReLU()
                )
                
            def forward(self_extractor, features):
                batch_size = features.shape[0]
                total_features = features.numel() // batch_size
                
                # Sanity check to avoid shape errors
                if total_features != self_extractor.features_per_timestep * self_extractor.sequence_length:
                    # Recalculate features_per_timestep based on the actual input
                    actual_features_per_timestep = total_features // self_extractor.sequence_length
                    print(f"Warning: Shape mismatch - expected {self_extractor.features_per_timestep} features per timestep, "
                          f"but got {actual_features_per_timestep} based on input of size {features.shape} "
                          f"and sequence length {self_extractor.sequence_length}")
                    # Update to use the corrected dimensions
                    self_extractor.features_per_timestep = actual_features_per_timestep
                
                # Reshape from (batch_size, features_dim) to 
                # (batch_size, features_per_timestep, sequence_length)
                try:
                    x = features.view(batch_size, self_extractor.features_per_timestep, self_extractor.sequence_length)
                except RuntimeError as e:
                    # If the reshape fails, try to adapt
                    print(f"Reshape error: {e}. Attempting to adapt shape...")
                    # Calculate how many timesteps we can actually fit
                    actual_features = features.shape[1]
                    actual_timesteps = actual_features // self_extractor.features_per_timestep
                    if actual_timesteps < 1:
                        # If we can't fit even one timestep, we need to reduce features_per_timestep
                        self_extractor.features_per_timestep = actual_features
                        x = features.view(batch_size, self_extractor.features_per_timestep, 1)
                        print(f"Adapted to shape: {x.shape}")
                    else:
                        # Otherwise use as many timesteps as we can
                        x = features.view(batch_size, self_extractor.features_per_timestep, actual_timesteps)
                        print(f"Adapted to shape: {x.shape}")
                
                # Apply TCN - output: (batch_size, num_filters, sequence_length)
                x = self_extractor.tcn(x)
                
                # Flatten for the MLP heads
                x_flat = x.reshape(batch_size, -1)
                
                return self_extractor.policy_net(x_flat), self_extractor.value_net(x_flat)
                
        return TcnExtractor(
            self._tcn, 
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