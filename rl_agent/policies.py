import torch
import torch.nn as nn
import numpy as np
import gym
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import logging

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
                # If 2D, assume (batch_size, features) and reshape appropriately
                batch_size = x.shape[0]
                features = x.shape[1]
                # Try to infer sequence_length and channels
                sequence_length = self.receptive_field
                channels = features // sequence_length
                x = x.view(batch_size, channels, sequence_length)
                print(f"Reshaped 2D input to {x.shape}")
            elif len(x.shape) > 3:
                # If more than 3D, try to flatten extra dimensions
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1, x.shape[-1])
                print(f"Reshaped to {x.shape}")
        else:
            # If already 3D, check if dimensions need to be transposed
            # TCN expects (batch_size, channels, sequence_length)
            # but sometimes we get (batch_size, sequence_length, channels)
            if x.shape[1] > x.shape[2]:
                # Likely (batch, seq_len, channels), transpose to (batch, channels, seq_len)
                x = x.transpose(1, 2)
                print(f"Transposed dimensions to {x.shape}")
        
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
        sequence_length: Optional[int] = 60,
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
        
        # Dynamically infer sequence_length and features_per_timestep
        features_dim = int(np.prod(observation_space.shape))
        orig_sequence_length = sequence_length
        orig_features_per_timestep = features_per_timestep
        # Try to use provided values if possible
        if sequence_length is not None and features_per_timestep is not None:
            if sequence_length * features_per_timestep != features_dim:
                print(f"[TCNPolicy] Provided sequence_length ({sequence_length}) * features_per_timestep ({features_per_timestep}) != obs dim ({features_dim}). Will infer dynamically.")
                sequence_length = None
                features_per_timestep = None
        # If not provided or not matching, infer
        if sequence_length is None or features_per_timestep is None:
            # Try to find the largest possible sequence_length that divides features_dim
            best_seq = None
            best_feat = None
            for seq in range(features_dim, 0, -1):
                if features_dim % seq == 0:
                    best_seq = seq
                    best_feat = features_dim // seq
                    break
            if best_seq is not None:
                sequence_length = best_seq
                features_per_timestep = best_feat
                print(f"[TCNPolicy] Inferred sequence_length={sequence_length}, features_per_timestep={features_per_timestep} from obs dim {features_dim}")
            else:
                raise ValueError(f"Cannot infer sequence_length/features_per_timestep for obs dim {features_dim}")
        else:
            print(f"[TCNPolicy] Using provided sequence_length={sequence_length}, features_per_timestep={features_per_timestep}")
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep
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
                
                self_extractor.tcn = tcn  # The TCN module
                self_extractor.features_per_timestep = features_per_timestep
                self_extractor.sequence_length = sequence_length
                self_extractor.features_dim = features_dim
                
                # Calculate the output dimension after TCN processing
                # This is the number of filters * sequence_length
                num_filters = tcn.num_filters if hasattr(tcn, 'num_filters') else 64  # Default if not available
                self_extractor.output_dim = num_filters * sequence_length
                
                # Initialize policy and value networks with the expected output dimensions
                self_extractor.policy_net = nn.Sequential(
                    nn.Linear(self_extractor.output_dim, 64),
                    nn.ReLU()
                )
                
                self_extractor.value_net = nn.Sequential(
                    nn.Linear(self_extractor.output_dim, 64),
                    nn.ReLU()
                )
                
            def forward(self_extractor, features):
                # Debug prints for shape diagnosis
                batch_size = features.shape[0]
                seq_len = self_extractor.sequence_length
                feat_per_timestep = self_extractor.features_per_timestep
                total_expected = batch_size * seq_len * feat_per_timestep
                print(f"[DEBUG] features.shape: {features.shape}")
                print(f"[DEBUG] batch_size: {batch_size}")
                print(f"[DEBUG] sequence_length: {seq_len}")
                print(f"[DEBUG] features_per_timestep: {feat_per_timestep}")
                print(f"[DEBUG] batch_size * sequence_length * features_per_timestep: {total_expected}")
                print(f"[DEBUG] features.numel(): {features.numel()}")
                # First reshape to (batch_size, sequence_length, features_per_timestep)
                reshaped_features = features.view(batch_size, self_extractor.sequence_length, 
                                                self_extractor.features_per_timestep)
                # Then transpose to (batch_size, features_per_timestep, sequence_length)
                reshaped_features = reshaped_features.transpose(1, 2)
                
                # Process through TCN
                tcn_output = self_extractor.tcn(reshaped_features)
                
                # Get the actual output shape from the TCN
                actual_output_shape = tcn_output.shape
                print(f"TCN output shape: {actual_output_shape}")
                
                # Flatten the output for the policy and value networks
                flattened = tcn_output.reshape(batch_size, -1)
                actual_flattened_dim = flattened.shape[1]
                
                # Check if the expected output dimension matches the actual output
                if self_extractor.output_dim != actual_flattened_dim:
                    print(f"TCN output dimension mismatch: expected {self_extractor.output_dim}, "
                          f"got {actual_flattened_dim}. Adjusting networks.")
                    
                    # Update the output dimension
                    self_extractor.output_dim = actual_flattened_dim
                    
                    # Recreate the policy and value networks with the correct dimensions
                    device = flattened.device
                    self_extractor.policy_net = nn.Sequential(
                        nn.Linear(actual_flattened_dim, 64),
                        nn.ReLU()
                    ).to(device)
                    
                    self_extractor.value_net = nn.Sequential(
                        nn.Linear(actual_flattened_dim, 64),
                        nn.ReLU()
                    ).to(device)
                
                # Forward through the policy and value networks
                policy_latent = self_extractor.policy_net(flattened)
                value_latent = self_extractor.value_net(flattened)
                
                return policy_latent, value_latent
                
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
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs

    def _get_action_dist_from_latent(self, latent_pi):
        """Get the action distribution from the latent features."""
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=mean_actions) 