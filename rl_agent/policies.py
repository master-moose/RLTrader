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
        print(f"[TCN][INIT] input_channels={input_channels}, num_filters={num_filters}, num_layers={num_layers}, kernel_size={kernel_size}, dropout={dropout}")
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
        # --- Add assertion for channel mismatch ---
        expected_channels = self.network[1].in_channels if hasattr(self.network[1], 'in_channels') else None
        if expected_channels is not None and x.shape[1] != expected_channels:
            print(f"[TCN][ERROR] Input channels: {x.shape[1]}, Expected: {expected_channels}, Input shape: {x.shape}")
            raise RuntimeError(f"TCN input channel mismatch: got {x.shape[1]}, expected {expected_channels}. Check features_per_timestep and sequence_length inference.")
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
        # ADDED: features_dim for clarity (will be inferred if None)
        features_dim: Optional[int] = None, 
        **kwargs
    ):
        # Initialize the parent class - IMPORTANT: This will call _build,
        # which will in turn call our overridden _build_mlp_extractor
        
        # --- REVISED INIT SEQUENCE ---
        # 1. Store TCN params and essential dimensions
        # Initialize default TCN parameters if not provided
        self.tcn_params = {
            "num_filters": 64,
            "num_layers": 4,
            "kernel_size": 3,
            "dropout": 0.2
        }
        if tcn_params is not None:
            self.tcn_params.update(tcn_params)
            
        # Use the provided sequence_length and features_per_timestep
        if sequence_length is None:
            raise ValueError("`sequence_length` must be provided to TcnPolicy.")
        if features_per_timestep is None:
            raise ValueError("`features_per_timestep` must be provided to TcnPolicy via policy_kwargs.")
            
        self.sequence_length = sequence_length
        self.features_per_timestep = features_per_timestep
        self.observation_space = observation_space
        self.action_space = action_space
        
        # --- Infer features_dim if not provided --- 
        if features_dim is None:
            if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1:
                self.features_dim = observation_space.shape[0]
            else:
                # Try to get it from kwargs or raise error
                self.features_dim = kwargs.get("features_dim")
                if self.features_dim is None:
                    raise ValueError("Could not infer features_dim. Please provide it in policy_kwargs.")
        else:
            self.features_dim = features_dim
        # --- End inference ---
            
        self._tcn = None # To be initialized in _build
        
        # 2. Call the parent __init__ which triggers _build
        # Remove 'features' from kwargs if present
        if 'features' in kwargs:
            del kwargs['features']
        # Remove features_dim if we inferred it
        if 'features_dim' in kwargs and features_dim is None:
             del kwargs['features_dim']
             
        super(TcnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        # --- END REVISED INIT SEQUENCE ---
    
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
            
        # Capture outer class members needed inside TcnExtractor
        outer_tcn = self._tcn
        outer_features_per_timestep = self.features_per_timestep
        outer_sequence_length = self.sequence_length
        outer_features_dim = self.features_dim # Total observation dimension
        
        class TcnExtractor(nn.Module):
            def __init__(self_extractor):
                super(TcnExtractor, self_extractor).__init__()
                
                self_extractor.tcn = outer_tcn
                self_extractor.features_per_timestep = outer_features_per_timestep
                self_extractor.sequence_length = outer_sequence_length
                self_extractor.features_dim = outer_features_dim # Store total obs dim
                self_extractor.time_series_dim = self_extractor.features_per_timestep * self_extractor.sequence_length
                self_extractor.state_vars_dim = self_extractor.features_dim - self_extractor.time_series_dim
                
                # Check if state variables exist
                if self_extractor.state_vars_dim < 0:
                    raise ValueError(f"Calculated state_vars_dim is negative ({self_extractor.state_vars_dim}). Check features_dim ({self_extractor.features_dim}), features_per_timestep ({self_extractor.features_per_timestep}), and sequence_length ({self_extractor.sequence_length}).")
                elif self_extractor.state_vars_dim > 0:
                    print(f"[TcnExtractor] Detected {self_extractor.state_vars_dim} state variables based on dimensions.")
                else:
                    print("[TcnExtractor] No state variables detected (time_series_dim matches features_dim).")
                    
                # Calculate the output dimension after TCN processing
                num_filters = self_extractor.tcn.output_dim # Use TCN's own output_dim property
                self_extractor.tcn_output_dim = num_filters * self_extractor.sequence_length
                
                # Calculate the input dimension for the policy/value networks
                # It's the flattened TCN output PLUS the state variables
                self_extractor.combined_latent_dim = self_extractor.tcn_output_dim + self_extractor.state_vars_dim
                
                # Define policy and value networks based on the combined dimension
                latent_layer_size = 64 # Example size, could be configurable
                self_extractor.policy_net = nn.Sequential(
                    nn.Linear(self_extractor.combined_latent_dim, latent_layer_size),
                    nn.ReLU()
                )
                self_extractor.value_net = nn.Sequential(
                    nn.Linear(self_extractor.combined_latent_dim, latent_layer_size),
                    nn.ReLU()
                )
                
                # --- Add latent_dim_pi and latent_dim_vf properties --- #
                # These should reflect the output size of the respective networks
                self_extractor.latent_dim_pi = latent_layer_size 
                self_extractor.latent_dim_vf = latent_layer_size
                # ------------------------------------------------------ #
                
            def forward(self_extractor, features):
                batch_size = features.shape[0]
                
                # Separate time-series data and state variables
                time_series_data = features[:, :self_extractor.time_series_dim]
                if self_extractor.state_vars_dim > 0:
                    state_vars = features[:, self_extractor.time_series_dim:]
                else:
                    state_vars = None # No state variables
                    
                # Reshape time-series data for TCN: (batch, features_per_step, seq_len)
                reshaped_ts_data = time_series_data.view(
                    batch_size, 
                    self_extractor.sequence_length, 
                    self_extractor.features_per_timestep
                ).transpose(1, 2)
                
                # Process through TCN
                tcn_output = self_extractor.tcn(reshaped_ts_data)
                
                # Flatten TCN output: (batch, tcn_output_dim)
                flattened_tcn_output = tcn_output.reshape(batch_size, -1)
                
                # Combine TCN output with state variables (if they exist)
                if state_vars is not None:
                    combined_features = torch.cat((flattened_tcn_output, state_vars), dim=1)
                else:
                    combined_features = flattened_tcn_output
                
                # --- Check combined dimension consistency --- #
                if combined_features.shape[1] != self_extractor.combined_latent_dim:
                     # This might happen if TCN output dim changes dynamically (unlikely but possible)
                     print(f"[WARN] Combined dimension mismatch. Expected {self_extractor.combined_latent_dim}, Got {combined_features.shape[1]}. Re-adjusting Linear layers.")
                     self_extractor.combined_latent_dim = combined_features.shape[1]
                     device = combined_features.device
                     latent_layer_size = 64 # Keep consistent
                     self_extractor.policy_net[0] = nn.Linear(self_extractor.combined_latent_dim, latent_layer_size).to(device)
                     self_extractor.value_net[0] = nn.Linear(self_extractor.combined_latent_dim, latent_layer_size).to(device)
                # --- End check --- #
                     
                # Forward through the policy and value networks
                policy_latent = self_extractor.policy_net(combined_features)
                value_latent = self_extractor.value_net(combined_features)
                
                return policy_latent, value_latent

            # forward_actor and forward_critic remain the same
            def forward_actor(self_extractor, features):
                policy_latent, _ = self_extractor.forward(features)
                return policy_latent

            def forward_critic(self_extractor, features):
                _, value_latent = self_extractor.forward(features)
                return value_latent
                
        # We don't pass features_dim here anymore, it's accessed via the outer scope
        return TcnExtractor()

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