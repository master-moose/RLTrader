import torch
import torch.nn as nn
from torch.nn.utils import weight_norm # Import weight_norm
import numpy as np
import gym
from typing import Dict, List, Tuple, Type, Union, Optional, Any
import logging

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


# --- Define the Residual Block ---
class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        # Apply weight normalization to convolutional layers
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        # Padding layer for causal convolution
        self.pad1 = nn.ConstantPad1d((padding, 0), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.pad2 = nn.ConstantPad1d((padding, 0), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers into a sequential block
        self.net = nn.Sequential(
            self.pad1, self.conv1, self.relu1, self.dropout1,
            self.pad2, self.conv2, self.relu2, self.dropout2
        )

        # 1x1 convolution for residual connection if dimensions mismatch
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU() # Final activation for the block

        self.init_weights()

    def init_weights(self):
        # Initialize weights for better stability
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Apply downsample if needed, otherwise use input directly
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # Add residual connection


# --- Refactored TCN using TemporalBlocks ---
class TCN(nn.Module):
    """
    Temporal Convolutional Network with residual blocks.
    """
    def __init__(
        self,
        input_channels: int,
        num_filters: int, # Can be a single int or list for varying filters per layer
        num_layers: Optional[int] = None, # Used if num_filters is int
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super(TCN, self).__init__()
        print(f"[TCN][INIT-Residual] input_channels={input_channels}, num_filters={num_filters}, num_layers={num_layers}, kernel_size={kernel_size}, dropout={dropout}")

        layers = []
        num_levels = num_layers if isinstance(num_filters, int) else len(num_filters)
        
        if num_layers is None and isinstance(num_filters, int):
             raise ValueError("num_layers must be specified if num_filters is an integer")

        for i in range(num_levels):
            dilation_size = 2**i
            # Determine input/output channels for this block
            in_channels = input_channels if i == 0 else num_filters_list[i - 1]
            out_channels = num_filters if isinstance(num_filters, int) else num_filters[i]
            
            # Create a list of filters if a single int was provided
            if isinstance(num_filters, int):
                 num_filters_list = [num_filters] * num_layers
            else:
                 num_filters_list = num_filters

            # Calculate padding for causal convolution
            # Padding depends on kernel size and dilation
            padding_size = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1, # Stride is typically 1 for TCN
                    dilation=dilation_size,
                    padding=padding_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        # Output dimension is the number of filters in the last layer
        self.output_dim = num_filters_list[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape handling (same as before)
        if len(x.shape) != 3:
            print(f"Warning: Expected 3D input (batch_size, channels, sequence_length), got {x.shape}")
            # Basic reshape/transpose attempts (might need adjustment based on actual input)
            if len(x.shape) == 2: # Simple case: assume (batch, features) -> (batch, channels, seq_len)
                # This reshape is highly dependent on how features are flattened
                # It's better to handle this upstream or make it explicit
                print("ERROR: 2D input handling in TCN forward is ambiguous. Ensure input is 3D (batch, channels, seq_len).")
                # Placeholder: Attempting a guess based on input_channels
                batch_size, total_features = x.shape
                # We don't know seq_len here without ambiguity
                # Need input_channels to be explicitly passed or known
                # For now, raise error to force correct input shape
                raise ValueError("TCN received 2D input. Reshape upstream to (batch, channels, sequence_length).")

            elif len(x.shape) > 3:
                 batch_size = x.shape[0]
                 # Assuming last dim is sequence length, flatten others into channels
                 channels = int(np.prod(x.shape[1:-1]))
                 seq_len = x.shape[-1]
                 x = x.reshape(batch_size, channels, seq_len)
                 print(f"Reshaped >3D input to {x.shape}")
        else:
             # Check if channels and sequence length are swapped
             # Heuristic: if dim 1 > dim 2, assume (batch, seq_len, channels)
             if x.shape[1] > x.shape[2]:
                 # Check if dim 2 matches expected input channels
                 first_block_in_channels = self.network[0].conv1.in_channels
                 if x.shape[2] == first_block_in_channels:
                      print(f"Input shape {x.shape} matches (batch, seq_len, channels). Transposing.")
                      x = x.transpose(1, 2)
                 else:
                      print(f"Warning: Input shape {x.shape} is 3D but doesn't fit (batch, channels={first_block_in_channels}, seq_len) or (batch, seq_len, channels={first_block_in_channels}). Check input format.")
             elif x.shape[1] != self.network[0].conv1.in_channels:
                 # If shape is (batch, channels, seq_len) but channels don't match
                 print(f"Warning: Input shape {x.shape} channels ({x.shape[1]}) don't match TCN expected input channels ({self.network[0].conv1.in_channels}).")


        # --- Add assertion for channel mismatch ---
        # Get expected channels from the first conv layer of the first block
        expected_channels = self.network[0].conv1.in_channels
        if x.shape[1] != expected_channels:
            print(f"[TCN][ERROR] Input channels: {x.shape[1]}, Expected: {expected_channels}, Input shape: {x.shape}")
            # Provide more context in the error
            raise RuntimeError(f"TCN input channel mismatch: got {x.shape[1]}, expected {expected_channels}. Ensure input data has the correct number of features per timestep and is shaped (batch, channels, sequence_length).")

        # Process through the network
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
            # Pass num_filters and num_layers from tcn_params
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
        # --- ADDED: Handle MultiDiscrete action spaces from Gymnasium --- #
        elif action_space_type == "MultiDiscrete" and hasattr(self.action_space, 'nvec'):
             action_net_output_dim = int(np.sum(self.action_space.nvec))
        # --------------------------------------------------------------- #
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
                # TCN output shape is (batch, num_filters, sequence_length)
                num_filters = self_extractor.tcn.output_dim # Use TCN's own output_dim property
                # TCN output, when flattened, should have dimension num_filters * sequence_length
                # We take the output of the TCN across the whole sequence length
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
                    
                # Reshape time-series data for TCN: (batch, channels, seq_len)
                # Ensure channels dimension matches features_per_timestep
                try:
                    reshaped_ts_data = time_series_data.view(
                        batch_size,
                        self_extractor.features_per_timestep, # Channels first
                        self_extractor.sequence_length
                    )
                except RuntimeError as e:
                     print(f"[TcnExtractor][ERROR] Reshape failed. Input time_series_data shape: {time_series_data.shape}")
                     print(f"  Attempted reshape to ({batch_size}, {self_extractor.features_per_timestep}, {self_extractor.sequence_length})")
                     print(f"  Total elements expected: {batch_size * self_extractor.features_per_timestep * self_extractor.sequence_length}")
                     print(f"  Total elements actual: {time_series_data.numel()}")
                     print(f"  Original features.shape: {features.shape}")
                     print(f"  Calculated time_series_dim: {self_extractor.time_series_dim}")
                     print(f"  Calculated state_vars_dim: {self_extractor.state_vars_dim}")
                     raise e # Re-raise the error after printing info

                # Transposing is NOT needed if view is done correctly above
                # reshaped_ts_data = time_series_data.view(
                #     batch_size,
                #     self_extractor.sequence_length,
                #     self_extractor.features_per_timestep
                # ).transpose(1, 2)

                # Process through TCN
                tcn_output = self_extractor.tcn(reshaped_ts_data)
                
                # Flatten TCN output: (batch, num_filters * seq_len)
                # Ensure the flatten operation is correct
                expected_flattened_dim = self_extractor.tcn.output_dim * self_extractor.sequence_length
                flattened_tcn_output = tcn_output.reshape(batch_size, -1)
                if flattened_tcn_output.shape[1] != expected_flattened_dim:
                     print(f"[TcnExtractor][WARN] Flattened TCN output dimension mismatch!")
                     print(f"  tcn_output.shape: {tcn_output.shape}")
                     print(f"  Expected flattened dim (filters*seq_len): {expected_flattened_dim}")
                     print(f"  Actual flattened dim: {flattened_tcn_output.shape[1]}")
                     # Adjust based on actual output dim - might indicate upstream issue
                     self_extractor.tcn_output_dim = flattened_tcn_output.shape[1]
                     self_extractor.combined_latent_dim = self_extractor.tcn_output_dim + self_extractor.state_vars_dim

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