import torch
import torch.nn as nn
from torch.nn.utils import weight_norm # Import weight_norm
import numpy as np
import gym
from typing import Dict, Optional, Any

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
        padding: int, # This is the padding calculated for causality
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        # Apply weight normalization to convolutional layers
        # Set padding=0 here because causal padding is handled by self.pad1/pad2
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=0, # Set internal Conv1d padding to 0
                dilation=dilation,
            )
        )
        # Padding layer for causal convolution (applied before conv)
        self.pad1 = nn.ConstantPad1d((padding, 0), 0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=0, # Set internal Conv1d padding to 0
                dilation=dilation,
            )
        )
        self.pad2 = nn.ConstantPad1d((padding, 0), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers into a sequential block
        # Padding -> Conv -> ReLU -> Dropout
        self.net = nn.Sequential(
            self.pad1, self.conv1, self.relu1, self.dropout1,
            self.pad2, self.conv2, self.relu2, self.dropout2
        )

        # 1x1 convolution for residual connection if dimensions mismatch
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()  # Final activation for the block

        self.init_weights()

    def init_weights(self):
        # Initialize weights for better stability
        # Use He initialization (Kaiming Normal) which is common for ReLU
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        # Optional: Initialize bias terms to zero
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        # Apply downsample if needed, otherwise use input directly
        res = x if self.downsample is None else self.downsample(x)
        # The causal padding should ensure out and res have the same length
        return self.relu(out + res)  # Add residual connection


# --- Refactored TCN using TemporalBlocks ---
class TCN(nn.Module):
    """
    Temporal Convolutional Network with residual blocks.
    """
    def __init__(
        self,
        input_channels: int,
        num_filters: int,  # Filters per layer if int, or list of filters
        num_layers: Optional[int] = None,  # Used if num_filters is int
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super(TCN, self).__init__()
        print(f"[TCN][INIT-Residual] input_channels={input_channels}, "
              f"num_filters={num_filters}, num_layers={num_layers}, "
              f"kernel_size={kernel_size}, dropout={dropout}")

        layers = []
        self._input_channels = input_channels # Store for checks

        # Determine the number of levels and filter list
        if isinstance(num_filters, int):
            if num_layers is None:
                raise ValueError("num_layers must be specified if num_filters is an integer")
            num_levels = num_layers
            num_filters_list = [num_filters] * num_layers
        elif isinstance(num_filters, list):
            num_levels = len(num_filters)
            num_filters_list = num_filters
            if num_layers is not None:
                print("Warning: num_layers ignored because num_filters is a list.")
        else:
            raise ValueError("num_filters must be an int or a list of ints")

        for i in range(num_levels):
            dilation_size = 2**i
            # Determine input/output channels for this block
            in_channels = self._input_channels if i == 0 else num_filters_list[i - 1]
            out_channels = num_filters_list[i]

            # Calculate padding for causal convolution within the block
            padding_size = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,  # Stride is typically 1 for TCN
                    dilation=dilation_size,
                    padding=padding_size,  # Pass calculated padding
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        # Output dimension is the number of filters in the last layer
        self.output_dim = num_filters_list[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: Expected (batch_size, channels, sequence_length)

        # 1. Initial Shape Check
        if x.ndim != 3:
            print(f"ERROR: TCN Expected 3D input (batch, channels, seq_len), got {x.shape}. Reshape upstream.")
            raise ValueError(f"TCN received non-3D input: {x.shape}. Must be (batch, channels, sequence_length).")

        # 2. Channel Dimension Check
        if x.shape[1] != self._input_channels:
            # Check if dimensions might be swapped (batch, seq_len, channels)
            if x.shape[2] == self._input_channels:
                print(f"Warning: TCN input shape {x.shape} suggests (batch, seq_len, channels). Transposing to (batch, channels, seq_len).")
                x = x.transpose(1, 2)
            else:
                # If not swapped, it's a definite channel mismatch
                print(f"[TCN][ERROR] Input channels mismatch! Expected: {self._input_channels}, Got: {x.shape[1]}, Input shape: {x.shape}")
                raise RuntimeError(
                    f"TCN input channel mismatch: got {x.shape[1]}, expected {self._input_channels}. "
                    f"Ensure input data has the correct number of features per timestep ({self._input_channels}) "
                    f"and is shaped (batch, channels={self._input_channels}, sequence_length)."
                )

        # --- Assertion (Redundant if above checks pass, but good sanity check) ---
        if x.shape[1] != self._input_channels:
            # This should ideally not be reached if the checks above are correct
            raise RuntimeError(f"Internal TCN Error: Channel dimension mismatch after checks. Expected {self._input_channels}, got {x.shape[1]}.")

        # Process through the network
        return self.network(x)


# --- TCN Feature Extractor --- #
class TcnExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using a TCN to process the time-series part of the observation.
    Assumes observation is flattened: [time_series_features | state_variables]
    Time series features are expected to be [timestep_1_feature_1, ..., timestep_1_feature_N, ...,
                                            timestep_L_feature_1, ..., timestep_L_feature_N]
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_per_timestep: int,
                 sequence_length: int,
                 tcn_params: Dict[str, Any]):

        # Calculate dimensions
        self.features_per_timestep = features_per_timestep
        self.sequence_length = sequence_length
        self.time_series_dim = features_per_timestep * sequence_length
        total_obs_dim = observation_space.shape[0]
        self.state_vars_dim = total_obs_dim - self.time_series_dim

        if self.state_vars_dim < 0:
            raise ValueError(
                f"Calculated state_vars_dim is negative ({self.state_vars_dim}). "
                f"Check observation_space shape ({total_obs_dim}), "
                f"features_per_timestep ({features_per_timestep}), "
                f"and sequence_length ({sequence_length})."
            )
        elif self.state_vars_dim > 0:
            print(f"[TcnExtractor] Detected {self.state_vars_dim} state variables.")
        else:
            print("[TcnExtractor] No state variables detected.")

        # --- Create the TCN --- #
        tcn_init_kwargs = {
            "input_channels": self.features_per_timestep,
            "num_filters": tcn_params.get("num_filters", 64),
            "kernel_size": tcn_params.get("kernel_size", 3),
            "dropout": tcn_params.get("dropout", 0.2)
        }
        # Add num_layers only if num_filters is an int
        if isinstance(tcn_init_kwargs["num_filters"], int):
            tcn_init_kwargs["num_layers"] = tcn_params.get("num_layers", 4)

        self.tcn = TCN(**tcn_init_kwargs)
        # --------------------- #

        # Output dim after TCN processing (flattened)
        tcn_output_filters = self.tcn.output_dim
        tcn_flat_output_dim = tcn_output_filters * self.sequence_length

        # Final features dim is flattened TCN output + state variables
        combined_features_dim = tcn_flat_output_dim + self.state_vars_dim

        # Call parent constructor AFTER calculating final features_dim
        super().__init__(observation_space, features_dim=combined_features_dim)

        print(f"[TcnExtractor] Initialized. Time series dim: {self.time_series_dim}, "
              f"State vars dim: {self.state_vars_dim}, TCN output dim (flat): {tcn_flat_output_dim}, "
              f"Total output features_dim: {self._features_dim}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, total_obs_dim)
        batch_size = observations.shape[0]

        # Separate time-series data and state variables
        time_series_data = observations[:, :self.time_series_dim]
        if self.state_vars_dim > 0:
            state_vars = observations[:, self.time_series_dim:]
        else:
            state_vars = None  # No state variables

        # Reshape time-series data for TCN: (batch, channels, seq_len)
        try:
            # Channels = features_per_timestep
            reshaped_ts_data = time_series_data.view(
                batch_size,
                self.features_per_timestep,
                self.sequence_length
            )
        except RuntimeError as e:
            print(f"[TcnExtractor][ERROR] Reshape failed. Input time_series_data shape: {time_series_data.shape}")
            print(f"  Attempted reshape to ({batch_size}, {self.features_per_timestep}, {self.sequence_length})")
            print(f"  Total elements expected: {batch_size * self.features_per_timestep * self.sequence_length}")
            print(f"  Total elements actual: {time_series_data.numel()}")
            print(f"  Original observations.shape: {observations.shape}")
            raise e # Re-raise the error after printing info

        # Process through TCN
        # tcn_output shape: (batch, num_filters, seq_len)
        tcn_output = self.tcn(reshaped_ts_data)

        # Flatten TCN output: (batch, num_filters * seq_len)
        flattened_tcn_output = tcn_output.reshape(batch_size, -1)

        # --- Check Flattened Dim --- # (Optional Sanity Check)
        expected_flattened_dim = self.tcn.output_dim * self.sequence_length
        if flattened_tcn_output.shape[1] != expected_flattened_dim:
            print(f"[TcnExtractor][WARN] Flattened TCN output dimension mismatch! Should not happen.")
            # Handle potential error or adjust downstream, but ideally TCN maintains dims
            # For safety, raise error if mismatch occurs
            raise RuntimeError(f"Flattened TCN dim mismatch: Expected {expected_flattened_dim}, got {flattened_tcn_output.shape[1]}")
        # ------------------------- #

        # Combine TCN output with state variables (if they exist)
        if state_vars is not None:
            # Ensure state_vars is 2D (batch_size, state_vars_dim)
            if state_vars.ndim != 2 or state_vars.shape[0] != batch_size:
                raise ValueError(f"State variables shape mismatch: Expected ({batch_size}, {self.state_vars_dim}), got {state_vars.shape}")
            combined_features = torch.cat((flattened_tcn_output, state_vars), dim=1)
        else:
            combined_features = flattened_tcn_output

        # --- Final Check on output dimension --- #
        if combined_features.shape[1] != self._features_dim:
             raise RuntimeError(
                 f"[TcnExtractor][ERROR] Final combined feature dimension mismatch. "
                 f"Expected self._features_dim: {self._features_dim}, Got: {combined_features.shape[1]}"
             )
        # --------------------------------------- #

        return combined_features


# --- TCN Policy (Uses TcnExtractor) --- #
class TcnPolicy(ActorCriticPolicy):
    """
    Policy using a TCN-based feature extractor.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args,
        # TCN specific parameters for the Extractor
        tcn_params: Optional[Dict[str, Any]] = None,
        sequence_length: Optional[int] = 60,
        features_per_timestep: Optional[int] = None,
        # Standard ActorCriticPolicy args (e.g., net_arch, activation_fn)
        **kwargs
    ):
        # 1. Validate required TCN parameters
        if sequence_length is None:
            raise ValueError("`sequence_length` must be provided to TcnPolicy policy_kwargs.")
        if features_per_timestep is None:
            raise ValueError("`features_per_timestep` must be provided via policy_kwargs.")

        # 2. Prepare policy_kwargs for ActorCriticPolicy's __init__
        # - Set the features_extractor_class
        # - Set the features_extractor_kwargs needed by TcnExtractor
        # - Do NOT pass TCN specific args directly to parent
        policy_kwargs = kwargs.copy()
        policy_kwargs["features_extractor_class"] = TcnExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_per_timestep": features_per_timestep,
            "sequence_length": sequence_length,
            "tcn_params": tcn_params if tcn_params is not None else {},
        }

        # 3. Call the parent __init__
        # It will internally create the TcnExtractor and calculate features_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **policy_kwargs # Pass the modified kwargs
        )

        # 4. Orthogonal initialization for policy/value heads (good practice)
        self.init_weights(self.mlp_extractor) # Initialize extractor layers if needed
        self.init_weights(self.action_net, std=0.01)
        self.init_weights(self.value_net, std=1)

    # Override _build method only if absolutely necessary
    # ActorCriticPolicy._build handles extractor creation and head setup
    # def _build(self, lr_schedule):
    #     super()._build(lr_schedule)

    # Override init_weights for orthogonal initialization
    @staticmethod
    def init_weights(module: nn.Module, std: float = np.sqrt(2)) -> None:
        """
        Initialize the weights of the network with orthogonal initialization.

        :param module: Module to initialize
        :param std: Standard deviation for initialization (gain)
        """
        for layer in module.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Orthogonal initialization for weights
                torch.nn.init.orthogonal_(layer.weight, gain=std)
                # Initialize bias terms to zero
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)

    # Forward method should typically NOT be overridden unless changing
    # how policy/value heads are used. ActorCriticPolicy handles this.
    # def forward(self, obs: torch.Tensor, deterministic: bool = False):
    #     return super().forward(obs, deterministic)

    # _get_action_dist_from_latent remains the same as ActorCriticPolicy

    # predict_values remains the same as ActorCriticPolicy

    @property
    def tcn(self):
        """Safe access to TCN with error handling."""
        if self.features_extractor is None:
            raise ValueError("TCN not initialized properly. Check if tcn_params is provided correctly.")
        return self.features_extractor.tcn

    def _build_mlp_extractor(self):
        """Build a TCN-based feature extractor."""
        if self.features_extractor is None:
            raise ValueError("TCN must be initialized before building the MLP extractor.")
            
        # Capture outer class members needed inside TcnExtractor
        outer_tcn = self.tcn
        outer_features_per_timestep = self.features_per_timestep
        outer_sequence_length = self.sequence_length
        outer_features_dim = self._features_dim # Total observation dimension
        
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