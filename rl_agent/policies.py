import torch
import torch.nn as nn
from torch.nn.utils import weight_norm # Import weight_norm
import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Any
import logging

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.sac.policies import SACPolicy # Added import
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, MlpExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution # Add others if needed
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.spaces import get_action_dim # <-- Added import

# Get module logger
logger = logging.getLogger("rl_agent.policies")

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

        # --- Step 1: Calculate all dimensions BEFORE calling super().__init__ --- #
        self.features_per_timestep = features_per_timestep
        self.sequence_length = sequence_length
        self.time_series_dim = features_per_timestep * sequence_length
        
        # Check observation space type using gymnasium alias
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError(f"TcnExtractor requires a 1D Box observation space, got {observation_space}")
        total_obs_dim = observation_space.shape[0]
        self.state_vars_dim = total_obs_dim - self.time_series_dim

        if self.state_vars_dim < 0:
            raise ValueError(
                f"Calculated state_vars_dim is negative ({self.state_vars_dim}). "
                f"Check observation_space shape ({total_obs_dim}), "
                f"features_per_timestep ({features_per_timestep}), "
                f"and sequence_length ({sequence_length})."
            )
        
        # Need to determine the TCN output dimension to calculate the final extractor output dim
        # Temporarily determine TCN output filters based on params
        temp_num_filters = tcn_params.get("num_filters", 64)
        if isinstance(temp_num_filters, list):
            tcn_output_filters = temp_num_filters[-1] 
        else:
            tcn_output_filters = temp_num_filters
        
        tcn_flat_output_dim = tcn_output_filters * self.sequence_length
        combined_features_dim = tcn_flat_output_dim + self.state_vars_dim
        # --- End Dimension Calculation ---

        # --- Step 2: Call the parent constructor with the FINAL features_dim --- #
        super().__init__(observation_space, features_dim=combined_features_dim)
        # ----------------------------------------------------------------------- #

        # --- Step 3: Now create and assign submodules (like the TCN) --- #
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
        # --- End Submodule Creation --- #

        # Print info after everything is set up
        if self.state_vars_dim > 0:
            print(f"[TcnExtractor] Detected {self.state_vars_dim} state variables.")
        else:
            print("[TcnExtractor] No state variables detected.")
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
        observation_space: gym.spaces.Box, # Ensure Box space
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
        # Check observation space type using gymnasium alias
        if not isinstance(observation_space, gym.spaces.Box):
             raise ValueError(f"TcnPolicy requires a Box observation space, got {type(observation_space)}")

        # --- Store TCN params needed by _build_mlp_extractor BEFORE super init ---
        self.features_per_timestep = features_per_timestep
        self.sequence_length = sequence_length
        # We also need access to the tcn_params dict later, let's store it too
        # Use an empty dict if None is passed
        self.tcn_params = tcn_params if tcn_params is not None else {}
        # -----------------------------------------------------------------------

        # 2. Prepare policy_kwargs for ActorCriticPolicy's __init__
        # - Set the features_extractor_class
        # - Set the features_extractor_kwargs needed by TcnExtractor
        # - Ensure standard kwargs (like net_arch, activation_fn) are handled
        # - Remove TCN specific args before calling parent
        policy_kwargs = kwargs.copy() # Start with user-provided kwargs

        # Set default net_arch and activation_fn if not provided
        policy_kwargs.setdefault("net_arch", [dict(pi=[64], vf=[64])]) # Default SB3 MLP arch
        policy_kwargs.setdefault("activation_fn", nn.Tanh) # Default SB3 activation

        policy_kwargs["features_extractor_class"] = TcnExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_per_timestep": features_per_timestep,
            "sequence_length": sequence_length,
            "tcn_params": tcn_params if tcn_params is not None else {},
        }

        # --- IMPORTANT: Remove keys not accepted by ActorCriticPolicy.__init__ ---
        policy_kwargs.pop("tcn_params", None)
        policy_kwargs.pop("sequence_length", None)
        policy_kwargs.pop("features_per_timestep", None)
        policy_kwargs.pop("features_dim", None) # Explicitly remove features_dim if present
        # ------------------------------------------------------------------------ #

        # 3. Call the parent __init__
        # It will internally create the TcnExtractor and calculate features_dim
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **policy_kwargs # Pass the cleaned kwargs
        )

        # 4. Initialize log_std for continuous actions (if applicable)
        if isinstance(action_space, gym.spaces.Box):
            action_dim = get_action_dim(self.action_space)
            # Initialize log_std as a learnable parameter
            self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)

        # 5. Orthogonal initialization for policy/value heads (good practice)
        # The parent class's _build method already applies init_weights,
        # so we don't need to call it manually here.
        # self.init_weights(self.mlp_extractor)
        # if hasattr(self, 'action_net'):
        #     self.init_weights(self.action_net, std=0.01)
        # if hasattr(self, 'value_net'):
        #     self.init_weights(self.value_net, std=1)

    # Override init_weights for orthogonal initialization of heads
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:  # Changed std to gain
        """
        Initialize the weights of the network with orthogonal initialization.

        :param module: Module to initialize
        :param gain: Gain factor for orthogonal initialization
        """
        for layer in module.modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Orthogonal initialization for weights
                torch.nn.init.orthogonal_(layer.weight, gain=gain) # Use gain here
                # Initialize bias terms to zero
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)

    @property
    def tcn(self):
        """Safe access to TCN with error handling."""
        if self.features_extractor is None:
            raise ValueError("TCN not initialized properly. Check if tcn_params is provided correctly.")
        return self.features_extractor.tcn

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

        # Check if action space is continuous
        if isinstance(self.action_space, gym.spaces.Box):
            if not hasattr(self, 'log_std'):
                raise ValueError("log_std not initialized for continuous action space.")
            # For continuous actions (Gaussian distribution)
            # Ensure log_std is broadcastable to mean_actions shape
            action_log_std = self.log_std.expand_as(mean_actions)
            return self.action_dist.proba_distribution(mean_actions, action_log_std)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # For discrete actions (Categorical distribution)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise NotImplementedError(
                f"Unsupported action space type: {type(self.action_space)}"
            )


# --- TCN SAC Policy (Uses TcnExtractor) --- #
# Import SACPolicy if not already imported at the top
# from stable_baselines3.sac.policies import SACPolicy

class TcnSacPolicy(SACPolicy):
    """
    Policy using a TCN-based feature extractor for SAC.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box, # SAC requires Box action space
        lr_schedule: Schedule,
        *args,
        # TCN specific parameters for the Extractor
        tcn_params: Optional[Dict[str, Any]] = None,
        sequence_length: Optional[int] = 60,
        features_per_timestep: Optional[int] = None,
        # Standard SACPolicy args (e.g., net_arch, activation_fn, n_critics)
        **kwargs
    ):
        # 1. Validate required TCN and SAC parameters
        if sequence_length is None:
            raise ValueError("`sequence_length` must be provided to TcnSacPolicy policy_kwargs.")
        if features_per_timestep is None:
            raise ValueError("`features_per_timestep` must be provided via policy_kwargs.")
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(f"TcnSacPolicy requires a Box observation space, got {type(observation_space)}")
        if not isinstance(action_space, gym.spaces.Box):
             raise ValueError(f"TcnSacPolicy requires a Box action space for SAC, got {type(action_space)}")

        # 2. Prepare kwargs for SACPolicy's __init__
        policy_kwargs = kwargs.copy()

        # Set defaults for SAC if not provided (common SAC defaults)
        policy_kwargs.setdefault("net_arch", [256, 256]) # Default SAC MLP arch for actor/critic
        policy_kwargs.setdefault("activation_fn", nn.ReLU) # Common activation for SAC
        policy_kwargs.setdefault("n_critics", 2) # Standard SAC setup
        policy_kwargs.setdefault("share_features_extractor", True) # Typically share extractor

        policy_kwargs["features_extractor_class"] = TcnExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_per_timestep": features_per_timestep,
            "sequence_length": sequence_length,
            "tcn_params": tcn_params if tcn_params is not None else {},
        }

        # --- IMPORTANT: Remove keys not accepted by SACPolicy.__init__ ---
        policy_kwargs.pop("tcn_params", None)
        policy_kwargs.pop("sequence_length", None)
        policy_kwargs.pop("features_per_timestep", None)
        # --------------------------------------------------------------- #

        # 3. Call the parent SACPolicy __init__
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **policy_kwargs # Pass the cleaned kwargs
        )

        # SACPolicy's _build method handles initialization

    # Override make_features_extractor if needed, but SACPolicy handles it
    # def make_features_extractor(self) -> TcnExtractor:
    #     """Creates the TCN feature extractor."""
    #     # The features_extractor is already created by the parent class
    #     # using features_extractor_class and features_extractor_kwargs
    #     return self.features_extractor

    # init_weights might be useful if customizing initialization beyond SB3 defaults
    # @staticmethod
    # def init_weights(module: nn.Module, gain: float = 1) -> None:
    #     TcnPolicy.init_weights(module, gain) # Reuse TcnPolicy's method if desired 