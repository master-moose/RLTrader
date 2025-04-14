#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural network models for RL agents.

This module contains model definitions for various RL architectures
including feature extractors and policy networks.
"""

import torch
import torch.nn as nn
import numpy as np
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy

from .config import logger, HOLD_ACTION


def safe_load_lstm_weights(model_path: str) -> Optional[Dict]:
    """
    Safely load LSTM weights with multiple fallback approaches for compatibility.
    
    Args:
        model_path: Path to the saved LSTM model
    
    Returns:
        Loaded state dictionary or None if loading failed
    """
    if not model_path:
        return None
    
    logger.info(f"Attempting to load LSTM weights from: {model_path}")
    
    loading_approaches = [
        # Approach 1: Modern PyTorch with weights_only=True
        lambda path: torch.load(path, map_location='cpu', weights_only=True),
        
        # Approach 2: Older PyTorch without weights_only
        lambda path: torch.load(path, map_location='cpu'),
        
        # Approach 3: Try with map_location='cpu' and potentially deserialize_pickle_tensors
        lambda path: torch.load(path, map_location='cpu', pickle_module=torch.serialization._load._get_default_load_module()),
        
        # Approach 4: Raw pickle load as last resort
        lambda path: torch.serialization._load._load(open(path, 'rb'), map_location='cpu')
    ]
    
    last_error = None
    for i, approach in enumerate(loading_approaches):
        try:
            logger.info(f"Trying loading approach {i+1}...")
            state_dict = approach(model_path)
            logger.info(f"Successfully loaded LSTM weights using approach {i+1}")
            return state_dict
        except Exception as e:
            last_error = e
            logger.warning(f"Loading approach {i+1} failed: {str(e)}")
            continue
    
    # If we get here, all approaches failed
    logger.error(f"All loading approaches failed. Last error: {last_error}")
    logger.error("LSTM weights could not be loaded. Continuing with random initialization.")
    return None


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses an LSTM model to process observations before 
    passing them to the policy network.
    
    This allows utilization of pre-trained LSTM models for feature extraction
    in reinforcement learning.
    """
    
    def __init__(self, observation_space, lstm_state_dict=None, features_dim=64):
        """
        Initialize the LSTM feature extractor.
        
        Args:
            observation_space: The observation space of the environment
            lstm_state_dict: The state dictionary of a pre-trained LSTM model
                             or path to a saved model
            features_dim: The output dimension of the LSTM features
        """
        # Call the parent constructor with the correct features_dim
        super().__init__(observation_space, features_dim=features_dim)
        
        # Create LSTM model architecture - this should match the saved model
        input_dim = observation_space.shape[0]
        self.lstm_model = torch.nn.LSTM(
            input_size=input_dim, 
            hidden_size=features_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Try to load the state dict
        if lstm_state_dict is not None:
            try:
                # Handle string paths to models
                if isinstance(lstm_state_dict, str):
                    # Use our safe loading function
                    lstm_state_dict = safe_load_lstm_weights(lstm_state_dict)
                    if lstm_state_dict is None:
                        logger.warning("Could not load LSTM weights, using random initialization")
                
                # Continue only if we have a valid state dict
                if lstm_state_dict is not None and hasattr(lstm_state_dict, 'items'):
                    # Extract LSTM weights from the state dict
                    filtered_state_dict = {}
                    valid_keys = ['lstm', 'encoder', 'rnn', 'model', 'net']
                    
                    # First, log all available keys for debugging
                    logger.info(f"Available keys in state dict: {list(lstm_state_dict.keys())}")
                    
                    # Try to find keys that could contain LSTM weights
                    for key, value in lstm_state_dict.items():
                        if any(key.startswith(prefix) for prefix in valid_keys) or 'lstm' in key.lower():
                            filtered_state_dict[key] = value
                    
                    # If we don't find any matching keys, try to use the entire state dict
                    if not filtered_state_dict and lstm_state_dict:
                        logger.warning("No LSTM-specific keys found, attempting to use entire state dict")
                        filtered_state_dict = lstm_state_dict
                    
                    # Log what we're loading
                    logger.info(f"Loading LSTM weights: {list(filtered_state_dict.keys())}")
                    
                    # Attempt to load with different approaches
                    try:
                        # First try direct loading
                        self.lstm_model.load_state_dict(filtered_state_dict)
                        logger.info("LSTM weights loaded successfully with direct loading")
                    except Exception as e1:
                        logger.warning(f"Direct loading failed: {e1}")
                        try:
                            # Try partial loading with strict=False
                            self.lstm_model.load_state_dict(filtered_state_dict, strict=False)
                            logger.info("LSTM weights loaded with partial loading (strict=False)")
                        except Exception as e2:
                            logger.error(f"Partial loading also failed: {e2}")
                            logger.error("Using random initialization for LSTM")
                else:
                    logger.warning("LSTM state dict is not valid, using random initialization")
            except Exception as e:
                logger.error(f"Error loading LSTM weights: {e}")
                logger.error(traceback.format_exc())
        
        # Set to eval mode since we're not training the LSTM
        self.lstm_model.eval()
        
    def forward(self, observations):
        """
        Process observations through the LSTM feature extractor.
        
        Args:
            observations: Tensor of observations from the environment
            
        Returns:
            Tensor of processed features
        """
        # Ensure observations are of the right shape
        # LSTM expects [batch_size, sequence_length, input_size]
        
        with torch.no_grad():  # No need to track gradients in feature extraction
            # Process as a single timestep sequence [batch_size, 1, input_size]
            # Add sequence dimension
            obs_seq = observations.unsqueeze(1)
            
            # Forward pass through LSTM
            lstm_out, _ = self.lstm_model(obs_seq)
            
            # Extract the output for the last timestep [batch_size, features_dim]
            features = lstm_out[:, -1, :]
            
        return features


class AntiHoldPolicy(DQNPolicy):
    """
    A custom DQN policy that discourages the hold action by artificially
    reducing its Q-value during action selection.
    """
    
    def __init__(self, *args, hold_action_bias=-1.0, **kwargs):
        """Initialize the anti-hold policy with a bias against the hold action"""
        super().__init__(*args, **kwargs)
        self.hold_action_bias = hold_action_bias
        self.hold_action = HOLD_ACTION  # The action index for hold
    
    def _predict(self, obs, deterministic=True):
        """
        Overrides the parent class _predict method to apply a bias against holding.
        This method reduces the Q-value of the hold action to discourage the agent from holding.
        """
        q_values = self.q_net(obs)
        
        # Apply a negative bias to the hold action's Q-value
        # Use a milder bias (-1.0 instead of -3.0) to discourage holding without causing oscillation
        q_values[:, self.hold_action] += self.hold_action_bias
        
        # Get the actions using the modified Q-values
        actions = q_values.argmax(dim=1).reshape(-1)
        return actions, q_values