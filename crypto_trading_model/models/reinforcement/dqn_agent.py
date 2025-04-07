"""
DQN (Deep Q-Network) agent for cryptocurrency trading.

This module implements a DQN agent for reinforcement learning-based
cryptocurrency trading, which uses predictions from time series models.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional, Any
import logging
import gym
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
import random
from collections import deque

# Import from parent directory
import sys
sys.path.append('../..')
from config import RL_SETTINGS, PATHS
from environment.trading_env import CryptoTradingEnv
from models.time_series.lstm_model import TimeSeriesLSTM
from models.time_series.model import MultiTimeframeModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dqn_agent')

class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Parameters:
        -----------
        capacity : int
            Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state
        done : bool
            Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences.
        
        Parameters:
        -----------
        batch_size : int
            Size of batch to sample
            
        Returns:
        --------
        Tuple
            Batch of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
        --------
        int
            Current size of buffer
        """
        return len(self.buffer)

class DQNTradingAgent:
    """
    DQN agent for cryptocurrency trading.
    """
    
    def __init__(
        self,
        env: gym.Env,
        time_series_model: Optional[Union[TimeSeriesLSTM, MultiTimeframeModel]] = None,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        batch_size: int = 64,
        update_target_every: int = 100,
        buffer_capacity: int = 10000,
        hidden_units: List[int] = None,
        double_dqn: bool = True,
        tensorboard_log: str = None,
        feature_extraction_mode: str = "concat"  # Options: "concat", "guidance", "transfer"
    ):
        """
        Initialize the DQN trading agent.
        
        Parameters:
        -----------
        env : gym.Env
            Trading environment
        time_series_model : TimeSeriesLSTM or MultiTimeframeModel, optional
            Trained time series model to incorporate predictions
        learning_rate : float
            Learning rate for the optimizer
        gamma : float
            Discount factor
        epsilon_start : float
            Initial exploration rate
        epsilon_end : float
            Final exploration rate
        epsilon_decay_steps : int
            Number of steps for epsilon decay
        batch_size : int
            Batch size for training
        update_target_every : int
            Frequency of target network updates
        buffer_capacity : int
            Capacity of replay buffer
        hidden_units : List[int]
            List of hidden units for each layer
        double_dqn : bool
            Whether to use Double DQN
        tensorboard_log : str
            Path for TensorBoard logs
        feature_extraction_mode : str
            How to use features from time series model
              - "concat": Concatenate features with state
              - "guidance": Use predictions as guidance signals
              - "transfer": Use transfer learning from time series model
        """
        # Environment
        self.env = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        # Time series model
        self.time_series_model = time_series_model
        self.use_predictions = time_series_model is not None
        self.feature_extraction_mode = feature_extraction_mode
        
        # Determine the type of time series model
        self.ts_model_type = None
        if self.use_predictions:
            self.ts_model_type = "multitimeframe" if isinstance(time_series_model, MultiTimeframeModel) else "lstm"
            logger.info(f"Using {self.ts_model_type} time series model for feature extraction")
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.double_dqn = double_dqn
        
        # Define model architecture
        if hidden_units is None:
            hidden_units = [128, 64]
        self.hidden_units = hidden_units
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        
        # Build networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.step_count = 0
        
        # TensorBoard logging
        if tensorboard_log is None:
            self.tensorboard_log = os.path.join(PATHS.get('logs', 'logs'), 'tensorboard', 'dqn')
        else:
            self.tensorboard_log = tensorboard_log
        
        # Create log directory
        os.makedirs(self.tensorboard_log, exist_ok=True)
        
        # Create summary writer for TensorBoard
        current_time = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_log, current_time)
        )
    
    def _build_model(self) -> Model:
        """
        Build the Q-network model.
        
        Returns:
        --------
        tf.keras.Model
            Q-network model
        """
        # Check if we need to incorporate time series predictions
        if self.use_predictions:
            # Main input for state
            state_input = Input(shape=self.state_dim)
            state_features = Flatten()(state_input)
            
            if self.feature_extraction_mode == "concat":
                # Add time series feature input
                if self.ts_model_type == "multitimeframe":
                    # For MultiTimeframeModel, we expect features from extract_features
                    # Estimate feature size based on model architecture
                    if hasattr(self.time_series_model, 'hidden_dims'):
                        feature_size = self.time_series_model.hidden_dims
                        if hasattr(self.time_series_model, 'bidirectional') and self.time_series_model.bidirectional:
                            feature_size *= 2
                        if not hasattr(self.time_series_model, 'attention') or not self.time_series_model.attention:
                            feature_size *= len(self.time_series_model.timeframes)
                    else:
                        # Default if we can't determine
                        feature_size = 128
                    
                    ts_features_input = Input(shape=(feature_size,))
                else:
                    # For LSTM model, simpler prediction
                    ts_features_input = Input(shape=(3,))  # 3 classes: buy, sell, hold
                
                # Concatenate state and time series features
                combined = Concatenate()([state_features, ts_features_input])
                
            elif self.feature_extraction_mode == "guidance":
                # Use time series predictions as guidance signals
                ts_prediction_input = Input(shape=(3,))  # 3 classes: buy, sell, hold
                
                # Process the state with initial layers
                x = Dense(self.hidden_units[0], activation='relu')(state_features)
                x = BatchNormalization()(x)
                
                # Incorporate prediction guidance
                combined = Concatenate()([x, ts_prediction_input])
                
            elif self.feature_extraction_mode == "transfer":
                # Create a similar architecture to the time series model
                # but use it for Q-learning instead
                combined = state_features
                # No additional input needed as we're transferring knowledge differently
            
            # Hidden layers with BatchNormalization for better training
            x = combined
            for units in self.hidden_units:
                x = Dense(units, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(0.2)(x)  # Add dropout for regularization
            
            # Output layer (Q-values for each action)
            outputs = Dense(self.action_dim)(x)
            
            # Build model based on feature extraction mode
            if self.feature_extraction_mode == "concat":
                model = Model(inputs=[state_input, ts_features_input], outputs=outputs)
            elif self.feature_extraction_mode == "guidance":
                model = Model(inputs=[state_input, ts_prediction_input], outputs=outputs)
            else:  # "transfer"
                model = Model(inputs=state_input, outputs=outputs)
                
        else:
            # Simple sequential model with improved regularization
            model = Sequential([
                Input(shape=self.state_dim),
                Flatten(),
                *[
                    Sequential([
                        Dense(units, activation='relu'),
                        BatchNormalization(),
                        Dropout(0.2)
                    ]) for units in self.hidden_units
                ],
                Dense(self.action_dim)
            ])
        
        # Compile model with Huber loss for stability
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=Huber()
        )
        
        return model
    
    def update_target_network(self):
        """
        Update target network weights from the Q-network.
        """
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state, prediction=None, deterministic=False) -> int:
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        prediction : np.ndarray, optional
            Prediction from time series model
        deterministic : bool
            Whether to use deterministic policy (for evaluation)
            
        Returns:
        --------
        int
            Selected action
        """
        # Use epsilon-greedy strategy for exploration
        if not deterministic and np.random.random() < self.epsilon:
            # Random action
            return np.random.randint(self.action_dim)
        
        # Get Q-values for each action
        if self.use_predictions:
            if self.feature_extraction_mode == "concat":
                if prediction is None:
                    prediction = self._get_time_series_features(state)
                q_values = self.q_network.predict([np.array([state]), np.array([prediction])], verbose=0)[0]
            elif self.feature_extraction_mode == "guidance":
                if prediction is None:
                    prediction = self._get_time_series_prediction(state)
                q_values = self.q_network.predict([np.array([state]), np.array([prediction])], verbose=0)[0]
            else:  # "transfer"
                q_values = self.q_network.predict(np.array([state]), verbose=0)[0]
        else:
            q_values = self.q_network.predict(np.array([state]), verbose=0)[0]
        
        # Select action with highest Q-value
        return np.argmax(q_values)
    
    def train(
        self,
        total_timesteps: int = 100000,
        log_interval: int = 1000,
        eval_interval: int = 10000,
        n_eval_episodes: int = 3
    ) -> Dict[str, List]:
        """
        Train the DQN agent.
        
        Parameters:
        -----------
        total_timesteps : int
            Total number of timesteps to train
        log_interval : int
            Interval for logging training metrics
        eval_interval : int
            Interval for evaluating the agent
        n_eval_episodes : int
            Number of episodes for evaluation
            
        Returns:
        --------
        Dict[str, List]
            Training history with loss, reward, and epsilon
        """
        logger.info(f"Training DQN agent for {total_timesteps} timesteps")
        
        self.step_count = 0
        episode = 0
        episode_rewards = []
        episode_reward = 0
        
        # Reset environment
        state = self.env.reset()
        
        while self.step_count < total_timesteps:
            # Get prediction from time series model
            prediction = None
            if self.use_predictions:
                prediction = self._get_time_series_prediction(state)
            
            # Select action
            action = self.select_action(state, prediction)
            
            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Add to replay buffer
            self.buffer.add(state, action, reward, next_state, done)
            
            # Update statistics
            episode_reward += reward
            self.step_count += 1
            
            # Update epsilon
            self.epsilon = max(
                self.epsilon_end, 
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * min(1.0, self.step_count / self.epsilon_decay_steps)
            )
            
            # Train on batch if enough samples
            if len(self.buffer) >= self.batch_size:
                self._train_step()
            
            # Update target network periodically
            if self.step_count % self.update_target_every == 0:
                self.update_target_network()
                logger.debug(f"Updated target network at step {self.step_count}")
            
            # Log training metrics
            if self.step_count % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                logger.info(f"Step: {self.step_count}, Episode: {episode}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.4f}")
                
                # Log to TensorBoard
                with self.summary_writer.as_default():
                    tf.summary.scalar('avgReward', avg_reward, step=self.step_count)
                    tf.summary.scalar('epsilon', self.epsilon, step=self.step_count)
                    if self.loss_history:
                        tf.summary.scalar('loss', self.loss_history[-1], step=self.step_count)
            
            # Evaluate the agent periodically
            if self.step_count % eval_interval == 0:
                eval_rewards = self._evaluate(n_eval_episodes)
                logger.info(f"Evaluation at step {self.step_count}: Mean reward = {np.mean(eval_rewards):.4f}")
                
                # Log to TensorBoard
                with self.summary_writer.as_default():
                    tf.summary.scalar('evalReward', np.mean(eval_rewards), step=self.step_count)
            
            # Reset environment if episode is done
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                self.reward_history.append(episode_reward)
                self.epsilon_history.append(self.epsilon)
                
                # Reset
                state = self.env.reset()
                episode_reward = 0
            else:
                state = next_state
        
        logger.info(f"Training completed after {episode} episodes")
        
        # Create training history dictionary
        training_history = {
            'loss': self.loss_history,
            'reward': self.reward_history,
            'epsilon': self.epsilon_history
        }
        
        return training_history
    
    def _get_time_series_features(self, state):
        """
        Extract features from the time series model
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
            
        Returns:
        --------
        np.ndarray
            Features from time series model
        """
        if self.ts_model_type == "multitimeframe":
            # For MultiTimeframeModel, we need to reshape state into multiple timeframes
            # This assumes state contains data for all timeframes concatenated
            
            # This is a simplified example - in practice, would need to:
            # 1. Reshape state into timeframe-specific tensors
            # 2. Convert to torch tensors
            # 3. Call extract_features
            # 4. Convert result back to numpy
            
            # Placeholder implementation - in practice, would need to be customized
            # based on exact state representation in environment
            try:
                import torch
                
                # Convert state to proper format for MultiTimeframeModel
                # This assumes environment provides state in the right format
                # for extracting timeframe data
                tf_data = {}
                feature_dim = self.env.feature_dims  # Assuming env has this attribute
                
                for i, tf in enumerate(self.time_series_model.timeframes):
                    # Calculate start and end indices for this timeframe's data in state
                    if hasattr(self.env, 'window_size'):
                        window_size = self.env.window_size
                    else:
                        window_size = 50  # default
                    
                    # Extract and reshape this timeframe's data
                    start_idx = i * feature_dim[tf] * window_size
                    end_idx = start_idx + feature_dim[tf] * window_size
                    
                    tf_data[tf] = torch.tensor(
                        state[start_idx:end_idx].reshape(1, window_size, feature_dim[tf]),
                        dtype=torch.float32
                    )
                
                # Extract features using the time series model
                with torch.no_grad():
                    features = self.time_series_model.extract_features(tf_data)
                    return features.numpy().flatten()
                    
            except Exception as e:
                logger.warning(f"Error extracting features from time series model: {e}")
                # Fallback: return dummy features
                if hasattr(self.time_series_model, 'hidden_dims'):
                    feature_size = self.time_series_model.hidden_dims
                    if hasattr(self.time_series_model, 'bidirectional') and self.time_series_model.bidirectional:
                        feature_size *= 2
                    if not hasattr(self.time_series_model, 'attention') or not self.time_series_model.attention:
                        feature_size *= len(self.time_series_model.timeframes)
                else:
                    feature_size = 128
                return np.zeros(feature_size)
        else:
            # For LSTM model, get prediction and use as features
            return self._get_time_series_prediction(state)
    
    def _get_time_series_prediction(self, state):
        """
        Get prediction from time series model
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
            
        Returns:
        --------
        np.ndarray
            Prediction from time series model (probabilities for buy, sell, hold)
        """
        # This method would need to be customized based on the specific
        # format of your time series model's predictions
        
        if self.ts_model_type == "multitimeframe":
            try:
                import torch
                
                # Convert state to proper format for MultiTimeframeModel
                # Similar to _get_time_series_features
                tf_data = {}
                feature_dim = self.env.feature_dims  # Assuming env has this attribute
                
                for i, tf in enumerate(self.time_series_model.timeframes):
                    # Calculate indices as in _get_time_series_features
                    if hasattr(self.env, 'window_size'):
                        window_size = self.env.window_size
                    else:
                        window_size = 50  # default
                    
                    start_idx = i * feature_dim[tf] * window_size
                    end_idx = start_idx + feature_dim[tf] * window_size
                    
                    tf_data[tf] = torch.tensor(
                        state[start_idx:end_idx].reshape(1, window_size, feature_dim[tf]),
                        dtype=torch.float32
                    )
                
                # Get prediction probabilities
                with torch.no_grad():
                    probs = self.time_series_model.predict_probabilities(tf_data)
                    return probs.numpy().flatten()
                    
            except Exception as e:
                logger.warning(f"Error getting prediction from time series model: {e}")
                # Fallback: return uniform probabilities
                return np.ones(3) / 3
                
        elif self.ts_model_type == "lstm":
            # Assuming LSTM model has a predict method that returns probabilities
            try:
                return self.time_series_model.predict(np.array([state]))[0]
            except:
                return np.ones(3) / 3
        else:
            # Fallback
            return np.ones(3) / 3
    
    def _train_step(self):
        """
        Perform a single training step.
        """
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Get predictions from time series model if available
        predictions = None
        next_predictions = None
        if self.use_predictions:
            predictions = np.array([self._get_time_series_features(state) for state in states])
            next_predictions = np.array([self._get_time_series_features(next_state) for next_state in next_states])
        
        # Current Q-values
        if self.use_predictions and predictions is not None:
            current_q = self.q_network.predict([states, predictions])
        else:
            current_q = self.q_network.predict(states)
        
        # Target Q-values
        if self.use_predictions and next_predictions is not None:
            if self.double_dqn:
                # Double DQN: use online network for action selection
                online_q = self.q_network.predict([next_states, next_predictions])
                best_actions = np.argmax(online_q, axis=1)
                
                # Use target network for Q-value estimation
                target_q = self.target_network.predict([next_states, next_predictions])
                next_q = np.array([target_q[i, action] for i, action in enumerate(best_actions)])
            else:
                # Standard DQN
                next_q = np.amax(self.target_network.predict([next_states, next_predictions]), axis=1)
        else:
            if self.double_dqn:
                # Double DQN: use online network for action selection
                online_q = self.q_network.predict(next_states)
                best_actions = np.argmax(online_q, axis=1)
                
                # Use target network for Q-value estimation
                target_q = self.target_network.predict(next_states)
                next_q = np.array([target_q[i, action] for i, action in enumerate(best_actions)])
            else:
                # Standard DQN
                next_q = np.amax(self.target_network.predict(next_states), axis=1)
        
        # Compute targets with reward normalization to improve stability
        rewards = np.clip(rewards, -10.0, 10.0)  # Clip rewards to prevent extreme values
        targets = rewards + (1 - dones) * self.gamma * next_q
        
        # Update only the Q-values for actions taken (Bellman update)
        target_q = current_q.copy()
        
        # Use vectorized operations
        for i in range(len(actions)):
            target_q[i, actions[i]] = targets[i]
        
        # Train the model with a lower batch size if needed for stability
        actual_batch_size = min(self.batch_size, 32)  # Smaller batches can be more stable
        
        # Train the model
        if self.use_predictions and predictions is not None:
            history = self.q_network.fit(
                [states, predictions], target_q, 
                batch_size=actual_batch_size, 
                verbose=0
            )
        else:
            history = self.q_network.fit(
                states, target_q, 
                batch_size=actual_batch_size, 
                verbose=0
            )
        
        # Record loss
        self.loss_history.append(history.history['loss'][0])
    
    def _evaluate(self, n_episodes: int = 5) -> List[float]:
        """
        Evaluate the agent's performance.
        
        Parameters:
        -----------
        n_episodes : int
            Number of episodes to evaluate
            
        Returns:
        --------
        List[float]
            Rewards for each evaluation episode
        """
        rewards = []
        
        for _ in range(n_episodes):
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get prediction if using time series model
                prediction = None
                if self.use_predictions:
                    prediction = self._get_time_series_prediction(state)
                
                # Select action deterministically
                action = self.select_action(state, prediction, deterministic=True)
                
                # Take step
                next_state, reward, done, _ = self.env.step(action)
                
                # Update statistics
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
        
        return rewards
    
    def save(self, path: str = None):
        """
        Save the agent.
        
        Parameters:
        -----------
        path : str
            Path to save the agent
        """
        if path is None:
            path = os.path.join(PATHS.get('rl_models', 'models/reinforcement'), 'dqn')
        
        os.makedirs(path, exist_ok=True)
        
        # Save Q-network
        q_network_path = os.path.join(path, 'q_network.h5')
        self.q_network.save(q_network_path)
        
        # Save target network
        target_network_path = os.path.join(path, 'target_network.h5')
        self.target_network.save(target_network_path)
        
        # Save hyperparameters
        hyperparams = {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_steps': self.epsilon_decay_steps,
            'batch_size': self.batch_size,
            'update_target_every': self.update_target_every,
            'hidden_units': self.hidden_units,
            'double_dqn': self.double_dqn,
            'use_predictions': self.use_predictions,
            'step_count': self.step_count
        }
        
        # Save hyperparameters to JSON
        import json
        with open(os.path.join(path, 'hyperparams.json'), 'w') as f:
            json.dump(hyperparams, f)
        
        logger.info(f"Agent saved to {path}")
    
    @classmethod
    def load(cls, path: str, env: gym.Env, time_series_model: Optional[Union[TimeSeriesLSTM, MultiTimeframeModel]] = None):
        """
        Load a saved agent.
        
        Parameters:
        -----------
        path : str
            Path to saved agent
        env : gym.Env
            Trading environment
        time_series_model : TimeSeriesLSTM or MultiTimeframeModel, optional
            Trained time series model
            
        Returns:
        --------
        DQNTradingAgent
            Loaded agent
        """
        # Load hyperparameters
        import json
        with open(os.path.join(path, 'hyperparams.json'), 'r') as f:
            hyperparams = json.load(f)
        
        # Create agent instance
        agent = cls(
            env=env,
            time_series_model=time_series_model,
            learning_rate=hyperparams['learning_rate'],
            gamma=hyperparams['gamma'],
            epsilon_start=hyperparams['epsilon_start'],
            epsilon_end=hyperparams['epsilon_end'],
            epsilon_decay_steps=hyperparams['epsilon_decay_steps'],
            batch_size=hyperparams['batch_size'],
            update_target_every=hyperparams['update_target_every'],
            hidden_units=hyperparams['hidden_units'],
            double_dqn=hyperparams['double_dqn']
        )
        
        # Set current epsilon
        agent.epsilon = hyperparams['epsilon']
        agent.step_count = hyperparams['step_count']
        
        # Load network weights
        agent.q_network = load_model(os.path.join(path, 'q_network.h5'))
        agent.target_network = load_model(os.path.join(path, 'target_network.h5'))
        
        logger.info(f"Agent loaded from {path}")
        
        return agent
    
    def plot_training_results(self, save_path: str = None):
        """
        Plot training results.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot rewards
        if self.reward_history:
            ax1.plot(self.reward_history)
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Rewards')
            ax1.grid(True)
            
            # Add moving average
            window_size = min(20, len(self.reward_history))
            if window_size > 0:
                rewards_ma = np.convolve(self.reward_history, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(self.reward_history)), rewards_ma, 'r', label=f'{window_size}-episode MA')
                ax1.legend()
        
        # Plot epsilon
        if self.epsilon_history:
            ax2.plot(self.epsilon_history)
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Exploration Rate')
            ax2.grid(True)
        
        # Plot loss
        if self.loss_history:
            # Smooth the loss curve
            window_size = min(100, len(self.loss_history))
            if window_size > 0:
                loss_ma = np.convolve(self.loss_history, np.ones(window_size)/window_size, mode='valid')
                ax3.plot(range(window_size-1, len(self.loss_history)), loss_ma)
                ax3.set_ylabel('Loss')
                ax3.set_xlabel('Training Steps')
                ax3.set_title('Training Loss')
                ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        
        return fig

# Function to create and configure DQN agent
def create_dqn_agent(
    env: gym.Env,
    time_series_model: Optional[Union[TimeSeriesLSTM, MultiTimeframeModel]] = None,
    **kwargs
) -> DQNTradingAgent:
    """
    Create a DQN agent with custom configuration.
    
    Parameters:
    -----------
    env : gym.Env
        Trading environment
    time_series_model : TimeSeriesLSTM or MultiTimeframeModel, optional
        Trained time series model
    **kwargs : dict
        Additional arguments for DQN agent
        
    Returns:
    --------
    DQNTradingAgent
        Configured DQN agent
    """
    # Set default parameters from config
    params = {
        'learning_rate': RL_SETTINGS.get('learning_rate', 0.0005),
        'gamma': RL_SETTINGS.get('gamma', 0.99),
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_steps': 10000,
        'batch_size': RL_SETTINGS.get('batch_size', 64),
        'update_target_every': 100,
        'buffer_capacity': 10000,
        'hidden_units': [64, 64],
        'double_dqn': True
    }
    
    # Update with custom parameters
    params.update(kwargs)
    
    # Create agent
    agent = DQNTradingAgent(
        env=env,
        time_series_model=time_series_model,
        **params
    )
    
    return agent 