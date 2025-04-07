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
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
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
        time_series_model: Optional[TimeSeriesLSTM] = None,
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
        tensorboard_log: str = None
    ):
        """
        Initialize the DQN trading agent.
        
        Parameters:
        -----------
        env : gym.Env
            Trading environment
        time_series_model : TimeSeriesLSTM, optional
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
        """
        # Environment
        self.env = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        # Time series model
        self.time_series_model = time_series_model
        self.use_predictions = time_series_model is not None
        
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
            hidden_units = [64, 64]
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
            
            # Add time series prediction input if available
            prediction_input = Input(shape=(1,))  # Single step prediction
            
            # Concatenate state and prediction
            combined = Concatenate()([state_features, prediction_input])
            
            # Hidden layers
            x = combined
            for units in self.hidden_units:
                x = Dense(units, activation='relu')(x)
            
            # Output layer (Q-values for each action)
            outputs = Dense(self.action_dim)(x)
            
            # Build model
            model = Model(inputs=[state_input, prediction_input], outputs=outputs)
        else:
            # Simple sequential model
            model = Sequential()
            model.add(Flatten(input_shape=self.state_dim))
            
            # Hidden layers
            for units in self.hidden_units:
                model.add(Dense(units, activation='relu'))
            
            # Output layer (Q-values for each action)
            model.add(Dense(self.action_dim))
        
        # Compile model
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
        Select an action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state
        prediction : np.ndarray, optional
            Price prediction from time series model
        deterministic : bool
            Whether to use a deterministic policy (for evaluation)
            
        Returns:
        --------
        int
            Selected action
        """
        # Use deterministic policy for evaluation
        if deterministic:
            # Get Q-values
            if self.use_predictions and prediction is not None:
                q_values = self.q_network.predict([np.expand_dims(state, axis=0), 
                                                  np.expand_dims(prediction, axis=0)])[0]
            else:
                q_values = self.q_network.predict(np.expand_dims(state, axis=0))[0]
            
            # Select best action
            return np.argmax(q_values)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random action
            return np.random.randint(self.action_dim)
        else:
            # Get Q-values
            if self.use_predictions and prediction is not None:
                q_values = self.q_network.predict([np.expand_dims(state, axis=0), 
                                                  np.expand_dims(prediction, axis=0)])[0]
            else:
                q_values = self.q_network.predict(np.expand_dims(state, axis=0))[0]
            
            # Select best action
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
                prediction = self._get_price_prediction(state)
            
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
    
    def _get_price_prediction(self, state):
        """
        Get price prediction from time series model.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state of the environment
            
        Returns:
        --------
        float
            Price prediction
        """
        if not self.use_predictions or self.time_series_model is None:
            return 0.0
        
        try:
            # Extract the price data from the state
            # This assumes the state contains the price history in a specific format
            # Adjust this based on your actual environment state structure
            
            # For CryptoTradingEnv, the state typically includes:
            # 1. Price/OHLCV data
            # 2. Technical indicators
            # 3. Position information
            
            # Extract the relevant price data
            env = self.env
            if hasattr(env, 'prices') and hasattr(env, 'current_step'):
                # Get the current price window
                window_start = max(0, env.current_step - env.lookback_window)
                window_end = env.current_step + 1  # +1 because of Python slicing
                
                # Extract the window of prices needed for prediction
                if hasattr(env, 'data'):
                    # If the environment has the full data
                    price_window = env.data.iloc[window_start:window_end][['close']].values
                    
                    # Normalize the price window if needed
                    if hasattr(env, 'normalizer') and env.normalizer is not None:
                        # Use the environment's normalizer if available
                        price_window = env.normalizer.transform(price_window)
                else:
                    # Fallback to the prices array
                    price_window = env.prices[window_start:window_end]
                
                # Get the sequence length required by the model
                required_seq_length = self.time_series_model.sequence_length
                
                # Pad or truncate to match the required sequence length
                if len(price_window) < required_seq_length:
                    # Pad with zeros if the window is too short
                    padding = np.zeros((required_seq_length - len(price_window), price_window.shape[1]))
                    price_window = np.vstack([padding, price_window])
                elif len(price_window) > required_seq_length:
                    # Use only the most recent data
                    price_window = price_window[-required_seq_length:]
                
                # Reshape for the model
                if hasattr(self.time_series_model, 'input_shape'):
                    expected_shape = self.time_series_model.input_shape
                    price_window = price_window.reshape((1, *expected_shape))
                else:
                    # Default reshape for sequence models
                    price_window = price_window.reshape((1, price_window.shape[0], price_window.shape[1]))
                
                # Get prediction from the model
                prediction = self.time_series_model.predict(price_window)
                
                # Return the first prediction value
                return prediction[0][0]  # Assuming prediction shape is (1, forecast_horizon)
            
            # Fallback: try to infer from raw state
            # This is less ideal but may work if the environment structure is unknown
            logger.warning("Using fallback method for price prediction, might be inaccurate")
            if len(state.shape) == 1:
                # Assume the first elements are price-related
                price_data = state[:min(30, len(state))].reshape(1, -1, 1)
                prediction = self.time_series_model.predict(price_data)
                return prediction[0][0]
            
        except Exception as e:
            logger.warning(f"Error getting price prediction: {e}")
        
        # Return default value if prediction failed
        return 0.0
    
    def _train_step(self):
        """
        Perform a single training step on a batch from the replay buffer.
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Get predictions if using time series model
        predictions = None
        next_predictions = None
        if self.use_predictions:
            # Get predictions for current and next states
            predictions = np.array([self._get_price_prediction(state) for state in states])
            next_predictions = np.array([self._get_price_prediction(state) for state in next_states])
            
            # Reshape to (batch_size, 1)
            predictions = predictions.reshape(-1, 1)
            next_predictions = next_predictions.reshape(-1, 1)
        
        # Compute target Q-values
        if self.double_dqn:
            # Double DQN: use online network to select actions, target network to evaluate
            if self.use_predictions and next_predictions is not None:
                # Select actions using online network
                online_q_values = self.q_network.predict([next_states, next_predictions])
                next_actions = np.argmax(online_q_values, axis=1)
                
                # Evaluate using target network
                target_q_values = self.target_network.predict([next_states, next_predictions])
                target_q_values = np.array([target_q_values[i, next_actions[i]] for i in range(len(next_actions))])
            else:
                # Select actions using online network
                online_q_values = self.q_network.predict(next_states)
                next_actions = np.argmax(online_q_values, axis=1)
                
                # Evaluate using target network
                target_q_values = self.target_network.predict(next_states)
                target_q_values = np.array([target_q_values[i, next_actions[i]] for i in range(len(next_actions))])
        else:
            # Standard DQN: use target network to select and evaluate
            if self.use_predictions and next_predictions is not None:
                target_q_values = np.max(self.target_network.predict([next_states, next_predictions]), axis=1)
            else:
                target_q_values = np.max(self.target_network.predict(next_states), axis=1)
        
        # Compute target values
        targets = rewards + self.gamma * target_q_values * (1 - dones)
        
        # Get current Q-values
        if self.use_predictions and predictions is not None:
            current_q = self.q_network.predict([states, predictions])
        else:
            current_q = self.q_network.predict(states)
        
        # Create target Q-value matrix
        target_q = current_q.copy()
        for i in range(len(actions)):
            target_q[i, actions[i]] = targets[i]
        
        # Train the model
        if self.use_predictions and predictions is not None:
            history = self.q_network.fit(
                [states, predictions], target_q, 
                batch_size=self.batch_size, 
                verbose=0
            )
        else:
            history = self.q_network.fit(
                states, target_q, 
                batch_size=self.batch_size, 
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
                    prediction = self._get_price_prediction(state)
                
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
    def load(cls, path: str, env: gym.Env, time_series_model: Optional[TimeSeriesLSTM] = None):
        """
        Load a saved agent.
        
        Parameters:
        -----------
        path : str
            Path to saved agent
        env : gym.Env
            Trading environment
        time_series_model : TimeSeriesLSTM, optional
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
    time_series_model: Optional[TimeSeriesLSTM] = None,
    **kwargs
) -> DQNTradingAgent:
    """
    Create a DQN agent with custom configuration.
    
    Parameters:
    -----------
    env : gym.Env
        Trading environment
    time_series_model : TimeSeriesLSTM, optional
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