import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import gym
import random
from collections import deque, namedtuple
import os
import json

from .policy import BasePolicy, MLPPolicy, LSTMPolicy, MultiTimeframePolicy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experience for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Replay buffer for storing experiences
    """
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Parameters:
        - capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Get current size of buffer"""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent
    """
    def __init__(self,
                 env,
                 policy_net: BasePolicy,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 double_dqn: bool = True,
                 prioritized_replay: bool = False,
                 device: str = 'cuda'):
        """
        Initialize DQN agent
        
        Parameters:
        - env: Environment
        - policy_net: Policy network
        - learning_rate: Learning rate
        - gamma: Discount factor
        - epsilon_start: Initial exploration rate
        - epsilon_end: Final exploration rate
        - epsilon_decay: Exploration rate decay
        - buffer_size: Replay buffer size
        - batch_size: Batch size for training
        - target_update_freq: Target network update frequency
        - double_dqn: Whether to use Double DQN
        - prioritized_replay: Whether to use prioritized experience replay
        - device: Device to use for tensor operations
        """
        self.env = env
        self.policy_net = policy_net
        self.target_net = type(policy_net)(*policy_net.__init__.__code__.co_varnames[1:])
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0
        
        # Metrics
        self.episode_rewards = []
        self.avg_losses = []
    
    def select_action(self, state, deterministic=False):
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        - state: Current state
        - deterministic: Whether to use deterministic policy
        
        Returns:
        - action: Selected action
        """
        if deterministic or random.random() > self.epsilon:
            # Exploit: use policy
            return self.policy_net.act(state, deterministic=True)
        else:
            # Explore: random action
            return random.randrange(self.env.action_space.n)
    
    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update(self):
        """Update policy network from replay buffer"""
        # Check if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = self._convert_states_to_tensor(states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = self._convert_states_to_tensor(next_states)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute Q values for current states
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next state values
        if self.double_dqn:
            # Double DQN: Use policy net to select action, target net to evaluate
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            # Standard DQN: Use target net for both
            next_q_values = self.target_net(next_states).max(dim=1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(q_values, target_q_values.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def _convert_states_to_tensor(self, states):
        """
        Convert states to tensors based on type
        
        Parameters:
        - states: List of states
        
        Returns:
        - tensor_states: Tensor representation of states
        """
        # Check if dictionary states (multi-timeframe)
        if isinstance(states[0], dict):
            # Dictionary states
            tensor_states = {}
            for tf in states[0].keys():
                # Stack states for this timeframe
                tf_states = [state[tf] for state in states]
                if isinstance(tf_states[0], np.ndarray):
                    tensor_states[tf] = torch.tensor(np.array(tf_states), dtype=torch.float32).to(self.device)
                else:
                    tensor_states[tf] = torch.stack([state[tf] for state in states]).to(self.device)
        else:
            # Simple states
            if isinstance(states[0], np.ndarray):
                tensor_states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            else:
                tensor_states = torch.stack([state for state in states]).to(self.device)
        
        return tensor_states
    
    def train(self, num_episodes, max_steps_per_episode=1000, eval_freq=10):
        """
        Train the agent
        
        Parameters:
        - num_episodes: Number of episodes to train
        - max_steps_per_episode: Maximum steps per episode
        - eval_freq: Evaluation frequency
        
        Returns:
        - metrics: Training metrics
        """
        metrics = {
            'episode_rewards': [],
            'avg_losses': [],
            'epsilons': [],
            'eval_returns': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update networks
                loss = self.update()
                if loss > 0:
                    episode_losses.append(loss)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update exploration rate
            self.update_epsilon()
            
            # Log metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['avg_losses'].append(np.mean(episode_losses) if episode_losses else 0)
            metrics['epsilons'].append(self.epsilon)
            
            # Evaluate periodically
            if episode % eval_freq == 0:
                eval_return = self.evaluate(5)
                metrics['eval_returns'].append(eval_return)
                
                logger.info(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                           f"Loss: {metrics['avg_losses'][-1]:.4f}, Epsilon: {self.epsilon:.4f}, "
                           f"Eval Return: {eval_return:.2f}")
            else:
                logger.info(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                           f"Loss: {metrics['avg_losses'][-1]:.4f}, Epsilon: {self.epsilon:.4f}")
        
        return metrics
    
    def evaluate(self, num_episodes=5):
        """
        Evaluate the agent
        
        Parameters:
        - num_episodes: Number of evaluation episodes
        
        Returns:
        - avg_return: Average episode return
        """
        returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_return = 0
            done = False
            
            while not done:
                # Select action deterministically
                action = self.select_action(state, deterministic=True)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update state and return
                state = next_state
                episode_return += reward
            
            returns.append(episode_return)
        
        avg_return = np.mean(returns)
        return avg_return
    
    def save(self, path):
        """
        Save agent
        
        Parameters:
        - path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save policy network
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy.pt'))
        
        # Save target network
        torch.save(self.target_net.state_dict(), os.path.join(path, 'target.pt'))
        
        # Save training metrics
        metrics = {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'avg_losses': self.avg_losses,
        }
        
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
    
    def load(self, path):
        """
        Load agent
        
        Parameters:
        - path: Directory path to load from
        """
        # Load policy network
        self.policy_net.load_state_dict(torch.load(os.path.join(path, 'policy.pt')))
        
        # Load target network
        self.target_net.load_state_dict(torch.load(os.path.join(path, 'target.pt')))
        
        # Load training metrics
        with open(os.path.join(path, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        self.epsilon = metrics['epsilon']
        self.steps = metrics['steps']
        self.episode_rewards = metrics['episode_rewards']
        self.avg_losses = metrics['avg_losses']


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent
    """
    def __init__(self,
                 env,
                 policy_net: BasePolicy,
                 value_net: nn.Module = None,
                 lr_policy: float = 3e-4,
                 lr_value: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = 'cuda'):
        """
        Initialize PPO agent
        
        Parameters:
        - env: Environment
        - policy_net: Policy network
        - value_net: Value network
        - lr_policy: Policy learning rate
        - lr_value: Value function learning rate
        - gamma: Discount factor
        - gae_lambda: GAE lambda parameter
        - clip_ratio: PPO clip ratio
        - value_coef: Value loss coefficient
        - entropy_coef: Entropy loss coefficient
        - max_grad_norm: Maximum gradient norm
        - device: Device to use for tensor operations
        """
        self.env = env
        self.policy_net = policy_net
        
        # Create simple value network if not provided
        if value_net is None:
            # Determine input size based on policy network
            if hasattr(policy_net, 'observation_dim'):
                input_size = policy_net.observation_dim
            else:
                input_size = env.observation_space.shape[0]
            
            # Create value network
            class ValueNet(nn.Module):
                def __init__(self, input_size):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )
                
                def forward(self, x):
                    return self.network(x).squeeze(-1)
            
            self.value_net = ValueNet(input_size)
        else:
            self.value_net = value_net
        
        # Hyperparameters
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # Metrics
        self.episode_rewards = []
        self.avg_policy_losses = []
        self.avg_value_losses = []
    
    def collect_trajectories(self, num_steps):
        """
        Collect trajectories from environment
        
        Parameters:
        - num_steps: Number of steps to collect
        
        Returns:
        - trajectory: Dictionary of trajectory data
        """
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        state = self.env.reset()
        episode_reward = 0
        
        for _ in range(num_steps):
            # Convert state to tensor
            state_tensor = self._convert_state_to_tensor(state)
            
            # Get action from policy
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
                action = self.policy_net.act(state_tensor)
                
                # Get value estimate
                value = self.value_net(state_tensor).item()
                
                # Get log probability of action
                log_prob = torch.log(action_probs[action]).item()
            
            # Take action in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Store step data
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['log_probs'].append(log_prob)
            trajectory['dones'].append(done)
            
            # Update state
            state = next_state
            episode_reward += reward
            
            if done:
                state = self.env.reset()
                self.episode_rewards.append(episode_reward)
                episode_reward = 0
        
        # Add final value estimate
        state_tensor = self._convert_state_to_tensor(state)
        final_value = self.value_net(state_tensor).item()
        trajectory['final_value'] = final_value
        
        return trajectory
    
    def compute_advantages(self, trajectory):
        """
        Compute advantages using Generalized Advantage Estimation (GAE)
        
        Parameters:
        - trajectory: Dictionary of trajectory data
        
        Returns:
        - returns: Discounted returns
        - advantages: Advantage estimates
        """
        rewards = trajectory['rewards']
        values = trajectory['values'] + [trajectory['final_value']]
        dones = trajectory['dones']
        
        returns = []
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            # Calculate TD error and GAE
            if t == len(rewards) - 1:
                # Last step
                next_non_terminal = 1.0 - dones[t]
                next_value = trajectory['final_value']
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Prepend returns and advantages
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)
        
        return returns, advantages
    
    def update(self, trajectory, batch_size=64, epochs=10):
        """
        Update policy and value networks
        
        Parameters:
        - trajectory: Dictionary of trajectory data
        - batch_size: Batch size for updates
        - epochs: Number of epochs to update
        
        Returns:
        - metrics: Update metrics
        """
        states = trajectory['states']
        actions = trajectory['actions']
        old_log_probs = trajectory['log_probs']
        
        # Compute returns and advantages
        returns, advantages = self.compute_advantages(trajectory)
        
        # Convert to tensors
        states_tensor = self._convert_states_to_tensor(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            states_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor
        )
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        policy_losses = []
        value_losses = []
        
        # Update networks for multiple epochs
        for _ in range(epochs):
            for batch in data_loader:
                # Unpack batch
                states_batch, actions_batch, old_log_probs_batch, returns_batch, advantages_batch = batch
                
                # Evaluate actions
                new_log_probs, entropy = self.policy_net.evaluate_actions(states_batch, actions_batch)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                # Compute surrogate losses
                surrogate1 = ratio * advantages_batch
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_batch
                
                # Compute policy loss
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Add entropy bonus
                policy_loss -= self.entropy_coef * entropy.mean()
                
                # Compute value loss
                value_pred = self.value_net(states_batch)
                value_loss = self.value_coef * F.mse_loss(value_pred, returns_batch)
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        # Compute average losses
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        
        # Store metrics
        self.avg_policy_losses.append(avg_policy_loss)
        self.avg_value_losses.append(avg_value_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }
    
    def _convert_state_to_tensor(self, state):
        """
        Convert state to tensor
        
        Parameters:
        - state: State to convert
        
        Returns:
        - tensor_state: Tensor representation of state
        """
        # Check if dictionary state (multi-timeframe)
        if isinstance(state, dict):
            # Dictionary state
            tensor_state = {}
            for tf in state.keys():
                if isinstance(state[tf], np.ndarray):
                    tensor_state[tf] = torch.tensor(state[tf], dtype=torch.float32).unsqueeze(0).to(self.device)
                else:
                    tensor_state[tf] = state[tf].unsqueeze(0).to(self.device)
        else:
            # Simple state
            if isinstance(state, np.ndarray):
                tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                tensor_state = state.unsqueeze(0).to(self.device)
        
        return tensor_state
    
    def _convert_states_to_tensor(self, states):
        """
        Convert states to tensors based on type
        
        Parameters:
        - states: List of states
        
        Returns:
        - tensor_states: Tensor representation of states
        """
        # Check if dictionary states (multi-timeframe)
        if isinstance(states[0], dict):
            # Dictionary states
            tensor_states = {}
            for tf in states[0].keys():
                # Stack states for this timeframe
                tf_states = [state[tf] for state in states]
                if isinstance(tf_states[0], np.ndarray):
                    tensor_states[tf] = torch.tensor(np.array(tf_states), dtype=torch.float32).to(self.device)
                else:
                    tensor_states[tf] = torch.stack([state[tf] for state in states]).to(self.device)
        else:
            # Simple states
            if isinstance(states[0], np.ndarray):
                tensor_states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            else:
                tensor_states = torch.stack([state for state in states]).to(self.device)
        
        return tensor_states
    
    def train(self, num_iterations, steps_per_iteration=2048, eval_freq=10):
        """
        Train the agent
        
        Parameters:
        - num_iterations: Number of training iterations
        - steps_per_iteration: Number of steps per iteration
        - eval_freq: Evaluation frequency
        
        Returns:
        - metrics: Training metrics
        """
        metrics = {
            'iteration_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'eval_returns': []
        }
        
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectory = self.collect_trajectories(steps_per_iteration)
            
            # Update policy and value networks
            update_info = self.update(trajectory)
            
            # Log metrics
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            metrics['iteration_rewards'].append(avg_reward)
            metrics['policy_losses'].append(update_info['policy_loss'])
            metrics['value_losses'].append(update_info['value_loss'])
            
            # Evaluate periodically
            if iteration % eval_freq == 0:
                eval_return = self.evaluate(5)
                metrics['eval_returns'].append(eval_return)
                
                logger.info(f"Iteration {iteration}/{num_iterations}, "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Policy Loss: {update_info['policy_loss']:.4f}, "
                           f"Value Loss: {update_info['value_loss']:.4f}, "
                           f"Eval Return: {eval_return:.2f}")
            else:
                logger.info(f"Iteration {iteration}/{num_iterations}, "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Policy Loss: {update_info['policy_loss']:.4f}, "
                           f"Value Loss: {update_info['value_loss']:.4f}")
        
        return metrics
    
    def evaluate(self, num_episodes=5):
        """
        Evaluate the agent
        
        Parameters:
        - num_episodes: Number of evaluation episodes
        
        Returns:
        - avg_return: Average episode return
        """
        returns = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_return = 0
            done = False
            
            while not done:
                # Convert state to tensor
                state_tensor = self._convert_state_to_tensor(state)
                
                # Select action deterministically
                action = self.policy_net.act(state_tensor, deterministic=True)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update state and return
                state = next_state
                episode_return += reward
            
            returns.append(episode_return)
        
        avg_return = np.mean(returns)
        return avg_return
    
    def save(self, path):
        """
        Save agent
        
        Parameters:
        - path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save policy network
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy.pt'))
        
        # Save value network
        torch.save(self.value_net.state_dict(), os.path.join(path, 'value.pt'))
        
        # Save training metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.avg_policy_losses,
            'value_losses': self.avg_value_losses,
        }
        
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
    
    def load(self, path):
        """
        Load agent
        
        Parameters:
        - path: Directory path to load from
        """
        # Load policy network
        self.policy_net.load_state_dict(torch.load(os.path.join(path, 'policy.pt')))
        
        # Load value network
        self.value_net.load_state_dict(torch.load(os.path.join(path, 'value.pt')))
        
        # Load training metrics
        with open(os.path.join(path, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        self.episode_rewards = metrics['episode_rewards']
        self.avg_policy_losses = metrics['policy_losses']
        self.avg_value_losses = metrics['value_losses'] 