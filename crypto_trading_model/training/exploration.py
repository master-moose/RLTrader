import numpy as np
from typing import Dict, Optional, Union
import torch
from abc import ABC, abstractmethod

class ExplorationStrategy(ABC):
    @abstractmethod
    def get_action(self, q_values: torch.Tensor) -> int:
        """Get action based on exploration strategy"""
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        """Update exploration parameters"""
        pass

class EpsilonGreedy(ExplorationStrategy):
    def __init__(
        self,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.01,
        decay_steps: int = 100000,
        decay_type: str = 'linear'
    ):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0
        self.epsilon = initial_epsilon
        
    def get_action(self, q_values: torch.Tensor) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(q_values.shape[0])
        return torch.argmax(q_values).item()
        
    def update(self, **kwargs):
        self.current_step += 1
        if self.decay_type == 'linear':
            self.epsilon = max(
                self.final_epsilon,
                self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * 
                (self.current_step / self.decay_steps)
            )
        elif self.decay_type == 'exponential':
            self.epsilon = self.final_epsilon + (
                self.initial_epsilon - self.final_epsilon
            ) * np.exp(-1. * self.current_step / self.decay_steps)

class BoltzmannExploration(ExplorationStrategy):
    def __init__(
        self,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.1,
        decay_steps: int = 100000
    ):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.decay_steps = decay_steps
        self.current_step = 0
        self.temperature = initial_temperature
        
    def get_action(self, q_values: torch.Tensor) -> int:
        # Apply temperature scaling
        scaled_q = q_values / self.temperature
        # Subtract max for numerical stability
        scaled_q = scaled_q - torch.max(scaled_q)
        # Compute softmax
        probs = torch.softmax(scaled_q, dim=0)
        # Sample action
        return torch.multinomial(probs, 1).item()
        
    def update(self, **kwargs):
        self.current_step += 1
        self.temperature = self.final_temperature + (
            self.initial_temperature - self.final_temperature
        ) * np.exp(-1. * self.current_step / self.decay_steps)

class NoisyNet(ExplorationStrategy):
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
        
    def get_action(self, q_values: torch.Tensor) -> int:
        # Add noise to Q-values
        noisy_q = q_values + torch.randn_like(q_values) * self.noise_std
        return torch.argmax(noisy_q).item()
        
    def update(self, **kwargs):
        # Optionally adjust noise based on performance
        if 'performance' in kwargs:
            performance = kwargs['performance']
            # Reduce noise as performance improves
            self.noise_std = max(0.01, self.noise_std * (1 - performance))

class UCBExploration(ExplorationStrategy):
    def __init__(self, c: float = 2.0):
        self.c = c
        self.action_counts = {}
        self.total_steps = 0
        
    def get_action(self, q_values: torch.Tensor) -> int:
        num_actions = q_values.shape[0]
        
        # Initialize counts for new actions
        for a in range(num_actions):
            if a not in self.action_counts:
                self.action_counts[a] = 1
                
        # Calculate UCB values
        ucb_values = torch.zeros_like(q_values)
        for a in range(num_actions):
            if self.action_counts[a] == 0:
                ucb_values[a] = float('inf')
            else:
                exploration_bonus = self.c * np.sqrt(
                    np.log(self.total_steps) / self.action_counts[a]
                )
                ucb_values[a] = q_values[a] + exploration_bonus
                
        # Select action
        action = torch.argmax(ucb_values).item()
        self.action_counts[action] += 1
        self.total_steps += 1
        return action
        
    def update(self, **kwargs):
        # Optionally adjust exploration parameter
        if 'performance' in kwargs:
            performance = kwargs['performance']
            # Reduce exploration as performance improves
            self.c = max(0.1, self.c * (1 - performance))

class AdaptiveExploration(ExplorationStrategy):
    def __init__(
        self,
        strategies: Dict[str, ExplorationStrategy],
        initial_weights: Optional[Dict[str, float]] = None
    ):
        self.strategies = strategies
        self.weights = initial_weights or {
            name: 1.0 / len(strategies) for name in strategies
        }
        self.strategy_performance = {
            name: [] for name in strategies
        }
        self.window_size = 100
        
    def get_action(self, q_values: torch.Tensor) -> int:
        # Select strategy based on weights
        strategy_name = np.random.choice(
            list(self.weights.keys()),
            p=list(self.weights.values())
        )
        return self.strategies[strategy_name].get_action(q_values)
        
    def update(self, **kwargs):
        # Update each strategy
        for strategy in self.strategies.values():
            strategy.update(**kwargs)
            
        # Update strategy weights based on performance
        if 'episode_rewards' in kwargs:
            rewards = kwargs['episode_rewards']
            for name, strategy in self.strategies.items():
                # Calculate strategy's contribution to rewards
                strategy_rewards = rewards.get(name, [])
                if strategy_rewards:
                    avg_reward = np.mean(strategy_rewards[-self.window_size:])
                    self.strategy_performance[name].append(avg_reward)
                    
                    # Update weights using softmax on recent performance
                    recent_performance = np.mean(
                        self.strategy_performance[name][-self.window_size:]
                    )
                    self.weights[name] = np.exp(recent_performance)
                    
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {
                name: weight / total_weight
                for name, weight in self.weights.items()
            } 