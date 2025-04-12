import numpy as np
from typing import Dict, List, Optional, Union
import torch
from datetime import datetime, timedelta

class CurriculumLearning:
    def __init__(
        self,
        difficulty_metrics: List[str] = ['volatility', 'trend_strength', 'market_regime'],
        difficulty_levels: int = 5,
        transition_threshold: float = 0.8,
        min_episodes_per_level: int = 100,
        max_episodes_per_level: int = 500
    ):
        """
        Initialize curriculum learning for trading environment
        
        Parameters:
        -----------
        difficulty_metrics : List[str]
            Metrics used to determine difficulty level
        difficulty_levels : int
            Number of difficulty levels
        transition_threshold : float
            Performance threshold required to advance to next level
        min_episodes_per_level : int
            Minimum number of episodes to spend at each level
        max_episodes_per_level : int
            Maximum number of episodes to spend at each level
        """
        self.difficulty_metrics = difficulty_metrics
        self.difficulty_levels = difficulty_levels
        self.transition_threshold = transition_threshold
        self.min_episodes_per_level = min_episodes_per_level
        self.max_episodes_per_level = max_episodes_per_level
        
        self.current_level = 0
        self.episodes_at_current_level = 0
        self.performance_history = []
        self.level_history = []
        
    def calculate_difficulty(self, market_data: Dict[str, np.ndarray]) -> float:
        """
        Calculate overall difficulty score for market data
        
        Parameters:
        -----------
        market_data : Dict[str, np.ndarray]
            Market data with different metrics
            
        Returns:
        --------
        float
            Difficulty score between 0 and 1
        """
        difficulty_scores = []
        
        if 'volatility' in self.difficulty_metrics:
            # Calculate volatility-based difficulty
            returns = np.diff(np.log(market_data['close']))
            volatility = np.std(returns)
            # Normalize volatility to 0-1 range
            volatility_score = min(volatility / 0.05, 1.0)  # 5% daily volatility as max
            difficulty_scores.append(volatility_score)
            
        if 'trend_strength' in self.difficulty_metrics:
            # Calculate trend strength using ADX
            if 'adx' in market_data:
                adx = market_data['adx']
                trend_score = min(adx / 50, 1.0)  # ADX > 50 considered strong trend
                difficulty_scores.append(trend_score)
                
        if 'market_regime' in self.difficulty_metrics:
            # Calculate market regime complexity
            if 'rsi' in market_data:
                rsi = market_data['rsi']
                # More complex when RSI oscillates between overbought/oversold
                regime_changes = np.sum((rsi > 70) | (rsi < 30))
                regime_score = min(regime_changes / len(rsi), 1.0)
                difficulty_scores.append(regime_score)
        
        # Average all available difficulty scores
        return np.mean(difficulty_scores) if difficulty_scores else 0.5
        
    def get_current_level_parameters(self) -> Dict:
        """
        Get parameters for current difficulty level
        
        Returns:
        --------
        Dict
            Parameters adjusted for current difficulty level
        """
        level_params = {
            'position_size_multiplier': 1.0 - (self.current_level * 0.1),  # Reduce position size at higher levels
            'transaction_cost_multiplier': 1.0 + (self.current_level * 0.1),  # Increase costs at higher levels
            'reward_scale': 1.0 - (self.current_level * 0.1),  # Reduce rewards at higher levels
            'max_holding_period': max(1, 10 - self.current_level),  # Shorter holding periods at higher levels
            'required_sharpe_ratio': 0.5 + (self.current_level * 0.1),  # Higher Sharpe requirement at higher levels
        }
        return level_params
        
    def update_level(self, episode_performance: Dict[str, float]) -> bool:
        """
        Update curriculum level based on performance
        
        Parameters:
        -----------
        episode_performance : Dict[str, float]
            Performance metrics for the episode
            
        Returns:
        --------
        bool
            Whether level was changed
        """
        self.episodes_at_current_level += 1
        self.performance_history.append(episode_performance)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(episode_performance)
        
        # Check if we should advance to next level
        if (self.current_level < self.difficulty_levels - 1 and
            performance_score >= self.transition_threshold and
            self.episodes_at_current_level >= self.min_episodes_per_level):
            
            self.current_level += 1
            self.episodes_at_current_level = 0
            self.level_history.append({
                'level': self.current_level,
                'episode': len(self.performance_history),
                'performance': performance_score
            })
            return True
            
        # Check if we should regress to previous level
        elif (self.current_level > 0 and
              performance_score < self.transition_threshold * 0.5 and
              self.episodes_at_current_level >= self.min_episodes_per_level):
            
            self.current_level -= 1
            self.episodes_at_current_level = 0
            self.level_history.append({
                'level': self.current_level,
                'episode': len(self.performance_history),
                'performance': performance_score
            })
            return True
            
        # Force level change if stuck too long
        elif self.episodes_at_current_level >= self.max_episodes_per_level:
            if performance_score >= self.transition_threshold:
                self.current_level = min(self.current_level + 1, self.difficulty_levels - 1)
            else:
                self.current_level = max(self.current_level - 1, 0)
            self.episodes_at_current_level = 0
            return True
            
        return False
        
    def _calculate_performance_score(self, performance: Dict[str, float]) -> float:
        """
        Calculate overall performance score from metrics
        
        Parameters:
        -----------
        performance : Dict[str, float]
            Performance metrics
            
        Returns:
        --------
        float
            Normalized performance score between 0 and 1
        """
        # Weight different metrics
        weights = {
            'sharpe_ratio': 0.4,
            'win_rate': 0.3,
            'profit_factor': 0.2,
            'max_drawdown': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in performance:
                if metric == 'max_drawdown':
                    # Invert drawdown (smaller is better)
                    normalized = 1.0 - min(performance[metric] / 0.5, 1.0)
                else:
                    # Normalize other metrics
                    normalized = min(performance[metric] / 2.0, 1.0)
                score += weight * normalized
                
        return score
        
    def get_curriculum_state(self) -> Dict:
        """
        Get current curriculum learning state
        
        Returns:
        --------
        Dict
            Current state including level, performance history, etc.
        """
        return {
            'current_level': self.current_level,
            'episodes_at_current_level': self.episodes_at_current_level,
            'performance_history': self.performance_history,
            'level_history': self.level_history,
            'current_parameters': self.get_current_level_parameters()
        } 