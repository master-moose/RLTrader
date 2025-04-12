import numpy as np
from typing import Dict, Optional, Union
import torch
from abc import ABC, abstractmethod

class RiskManager(ABC):
    @abstractmethod
    def calculate_position_size(
        self,
        current_price: float,
        account_balance: float,
        market_data: Dict[str, np.ndarray],
        model_confidence: float
    ) -> float:
        """Calculate position size based on risk parameters"""
        pass
    
    @abstractmethod
    def update_risk_metrics(self, trade_results: Dict[str, float]):
        """Update risk metrics based on trade results"""
        pass

class KellyCriterionRiskManager(RiskManager):
    def __init__(
        self,
        max_position_size: float = 0.2,  # Maximum 20% of account per trade
        min_position_size: float = 0.01,  # Minimum 1% of account per trade
        win_rate_window: int = 100,  # Window for calculating win rate
        profit_factor_window: int = 100  # Window for calculating profit factor
    ):
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.win_rate_window = win_rate_window
        self.profit_factor_window = profit_factor_window
        self.trade_history = []
        
    def calculate_position_size(
        self,
        current_price: float,
        account_balance: float,
        market_data: Dict[str, np.ndarray],
        model_confidence: float
    ) -> float:
        if not self.trade_history:
            return self.min_position_size
            
        # Calculate win rate and profit factor
        recent_trades = self.trade_history[-self.win_rate_window:]
        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        total_trades = len(recent_trades)
        win_rate = wins / total_trades if total_trades > 0 else 0.5
        
        # Calculate average win/loss
        avg_win = np.mean([t['pnl'] for t in recent_trades if t['pnl'] > 0]) if wins > 0 else 1.0
        avg_loss = abs(np.mean([t['pnl'] for t in recent_trades if t['pnl'] < 0])) if total_trades - wins > 0 else 1.0
        
        # Kelly criterion formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = avg_win / avg_loss
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b
        
        # Adjust for model confidence
        kelly_fraction *= model_confidence
        
        # Apply position size limits
        position_size = np.clip(
            kelly_fraction,
            self.min_position_size,
            self.max_position_size
        )
        
        return position_size
        
    def update_risk_metrics(self, trade_results: Dict[str, float]):
        self.trade_history.append(trade_results)

class VolatilityAdjustedRiskManager(RiskManager):
    def __init__(
        self,
        target_volatility: float = 0.2,  # Target annualized volatility
        max_drawdown: float = 0.2,  # Maximum allowed drawdown
        volatility_window: int = 20,  # Window for volatility calculation
        position_adjustment_factor: float = 0.5  # How much to adjust position based on volatility
    ):
        self.target_volatility = target_volatility
        self.max_drawdown = max_drawdown
        self.volatility_window = volatility_window
        self.position_adjustment_factor = position_adjustment_factor
        self.portfolio_value_history = []
        
    def calculate_position_size(
        self,
        current_price: float,
        account_balance: float,
        market_data: Dict[str, np.ndarray],
        model_confidence: float
    ) -> float:
        # Calculate current portfolio volatility
        if len(self.portfolio_value_history) >= self.volatility_window:
            returns = np.diff(np.log(self.portfolio_value_history))
            current_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        else:
            current_volatility = self.target_volatility
            
        # Calculate volatility adjustment
        volatility_ratio = self.target_volatility / current_volatility
        volatility_adjustment = min(
            1.0,
            volatility_ratio ** self.position_adjustment_factor
        )
        
        # Calculate drawdown adjustment
        if self.portfolio_value_history:
            current_drawdown = 1 - account_balance / max(self.portfolio_value_history)
            drawdown_adjustment = max(
                0.0,
                1 - (current_drawdown / self.max_drawdown)
            )
        else:
            drawdown_adjustment = 1.0
            
        # Calculate market volatility
        if 'close' in market_data:
            market_returns = np.diff(np.log(market_data['close']))
            market_volatility = np.std(market_returns) * np.sqrt(252)
            market_adjustment = min(
                1.0,
                (self.target_volatility / market_volatility) ** 0.5
            )
        else:
            market_adjustment = 1.0
            
        # Combine adjustments
        position_size = (
            volatility_adjustment *
            drawdown_adjustment *
            market_adjustment *
            model_confidence
        )
        
        return position_size
        
    def update_risk_metrics(self, trade_results: Dict[str, float]):
        self.portfolio_value_history.append(trade_results['portfolio_value'])

class AdaptiveRiskManager(RiskManager):
    def __init__(
        self,
        risk_managers: Dict[str, RiskManager],
        initial_weights: Optional[Dict[str, float]] = None,
        performance_window: int = 100
    ):
        self.risk_managers = risk_managers
        self.weights = initial_weights or {
            name: 1.0 / len(risk_managers) for name in risk_managers
        }
        self.performance_window = performance_window
        self.manager_performance = {
            name: [] for name in risk_managers
        }
        
    def calculate_position_size(
        self,
        current_price: float,
        account_balance: float,
        market_data: Dict[str, np.ndarray],
        model_confidence: float
    ) -> float:
        # Get position sizes from all managers
        position_sizes = {}
        for name, manager in self.risk_managers.items():
            position_sizes[name] = manager.calculate_position_size(
                current_price,
                account_balance,
                market_data,
                model_confidence
            )
            
        # Weighted average of position sizes
        weighted_size = sum(
            size * self.weights[name]
            for name, size in position_sizes.items()
        )
        
        return weighted_size
        
    def update_risk_metrics(self, trade_results: Dict[str, float]):
        # Update each risk manager
        for manager in self.risk_managers.values():
            manager.update_risk_metrics(trade_results)
            
        # Update weights based on performance
        for name, manager in self.risk_managers.items():
            # Calculate manager's performance
            if len(self.manager_performance[name]) >= self.performance_window:
                recent_performance = np.mean(
                    self.manager_performance[name][-self.performance_window:]
                )
                self.weights[name] = np.exp(recent_performance)
                
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {
            name: weight / total_weight
            for name, weight in self.weights.items()
        }
        
    def add_performance_metric(
        self,
        manager_name: str,
        metric_name: str,
        value: float
    ):
        """Add a performance metric for a specific risk manager"""
        if manager_name in self.manager_performance:
            self.manager_performance[manager_name].append(value)
            
class RiskMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0
        
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        return np.mean(excess_returns) / np.std(downside_returns) if np.std(downside_returns) != 0 else 0.0
        
    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown)
        
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        max_dd = RiskMetrics.calculate_max_drawdown(portfolio_values)
        return np.mean(returns) / max_dd if max_dd != 0 else 0.0
        
    @staticmethod
    def calculate_win_rate(trades: np.ndarray) -> float:
        """Calculate win rate"""
        return np.mean(trades > 0)
        
    @staticmethod
    def calculate_profit_factor(trades: np.ndarray) -> float:
        """Calculate profit factor"""
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        gross_profit = np.sum(winning_trades)
        gross_loss = abs(np.sum(losing_trades))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
    @staticmethod
    def calculate_risk_metrics(
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        trades: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        return {
            'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': RiskMetrics.calculate_sortino_ratio(returns, risk_free_rate),
            'max_drawdown': RiskMetrics.calculate_max_drawdown(portfolio_values),
            'calmar_ratio': RiskMetrics.calculate_calmar_ratio(returns, portfolio_values),
            'win_rate': RiskMetrics.calculate_win_rate(trades),
            'profit_factor': RiskMetrics.calculate_profit_factor(trades)
        } 