"""
Performance metrics utilities for the crypto trading model.

This module provides functions for calculating and analyzing trading
performance metrics, including returns, risk metrics, and trade statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple, Optional
import logging
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_metrics')

class PerformanceMetrics:
    """
    Class that calculates and analyzes trading performance metrics.
    """
    
    def __init__(self, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize the performance metrics calculator.
        
        Parameters:
        -----------
        benchmark_returns : pd.Series, optional
            Returns of a benchmark for comparison
        """
        self.benchmark_returns = benchmark_returns
    
    def calculate_returns(
        self,
        equity_curve: pd.Series,
        initial_capital: float = 10000.0
    ) -> pd.DataFrame:
        """
        Calculate return metrics from an equity curve.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Series of account equity values over time
        initial_capital : float
            Initial capital amount
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with return metrics
        """
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            raise ValueError("Equity curve must have a DatetimeIndex")
        
        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Calculate drawdowns
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        
        # Create results DataFrame
        results = pd.DataFrame({
            'equity': equity_curve,
            'returns': returns,
            'cumulative_returns': cumulative_returns,
            'drawdowns': drawdowns
        })
        
        # Calculate total return
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
        
        # Calculate annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        # Calculate monthly returns
        monthly_returns = equity_curve.resample('M').last().pct_change().fillna(0)
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Return',
                'Monthly Avg Return',
                'Monthly Std Dev',
                'Positive Months',
                'Max Drawdown'
            ],
            'Value': [
                f"{total_return:.2%}",
                f"{annualized_return:.2%}",
                f"{monthly_returns.mean():.2%}",
                f"{monthly_returns.std():.2%}",
                f"{(monthly_returns > 0).sum()} / {len(monthly_returns)} ({(monthly_returns > 0).mean():.2%})",
                f"{drawdowns.min():.2%}"
            ]
        })
        
        return results, summary
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate risk metrics from a series of returns.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of period returns
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with calculated risk metrics
        """
        # Annualization factor (assuming daily returns)
        ann_factor = 252
        
        # Basic statistics
        mean_return = returns.mean()
        std_dev = returns.std()
        
        # Annualized statistics
        ann_return = (1 + mean_return) ** ann_factor - 1
        ann_vol = std_dev * np.sqrt(ann_factor)
        
        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe_ratio = (ann_return) / ann_vol if ann_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(ann_factor)
        sortino_ratio = (ann_return) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        # Calmar ratio
        calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (CVaR) / Expected Shortfall
        cvar_95 = returns[returns <= var_95].mean()
        
        # Create metrics dictionary
        metrics = {
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
        
        return metrics
    
    def analyze_trades(
        self,
        trades: pd.DataFrame,
        capital: float = 10000.0
    ) -> Dict[str, float]:
        """
        Analyze trade results for performance statistics.
        
        Parameters:
        -----------
        trades : pd.DataFrame
            DataFrame with trade results, must contain columns:
            - 'entry_date': entry timestamp
            - 'exit_date': exit timestamp
            - 'entry_price': entry price
            - 'exit_price': exit price
            - 'direction': 1 for long, -1 for short
            - 'size': trade size (units)
            - 'pnl': profit/loss in currency units
        capital : float
            Account capital for calculating return metrics
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with trade analysis metrics
        """
        if len(trades) == 0:
            logger.warning("No trades to analyze")
            return {}
        
        # Calculate basic trade statistics
        trades['return'] = trades['pnl'] / capital
        trades['duration'] = (trades['exit_date'] - trades['entry_date']).dt.total_seconds() / 3600  # hours
        
        # Separate winning and losing trades
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] <= 0]
        
        # Calculate trade metrics
        num_trades = len(trades)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / num_trades if num_trades > 0 else 0
        
        avg_winner = winning_trades['pnl'].mean() if num_winning > 0 else 0
        avg_loser = losing_trades['pnl'].mean() if num_losing > 0 else 0
        
        # Avoid division by zero
        profit_factor = (
            abs(winning_trades['pnl'].sum()) / abs(losing_trades['pnl'].sum())
            if num_losing > 0 and losing_trades['pnl'].sum() != 0
            else float('inf')
        )
        
        # Expected value per trade
        expectancy = (win_rate * avg_winner + (1 - win_rate) * avg_loser) if num_trades > 0 else 0
        
        # Average trade duration
        avg_duration = trades['duration'].mean()
        
        # Average risk-reward ratio (using avg winner/loser as proxy)
        risk_reward = abs(avg_winner / avg_loser) if avg_loser < 0 else float('inf')
        
        # Maximum consecutive winners and losers
        trades['win'] = trades['pnl'] > 0
        consecutive_wins = max(self._count_consecutive(trades['win'], True))
        consecutive_losses = max(self._count_consecutive(trades['win'], False))
        
        # Create metrics dictionary
        metrics = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_duration': avg_duration,
            'risk_reward': risk_reward,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
        
        return metrics
    
    def _count_consecutive(self, series: pd.Series, value) -> List[int]:
        """
        Count consecutive occurrences of a value in a series.
        
        Parameters:
        -----------
        series : pd.Series
            Series to analyze
        value : any
            Value to count consecutive occurrences of
            
        Returns:
        --------
        List[int]
            List of lengths of consecutive occurrences
        """
        # Convert series to numpy array for faster processing
        s = series.values
        
        # Find indices where values change
        change_points = np.where(np.diff(np.hstack(([False], s == value, [False]))))[0]
        
        # Compute run lengths
        run_lengths = np.diff(change_points)[::2]
        
        return run_lengths.tolist() if len(run_lengths) > 0 else [0]
    
    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark: Optional[pd.Series] = None,
        title: str = 'Equity Curve',
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot equity curve with optional benchmark comparison.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Series of account equity values over time
        benchmark : pd.Series, optional
            Benchmark values for comparison
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Normalize equity curve and benchmark to start at 1
        norm_equity = equity_curve / equity_curve.iloc[0]
        norm_equity.plot(ax=axes[0], label='Strategy')
        
        if benchmark is not None:
            if len(benchmark) != len(equity_curve):
                logger.warning("Benchmark and equity curve have different lengths")
            
            # Align benchmark to equity curve dates if necessary
            if benchmark.index[0] != equity_curve.index[0]:
                benchmark = benchmark.reindex(equity_curve.index, method='ffill')
            
            # Normalize benchmark to start at 1
            norm_benchmark = benchmark / benchmark.iloc[0]
            norm_benchmark.plot(ax=axes[0], label='Benchmark')
        
        axes[0].set_title(title)
        axes[0].set_ylabel('Growth of $1')
        axes[0].legend()
        axes[0].grid(True)
        
        # Drawdown subplot
        drawdown = (norm_equity / norm_equity.cummax() - 1) * 100
        drawdown.plot(ax=axes[1], color='red')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_ylim(bottom=min(drawdown.min() * 1.1, -1), top=1)  # Ensure 0 is visible
        axes[1].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_return_distribution(
        self,
        returns: pd.Series,
        bins: int = 50,
        title: str = 'Return Distribution',
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot distribution of returns with normal distribution fit.
        
        Parameters:
        -----------
        returns : pd.Series
            Series of period returns
        bins : int
            Number of histogram bins
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram
        returns.hist(bins=bins, density=True, alpha=0.6, ax=ax)
        
        # Plot normal distribution fit
        x = np.linspace(returns.min(), returns.max(), 100)
        y = self._normal_pdf(x, returns.mean(), returns.std())
        ax.plot(x, y, 'r--', linewidth=2, label='Normal Distribution')
        
        # Plot VaR and CVaR lines
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        ax.axvline(var_95, color='red', linestyle='-', linewidth=1, label=f'VaR (95%): {var_95:.2%}')
        ax.axvline(cvar_95, color='darkred', linestyle='-', linewidth=1, label=f'CVaR (95%): {cvar_95:.2%}')
        
        ax.set_title(title)
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def _normal_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """
        Calculate normal probability density function.
        
        Parameters:
        -----------
        x : np.ndarray
            Array of x values
        mean : float
            Mean of the distribution
        std : float
            Standard deviation of the distribution
            
        Returns:
        --------
        np.ndarray
            Array of PDF values
        """
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    def generate_performance_report(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        initial_capital: float = 10000.0,
        report_title: str = 'Trading Performance Report'
    ) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Parameters:
        -----------
        equity_curve : pd.Series
            Series of account equity values over time
        trades : pd.DataFrame, optional
            DataFrame with trade results
        benchmark : pd.Series, optional
            Benchmark values for comparison
        initial_capital : float
            Initial capital amount
        report_title : str
            Title for the report
            
        Returns:
        --------
        Dict
            Dictionary containing all performance metrics and figures
        """
        # Calculate returns
        returns_data, returns_summary = self.calculate_returns(equity_curve, initial_capital)
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(returns_data['returns'])
        
        # Calculate trade metrics if trades are provided
        trade_metrics = {}
        if trades is not None and len(trades) > 0:
            trade_metrics = self.analyze_trades(trades, initial_capital)
        
        # Create figures
        equity_fig = self.plot_equity_curve(equity_curve, benchmark, 'Equity Curve')
        returns_fig = self.plot_return_distribution(returns_data['returns'], 50, 'Return Distribution')
        
        # Combine all results
        report = {
            'title': report_title,
            'returns_data': returns_data,
            'returns_summary': returns_summary,
            'risk_metrics': risk_metrics,
            'trade_metrics': trade_metrics,
            'figures': {
                'equity_curve': equity_fig,
                'return_distribution': returns_fig
            }
        }
        
        return report

# Function to calculate directional accuracy
def calculate_directional_accuracy(
    predictions: np.ndarray,
    actual: np.ndarray
) -> float:
    """
    Calculate directional accuracy of predictions.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted price changes
    actual : np.ndarray
        Actual price changes
        
    Returns:
    --------
    float
        Directional accuracy (0-1)
    """
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actual)
    
    correct = (pred_direction == actual_direction).sum()
    total = len(predictions)
    
    return correct / total if total > 0 else 0

# Function to calculate maximum adverse excursion (MAE)
def calculate_mae(
    trades: pd.DataFrame,
    price_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate Maximum Adverse Excursion for each trade.
    
    Parameters:
    -----------
    trades : pd.DataFrame
        DataFrame with trade information
    price_data : pd.DataFrame
        DataFrame with OHLCV price data
        
    Returns:
    --------
    pd.DataFrame
        Original trades DataFrame with MAE information added
    """
    result = trades.copy()
    result['mae'] = 0.0
    result['mae_pct'] = 0.0
    
    for i, trade in trades.iterrows():
        # Extract price data for the trade period
        mask = (price_data.index >= trade['entry_date']) & (price_data.index <= trade['exit_date'])
        trade_prices = price_data.loc[mask]
        
        if len(trade_prices) == 0:
            continue
        
        # Calculate MAE based on trade direction
        if trade['direction'] > 0:  # Long trade
            # Worst case is the lowest price during the trade
            worst_price = trade_prices['low'].min()
            result.loc[i, 'mae'] = trade['entry_price'] - worst_price
        else:  # Short trade
            # Worst case is the highest price during the trade
            worst_price = trade_prices['high'].max()
            result.loc[i, 'mae'] = worst_price - trade['entry_price']
        
        # Calculate MAE as percentage of entry price
        result.loc[i, 'mae_pct'] = result.loc[i, 'mae'] / trade['entry_price']
    
    return result

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Create sample equity curve
    dates = pd.date_range(start='2022-01-01', periods=252, freq='D')
    equity = pd.Series(
        data=10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates)))),
        index=dates
    )
    
    # Create sample benchmark (e.g., Bitcoin price)
    benchmark = pd.Series(
        data=40000 * (1 + np.cumsum(np.random.normal(0.0005, 0.03, len(dates)))),
        index=dates
    )
    
    # Initialize metrics calculator
    metrics_calc = PerformanceMetrics()
    
    # Generate performance report
    report = metrics_calc.generate_performance_report(
        equity_curve=equity,
        benchmark=benchmark,
        initial_capital=10000.0,
        report_title='Sample Trading Performance Report'
    )
    
    # Print key metrics
    for metric, value in report['risk_metrics'].items():
        print(f"{metric}: {value:.4f}") 