#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import logging

# Configure logging
logger = logging.getLogger('crypto_trading_model.evaluation')

def calculate_returns(positions: np.ndarray, price_changes: np.ndarray, 
                     transaction_cost: float = 0.001) -> np.ndarray:
    """
    Calculate returns from a series of positions and price changes.
    
    Parameters:
    -----------
    positions : np.ndarray
        Array of positions (1 for long, 0 for neutral, -1 for short)
    price_changes : np.ndarray
        Array of price changes (percentage)
    transaction_cost : float, optional
        Transaction cost as a percentage, by default 0.001 (0.1%)
    
    Returns:
    --------
    np.ndarray
        Array of returns
    """
    # Calculate returns from positions
    position_returns = positions * price_changes
    
    # Calculate transaction costs
    position_changes = np.diff(np.append(0, positions))
    transaction_costs = np.abs(position_changes) * transaction_cost
    
    # Subtract transaction costs from returns
    returns = position_returns - transaction_costs
    
    return returns

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                         annualization_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.0
    annualization_factor : int, optional
        Annualization factor (252 for daily data), by default 252
    
    Returns:
    --------
    float
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(annualization_factor)
    
    return sharpe

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                          annualization_factor: int = 252) -> float:
    """
    Calculate the Sortino ratio.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    risk_free_rate : float, optional
        Risk-free rate, by default 0.0
    annualization_factor : int, optional
        Annualization factor (252 for daily data), by default 252
    
    Returns:
    --------
    float
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(annualization_factor)
    
    return sortino

def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate the maximum drawdown and its start/end points.
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Array of equity values
    
    Returns:
    --------
    tuple
        (max_drawdown, start_idx, end_idx)
    """
    if len(equity_curve) == 0:
        return 0.0, 0, 0
    
    # Calculate cumulative max
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    
    # Find the maximum drawdown
    max_drawdown = np.min(drawdown)
    end_idx = np.argmin(drawdown)
    
    # Find the start of the drawdown period
    start_idx = np.argmax(equity_curve[:end_idx+1])
    
    return max_drawdown, start_idx, end_idx

def calculate_calmar_ratio(returns: np.ndarray, period: int = 252) -> float:
    """
    Calculate the Calmar ratio (CAGR / Max Drawdown).
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    period : int, optional
        Period for annualization, by default 252
    
    Returns:
    --------
    float
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert returns to equity curve (starting at 1.0)
    equity_curve = (1 + returns).cumprod()
    
    # Calculate CAGR
    years = len(returns) / period
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate max drawdown
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    # Avoid division by zero
    if max_dd == 0:
        return np.inf if cagr > 0 else 0.0
    
    return cagr / abs(max_dd)

def calculate_win_rate(returns: np.ndarray) -> float:
    """
    Calculate the win rate (percentage of positive returns).
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    
    Returns:
    --------
    float
        Win rate (0.0 to 1.0)
    """
    if len(returns) == 0:
        return 0.0
    
    win_count = np.sum(returns > 0)
    total_count = len(returns)
    
    return win_count / total_count

def calculate_profit_factor(returns: np.ndarray) -> float:
    """
    Calculate the profit factor (sum of profits / sum of losses).
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    
    Returns:
    --------
    float
        Profit factor
    """
    if len(returns) == 0:
        return 0.0
    
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return np.inf if profits > 0 else 0.0
    
    return profits / losses

def calculate_average_trade(returns: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate average trade metrics.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    
    Returns:
    --------
    tuple
        (average_trade, average_win, average_loss)
    """
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    
    average_trade = np.mean(returns)
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    average_win = np.mean(wins) if len(wins) > 0 else 0.0
    average_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    return average_trade, average_win, average_loss

def calculate_expectancy(returns: np.ndarray) -> float:
    """
    Calculate the expectancy (expected return per trade).
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    
    Returns:
    --------
    float
        Expectancy
    """
    if len(returns) == 0:
        return 0.0
    
    win_rate = calculate_win_rate(returns)
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    average_win = np.mean(wins) if len(wins) > 0 else 0.0
    average_loss = np.mean(losses) if len(losses) > 0 else 0.0
    
    expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss))
    
    return expectancy

def calculate_recovery_factor(returns: np.ndarray) -> float:
    """
    Calculate the recovery factor (total return / max drawdown).
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    
    Returns:
    --------
    float
        Recovery factor
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert returns to equity curve (starting at 1.0)
    equity_curve = (1 + returns).cumprod()
    
    # Calculate total return
    total_return = equity_curve[-1] - equity_curve[0]
    
    # Calculate max drawdown
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    
    # Avoid division by zero
    if max_dd == 0:
        return np.inf if total_return > 0 else 0.0
    
    return total_return / abs(max_dd)

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate classification metrics for trading signals.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (1 for buy, 0 for hold, -1 for sell)
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities, by default None
    
    Returns:
    --------
    dict
        Dictionary of classification metrics
    """
    # Convert to integer classes if they are -1, 0, 1
    if set(np.unique(y_true)).issubset({-1, 0, 1}):
        # Map to 0, 1, 2 for sklearn metrics
        y_true_mapped = np.array(y_true) + 1
        y_pred_mapped = np.array(y_pred) + 1
    else:
        y_true_mapped = y_true
        y_pred_mapped = y_pred
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped)
    
    # Classification report
    report = classification_report(y_true_mapped, y_pred_mapped, output_dict=True)
    
    metrics = {
        'confusion_matrix': cm,
        'accuracy': report['accuracy'],
        'precision': {k: v['precision'] for k, v in report.items() if k.isdigit()},
        'recall': {k: v['recall'] for k, v in report.items() if k.isdigit()},
        'f1_score': {k: v['f1-score'] for k, v in report.items() if k.isdigit()}
    }
    
    # Calculate ROC curve and AUC if probabilities are provided
    if y_proba is not None and y_proba.shape[1] > 2:  # Multi-class case
        # One-vs-Rest ROC AUC
        metrics['roc_auc'] = {}
        for i in range(y_proba.shape[1]):
            fpr, tpr, _ = roc_curve((y_true_mapped == i).astype(int), y_proba[:, i])
            metrics['roc_auc'][i] = auc(fpr, tpr)
    
    return metrics

def evaluate_backtest(positions: np.ndarray, prices: np.ndarray, 
                    initial_capital: float = 10000.0,
                    transaction_cost: float = 0.001) -> Dict[str, Any]:
    """
    Evaluate a backtest given positions and prices.
    
    Parameters:
    -----------
    positions : np.ndarray
        Array of positions (1 for long, 0 for neutral, -1 for short)
    prices : np.ndarray
        Array of prices
    initial_capital : float, optional
        Initial capital, by default 10000.0
    transaction_cost : float, optional
        Transaction cost as a percentage, by default 0.001 (0.1%)
    
    Returns:
    --------
    dict
        Dictionary of backtest metrics
    """
    # Calculate price changes
    price_changes = np.diff(prices) / prices[:-1]
    price_changes = np.append(0, price_changes)  # Add 0 for the first period
    
    # Adjust positions to match price_changes length
    if len(positions) > len(price_changes):
        positions = positions[:len(price_changes)]
    elif len(positions) < len(price_changes):
        price_changes = price_changes[:len(positions)]
    
    # Calculate returns
    returns = calculate_returns(positions, price_changes, transaction_cost)
    
    # Calculate equity curve
    equity_curve = initial_capital * (1 + returns).cumprod()
    
    # Calculate various performance metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, max_dd_start, max_dd_end = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(returns)
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    avg_trade, avg_win, avg_loss = calculate_average_trade(returns)
    expectancy = calculate_expectancy(returns)
    recovery_factor = calculate_recovery_factor(returns)
    
    # Trade analysis
    trades = []
    curr_pos = 0
    entry_price = 0
    entry_idx = 0
    
    for i, pos in enumerate(positions):
        if pos != curr_pos:
            # Close previous position if it exists
            if curr_pos != 0:
                exit_price = prices[i]
                pnl = (exit_price - entry_price) / entry_price * curr_pos  # Adjust for direction
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': curr_pos,
                    'pnl': pnl,
                    'pnl_pct': pnl * 100,
                    'duration': i - entry_idx
                })
            
            # Open new position if not neutral
            if pos != 0:
                entry_price = prices[i]
                entry_idx = i
            
            curr_pos = pos
    
    # Close any open position at the end
    if curr_pos != 0:
        exit_price = prices[-1]
        pnl = (exit_price - entry_price) / entry_price * curr_pos
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': len(positions) - 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position': curr_pos,
            'pnl': pnl,
            'pnl_pct': pnl * 100,
            'duration': len(positions) - 1 - entry_idx
        })
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate trade statistics
    if not trades_df.empty:
        avg_duration = trades_df['duration'].mean()
        max_consecutive_losses = 0
        curr_consecutive_losses = 0
        
        for pnl in trades_df['pnl']:
            if pnl < 0:
                curr_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, curr_consecutive_losses)
            else:
                curr_consecutive_losses = 0
    else:
        avg_duration = 0
        max_consecutive_losses = 0
    
    # Compile all metrics
    metrics = {
        'initial_capital': initial_capital,
        'final_capital': equity_curve[-1],
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'max_drawdown_start': max_dd_start,
        'max_drawdown_end': max_dd_end,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'average_trade': avg_trade,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'expectancy': expectancy,
        'recovery_factor': recovery_factor,
        'num_trades': len(trades),
        'average_duration': avg_duration,
        'max_consecutive_losses': max_consecutive_losses,
        'equity_curve': equity_curve,
        'returns': returns,
        'trades': trades_df
    }
    
    return metrics

def plot_equity_curve(equity_curve: np.ndarray, title: str = 'Equity Curve', 
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the equity curve.
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Array of equity values
    title : str, optional
        Plot title, by default 'Equity Curve'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Equity')
    ax.grid(True)
    
    # Add horizontal line at initial equity
    ax.axhline(y=equity_curve[0], color='r', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_drawdown(equity_curve: np.ndarray, title: str = 'Drawdown', 
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the drawdown curve.
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Array of equity values
    title : str, optional
        Plot title, by default 'Drawdown'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='r')
    ax.plot(drawdown, color='r')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown')
    ax.grid(True)
    
    # Find and mark the maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    ax.scatter(max_dd_idx, drawdown[max_dd_idx], color='darkred', s=100, zorder=5)
    ax.annotate(f"Max DD: {drawdown[max_dd_idx]:.2%}", 
               (max_dd_idx, drawdown[max_dd_idx]),
               xytext=(30, 30),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='black'))
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_returns_distribution(returns: np.ndarray, title: str = 'Returns Distribution', 
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of returns.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    title : str, optional
        Plot title, by default 'Returns Distribution'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(returns, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Return')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    
    # Add mean and median lines
    ax.axvline(x=np.mean(returns), color='r', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
    ax.axvline(x=np.median(returns), color='g', linestyle='-.', label=f'Median: {np.median(returns):.4f}')
    
    # Add zero line
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_rolling_sharpe(returns: np.ndarray, window: int = 252, 
                      title: str = 'Rolling Sharpe Ratio', 
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the rolling Sharpe ratio.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    window : int, optional
        Rolling window size, by default 252
    title : str, optional
        Plot title, by default 'Rolling Sharpe Ratio'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Convert to pandas Series for rolling calculations
    returns_series = pd.Series(returns)
    
    # Calculate rolling Sharpe ratio
    rolling_sharpe = (returns_series.rolling(window).mean() / 
                     returns_series.rolling(window).std() * np.sqrt(252))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_sharpe.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True)
    
    # Add zero line
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_monthly_returns_heatmap(returns: np.ndarray, dates: pd.DatetimeIndex, 
                               title: str = 'Monthly Returns', 
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a heatmap of monthly returns.
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    dates : pd.DatetimeIndex
        Array of dates corresponding to returns
    title : str, optional
        Plot title, by default 'Monthly Returns'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Convert to pandas Series with DatetimeIndex
    returns_series = pd.Series(returns, index=dates)
    
    # Resample to get monthly returns
    monthly_returns = returns_series.resample('M').sum()
    
    # Create a pivot table for the heatmap
    monthly_pivot = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    pivot_table = monthly_pivot.pivot_table(index='Year', columns='Month', values='Return')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='RdYlGn', center=0, annot=True, fmt='.2%', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def generate_performance_report(metrics: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a comprehensive performance report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics
    output_dir : str
        Directory to save the report
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics_to_save = {k: v for k, v in metrics.items() 
                      if k not in ['equity_curve', 'returns', 'trades']}
    pd.Series(metrics_to_save).to_json(os.path.join(output_dir, 'metrics.json'))
    
    # Save trades to CSV
    if 'trades' in metrics and not metrics['trades'].empty:
        metrics['trades'].to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
    
    # Generate plots
    if 'equity_curve' in metrics:
        fig = plot_equity_curve(metrics['equity_curve'], 
                              title='Equity Curve',
                              save_path=os.path.join(output_dir, 'equity_curve.png'))
        plt.close(fig)
        
        fig = plot_drawdown(metrics['equity_curve'], 
                          title='Drawdown',
                          save_path=os.path.join(output_dir, 'drawdown.png'))
        plt.close(fig)
    
    if 'returns' in metrics:
        fig = plot_returns_distribution(metrics['returns'], 
                                      title='Returns Distribution',
                                      save_path=os.path.join(output_dir, 'returns_distribution.png'))
        plt.close(fig)
        
        if len(metrics['returns']) > 252:  # Only if we have enough data
            fig = plot_rolling_sharpe(metrics['returns'], 
                                    title='Rolling Sharpe Ratio (252-day)',
                                    save_path=os.path.join(output_dir, 'rolling_sharpe.png'))
            plt.close(fig)
    
    # Generate HTML report
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metrics {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .metric span {{ font-weight: bold; }}
                .images {{ display: flex; flex-wrap: wrap; }}
                .image-container {{ margin: 10px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Trading Performance Report</h1>
            
            <div class="metrics">
                <h2>Performance Metrics</h2>
                <div class="metric"><span>Initial Capital:</span> ${metrics['initial_capital']:.2f}</div>
                <div class="metric"><span>Final Capital:</span> ${metrics['final_capital']:.2f}</div>
                <div class="metric"><span>Total Return:</span> {metrics['total_return']:.2%}</div>
                <div class="metric"><span>Sharpe Ratio:</span> {metrics['sharpe_ratio']:.4f}</div>
                <div class="metric"><span>Sortino Ratio:</span> {metrics['sortino_ratio']:.4f}</div>
                <div class="metric"><span>Max Drawdown:</span> {metrics['max_drawdown']:.2%}</div>
                <div class="metric"><span>Calmar Ratio:</span> {metrics['calmar_ratio']:.4f}</div>
                <div class="metric"><span>Win Rate:</span> {metrics['win_rate']:.2%}</div>
                <div class="metric"><span>Profit Factor:</span> {metrics['profit_factor']:.4f}</div>
                <div class="metric"><span>Average Trade:</span> {metrics['average_trade']:.4%}</div>
                <div class="metric"><span>Average Win:</span> {metrics['average_win']:.4%}</div>
                <div class="metric"><span>Average Loss:</span> {metrics['average_loss']:.4%}</div>
                <div class="metric"><span>Expectancy:</span> {metrics['expectancy']:.4%}</div>
                <div class="metric"><span>Recovery Factor:</span> {metrics['recovery_factor']:.4f}</div>
                <div class="metric"><span>Number of Trades:</span> {metrics['num_trades']}</div>
                <div class="metric"><span>Average Trade Duration:</span> {metrics['average_duration']:.2f}</div>
                <div class="metric"><span>Max Consecutive Losses:</span> {metrics['max_consecutive_losses']}</div>
            </div>
            
            <div class="images">
                <div class="image-container">
                    <h3>Equity Curve</h3>
                    <img src="equity_curve.png" alt="Equity Curve">
                </div>
                
                <div class="image-container">
                    <h3>Drawdown</h3>
                    <img src="drawdown.png" alt="Drawdown">
                </div>
                
                <div class="image-container">
                    <h3>Returns Distribution</h3>
                    <img src="returns_distribution.png" alt="Returns Distribution">
                </div>
                
                <div class="image-container">
                    <h3>Rolling Sharpe Ratio</h3>
                    <img src="rolling_sharpe.png" alt="Rolling Sharpe Ratio">
                </div>
            </div>
        </body>
        </html>
        """)
    
    logger.info(f"Performance report generated in {output_dir}") 