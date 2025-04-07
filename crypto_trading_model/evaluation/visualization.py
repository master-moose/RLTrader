#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Configure logging
logger = logging.getLogger('crypto_trading_model.evaluation')

def plot_price_with_signals(prices: np.ndarray, signals: np.ndarray, 
                          dates: Optional[pd.DatetimeIndex] = None,
                          title: str = 'Price Chart with Trade Signals',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot price chart with buy/sell signals.
    
    Parameters:
    -----------
    prices : np.ndarray
        Array of prices
    signals : np.ndarray
        Array of signals (1 for buy, 0 for hold, -1 for sell)
    dates : pd.DatetimeIndex, optional
        Array of dates, by default None
    title : str, optional
        Plot title, by default 'Price Chart with Trade Signals'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if dates is None:
        dates = np.arange(len(prices))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, prices, label='Price')
    
    # Find buy and sell signals
    buy_indices = np.where(signals == 1)[0]
    sell_indices = np.where(signals == -1)[0]
    
    # Plot buy signals
    if len(buy_indices) > 0:
        ax.scatter(dates[buy_indices], prices[buy_indices], 
                 marker='^', color='green', s=100, label='Buy')
    
    # Plot sell signals
    if len(sell_indices) > 0:
        ax.scatter(dates[sell_indices], prices[sell_indices], 
                 marker='v', color='red', s=100, label='Sell')
    
    ax.set_title(title)
    ax.set_xlabel('Date' if isinstance(dates, pd.DatetimeIndex) else 'Time')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    
    if isinstance(dates, pd.DatetimeIndex):
        fig.autofmt_xdate()  # Rotate date labels
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_price_with_positions(prices: np.ndarray, positions: np.ndarray, 
                            dates: Optional[pd.DatetimeIndex] = None,
                            title: str = 'Price Chart with Positions',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot price chart with position overlay.
    
    Parameters:
    -----------
    prices : np.ndarray
        Array of prices
    positions : np.ndarray
        Array of positions (1 for long, 0 for neutral, -1 for short)
    dates : pd.DatetimeIndex, optional
        Array of dates, by default None
    title : str, optional
        Plot title, by default 'Price Chart with Positions'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if dates is None:
        dates = np.arange(len(prices))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price on top subplot
    ax1.plot(dates, prices, label='Price')
    ax1.set_title(title)
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot positions on bottom subplot
    ax2.bar(dates, positions, width=1.0, color=np.where(positions > 0, 'green', np.where(positions < 0, 'red', 'gray')))
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_ylabel('Position')
    ax2.set_xlabel('Date' if isinstance(dates, pd.DatetimeIndex) else 'Time')
    ax2.grid(True)
    
    # Add position labels
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
    
    if isinstance(dates, pd.DatetimeIndex):
        fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_equity_with_drawdown(equity_curve: np.ndarray, 
                            dates: Optional[pd.DatetimeIndex] = None,
                            title: str = 'Equity Curve with Drawdown',
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot equity curve with drawdown.
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Array of equity values
    dates : pd.DatetimeIndex, optional
        Array of dates, by default None
    title : str, optional
        Plot title, by default 'Equity Curve with Drawdown'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if dates is None:
        dates = np.arange(len(equity_curve))
    
    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot equity curve on top subplot
    ax1.plot(dates, equity_curve, label='Equity')
    ax1.set_title(title)
    ax1.set_ylabel('Equity')
    ax1.grid(True)
    
    # Plot drawdown on bottom subplot
    ax2.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(dates, drawdown, color='red')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date' if isinstance(dates, pd.DatetimeIndex) else 'Time')
    ax2.grid(True)
    
    # Find and mark the maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    ax2.scatter(dates[max_dd_idx], drawdown[max_dd_idx], color='darkred', s=100, zorder=5)
    ax2.annotate(f"Max DD: {drawdown[max_dd_idx]:.2%}", 
               (dates[max_dd_idx], drawdown[max_dd_idx]),
               xytext=(30, 30),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='black'))
    
    if isinstance(dates, pd.DatetimeIndex):
        fig.autofmt_xdate()  # Rotate date labels
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                        class_names: List[str] = ['Sell', 'Hold', 'Buy'],
                        title: str = 'Confusion Matrix',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix for trading signals.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (1 for buy, 0 for hold, -1 for sell)
    y_pred : np.ndarray
        Predicted labels
    class_names : List[str], optional
        Names of classes, by default ['Sell', 'Hold', 'Buy']
    title : str, optional
        Plot title, by default 'Confusion Matrix'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
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
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
              yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{title} (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Plot normalized values
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names,
              yticklabels=class_names, ax=ax2)
    ax2.set_title(f'{title} (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                 class_names: List[str] = ['Sell', 'Hold', 'Buy'],
                 title: str = 'ROC Curve',
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curve for multi-class classification.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (one-hot encoded or integer classes)
    y_score : np.ndarray
        Predicted probabilities
    class_names : List[str], optional
        Names of classes, by default ['Sell', 'Hold', 'Buy']
    title : str, optional
        Plot title, by default 'ROC Curve'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Convert to one-hot encoding if needed
    if len(y_true.shape) == 1:
        # Check if classes are -1, 0, 1
        if set(np.unique(y_true)).issubset({-1, 0, 1}):
            # Map to 0, 1, 2 for one-hot encoding
            y_true_mapped = np.array(y_true) + 1
        else:
            y_true_mapped = y_true
            
        # Convert to one-hot encoding
        n_classes = len(class_names)
        y_true_onehot = np.zeros((len(y_true_mapped), n_classes))
        for i, val in enumerate(y_true_mapped):
            y_true_onehot[i, int(val)] = 1
    else:
        y_true_onehot = y_true
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray, 
                              class_names: List[str] = ['Sell', 'Hold', 'Buy'],
                              title: str = 'Precision-Recall Curve',
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot precision-recall curve for multi-class classification.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (one-hot encoded or integer classes)
    y_score : np.ndarray
        Predicted probabilities
    class_names : List[str], optional
        Names of classes, by default ['Sell', 'Hold', 'Buy']
    title : str, optional
        Plot title, by default 'Precision-Recall Curve'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Convert to one-hot encoding if needed
    if len(y_true.shape) == 1:
        # Check if classes are -1, 0, 1
        if set(np.unique(y_true)).issubset({-1, 0, 1}):
            # Map to 0, 1, 2 for one-hot encoding
            y_true_mapped = np.array(y_true) + 1
        else:
            y_true_mapped = y_true
            
        # Convert to one-hot encoding
        n_classes = len(class_names)
        y_true_onehot = np.zeros((len(y_true_mapped), n_classes))
        for i, val in enumerate(y_true_mapped):
            y_true_onehot[i, int(val)] = 1
    else:
        y_true_onehot = y_true
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_interactive_equity_curve(equity_curve: np.ndarray, 
                                dates: Optional[pd.DatetimeIndex] = None,
                                title: str = 'Interactive Equity Curve',
                                save_path: Optional[str] = None) -> go.Figure:
    """
    Create an interactive equity curve plot using Plotly.
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        Array of equity values
    dates : pd.DatetimeIndex, optional
        Array of dates, by default None
    title : str, optional
        Plot title, by default 'Interactive Equity Curve'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if dates is None:
        dates = np.arange(len(equity_curve))
    
    # Calculate drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    
    # Calculate returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = np.insert(returns, 0, 0)  # Add 0 for the first period
    
    # Create figure
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                      subplot_titles=('Equity Curve', 'Drawdown', 'Daily Returns'),
                      row_heights=[0.5, 0.25, 0.25],
                      vertical_spacing=0.05)
    
    # Add equity curve trace
    fig.add_trace(
        go.Scatter(x=dates, y=equity_curve, mode='lines', name='Equity',
                 line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add drawdown trace
    fig.add_trace(
        go.Scatter(x=dates, y=drawdown, mode='lines', name='Drawdown',
                 line=dict(color='red'), fill='tozeroy'),
        row=2, col=1
    )
    
    # Add returns trace
    fig.add_trace(
        go.Bar(x=dates, y=returns, name='Returns',
             marker=dict(color=np.where(returns >= 0, 'green', 'red'))),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text='Equity', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown', row=2, col=1, tickformat='.0%')
    fig.update_yaxes(title_text='Returns', row=3, col=1, tickformat='.0%')
    
    # Update x-axis label
    fig.update_xaxes(title_text='Date' if isinstance(dates, pd.DatetimeIndex) else 'Time',
                   row=3, col=1)
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

def plot_interactive_trades(prices: np.ndarray, trades_df: pd.DataFrame,
                         dates: Optional[pd.DatetimeIndex] = None,
                         title: str = 'Interactive Trade Analysis',
                         save_path: Optional[str] = None) -> go.Figure:
    """
    Create an interactive plot of trades using Plotly.
    
    Parameters:
    -----------
    prices : np.ndarray
        Array of prices
    trades_df : pd.DataFrame
        DataFrame of trades with columns: entry_idx, exit_idx, entry_price, exit_price, position, pnl
    dates : pd.DatetimeIndex, optional
        Array of dates, by default None
    title : str, optional
        Plot title, by default 'Interactive Trade Analysis'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if dates is None:
        dates = np.arange(len(prices))
    
    # Create figure
    fig = go.Figure()
    
    # Add price trace
    fig.add_trace(
        go.Scatter(x=dates, y=prices, mode='lines', name='Price',
                 line=dict(color='black', width=1))
    )
    
    # Add trades
    for _, trade in trades_df.iterrows():
        # Get trade information
        entry_idx = int(trade['entry_idx'])
        exit_idx = int(trade['exit_idx'])
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position = trade['position']
        pnl = trade['pnl']
        
        # Determine trade color
        color = 'green' if pnl > 0 else 'red'
        
        # Add trade entry point
        fig.add_trace(
            go.Scatter(
                x=[dates[entry_idx]],
                y=[entry_price],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=10,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                name=f"Entry: {position:.0f}",
                hoverinfo='text',
                hovertext=f"Entry: {dates[entry_idx]}<br>Price: {entry_price:.2f}<br>Position: {'Long' if position > 0 else 'Short'}",
                showlegend=False
            )
        )
        
        # Add trade exit point
        fig.add_trace(
            go.Scatter(
                x=[dates[exit_idx]],
                y=[exit_price],
                mode='markers',
                marker=dict(
                    symbol='square',
                    size=10,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                name=f"Exit: {pnl:.2%}",
                hoverinfo='text',
                hovertext=f"Exit: {dates[exit_idx]}<br>Price: {exit_price:.2f}<br>PnL: {pnl:.2%}",
                showlegend=False
            )
        )
        
        # Add line connecting entry and exit
        fig.add_trace(
            go.Scatter(
                x=[dates[entry_idx], dates[exit_idx]],
                y=[entry_price, exit_price],
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=f"Trade: {pnl:.2%}",
                hoverinfo='text',
                hovertext=f"Trade<br>Entry: {dates[entry_idx]}<br>Exit: {dates[exit_idx]}<br>PnL: {pnl:.2%}",
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date' if isinstance(dates, pd.DatetimeIndex) else 'Time',
        yaxis_title='Price',
        hovermode='closest',
        height=600,
        width=1000
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig

def plot_comparative_equity_curves(equity_curves: Dict[str, np.ndarray],
                                dates: Optional[pd.DatetimeIndex] = None,
                                title: str = 'Comparative Equity Curves',
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple equity curves for comparison.
    
    Parameters:
    -----------
    equity_curves : Dict[str, np.ndarray]
        Dictionary of equity curves, where keys are model names
    dates : pd.DatetimeIndex, optional
        Array of dates, by default None
    title : str, optional
        Plot title, by default 'Comparative Equity Curves'
    save_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    if dates is None:
        # Use the length of the first equity curve
        first_curve = next(iter(equity_curves.values()))
        dates = np.arange(len(first_curve))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each equity curve
    for model_name, equity_curve in equity_curves.items():
        ax.plot(dates, equity_curve, label=model_name)
    
    ax.set_title(title)
    ax.set_xlabel('Date' if isinstance(dates, pd.DatetimeIndex) else 'Time')
    ax.set_ylabel('Equity')
    ax.grid(True)
    ax.legend()
    
    if isinstance(dates, pd.DatetimeIndex):
        fig.autofmt_xdate()  # Rotate date labels
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def create_dashboard(metrics: Dict[str, Any], output_dir: str) -> None:
    """
    Create an interactive dashboard of trading performance.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics
    output_dir : str
        Directory to save the dashboard
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key metrics
    equity_curve = metrics['equity_curve']
    returns = metrics['returns']
    trades = metrics.get('trades', pd.DataFrame())
    
    # Generate dates (placeholder)
    dates = pd.date_range(start='2023-01-01', periods=len(equity_curve), freq='D')
    
    # Create interactive plots
    interactive_equity = plot_interactive_equity_curve(
        equity_curve=equity_curve,
        dates=dates,
        title='Equity Curve Analysis',
        save_path=os.path.join(output_dir, 'interactive_equity.html')
    )
    
    if not trades.empty and 'entry_idx' in trades.columns:
        interactive_trades = plot_interactive_trades(
            prices=metrics.get('prices', np.zeros(len(equity_curve))),
            trades_df=trades,
            dates=dates,
            title='Trade Analysis',
            save_path=os.path.join(output_dir, 'interactive_trades.html')
        )
    
    # Create dashboard HTML
    with open(os.path.join(output_dir, 'dashboard.html'), 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .metrics {{ margin: 20px 0; display: flex; flex-wrap: wrap; }}
                .metric-card {{ 
                    background-color: #f8f9fa; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 10px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    width: 200px;
                }}
                .metric-title {{ font-size: 14px; color: #666; margin-bottom: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-positive {{ color: green; }}
                .metric-negative {{ color: red; }}
                .metric-neutral {{ color: #333; }}
                .iframe-container {{ margin: 20px 0; }}
                iframe {{ border: none; width: 100%; height: 600px; }}
            </style>
        </head>
        <body>
            <h1>Trading Performance Dashboard</h1>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Total Return</div>
                    <div class="metric-value {{'metric-positive' if metrics['total_return'] > 0 else 'metric-negative'}}">
                        {metrics['total_return']:.2%}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Sharpe Ratio</div>
                    <div class="metric-value {{'metric-positive' if metrics['sharpe_ratio'] > 1 else 'metric-neutral'}}">
                        {metrics['sharpe_ratio']:.2f}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Max Drawdown</div>
                    <div class="metric-value metric-negative">
                        {metrics['max_drawdown']:.2%}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Win Rate</div>
                    <div class="metric-value {{'metric-positive' if metrics['win_rate'] > 0.5 else 'metric-neutral'}}">
                        {metrics['win_rate']:.2%}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Profit Factor</div>
                    <div class="metric-value {{'metric-positive' if metrics['profit_factor'] > 1 else 'metric-negative'}}">
                        {metrics['profit_factor']:.2f}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Number of Trades</div>
                    <div class="metric-value metric-neutral">
                        {metrics['num_trades']}
                    </div>
                </div>
            </div>
            
            <div class="iframe-container">
                <h2>Equity Curve Analysis</h2>
                <iframe src="interactive_equity.html"></iframe>
            </div>
            
            {'<div class="iframe-container"><h2>Trade Analysis</h2><iframe src="interactive_trades.html"></iframe></div>' 
             if not trades.empty and 'entry_idx' in trades.columns else ''}
            
            <div style="margin-top: 30px; text-align: center; color: #666;">
                Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </body>
        </html>
        """)
    
    logger.info(f"Interactive dashboard generated in {output_dir}") 