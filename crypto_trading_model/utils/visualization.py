"""
Visualization utilities for the crypto trading model.

This module provides functions for creating plots and visualizations
of market data, trading signals, and model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization')

class MarketVisualizer:
    """
    Class for visualizing market data and trading signals.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn-darkgrid'):
        """
        Initialize the market visualizer.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        style : str
            Matplotlib style to use
        """
        self.figsize = figsize
        plt.style.use(style)
    
    def plot_price_series(
        self,
        data: pd.DataFrame,
        title: str = 'Price Chart',
        volume: bool = True,
        ma_periods: List[int] = None,
        signals: pd.DataFrame = None
    ) -> plt.Figure:
        """
        Plot price series with optional volume, moving averages, and trading signals.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data
        title : str
            Chart title
        volume : bool
            Whether to include volume subplot
        ma_periods : List[int]
            List of periods for moving averages
        signals : pd.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if ma_periods is None:
            ma_periods = [20, 50, 200]
        
        # Create figure with subplots
        n_subplots = 1 + int(volume)
        fig, axes = plt.subplots(n_subplots, 1, figsize=self.figsize, 
                                gridspec_kw={'height_ratios': [3, 1] if volume else [1]})
        
        if not volume:
            axes = [axes]  # Make it a list for consistent indexing
        
        # Plot price on the main subplot
        price_ax = axes[0]
        price_ax.plot(data.index, data['close'], label='Close')
        
        # Add moving averages
        for period in ma_periods:
            if len(data) >= period:
                ma = data['close'].rolling(window=period).mean()
                price_ax.plot(data.index, ma, label=f'MA({period})')
        
        # Add trading signals if provided
        if signals is not None:
            # Buy signals
            buy_signals = signals[signals['signal'] > 0]
            if not buy_signals.empty:
                price_ax.scatter(buy_signals.index, data.loc[buy_signals.index, 'close'],
                               marker='^', color='green', s=100, label='Buy')
            
            # Sell signals
            sell_signals = signals[signals['signal'] < 0]
            if not sell_signals.empty:
                price_ax.scatter(sell_signals.index, data.loc[sell_signals.index, 'close'],
                               marker='v', color='red', s=100, label='Sell')
        
        price_ax.set_title(title)
        price_ax.set_ylabel('Price')
        price_ax.legend()
        price_ax.grid(True)
        
        # Format x-axis
        price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Plot volume if requested
        if volume:
            volume_ax = axes[1]
            volume_ax.bar(data.index, data['volume'], color='darkgray', alpha=0.8)
            volume_ax.set_ylabel('Volume')
            volume_ax.grid(True)
            volume_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        return fig
    
    def plot_indicators(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, List[str]],
        title: str = 'Technical Indicators',
        subplots: bool = True
    ) -> plt.Figure:
        """
        Plot technical indicators.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with indicator data
        indicators : Dict[str, List[str]]
            Dictionary mapping indicator groups to column names
        title : str
            Chart title
        subplots : bool
            Whether to use separate subplots for each indicator group
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        n_groups = len(indicators)
        
        if subplots:
            # Create separate subplot for each indicator group
            fig, axes = plt.subplots(n_groups, 1, figsize=self.figsize, sharex=True)
            
            if n_groups == 1:
                axes = [axes]  # Make it a list for consistent indexing
            
            for (group_name, columns), ax in zip(indicators.items(), axes):
                for col in columns:
                    if col in data.columns:
                        ax.plot(data.index, data[col], label=col)
                
                ax.set_title(group_name)
                ax.legend()
                ax.grid(True)
            
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
        else:
            # Plot all indicators on the same axes
            fig, ax = plt.subplots(figsize=self.figsize)
            
            for group_name, columns in indicators.items():
                for col in columns:
                    if col in data.columns:
                        ax.plot(data.index, data[col], label=col)
            
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_candlestick(
        self,
        data: pd.DataFrame,
        title: str = 'Candlestick Chart',
        volume: bool = True,
        support_resistance: List[float] = None,
        use_plotly: bool = False
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot candlestick chart with optional volume and support/resistance levels.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data
        title : str
            Chart title
        volume : bool
            Whether to include volume subplot
        support_resistance : List[float]
            List of support/resistance price levels
        use_plotly : bool
            Whether to use Plotly for interactive charts
            
        Returns:
        --------
        Union[plt.Figure, go.Figure]
            Matplotlib or Plotly figure object
        """
        if use_plotly:
            # Create Plotly candlestick chart
            if volume:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02,
                    row_heights=[0.8, 0.2]
                )
            else:
                fig = go.Figure()
            
            # Add candlestick trace
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add volume trace if requested
            if volume:
                colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in data.iterrows()]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['volume'],
                        name='Volume',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
            
            # Add support/resistance levels if provided
            if support_resistance and len(support_resistance) > 0:
                for level in support_resistance:
                    fig.add_shape(
                        type='line',
                        x0=data.index[0],
                        x1=data.index[-1],
                        y0=level,
                        y1=level,
                        line=dict(color='blue', width=2, dash='dash'),
                        row=1, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        else:
            # Use Matplotlib for static charts
            n_subplots = 1 + int(volume)
            fig, axes = plt.subplots(n_subplots, 1, figsize=self.figsize, 
                                    gridspec_kw={'height_ratios': [3, 1] if volume else [1]})
            
            if not volume:
                axes = [axes]  # Make it a list for consistent indexing
            
            # Plot candlesticks
            price_ax = axes[0]
            
            # Plot candlesticks using colored bars
            up = data[data['close'] >= data['open']]
            down = data[data['close'] < data['open']]
            
            # Plot up candles
            if not up.empty:
                price_ax.bar(
                    up.index, up['close'] - up['open'],
                    bottom=up['open'], width=0.6,
                    color='green', alpha=0.5
                )
                price_ax.bar(
                    up.index, up['high'] - up['close'],
                    bottom=up['close'], width=0.1,
                    color='green', alpha=0.5
                )
                price_ax.bar(
                    up.index, up['open'] - up['low'],
                    bottom=up['low'], width=0.1,
                    color='green', alpha=0.5
                )
            
            # Plot down candles
            if not down.empty:
                price_ax.bar(
                    down.index, down['close'] - down['open'],
                    bottom=down['open'], width=0.6,
                    color='red', alpha=0.5
                )
                price_ax.bar(
                    down.index, down['high'] - down['open'],
                    bottom=down['open'], width=0.1,
                    color='red', alpha=0.5
                )
                price_ax.bar(
                    down.index, down['close'] - down['low'],
                    bottom=down['low'], width=0.1,
                    color='red', alpha=0.5
                )
            
            # Add support/resistance levels if provided
            if support_resistance and len(support_resistance) > 0:
                for level in support_resistance:
                    price_ax.axhline(y=level, color='blue', linestyle='--', alpha=0.7)
            
            price_ax.set_title(title)
            price_ax.set_ylabel('Price')
            price_ax.grid(True)
            
            # Format x-axis
            price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Plot volume if requested
            if volume:
                volume_ax = axes[1]
                volume_colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in data.iterrows()]
                volume_ax.bar(data.index, data['volume'], color=volume_colors, alpha=0.5)
                volume_ax.set_ylabel('Volume')
                volume_ax.grid(True)
                volume_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            plt.tight_layout()
            return fig

class ModelVisualizer:
    """
    Class for visualizing model training results and predictions.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the model visualizer.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = None,
        title: str = 'Training History'
    ) -> plt.Figure:
        """
        Plot training history metrics.
        
        Parameters:
        -----------
        history : Dict[str, List[float]]
            Dictionary of training metrics
        metrics : List[str]
            List of metrics to plot
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if metrics is None:
            metrics = ['loss']
            if 'val_loss' in history:
                metrics.append('val_loss')
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for metric in metrics:
            if metric in history:
                ax.plot(history[metric], label=metric)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_predictions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: pd.DatetimeIndex,
        title: str = 'Model Predictions vs Actual'
    ) -> plt.Figure:
        """
        Plot model predictions against actual values.
        
        Parameters:
        -----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
        dates : pd.DatetimeIndex
            Dates for x-axis
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(dates, actual, label='Actual', marker='o', markersize=2)
        ax.plot(dates, predicted, label='Predicted', marker='x', markersize=2)
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: pd.Series,
        title: str = 'Feature Importance'
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        feature_importance : pd.Series
            Series with feature importance values
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Sort by importance
        sorted_importance = feature_importance.sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sorted_importance.plot(kind='bar', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Importance')
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        title: str = 'Confusion Matrix'
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        class_names : List[str]
            Names of classes
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if class_names is None:
            class_names = ['Negative', 'Neutral', 'Positive']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(
        self,
        fpr: Dict[int, np.ndarray],
        tpr: Dict[int, np.ndarray],
        roc_auc: Dict[int, float],
        title: str = 'ROC Curve'
    ) -> plt.Figure:
        """
        Plot ROC curve for multi-class classification.
        
        Parameters:
        -----------
        fpr : Dict[int, np.ndarray]
            False positive rates for each class
        tpr : Dict[int, np.ndarray]
            True positive rates for each class
        roc_auc : Dict[int, float]
            ROC AUC for each class
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i in sorted(fpr.keys()):
            ax.plot(
                fpr[i], tpr[i],
                lw=2,
                label=f'Class {i} (AUC = {roc_auc[i]:.2f})'
            )
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)
        
        return fig

# Function to visualize multi-timeframe data
def plot_multi_timeframe_data(
    data_dict: Dict[str, pd.DataFrame],
    title: str = 'Multi-Timeframe Price Comparison',
    price_col: str = 'close'
) -> plt.Figure:
    """
    Plot price data from multiple timeframes for comparison.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping timeframe names to DataFrames
    title : str
        Plot title
    price_col : str
        Column to plot from each DataFrame
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for timeframe, df in data_dict.items():
        # Resample to daily for consistent comparison if needed
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            # Normalize to starting at 1 for comparison
            normalized = df[price_col] / df[price_col].iloc[0]
            normalized.plot(ax=ax, label=f'{timeframe}')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Normalized {price_col}')
    ax.legend()
    ax.grid(True)
    
    return fig

if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, len(dates)),
        'high': np.random.normal(105, 5, len(dates)),
        'low': np.random.normal(95, 5, len(dates)),
        'close': np.random.normal(102, 5, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    
    # Add some technical indicators
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['rsi_14'] = np.random.normal(50, 10, len(dates))
    
    # Create sample trade signals
    signals = pd.DataFrame({
        'signal': [0] * len(dates)
    }, index=dates)
    signals.loc[dates[25], 'signal'] = 1  # Buy signal
    signals.loc[dates[75], 'signal'] = -1  # Sell signal
    
    # Initialize visualizer
    market_viz = MarketVisualizer()
    
    # Plot price chart
    fig = market_viz.plot_price_series(
        data=data,
        title='Sample Price Chart',
        volume=True,
        ma_periods=[20, 50],
        signals=signals
    )
    
    plt.tight_layout()
    plt.show() 