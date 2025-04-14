#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import argparse
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf
import mplfinance as mpf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_evaluation.log')
    ]
)
logger = logging.getLogger('synthetic_data_evaluation')

def setup_output_dir(output_dir):
    """Create output directory for evaluation results."""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory: {str(e)}")
        sys.exit(1)

def load_data(data_path, timeframe, sample_size=None):
    """Load data from HDF5 file."""
    try:
        store = pd.HDFStore(data_path, mode='r')
        
        # Check if the timeframe exists
        if f'/{timeframe}' not in store:
            available_timeframes = [k[1:] for k in store.keys()]
            logger.error(f"Timeframe {timeframe} not found in {data_path}. "
                         f"Available timeframes: {available_timeframes}")
            store.close()
            sys.exit(1)
        
        # Load the data
        df = store.get(f'/{timeframe}')
        store.close()
        
        # Apply sampling if specified
        if sample_size is not None and sample_size < len(df):
            logger.info(f"Sampling {sample_size} rows from {len(df)} total rows")
            df = df.sample(n=sample_size, random_state=42)
        
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex. Converting...")
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                df.index = pd.date_range(
                    start='2010-01-01', 
                    periods=len(df),
                    freq='15min' if timeframe == '15m' else '4h' if timeframe == '4h' else '1D'
                )
        
        logger.info(f"Loaded {len(df)} rows of {timeframe} data from {data_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        sys.exit(1)

def plot_price_series(df, output_dir):
    """Plot price series over time."""
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, df['close'], label='Close Price', linewidth=1)
    
    # Optional: Plot moving averages if available
    if 'sma_25' in df.columns:
        plt.plot(df.index, df['sma_25'], label='25-period MA', linewidth=1, alpha=0.7)
    if 'ema_21' in df.columns:
        plt.plot(df.index, df['ema_21'], label='21-period EMA', linewidth=1, alpha=0.7)
    
    # Add market regime coloring if available
    if 'market_regime' in df.columns:
        unique_regimes = df['market_regime'].unique()
        for regime in unique_regimes:
            regime_df = df[df['market_regime'] == regime]
            plt.scatter(
                regime_df.index, 
                regime_df['close'], 
                s=5, 
                alpha=0.5, 
                label=f'Regime: {regime}'
            )
    
    plt.title('Synthetic Price Series')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_series.png'), dpi=300)
    plt.close()
    logger.info("Generated price series plot")

def plot_volume_series(df, output_dir):
    """Plot volume over time with price overlay."""
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot price on primary y-axis
    ax1.plot(df.index, df['close'], color='blue', linewidth=1, label='Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create secondary y-axis for volume
    ax2 = ax1.twinx()
    ax2.bar(df.index, df['volume'], color='gray', alpha=0.3, label='Volume')
    ax2.set_ylabel('Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Title and grid
    plt.title('Price and Volume Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_series.png'), dpi=300)
    plt.close()
    logger.info("Generated volume series plot")

def plot_candlestick(df, output_dir, segment_size=200):
    """Plot candlestick chart for a segment of the data."""
    # Make a copy of the data for plotting
    plot_df = df.copy()
    
    # Ensure plot_df has OHLCV columns
    if not all(col in plot_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        logger.error("DataFrame does not have required OHLCV columns for candlestick plot")
        return
    
    # Select a segment for plotting
    if len(plot_df) > segment_size:
        # Take a segment from the middle for more mature data
        start_idx = len(plot_df) // 2
        plot_df = plot_df.iloc[start_idx:start_idx+segment_size]
    
    # Create a new figure with custom style
    mc = mpf.make_marketcolors(
        up='green', down='red',
        wick={'up':'green', 'down':'red'},
        volume='blue',
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', facecolor='white')
    
    # Add moving averages if available
    if all(col in plot_df.columns for col in ['sma_7', 'sma_25']):
        apds = [
            mpf.make_addplot(plot_df['sma_7'], color='blue', width=0.7),
            mpf.make_addplot(plot_df['sma_25'], color='purple', width=1.5)
        ]
        # Save the figure
        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            volume=True,
            style=s,
            addplot=apds,
            returnfig=True,
            title='Candlestick Chart (Segment)',
            figratio=(16, 9),
            figscale=1.2
        )
    else:
        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            volume=True,
            style=s,
            returnfig=True,
            title='Candlestick Chart (Segment)',
            figratio=(16, 9),
            figscale=1.2
        )
    
    # Save the figure
    fig.savefig(os.path.join(output_dir, 'candlestick_chart.png'), dpi=300)
    plt.close(fig)
    logger.info(f"Generated candlestick chart with {len(plot_df)} periods")

def calculate_descriptive_stats(df, output_dir):
    """Calculate descriptive statistics for key columns."""
    # Calculate returns if not already present
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Identify key columns
    key_columns = [
        'close', 'volume', 'return', 'log_return', 
        'high_low_range', 'body_size'
    ]
    key_columns = [col for col in key_columns if col in df.columns]
    
    # Calculate descriptive statistics
    stats_df = df[key_columns].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    
    # Add additional statistics
    for col in key_columns:
        # Add skewness and kurtosis
        stats_df.loc['skewness', col] = df[col].skew()
        stats_df.loc['kurtosis', col] = df[col].kurtosis()
        
        # Add autocorrelation (lag 1)
        stats_df.loc['autocorr_lag1', col] = df[col].autocorr(lag=1)
    
    # Save to file
    stats_output = os.path.join(output_dir, 'summary_stats.txt')
    with open(stats_output, 'w') as f:
        f.write("Descriptive Statistics for Synthetic Data\n")
        f.write("========================================\n\n")
        f.write(stats_df.to_string())
        
        # Add additional market statistics if we have returns
        if 'return' in key_columns:
            f.write("\n\nMarket Statistics\n")
            f.write("================\n\n")
            
            # Perform normality test on returns
            jb_stat, jb_pval = stats.jarque_bera(df['return'].dropna())
            f.write(f"Jarque-Bera test for normality of returns:\n")
            f.write(f"  Statistic: {jb_stat:.4f}\n")
            f.write(f"  p-value: {jb_pval:.6f}\n")
            f.write(f"  Conclusion: {'Non-normal' if jb_pval < 0.05 else 'Cannot reject normality'}\n\n")
            
            # Calculate volatility clustering indicators
            abs_returns = df['return'].abs()
            abs_returns_autocorr = abs_returns.autocorr(lag=1)
            f.write(f"Volatility clustering (autocorrelation of absolute returns):\n")
            f.write(f"  Lag-1 autocorrelation: {abs_returns_autocorr:.4f}\n")
            f.write(f"  Conclusion: {'Present' if abs_returns_autocorr > 0.1 else 'Weak or absent'}\n")
    
    logger.info(f"Saved descriptive statistics to {stats_output}")
    return stats_df

def analyze_return_distribution(df, output_dir):
    """Analyze and plot the distribution of returns."""
    # Ensure returns are calculated
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Drop NaN values
    returns = df['return'].dropna()
    log_returns = df['log_return'].dropna()
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot histogram of returns with normal distribution overlay
    sns.histplot(returns, kde=True, stat='density', ax=ax1)
    
    # Add normal distribution overlay
    mu, std = returns.mean(), returns.std()
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    normal_dist = stats.norm.pdf(x, mu, std)
    ax1.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
    
    ax1.set_title('Distribution of Returns')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.legend()
    
    # Q-Q plot of returns against normal distribution
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Returns vs. Normal Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'return_distribution.png'), dpi=300)
    plt.close()
    
    # Log-scale plot to see fat tails more clearly
    plt.figure(figsize=(12, 8))
    sns.histplot(returns, kde=False, stat='density', log_scale=(False, True))
    plt.title('Return Distribution (Log Scale)')
    plt.xlabel('Return')
    plt.ylabel('Log Density')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'return_distribution_log.png'), dpi=300)
    plt.close()
    
    logger.info("Generated return distribution plots")

def analyze_autocorrelation(df, output_dir):
    """Analyze and plot autocorrelation of returns and volatility."""
    # Ensure returns are calculated
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()
    
    # Drop NaN values
    returns = df['return'].dropna()
    abs_returns = returns.abs()  # Proxy for volatility
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot ACF of returns
    plot_acf(returns, ax=ax1, lags=50, alpha=0.05)
    ax1.set_title('Autocorrelation of Returns')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    
    # Plot ACF of absolute returns (volatility proxy)
    plot_acf(abs_returns, ax=ax2, lags=50, alpha=0.05)
    ax2.set_title('Autocorrelation of Absolute Returns (Volatility)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorrelation.png'), dpi=300)
    plt.close()
    
    logger.info("Generated autocorrelation plots")

def analyze_volatility(df, output_dir, window=20):
    """Analyze and plot volatility over time."""
    # Calculate rolling volatility if not already present
    if 'true_volatility' in df.columns:
        # Use the true volatility from the simulation if available
        volatility = df['true_volatility']
        title = 'True Volatility Over Time'
    else:
        # Calculate rolling volatility from returns
        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change()
        
        # Calculate annualized volatility (approximate)
        periods_per_year = 35040 if '15m' in df.index.freq.name else 2190 if '4h' in df.index.freq.name else 365
        volatility = df['return'].rolling(window=window).std() * np.sqrt(periods_per_year)
        title = f'{window}-Period Rolling Volatility (Annualized)'
    
    plt.figure(figsize=(15, 8))
    plt.plot(df.index, volatility, linewidth=1)
    
    # Add regime information if available
    if 'market_regime' in df.columns:
        # Mark volatility expansion periods
        vol_exp_df = df[df['market_regime'] == 'volatility_expansion']
        plt.scatter(
            vol_exp_df.index, 
            volatility.loc[vol_exp_df.index], 
            color='red', 
            s=10, 
            alpha=0.6,
            label='Volatility Expansion Regime'
        )
        
        # Mark volatility contraction periods
        vol_con_df = df[df['market_regime'] == 'volatility_contraction']
        plt.scatter(
            vol_con_df.index, 
            volatility.loc[vol_con_df.index], 
            color='green', 
            s=10, 
            alpha=0.6,
            label='Volatility Contraction Regime'
        )
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True, alpha=0.3)
    
    if 'market_regime' in df.columns:
        plt.legend()
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volatility_over_time.png'), dpi=300)
    plt.close()
    logger.info("Generated volatility plot")

def analyze_volume_price_relationship(df, output_dir):
    """Analyze and plot the relationship between volume and price changes."""
    # Ensure returns are calculated
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()
    
    # Calculate absolute returns
    df['abs_return'] = df['return'].abs()
    
    # Scatter plot of volume vs absolute returns
    plt.figure(figsize=(12, 8))
    plt.scatter(df['abs_return'], df['volume'], alpha=0.3, s=5)
    
    # Add a trend line
    z = np.polyfit(df['abs_return'].dropna(), df['volume'].dropna(), 1)
    p = np.poly1d(z)
    plt.plot(sorted(df['abs_return'].dropna()), 
             p(sorted(df['abs_return'].dropna())), 
             "r--", linewidth=2)
    
    # Calculate and display correlation
    corr = df['abs_return'].corr(df['volume'])
    plt.title(f'Volume vs. Absolute Returns (Correlation: {corr:.4f})')
    plt.xlabel('Absolute Return')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_vs_return.png'), dpi=300)
    plt.close()
    
    # Log correlation information
    with open(os.path.join(output_dir, 'summary_stats.txt'), 'a') as f:
        f.write("\n\nVolume-Price Relationships\n")
        f.write("=========================\n\n")
        f.write(f"Correlation between volume and absolute returns: {corr:.4f}\n")
        
        # Calculate correlation between volume and volatility if available
        if 'true_volatility' in df.columns:
            vol_corr = df['volume'].corr(df['true_volatility'])
            f.write(f"Correlation between volume and volatility: {vol_corr:.4f}\n")
        
        # Create a correlation matrix for key variables
        key_vars = ['volume', 'abs_return', 'close', 'high_low_range']
        key_vars = [var for var in key_vars if var in df.columns]
        corr_matrix = df[key_vars].corr()
        
        f.write("\nCorrelation Matrix:\n")
        f.write(corr_matrix.to_string())
    
    logger.info("Generated volume-price relationship analysis")

def analyze_indicators(df, output_dir):
    """Analyze and plot key technical indicators."""
    # Check which indicators are available
    indicators = {
        'Moving Averages': ['sma_7', 'sma_25', 'ema_9', 'ema_21'],
        'Oscillators': ['rsi_14', 'stoch_k', 'stoch_d'],
        'MACD': ['macd', 'macd_signal', 'macd_hist'],
        'Bollinger Bands': ['bb_upper', 'bb_middle', 'bb_lower'],
        'Volume Indicators': ['obv', 'cmf_20', 'volume_ratio']
    }
    
    available_groups = []
    for group, inds in indicators.items():
        if any(ind in df.columns for ind in inds):
            available_groups.append(group)
    
    if not available_groups:
        logger.warning("No technical indicators found in the data")
        return
    
    # Select a segment for plotting (using the latest data)
    plot_df = df.iloc[-500:] if len(df) > 500 else df
    
    # Create one plot per indicator group
    for group in available_groups:
        inds = [ind for ind in indicators[group] if ind in df.columns]
        if not inds:
            continue
        
        # Special handling for Bollinger Bands (need price as well)
        if group == 'Bollinger Bands':
            plt.figure(figsize=(15, 8))
            plt.plot(plot_df.index, plot_df['close'], label='Close', linewidth=1)
            for ind in inds:
                plt.plot(plot_df.index, plot_df[ind], label=ind, linewidth=1, alpha=0.7)
        
        # Special handling for MACD (histogram)
        elif group == 'MACD':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Price on the first subplot
            ax1.plot(plot_df.index, plot_df['close'], label='Close', linewidth=1)
            ax1.set_title('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MACD on the second subplot
            if 'macd' in inds and 'macd_signal' in inds:
                ax2.plot(plot_df.index, plot_df['macd'], label='MACD', linewidth=1)
                ax2.plot(plot_df.index, plot_df['macd_signal'], label='Signal', linewidth=1)
            
            if 'macd_hist' in inds:
                # MACD histogram as bar chart
                ax2.bar(plot_df.index, plot_df['macd_hist'], alpha=0.3, width=2, label='Histogram')
            
            ax2.set_title('MACD')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Skip the rest of the loop for MACD since we've already created the plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'indicator_{group.lower().replace(" ", "_")}.png'), dpi=300)
            plt.close()
            continue
        
        # Regular indicators
        else:
            plt.figure(figsize=(15, 8))
            for ind in inds:
                plt.plot(plot_df.index, plot_df[ind], label=ind, linewidth=1)
        
        plt.title(f'{group} - Last {len(plot_df)} Periods')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'indicator_{group.lower().replace(" ", "_")}.png'), dpi=300)
        plt.close()
    
    logger.info(f"Generated plots for {len(available_groups)} indicator groups")

def main():
    """Main entry point for synthetic data evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize synthetic data for cryptocurrency trading model"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to the HDF5 data file"
    )
    parser.add_argument(
        "--timeframe", 
        type=str, 
        default="15m",
        choices=["15m", "4h", "1d"],
        help="Timeframe to analyze (default: 15m)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output/data_evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=None,
        help="Number of samples to use (random sample if less than total)"
    )
    parser.add_argument(
        "--candlestick_segment", 
        type=int, 
        default=200,
        help="Number of periods for candlestick chart"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    setup_output_dir(args.output_dir)
    
    # Load data
    df = load_data(args.data_path, args.timeframe, args.sample_size)
    
    # Generate visual plots
    plot_price_series(df, args.output_dir)
    plot_volume_series(df, args.output_dir)
    plot_candlestick(df, args.output_dir, args.candlestick_segment)
    
    # Statistical analysis
    calculate_descriptive_stats(df, args.output_dir)
    analyze_return_distribution(df, args.output_dir)
    analyze_autocorrelation(df, args.output_dir)
    analyze_volatility(df, args.output_dir)
    analyze_volume_price_relationship(df, args.output_dir)
    analyze_indicators(df, args.output_dir)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 