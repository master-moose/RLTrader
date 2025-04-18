#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the RL agent.

This module contains various utility functions for resource monitoring,
data visualization, file management, and other helper functions.
"""

import gc
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns
from stable_baselines3.common.logger import configure

# Try to import GPU monitoring tools
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from gputil import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Setup logger
logger = logging.getLogger("rl_agent")

# Define a small threshold for floating point comparisons
ZERO_THRESHOLD = 1e-9  # noqa E221


def setup_logger(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    console_level: int = logging.INFO,
    log_filename: str = "rl_agent.log"
) -> logging.Logger:
    """
    Setup logger for the RL agent.

    Args:
        log_dir: Directory to save logs
        log_level: Logging level for file handler
        console_level: Logging level for console handler
        log_filename: Name of the log file

    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    ensure_dir_exists(log_dir)

    # Configure logger
    logger = logging.getLogger("rl_agent")
    # Set logger level based on the file log level argument
    logger.setLevel(log_level)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create file handler
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger setup complete. Log file: {log_path}")

    return logger


def setup_sb3_logger(
    log_dir: str = "./logs",
    format_strings: Optional[List[str]] = None
) -> None:
    """
    Setup Stable-Baselines3 logger.

    Args:
        log_dir: Directory to save logs
        format_strings: List of format strings for the logger
    """
    if format_strings is None:
        format_strings = ["stdout", "csv", "tensorboard"]

    # Create log directory
    ensure_dir_exists(log_dir)

    # Configure SB3 logger
    sb3_logger = configure(
        folder=log_dir,
        format_strings=format_strings
    )

    return sb3_logger


def ensure_dir_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def check_resources(
    logger: Optional[logging.Logger] = None,
    warning_threshold: float = 0.9,
    garbage_collect: bool = True
) -> Dict[str, Any]:
    """
    Check system resources (CPU, memory, GPU).

    Args:
        logger: Logger to log resource information
        warning_threshold: Memory usage percentage to trigger warning
        garbage_collect: Whether to run garbage collection on high memory usage

    Returns:
        Dictionary with resource information
    """
    resources = {}

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    resources["cpu_percent"] = cpu_percent

    # Memory usage
    memory = psutil.virtual_memory()
    resources["memory_total"] = memory.total / (1024 ** 3)  # GB
    resources["memory_used"] = memory.used / (1024 ** 3)    # GB
    resources["memory_percent"] = memory.percent

    # GPU usage if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        resources["gpu_count"] = torch.cuda.device_count()
        resources["gpu_memory_allocated"] = (
            torch.cuda.memory_allocated(0) / (1024 ** 2)  # MB
        )
        resources["gpu_memory_reserved"] = (
            torch.cuda.memory_reserved(0) / (1024 ** 2)    # MB
        )

        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get the first GPU
                    resources["gpu_memory_used"] = gpu.memoryUsed
                    resources["gpu_memory_total"] = gpu.memoryTotal
                    resources["gpu_utilization"] = gpu.load * 100
            except Exception as e:
                if logger:
                    logger.warning(f"Error getting GPU info: {e}")

    # Log resource information
    if logger:
        logger.debug(f"CPU Usage: {cpu_percent:.1f}%")
        logger.debug(
            f"Memory Usage: {memory.percent:.1f}% "
            f"({memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB)"
        )

        if "gpu_memory_used" in resources:
            gpu_name = torch.cuda.get_device_name(0) if TORCH_AVAILABLE and torch.cuda.is_available() else "GPU"
            logger.debug(f"GPU ({gpu_name}): {resources['gpu_memory_used']:.1f}MB / "
                       f"{resources['gpu_memory_total']:.1f}MB "
                       f"({resources.get('gpu_utilization', 'N/A'):.1f}% Utilization)")

    # Check for high memory usage
    if memory.percent > warning_threshold * 100:
        if logger:
            logger.warning(f"High memory usage: {memory.percent:.1f}%")

        if garbage_collect:
            # Run garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get memory usage after garbage collection
            memory_after = psutil.virtual_memory()
            memory_freed = (memory.used - memory_after.used) / (1024 ** 3)  # GB

            if logger:
                logger.info(
                    f"GC freed {memory_freed:.2f}GB. "
                    f"Mem now: {memory_after.percent:.1f}%"
                )

            resources["memory_after_gc"] = memory_after.percent
            resources["memory_freed_gb"] = memory_freed

    return resources


def save_config(
    config: Dict[str, Any], log_dir: str, filename: str = "config.json"
) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config: Configuration dictionary
        log_dir: Directory to save the configuration
        filename: Name of the configuration file
    """
    # Create log directory if it doesn't exist
    ensure_dir_exists(log_dir)

    # Create a copy of config to modify
    config_to_save = {}

    # Handle non-serializable objects
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple, type(None))):
            config_to_save[key] = value
        else:
            try:
                # Try to serialize with json
                json.dumps({key: value})
                config_to_save[key] = value
            except (TypeError, OverflowError):
                # If not serializable, convert to string
                config_to_save[key] = str(value)

    # Save to file
    config_path = os.path.join(log_dir, filename)
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=4)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)

    return config


def create_evaluation_plots(
    portfolio_values: np.ndarray,
    actions: Optional[List[int]] = None,
    rewards: Optional[List[float]] = None,
    positions: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = False,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Create evaluation plots for the trading agent.

    Args:
        portfolio_values: List of portfolio values over time
        actions: List of actions taken by the agent
        rewards: List of rewards received by the agent
        positions: List of positions held by the agent
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    """
    plot_logger = logging.getLogger("rl_agent.plotting") # Use a specific logger
    plot_logger.debug(f"create_evaluation_plots called. save_path={save_path}")

    if portfolio_values.size == 0:
        plot_logger.warning("No portfolio values to plot, exiting.")
        return

    # Handle None values
    actions = [] if actions is None else actions
    rewards = [] if rewards is None else rewards
    positions = [] if positions is None else positions

    # Create figure and axes
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

    try:
        # Set style
        sns.set_style("whitegrid")

        # Plot portfolio values
        steps = list(range(len(portfolio_values)))
        axes[0].plot(steps, portfolio_values, label="Portfolio Value", color="blue",
                     linewidth=2)
        axes[0].set_title("Portfolio Value Over Time", fontsize=14)
        axes[0].set_ylabel("Value", fontsize=12)
        axes[0].legend(loc="upper left")
        axes[0].grid(True)

        # Calculate returns if we have enough data
        if len(portfolio_values) > 1:
            # Add basic check for zero/negative values before division
            safe_portfolio_values = np.maximum(portfolio_values[:-1], 1e-9)
            returns = np.diff(portfolio_values) / safe_portfolio_values
            # Check for NaNs/Infs in returns before cumprod
            if np.any(~np.isfinite(returns)):
                plot_logger.warning("NaN or Inf detected in returns, cannot plot cumulative.")
            else:
                cumulative_returns = np.cumprod(1 + returns) - 1
                # Add returns to the first plot
                ax_returns = axes[0].twinx()
                ax_returns.plot(
                    steps[1:], cumulative_returns,
                    label="Cumulative Return", color="green",
                    linestyle="--", linewidth=1.5
                )
                ax_returns.set_ylabel("Cumulative Return (%)", fontsize=12)
                ax_returns.legend(loc="upper right")

        # Plot actions if available
        if actions:
            if len(actions) > len(steps):
                actions = actions[:len(steps)]
                plot_logger.debug(f"Trimmed actions to match steps: {len(actions)}")
            elif len(actions) < len(steps):
                padding_needed = len(steps) - len(actions)
                actions.extend([0] * padding_needed)
                plot_logger.debug(f"Padded actions by {padding_needed} to match steps.")

            # Plot actions as scatter points
            action_labels = {0: "Sell", 1: "Hold", 2: "Buy"}
            action_colors = {0: "red", 1: "gray", 2: "green"}

            # Count actions
            unique_actions, counts = np.unique(actions, return_counts=True)
            action_counts = dict(zip(unique_actions, counts))

            # Create action array for plotting
            action_array = np.array(actions)

            for action in sorted(action_counts.keys()):
                mask = action_array == action
                if np.any(mask):
                    axes[1].scatter(
                        np.array(steps)[mask],
                        np.ones(mask.sum()) * action,
                        label=f"{action_labels.get(action, action)} "
                              f"({action_counts.get(action, 0)})",
                        color=action_colors.get(action, "blue"),
                        s=50,
                        alpha=0.7
                    )

            axes[1].set_title("Actions Taken by Agent", fontsize=14)
            axes[1].set_ylabel("Action", fontsize=12)
            if unique_actions.size > 0:
                axes[1].set_yticks(list(sorted(unique_actions)))
                axes[1].set_yticklabels([
                    action_labels.get(a, a) for a in sorted(unique_actions)
                ])
            axes[1].legend(loc="upper right")
            axes[1].grid(True)

        # Plot rewards or positions in the third subplot
        if rewards:
            reward_steps = steps[:len(rewards)]
            axes[2].plot(reward_steps, rewards, label="Reward",
                         color="purple", linewidth=1.5)
            axes[2].set_title("Rewards", fontsize=14)
            axes[2].set_ylabel("Reward", fontsize=12)
            axes[2].set_xlabel("Step", fontsize=12)
            axes[2].legend(loc="upper left")
            axes[2].grid(True)

            # Plot cumulative rewards
            ax_cum_rewards = axes[2].twinx()
            cum_rewards = np.cumsum(rewards)
            ax_cum_rewards.plot(
                reward_steps, cum_rewards,
                label="Cumulative Reward", color="orange",
                linestyle="--", linewidth=1.5
            )
            ax_cum_rewards.set_ylabel("Cumulative Reward", fontsize=12)
            ax_cum_rewards.legend(loc="upper right")

        # Add positions if available
        if positions:
            position_steps = steps[:len(positions)]
            if not rewards:  # Only if we haven't plotted rewards
                axes[2].plot(position_steps, positions, label="Position",
                             color="brown", linewidth=1.5)
                axes[2].set_title("Positions", fontsize=14)
                axes[2].set_ylabel("Position", fontsize=12)
                axes[2].set_xlabel("Step", fontsize=12)
                axes[2].legend(loc="upper left")
                axes[2].grid(True)
            else:
                # Add positions to the rewards plot
                ax_pos = axes[2].twinx()
                # Offset the axis slightly if rewards were plotted
                spine_offset = 60
                ax_pos.spines["right"].set_position(("axes", 1.0 + spine_offset / 72.0))
                ax_pos.plot(
                    position_steps, positions, label="Position",
                    color="brown", linestyle=":", linewidth=1.5
                )
                ax_pos.set_ylabel("Position", fontsize=12)
                ax_pos.legend(loc="lower right")

        # Adjust layout
        plt.tight_layout()

    except Exception as e:
        plot_logger.error(f"Error occurred during plot generation: {e}", exc_info=True)
        plt.close(fig) # Close the potentially broken figure
        return # Exit if plot generation failed

    # Save or show the plot
    if save_path:
        try:
            # Construct the full file path here
            full_file_path = os.path.join(save_path, "evaluation_plots.png")
            plot_logger.info(f"Attempting to save plot to: {full_file_path}")
            plt.savefig(full_file_path, dpi=300, bbox_inches="tight")
            plot_logger.info(f"Plot successfully saved to {full_file_path}")
        except Exception as e:
            plot_logger.error(f"Error saving plot to {full_file_path}: {e}", exc_info=True)

    if show_plot:
        plot_logger.debug("Displaying plot.")
        plt.show()
    else:
        # Always close the figure if not shown to free memory
        plot_logger.debug("Closing plot figure.")
        plt.close(fig)


def calculate_trading_metrics(
    portfolio_values: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate trading performance metrics.

    Args:
        portfolio_values: List of portfolio values over time
        benchmark_returns: List of benchmark returns (optional)
        risk_free_rate: Annual risk-free rate (decimal, e.g., 0.02 for 2%)

    Returns:
        Dictionary with trading metrics
    """
    metrics = {}

    if portfolio_values.size < 2:
        return {"error": "Not enough data points"}

    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # Total return
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value
    metrics["total_return"] = total_return

    # Annualized return (assuming 252 trading days per year)
    # Adjust the 252 based on your data frequency
    n_periods = len(portfolio_values) - 1
    annualized_return = (1 + total_return) ** (252 / n_periods) - 1
    metrics["annualized_return"] = annualized_return

    # Volatility
    daily_volatility = np.std(returns)
    annualized_volatility = daily_volatility * np.sqrt(252)
    metrics["daily_volatility"] = daily_volatility
    metrics["annualized_volatility"] = annualized_volatility

    # Sharpe ratio
    daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_risk_free
    sharpe_ratio = (np.mean(excess_returns) /
                    (np.std(excess_returns) + 1e-10) * np.sqrt(252))
    metrics["sharpe_ratio"] = sharpe_ratio

    # Maximum drawdown
    max_drawdown = calculate_max_drawdown(portfolio_values)
    metrics["max_drawdown"] = max_drawdown

    # Calmar ratio
    calmar_ratio = annualized_return / (max_drawdown + 1e-10)
    metrics["calmar_ratio"] = calmar_ratio

    # Sortino ratio (downside deviation)
    negative_returns = returns[returns < 0]
    # Correct check for empty NumPy array
    if negative_returns.size > 0:  # <--- CORRECTED LINE
        downside_deviation = np.std(negative_returns) * np.sqrt(252)
        sortino_ratio = annualized_return / (downside_deviation + 1e-10)
    else:
        downside_deviation = 0
        sortino_ratio = annualized_return * 1000  # Arbitrarily large value

    metrics["downside_deviation"] = downside_deviation
    metrics["sortino_ratio"] = sortino_ratio

    # Benchmark comparison if provided
    if benchmark_returns is not None and benchmark_returns.size >= returns.size:
        # Adjust benchmark returns to match the length of portfolio returns
        benchmark_returns = benchmark_returns[:returns.size]

        # Beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / (benchmark_variance + 1e-10)
        metrics["beta"] = beta

        # Alpha (Jensen's alpha)
        benchmark_return = np.mean(benchmark_returns) * 252
        expected_return = (daily_risk_free * 252 + beta *
                           (benchmark_return - daily_risk_free * 252))
        alpha = annualized_return - expected_return
        metrics["alpha"] = alpha

        # Information ratio
        tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
        information_ratio = (annualized_return - benchmark_return) \
            / (tracking_error + 1e-10)
        metrics["tracking_error"] = tracking_error
        metrics["information_ratio"] = information_ratio

    return metrics


def calculate_max_drawdown(values: np.ndarray) -> float:
    """
    Calculate maximum drawdown from a series of values.

    Args:
        values: List of values (e.g., portfolio values)

    Returns:
        Maximum drawdown as a decimal
    """
    if values.size == 0:
        return 0.0

    # Ensure input is a NumPy array for vectorized operations if possible
    # Although iteration works, explicit array is safer
    values = np.asarray(values)

    max_so_far = values[0]
    max_drawdown = 0

    for value in values:
        if value > max_so_far:
            max_so_far = value
        drawdown = (max_so_far - value) / max_so_far
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def execute_with_retry(
    func: callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Execute a function with retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries (seconds)
        backoff_factor: Factor to increase delay on each retry
        allowed_exceptions: Exceptions to retry on
        logger: Logger to log retry information

    Returns:
        Result of the function
    """
    retries = 0
    delay = retry_delay

    while True:
        try:
            return func()
        except allowed_exceptions as e:
            retries += 1

            if retries > max_retries:
                if logger:
                    logger.error(f"Failed after {max_retries} retries: {e}")
                raise

            if logger:
                logger.warning(f"Retry {retries}/{max_retries} after error: {e}")

            time.sleep(delay)
            delay *= backoff_factor


def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Potentially add these for full determinism (can impact performance)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")