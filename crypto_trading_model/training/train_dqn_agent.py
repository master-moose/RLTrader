import os
import pandas as pd
import numpy as np
import argparse
from typing import Dict
import logging
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from crypto_trading_model.models.lstm import LSTMModel
from crypto_trading_model.environment.crypto_env import CryptocurrencyTradingEnv
from crypto_trading_model.data.data_loader import load_crypto_data
from crypto_trading_model.utils.logging import setup_logging

# Setup logging
logger = setup_logging()

# FinRL imports
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processor import DataProcessor

def prepare_crypto_data_for_finrl(
    data: Dict[str, pd.DataFrame],
    timeframe: str = '15m',
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    Prepare cryptocurrency data for FinRL environment
    
    Args:
        data: Dictionary of dataframes for different timeframes
        timeframe: Selected timeframe
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        DataFrame in FinRL format
    """
    logger.info("Preparing data for FinRL environment")
    
    # Get the selected timeframe data
    df = data[timeframe].copy()
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Filter by date range if specified
    if start_date and end_date:
        df = df[start_date:end_date]
    
    # Add required columns for FinRL
    df['tic'] = 'BTC'  # Add ticker column
    df['date'] = df.index  # Add date column
    
    # Ensure we have the required columns for FinRL
    required_columns = [
        'date', 'tic', 'open', 'high', 'low', 'close', 'volume'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns. Need: {required_columns}")
    
    # Select and order columns for FinRL
    finrl_columns = (
        required_columns + 
        [col for col in df.columns if col not in required_columns]
    )
    df = df[finrl_columns]
    
    logger.info(f"Data prepared for FinRL with shape: {df.shape}")
    return df 