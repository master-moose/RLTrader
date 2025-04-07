#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd

# Try to import torch with fallback for NCCL errors
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    TORCH_IMPORT_ERROR = str(e)
    # We'll handle this later

from pathlib import Path
import tables
import time
from collections import defaultdict

# Try to import sklearn metrics
try:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
except ImportError:
    print("Warning: sklearn not available, some metrics will be disabled")

# Try to import CuPy for GPU accelerated array operations if CUDA is available
try:
    import cupy as cp
    HAS_CUPY = True
    logger_message = "CuPy imported successfully, will use GPU-accelerated array operations"
except ImportError:
    HAS_CUPY = False
    cp = np  # Fallback to NumPy
    logger_message = "CuPy not available, falling back to NumPy"

# Function to use the appropriate array module based on device
def get_array_module(device):
    """Return the appropriate array module (NumPy or CuPy) based on device"""
    if device and device.type == 'cuda' and HAS_CUPY:
        return cp
    return np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('crypto_trading_model.training')

# If there was a torch import error, log it now
if TORCH_IMPORT_ERROR:
    logger.error(f"Error importing PyTorch: {TORCH_IMPORT_ERROR}")
    logger.warning("If this is an NCCL error, will try to continue with CPU-only mode")
    
    # Try importing torch again with CPU-only mode
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPUs
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        logger.info("Successfully imported PyTorch in CPU-only mode")
    except ImportError as e:
        logger.error(f"Still couldn't import PyTorch: {e}")
        logger.error("Exiting...")
        sys.exit(1)

def setup_directories():
    """Create necessary output directories for the model training results."""
    directories = [
        'output/time_series',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def load_config(config_path):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        sys.exit(1)

class TimeSeriesDataset(Dataset):
    """Dataset for loading time series data from HDF5 files"""
    
    def __init__(self, 
                 data_path: str, 
                 timeframes: list = ['15m', '4h', '1d'],
                 sequence_length: dict = None,
                 target_column: str = 'close',
                 feature_columns: list = None,
                 normalize: bool = True,
                 use_cupy: bool = None):
        """
        Initialize the dataset
        
        Parameters:
        - data_path: Path to the HDF5 file with time series data
        - timeframes: List of timeframes to include
        - sequence_length: Dict mapping timeframes to their sequence lengths or single int for all timeframes
        - target_column: Column to use as the prediction target
        - feature_columns: List of columns to use as features (None = use all)
        - normalize: Whether to normalize the features
        - use_cupy: Whether to use CuPy for computations (None = auto-detect)
        """
        self.data_path = data_path
        self.timeframes = timeframes
        
        # Determine whether to use CuPy - store as flag but don't keep module reference
        self.use_cupy = use_cupy if use_cupy is not None else HAS_CUPY
        
        # Handle sequence_length as either dict or int
        if isinstance(sequence_length, dict):
            self.sequence_length = sequence_length
        else:
            # Default to 50 if not specified
            default_length = 50 if sequence_length is None else sequence_length
            self.sequence_length = {tf: default_length for tf in timeframes}
            
        self.target_column = target_column
        self.normalize = normalize
        
        # Load data from HDF5 file
        self.data = {}
        timestamps = {}
        
        with pd.HDFStore(data_path, mode='r') as store:
            # Check available timeframes in the store
            available_timeframes = [key[1:] for key in store.keys()]  # Remove leading '/'
            logger.info(f"Available timeframes in {data_path}: {available_timeframes}")
            
            # Log CuPy usage
            if self.use_cupy and HAS_CUPY:
                logger.info("Using CuPy for data normalization")
            
            # Load each requested timeframe
            for tf in timeframes:
                if f'/{tf}' in store:
                    # Load the dataframe
                    df = store[f'/{tf}']
                    
                    # Ensure timestamp column exists for alignment
                    if 'timestamp' not in df.columns:
                        logger.warning(f"No timestamp column in {tf} data, using index as timestamp")
                        # Use a different name to prevent ambiguity with index
                        df['time_col'] = df.index
                        timestamps[tf] = set(df['time_col'])
                    else:
                        timestamps[tf] = set(df['timestamp'])
                    
                    # Store timestamps for alignment
                    
                    # Select feature columns if specified
                    if feature_columns is not None:
                        # Ensure target column is included
                        if target_column not in feature_columns:
                            cols = feature_columns + [target_column]
                        else:
                            cols = feature_columns
                            
                        # Select only the specified columns that exist in the dataframe
                        existing_cols = [col for col in cols if col in df.columns]
                        df = df[existing_cols]
                    
                    # Normalize if requested
                    if normalize:
                        for col in df.columns:
                            # Skip timestamp columns and any non-numeric columns
                            if col not in ['timestamp', 'time_col'] and np.issubdtype(df[col].dtype, np.number):
                                # Use CuPy if available for normalization
                                if self.use_cupy and HAS_CUPY:
                                    # Convert to CuPy array for processing
                                    values = cp.array(df[col].values)
                                    mean = cp.mean(values)
                                    std = cp.std(values)
                                    if std > 0:
                                        normalized = (values - mean) / std
                                        # Convert back to numpy for pandas
                                        df[col] = cp.asnumpy(normalized)
                                else:
                                    # Use pandas/numpy directly
                                    mean = df[col].mean()
                                    std = df[col].std()
                                    if std > 0:
                                        df[col] = (df[col] - mean) / std
                    
                    self.data[tf] = df
                else:
                    logger.warning(f"Timeframe '{tf}' not found in {data_path}")
        
        # Find common timestamps across all timeframes for alignment
        if len(timestamps) > 0:
            common_timestamps = set.intersection(*timestamps.values())
            logger.info(f"Found {len(common_timestamps)} common timestamps across all timeframes")
            
            # Filter data to only include common timestamps
            for tf in self.data:
                df = self.data[tf]
                # Use the appropriate timestamp column name
                time_col = 'time_col' if 'time_col' in df.columns else 'timestamp'
                
                # Filter to common timestamps
                self.data[tf] = df[df[time_col].isin(common_timestamps)]
                
                # Sort by timestamp
                self.data[tf] = self.data[tf].sort_values(time_col)
                
                # Remove timestamp columns before modeling to avoid issues
                if 'time_col' in self.data[tf].columns:
                    self.data[tf] = self.data[tf].drop(columns=['time_col'])
                if 'timestamp' in self.data[tf].columns:
                    self.data[tf] = self.data[tf].drop(columns=['timestamp'])
        
        # Calculate valid indices (ensuring we have enough sequential data)
        # For valid indices, we need to ensure we have enough data for the longest sequence
        max_sequence_length = max(self.sequence_length.values())
        min_length = min(len(df) for df in self.data.values())
        self.valid_indices = range(max_sequence_length, min_length)
        logger.info(f"Loaded dataset with {len(self.valid_indices)} valid samples")
        logger.info(f"Using sequence lengths: {self.sequence_length}")
    
    def get_array_module(self):
        """Helper method to get the appropriate array module (NumPy or CuPy)"""
        if self.use_cupy and HAS_CUPY:
            return cp
        return np

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual index in the dataframes
        index = self.valid_indices[idx]
        
        # Extract sequences for each timeframe
        sequences = {}
        for tf, df in self.data.items():
            # Get the sequence using timeframe-specific length
            tf_seq_length = self.sequence_length[tf]
            sequence = df.iloc[index - tf_seq_length:index].copy()
            
            # Convert to tensor
            feature_tensor = torch.tensor(sequence.values, dtype=torch.float32)
            sequences[tf] = feature_tensor
        
        # Extract target (future price movement)
        target = {}
        for tf, df in self.data.items():
            current_price = df[self.target_column].iloc[index - 1]
            next_price = df[self.target_column].iloc[index]
            
            # Calculate price change (can be used for regression)
            price_change = (next_price - current_price) / current_price
            
            # Create a classification target (1=up, 0=neutral, -1=down)
            if price_change > 0.001:  # 0.1% threshold for up
                direction = 1  # Buy signal
            elif price_change < -0.001:  # -0.1% threshold for down
                direction = 2  # Sell signal
            else:
                direction = 0  # Hold signal
                
            target[tf] = direction
        
        # Use the smallest timeframe for the primary target
        primary_tf = self.timeframes[0]
        primary_target = target[primary_tf]
        
        return sequences, torch.tensor(primary_target, dtype=torch.long)

def create_dataloaders(train_path, val_path, batch_size=32, use_cupy=None, disable_mp=False, balanced_sampling=True, **dataset_kwargs):
    """
    Create DataLoader objects for training and validation
    
    Parameters:
    - train_path: Path to training data
    - val_path: Path to validation data
    - batch_size: Batch size
    - use_cupy: Whether to use CuPy for computations
    - disable_mp: Whether to disable multiprocessing
    - balanced_sampling: Whether to use balanced sampling for training data
    - **dataset_kwargs: Additional arguments to pass to TimeSeriesDataset
    
    Returns:
    - (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(train_path, use_cupy=use_cupy, **dataset_kwargs)
    val_dataset = TimeSeriesDataset(val_path, use_cupy=use_cupy, **dataset_kwargs)
    
    logger.info(f"Created datasets with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Determine whether to use multiprocessing
    # We can't use multiprocessing if CuPy arrays are involved (they're not picklable)
    if use_cupy or disable_mp:
        num_workers = 0
        logger.info("Using single-process data loading (CuPy detected or multiprocessing disabled)")
    else:
        num_workers = min(4, os.cpu_count() or 1)
        logger.info(f"Using {num_workers} processes for data loading")
    
    # If using balanced sampling, create a sampler that rebalances class weights
    if balanced_sampling:
        # Get all class labels
        train_labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            train_labels.append(label.item() if hasattr(label, 'item') else label)
        
        # Count samples per class
        class_sample_count = np.unique(train_labels, return_counts=True)[1]
        logger.info(f"Class distribution in training data: {class_sample_count}")
        
        # Calculate weights per sample (inversely proportional to class frequency)
        weight = 1. / class_sample_count
        samples_weight = torch.tensor([weight[t] for t in train_labels])
        
        # Create a WeightedRandomSampler
        train_sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )
        logger.info("Using weighted sampler to balance class distribution during training")
        
        # Create DataLoader with the sampler
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # Standard DataLoader with shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Create validation loader (no need for balanced sampling here)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class MultiTimeframeModel(torch.nn.Module):
    """
    Deep learning model that processes multiple timeframes
    
    Features:
    - Separate input paths for each timeframe
    - LSTM layers for sequence processing
    - Attention mechanisms between timeframes
    - Dense output layers for predictions
    """
    
    def __init__(self, 
                input_dims: dict, 
                hidden_dims: int = 128, 
                num_layers: int = 2,
                dropout: float = 0.2,
                bidirectional: bool = True,
                attention: bool = True,
                num_classes: int = 3):
        """
        Initialize the multi-timeframe model
        
        Parameters:
        - input_dims: Dictionary mapping timeframe names to input feature dimensions
        - hidden_dims: Dimension of hidden layers
        - num_layers: Number of LSTM layers
        - dropout: Dropout probability
        - bidirectional: Whether to use bidirectional LSTM
        - attention: Whether to use attention mechanism
        - num_classes: Number of output classes (3 for buy/hold/sell)
        """
        super().__init__()
        
        self.timeframes = list(input_dims.keys())
        self.hidden_dims = hidden_dims
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_directions = 2 if bidirectional else 1
        
        # Create encoder LSTMs for each timeframe
        self.encoders = torch.nn.ModuleDict()
        for tf, dim in input_dims.items():
            self.encoders[tf] = torch.nn.LSTM(
                input_size=dim,
                hidden_size=hidden_dims,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Attention mechanism
        if attention:
            # Simplified attention mechanism
            self.attn_hidden_dim = hidden_dims * self.num_directions
            
            # Query vectors for each timeframe
            self.query_vectors = torch.nn.ParameterDict()
            for tf in self.timeframes:
                self.query_vectors[tf] = torch.nn.Parameter(torch.randn(self.attn_hidden_dim))
            
            # Attention projection
            self.attention_projection = torch.nn.Linear(
                self.attn_hidden_dim * len(self.timeframes), 
                self.attn_hidden_dim
            )
        
        # Output layers
        if attention:
            output_dim = self.attn_hidden_dim
        else:
            output_dim = self.attn_hidden_dim * len(self.timeframes)
        
        self.fc1 = torch.nn.Linear(output_dim, hidden_dims)
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims // 2)
        self.fc3 = torch.nn.Linear(hidden_dims // 2, num_classes)
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, inputs):
        """
        Forward pass through the model
        
        Parameters:
        - inputs: Dictionary mapping timeframe names to input tensors
                  Each tensor has shape (batch_size, seq_len, input_dim)
        
        Returns:
        - Tensor with shape (batch_size, num_classes)
        """
        # Process each timeframe through its encoder
        encoded_timeframes = {}
        for tf, encoder in self.encoders.items():
            if tf in inputs:
                # Run the encoder
                output, (hidden, cell) = encoder(inputs[tf])
                
                # Get the final hidden state(s)
                if self.bidirectional:
                    # Concatenate the last hidden state from both directions
                    final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
                else:
                    final_hidden = hidden[-1]
                
                encoded_timeframes[tf] = final_hidden
            else:
                # Handle missing timeframe
                batch_size = next(iter(inputs.values())).size(0)
                device = next(iter(inputs.values())).device
                
                # Create a zero tensor of appropriate size
                encoded_timeframes[tf] = torch.zeros(
                    batch_size, 
                    self.hidden_dims * self.num_directions,
                    device=device
                )
        
        # Apply attention mechanism if enabled
        if self.attention:
            # Calculate attention scores for each timeframe
            attended_encodings = []
            
            for tf in self.timeframes:
                # Compute attention scores between query vector and encoded timeframe
                encoding = encoded_timeframes[tf]
                query = self.query_vectors[tf].unsqueeze(0).expand(encoding.size(0), -1)
                
                # Simple dot-product attention
                attention_score = torch.sum(query * encoding, dim=1, keepdim=True)
                attention_weights = torch.nn.functional.softmax(attention_score, dim=0)
                
                # Apply attention weights
                attended = attention_weights * encoding
                attended_encodings.append(attended)
            
            # Concatenate all attended vectors
            concat_encodings = torch.cat(attended_encodings, dim=1)
            
            # Project to final representation
            combined = self.attention_projection(concat_encodings)
        else:
            # Simple concatenation of all timeframe encodings
            combined = torch.cat([encoded_timeframes[tf] for tf in self.timeframes], dim=1)
        
        # Pass through fully connected layers
        x = torch.nn.functional.relu(self.fc1(self.dropout(combined)))
        x = torch.nn.functional.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    
    start_time = time.time()
    
    # Detailed timing metrics
    timing_stats = {
        'data_transfer': 0.0,
        'forward_pass': 0.0,
        'backward_pass': 0.0,
        'optimization': 0.0
    }
    
    # GPU memory tracking
    gpu_memory_usage = []
    
    total_samples = 0
    
    for batch_idx, (features, targets) in enumerate(dataloader):
        # Track GPU memory if using CUDA
        if device.type == 'cuda':
            gpu_memory_usage.append(torch.cuda.memory_allocated(device) / (1024 * 1024))  # MB
            
        # Track number of samples
        batch_size = targets.size(0)
        total_samples += batch_size
        
        # Measure data transfer time
        data_start = time.time()
        
        # Move data to device
        for tf in features:
            features[tf] = features[tf].to(device)
        targets = targets.to(device)
        
        data_end = time.time()
        timing_stats['data_transfer'] += data_end - data_start
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass - measure time
        forward_start = time.time()
        outputs = model(features)
        forward_end = time.time()
        timing_stats['forward_pass'] += forward_end - forward_start
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize - measure time
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_stats['backward_pass'] += backward_end - backward_start
        
        # Optimization step - measure time
        optim_start = time.time()
        optimizer.step()
        optim_end = time.time()
        timing_stats['optimization'] += optim_end - optim_start
        
        # Track statistics
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        batch_correct = (predicted == targets).sum().item()
        epoch_correct += batch_correct
        epoch_total += targets.size(0)
        
        if batch_idx % 10 == 0:
            batch_time = time.time() - (data_start if batch_idx > 0 else start_time)
            samples_per_sec = batch_size / (batch_time / 10 if batch_idx % 10 == 0 else batch_time)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Add GPU info if available
            gpu_info = ""
            if device.type == 'cuda':
                gpu_info = f", GPU Mem: {torch.cuda.memory_allocated(device) / (1024 * 1024):.1f}MB"
                
            logger.info(f"Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, "
                       f"Acc={batch_correct/targets.size(0):.4f}, "
                       f"LR={current_lr:.6f}, "
                       f"Throughput={samples_per_sec:.1f} samples/sec{gpu_info}")
    
    # Calculate epoch metrics
    avg_loss = epoch_loss / len(dataloader)
    accuracy = epoch_correct / epoch_total
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    # Calculate throughput
    throughput = total_samples / epoch_time
    
    # Add GPU stats if available
    gpu_stats = {}
    if device.type == 'cuda':
        gpu_stats = {
            'avg_memory_usage_mb': sum(gpu_memory_usage) / len(gpu_memory_usage) if gpu_memory_usage else 0,
            'peak_memory_usage_mb': max(gpu_memory_usage) if gpu_memory_usage else 0,
            'gpu_utilization': None  # Placeholder, not easily accessible from PyTorch
        }
        # Log GPU memory usage
        logger.info(f"GPU Memory: Avg={gpu_stats['avg_memory_usage_mb']:.1f}MB, Peak={gpu_stats['peak_memory_usage_mb']:.1f}MB")
    
    # Log timing breakdown
    logger.info(f"Epoch time breakdown: "
               f"Data transfer: {timing_stats['data_transfer']:.2f}s ({timing_stats['data_transfer']/epoch_time*100:.1f}%), "
               f"Forward: {timing_stats['forward_pass']:.2f}s ({timing_stats['forward_pass']/epoch_time*100:.1f}%), "
               f"Backward: {timing_stats['backward_pass']:.2f}s ({timing_stats['backward_pass']/epoch_time*100:.1f}%), "
               f"Optimization: {timing_stats['optimization']:.2f}s ({timing_stats['optimization']/epoch_time*100:.1f}%)")
    logger.info(f"Overall throughput: {throughput:.2f} samples/second")
    
    return avg_loss, accuracy, epoch_time, throughput, timing_stats, gpu_stats if device.type == 'cuda' else None

def validate(model, dataloader, criterion, device, num_classes=3):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    # For trading metrics
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    total_samples = 0
    
    with torch.no_grad():
        for features, targets in dataloader:
            # Track number of samples
            batch_size = targets.size(0)
            total_samples += batch_size
            
            # Move data to device
            for tf in features:
                features[tf] = features[tf].to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == targets).sum().item()
            val_correct += batch_correct
            val_total += targets.size(0)
            
            # Store predictions and targets for additional metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate validation metrics
    avg_loss = val_loss / len(dataloader)
    accuracy = val_correct / val_total
    
    # Calculate trading-specific metrics
    metrics = calculate_trading_metrics(all_predictions, all_targets, device, num_classes=num_classes)
    
    end_time = time.time()
    val_time = end_time - start_time
    
    # Calculate throughput
    throughput = total_samples / val_time
    logger.info(f"Validation throughput: {throughput:.2f} samples/second")
    
    # Log trading metrics
    logger.info(f"Trading metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Log class breakdown, safely handling missing classes
    class_logs = []
    class_names = ["Hold", "Buy", "Sell"]
    for i, name in enumerate(class_names):
        if i < len(metrics['class_accuracy']):
            class_logs.append(f"{name}: {metrics['class_accuracy'][i]:.4f}")
        else:
            class_logs.append(f"{name}: N/A (no samples)")
    
    logger.info(f"Class breakdown - {', '.join(class_logs)}")
    
    return avg_loss, accuracy, val_time, throughput, metrics

def calculate_trading_metrics(predictions, targets, device=None, num_classes=3):
    """
    Calculate trading-specific metrics
    
    Parameters:
    - predictions: List of model predictions (0=hold, 1=buy, 2=sell)
    - targets: List of actual targets
    - device: The torch device being used (to determine whether to use CuPy)
    - num_classes: Number of classes in the classification task
    
    Returns:
    - Dictionary of metrics
    """
    # Get the appropriate array module
    if device and device.type == 'cuda' and HAS_CUPY:
        xp = cp
    else:
        xp = np
    
    # Convert inputs to appropriate array type
    predictions = xp.array(predictions)
    targets = xp.array(targets)
    
    # Special case: if predictions or targets are all the same value, metrics will be undefined
    if len(xp.unique(predictions)) <= 1 or len(xp.unique(targets)) <= 1:
        logger.warning("All predictions or targets are the same class. Metrics will be unreliable.")
    
    # If using CuPy, need to convert back to NumPy for sklearn metrics
    if xp is cp:
        # Move arrays to CPU to use sklearn
        np_predictions = cp.asnumpy(predictions)
        np_targets = cp.asnumpy(targets)
        
        try:
            # Calculate sklearn metrics with NumPy arrays
            precision, recall, f1, _ = precision_recall_fscore_support(
                np_targets, np_predictions, average='macro', zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error calculating precision_recall_fscore: {e}")
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        try:
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(np_targets, np_predictions)
            conf_matrix = xp.array(conf_matrix)  # Convert back to CuPy if needed
        except Exception as e:
            logger.warning(f"Error calculating confusion matrix: {e}")
            # Create a dummy 3x3 confusion matrix with zeros
            conf_matrix = xp.zeros((num_classes, num_classes))
        
        # Get unique classes
        unique_classes = xp.unique(xp.concatenate([predictions, targets]))
    else:
        # Using NumPy directly
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions, average='macro', zero_division=0
            )
        except Exception as e:
            logger.warning(f"Error calculating precision_recall_fscore: {e}")
            precision, recall, f1 = 0.0, 0.0, 0.0
            
        try:
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(targets, predictions)
        except Exception as e:
            logger.warning(f"Error calculating confusion matrix: {e}")
            # Create a dummy 3x3 confusion matrix
            conf_matrix = xp.zeros((3, 3))
        
        # Get unique classes
        unique_classes = xp.unique(xp.concatenate([predictions, targets]))
    
    # Calculate class-specific accuracy
    # First make sure the confusion matrix has the right shape for num_classes
    if conf_matrix.shape[0] < num_classes:
        # Extend with zeros if some classes are missing
        extended_conf_matrix = xp.zeros((num_classes, num_classes))
        extended_conf_matrix[:conf_matrix.shape[0], :conf_matrix.shape[1]] = conf_matrix
        conf_matrix = extended_conf_matrix
    
    # CuPy doesn't have errstate context manager, so handle differently based on array module
    if xp is np:
        with np.errstate(divide='ignore', invalid='ignore'):
            class_accuracy = conf_matrix.diagonal() / xp.maximum(conf_matrix.sum(axis=1), 1e-10)
    else:
        # For CuPy, just do the division and handle NaNs afterwards
        class_accuracy = conf_matrix.diagonal() / xp.maximum(conf_matrix.sum(axis=1), 1e-10)
    
    # Replace NaN values (from division by zero) with zero
    class_accuracy = xp.nan_to_num(class_accuracy)
    
    # Calculate additional trading metrics
    # A basic trading score: higher weight for correctly predicting buy/sell signals
    # and lower penalty for mispredicting hold signals
    trading_score = 0
    
    # Convert to numpy or CPU if needed for iteration
    if xp is cp:
        iter_predictions = cp.asnumpy(predictions)
        iter_targets = cp.asnumpy(targets)
    else:
        iter_predictions = predictions
        iter_targets = targets
        
    for pred, target in zip(iter_predictions, iter_targets):
        if pred == target:
            if target in [1, 2]:  # Buy or Sell
                trading_score += 2  # Higher weight for correct buy/sell
            else:  # Hold
                trading_score += 1
        else:
            if target == 0 and pred in [1, 2]:  # Predicted action when should hold
                trading_score -= 1
            elif target in [1, 2] and pred == 0:  # Missed an action
                trading_score -= 2
            else:  # Wrong action
                trading_score -= 3
    
    trading_score = trading_score / max(len(predictions), 1)  # Avoid division by zero
    
    # Convert to CPU for serialization if using CuPy
    if xp is cp:
        class_accuracy = cp.asnumpy(class_accuracy)
        unique_classes = cp.asnumpy(unique_classes)
    
    # Convert numpy arrays to lists for JSON serialization
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'trading_score': float(trading_score),
        'class_accuracy': class_accuracy.tolist() if hasattr(class_accuracy, 'tolist') else list(class_accuracy),
        'unique_classes': unique_classes.tolist() if hasattr(unique_classes, 'tolist') else list(unique_classes)
    }

def check_gpu_availability():
    """Check if a GPU is available and return information about it"""
    # If we're in forced CPU mode due to NCCL errors, return false
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
        logger.info("Running in CPU-only mode due to CUDA/NCCL initialization issues")
        return False, None
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        cuda_version = torch.version.cuda
        
        logger.info(f"GPU detected: {gpu_name} (CUDA {cuda_version})")
        logger.info(f"Number of GPUs available: {gpu_count}")
        
        # Log CuPy availability
        if HAS_CUPY:
            logger.info("CuPy is available and will be used for GPU-accelerated array operations")
            # Log CuPy version if possible
            try:
                cupy_version = cp.__version__
                logger.info(f"CuPy version: {cupy_version}")
            except:
                logger.info("CuPy version information unavailable")
        else:
            logger.info("CuPy is not available, using NumPy for array operations")
        
        # Get memory info if possible
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"GPU memory: {total_mem:.2f} GB")
        except:
            logger.info("Could not retrieve GPU memory information")
            
        return True, gpu_name
    else:
        logger.warning("No GPU detected, using CPU")
        return False, None

def configure_gpu_settings():
    """Configure settings for optimal GPU performance"""
    # Skip if we're in forced CPU mode
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
        return
        
    if torch.cuda.is_available():
        # Enable cuDNN benchmark mode for optimal performance
        # This will benchmark several algorithms and select the fastest
        torch.backends.cudnn.benchmark = True
        
        # Set device precision
        torch.set_float32_matmul_precision('high')
        
        logger.info("Configured GPU settings for optimal performance")
    else:
        logger.warning("Cannot configure GPU settings - CUDA not available")

def train_model(config_path="crypto_trading_model/config/time_series_config.json", disable_mp=False):
    """Train the time series model using parameters from config file"""
    # Load configuration
    config = load_config(config_path)
    
    # Check for GPU availability
    gpu_available, gpu_name = check_gpu_availability()
    if gpu_available:
        configure_gpu_settings()
    else:
        logger.warning("Training will run on CPU, which may be significantly slower")
    
    # Extract parameters
    data_path = config['data']['data_path']
    output_dir = config['data']['output_dir']
    timeframes = config['data']['timeframes']
    seq_length = config['data']['sequence_length']
    use_synthetic = config['data']['use_synthetic']
    synthetic_path = config['data']['synthetic_path']
    
    # Model settings
    model_type = config['model']['type']
    hidden_dims = config['model']['hidden_dims']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    bidirectional = config['model']['bidirectional']
    attention = config['model']['attention']
    feature_dims = config['model']['feature_dims']
    
    # Training settings
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    patience = config['training']['patience']
    device_cfg = config['training']['device']
    
    # Modify training parameters
    early_stopping_patience = config.get('early_stopping_patience', 30)  # Increased patience
    optimizer_config = config.get('optimizer', {})
    learning_rate = optimizer_config.get('learning_rate', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 1e-5)
    
    # Ensure there's a validation set for early stopping
    val_ratio = config.get('validation_ratio', 0.2)
    
    # Determine data paths
    if use_synthetic:
        train_path = os.path.join(synthetic_path, 'train_data.h5')
        val_path = os.path.join(synthetic_path, 'val_data.h5')
    else:
        train_path = os.path.join(data_path, 'train_data.h5')
        val_path = os.path.join(data_path, 'val_data.h5')
    
    logger.info(f"Training with data: {train_path}, {val_path}")
    
    # Setup device - use CUDA if available and configured, else CPU
    if device_cfg == 'cuda' and torch.cuda.is_available() and ('CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] != ''):
        device = torch.device('cuda')
        logger.info("Using CUDA for training")
        # Use CuPy for array operations if available
        use_cupy = HAS_CUPY
        if use_cupy:
            logger.info("CuPy will be used for array operations")
            if disable_mp:
                logger.info("Multiprocessing disabled to avoid pickling issues with CuPy")
        else:
            logger.info("CuPy not available, using NumPy for array operations")
    else:
        device = torch.device('cpu')
        use_cupy = False  # Don't use CuPy with CPU
        if device_cfg == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
        elif 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '':
            logger.warning("CUDA disabled due to initialization errors, using CPU")
        else:
            logger.info("Using CPU for training")
    
    # Create dataloaders with CuPy option
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        timeframes=timeframes,
        sequence_length=seq_length,
        use_cupy=use_cupy,
        disable_mp=disable_mp,
        balanced_sampling=True  # Enable balanced sampling to address class imbalance
    )
    
    # Create model
    input_dims = {tf: feature_dims[tf] for tf in timeframes}
    num_classes = config['model']['num_classes']
    
    if model_type == 'multi_timeframe':
        model = MultiTimeframeModel(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            attention=attention,
            num_classes=num_classes
        )
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)
    
    model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Calculate class weights to address imbalance (hold, buy, sell)
    # This section should be after data is loaded and before model training
    if num_classes > 1:  # Classification task
        # Extract targets from the dataloader instead of directly from dataset
        logger.info("Computing class distribution for weighting...")
        class_counts = np.zeros(num_classes, dtype=np.int32)
        
        # Iterate through the training loader to count class occurrences
        for batch in train_loader:
            targets = batch[1].cpu().numpy()
            for c in range(num_classes):
                class_counts[c] += np.sum(targets == c)
        
        total_samples = np.sum(class_counts)
        logger.info(f"Class distribution: {class_counts}")
        
        # Compute class weights (inverse frequency)
        class_weights = {}
        for cls in range(num_classes):
            if class_counts[cls] > 0:
                # Weight is inversely proportional to frequency
                class_weights[cls] = total_samples / (num_classes * class_counts[cls])
            else:
                class_weights[cls] = 1.0
        
        logger.info(f"Using class weights to handle imbalance: {class_weights}")
    else:
        class_weights = None
    
    # Create criterion with weights for classification
    if num_classes > 1 and class_weights:
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([class_weights[i] for i in range(num_classes)], 
                               dtype=torch.float32,  # Explicitly use float32 to match model outputs
                               device=device)
        )
    else:
        criterion = torch.nn.CrossEntropyLoss() if num_classes > 1 else torch.nn.MSELoss()
    
    # Setup optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay  # L2 regularization
    )
    
    # Learning rate scheduler with slower decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce by 50% (less aggressive)
        patience=10,  # Wait longer before reducing
        min_lr=1e-6,
        verbose=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    best_f1_score = 0.0
    best_trading_score = float('-inf')
    best_model_state = None
    no_improve_epochs = 0
    
    # Function to check if we have improvement based on multiple metrics
    def is_improvement(new_loss, new_f1, new_trading_score):
        # Weighted combination of metrics (lower is better for loss, higher is better for F1 and trading score)
        # We convert each to a value where higher is better
        loss_component = 1.0 / (1.0 + new_loss)  # Transform loss so higher is better
        
        # Previous best
        prev_loss_component = 1.0 / (1.0 + best_val_loss)
        
        # Calculate overall scores (higher is better)
        # Putting more weight on F1 score to prioritize balanced class performance
        new_score = 0.3 * loss_component + 0.5 * new_f1 + 0.2 * new_trading_score
        old_score = 0.3 * prev_loss_component + 0.5 * best_f1_score + 0.2 * best_trading_score
        
        return new_score > old_score
    
    # Performance metrics tracking
    total_training_time = 0
    throughput_history = {
        'train': [],
        'val': []
    }
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_time': [],
        'val_time': [],
        'train_throughput': [],
        'val_throughput': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'trading_score': [],
        'class_accuracy_buy': [],
        'class_accuracy_hold': [],
        'class_accuracy_sell': [],
        'learning_rates': []  # Add learning rate tracking
    }
    
    # Track timing of each phase
    timing_breakdown = defaultdict(list)
    
    # Record training start time
    training_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Get current learning rate using get_last_lr()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{epochs} (Learning Rate: {current_lr:.6f})")
        
        # Train
        train_result = train_epoch(model, train_loader, criterion, optimizer, device)
        train_loss, train_acc, train_time, train_throughput, train_timing, gpu_stats = (
            train_result if len(train_result) == 6 else train_result + (None,)
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Time: {train_time:.2f} seconds, Train Throughput: {train_throughput:.2f} samples/second")
        
        # Save timing breakdown
        for key, value in train_timing.items():
            timing_breakdown[f'train_{key}'].append(value)
        
        # Validate
        val_loss, val_acc, val_time, val_throughput, val_metrics = validate(model, val_loader, criterion, device, num_classes=num_classes)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Time: {val_time:.2f} seconds, Val Throughput: {val_throughput:.2f} samples/second")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Get updated learning rate
        new_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate updated: {current_lr:.6f} -> {new_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)
        history['train_throughput'].append(train_throughput)
        history['val_throughput'].append(val_throughput)
        history['learning_rates'].append(new_lr)  # Store learning rate in history
        
        # Save trading metrics
        history['precision'].append(val_metrics['precision'])
        history['recall'].append(val_metrics['recall'])
        history['f1'].append(val_metrics['f1'])
        history['trading_score'].append(val_metrics['trading_score'])
        
        # Safely get class accuracies, using 0.0 if not available
        class_acc = val_metrics['class_accuracy']
        history['class_accuracy_hold'].append(class_acc[0] if 0 < len(class_acc) else 0.0)
        history['class_accuracy_buy'].append(class_acc[1] if 1 < len(class_acc) else 0.0)
        history['class_accuracy_sell'].append(class_acc[2] if 2 < len(class_acc) else 0.0)
        
        # Update throughput history
        throughput_history['train'].append(train_throughput)
        throughput_history['val'].append(val_throughput)
        
        # Track total training time
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        # Check for improvement
        if is_improvement(val_loss, val_metrics['f1'], val_metrics['trading_score']):
            best_val_loss = val_loss
            best_f1_score = val_metrics['f1']
            best_trading_score = val_metrics['trading_score']
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Calculate training statistics
    total_time = time.time() - training_start_time
    avg_epoch_time = total_training_time / epoch
    avg_train_throughput = sum(throughput_history['train']) / len(throughput_history['train'])
    avg_val_throughput = sum(throughput_history['val']) / len(throughput_history['val'])
    
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    logger.info(f"Average training throughput: {avg_train_throughput:.2f} samples/second")
    logger.info(f"Average validation throughput: {avg_val_throughput:.2f} samples/second")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'val_loss': best_val_loss,
        'timeframes': timeframes,
        'feature_dims': feature_dims,
        'epoch': epoch
    }, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert all values to native Python types for JSON serialization
        serializable_history = {}
        for k, v in history.items():
            if isinstance(v, list):
                # Handle list values (could contain numpy types)
                serializable_history[k] = []
                for item in v:
                    # Convert each item in the list
                    if hasattr(item, 'tolist'):
                        # For numpy arrays
                        serializable_history[k].append(item.tolist())
                    elif hasattr(item, 'item'):
                        # For scalar numpy or torch values
                        serializable_history[k].append(item.item())
                    else:
                        # For other types, convert to float if numeric
                        try:
                            serializable_history[k].append(float(item))
                        except (TypeError, ValueError):
                            # If conversion fails, keep original
                            serializable_history[k].append(item)
            else:
                # Handle non-list values
                serializable_history[k] = v
                
        json.dump(serializable_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save throughput metrics
    throughput_path = os.path.join(output_dir, 'throughput_metrics.json')
    with open(throughput_path, 'w') as f:
        # Create serializable metrics by converting numpy arrays to lists
        throughput_metrics = {
            'avg_train_throughput': float(avg_train_throughput),
            'avg_val_throughput': float(avg_val_throughput),
            'train_throughput_history': [float(t) for t in throughput_history['train']],
            'val_throughput_history': [float(t) for t in throughput_history['val']],
            'total_training_time': float(total_time),
            'avg_epoch_time': float(avg_epoch_time),
            'timing_breakdown': {k: [float(v) for v in vals] for k, vals in timing_breakdown.items()},
            'gpu_stats': gpu_stats if device.type == 'cuda' else None
        }
        
        # Convert val_metrics to JSON-serializable format
        serializable_val_metrics = {}
        for key, value in val_metrics.items():
            if key == 'class_accuracy' or key == 'unique_classes':
                # Convert numpy arrays to lists
                serializable_val_metrics[key] = value.tolist() if hasattr(value, 'tolist') else [float(v) for v in value]
            else:
                # Convert scalar values to float
                serializable_val_metrics[key] = float(value)
        
        throughput_metrics['val_metrics'] = serializable_val_metrics
        
        json.dump(throughput_metrics, f, indent=2)
    logger.info(f"Throughput metrics saved to {throughput_path}")
    
    # Plot trading metrics if available
    if 'precision' in serializable_history and 'recall' in serializable_history and 'f1' in serializable_history:
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(serializable_history['precision']) + 1)
        plt.plot(epochs, serializable_history['precision'], label='Precision')
        plt.plot(epochs, serializable_history['recall'], label='Recall')
        plt.plot(epochs, serializable_history['f1'], label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Trading Metrics Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'visualizations/trading_metrics.png'))
        
        # Plot class-specific accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, serializable_history['class_accuracy_buy'], label='Buy Accuracy')
        plt.plot(epochs, serializable_history['class_accuracy_hold'], label='Hold Accuracy')
        plt.plot(epochs, serializable_history['class_accuracy_sell'], label='Sell Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Class-Specific Accuracy Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'visualizations/class_accuracy.png'))
        
        # Plot trading score
        if 'trading_score' in serializable_history:
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, serializable_history['trading_score'])
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Trading Score Over Time')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'visualizations/trading_score.png'))
    
    return model, history

def visualize_throughput(metrics_path, output_dir):
    """
    Visualize throughput metrics from training
    
    Parameters:
    - metrics_path: Path to the throughput metrics JSON file
    - output_dir: Directory to save the plots
    """
    try:
        import matplotlib.pyplot as plt
        
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot throughput history
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['train_throughput_history'], label='Training')
        plt.plot(metrics['val_throughput_history'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Samples/second')
        plt.title('Training Throughput')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'throughput_history.png'))
        
        # Plot timing breakdown
        if 'timing_breakdown' in metrics:
            # Calculate average time per category
            avg_timing = {
                k.replace('train_', ''): sum(v) / len(v)
                for k, v in metrics['timing_breakdown'].items()
            }
            
            # Plot pie chart of time distribution
            plt.figure(figsize=(10, 10))
            plt.pie(
                avg_timing.values(),
                labels=avg_timing.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            plt.axis('equal')
            plt.title('Training Time Breakdown')
            plt.savefig(os.path.join(output_dir, 'time_breakdown.png'))
        
        # Plot GPU metrics if available
        if metrics.get('gpu_stats') and isinstance(metrics['gpu_stats'], dict):
            plt.figure(figsize=(10, 6))
            
            # If we have memory usage data
            if 'avg_memory_usage_mb' in metrics['gpu_stats']:
                plt.bar(['Average', 'Peak'], 
                       [metrics['gpu_stats']['avg_memory_usage_mb'], 
                        metrics['gpu_stats']['peak_memory_usage_mb']])
                plt.ylabel('Memory Usage (MB)')
                plt.title('GPU Memory Usage')
                plt.savefig(os.path.join(output_dir, 'gpu_memory.png'))
                
        # Load training history to plot learning rate
        history_path = os.path.join(os.path.dirname(metrics_path), 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            # Plot learning rate if available
            if 'learning_rates' in history:
                plt.figure(figsize=(10, 6))
                plt.plot(history['learning_rates'])
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.yscale('log')  # Log scale for better visualization
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        
        logger.info(f"Throughput visualizations saved to {output_dir}")
    except ImportError:
        logger.warning("Matplotlib not installed, skipping visualization")
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")

def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train time series model for cryptocurrency trading")
    parser.add_argument("--config", type=str, default="crypto_trading_model/config/time_series_config.json",
                      help="Path to configuration file")
    parser.add_argument("--visualize", action="store_true", 
                      help="Generate visualizations of throughput metrics")
    parser.add_argument("--disable_mp", action="store_true",
                      help="Disable multiprocessing for data loading")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Force single-process mode on Windows with CuPy
    if sys.platform.startswith('win') and HAS_CUPY:
        # Set environment variable for PyTorch dataloader
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        # Set global flag to disable multiprocessing in DataLoader
        args.disable_mp = True
        logger.warning("Windows with CuPy detected: multiprocessing disabled to avoid pickling errors")
    
    # Train the model
    model, history = train_model(args.config, disable_mp=args.disable_mp)
    
    # Generate visualizations if requested
    if args.visualize:
        config = load_config(args.config)
        output_dir = config['data']['output_dir']
        metrics_path = os.path.join(output_dir, 'throughput_metrics.json')
        visualize_throughput(metrics_path, os.path.join(output_dir, 'visualizations'))
    
    logger.info("Model training complete.")

if __name__ == "__main__":
    main() 