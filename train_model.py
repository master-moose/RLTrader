#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import tables
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('crypto_trading_model.training')

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
                 sequence_length: int = 50,
                 target_column: str = 'close',
                 feature_columns: list = None,
                 normalize: bool = True):
        """
        Initialize the dataset
        
        Parameters:
        - data_path: Path to the HDF5 file with time series data
        - timeframes: List of timeframes to include
        - sequence_length: Number of time steps to include in each sample
        - target_column: Column to use as the prediction target
        - feature_columns: List of columns to use as features (None = use all)
        - normalize: Whether to normalize the features
        """
        self.data_path = data_path
        self.timeframes = timeframes
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.normalize = normalize
        
        # Load data from HDF5 file
        self.data = {}
        timestamps = {}
        
        with pd.HDFStore(data_path, mode='r') as store:
            # Check available timeframes in the store
            available_timeframes = [key[1:] for key in store.keys()]  # Remove leading '/'
            logger.info(f"Available timeframes in {data_path}: {available_timeframes}")
            
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
        min_length = min(len(df) for df in self.data.values())
        self.valid_indices = range(sequence_length, min_length)
        logger.info(f"Loaded dataset with {len(self.valid_indices)} valid samples")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual index in the dataframes
        index = self.valid_indices[idx]
        
        # Extract sequences for each timeframe
        sequences = {}
        for tf, df in self.data.items():
            # Get the sequence
            sequence = df.iloc[index - self.sequence_length:index].copy()
            
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

def create_dataloaders(train_path, val_path, batch_size=32, **dataset_kwargs):
    """Create DataLoader objects for training and validation"""
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_path, **dataset_kwargs)
    val_dataset = TimeSeriesDataset(val_path, **dataset_kwargs)
    
    # Determine number of workers based on system CPU count
    num_workers = min(os.cpu_count() - 1, 16)  # Use n-1 CPUs, max 16
    logger.info(f"Using {num_workers} worker processes for data loading")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
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
            
            # Add GPU info if available
            gpu_info = ""
            if device.type == 'cuda':
                gpu_info = f", GPU Mem: {torch.cuda.memory_allocated(device) / (1024 * 1024):.1f}MB"
                
            logger.info(f"Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}, "
                       f"Acc={batch_correct/targets.size(0):.4f}, "
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

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
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
    
    # Calculate validation metrics
    avg_loss = val_loss / len(dataloader)
    accuracy = val_correct / val_total
    
    end_time = time.time()
    val_time = end_time - start_time
    
    # Calculate throughput
    throughput = total_samples / val_time
    logger.info(f"Validation throughput: {throughput:.2f} samples/second")
    
    return avg_loss, accuracy, val_time, throughput

def train_model(config_path="crypto_trading_model/config/time_series_config.json"):
    """Train the time series model using parameters from config file"""
    # Load configuration
    config = load_config(config_path)
    
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
    device = config['training']['device']
    
    # Determine data paths
    if use_synthetic:
        train_path = os.path.join(synthetic_path, 'train_data.h5')
        val_path = os.path.join(synthetic_path, 'val_data.h5')
    else:
        train_path = os.path.join(data_path, 'train_data.h5')
        val_path = os.path.join(data_path, 'val_data.h5')
    
    logger.info(f"Training with data: {train_path}, {val_path}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        timeframes=timeframes,
        sequence_length=seq_length
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
    
    # Setup device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA for training")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
    
    model.to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'],
        verbose=True
    )
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    
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
        'val_throughput': []
    }
    
    # Track timing of each phase
    timing_breakdown = defaultdict(list)
    
    # Record training start time
    training_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch}/{epochs}")
        
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
        val_loss, val_acc, val_time, val_throughput = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Time: {val_time:.2f} seconds, Val Throughput: {val_throughput:.2f} samples/second")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)
        history['train_throughput'].append(train_throughput)
        history['val_throughput'].append(val_throughput)
        
        # Update throughput history
        throughput_history['train'].append(train_throughput)
        throughput_history['val'].append(val_throughput)
        
        # Track total training time
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        else:
            no_improve_epochs += 1
            logger.info(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= patience:
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
        # Convert values to float for JSON serialization
        serializable_history = {
            k: [float(val) for val in v] 
            for k, v in history.items()
        }
        json.dump(serializable_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save throughput metrics
    throughput_path = os.path.join(output_dir, 'throughput_metrics.json')
    with open(throughput_path, 'w') as f:
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
        json.dump(throughput_metrics, f, indent=2)
    logger.info(f"Throughput metrics saved to {throughput_path}")
    
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
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Train the model
    model, history = train_model(args.config)
    
    # Generate visualizations if requested
    if args.visualize:
        config = load_config(args.config)
        output_dir = config['data']['output_dir']
        metrics_path = os.path.join(output_dir, 'throughput_metrics.json')
        visualize_throughput(metrics_path, os.path.join(output_dir, 'visualizations'))
    
    logger.info("Model training complete.")

if __name__ == "__main__":
    main() 