#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Lightning implementation of the LSTM-based trading model
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F

# Import from local modules
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.models.time_series.trainer import TimeSeriesDataset, collate_fn

# Set up logging - cleaned version, no null bytes
logger = logging.getLogger(__name__)

class LightningTimeSeriesModel(pl.LightningModule):
    """
    PyTorch Lightning implementation of the LSTM-based trading model
    """
    
    def __init__(
        self,
        input_dims: dict,
        hidden_dims: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        attention: bool = True,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_batch_norm: bool = True,
        use_residual: bool = True,
        embedding_dim: int = None,
        class_weights: list = None,
        warm_up_steps: int = 100,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 10,
        mixup_alpha: float = 0.2,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        """
        Initialize the model
        
        Parameters:
        -----------
        input_dims : dict
            Dictionary of input dimensions for each timeframe
        hidden_dims : int
            Size of hidden dimensions in LSTM
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        attention : bool
            Whether to use attention mechanism
        num_classes : int
            Number of output classes
        learning_rate : float
            Learning rate for optimization
        weight_decay : float
            Weight decay for regularization
        use_batch_norm : bool
            Whether to use batch normalization
        use_residual : bool
            Whether to use residual connections
        embedding_dim : int
            Dimension for initial feature embedding or None
        class_weights : list
            Optional weights for each class to handle imbalance
        warm_up_steps : int
            Number of warmup steps for learning rate scheduler
        lr_scheduler_factor : float
            Factor to reduce learning rate by on plateau
        lr_scheduler_patience : int
            Patience for learning rate scheduler
        mixup_alpha : float
            Alpha parameter for mixup data augmentation or 0 to disable
        use_focal_loss : bool
            Whether to use focal loss instead of cross entropy
        focal_gamma : float
            Gamma parameter for focal loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the PyTorch model
        self.model = MultiTimeframeModel(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            attention=attention,
            num_classes=num_classes,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            embedding_dim=embedding_dim
        )
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.warm_up_steps = warm_up_steps
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        
        # Setup loss function with class weighting if provided
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
            
        # Track metrics for each class
        self.class_names = ['Sell', 'Hold', 'Buy'] if num_classes == 3 else [f'Class_{i}' for i in range(num_classes)]
        self.train_metrics = pl.metrics.MetricCollection({
            'accuracy': pl.metrics.Accuracy(num_classes=num_classes, average='macro'),
            'precision': pl.metrics.Precision(num_classes=num_classes, average='macro'),
            'recall': pl.metrics.Recall(num_classes=num_classes, average='macro'),
            'f1': pl.metrics.F1Score(num_classes=num_classes, average='macro')
        })
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        
        # Initialize loss function
        self._init_loss_function()
        
    def _init_loss_function(self):
        """Initialize the appropriate loss function based on configuration"""
        if self.use_focal_loss:
            def focal_loss(logits, target):
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=1)
                # Get probability of correct class
                pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
                # Apply weights if provided
                if self.class_weights is not None:
                    weights = self.class_weights.to(target.device)
                    class_weights = weights.gather(0, target)
                    loss = -class_weights * (1 - pt) ** self.focal_gamma * torch.log(pt + 1e-10)
                else:
                    loss = -(1 - pt) ** self.focal_gamma * torch.log(pt + 1e-10)
                return loss.mean()
            
            self.criterion = focal_loss
        else:
            # Standard cross entropy with optional class weights
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights if self.class_weights is not None else None
            )
    
    def forward(self, inputs):
        """Forward pass"""
        return self.model(inputs)
    
    def mixup_data(self, inputs, labels):
        """Apply mixup data augmentation"""
        if self.mixup_alpha <= 0 or not self.training:
            return inputs, labels, labels, 1.0
        
        # Generate mixup coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Get batch size
        batch_size = labels.size(0)
        
        # Generate permutation
        index = torch.randperm(batch_size).to(labels.device)
        
        # Mix inputs for each timeframe
        mixed_inputs = {}
        for tf, tensor in inputs.items():
            mixed_inputs[tf] = lam * tensor + (1 - lam) * tensor[index]
        
        # Return mixed inputs and labels
        return mixed_inputs, labels, labels[index], lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Apply mixup to the criterion"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def training_step(self, batch, batch_idx):
        """Training step with mixup data augmentation and improved logging"""
        # Remove label from inputs
        labels = batch.pop('label')
        
        # Apply mixup if enabled
        if self.mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = self.mixup_data(batch, labels)
            outputs = self(inputs)
            loss = self.mixup_criterion(outputs, labels_a, labels_b, lam)
            # For metrics, use the dominant label
            _, predicted = torch.max(outputs, 1)
            labels_for_metrics = labels_a  # Use primary labels for metrics
        else:
            # Standard forward pass
            outputs = self(batch)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            labels_for_metrics = labels
        
        # Update and log metrics
        metrics = self.train_metrics(predicted, labels_for_metrics)
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        
        # Track per-class accuracy
        for i, class_name in enumerate(self.class_names):
            class_mask = labels_for_metrics == i
            if class_mask.sum() > 0:
                class_correct = (predicted[class_mask] == labels_for_metrics[class_mask]).float().mean()
                self.log(f'train_acc_{class_name}', class_correct, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with improved metrics"""
        # Remove label from inputs
        labels = batch.pop('label')
        
        # Forward pass
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        
        # Calculate and log metrics
        _, predicted = torch.max(outputs, 1)
        metrics = self.val_metrics(predicted, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        
        # Track per-class accuracy
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if class_mask.sum() > 0:
                class_correct = (predicted[class_mask] == labels[class_mask]).float().mean()
                self.log(f'val_acc_{class_name}', class_correct, prog_bar=False)
                
        # Log confusion matrix every N validation steps
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            # Create confusion matrix
            conf_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=self.device)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
                
            # Convert to numpy for logging
            conf_matrix_np = conf_matrix.cpu().numpy()
            
            # Log normalized confusion matrix
            cm_norm = conf_matrix_np / (conf_matrix_np.sum(axis=1, keepdims=True) + 1e-10)
            
            for i, class_name_i in enumerate(self.class_names):
                for j, class_name_j in enumerate(self.class_names):
                    self.logger.experiment.add_scalar(
                        f'confusion_matrix/val_{class_name_i}_predicted_as_{class_name_j}',
                        cm_norm[i, j],
                        self.current_epoch
                    )
        
        return {'val_loss': loss, 'val_metrics': metrics}
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        # Remove label from inputs
        labels = batch.pop('label')
        
        # Forward pass
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        
        # Calculate metrics
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        
        # Store predictions for later analysis
        return {
            'test_loss': loss,
            'test_acc': accuracy,
            'predictions': predicted.detach(),
            'labels': labels.detach(),
            'outputs': outputs.detach()
        }
    
    def configure_optimizers(self):
        """Configure optimizers with warmup and cosine scheduling"""
        # Adam optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True
        )
        
        # Create scheduler with warmup
        schedulers = []
        
        # Learning rate warmup
        if self.warm_up_steps > 0:
            warmup_scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lambda epoch: min(1.0, epoch / self.warm_up_steps)
                ),
                'interval': 'step',
                'frequency': 1,
                'name': 'warmup'
            }
            schedulers.append(warmup_scheduler)
        
        # LR scheduler for after warmup
        main_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
            "name": "plateau"
        }
        schedulers.append(main_scheduler)
        
        return [optimizer], schedulers

def train_lightning_model(
    config_path: str,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
    verbose: bool = False,
    trainer_class=None
):
    """
    Train the model using PyTorch Lightning
    
    Parameters:
    - config_path: Path to configuration JSON file
    - max_epochs: Maximum number of epochs to train
    - early_stopping_patience: Patience for early stopping
    - verbose: Whether to enable verbose logging
    - trainer_class: Optional custom trainer class to use instead of pl.Trainer
    
    Returns:
    - model: Trained model
    - trainer: Lightning trainer
    """
    # Enable tensor cores for better performance on CUDA devices
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        logger.info("Set float32 matmul precision to 'high' for tensor cores")
    
    logger.info("Starting PyTorch Lightning model training")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("Configuration loaded successfully")
    
    # Extract model parameters from config
    model_params = config.get('model', {})
    hidden_dims = model_params.get('hidden_dims', 128)
    num_layers = model_params.get('num_layers', 2)
    dropout = model_params.get('dropout', 0.2)
    bidirectional = model_params.get('bidirectional', True)
    attention = model_params.get('attention', True)
    num_classes = model_params.get('num_classes', 3)
    
    # Get training parameters
    training_params = config.get('training', {})
    learning_rate = training_params.get('learning_rate', 0.001)
    weight_decay = training_params.get('weight_decay', 1e-5)
    
    # Setup data
    data_params = config.get('data', {})
    train_data_path = data_params.get('train_data_path', '')
    val_data_path = data_params.get('val_data_path', '')
    batch_size = config.get('training', {}).get('batch_size', 32)
    
    if not train_data_path or not val_data_path:
        logger.warning("No data paths provided in config, skipping data loading")
        return None, None
    
    # Load data
    logger.info(f"Loading data from {train_data_path} and {val_data_path}")
    try:
        import h5py
        import numpy as np
        from torch.utils.data import TensorDataset, DataLoader
        
        # Load training data
        with h5py.File(train_data_path, 'r') as f:
            # Print file structure only in verbose mode
            if verbose:
                logger.info(f"HDF5 file structure for {train_data_path}:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        logger.info(f"  Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        logger.info(f"  Group: {name}")
                f.visititems(print_structure)
            
            # The data is organized as timeframes with tables
            # Extract the data from the main tables, not the _i_table indices
            train_data = {}
            timeframes = []
            
            for tf in f.keys():
                if isinstance(f[tf], h5py.Group) and 'table' in f[tf]:
                    timeframes.append(tf)
                    if verbose:
                        logger.info(f"Loading timeframe: {tf}")
                    try:
                        # Access the main table, which contains all features as a structured array
                        table_data = f[tf]['table'][:]
                        if verbose:
                            logger.info(f"  Found table with shape {table_data.shape} and type {table_data.dtype}")
                        
                        # Convert structured array to separate tensors
                        for field_name in table_data.dtype.names:
                            if field_name != 'index':  # Skip the index field
                                feature_name = f"{tf}_{field_name}"
                                train_data[feature_name] = torch.tensor(
                                    table_data[field_name], dtype=torch.float32
                                )
                                if verbose:
                                    logger.info(f"    Added feature: {feature_name}, Shape: {train_data[feature_name].shape}")
                    except Exception as e:
                        logger.error(f"Error loading timeframe {tf}: {str(e)}")
            
            # Create labels from price direction in the shortest timeframe
            # Assuming labels will be based on price_direction field of the first timeframe
            if timeframes:
                primary_tf = timeframes[0]
                try:
                    table_data = f[primary_tf]['table'][:]
                    if 'price_direction' in table_data.dtype.names:
                        # Extract price direction labels and ensure they're valid for num_classes
                        raw_labels = table_data['price_direction']
                        # Clip labels to be in valid range [0, num_classes-1]
                        valid_labels = np.clip(raw_labels, 0, num_classes - 1)
                        train_labels = torch.tensor(valid_labels, dtype=torch.long)
                        if np.any(raw_labels != valid_labels):
                            logger.warning(f"Some training labels were outside valid range [0, {num_classes-1}] and have been clipped")
                        if verbose:
                            logger.info(f"Using price_direction from {primary_tf} as labels, Shape: {train_labels.shape}")
                    else:
                        logger.warning(f"No price_direction field found in {primary_tf}, generating dummy labels")
                        train_labels = torch.zeros(len(table_data), dtype=torch.long)
                except Exception as e:
                    logger.error(f"Error loading labels: {str(e)}")
                    train_labels = torch.zeros(len(next(iter(train_data.values()))), dtype=torch.long)
            else:
                logger.warning("No valid timeframes found, generating dummy labels")
                train_labels = torch.zeros(len(next(iter(train_data.values())) if train_data else 0), dtype=torch.long)
            
            logger.info(f"Created training dataset with {len(train_data)} features and {len(train_labels)} samples")
        
        # Load validation data with the same approach
        with h5py.File(val_data_path, 'r') as f:
            # Print file structure only in verbose mode
            if verbose:
                logger.info(f"HDF5 file structure for {val_data_path}:")
                f.visititems(print_structure)
            
            # Extract the data from the tables
            val_data = {}
            timeframes = []
            
            for tf in f.keys():
                if isinstance(f[tf], h5py.Group) and 'table' in f[tf]:
                    timeframes.append(tf)
                    if verbose:
                        logger.info(f"Loading timeframe: {tf}")
                    try:
                        # Access the main table, which contains all features
                        table_data = f[tf]['table'][:]
                        if verbose:
                            logger.info(f"  Found table with shape {table_data.shape} and type {table_data.dtype}")
                        
                        # Convert structured array to separate tensors
                        for field_name in table_data.dtype.names:
                            if field_name != 'index':  # Skip the index field
                                feature_name = f"{tf}_{field_name}"
                                val_data[feature_name] = torch.tensor(
                                    table_data[field_name], dtype=torch.float32
                                )
                                if verbose:
                                    logger.info(f"    Added feature: {feature_name}, Shape: {val_data[feature_name].shape}")
                    except Exception as e:
                        logger.error(f"Error loading timeframe {tf}: {str(e)}")
            
            # Create labels from price direction as we did for training data
            if timeframes:
                primary_tf = timeframes[0]
                try:
                    table_data = f[primary_tf]['table'][:]
                    if 'price_direction' in table_data.dtype.names:
                        # Extract price direction labels and ensure they're valid for num_classes
                        raw_labels = table_data['price_direction']
                        # Clip labels to be in valid range [0, num_classes-1]
                        valid_labels = np.clip(raw_labels, 0, num_classes - 1)
                        val_labels = torch.tensor(valid_labels, dtype=torch.long)
                        if np.any(raw_labels != valid_labels):
                            logger.warning(f"Some validation labels were outside valid range [0, {num_classes-1}] and have been clipped")
                        if verbose:
                            logger.info(f"Using price_direction from {primary_tf} as labels, Shape: {val_labels.shape}")
                    else:
                        logger.warning(f"No price_direction field found in {primary_tf}, generating dummy labels")
                        val_labels = torch.zeros(len(table_data), dtype=torch.long)
                except Exception as e:
                    logger.error(f"Error loading labels: {str(e)}")
                    val_labels = torch.zeros(len(next(iter(val_data.values()))), dtype=torch.long)
            else:
                logger.warning("No valid timeframes found, generating dummy labels")
                val_labels = torch.zeros(len(next(iter(val_data.values())) if val_data else 0), dtype=torch.long)
            
            logger.info(f"Created validation dataset with {len(val_data)} features and {len(val_labels)} samples")
        
        # Create datasets
        class TimeSeriesDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
                
                # Group features by timeframe
                self.timeframes = {}
                for key in data.keys():
                    if '_' in key:
                        tf, feature = key.split('_', 1)
                        if tf not in self.timeframes:
                            self.timeframes[tf] = []
                        self.timeframes[tf].append(feature)
                
                # Find the minimum length across all tensors
                self.data_length = len(labels)
                for key, tensor in data.items():
                    if len(tensor) < self.data_length:
                        if verbose:
                            logger.warning(f"Feature {key} has length {len(tensor)}, shorter than labels with length {self.data_length}.")
                        self.data_length = min(self.data_length, len(tensor))
                
                # Truncate all tensors to the minimum length
                for key, tensor in data.items():
                    if len(tensor) > self.data_length:
                        if verbose:
                            logger.warning(f"Feature {key} has length {len(tensor)}, truncating to {self.data_length}.")
                        self.data[key] = tensor[:self.data_length]
                
                # Truncate labels to match
                if len(self.labels) > self.data_length:
                    if verbose:
                        logger.warning(f"Labels has length {len(self.labels)}, truncating to {self.data_length}.")
                    self.labels = self.labels[:self.data_length]
                
                if verbose:
                    logger.info(f"Dataset organized with timeframes: {list(self.timeframes.keys())} and {self.data_length} samples")
            
            def __len__(self):
                return self.data_length
            
            def __getitem__(self, idx):
                # Ensure index is in bounds
                if idx >= self.data_length:
                    raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.data_length}")
                
                # Organize data by timeframe for model consumption
                sample = {}
                for tf in self.timeframes:
                    # Stack features into a tensor of shape (num_features,)
                    try:
                        features = torch.stack([
                            self.data[f"{tf}_{feature}"][idx] 
                            for feature in self.timeframes[tf]
                        ])
                        
                        # LSTM expects input shape (batch_size, seq_len, input_size)
                        # For a single sample, we need shape (seq_len, input_size)
                        # Since we only have one time step per sample, seq_len=1
                        # input_size = number of features
                        # The features tensor is currently shape (num_features,)
                        # We need to reshape to (1, num_features) for the LSTM
                        sample[tf] = features.unsqueeze(0)  # Add sequence dimension -> (1, num_features)
                    except IndexError as e:
                        logger.error(f"Index error for {tf} at idx={idx}: {str(e)}")
                        raise
                
                sample['label'] = self.labels[idx]
                return sample
        
        train_dataset = TimeSeriesDataset(train_data, train_labels)
        val_dataset = TimeSeriesDataset(val_data, val_labels)
        
        # Create DataLoaders with safer settings
        # Use fewer workers to reduce likelihood of race conditions
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced from 4
            pin_memory=True,
            drop_last=True  # Drop the last incomplete batch
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # Reduced from 4
            pin_memory=True,
            drop_last=True  # Drop the last incomplete batch
        )
        
        logger.info(f"Created data loaders with batch size {batch_size}")
        
        # Determine input dimensions from actual data
        input_dims = {}
        for tf in train_dataset.timeframes:
            # Number of features for this timeframe
            input_dims[tf] = len(train_dataset.timeframes[tf])
        
        logger.info(f"Computed input dimensions from data: {input_dims}")
        
        # Create Lightning model
        model = LightningTimeSeriesModel(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            attention=attention,
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    
    # Setup callbacks with better default values
    os.makedirs(os.path.join('models', 'checkpoints'), exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min',
            min_delta=0.001,  # Small improvement threshold
            verbose=True
        ),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join('models', 'checkpoints'),
            filename='lstm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min',
            verbose=True
        )
    ]
    
    # Setup logger
    os.makedirs(os.path.join('logs', 'lightning_logs'), exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join('logs'),
        name='lightning_logs'
    )
    
    # Setup trainer - use custom trainer class if provided
    trainer_kwargs = {
        'max_epochs': max_epochs,
        'callbacks': callbacks,
        'logger': tb_logger,
        'log_every_n_steps': 50,  # Reduced logging frequency
        'accelerator': 'auto',  # Use GPU if available
        'enable_progress_bar': True,
        'enable_model_summary': True
    }
    
    if trainer_class:
        trainer = trainer_class(**trainer_kwargs)
    else:
        trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    logger.info("Starting training")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    logger.info("Training complete")
    
    return model, trainer
