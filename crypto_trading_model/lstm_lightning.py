#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch Lightning implementation of the LSTM-based trading model
"""

import os
import json
import time
import torch
import logging
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchmetrics
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logger = logging.getLogger(__name__)

class MultiTimeframeLSTMModule(pl.LightningModule):
    """
    PyTorch Lightning module for multi-timeframe LSTM model
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dims: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
        attention: bool = True,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        focal_loss_gamma: float = 2.0,
        lr_scheduler_patience: int = 15,
        lr_scheduler_factor: float = 0.5
    ):
        """
        Initialize the Lightning module
        
        Parameters:
        - input_dims: Dictionary mapping timeframe names to input feature dimensions
        - hidden_dims: Dimension of hidden layers
        - num_layers: Number of LSTM layers
        - dropout: Dropout probability
        - bidirectional: Whether to use bidirectional LSTM
        - attention: Whether to use attention mechanism
        - num_classes: Number of output classes (3 for buy/hold/sell)
        - learning_rate: Initial learning rate
        - weight_decay: L2 regularization strength
        - class_weights: Optional tensor of class weights for imbalanced datasets
        - focal_loss_gamma: Gamma parameter for focal loss (higher gives more weight to hard examples)
        - lr_scheduler_patience: Patience for learning rate scheduler
        - lr_scheduler_factor: Factor to reduce learning rate by
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.attention = attention
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.focal_loss_gamma = focal_loss_gamma
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        
        # Number of directions for LSTM (1 or 2 if bidirectional)
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
                dropout=self.dropout_rate if num_layers > 1 else 0
            )
        
        # Attention mechanism
        if attention:
            # Hidden dimension accounting for bidirectionality
            self.attn_hidden_dim = hidden_dims * self.num_directions
            
            # Query vectors for each timeframe
            self.query_vectors = torch.nn.ParameterDict()
            for tf in self.input_dims.keys():
                param = torch.nn.Parameter(torch.zeros(self.attn_hidden_dim))
                # Use normal_ for 1D tensors
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
                self.query_vectors[tf] = param
            
            # Attention projection
            self.attention_projection = torch.nn.Linear(
                self.attn_hidden_dim * len(self.input_dims), 
                self.attn_hidden_dim
            )
            self.attention_ln = torch.nn.LayerNorm(self.attn_hidden_dim)
        
        # Output dimension depends on whether we use attention
        output_dim = self.attn_hidden_dim if attention else self.attn_hidden_dim * len(self.input_dims)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(output_dim, hidden_dims)
        self.ln1 = torch.nn.LayerNorm(hidden_dims)
        
        self.fc2 = torch.nn.Linear(hidden_dims, hidden_dims // 2)
        self.ln2 = torch.nn.LayerNorm(hidden_dims // 2)
        
        self.fc3 = torch.nn.Linear(hidden_dims // 2, num_classes)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        
        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # F1 Score for each class
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=None)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=None)
        
        # Precision and Recall
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None)
        
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None)
        
        # Confusion matrix for more detailed analysis
        self.train_confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)
    
    def focal_loss(self, logits, targets):
        """
        Compute focal loss for addressing class imbalance
        
        Parameters:
        - logits: Model output logits, shape (batch_size, num_classes)
        - targets: Target labels, shape (batch_size,)
        
        Returns:
        - loss: Focal loss value
        """
        # Apply log_softmax to get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        
        # Gather log probabilities of target classes
        target_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term: (1 - pt)^gamma 
        focal_term = torch.exp(target_probs).neg().add(1).pow(self.focal_loss_gamma)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            alpha_weights = self.class_weights.gather(0, targets)
            focal_term = focal_term * alpha_weights
        
        # Calculate final loss
        loss = -focal_term * target_probs
        
        return loss.mean()
    
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
            
            for tf in self.input_dims.keys():
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
            combined = self.attention_ln(combined)
        else:
            # Simple concatenation of all timeframe encodings
            combined = torch.cat([encoded_timeframes[tf] for tf in self.input_dims.keys()], dim=1)
        
        # Apply clipping to prevent gradient explosion
        combined = torch.clamp(combined, -5, 5)
        
        # Pass through fully connected layers
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.nn.functional.relu(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.nn.functional.relu(x)
        
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """
        Compute appropriate loss based on configuration
        """
        if self.focal_loss_gamma > 0:
            # Use focal loss
            return self.focal_loss(logits, targets)
        elif self.class_weights is not None:
            # Use weighted cross entropy
            return torch.nn.functional.cross_entropy(logits, targets, weight=self.class_weights)
        else:
            # Use regular cross entropy
            return torch.nn.functional.cross_entropy(logits, targets)
    
    def training_step(self, batch, batch_idx):
        """
        Training step for Lightning
        """
        features, targets = batch
        
        # Apply mixup augmentation with 50% probability
        if np.random.random() < 0.5 and features[next(iter(features.keys()))].size(0) > 1:
            mixed_features, targets_a, targets_b, lam = self.mixup_data(features, targets)
            logits = self(mixed_features)
            
            # Mixup loss
            loss_a = self.compute_loss(logits, targets_a)
            loss_b = self.compute_loss(logits, targets_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            
            # For metrics, use the dominant class
            preds = torch.argmax(logits, dim=1)
            targets_for_metrics = targets_a  # Use dominant class for metrics
        else:
            logits = self(features)
            loss = self.compute_loss(logits, targets)
            preds = torch.argmax(logits, dim=1)
            targets_for_metrics = targets
        
        # Update metrics
        self.train_acc(preds, targets_for_metrics)
        self.train_f1(preds, targets_for_metrics)
        self.train_precision(preds, targets_for_metrics)
        self.train_recall(preds, targets_for_metrics)
        self.train_confmat(preds, targets_for_metrics)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        # Log learning rate
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for Lightning
        """
        features, targets = batch
        logits = self(features)
        
        # Compute validation loss
        val_loss = self.compute_loss(logits, targets)
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_confmat(preds, targets)
        
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        return val_loss
    
    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch
        """
        # Compute per-class F1 scores
        val_f1_values = self.val_f1.compute()
        
        # Log per-class F1 scores
        for i, f1 in enumerate(val_f1_values):
            self.log(f'val_f1_class_{i}', f1)
        
        # Log average F1 score
        self.log('val_f1_avg', val_f1_values.mean())
        
        # Get confusion matrix
        conf_matrix = self.val_confmat.compute()
        
        # Calculate class distribution in predictions
        class_distribution = conf_matrix.sum(dim=0) / conf_matrix.sum()
        
        # Log class distribution
        for i, dist in enumerate(class_distribution):
            self.log(f'val_class_dist_{i}', dist)
        
        # Compute class balance metric (max/min ratio of predicted class frequencies)
        non_zero_dist = class_distribution[class_distribution > 0]
        if len(non_zero_dist) > 1:
            balance_metric = non_zero_dist.min() / non_zero_dist.max()
            self.log('val_class_balance', balance_metric)
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler
        """
        # Adam optimizer with weight decay
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def mixup_data(self, x, y, alpha=0.2):
        """
        Create mixed samples for training
        
        Parameters:
        - x: Dictionary of features
        - y: Target labels
        - alpha: Mixup parameter (higher = more mixing)
        
        Returns:
        - mixed_x: Mixed features
        - y_a, y_b: Original labels
        - lam: Lambda parameter (mix ratio)
        """
        batch_size = y.size(0)
        
        # Sample lambda from Beta distribution
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
            
        lam = max(lam, 1-lam)  # Ensure lambda is at least 0.5
        
        # Create tensor version of lambda
        lam_tensor = torch.tensor(lam, device=y.device)
        
        # Generate permutation of indices
        index = torch.randperm(batch_size, device=y.device)
        
        # Mix features
        mixed_x = {}
        for key, tensor in x.items():
            mixed_x[key] = lam_tensor * tensor + (1 - lam_tensor) * tensor[index]
        
        # Return mixed features and original labels
        return mixed_x, y, y[index], lam_tensor


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for time series data
    """
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        balance_classes: bool = True,
        sequence_lengths: Optional[Dict[str, int]] = None,
        feature_columns: Optional[List[str]] = None,
        normalize: bool = True
    ):
        """
        Initialize the DataModule
        
        Parameters:
        - train_data_path: Path to training data HDF5 file
        - val_data_path: Path to validation data HDF5 file
        - batch_size: Batch size for dataloaders
        - num_workers: Number of workers for dataloaders
        - balance_classes: Whether to balance classes during training
        - sequence_lengths: Dictionary mapping timeframes to sequence lengths
        - feature_columns: List of columns to use as features
        - normalize: Whether to normalize features
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance_classes = balance_classes
        self.sequence_lengths = sequence_lengths or {}
        self.feature_columns = feature_columns
        self.normalize = normalize
        
        # These will be set in setup
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None
        self.class_counts = None
    
    def prepare_data(self):
        """
        Perform any downloads or preprocessing steps that should be done once
        """
        # Nothing to do here since data is assumed to be already prepared
        pass
    
    def setup(self, stage=None):
        """
        Load datasets
        
        Parameters:
        - stage: 'fit', 'validate', 'test', or 'predict'
        """
        from crypto_trading_model.dataset import TimeSeriesDataset
        
        # Load training dataset
        self.train_dataset = TimeSeriesDataset(
            data_path=self.train_data_path,
            sequence_lengths=self.sequence_lengths,
            feature_columns=self.feature_columns,
            normalize=self.normalize
        )
        
        # Load validation dataset
        self.val_dataset = TimeSeriesDataset(
            data_path=self.val_data_path,
            sequence_lengths=self.sequence_lengths,
            feature_columns=self.feature_columns,
            normalize=self.normalize,
            normalization_params=self.train_dataset.normalization_params  # Use same normalization as training
        )
        
        # Calculate class counts and weights if needed
        if self.balance_classes:
            # Count occurrences of each class
            classes = torch.tensor([y for _, y in self.train_dataset])
            class_counts = torch.bincount(classes)
            self.class_counts = class_counts.tolist()
            
            # Compute inverse frequency weights
            total_samples = float(len(self.train_dataset))
            num_classes = len(class_counts)
            
            # Weighted with squared inverse frequency
            class_weights = torch.zeros(num_classes)
            for i in range(num_classes):
                if class_counts[i] > 0:
                    # Use squared inverse frequency for stronger weighting
                    class_weights[i] = (total_samples / class_counts[i]) ** 2
                else:
                    class_weights[i] = 1.0
            
            # Normalize weights
            class_weights = class_weights * num_classes / class_weights.sum()
            
            # Store class weights
            self.class_weights = class_weights
    
    def train_dataloader(self):
        """
        Return the training dataloader
        """
        if self.balance_classes and self.class_counts:
            # Create weighted sampler for class balancing
            classes = torch.tensor([y for _, y in self.train_dataset])
            weights = torch.zeros_like(classes, dtype=torch.float)
            
            # Assign weight to each sample based on its class
            for i, cls in enumerate(classes):
                weights[i] = 1.0 / self.class_counts[cls.item()]
                
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            
            # Create weighted sampler
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, 
                num_samples=len(self.train_dataset), 
                replacement=True
            )
            
            # Create dataloader with sampler
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            # Create regular dataloader with shuffle
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
    
    def val_dataloader(self):
        """
        Return the validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def train_lightning_model(
    config_path: str,
    max_epochs: int = 200,
    early_stopping_patience: int = 30
):
    """
    Train the model using PyTorch Lightning
    
    Parameters:
    - config_path: Path to configuration JSON file
    - max_epochs: Maximum number of epochs to train
    - early_stopping_patience: Patience for early stopping
    
    Returns:
    - model: Trained model
    - trainer: Lightning trainer
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get data paths
    train_data_path = config.get('train_data_path', './data/synthetic/train_data.h5')
    val_data_path = config.get('val_data_path', './data/synthetic/val_data.h5')
    
    # Get model hyperparameters
    model_config = config.get('model', {})
    hidden_dims = model_config.get('hidden_dims', 128)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.4)
    bidirectional = model_config.get('bidirectional', True)
    attention = model_config.get('attention', True)
    
    # Get sequence lengths and features
    sequence_lengths = config.get('sequence_lengths', {})
    feature_columns = config.get('feature_columns', None)
    
    # Get training parameters
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    num_workers = config.get('num_workers', 4)
    
    # Create datamodule
    datamodule = TimeSeriesDataModule(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        balance_classes=True,
        sequence_lengths=sequence_lengths,
        feature_columns=feature_columns,
        normalize=True
    )
    
    # Set up datamodule to get feature dimensions
    datamodule.setup()
    
    # Calculate feature dimensions for each timeframe
    input_dims = {}
    # We need to inspect a sample from the dataset to get dimensions
    sample_features, _ = datamodule.train_dataset[0]
    
    for tf, tensor in sample_features.items():
        # Feature dimension is the last dimension of the tensor
        input_dims[tf] = tensor.shape[-1]
    
    # Create the model
    model = MultiTimeframeLSTMModule(
        input_dims=input_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        attention=attention,
        num_classes=3,  # Buy, hold, sell
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=datamodule.class_weights,
        focal_loss_gamma=2.0,
        lr_scheduler_patience=15,
        lr_scheduler_factor=0.5
    )
    
    # Set up callbacks
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            dirpath='./output/time_series/',
            filename='best_model-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        
        # Early stopping based on validation loss
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min'
        ),
        
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=pl.loggers.TensorBoardLogger('./logs/lightning_logs'),
        accelerator='auto',  # Automatically use GPU if available
        devices='auto',
        precision='32-true',  # Use 32-bit precision
        log_every_n_steps=10,
        default_root_dir='./output/time_series/'
    )
    
    # Start training
    trainer.fit(model, datamodule=datamodule)
    
    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path}")
        model = MultiTimeframeLSTMModule.load_from_checkpoint(best_model_path)
    
    # Save model
    model_path = os.path.join('./output/time_series/', 'model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/lightning_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='crypto_trading_model/config/time_series_config.json',
                       help='Path to configuration file')
    parser.add_argument('--max-epochs', type=int, default=200,
                      help='Maximum number of epochs to train')
    parser.add_argument('--patience', type=int, default=30,
                     help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Train model
    train_lightning_model(
        config_path=args.config,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience
    ) 