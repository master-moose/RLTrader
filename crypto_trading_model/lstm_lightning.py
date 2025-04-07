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

# Import from local modules
from crypto_trading_model.models.time_series.model import MultiTimeframeModel
from crypto_trading_model.models.time_series.trainer import TimeSeriesDataset, collate_fn

# Set up logging
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
        weight_decay: float = 1e-5
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
            num_classes=num_classes
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, inputs):
        """Forward pass"""
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Remove label from inputs
        labels = batch.pop('label')
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        
        # Log metrics
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Remove label from inputs
        labels = batch.pop('label')
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        
        # Log metrics
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        # Remove label from inputs
        labels = batch.pop('label')
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        
        # Log metrics
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        
        return {'test_loss': loss, 'test_acc': accuracy}
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

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
    logger.info("Starting PyTorch Lightning model training")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Early stopping patience: {early_stopping_patience}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("Configuration loaded successfully")
    
    # Extract model parameters from config
    model_params = config.get('model_params', {})
    input_dims = model_params.get('input_dims', {
        '15m': 32,
        '4h': 32,
        '1d': 32
    })
    hidden_dims = model_params.get('hidden_dims', 128)
    num_layers = model_params.get('num_layers', 2)
    dropout = model_params.get('dropout', 0.2)
    bidirectional = model_params.get('bidirectional', True)
    attention = model_params.get('attention', True)
    num_classes = model_params.get('num_classes', 3)
    learning_rate = model_params.get('learning_rate', 0.001)
    weight_decay = model_params.get('weight_decay', 1e-5)
    
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
    
    # Setup data
    # Note: This is a placeholder. You'll need to implement the data loading
    # specific to your application
    
    # For example, assuming your config has data paths:
    data_params = config.get('data_params', {})
    train_data_path = data_params.get('train_data_path', '')
    val_data_path = data_params.get('val_data_path', '')
    batch_size = data_params.get('batch_size', 32)
    
    if train_data_path and val_data_path:
        # Load data
        logger.info(f"Loading data from {train_data_path} and {val_data_path}")
        # Implement your data loading logic here
        # ...
        
        # Create DataLoaders
        # train_dataloader = DataLoader(...)
        # val_dataloader = DataLoader(...)
    else:
        logger.warning("No data paths provided in config, skipping data loading")
        # In a real implementation, you would raise an error here
        return None, None
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=os.path.join('models', 'checkpoints'),
            filename='lstm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min'
        )
    ]
    
    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join('logs'),
        name='lightning_logs'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=10,
        accelerator='auto',  # Use GPU if available
    )
    
    # Train model
    logger.info("Starting training")
    # In a real implementation, you would uncomment this:
    # trainer.fit(model, train_dataloader, val_dataloader)
    
    logger.info("Training complete")
    
    return model, trainer
