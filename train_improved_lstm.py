#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the improved LSTM model with enhanced multi-timeframe features and labeling.
"""

import os
import argparse
import logging
import json
import numpy as np
import torch
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from crypto_trading_model.lstm_lightning import LightningTimeSeriesModel, train_lightning_model
from crypto_trading_model.data_processing.feature_engineering import calculate_multi_timeframe_signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_dataset(data_dir, regenerate_labels=True, threshold_pct=0.015):
    """
    Prepare the dataset for training with optional label regeneration.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    regenerate_labels : bool
        Whether to regenerate labels using the multi-timeframe approach
    threshold_pct : float
        Threshold percentage for price movement to generate labels
        
    Returns:
    --------
    dict
        Dictionary containing paths to the processed data files
    """
    # Paths to the data files
    train_path = os.path.join(data_dir, 'train_data.h5')
    val_path = os.path.join(data_dir, 'val_data.h5')
    test_path = os.path.join(data_dir, 'test_data.h5')
    
    if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        logger.error(f"Data files not found in {data_dir}. Please generate the data first.")
        return None
    
    # If we need to regenerate labels
    if regenerate_labels:
        logger.info("Regenerating labels using multi-timeframe approach...")
        
        # Process each split
        for split_name, file_path in [('train', train_path), ('val', val_path), ('test', test_path)]:
            # Load data from HDF5
            with pd.HDFStore(file_path, mode='r') as store:
                # Extract timeframes
                timeframes = [key.strip('/') for key in store.keys()]
                
                # Create a dictionary to hold dataframes for each timeframe
                data_dict = {}
                for tf in timeframes:
                    data_dict[tf] = store[f'/{tf}']
                
                # Calculate multi-timeframe signals
                lookforward_periods = {'15m': 16, '4h': 4, '1d': 1}
                multi_tf_signals = calculate_multi_timeframe_signal(
                    data_dict,
                    primary_tf='15m',
                    threshold_pct=threshold_pct,
                    lookforward_periods=lookforward_periods
                )
                
                # Map signals to 0-2 range for compatibility with standard class indices
                signal_mapping = {-1: 0, 0: 1, 1: 2}
                mapped_signals = multi_tf_signals.map(signal_mapping)
                
                # Update labels in the datasets
                with h5py.File(file_path, 'r+') as f:
                    # Update primary timeframe
                    if '15m' in f:
                        try:
                            # Get table data
                            table_data = f['15m']['table']
                            
                            # Create a copy of the data
                            data_copy = table_data[:]
                            
                            # Update price_direction field
                            data_copy['price_direction'] = mapped_signals.values
                            
                            # Replace the dataset
                            del f['15m']['table']
                            f['15m'].create_dataset('table', data=data_copy)
                            
                            logger.info(f"Updated price_direction in {split_name}/15m")
                        except Exception as e:
                            logger.error(f"Error updating 15m data: {str(e)}")
                    
                    # Update other timeframes by downsampling
                    for tf, step in [('4h', 16), ('1d', 96)]:
                        if tf in f:
                            try:
                                # Get table data
                                table_data = f[tf]['table']
                                
                                # Create a copy of the data
                                data_copy = table_data[:]
                                
                                # Get indices for downsampling
                                tf_indices = list(range(0, len(mapped_signals), step))
                                if len(tf_indices) > len(data_copy):
                                    tf_indices = tf_indices[:len(data_copy)]
                                
                                # Downsample signals
                                downsampled_signals = mapped_signals.iloc[tf_indices].values
                                
                                # Update price_direction field (only up to the available length)
                                length = min(len(data_copy), len(downsampled_signals))
                                data_copy['price_direction'][:length] = downsampled_signals[:length]
                                
                                # Replace the dataset
                                del f[tf]['table']
                                f[tf].create_dataset('table', data=data_copy)
                                
                                logger.info(f"Updated price_direction in {split_name}/{tf}")
                            except Exception as e:
                                logger.error(f"Error updating {tf} data: {str(e)}")
    
    # Return paths to the data files
    return {
        'train_path': train_path,
        'val_path': val_path,
        'test_path': test_path
    }

def analyze_label_distribution(data_path):
    """
    Analyze the distribution of labels in the dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file
        
    Returns:
    --------
    dict
        Dictionary containing label distribution and class weights
    """
    # Initialize results
    results = {'distribution': {}, 'class_weights': {}}
    
    # Extract labels from each timeframe
    with h5py.File(data_path, 'r') as f:
        for tf in f.keys():
            if 'table' in f[tf]:
                # Get table data
                table_data = f[tf]['table'][:]
                
                # Extract price_direction field
                if 'price_direction' in table_data.dtype.names:
                    labels = table_data['price_direction']
                    
                    # Count occurrences
                    values, counts = np.unique(labels, return_counts=True)
                    
                    # Store distribution
                    distribution = {int(val): int(count) for val, count in zip(values, counts)}
                    results['distribution'][tf] = distribution
                    
                    # Calculate class weights
                    try:
                        class_weights = compute_class_weight(
                            class_weight='balanced',
                            classes=np.unique(labels),
                            y=labels
                        )
                        results['class_weights'][tf] = {int(cls): float(weight) for cls, weight in zip(np.unique(labels), class_weights)}
                    except Exception as e:
                        logger.warning(f"Could not compute class weights for {tf}: {str(e)}")
    
    return results

def train_model(data_paths, config_path=None, output_dir='models/lstm_improved', max_epochs=100):
    """
    Train the improved LSTM model.
    
    Parameters:
    -----------
    data_paths : dict
        Dictionary containing paths to the data files
    config_path : str
        Path to the configuration file
    output_dir : str
        Directory to save model and logs
    max_epochs : int
        Maximum number of epochs for training
        
    Returns:
    --------
    LightningTimeSeriesModel
        Trained model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        # Default configuration
        config = {
            "model_params": {
                "hidden_dims": 256,
                "num_layers": 3,
                "dropout": 0.3,
                "bidirectional": True,
                "attention": True,
                "use_batch_norm": True,
                "use_residual": True,
                "embedding_dim": 64
            },
            "training_params": {
                "learning_rate": 0.0005,
                "weight_decay": 1e-4,
                "batch_size": 128,
                "early_stopping_patience": 20,
                "warm_up_steps": 100,
                "lr_scheduler_factor": 0.5,
                "lr_scheduler_patience": 10,
                "mixup_alpha": 0.2,
                "use_focal_loss": True,
                "focal_gamma": 2.0
            }
        }
        
        # Save configuration
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created new configuration and saved to {config_path}")
    
    # Analyze label distribution
    logger.info("Analyzing label distribution in the training data...")
    label_analysis = analyze_label_distribution(data_paths['train_path'])
    
    # Save label analysis
    analysis_path = os.path.join(output_dir, 'label_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(label_analysis, f, indent=2)
    logger.info(f"Saved label distribution analysis to {analysis_path}")
    
    # Extract class weights from the primary timeframe
    class_weights = None
    if '15m' in label_analysis['class_weights']:
        class_weights_dict = label_analysis['class_weights']['15m']
        class_weights = [class_weights_dict.get(i, 1.0) for i in range(3)]
        logger.info(f"Using class weights: {class_weights}")
    
    # Set up training
    logger.info("Setting up model training...")
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['training_params']['early_stopping_patience'],
        mode='min'
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='lstm-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create logger
    tensorboard_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='lstm_improved'
    )
    
    # Train the model
    logger.info("Starting model training...")
    model, trainer = train_lightning_model(
        config_path=config_path,
        max_epochs=max_epochs,
        early_stopping_patience=config['training_params']['early_stopping_patience'],
        verbose=True,
        callbacks=[early_stopping, checkpoint_callback, lr_monitor],
        logger=tensorboard_logger,
        class_weights=class_weights,
        **config['training_params']
    )
    
    # Save the model
    model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(model.model.state_dict(), model_path)
    logger.info(f"Saved final model to {model_path}")
    
    return model

def main():
    """Main entry point for training the improved LSTM model."""
    parser = argparse.ArgumentParser(description="Train the improved LSTM model for cryptocurrency trading")
    parser.add_argument("--data_dir", type=str, default="data/synthetic",
                      help="Directory containing the data files")
    parser.add_argument("--config", type=str, default=None,
                      help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="models/lstm_improved",
                      help="Directory to save model and logs")
    parser.add_argument("--max_epochs", type=int, default=100,
                      help="Maximum number of epochs for training")
    parser.add_argument("--regenerate_labels", action="store_true",
                      help="Whether to regenerate labels using the multi-timeframe approach")
    parser.add_argument("--threshold_pct", type=float, default=0.015,
                      help="Threshold percentage for price movement to generate labels")
    
    args = parser.parse_args()
    
    # Prepare the dataset
    data_paths = prepare_dataset(
        args.data_dir,
        regenerate_labels=args.regenerate_labels,
        threshold_pct=args.threshold_pct
    )
    
    if data_paths:
        # Train the model
        model = train_model(
            data_paths,
            config_path=args.config,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs
        )
        
        logger.info("Model training complete.")
    else:
        logger.error("Could not prepare dataset. Exiting.")

if __name__ == "__main__":
    main() 