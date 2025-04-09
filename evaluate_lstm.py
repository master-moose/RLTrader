#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate the trained LSTM model on the test dataset.
"""

import os
import json
import logging
import argparse
import sys  # Add sys import for sys.platform check
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from torch.utils.data import DataLoader

# Import from local modules
from crypto_trading_model.lstm_lightning import train_lightning_model, LightningTimeSeriesModel

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model_dir, data_dir, batch_size=256, num_workers=None, model_path=None):
    """
    Evaluate the trained model on the test dataset.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the trained model and configuration
    data_dir : str
        Directory containing the data files
    batch_size : int
        Batch size for evaluation
    num_workers : int, optional
        Number of worker processes for data loading
    model_path : str, optional
        Path to the specific model file. If None, looks for 'final_model.pt' 
        in model_dir
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Paths to the model and configuration files
    local_config_path = os.path.join(model_dir, 'config.json')
    time_series_config_path = 'crypto_trading_model/config/time_series_config.json'
    
    # If model_path not specified, use default location
    if model_path is None:
        model_path = os.path.join(model_dir, 'final_model.pt')
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    
    # Path to the test data
    test_path = os.path.join(data_dir, 'test_data.h5')
    if not os.path.exists(test_path):
        logger.error(f"Test data file not found at {test_path}")
        return None
    
    # First try to load time_series_config.json, then fall back to local config
    if os.path.exists(time_series_config_path):
        logger.info(f"Loading configuration from {time_series_config_path}")
        with open(time_series_config_path, 'r') as f:
            config = json.load(f)
    elif os.path.exists(local_config_path):
        logger.info(f"Loading configuration from {local_config_path}")
        with open(local_config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.error(f"Configuration file not found at {local_config_path} or {time_series_config_path}")
        return None
    
    # Make sure the configuration has all required keys
    if 'data' not in config:
        config['data'] = {}
    
    # Update configuration with test data path and batch size
    # Normalize paths for cross-platform compatibility
    test_path_normalized = os.path.normpath(test_path).replace('\\', '/')
    config['data']['test_data_path'] = test_path_normalized
    
    # Also normalize the train and val paths if they exist
    if 'train_data_path' in config['data']:
        train_path = config['data']['train_data_path']
        config['data']['train_data_path'] = os.path.normpath(train_path).replace('\\', '/')
    
    if 'val_data_path' in config['data']:
        val_path = config['data']['val_data_path']
        config['data']['val_data_path'] = os.path.normpath(val_path).replace('\\', '/')
    
    if num_workers is not None:
        config['data']['num_workers'] = num_workers
    if 'training' not in config:
        config['training'] = {}
    config['training']['batch_size'] = batch_size
    
    # Ensure we have model feature dimensions information
    if 'model' not in config:
        logger.error("No model configuration found in config file")
        return None
    
    # Add timeframes from data if not present in config
    if 'data' in config and 'timeframes' not in config['data']:
        import h5py
        with h5py.File(test_path, 'r') as f:
            timeframes = [
                tf for tf in f.keys() 
                if isinstance(f[tf], h5py.Group) and 'table' in f[tf]
            ]
            config['data']['timeframes'] = timeframes
            logger.info(f"Detected timeframes: {timeframes}")

    # Save updated configuration
    temp_config_path = os.path.join(model_dir, 'eval_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Using model configuration: {config}")
    
    # Load model using train_lightning_model
    try:
        # Check if this is a Lightning checkpoint file
        if model_path.endswith('.ckpt'):
            logger.info(f"Loading model directly from checkpoint: {model_path}")
            try:
                # Import the LightningModule class
                from crypto_trading_model.lstm_lightning import LightningTimeSeriesModel
                
                # Load the checkpoint directly
                model = LightningTimeSeriesModel.load_from_checkpoint(
                    model_path,
                    map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                )
                logger.info("Model loaded successfully from checkpoint")
                
            except Exception as e:
                logger.error(f"Error loading model from checkpoint: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Fall back to regular initialization
                logger.warning("Falling back to configuration-based initialization")
                model, _ = train_lightning_model(
                    config_path=temp_config_path,
                    max_epochs=1,  # We won't actually train
                    verbose=True
                )
                
                if model is None:
                    logger.error("Failed to initialize model architecture")
                    return None
                
                # Try to load checkpoint with strict=False
                logger.info("Attempting to load checkpoint with strict=False...")
                checkpoint = torch.load(model_path)
                if 'state_dict' in checkpoint:
                    # Get the state dict from the checkpoint
                    state_dict = checkpoint['state_dict']
                    
                    # Remove 'model.' prefix if present
                    clean_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('model.'):
                            clean_key = key[len('model.'):]
                            clean_state_dict[clean_key] = value
                        else:
                            clean_state_dict[key] = value
                    
                    # Load the state dict with strict=False
                    try:
                        model.model.load_state_dict(clean_state_dict, strict=False)
                    except Exception as e:
                        logger.error(f"Error loading checkpoint state_dict: {str(e)}")
                        return None
                else:
                    # Try direct loading
                    try:
                        model.load_state_dict(checkpoint, strict=False)
                    except Exception as e:
                        logger.error(f"Error loading checkpoint directly: {str(e)}")
                        return None
                    
                logger.warning("Model loaded with strict=False - some parameters may be missing or unused")
        else:
            # Create a dummy train/val loader to initialize the model
            logger.info("Loading model architecture...")
            model, _ = train_lightning_model(
                config_path=temp_config_path,
                max_epochs=1,  # We won't actually train
                verbose=True
            )
            
            if model is None:
                logger.error("Failed to initialize model architecture")
                return None
            
            # Load the saved model weights
            logger.info(f"Loading model weights from {model_path}...")
            try:
                # Attempt to load the state dict directly
                model.model.load_state_dict(torch.load(model_path))
            except Exception as direct_load_error:
                logger.warning(f"Direct loading failed: {str(direct_load_error)}")
                logger.info("Attempting to load with strict=False...")
                
                # Try loading with strict=False to allow for parameter mismatches
                try:
                    model.model.load_state_dict(torch.load(model_path), strict=False)
                    logger.warning(
                        "Model loaded with strict=False - some parameters may "
                        "be missing or unused"
                    )
                except Exception as e:
                    logger.error(f"Could not load model weights: {str(e)}")
                    return None
        
        model.eval()  # Set to evaluation mode
        logger.info("Model ready for evaluation")
        
        # Load test data
        import h5py
        logger.info(f"Loading test data from {test_path}...")
        with h5py.File(test_path, 'r') as f:
            # Extract the data from the tables
            test_data = {}
            timeframes = []
            
            for tf in f.keys():
                if isinstance(f[tf], h5py.Group) and 'table' in f[tf]:
                    timeframes.append(tf)
                    logger.info(f"Loading timeframe: {tf}")
                    try:
                        # Access the main table
                        table_data = f[tf]['table'][:]
                        
                        # Copy the structured array
                        table_data = np.array(table_data.copy())
                        
                        # Convert structured array to separate tensors
                        for field_name in table_data.dtype.names:
                            if field_name != 'index':
                                feature_name = f"{tf}_{field_name}"
                                field_data = np.copy(table_data[field_name])
                                test_data[feature_name] = torch.tensor(
                                    field_data, dtype=torch.float32
                                )
                    except Exception as e:
                        logger.error(f"Error loading timeframe {tf}: {str(e)}")
            
            # Create labels from price direction
            if timeframes:
                primary_tf = timeframes[0]
                try:
                    table_data = f[primary_tf]['table'][:]
                    if 'price_direction' in table_data.dtype.names:
                        raw_labels = table_data['price_direction']
                        num_classes = config['model']['num_classes']
                        valid_labels = np.clip(
                            raw_labels, 0, num_classes - 1
                        )
                        test_labels = torch.tensor(
                            valid_labels, dtype=torch.long
                        )
                    else:
                        logger.warning(
                            f"No price_direction field found in {primary_tf}"
                        )
                        test_labels = torch.zeros(
                            len(next(iter(test_data.values()))), 
                            dtype=torch.long
                        )
                except Exception as e:
                    logger.error(f"Error loading labels: {str(e)}")
                    test_labels = torch.zeros(
                        len(next(iter(test_data.values()))), 
                        dtype=torch.long
                    )
        
        # Create our own TimeSeriesDataset class for evaluation
        class TimeSeriesDataset(torch.utils.data.Dataset):
            def __init__(self, data, labels):
                # Deep copy the data to avoid pickling issues with references
                self.data = {}
                for key, tensor in data.items():
                    self.data[key] = (
                        tensor.clone() if isinstance(tensor, torch.Tensor) 
                        else torch.tensor(tensor, dtype=torch.float32)
                    )
                
                # Make sure labels are properly copied
                self.labels = (
                    labels.clone() if isinstance(labels, torch.Tensor) 
                    else torch.tensor(labels, dtype=torch.long)
                )
                
                # Group features by timeframe
                self.timeframes = {}
                for key in data.keys():
                    if '_' in key:
                        tf, feature = key.split('_', 1)
                        if tf not in self.timeframes:
                            self.timeframes[tf] = []
                        self.timeframes[tf].append(feature)
                
                # Find the base timeframe (highest frequency) and its length
                base_tf = None
                base_length = 0
                
                # Determine the timeframe with the most samples (highest frequency)
                timeframe_lengths = {}
                for tf in self.timeframes.keys():
                    # Get the length of one feature from this timeframe
                    if self.timeframes[tf]:
                        feature = self.timeframes[tf][0]
                        tf_length = len(self.data[f"{tf}_{feature}"])
                        timeframe_lengths[tf] = tf_length
                        if base_tf is None or tf_length > base_length:
                            base_tf = tf
                            base_length = tf_length
                
                logger.info(
                    f"Base timeframe is {base_tf} with {base_length} samples"
                )
                for tf, length in timeframe_lengths.items():
                    logger.info(f"Timeframe {tf} has {length} samples")
                
                # Use the length of the labels if available
                self.data_length = len(self.labels)
                
                # Ensure labels match the base timeframe length
                if len(self.labels) != base_length:
                    logger.warning(
                        f"Labels length ({len(self.labels)}) doesn't match "
                        f"base timeframe length ({base_length}), adjusting..."
                    )
                    
                    # If labels are longer, truncate
                    if len(self.labels) > base_length:
                        self.labels = self.labels[:base_length]
                    # If labels are shorter, pad with zeros (this should be rare)
                    elif len(self.labels) < base_length:
                        padding = torch.zeros(
                            base_length - len(self.labels), 
                            dtype=torch.long
                        )
                        self.labels = torch.cat([self.labels, padding])
                    
                    self.data_length = base_length
                
                # Resample lower frequency data to match the base timeframe length
                for tf in self.timeframes.keys():
                    if tf != base_tf:
                        tf_length = timeframe_lengths.get(tf, 0)
                        
                        # Only process if we have features for this timeframe
                        if tf_length > 0:
                            for feature in self.timeframes[tf]:
                                feature_key = f"{tf}_{feature}"
                                source_tensor = self.data[feature_key]
                                
                                # If source is shorter than target (lower frequency)
                                if len(source_tensor) < base_length:
                                    logger.info(
                                        f"Upsampling {feature_key} from "
                                        f"{len(source_tensor)} to {base_length} samples"
                                    )
                                    
                                    # Create indices for nearest-neighbor upsampling
                                    # This repeats each value the appropriate number of times
                                    indices = torch.linspace(
                                        0, len(source_tensor) - 1, base_length
                                    ).long()
                                    upsampled = source_tensor[indices]
                                    self.data[feature_key] = upsampled
                                # If source is longer than target (this should be rare)
                                elif len(source_tensor) > base_length:
                                    logger.warning(
                                        f"Feature {feature_key} has length "
                                        f"{len(source_tensor)}, truncating to {base_length}."
                                    )
                                    self.data[feature_key] = source_tensor[:base_length]
                
                logger.info(
                    f"Dataset organized with timeframes: {list(self.timeframes.keys())} "
                    f"and {self.data_length} samples"
                )
            
            def __len__(self):
                return self.data_length
            
            def __getitem__(self, idx):
                # Ensure index is in bounds
                if idx >= self.data_length:
                    raise IndexError(
                        f"Index {idx} is out of bounds for dataset "
                        f"with length {self.data_length}"
                    )
                
                # Organize data by timeframe for model consumption
                sample = {}
                for tf in self.timeframes:
                    # Check if this timeframe has any features
                    if not self.timeframes[tf]:
                        continue  # Skip empty timeframes
                        
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
                        sample[tf] = features.unsqueeze(0)  # Add sequence dimension
                    except IndexError as e:
                        logger.error(f"Index error for {tf} at idx={idx}: {str(e)}")
                        raise
                    except Exception as e:
                        logger.error(
                            f"Error processing timeframe {tf} at idx={idx}: {str(e)}"
                        )
                        # Provide an empty tensor as fallback
                        sample[tf] = torch.zeros((1, 1), dtype=torch.float32)
                
                sample['label'] = self.labels[idx]
                return sample
        
        # Create test dataset
        test_dataset = TimeSeriesDataset(test_data, test_labels)
        
        # Get the number of workers
        if num_workers is not None:
            workers = num_workers
        elif sys.platform.startswith('win'):
            workers = 0
        else:
            workers = 2
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True
        )
        
        # Evaluate model
        logger.info("Evaluating model on test data...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                # Move tensors to the device
                sample = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items() if k != 'label'
                }
                
                if 'label' in batch:
                    labels = batch['label'].to(device)
                else:
                    continue
                
                # Forward pass
                outputs = model(sample)
                _, predicted = torch.max(outputs, 1)
                
                # Collect true and predicted labels
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        report = classification_report(
            y_true, y_pred, target_names=['Sell', 'Hold', 'Buy']
        )
        cm = confusion_matrix(y_true, y_pred)
        
        # Log results
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Sell', 'Hold', 'Buy'],
            yticklabels=['Sell', 'Hold', 'Buy']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        cm_path = os.path.join(model_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        logger.info(f"Saved confusion matrix to {cm_path}")
        
        # Save results to file
        results = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        results_path = os.path.join(model_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved test results to {results_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def main():
    """Main entry point for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the trained LSTM model on test data"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models/lstm_improved",
        help="Directory containing the trained model and configuration"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to specific model file (default: looks for final_model.pt in model_dir)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/synthetic",
        help="Directory containing the data files"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of worker processes for data loading (default: auto-detect)"
    )
    parser.add_argument(
        "--use_checkpoint", action="store_true", 
        help="Use the checkpoint from the models/checkpoints directory if available"
    )
    
    args = parser.parse_args()
    
    # Look for checkpoint file if requested
    if args.use_checkpoint and args.model_path is None:
        checkpoint_dir = "models/checkpoints"
        if os.path.exists(checkpoint_dir):
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if ckpt_files:
                # Sort by val_loss if possible, otherwise by modification time
                ckpt_files.sort()
                selected_ckpt = os.path.join(checkpoint_dir, ckpt_files[0])
                logger.info(f"Using checkpoint file: {selected_ckpt}")
                args.model_path = selected_ckpt
    
    # Evaluate the model
    results = evaluate_model(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_path=args.model_path
    )
    
    if results:
        logger.info("Model evaluation complete.")
    else:
        logger.error("Model evaluation failed.")


if __name__ == "__main__":
    import sys
    main() 