#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate the trained LSTM model on the test dataset.
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader

# Import from local modules
from crypto_trading_model.lstm_lightning import train_lightning_model, LightningTimeSeriesModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model_dir, data_dir, batch_size=256, num_workers=None):
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
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Paths to the model and configuration files
    config_path = os.path.join(model_dir, 'config.json')
    model_path = os.path.join(model_dir, 'final_model.pt')
    
    # Check if files exist
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        return None
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None
    
    # Path to the test data
    test_path = os.path.join(data_dir, 'test_data.h5')
    if not os.path.exists(test_path):
        logger.error(f"Test data file not found at {test_path}")
        return None
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update configuration with test data path and batch size
    if 'data' not in config:
        config['data'] = {}
    config['data']['test_data_path'] = test_path
    if num_workers is not None:
        config['data']['num_workers'] = num_workers
    if 'training' not in config:
        config['training'] = {}
    config['training']['batch_size'] = batch_size
    
    # Save updated configuration
    temp_config_path = os.path.join(model_dir, 'eval_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load model using train_lightning_model
    try:
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
        model.model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        
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
                        valid_labels = np.clip(raw_labels, 0, num_classes - 1)
                        test_labels = torch.tensor(valid_labels, dtype=torch.long)
                    else:
                        logger.warning(f"No price_direction field found in {primary_tf}")
                        test_labels = torch.zeros(len(next(iter(test_data.values()))), dtype=torch.long)
                except Exception as e:
                    logger.error(f"Error loading labels: {str(e)}")
                    test_labels = torch.zeros(len(next(iter(test_data.values()))), dtype=torch.long)
        
        # Create dataset and dataloader
        from crypto_trading_model.lstm_lightning import TimeSeriesDataset
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
                sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items() if k != 'label'}
                
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
        report = classification_report(y_true, y_pred, target_names=['Sell', 'Hold', 'Buy'])
        cm = confusion_matrix(y_true, y_pred)
        
        # Log results
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        logger.info(f"Classification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sell', 'Hold', 'Buy'],
                   yticklabels=['Sell', 'Hold', 'Buy'])
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
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM model on test data")
    parser.add_argument("--model_dir", type=str, default="models/lstm_improved",
                      help="Directory containing the trained model and configuration")
    parser.add_argument("--data_dir", type=str, default="data/synthetic",
                      help="Directory containing the data files")
    parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=None,
                      help="Number of worker processes for data loading (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Evaluate the model
    results = evaluate_model(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if results:
        logger.info("Model evaluation complete.")
    else:
        logger.error("Model evaluation failed.")

if __name__ == "__main__":
    import sys
    main() 