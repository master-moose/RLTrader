import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F

from .model import MultiTimeframeModel, TimeSeriesTransformer, TimeSeriesForecaster

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset for multi-timeframe time series data"""
    
    def __init__(self, 
                data: List[Dict[str, pd.DataFrame]], 
                labels: Optional[np.ndarray] = None,
                sequence_length: int = 50,
                timeframes: List[str] = ["15m", "4h", "1d"],
                forecast_horizon: Optional[int] = None):
        """
        Initialize the dataset
        
        Parameters:
        - data: List of dictionaries containing DataFrames for each timeframe
        - labels: Optional array of labels for classification
        - sequence_length: Length of input sequences
        - timeframes: List of timeframes to include
        - forecast_horizon: Optional forecast horizon for regression tasks
        """
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.timeframes = timeframes
        self.forecast_horizon = forecast_horizon
        self.is_forecasting = forecast_horizon is not None
        
        # Verify data structure
        if len(data) > 0:
            sample = data[0]
            for tf in timeframes:
                if tf not in sample:
                    logger.warning(f"Timeframe {tf} not found in data, will be padded with zeros")
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset
        
        Returns dictionary with:
        - For each timeframe: tensor of shape (sequence_length, num_features)
        - 'label': classification label or None
        - 'target': regression target or None
        """
        sample = self.data[idx]
        result = {}
        
        # Process each timeframe
        for tf in self.timeframes:
            if tf in sample:
                # Get the DataFrame for this timeframe
                df = sample[tf]
                
                # Use the most recent sequence_length candles
                if len(df) > self.sequence_length:
                    df = df.iloc[-self.sequence_length:]
                
                # Get feature columns (exclude timestamp/index)
                features = df.select_dtypes(include=[np.number]).values
                
                # Pad if necessary
                if len(features) < self.sequence_length:
                    padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
                    features = np.vstack([padding, features])
                
                # Add to result
                result[tf] = torch.tensor(features, dtype=torch.float32)
            else:
                # Timeframe not found, use zeros
                # Assuming all timeframes have same number of features
                if len(result) > 0:
                    # Get feature dimension from another timeframe
                    feat_dim = next(iter(result.values())).shape[1]
                else:
                    # Default feature dimension if no other timeframe available
                    feat_dim = 5  # Default: OHLCV
                
                result[tf] = torch.zeros(self.sequence_length, feat_dim, dtype=torch.float32)
        
        # Add label for classification tasks
        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Add target for forecasting tasks
        if self.is_forecasting:
            # Get the 'close' price for the next forecast_horizon candles
            # Assuming 'close' is the 4th column (index 3) in OHLCV data
            target = np.zeros(self.forecast_horizon)
            
            # Use the base timeframe (first in list) for target
            base_tf = self.timeframes[0]
            if base_tf in sample:
                df = sample[base_tf]
                if len(df) > self.sequence_length:
                    future_prices = df.iloc[self.sequence_length:self.sequence_length+self.forecast_horizon]
                    if len(future_prices) > 0:
                        # Use 'close' column if available, otherwise use column index 3
                        if 'close' in future_prices.columns:
                            target[:len(future_prices)] = future_prices['close'].values
                        else:
                            # Fallback to column index 3 (typical position of 'close' in OHLCV)
                            target[:len(future_prices)] = future_prices.iloc[:, 3].values
            
            result['target'] = torch.tensor(target, dtype=torch.float32)
        
        return result

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    
    Parameters:
    - batch: List of samples from the dataset
    
    Returns:
    - Dictionary with batched tensors
    """
    result = {}
    
    # Get all keys
    keys = batch[0].keys()
    
    for key in keys:
        # Stack tensors along batch dimension
        result[key] = torch.stack([item[key] for item in batch])
    
    return result

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Parameters:
        - patience: Number of epochs with no improvement after which training will be stopped
        - min_delta: Minimum change in the monitored quantity to qualify as an improvement
        - restore_best_weights: Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Update early stopping state
        
        Parameters:
        - score: Current validation score (higher is better)
        - model: Model to save weights from
        
        Returns:
        - True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            # No improvement
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        
        return self.early_stop
    
    def restore_weights(self, model: nn.Module) -> None:
        """Restore best weights to model"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best weights to model")

def train_time_series_model(model: nn.Module,
                           train_data: TimeSeriesDataset,
                           val_data: TimeSeriesDataset,
                           epochs: int = 100,
                           batch_size: int = 32,
                           learning_rate: float = 0.001,
                           weight_decay: float = 1e-5,
                           patience: int = 10,
                           device: str = "cuda" if torch.cuda.is_available() else "cpu",
                           model_save_path: Optional[str] = None,
                           scheduler_factor: float = 0.5,
                           scheduler_patience: int = 5) -> Dict[str, List[float]]:
    """
    Train the time series model
    
    Parameters:
    - model: Model instance to train
    - train_data: Training data
    - val_data: Validation data
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    - learning_rate: Learning rate for optimizer
    - weight_decay: L2 regularization factor
    - patience: Patience for early stopping
    - device: Device to train on ('cuda' or 'cpu')
    - model_save_path: Optional path to save model weights
    - scheduler_factor: Factor to reduce learning rate by
    - scheduler_patience: Patience for learning rate scheduler
    
    Returns:
    - Training history (loss and metrics)
    """
    logger.info(f"Training model on {device}")
    model = model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Determine if this is a classification or regression task
    is_forecasting = isinstance(model, TimeSeriesForecaster)
    
    # Set up criterion
    if is_forecasting:
        criterion = nn.MSELoss()
    else:
        # CrossEntropyLoss for classification (assumes classes start at 0)
        criterion = nn.CrossEntropyLoss()
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max' if not is_forecasting else 'min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True
    )
    
    # Set up early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
    }
    
    if not is_forecasting:
        history.update({
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        })
    else:
        history.update({
            'train_mse': [],
            'val_mse': [],
            'train_mae': [],
            'val_mae': []
        })
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            if is_forecasting:
                outputs = model({k: v for k, v in batch.items() if k not in ['label', 'target']}, 
                               batch.get('target'))
                loss = criterion(outputs, batch['target'])
            else:
                outputs = model({k: v for k, v in batch.items() if k != 'label'})
                loss = criterion(outputs, batch['label'])
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Save predictions and targets for metrics
            train_losses.append(loss.detach().item())
            
            if is_forecasting:
                train_predictions.append(outputs.detach().cpu())
                train_targets.append(batch['target'].detach().cpu())
            else:
                train_predictions.append(outputs.argmax(dim=1).detach().cpu())
                train_targets.append(batch['label'].detach().cpu())
        
        # Calculate training metrics
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        
        # Concatenate predictions and targets
        train_predictions = torch.cat(train_predictions)
        train_targets = torch.cat(train_targets)
        
        if is_forecasting:
            train_mse = ((train_predictions - train_targets) ** 2).mean().item()
            train_mae = (train_predictions - train_targets).abs().mean().item()
            history['train_mse'].append(train_mse)
            history['train_mae'].append(train_mae)
        else:
            train_accuracy = accuracy_score(train_targets, train_predictions)
            train_f1 = f1_score(train_targets, train_predictions, average='weighted')
            history['train_accuracy'].append(train_accuracy)
            history['train_f1'].append(train_f1)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                # Move data to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                if is_forecasting:
                    outputs = model({k: v for k, v in batch.items() if k not in ['label', 'target']})
                    loss = criterion(outputs, batch['target'])
                else:
                    outputs = model({k: v for k, v in batch.items() if k != 'label'})
                    loss = criterion(outputs, batch['label'])
                
                # Save predictions and targets for metrics
                val_losses.append(loss.detach().item())
                
                if is_forecasting:
                    val_predictions.append(outputs.detach().cpu())
                    val_targets.append(batch['target'].detach().cpu())
                else:
                    val_predictions.append(outputs.argmax(dim=1).detach().cpu())
                    val_targets.append(batch['label'].detach().cpu())
        
        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        history['val_loss'].append(val_loss)
        
        # Concatenate predictions and targets
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
        
        if is_forecasting:
            val_mse = ((val_predictions - val_targets) ** 2).mean().item()
            val_mae = (val_predictions - val_targets).abs().mean().item()
            history['val_mse'].append(val_mse)
            history['val_mae'].append(val_mae)
            
            # Print metrics
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                       f"MSE: {train_mse:.4f}/{val_mse:.4f} - "
                       f"MAE: {train_mae:.4f}/{val_mae:.4f} - "
                       f"Time: {time.time() - start_time:.2f}s")
            
            # Update scheduler
            scheduler.step(val_mse)
            
            # Check early stopping
            if early_stopping(val_mse, model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
        else:
            val_accuracy = accuracy_score(val_targets, val_predictions)
            val_f1 = f1_score(val_targets, val_predictions, average='weighted')
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            
            # Print metrics
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                       f"Accuracy: {train_accuracy:.4f}/{val_accuracy:.4f} - "
                       f"F1: {train_f1:.4f}/{val_f1:.4f} - "
                       f"Time: {time.time() - start_time:.2f}s")
            
            # Update scheduler
            scheduler.step(val_f1)
            
            # Check early stopping
            if early_stopping(-val_f1, model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best weights
    early_stopping.restore_weights(model)
    
    # Save model if path provided
    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
    
    return history

def plot_training_history(history: Dict[str, List[float]], 
                        is_forecasting: bool = False, 
                        save_path: Optional[str] = None):
    """
    Plot training history
    
    Parameters:
    - history: Training history dictionary
    - is_forecasting: Whether this is a forecasting model
    - save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot metrics
    if is_forecasting:
        axes[1].plot(history['train_mse'], label='Train MSE')
        axes[1].plot(history['val_mse'], label='Validation MSE')
        axes[1].set_title('Mean Squared Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
    else:
        axes[1].plot(history['train_accuracy'], label='Train Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
    
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def evaluate_model(model: nn.Module,
                  test_data: TimeSeriesDataset,
                  batch_size: int = 32,
                  device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, float]:
    """
    Evaluate the model on test data
    
    Parameters:
    - model: Trained model to evaluate
    - test_data: Test dataset
    - batch_size: Batch size for evaluation
    - device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
    - Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating model on {device}")
    model = model.to(device)
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Determine if this is a classification or regression task
    is_forecasting = isinstance(model, TimeSeriesForecaster)
    
    # Set up criterion
    if is_forecasting:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Evaluation metrics
    all_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            if is_forecasting:
                outputs = model({k: v for k, v in batch.items() if k not in ['label', 'target']})
                loss = criterion(outputs, batch['target'])
            else:
                outputs = model({k: v for k, v in batch.items() if k != 'label'})
                loss = criterion(outputs, batch['label'])
            
            # Save predictions and targets for metrics
            all_losses.append(loss.item())
            
            if is_forecasting:
                all_predictions.append(outputs.detach().cpu().numpy())
                all_targets.append(batch['target'].detach().cpu().numpy())
            else:
                all_predictions.append(outputs.argmax(dim=1).detach().cpu().numpy())
                all_targets.append(batch['label'].detach().cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    test_loss = np.mean(all_losses)
    
    if is_forecasting:
        # Regression metrics
        test_mse = ((all_predictions - all_targets) ** 2).mean()
        test_mae = np.abs(all_predictions - all_targets).mean()
        test_rmse = np.sqrt(test_mse)
        
        # Calculate directional accuracy (if price movement direction is predicted correctly)
        pred_direction = np.diff(all_predictions, axis=1) > 0
        target_direction = np.diff(all_targets, axis=1) > 0
        directional_accuracy = np.mean(pred_direction == target_direction)
        
        metrics = {
            'loss': test_loss,
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test MSE: {test_mse:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Directional Accuracy: {directional_accuracy:.4f}")
        
    else:
        # Classification metrics
        test_accuracy = accuracy_score(all_targets, all_predictions)
        test_precision = precision_score(all_targets, all_predictions, average='weighted')
        test_recall = recall_score(all_targets, all_predictions, average='weighted')
        test_f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        metrics = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        }
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1 Score: {test_f1:.4f}")
        
        # Print confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        logger.info("Confusion Matrix:")
        logger.info(cm)
    
    return metrics 