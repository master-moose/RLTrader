#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated training script for the cryptocurrency trading model using PyTorch Lightning.
This script provides a more streamlined and automated training process with cleaner logs.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'model_training_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('automated_training')

def setup_directories():
    """Create necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'data/synthetic',
        'output/time_series',
        'output/reinforcement',
        'output/ensemble',
        'output/backtest',
        'logs',
        'models/checkpoints',
        'logs/lightning_logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")
    
    logger.debug("Directory setup complete.")

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

def summarize_config(config):
    """Print a summary of the configuration."""
    logger.info("Configuration summary:")
    
    # Data settings
    data_config = config.get('data', {})
    logger.info(f"  Data paths: train={data_config.get('train_data_path')}, val={data_config.get('val_data_path')}")
    logger.info(f"  Timeframes: {data_config.get('timeframes', [])}")
    
    # Model settings
    model_config = config.get('model', {})
    logger.info(f"  Model type: {model_config.get('type', 'unknown')}")
    logger.info(f"  Hidden dimensions: {model_config.get('hidden_dims', 'N/A')}")
    logger.info(f"  Num layers: {model_config.get('num_layers', 'N/A')}")
    
    # Training settings
    training_config = config.get('training', {})
    logger.info(f"  Batch size: {training_config.get('batch_size', 'N/A')}")
    logger.info(f"  Learning rate: {training_config.get('learning_rate', 'N/A')}")
    logger.info(f"  Max epochs: {training_config.get('epochs', 'N/A')}")

def evaluate_model_quality(trainer, min_accuracy=0.55):
    """
    Evaluate if the model has learned effectively based on metrics.
    
    Args:
        trainer: PyTorch Lightning trainer instance with training history
        min_accuracy: Minimum acceptable validation accuracy

    Returns:
        tuple: (is_model_good, message)
            - is_model_good: Boolean indicating if model quality is acceptable
            - message: String explanation of the quality assessment
    """
    try:
        # Get training metrics
        callback_metrics = trainer.callback_metrics
        
        # Extract current metrics
        val_loss = callback_metrics.get('val_loss', float('inf')).item()
        val_acc = callback_metrics.get('val_acc', 0).item()
        train_loss = callback_metrics.get('train_loss', float('inf')).item()
        train_acc = callback_metrics.get('train_acc', 0).item()
        
        # Get information about why training stopped
        current_epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        
        # Generate quality assessment
        issues = []
        
        # Check validation metrics
        if val_acc < min_accuracy:
            issues.append(f"Low validation accuracy: {val_acc:.4f} < {min_accuracy}")
        
        # Check for overfitting
        if train_acc - val_acc > 0.2:
            issues.append(f"Potential overfitting: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Check if stopped very early due to no improvement
        if current_epoch < max_epochs * 0.3:  # Stopped in first 30% of epochs
            issues.append(f"Training stopped early at epoch {current_epoch}/{max_epochs}, " 
                         f"suggesting no improvement in validation metrics")
        
        # Assess loss values
        if val_loss > 0.69:  # Close to log(2) which is random guessing for binary classification
            issues.append(f"Validation loss ({val_loss:.4f}) is close to random guessing")
        
        # Make final determination
        if issues:
            quality_message = "Model quality concerns detected:\n - " + "\n - ".join(issues)
            logger.warning(quality_message)
            return False, quality_message
        else:
            quality_message = (f"Model quality looks good: val_acc={val_acc:.4f}, "
                              f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}")
            logger.info(quality_message)
            return True, quality_message
            
    except Exception as e:
        logger.error(f"Error evaluating model quality: {str(e)}")
        return False, f"Could not assess model quality: {str(e)}"

class ModelQualityCallback(object):
    """Callback to monitor training progress and detect potential issues."""
    
    def __init__(self):
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.plateau_count = 0
        self.improvement_threshold = 0.001
        
    def on_epoch_end(self, trainer, loss_dict):
        """Update metrics after each epoch"""
        if not loss_dict:
            return
            
        train_loss = loss_dict.get('train_loss', float('inf'))
        val_loss = loss_dict.get('val_loss', float('inf'))
        train_acc = loss_dict.get('train_acc', 0)
        val_acc = loss_dict.get('val_acc', 0)
        
        # Add to history
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        
        # Check for plateau
        if len(self.val_loss_history) >= 5:
            # Check last 5 values for plateau
            recent = self.val_loss_history[-5:]
            max_diff = max(recent) - min(recent)
            
            if max_diff < self.improvement_threshold:
                self.plateau_count += 1
                if self.plateau_count >= 2:  # Two consecutive plateaus
                    logger.warning(f"Plateau detected in validation loss for {self.plateau_count * 5} epochs")
            else:
                self.plateau_count = 0
                
        # Log current progress every 10 epochs
        epoch = len(self.train_loss_history)
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                       f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

def run_lightning_training(config_path, max_epochs=None, patience=None, verbose=False, retry_on_poor_quality=True, min_accuracy=0.55):
    """
    Run the PyTorch Lightning training process.
    
    Args:
        config_path (str): Path to the configuration file
        max_epochs (int, optional): Maximum number of epochs to train
        patience (int, optional): Patience for early stopping
        verbose (bool, optional): Whether to enable verbose logging
        retry_on_poor_quality (bool): Whether to retry training with different params if model quality is poor
        min_accuracy (float): Minimum acceptable validation accuracy
    """
    from crypto_trading_model.lstm_lightning import train_lightning_model
    import pytorch_lightning as pl
    
    # Load the config to extract parameters
    config = load_config(config_path)
    
    # Summarize the configuration
    summarize_config(config)
    
    # Set defaults if not provided
    if max_epochs is None:
        max_epochs = config.get('training', {}).get('epochs', 100)
    
    if patience is None:
        patience = config.get('training', {}).get('patience', 15)
    
    logger.info(f"Starting Lightning training with max_epochs={max_epochs}, patience={patience}")
    
    # Create custom callback for monitoring
    quality_callback = ModelQualityCallback()
    
    # Override the default Lightning trainer
    class CustomTrainer(pl.Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quality_callback = quality_callback
            
        def on_validation_epoch_end(self):
            super().on_validation_epoch_end()
            if hasattr(self, 'callback_metrics'):
                metrics = {k: v.item() if hasattr(v, 'item') else v 
                          for k, v in self.callback_metrics.items()}
                self.quality_callback.on_epoch_end(self, metrics)
    
    # Run the Lightning training
    model, trainer = train_lightning_model(
        config_path=config_path,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        verbose=verbose,
        trainer_class=CustomTrainer
    )
    
    if model is None:
        logger.error("Training failed or no model was returned")
        return None, None
    
    # Evaluate model quality
    is_model_good, quality_message = evaluate_model_quality(trainer, min_accuracy=min_accuracy)
    
    # Retry with different parameters if model quality is poor
    if not is_model_good and retry_on_poor_quality:
        logger.warning("Model quality is poor. Attempting to retrain with modified parameters...")
        
        # Try different strategies based on the issues
        if "overfitting" in quality_message:
            # Increase regularization
            logger.info("Adjusting for overfitting by increasing dropout and weight decay")
            config['model']['dropout'] = min(config.get('model', {}).get('dropout', 0.3) + 0.1, 0.5)
            config['training']['weight_decay'] = config.get('training', {}).get('weight_decay', 1e-5) * 5
            
            # Save the adjusted config
            adjusted_config_path = config_path.replace('.json', '_adjusted.json')
            with open(adjusted_config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            # Rerun training with adjusted config
            logger.info(f"Retraining with adjusted config: {adjusted_config_path}")
            return run_lightning_training(
                config_path=adjusted_config_path,
                max_epochs=max_epochs,
                patience=patience,
                verbose=verbose,
                retry_on_poor_quality=False  # Prevent infinite loop
            )
            
        elif "random guessing" in quality_message or "Low validation accuracy" in quality_message:
            # Model is not learning at all - try with different learning rate and architecture
            logger.info("Model is not learning effectively. Adjusting learning rate and architecture")
            config['training']['learning_rate'] = config.get('training', {}).get('learning_rate', 0.001) * 3
            config['model']['hidden_dims'] = int(config.get('model', {}).get('hidden_dims', 256) * 1.5)
            
            # Save the adjusted config
            adjusted_config_path = config_path.replace('.json', '_adjusted.json')
            with open(adjusted_config_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            # Rerun training with adjusted config
            logger.info(f"Retraining with adjusted config: {adjusted_config_path}")
            return run_lightning_training(
                config_path=adjusted_config_path,
                max_epochs=int(max_epochs * 1.5),  # Increase epochs
                patience=patience,
                verbose=verbose,
                retry_on_poor_quality=False  # Prevent infinite loop
            )
    
    logger.info(f"Training completed successfully")
    
    # Save final model if needed
    output_dir = config.get('data', {}).get('output_dir', 'output/time_series')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model with quality assessment
    model_info = {
        "training_epochs": trainer.current_epoch,
        "max_epochs": trainer.max_epochs,
        "early_stopped": trainer.current_epoch < trainer.max_epochs,
        "metrics": {
            "train_loss": trainer.callback_metrics.get('train_loss', float('inf')).item(),
            "train_acc": trainer.callback_metrics.get('train_acc', 0).item(),
            "val_loss": trainer.callback_metrics.get('val_loss', float('inf')).item(),
            "val_acc": trainer.callback_metrics.get('val_acc', 0).item()
        },
        "quality_assessment": {
            "is_model_good": is_model_good,
            "quality_message": quality_message
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save model info
    model_info_path = os.path.join(output_dir, 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"Model quality assessment saved to {model_info_path}")
    
    model_path = os.path.join(output_dir, 'final_model.pt')
    try:
        import torch
        torch.save(model.state_dict(), model_path)
        logger.info(f"Final model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {str(e)}")
    
    # Return final quality assessment
    return model, trainer, is_model_good, quality_message

def main():
    """Main entry point for the automated training script."""
    parser = argparse.ArgumentParser(description='Automated Cryptocurrency Trading Model Training')
    parser.add_argument('--config', type=str, default='crypto_trading_model/config/time_series_config.json',
                      help='Path to config file')
    parser.add_argument('--max-epochs', type=int, default=None,
                      help='Maximum number of epochs to train')
    parser.add_argument('--patience', type=int, default=None,
                      help='Early stopping patience')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--min-accuracy', type=float, default=0.55,
                      help='Minimum acceptable validation accuracy')
    parser.add_argument('--no-retry', action='store_true',
                      help='Disable automatic retraining on poor model quality')
    
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    # Log system info
    try:
        import torch
        import pytorch_lightning as pl
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"PyTorch Lightning version: {pl.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("Could not import PyTorch or PyTorch Lightning")
    
    # Run Lightning training
    results = run_lightning_training(
        config_path=args.config,
        max_epochs=args.max_epochs,
        patience=args.patience,
        verbose=args.verbose,
        retry_on_poor_quality=not args.no_retry,
        min_accuracy=args.min_accuracy
    )
    
    if results and len(results) >= 4:
        model, trainer, is_model_good, quality_message = results
        
        if is_model_good:
            logger.info("✅ Automated training completed successfully with a good quality model")
            # Exit with success
            sys.exit(0)
        else:
            logger.warning("⚠️ Automated training completed but the model quality is questionable")
            logger.warning(quality_message)
            # Exit with warning code
            sys.exit(2)
    else:
        logger.error("❌ Automated training failed")
        # Exit with error
        sys.exit(1)

if __name__ == "__main__":
    main() 