#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger('crypto_trading_model.ensemble')

class EnsembleModel:
    """
    Ensemble model for combining predictions from multiple trading models.
    
    This class implements various ensemble methods including:
    - Weighted voting
    - Majority voting
    - Stacking
    - Probabilistic ensemble
    
    It supports both classification (buy/hold/sell) and regression (price prediction) tasks.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the ensemble model.
        
        Parameters:
        -----------
        config_path : str
            Path to the ensemble configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Parse configuration
        self.output_dir = self.config.get('output_dir', './output/ensemble')
        self.ensemble_method = self.config.get('ensemble_method', 'weighted_vote')
        self.models_config = self.config.get('models', [])
        self.weights = self.config.get('weights', [])
        self.tie_breaking = self.config.get('tie_breaking', 'conservative')
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.min_agreement = self.config.get('min_agreement', 0.5)
        self.use_probabilistic = self.config.get('use_probabilistic', True)
        
        # Advanced settings
        advanced = self.config.get('advanced_settings', {})
        self.decision_strategy = advanced.get('decision_strategy', 'majority_vote')
        self.calibration_method = advanced.get('calibration_method', 'temperature_scaling')
        self.calibration_data_path = advanced.get('calibration_data_path', None)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models
        self.models = []
        self._load_models()
        
        # Initialize calibration
        self.calibration_models = {}
        if self.calibration_data_path and os.path.exists(self.calibration_data_path):
            self._calibrate_models()
    
    def _load_models(self):
        """Load all models specified in the configuration."""
        logger.info(f"Loading models for ensemble ({self.ensemble_method})")
        
        for i, model_config in enumerate(self.models_config):
            model_path = model_config.get('path')
            model_type = model_config.get('type')
            model_weight = model_config.get('weight', 1.0)
            
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}")
                continue
            
            try:
                if model_type == 'time_series':
                    # Load PyTorch model
                    model_state = torch.load(model_path, map_location=torch.device('cpu'))
                    model = {'type': 'time_series', 'state': model_state, 'path': model_path, 'weight': model_weight}
                    self.models.append(model)
                    logger.info(f"Loaded time series model from {model_path}")
                    
                elif model_type == 'reinforcement':
                    # For RL models, we just store the path as they are loaded differently during prediction
                    model = {'type': 'reinforcement', 'path': model_path, 'weight': model_weight}
                    self.models.append(model)
                    logger.info(f"Registered reinforcement learning model from {model_path}")
                    
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
        
        # If weights not provided, use equal weights
        if not self.weights:
            self.weights = [model['weight'] for model in self.models]
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Loaded {len(self.models)} models with weights: {self.weights}")
    
    def _calibrate_models(self):
        """Calibrate model probabilities using the specified method."""
        logger.info(f"Calibrating models using {self.calibration_method}")
        
        try:
            # Load calibration data
            calibration_data = pd.read_csv(self.calibration_data_path)
            X = calibration_data.drop(['label', 'timestamp'], axis=1, errors='ignore').values
            y = calibration_data['label'].values
            
            # Calibrate each model
            for i, model in enumerate(self.models):
                if model['type'] == 'time_series':
                    # For time series models, we use CalibratedClassifierCV as a wrapper
                    model_wrapper = TimeSeriesModelWrapper(model['state'])
                    calibrated_model = CalibratedClassifierCV(
                        base_estimator=model_wrapper,
                        method='isotonic',  # or 'sigmoid'
                        cv='prefit'
                    )
                    calibrated_model.fit(X, y)
                    self.calibration_models[i] = calibrated_model
                    logger.info(f"Calibrated model {i} ({model['type']})")
            
            logger.info(f"Calibrated {len(self.calibration_models)} models")
            
        except Exception as e:
            logger.error(f"Error calibrating models: {str(e)}")
            logger.warning("Proceeding without calibration")
    
    def predict(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble model.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary of DataFrames, where keys are timeframes
        
        Returns:
        --------
        tuple
            (predictions, probabilities) for classification or
            (predictions, confidence) for regression
        """
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            try:
                if model['type'] == 'time_series':
                    # Predict using time series model
                    preds, probs = self._predict_time_series(model['state'], data)
                elif model['type'] == 'reinforcement':
                    # Predict using reinforcement learning model
                    preds, probs = self._predict_reinforcement(model['path'], data)
                else:
                    logger.warning(f"Unknown model type: {model['type']}")
                    continue
                
                all_predictions.append(preds)
                all_probabilities.append(probs)
                
            except Exception as e:
                logger.error(f"Error predicting with model {i}: {str(e)}")
        
        # Check if we have any valid predictions
        if len(all_predictions) == 0:
            logger.error("No valid predictions from any model")
            # Return neutral predictions (all 0s)
            return np.zeros(len(data[next(iter(data))]), dtype=int), \
                   np.zeros((len(data[next(iter(data))]), 3))
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_vote':
            ensemble_preds, ensemble_probs = self._weighted_vote(all_predictions, all_probabilities)
        elif self.ensemble_method == 'majority_vote':
            ensemble_preds, ensemble_probs = self._majority_vote(all_predictions, all_probabilities)
        elif self.ensemble_method == 'stacking':
            # For stacking, we need a meta-model trained separately
            # Here we just fall back to weighted voting
            logger.warning("Stacking not yet implemented, falling back to weighted voting")
            ensemble_preds, ensemble_probs = self._weighted_vote(all_predictions, all_probabilities)
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}")
            # Fall back to simple averaging
            ensemble_preds, ensemble_probs = self._weighted_vote(all_predictions, all_probabilities)
        
        # Apply confidence threshold if enabled
        if self.confidence_threshold > 0:
            # For each prediction, if max probability is below threshold, change to hold (0)
            max_probs = ensemble_probs.max(axis=1)
            ensemble_preds[max_probs < self.confidence_threshold] = 0
        
        return ensemble_preds, ensemble_probs
    
    def _predict_time_series(self, model_state: Dict, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a time series model.
        
        Parameters:
        -----------
        model_state : Dict
            The loaded model state
        data : Dict[str, pd.DataFrame]
            Dictionary of DataFrames, where keys are timeframes
        
        Returns:
        --------
        tuple
            (predictions, probabilities)
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Create a model instance
        # 2. Load the state dict
        # 3. Preprocess the data
        # 4. Make predictions
        
        # For now, we'll simulate the process with random predictions
        n_samples = len(data[next(iter(data))])
        
        # Generate random probabilities (placeholder)
        probabilities = np.random.rand(n_samples, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Convert to predictions (-1, 0, 1 for sell, hold, buy)
        predictions = np.argmax(probabilities, axis=1) - 1
        
        return predictions, probabilities
    
    def _predict_reinforcement(self, model_path: str, data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a reinforcement learning model.
        
        Parameters:
        -----------
        model_path : str
            Path to the RL model
        data : Dict[str, pd.DataFrame]
            Dictionary of DataFrames, where keys are timeframes
        
        Returns:
        --------
        tuple
            (predictions, probabilities)
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Load the agent
        # 2. Create the environment
        # 3. Run the agent to get actions
        
        # For now, we'll simulate the process with random predictions
        n_samples = len(data[next(iter(data))])
        
        # Generate random probabilities (placeholder)
        probabilities = np.random.rand(n_samples, 3)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        # Convert to predictions (-1, 0, 1 for sell, hold, buy)
        predictions = np.argmax(probabilities, axis=1) - 1
        
        return predictions, probabilities
    
    def _weighted_vote(self, all_predictions: List[np.ndarray], all_probabilities: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine predictions using weighted voting.
        
        Parameters:
        -----------
        all_predictions : List[np.ndarray]
            List of prediction arrays from each model
        all_probabilities : List[np.ndarray]
            List of probability arrays from each model
        
        Returns:
        --------
        tuple
            (predictions, probabilities)
        """
        # Stack predictions and probabilities
        stacked_preds = np.stack(all_predictions)
        stacked_probs = np.stack(all_probabilities)
        
        # Weighted probabilities
        weighted_probs = np.zeros_like(stacked_probs[0])
        for i, probs in enumerate(stacked_probs):
            weight = self.weights[i] if i < len(self.weights) else 1.0 / len(stacked_probs)
            weighted_probs += probs * weight
        
        # Convert to predictions
        if self.use_probabilistic:
            # Use the weighted probabilities directly
            ensemble_preds = np.argmax(weighted_probs, axis=1) - 1  # Convert to -1, 0, 1
        else:
            # Count votes for each class
            votes = np.zeros((stacked_preds.shape[1], 3))  # 3 classes: sell, hold, buy
            
            for i, preds in enumerate(stacked_preds):
                weight = self.weights[i] if i < len(self.weights) else 1.0 / len(stacked_preds)
                
                # Convert -1, 0, 1 to 0, 1, 2 for indexing
                idx_preds = preds + 1
                
                for j, pred_class in enumerate(idx_preds):
                    votes[j, pred_class] += weight
            
            # Get the class with the highest vote
            ensemble_preds = np.argmax(votes, axis=1) - 1  # Convert back to -1, 0, 1
        
        return ensemble_preds, weighted_probs
    
    def _majority_vote(self, all_predictions: List[np.ndarray], all_probabilities: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine predictions using majority voting.
        
        Parameters:
        -----------
        all_predictions : List[np.ndarray]
            List of prediction arrays from each model
        all_probabilities : List[np.ndarray]
            List of probability arrays from each model
        
        Returns:
        --------
        tuple
            (predictions, probabilities)
        """
        # Stack predictions
        stacked_preds = np.stack(all_predictions)
        stacked_probs = np.stack(all_probabilities)
        
        # Count votes for each class
        votes = np.zeros((stacked_preds.shape[1], 3))  # 3 classes: sell, hold, buy
        
        for preds in stacked_preds:
            # Convert -1, 0, 1 to 0, 1, 2 for indexing
            idx_preds = preds + 1
            
            for j, pred_class in enumerate(idx_preds):
                votes[j, pred_class] += 1
        
        # Normalize votes to get "probabilities"
        ensemble_probs = votes / votes.sum(axis=1, keepdims=True)
        
        # Get the class with the most votes
        ensemble_preds = np.argmax(votes, axis=1) - 1  # Convert back to -1, 0, 1
        
        # Handle ties according to tie-breaking strategy
        if self.tie_breaking == 'conservative':
            # If there's a tie, choose hold (0)
            max_votes = votes.max(axis=1)
            for i, row in enumerate(votes):
                if np.sum(row == max_votes[i]) > 1:  # Tie
                    ensemble_preds[i] = 0  # Hold
        
        # Check minimum agreement
        if self.min_agreement > 0:
            agreement = votes.max(axis=1) / votes.sum(axis=1)
            ensemble_preds[agreement < self.min_agreement] = 0  # Set to hold if agreement is too low
        
        return ensemble_preds, ensemble_probs
    
    def evaluate(self, data: Dict[str, pd.DataFrame], true_labels: np.ndarray, 
                prices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate the ensemble model against true labels.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary of DataFrames, where keys are timeframes
        true_labels : np.ndarray
            True labels (-1, 0, 1 for sell, hold, buy)
        prices : np.ndarray, optional
            Array of prices for calculating returns, by default None
        
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Make predictions
        predictions, probabilities = self.predict(data)
        
        # Calculate classification metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1_score': f1_score(true_labels, predictions, average='weighted')
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        metrics['confusion_matrix'] = cm
        
        # Visualize
        visualization = self.config.get('visualization', {})
        if visualization.get('create_confusion_matrix', True):
            self._plot_confusion_matrix(cm, ['Sell', 'Hold', 'Buy'], 
                                      os.path.join(self.output_dir, 'confusion_matrix.png'))
        
        # Calculate trading metrics if prices are provided
        if prices is not None:
            # Simulate trading based on predictions
            returns = self._calculate_trading_returns(predictions, prices)
            
            # Add trading metrics
            metrics.update(self._calculate_trading_metrics(returns))
        
        return metrics
    
    def _calculate_trading_returns(self, predictions: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Calculate returns from trading based on predictions.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Array of predictions (-1, 0, 1 for sell, hold, buy)
        prices : np.ndarray
            Array of prices
        
        Returns:
        --------
        np.ndarray
            Array of returns
        """
        # Calculate price changes
        price_changes = np.diff(prices) / prices[:-1]
        price_changes = np.append(0, price_changes)  # Add 0 for the first period
        
        # Calculate returns based on positions
        returns = predictions * price_changes
        
        # Apply transaction costs (simplified)
        transaction_cost = 0.001  # 0.1%
        position_changes = np.diff(np.append(0, predictions))
        transaction_costs = np.abs(position_changes) * transaction_cost
        
        # Subtract transaction costs
        returns = returns - transaction_costs
        
        return returns
    
    def _calculate_trading_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate trading performance metrics.
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns
        
        Returns:
        --------
        dict
            Dictionary of trading metrics
        """
        # Cumulative returns
        cumulative_return = (1 + returns).cumprod()[-1] - 1
        
        # Sharpe ratio (annualized)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate((1 + returns).cumprod())
        drawdown = ((1 + returns).cumprod() - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Profit factor
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = np.abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        return {
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: str) -> None:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        class_names : List[str]
            Names of classes
        save_path : str
            Path to save the plot
        """
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                  yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Plot normalized values
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names,
                  yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the ensemble model.
        
        Parameters:
        -----------
        path : str, optional
            Path to save the model, by default None (uses output_dir/ensemble_model.pkl)
        """
        if path is None:
            path = os.path.join(self.output_dir, 'ensemble_model.pkl')
        
        # Create a dictionary with all necessary data
        model_data = {
            'config': self.config,
            'weights': self.weights,
            'calibration_models': self.calibration_models
        }
        
        # Save the model
        joblib.dump(model_data, path)
        logger.info(f"Ensemble model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """
        Load an ensemble model from file.
        
        Parameters:
        -----------
        path : str
            Path to the saved model
        
        Returns:
        --------
        EnsembleModel
            Loaded ensemble model
        """
        # Load the model data
        model_data = joblib.load(path)
        
        # Create a temporary config file
        temp_config_path = os.path.join(os.path.dirname(path), 'temp_config.json')
        with open(temp_config_path, 'w') as f:
            json.dump(model_data['config'], f)
        
        # Create the model
        model = cls(temp_config_path)
        
        # Restore additional data
        model.weights = model_data['weights']
        model.calibration_models = model_data['calibration_models']
        
        # Clean up temporary file
        os.remove(temp_config_path)
        
        return model


class TimeSeriesModelWrapper:
    """Wrapper class for time series models to be used with sklearn's CalibratedClassifierCV."""
    
    def __init__(self, model_state):
        self.model_state = model_state
    
    def predict(self, X):
        # This is a placeholder - in a real implementation,
        # you would load the model and make predictions
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        # This is a placeholder - in a real implementation,
        # you would load the model and make probability predictions
        probas = np.random.rand(X.shape[0], 3)
        return probas / probas.sum(axis=1, keepdims=True)
    
    def fit(self, X, y):
        # This is a placeholder - the model is already trained
        return self


def create_ensemble(config_path: str) -> EnsembleModel:
    """
    Create and train an ensemble model.
    
    Parameters:
    -----------
    config_path : str
        Path to the ensemble configuration file
    
    Returns:
    --------
    EnsembleModel
        Trained ensemble model
    """
    # Create ensemble model
    ensemble = EnsembleModel(config_path)
    
    # Save the model
    ensemble.save()
    
    return ensemble 