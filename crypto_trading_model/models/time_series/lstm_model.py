"""
LSTM-based time series model for cryptocurrency price prediction.

This module implements a deep learning model using LSTM layers
for time series forecasting of cryptocurrency prices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
import logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Concatenate, Bidirectional,
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import from parent directory
import sys
sys.path.append('../..')
from config import TIME_SERIES_MODEL_SETTINGS, PATHS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lstm_model')

class TimeSeriesLSTM:
    """
    LSTM-based time series forecasting model.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = None,
        n_outputs: int = 1,
        lstm_units: List[int] = None,
        dense_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        model_type: str = 'lstm'
    ):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        n_features : int
            Number of input features
        n_outputs : int
            Number of output variables to predict
        lstm_units : List[int]
            List of LSTM layer units
        dense_units : List[int]
            List of dense layer units
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimization
        model_type : str
            Type of model: 'lstm', 'bilstm', 'cnn_lstm', 'attention'
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        # Default to model settings if not provided
        if lstm_units is None:
            lstm_units = TIME_SERIES_MODEL_SETTINGS['lstm_units']
        if dense_units is None:
            dense_units = TIME_SERIES_MODEL_SETTINGS['dense_units']
            
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_type = model_type
        
        # Placeholder for model
        self.model = None
        
        # Training history
        self.history = None
    
    def build_model(self):
        """
        Build the LSTM model.
        
        Returns:
        --------
        tf.keras.models.Model
            Built model
        """
        if self.n_features is None:
            raise ValueError("Number of features must be specified before building the model")
        
        if self.model_type == 'lstm':
            self.model = self._build_lstm_model()
        elif self.model_type == 'bilstm':
            self.model = self._build_bidirectional_lstm_model()
        elif self.model_type == 'cnn_lstm':
            self.model = self._build_cnn_lstm_model()
        elif self.model_type == 'attention':
            self.model = self._build_attention_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def _build_lstm_model(self):
        """
        Build standard LSTM model.
        
        Returns:
        --------
        tf.keras.models.Model
            LSTM model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.n_outputs))
        
        return model
    
    def _build_bidirectional_lstm_model(self):
        """
        Build bidirectional LSTM model.
        
        Returns:
        --------
        tf.keras.models.Model
            Bidirectional LSTM model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.n_outputs))
        
        return model
    
    def _build_cnn_lstm_model(self):
        """
        Build CNN-LSTM model.
        
        Returns:
        --------
        tf.keras.models.Model
            CNN-LSTM model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.n_features)))
        
        # CNN layers
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(self.n_outputs))
        
        return model
    
    def _build_attention_lstm_model(self):
        """
        Build attention-based LSTM model using Functional API.
        
        Returns:
        --------
        tf.keras.models.Model
            Attention LSTM model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM with attention
        lstm_out = LSTM(self.lstm_units[0], return_sequences=True)(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Simple attention mechanism using GlobalAveragePooling
        attention = Dense(self.lstm_units[0], activation='tanh')(lstm_out)
        attention = Dense(1, activation='softmax')(attention)
        
        # Apply attention weights
        weighted = tf.multiply(lstm_out, attention)
        context = GlobalAveragePooling1D()(weighted)
        
        # Additional dense layers
        x = context
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(self.n_outputs)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 64,
        callbacks: List = None,
        verbose: int = 1
    ):
        """
        Train the model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training input data
        y_train : np.ndarray
            Training target data
        X_val : np.ndarray, optional
            Validation input data
        y_val : np.ndarray, optional
            Validation target data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        callbacks : List, optional
            List of callbacks
        verbose : int
            Verbosity level
            
        Returns:
        --------
        History
            Training history
        """
        # Build model if not already built
        if self.model is None:
            self.n_features = X_train.shape[2]
            self.build_model()
        
        # Setup callbacks if not provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        # Create validation data tuple if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def _get_default_callbacks(self):
        """
        Get default callbacks for training.
        
        Returns:
        --------
        List
            List of callbacks
        """
        # Create model directory if it doesn't exist
        os.makedirs(PATHS['time_series_models'], exist_ok=True)
        
        # Create log directory for TensorBoard
        log_dir = os.path.join(PATHS['logs'], 'tensorboard', 
                              f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=TIME_SERIES_MODEL_SETTINGS['early_stopping_patience'],
            restore_best_weights=True
        )
        
        # Model checkpoint
        model_checkpoint = ModelCheckpoint(
            os.path.join(PATHS['time_series_models'], f"{self.model_type}_best.h5"),
            monitor='val_loss',
            save_best_only=True
        )
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
        
        return [early_stopping, model_checkpoint, reduce_lr, tensorboard]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before predicting")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target data
            
        Returns:
        --------
        Tuple[float, float]
            Loss and mean absolute error
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before evaluating")
        
        return self.model.evaluate(X, y)
    
    def save(self, filepath: str = None):
        """
        Save the model.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        if filepath is None:
            os.makedirs(PATHS['time_series_models'], exist_ok=True)
            filepath = os.path.join(PATHS['time_series_models'], f"{self.model_type}_model.h5")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a saved model.
        
        Parameters:
        -----------
        filepath : str
            Path to saved model
            
        Returns:
        --------
        TimeSeriesLSTM
            Loaded model
        """
        # Create instance
        instance = cls()
        
        # Load model
        instance.model = tf.keras.models.load_model(filepath)
        
        # Extract model parameters from the loaded model
        instance.sequence_length = instance.model.input_shape[1]
        instance.n_features = instance.model.input_shape[2]
        instance.n_outputs = instance.model.output_shape[1]
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def plot_history(self):
        """
        Plot training history.
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot training and validation loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training and validation MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Mean Absolute Error')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

class MultiTimeframeModel:
    """
    Multi-timeframe model that combines predictions from models trained on different timeframes.
    """
    
    def __init__(self, timeframes: List[str]):
        """
        Initialize the multi-timeframe model.
        
        Parameters:
        -----------
        timeframes : List[str]
            List of timeframes to use
        """
        self.timeframes = timeframes
        self.models = {}
        
    def add_model(self, timeframe: str, model: TimeSeriesLSTM):
        """
        Add a model for a specific timeframe.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe identifier
        model : TimeSeriesLSTM
            Model for the timeframe
        """
        self.models[timeframe] = model
    
    def predict(self, X_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Make predictions for all timeframes.
        
        Parameters:
        -----------
        X_dict : Dict[str, np.ndarray]
            Dictionary mapping timeframes to input data
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary mapping timeframes to predictions
        """
        predictions = {}
        
        for timeframe, model in self.models.items():
            if timeframe in X_dict:
                predictions[timeframe] = model.predict(X_dict[timeframe])
        
        return predictions
    
    def ensemble_predict(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make an ensemble prediction by combining predictions from all timeframes.
        
        Parameters:
        -----------
        X_dict : Dict[str, np.ndarray]
            Dictionary mapping timeframes to input data
            
        Returns:
        --------
        np.ndarray
            Ensemble predictions
        """
        # Get individual predictions
        predictions = self.predict(X_dict)
        
        # Combine predictions (simple average)
        if not predictions:
            raise ValueError("No predictions available")
        
        all_preds = list(predictions.values())
        ensemble = np.mean(all_preds, axis=0)
        
        return ensemble
    
    def save_all(self, base_dir: str = None):
        """
        Save all models.
        
        Parameters:
        -----------
        base_dir : str, optional
            Base directory to save models
        """
        if base_dir is None:
            base_dir = PATHS['time_series_models']
        
        os.makedirs(base_dir, exist_ok=True)
        
        for timeframe, model in self.models.items():
            filepath = os.path.join(base_dir, f"{timeframe}_model.h5")
            model.save(filepath)
    
    @classmethod
    def load_all(cls, timeframes: List[str], base_dir: str = None):
        """
        Load models for all timeframes.
        
        Parameters:
        -----------
        timeframes : List[str]
            List of timeframes to load
        base_dir : str, optional
            Base directory where models are saved
            
        Returns:
        --------
        MultiTimeframeModel
            Loaded multi-timeframe model
        """
        if base_dir is None:
            base_dir = PATHS['time_series_models']
        
        # Create instance
        instance = cls(timeframes)
        
        # Load models for each timeframe
        for timeframe in timeframes:
            filepath = os.path.join(base_dir, f"{timeframe}_model.h5")
            if os.path.exists(filepath):
                model = TimeSeriesLSTM.load(filepath)
                instance.add_model(timeframe, model)
            else:
                logger.warning(f"Model for timeframe {timeframe} not found at {filepath}")
        
        return instance

# Function to prepare data for time series model
def prepare_time_series_data(
    data: pd.DataFrame,
    sequence_length: int,
    target_column: str = 'close',
    feature_columns: List[str] = None,
    target_steps: int = 1,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for time series model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with price data and features
    sequence_length : int
        Length of input sequences
    target_column : str
        Column to predict
    feature_columns : List[str], optional
        List of feature columns to use
    target_steps : int
        Number of steps ahead to predict
    train_ratio : float
        Ratio of data to use for training
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, y_train, X_test, y_test
    """
    # Use all columns except target as features if not specified
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    
    # Extract features and target
    features = data[feature_columns].values
    target = data[target_column].values
    
    # Normalize features
    feature_mean = np.mean(features, axis=0)
    feature_std = np.std(features, axis=0)
    features_norm = (features - feature_mean) / (feature_std + 1e-10)
    
    # Normalize target
    target_mean = np.mean(target)
    target_std = np.std(target)
    target_norm = (target - target_mean) / (target_std + 1e-10)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_norm) - sequence_length - target_steps + 1):
        X.append(features_norm[i:i+sequence_length])
        y.append(target_norm[i+sequence_length+target_steps-1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, (target_mean, target_std)

# Usage example
if __name__ == "__main__":
    logger.info("LSTM time series model example")
    
    # Create synthetic data
    import numpy as np
    
    # Synthetic time series data
    n_samples = 1000
    n_features = 5
    
    # Features (random)
    np.random.seed(42)
    features = np.random.randn(n_samples, n_features)
    
    # Target (simple function of features with noise)
    target = 0.5 * features[:, 0] - 0.3 * features[:, 1] + 0.1 * features[:, 2] + 0.2 * np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples) * 0.1
    
    # Create DataFrame
    data = pd.DataFrame(
        data=np.column_stack([features, target]),
        columns=[f'feature_{i}' for i in range(n_features)] + ['target']
    )
    
    # Prepare data
    sequence_length = 50
    X_train, y_train, X_test, y_test, _ = prepare_time_series_data(
        data=data,
        sequence_length=sequence_length,
        target_column='target',
        train_ratio=0.8
    )
    
    # Create and train model
    model = TimeSeriesLSTM(
        sequence_length=sequence_length,
        n_features=X_train.shape[2],
        lstm_units=[64, 32],
        dense_units=[16],
        model_type='lstm'
    )
    
    model.build_model()
    model.model.summary()
    
    # Train model (with smaller number of epochs for example)
    history = model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=20,
        batch_size=32
    )
    
    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test)
    logger.info(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
    
    # Save model
    model.save()
    
    # Plot history
    history_plot = model.plot_history()
    plt.savefig(os.path.join(PATHS['results'], 'lstm_training_history.png'))
    plt.close(history_plot) 