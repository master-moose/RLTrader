#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for a TCN model to predict cryptocurrency prices.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Logger Setup --- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- TCN Model Components --- #

class Chomp1d(nn.Module):
    """Removes the padding added by causal convolutions."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """A single block of the TCN."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # Initialize weights using Xavier initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight)
                 nn.init.zeros_(m.bias)


    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """The main TCN architecture."""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size, num_features, sequence_length)
        return self.network(x)

class TCNPricePredictor(nn.Module):
    """Full TCN model with final layer for price prediction."""
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCNPricePredictor, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # Use the output of the last TCN layer directly
        # self.linear = nn.Linear(num_channels[-1], output_size)
        # Output layer projects from the last TCN channel size to the desired output size (1 for price)
        self.output_layer = nn.Linear(num_channels[-1], output_size)


    def forward(self, inputs):
        """Inputs have shape (batch, seq_len, num_features)"""
        # TCN expects (batch, num_features, seq_len)
        y = self.tcn(inputs.transpose(1, 2))
        # Get the output from the last time step of the TCN
        # Output shape from TCN: (batch, num_channels[-1], seq_len)
        last_time_step_output = y[:, :, -1]
        # Pass the last time step output through the final linear layer
        output = self.output_layer(last_time_step_output)
        return output # Shape: (batch, output_size)


# --- Data Handling --- #

class PriceDataset(Dataset):
    """PyTorch Dataset for price prediction."""
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.targets[idx]]) # Target needs to be float tensor


def load_and_prepare_data(h5_path, sequence_length, target_col='close_4h', prediction_steps=1):
    """Loads data from HDF5, merges timeframes, scales, and creates sequences."""
    logger.info(f"Loading data from {h5_path}")
    dfs = {}
    keys = ['/15m', '/4h', '/1d']
    try:
        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                if key in f:
                    logger.info(f"Reading key: {key}")
                    data = f[key][()] # Read dataset into numpy array
                    # Convert structured numpy array to pandas DataFrame
                    df = pd.DataFrame(data)
                    # Assuming 'timestamp' column exists and is suitable for index
                    if 'timestamp' in df.columns:
                         # Convert timestamp if it's not already datetime
                         if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                              # Attempt conversion from seconds or milliseconds
                              try:
                                   # Try seconds first
                                   df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                              except (ValueError, TypeError):
                                   try:
                                        # Try milliseconds if seconds fail
                                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                   except (ValueError, TypeError):
                                        logger.error(f"Could not convert 'timestamp' column in {key} to datetime.")
                                        raise
                         df.set_index('timestamp', inplace=True)
                         df.sort_index(inplace=True)
                         dfs[key.strip('/')] = df
                         logger.info(f"Loaded {key}: {df.shape[0]} rows, {df.shape[1]} columns. Index type: {df.index.dtype}")
                    else:
                        logger.error(f"'timestamp' column not found in key {key}. Cannot process.")
                        raise ValueError(f"'timestamp' column missing in {key}")
                else:
                    logger.warning(f"Key {key} not found in {h5_path}")
                    raise FileNotFoundError(f"Required key {key} not found in {h5_path}")

    except Exception as e:
        logger.error(f"Failed to load data from HDF5: {e}", exc_info=True)
        raise

    # Use 4h as the base timeframe
    base_df = dfs['4h'].copy()
    logger.info(f"Base DataFrame (4h): {base_df.shape}")

    # Merge other timeframes onto the 4h index
    for key, df in dfs.items():
        if key != '4h':
            # Ensure consistent timestamp index type before merging
            base_df = base_df[~base_df.index.duplicated(keep='first')] # Remove duplicates in base index
            df = df[~df.index.duplicated(keep='first')] # Remove duplicates in joining index

            # Use merge_asof for robust time-based joining with tolerance
            # Sort both dataframes by index first
            base_df = base_df.sort_index()
            df = df.sort_index()

            logger.info(f"Merging '{key}' into '4h'. Base index range: {base_df.index.min()} to {base_df.index.max()}, Joining index range: {df.index.min()} to {df.index.max()}")
            # Allow merging based on nearest past timestamp within tolerance (e.g., slightly more than the target frequency)
            tolerance = pd.Timedelta('4 hours 1 minute') if key == '1d' else pd.Timedelta('16 minutes')
            merged = pd.merge_asof(
                base_df,
                df.add_suffix(f'_{key}'), # Add suffix before merging
                left_index=True,
                right_index=True,
                direction='backward', # Use the latest value from 'df' at or before the 'base_df' timestamp
                tolerance=tolerance
            )
            # Check merge quality
            null_after_merge = merged[[c for c in merged.columns if c.endswith(f'_{key}')]].isnull().sum().sum()
            if null_after_merge > 0:
                 logger.warning(f"Merge for '{key}' resulted in {null_after_merge} null values (out of {len(merged) * df.shape[1]}). Consider adjusting tolerance or checking data alignment.")

            base_df = merged # Update base_df with merged columns

    logger.info(f"DataFrame after merging: {base_df.shape}. Columns: {list(base_df.columns)}")

    # --- Feature Engineering / Selection ---
    # Define target column explicitly using the base timeframe name
    if target_col not in base_df.columns:
        logger.error(f"Target column '{target_col}' not found after merging. Available columns: {list(base_df.columns)}")
        raise ValueError(f"Target column '{target_col}' not found.")

    base_df['target'] = base_df[target_col].shift(-prediction_steps)
    logger.info(f"Created 'target' column by shifting '{target_col}' by {-prediction_steps} steps.")

    # Drop rows with NaN target (typically the last 'prediction_steps' rows)
    initial_rows = len(base_df)
    base_df.dropna(subset=['target'], inplace=True)
    logger.info(f"Dropped {initial_rows - len(base_df)} rows with NaN target.")

    # Handle potential NaNs introduced by merging/feature engineering
    # Option 1: Forward fill (simple, assumes persistence)
    base_df.fillna(method='ffill', inplace=True)
    # Option 2: Drop rows with any NaNs (can lose significant data)
    # base_df.dropna(inplace=True)
    # Option 3: Interpolate (linear or other methods)
    # base_df.interpolate(method='linear', inplace=True)

    # Drop remaining NaNs just in case ffill didn't catch everything at the start
    rows_before_final_na_drop = len(base_df)
    base_df.dropna(inplace=True)
    if len(base_df) < rows_before_final_na_drop:
        logger.warning(f"Dropped an additional {rows_before_final_na_drop - len(base_df)} rows due to NaNs after forward fill.")

    if base_df.empty:
         logger.error("DataFrame is empty after processing NaNs. Check data quality and merging logic.")
         raise ValueError("No valid data remaining after preprocessing.")

    logger.info(f"DataFrame after handling NaNs: {base_df.shape}")


    # Separate features and target
    features_df = base_df.drop(columns=['target', target_col]) # Drop original target col too
    target_series = base_df['target']

    # --- Scaling --- #
    logger.info(f"Scaling {features_df.shape[1]} features and the target...")
    feature_scaler = MinMaxScaler()
    # Fit only on non-NaN values if any remained (shouldn't if dropna used)
    valid_features = features_df.values[~np.isnan(features_df.values).any(axis=1)]
    if valid_features.shape[0] == 0:
         logger.error("No valid feature rows left to fit the scaler.")
         raise ValueError("Cannot fit scaler on empty feature data.")
    feature_scaler.fit(valid_features)
    features_scaled = feature_scaler.transform(features_df.values)


    target_scaler = MinMaxScaler()
    valid_targets = target_series.values[~np.isnan(target_series.values)].reshape(-1, 1)
    if valid_targets.shape[0] == 0:
        logger.error("No valid target values left to fit the scaler.")
        raise ValueError("Cannot fit scaler on empty target data.")
    target_scaler.fit(valid_targets)
    target_scaled = target_scaler.transform(target_series.values.reshape(-1, 1)).flatten()

    logger.info("Scaling complete.")


    # --- Create Sequences --- #
    logger.info(f"Creating sequences of length {sequence_length}...")
    X, y = [], []
    if len(features_scaled) < sequence_length + prediction_steps:
         logger.error(f"Not enough data ({len(features_scaled)} rows) to create sequences of length {sequence_length} with prediction step {prediction_steps}.")
         raise ValueError("Insufficient data for sequence creation.")

    # Adjusted loop to ensure we don't go out of bounds
    for i in range(len(features_scaled) - sequence_length): # No +1 needed here
        X.append(features_scaled[i:(i + sequence_length)])
        # The target 'y' corresponds to the time step *after* the sequence ends
        y.append(target_scaled[i + sequence_length -1 + prediction_steps]) # Get target after sequence

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Created {len(X)} sequences.")
    logger.info(f"Feature sequence shape: {X.shape}") # (num_sequences, seq_len, num_features)
    logger.info(f"Target shape: {y.shape}") # (num_sequences,)

    return X, y, feature_scaler, target_scaler

# --- Training and Evaluation --- #

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, patience=10):
    """Trains the TCN model."""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()

        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        epoch_duration = time.time() - start_time

        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"Duration: {epoch_duration:.2f}s")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_tcn_model.pth')
            logger.info(f"Validation loss improved. Saved best model to 'best_tcn_model.pth'")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training finished.")
    # Load the best model weights back
    if os.path.exists('best_tcn_model.pth'):
         logger.info("Loading best model weights.")
         model.load_state_dict(torch.load('best_tcn_model.pth'))
    return history

def evaluate_model(model, test_loader, criterion, device, target_scaler):
    """Evaluates the model on the test set."""
    model.to(device)
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    logger.info("Starting evaluation on the test set...")
    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            test_loss += loss.item()

            # Inverse transform predictions and targets
            preds_scaled = outputs.cpu().numpy()
            targets_scaled = batch_targets.cpu().numpy()

            preds_original = target_scaler.inverse_transform(preds_scaled)
            targets_original = target_scaler.inverse_transform(targets_scaled)

            all_preds.extend(preds_original.flatten())
            all_targets.extend(targets_original.flatten())

    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Average Test Loss (Scaled): {avg_test_loss:.6f}")

    # Calculate metrics on original scale
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
    logger.info(f"MAE (Original Scale): {mae:.4f}")
    logger.info(f"RMSE (Original Scale): {rmse:.4f}")

    return all_preds, all_targets, avg_test_loss

def plot_results(predictions, actuals, filename="prediction_vs_actual.png"):
    """Plots predictions vs actual values."""
    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual Prices', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--', alpha=0.7)
    plt.title('TCN Price Prediction vs Actual')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    logger.info(f"Saved prediction plot to {filename}")
    plt.close()

# --- Main Execution --- #

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train a TCN model for price prediction.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the HDF5 data file.")
    parser.add_argument("--sequence_length", type=int, default=60, help="Input sequence length for TCN.")
    parser.add_argument("--prediction_steps", type=int, default=1, help="Number of steps ahead to predict (default: 1, i.e., next step).")
    parser.add_argument("--target_col", type=str, default="close_4h", help="Column name of the target variable in the 4h dataset.")
    parser.add_argument("--tcn_channels", type=int, nargs='+', default=[64, 128, 256], help="Number of channels in TCN layers.")
    parser.add_argument("--tcn_kernel_size", type=int, default=3, help="Kernel size for TCN convolutions.")
    parser.add_argument("--tcn_dropout", type=float, default=0.2, help="Dropout rate for TCN layers.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data to use for testing.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training data to use for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load data
    try:
        X, y, feature_scaler, target_scaler = load_and_prepare_data(
            args.data_path, args.sequence_length, args.target_col, args.prediction_steps
        )
    except (FileNotFoundError, ValueError, Exception) as e:
         logger.error(f"Failed to load or prepare data: {e}", exc_info=True)
         sys.exit(1)


    # Create dataset
    full_dataset = PriceDataset(X, y)

    # Split data
    test_size = int(len(full_dataset) * args.test_split)
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size])

    val_size = int(train_val_size * args.val_split)
    train_size = train_val_size - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    logger.info(f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    num_features = X.shape[2] # Get number of features from loaded data
    output_size = 1 # Predicting a single price value
    model = TCNPricePredictor(
        input_size=num_features,
        output_size=output_size,
        num_channels=args.tcn_channels,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout
    )
    logger.info(f"Model initialized:
{model}")
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params:,}")


    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Mean Squared Error for regression

    # Train the model
    train_history = train_model(
        model, train_loader, val_loader, optimizer, criterion, args.epochs, device, args.patience
    )

    # Evaluate the model
    predictions, actuals, test_loss = evaluate_model(
        model, test_loader, criterion, device, target_scaler
    )

    # Plot results
    plot_results(predictions, actuals)

    logger.info("TCN Price Prediction Training Complete.")

if __name__ == "__main__":
    main() 