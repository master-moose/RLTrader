#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for a TCN model to predict cryptocurrency prices.
"""

# --- Ensure HDF5 plugin is loaded --- #
try:
    import hdf5plugin
except ImportError:
    print("WARNING: hdf5plugin not found. Install with 'pip install hdf5plugin' if reading compressed HDF5 files.")
# ------------------------------------ #

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


def load_and_prepare_data(h5_path, sequence_length, target_col='close_4h', prediction_steps=1,
                          fit_scalers=True, existing_target_scaler=None):
    """
    Loads data from HDF5, merges timeframes, selects scaled features, and creates sequences with raw target values.

    Args:
        h5_path (str): Path to the HDF5 file.
        sequence_length (int): Length of input sequences.
        target_col (str): Name of the target column (e.g., 'close_4h') - This should be the NON-SCALED version for prediction.
        prediction_steps (int): Number of steps ahead to predict.
        fit_scalers: **UNUSED** (no scaling performed).
        existing_target_scaler: **UNUSED**.

    Returns:
        tuple: (X, y, feature_columns)
                feature_columns (list): List of feature names used.
                y (np.ndarray): Array of raw target values.
    """
    logger.info(f"Loading data from {h5_path}")
    dfs = {}
    # Define keys and corresponding suffixes, using the CORRECT nested paths
    # The suffix applies to the columns *within* the table
    key_suffix_map = {
        '/15m/table': None,  # Base features, no suffix needed for columns
        '/4h/table': '_4h',
        '/1d/table': '_1d'
    }
    base_key_path = '/15m/table' # Path to the base feature set
    base_index_key_path = '/4h/table' # Path for the DataFrame index

    try:
        # Use pandas to read HDF5, leveraging PyTables backend
        for key_path, suffix in key_suffix_map.items():
            try:
                # pd.read_hdf handles opening/closing the file
                df = pd.read_hdf(h5_path, key=key_path)
                logger.info(f"Reading dataset: {key_path} using pandas")

                # --- Post-read processing (Use Index for Timestamp) --- #
                # Check if index is already DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"Index for {key_path} is not DatetimeIndex (type: {type(df.index)}). Attempting conversion.")
                    try:
                        # Try converting assuming it's numeric (unix timestamp)
                        df.index = pd.to_datetime(df.index, unit='s') # Assume seconds first
                    except (ValueError, TypeError):
                        try:
                           df.index = pd.to_datetime(df.index, unit='ms') # Try milliseconds
                        except (ValueError, TypeError):
                            try:
                                # Try converting assuming it's a string representation
                                df.index = pd.to_datetime(df.index)
                            except (ValueError, TypeError) as e:
                                logger.error(f"Failed to convert index to Datetime for {key_path}. Error: {e}", exc_info=True)
                                raise ValueError(f"Cannot interpret index as timestamp in {key_path}")
                
                # Ensure index is named 'timestamp' for consistency if needed later (though merge_asof uses index directly)
                # df.index.name = 'timestamp' 

                # Sort by index (timestamp)
                df.sort_index(inplace=True)

                # Add suffix to columns for non-base timeframes
                if suffix:
                    df = df.add_suffix(suffix)

                # Store using the timeframe name (derived from key_path)
                timeframe_key = key_path.split('/')[1]
                dfs[timeframe_key] = df
                logger.info(f"Loaded {key_path}: {df.shape[0]} rows, {df.shape[1]} columns. Index type: {type(df.index)}")

            except KeyError:
                # Handle case where the key_path doesn't exist in the HDF5 file
                logger.warning(f"Dataset path {key_path} not found in {h5_path}. Some features might be missing.")
                timeframe_key = key_path.split('/')[1]
                dfs[timeframe_key] = pd.DataFrame() if not suffix else pd.DataFrame().add_suffix(suffix)
            except Exception as e:
                # Catch other potential errors during read_hdf or processing
                logger.error(f"Error processing path {key_path} from {h5_path}: {e}", exc_info=True)
                # Assign empty df to avoid breaking later parts
                timeframe_key = key_path.split('/')[1]
                dfs[timeframe_key] = pd.DataFrame() if not suffix else pd.DataFrame().add_suffix(suffix)

        # --- Check if essential dataframes were loaded --- #
        if base_key_path.split('/')[1] not in dfs or dfs[base_key_path.split('/')[1]].empty:
            logger.error(f"Base feature dataset for {base_key_path} failed to load or was empty.")
            raise ValueError(f"Could not load base feature data from {base_key_path}")
        if base_index_key_path.split('/')[1] not in dfs or dfs[base_index_key_path.split('/')[1]].empty:
            logger.error(f"Base index dataset for {base_index_key_path} failed to load or was empty.")
            raise ValueError(f"Could not load base index data from {base_index_key_path}")

    except Exception as e:
        logger.error(f"Failed to load data from HDF5: {e}", exc_info=True)
        raise

    # Use 4h as the base *index* timeframe for alignment and target definition
    base_index_key = '4h' # Now refers to the dictionary key ('15m', '4h', '1d')
    if base_index_key not in dfs or dfs[base_index_key].empty:
         logger.error(f"Base index timeframe '{base_index_key}' data is missing or empty after loading from {h5_path}.")
         raise ValueError(f"Missing or empty '{base_index_key}' data in {h5_path}")

    # Start with the 4h dataframe for index alignment
    aligned_df = dfs[base_index_key].copy()
    logger.info(f"Base Index DataFrame ({base_index_key}): {aligned_df.shape}")

    # Merge other timeframes onto the 4h index
    for timeframe_key, df in dfs.items():
        if timeframe_key != base_index_key and not df.empty:
            aligned_df = aligned_df[~aligned_df.index.duplicated(keep='first')]
            df = df[~df.index.duplicated(keep='first')]
            aligned_df = aligned_df.sort_index()
            df = df.sort_index()

            # Determine merge tolerance based on the timeframe being merged
            if timeframe_key == '1d': tolerance = pd.Timedelta('1 day 1 minute')
            elif timeframe_key == '15m': tolerance = pd.Timedelta('16 minutes')
            else: tolerance = pd.Timedelta('5 minutes')

            logger.info(f"Merging '{timeframe_key}' onto '{base_index_key}' index. Tolerance: {tolerance}")

            # Suffixes were added when loading, so just merge
            merged = pd.merge_asof(
                aligned_df,
                df,
                left_index=True,
                right_index=True,
                direction='backward',
                tolerance=tolerance
            )
            # Check merge quality based on columns of the dataframe being merged
            suffix = key_suffix_map.get(f'/{timeframe_key}/table') # Get suffix using the original path structure pattern
            merged_cols = [c for c in df.columns if c != f'timestamp{suffix if suffix else ""}']
            null_after_merge = merged[merged_cols].isnull().sum().sum()
            if null_after_merge > 0:
                 logger.warning(f"Merge for '{timeframe_key}' resulted in {null_after_merge} null values in its original columns. Check alignment.")

            aligned_df = merged
        elif timeframe_key != base_index_key and df.empty:
             logger.warning(f"Skipping merge for empty dataframe from timeframe '{timeframe_key}'")

    logger.info(f"DataFrame after merging all timeframes: {aligned_df.shape}. Columns: {len(aligned_df.columns)}")

    # --- Target Definition --- #
    # Target column should be the UNSCALED 4h close price
    raw_target_col = target_col # e.g., 'close_4h'
    if raw_target_col not in aligned_df.columns:
        logger.error(f"Target column '{raw_target_col}' not found after merging. Available columns: {list(aligned_df.columns)}")
        raise ValueError(f"Target column '{raw_target_col}' not found.")

    aligned_df['target'] = aligned_df[raw_target_col].shift(-prediction_steps)
    logger.info(f"Created 'target' column by shifting '{raw_target_col}' by {-prediction_steps} steps.")

    # --- Feature Selection (Select only *_scaled features) --- #
    feature_columns = []
    for col in aligned_df.columns:
        # Keep base (15m) scaled features OR suffixed scaled features
        if col.endswith('_scaled') or col.endswith('_scaled_4h') or col.endswith('_scaled_1d'):
             # Basic check for excessive NaNs in potential feature columns BEFORE ffill
             if aligned_df[col].isnull().sum() / len(aligned_df) < 0.5: # Allow up to 50% NaNs pre-fill
                 feature_columns.append(col)
             else:
                 logger.warning(f"Skipping potential feature '{col}' due to excessive NaNs ({aligned_df[col].isnull().sum() / len(aligned_df):.1%}).")

    # Add check if feature_columns is empty
    if not feature_columns:
        logger.error("No suitable feature columns (ending in '_scaled', '_scaled_4h', '_scaled_1d') found or kept after NaN check.")
        raise ValueError("No feature columns selected.")

    logger.info(f"Selected {len(feature_columns)} scaled feature columns: {feature_columns[:5]}...") # Log first few

    # Select only the chosen features and the target column for further processing
    final_df = aligned_df[feature_columns + ['target']].copy()

    # --- Handle NaNs --- #
    initial_rows = len(final_df)
    final_df.dropna(subset=['target'], inplace=True) # Drop rows where target is NaN
    logger.info(f"Dropped {initial_rows - len(final_df)} rows with NaN target.")

    # Forward fill remaining NaNs in feature columns (from merging)
    final_df.fillna(method='ffill', inplace=True)

    # Final drop of any remaining NaNs (e.g., at the beginning after ffill)
    rows_before_final_na_drop = len(final_df)
    final_df.dropna(inplace=True)
    if len(final_df) < rows_before_final_na_drop:
        logger.warning(f"Dropped an additional {rows_before_final_na_drop - len(final_df)} rows due to NaNs after forward fill.")

    if final_df.empty:
         logger.error("DataFrame is empty after processing NaNs and selecting features.")
         raise ValueError("No valid data remaining after preprocessing.")

    logger.info(f"Final DataFrame shape after NaN handling: {final_df.shape}")

    # Separate features and target
    features_df = final_df[feature_columns]
    target_series = final_df['target']

    # --- Target values are used directly (NO SCALING) --- #
    target_raw = target_series.values
    logger.info("Using raw target values.")

    # Features are already scaled, directly use their values
    features_final = features_df.values
    logger.info(f"Using pre-scaled features directly. Shape: {features_final.shape}")

    # --- Create Sequences --- #
    logger.info(f"Creating sequences of length {sequence_length}...")
    X, y = [], []
    # Check if enough data for at least one sequence
    if len(features_final) < sequence_length:
        # This check applies regardless of fit_scalers now
        logger.error(f"Not enough data ({len(features_final)} rows) in {h5_path} to create sequences of length {sequence_length}. Returning empty arrays.")
        return np.array([]), np.array([]), feature_columns

    # Create sequences
    for i in range(len(features_final) - sequence_length + 1):
        X.append(features_final[i:(i + sequence_length)])
        # Target corresponds to the end of the sequence for price prediction
        # If prediction_steps=1, target is one step after sequence end
        target_idx = i + sequence_length -1 + prediction_steps
        if target_idx < len(target_raw):
            y.append(target_raw[target_idx]) # Use raw target value
        else:
            # This happens if prediction_steps pushes index out of bounds near the end
            # We need to shorten X to match the available y
            X = X[:-1] # Remove the last sequence added
            break

    X = np.array(X)
    y = np.array(y)

    # Final check if sequence creation resulted in empty arrays or mismatch
    if X.shape[0] == 0 or y.shape[0] == 0:
        logger.warning(f"Sequence creation resulted in empty arrays for {h5_path}.")
        return np.array([]), np.array([]), feature_columns
    elif X.shape[0] != y.shape[0]:
         logger.warning(f"Sequence length mismatch after creation: X={X.shape[0]}, y={y.shape[0]}. Trimming X.")
         min_len = min(X.shape[0], y.shape[0])
         X = X[:min_len]
         y = y[:min_len]
         if X.shape[0] == 0:
             logger.error("Trimming resulted in empty sequences for training data.")
             raise ValueError("Empty sequences after trimming training data.")

    logger.info(f"Created {len(X)} sequences.")
    logger.info(f"Feature sequence shape: {X.shape}") # (num_sequences, seq_len, num_features)
    logger.info(f"Target shape: {y.shape}") # (num_sequences,)

    # Return the features, raw targets, and the list of feature names used
    return X, y, feature_columns

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

        # Check if train_loader is empty
        if len(train_loader) == 0:
            logger.warning(f"Epoch {epoch+1}: Training loader is empty, skipping training phase.")
            avg_train_loss = 0.0 # Or perhaps np.nan?
        else:
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
        # Check if val_loader is empty
        if len(val_loader) == 0:
             logger.warning(f"Epoch {epoch+1}: Validation loader is empty, skipping validation phase.")
             avg_val_loss = float('inf') # Assign high loss if no validation data
        else:
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    epoch_val_loss += loss.item()
            avg_val_loss = epoch_val_loss / len(val_loader)

        history['val_loss'].append(avg_val_loss)
        epoch_duration = time.time() - start_time

        # Handle potential division by zero if loaders were empty
        train_loss_str = f"{avg_train_loss:.6f}" if len(train_loader) > 0 else "N/A (empty loader)"
        val_loss_str = f"{avg_val_loss:.6f}" if len(val_loader) > 0 else "N/A (empty loader)"


        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss_str}, "
                    f"Val Loss: {val_loss_str}, "
                    f"Duration: {epoch_duration:.2f}s")

        # Early stopping logic requires a valid avg_val_loss
        if len(val_loader) > 0:
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
        else:
             # If no validation data, train for full epochs unless training data also runs out
             if len(train_loader) == 0:
                  logger.warning("Both train and validation loaders are empty. Stopping training.")
                  break


    logger.info("Training finished.")
    # Load the best model weights back if they exist
    if os.path.exists('best_tcn_model.pth'):
         logger.info("Loading best model weights.")
         model.load_state_dict(torch.load('best_tcn_model.pth'))
    return history

def evaluate_model(model, test_loader, criterion, device):
    """Evaluates the model on the test set."""
    # Check if test_loader is empty
    if len(test_loader) == 0:
        logger.warning("Test loader is empty. Skipping evaluation.")
        return [], [], float('nan') # Return empty lists and NaN loss

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

            # Get predictions and targets directly (they are already on the original scale)
            preds_original = outputs.cpu().numpy()
            targets_original = batch_targets.cpu().numpy()

            all_preds.extend(preds_original.flatten())
            all_targets.extend(targets_original.flatten())

    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Average Test Loss (Original Scale): {avg_test_loss:.6f}") # Loss is now on original scale

    # Calculate metrics on original scale
    if not all_preds or not all_targets: # Check if lists are empty
         logger.warning("No predictions or targets collected during evaluation.")
         mae = float('nan')
         rmse = float('nan')
    else:
         mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
         rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
         logger.info(f"MAE (Original Scale): {mae:.4f}")
         logger.info(f"RMSE (Original Scale): {rmse:.4f}")


    return all_preds, all_targets, avg_test_loss

def plot_results(predictions, actuals, filename="prediction_vs_actual.png"):
    """Plots predictions vs actual values."""
    # Check if predictions or actuals are empty
    if not predictions or not actuals:
        logger.warning("Cannot plot results because predictions or actuals are empty.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual Prices', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--', alpha=0.7)
    plt.title('TCN Price Prediction vs Actual')
    plt.xlabel('Time Steps (Test Set)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(filename)
        logger.info(f"Saved prediction plot to {filename}")
    except Exception as e:
        logger.error(f"Failed to save plot '{filename}': {e}")
    plt.close()


# --- Main Execution --- #

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Train a TCN model for price prediction.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the HDF5 training data file.")
    parser.add_argument("--val_data_path", type=str, default=None, help="Optional path to the HDF5 validation data file.")
    parser.add_argument("--test_data_path", type=str, default=None, help="Optional path to the HDF5 test data file.")
    parser.add_argument("--sequence_length", type=int, default=60, help="Input sequence length for TCN.")
    parser.add_argument("--prediction_steps", type=int, default=1, help="Number of steps ahead to predict (default: 1, i.e., next step).")
    parser.add_argument("--target_col", type=str, default="close_4h", help="Column name of the target variable in the 4h dataset.")
    parser.add_argument("--tcn_channels", type=int, nargs='+', default=[64, 128, 256], help="Number of channels in TCN layers.")
    parser.add_argument("--tcn_kernel_size", type=int, default=3, help="Kernel size for TCN convolutions.")
    parser.add_argument("--tcn_dropout", type=float, default=0.2, help="Dropout rate for TCN layers.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for optimizer.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data to use for testing (if --test_data_path is not provided).")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training data to use for validation (if --val_data_path is not provided).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training.")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup --- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # --- Load Training Data & Fit Scalers --- #
    logger.info("--- Loading Training Data ---")
    try:
        X_train, y_train, feature_columns = load_and_prepare_data(
            args.data_path, args.sequence_length, args.target_col, args.prediction_steps,
            fit_scalers=True # Fit scalers on training data
        )
        if X_train.size == 0:
             logger.error("Training data loading resulted in empty sequences. Exiting.")
             sys.exit(1)
        train_dataset = PriceDataset(X_train, y_train)
    except (FileNotFoundError, ValueError, Exception) as e:
         logger.error(f"Failed to load or prepare training data from {args.data_path}: {e}", exc_info=True)
         sys.exit(1)

    # --- Load or Split Validation Data --- #
    logger.info("--- Loading/Splitting Validation Data ---")
    val_dataset = None
    if args.val_data_path:
        if os.path.exists(args.val_data_path):
            try:
                X_val, y_val = load_and_prepare_data(
                    args.val_data_path, args.sequence_length, args.target_col, args.prediction_steps,
                    fit_scalers=False
                )
                if X_val.size > 0:
                     val_dataset = PriceDataset(X_val, y_val)
                     logger.info(f"Loaded validation data from {args.val_data_path}. Size: {len(val_dataset)}")
                else:
                     logger.warning(f"Validation data file {args.val_data_path} resulted in zero sequences. Proceeding without validation set.")
            except (FileNotFoundError, ValueError, Exception) as e:
                logger.warning(f"Failed to load validation data from {args.val_data_path}: {e}. Proceeding without validation set.", exc_info=True)
        else:
             logger.warning(f"Validation data path specified but not found: {args.val_data_path}. Proceeding without validation set.")

    # If validation dataset wasn't loaded, split from training data
    if val_dataset is None:
        logger.info(f"Splitting validation data from training data (split ratio: {args.val_split})...")
        if len(train_dataset) < 2: # Need at least 2 samples to split
            logger.warning("Not enough training data to create a validation split. Proceeding without validation set.")
            # Keep train_dataset as is
        else:
            val_size = int(len(train_dataset) * args.val_split)
            train_size = len(train_dataset) - val_size
            if train_size == 0 or val_size == 0: # Avoid empty splits
                 logger.warning(f"Split resulted in zero size for train ({train_size}) or validation ({val_size}). Adjust split ratio or provide more data. Proceeding without validation split.")
                 # Keep train_dataset as is
            else:
                 train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                                          generator=torch.Generator().manual_seed(args.seed)) # Use generator for reproducibility
                 logger.info(f"Split validation data. New train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # --- Load or Split Test Data --- #
    logger.info("--- Loading/Splitting Test Data ---")
    test_dataset = None
    if args.test_data_path:
        if os.path.exists(args.test_data_path):
            try:
                X_test, y_test = load_and_prepare_data(
                    args.test_data_path, args.sequence_length, args.target_col, args.prediction_steps,
                    fit_scalers=False
                )
                if X_test.size > 0:
                     test_dataset = PriceDataset(X_test, y_test)
                     logger.info(f"Loaded test data from {args.test_data_path}. Size: {len(test_dataset)}")
                else:
                     logger.warning(f"Test data file {args.test_data_path} resulted in zero sequences. Test set will be empty.")
            except (FileNotFoundError, ValueError, Exception) as e:
                logger.warning(f"Failed to load test data from {args.test_data_path}: {e}. Test set will be empty.", exc_info=True)
        else:
             logger.warning(f"Test data path specified but not found: {args.test_data_path}. Test set will be empty.")

    # If test dataset wasn't loaded, split from the *original* training data pool
    if test_dataset is None:
         logger.info(f"Splitting test data from original training data pool (split ratio: {args.test_split})...")
         # Recreate full dataset from original X_train, y_train before validation split (if any)
         full_initial_dataset = PriceDataset(X_train, y_train)
         if len(full_initial_dataset) < 2:
             logger.warning("Not enough original training data to create a test split. Test set will be empty.")
             train_dataset_final = train_dataset # Use the potentially validation-split train set
         else:
             test_size_split = int(len(full_initial_dataset) * args.test_split)
             train_val_size_split = len(full_initial_dataset) - test_size_split
             if train_val_size_split == 0 or test_size_split == 0:
                  logger.warning(f"Split resulted in zero size for train/val pool ({train_val_size_split}) or test ({test_size_split}). Adjust split ratio or provide more data. Test set will be empty.")
                  train_dataset_final = train_dataset # Use the potentially validation-split train set
             else:
                  # Split original data pool into train/val leftovers and test set
                  train_val_pool_dataset, test_dataset = random_split(full_initial_dataset, [train_val_size_split, test_size_split],
                                                                     generator=torch.Generator().manual_seed(args.seed)) # Use generator
                  logger.info(f"Split test data. Test size: {len(test_dataset)}")

                  # Re-split train/val from the train_val_pool_dataset if val wasn't loaded separately
                  if not args.val_data_path:
                       logger.info("Re-splitting validation data from the remaining pool...")
                       if len(train_val_pool_dataset) < 2:
                            logger.warning("Not enough data in pool for train/val re-split. Using pool as train set.")
                            train_dataset_final = train_val_pool_dataset
                            val_dataset = None # Reset val dataset as it couldn't be split
                       else:
                            # Calculate val_size relative to the *new* pool size
                            val_split_ratio_adjusted = args.val_split / (1.0 - args.test_split) if (1.0 - args.test_split) > 0 else args.val_split
                            val_size_resplit = int(len(train_val_pool_dataset) * val_split_ratio_adjusted)
                            train_size_resplit = len(train_val_pool_dataset) - val_size_resplit

                            if train_size_resplit == 0 or val_size_resplit == 0:
                                 logger.warning(f"Re-split resulted in zero size for train ({train_size_resplit}) or validation ({val_size_resplit}). Using pool as train set.")
                                 train_dataset_final = train_val_pool_dataset
                                 val_dataset = None
                            else:
                                 train_dataset_final, val_dataset = random_split(train_val_pool_dataset, [train_size_resplit, val_size_resplit],
                                                                                 generator=torch.Generator().manual_seed(args.seed + 1)) # Use different seed for re-split
                                 logger.info(f"Re-split train/val. Final train size: {len(train_dataset_final)}, Val size: {len(val_dataset)}")
                  else:
                       # If val_data was loaded separately, the pool is just the final train set
                       train_dataset_final = train_val_pool_dataset
                       logger.info(f"Using remaining pool as final train set. Size: {len(train_dataset_final)}")
                  train_dataset = train_dataset_final # Assign the final train dataset

    # Handle cases where datasets might be None or empty after loading/splitting
    train_dataset = train_dataset if train_dataset is not None else PriceDataset(np.array([]), np.array([]))
    val_dataset = val_dataset if val_dataset is not None else PriceDataset(np.array([]), np.array([]))
    test_dataset = test_dataset if test_dataset is not None else PriceDataset(np.array([]), np.array([]))

    logger.info(f"Final Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")

    # Create DataLoaders
    # Use persistent_workers and pin_memory if CUDA is available for potential speedup
    num_workers = 4 if device == torch.device("cuda") else 0 # Use workers only with CUDA
    pin_memory = True if device == torch.device("cuda") else False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    # Log the number of batches in each loader
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    logger.info(f"Number of test batches: {len(test_loader)}")

    # --- Initialize Model --- #
    # Ensure X_train has the expected shape before accessing shape[2]
    if len(X_train.shape) < 3:
        logger.error(f"Training data features (X_train) have unexpected shape: {X_train.shape}. Expected 3 dimensions. Exiting.")
        sys.exit(1)
    num_features = X_train.shape[2]
    output_size = 1 # Predicting a single price value
    model = TCNPricePredictor(
        input_size=num_features,
        output_size=output_size,
        num_channels=args.tcn_channels,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout
    ).to(device) # Move model to device right away

    logger.info(f"Model initialized:")
    logger.info(f"{model}") # Log model structure separately
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_params:,}")

    # --- Optimizer and Loss --- #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # --- Train --- #
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Cannot train the model. Exiting.")
        sys.exit(1)

    logger.info("--- Starting Training ---")
    train_history = train_model(
        model, train_loader, val_loader, optimizer, criterion, args.epochs, device, args.patience
    )

    # --- Evaluate --- #
    logger.info("--- Starting Evaluation ---")
    predictions, actuals, test_loss = evaluate_model(
        model, test_loader, criterion, device
    )

    # --- Plot Results --- #
    if predictions and actuals:
        logger.info("--- Plotting Results ---")
        plot_results(predictions, actuals)
    else:
        logger.warning("Skipping plotting due to empty predictions or actuals.")


    logger.info("TCN Price Prediction Script Finished.")


if __name__ == "__main__":
    main() 