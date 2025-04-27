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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
                          fit_scalers=None, existing_target_scaler=None):
    """
    Loads data from HDF5, merges timeframes, selects scaled features, and creates sequences 
    with percentage change target values.

    Args:
        h5_path (str): Path to the HDF5 file.
        sequence_length (int): Length of input sequences.
        target_col (str): Name of the target column (e.g., 'close_4h') - This should be the NON-SCALED version for calculating % change.
        prediction_steps (int): Number of steps ahead to predict.
        fit_scalers: **UNUSED** (no scaling performed).
        existing_target_scaler: **UNUSED**.

    Returns:
        tuple: (X, y, feature_columns)
                feature_columns (list): List of feature names used.
                y (np.ndarray): Array of target percentage changes.
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

    # Keep the original target column temporarily for % change calculation
    cols_to_keep = feature_columns + ['target', raw_target_col]
    # Ensure no duplicates if raw_target_col somehow ended up in feature_columns
    cols_to_keep = list(dict.fromkeys(cols_to_keep))
    final_df = aligned_df[cols_to_keep].copy()

    # --- Handle NaNs --- #
    initial_rows = len(final_df)
    # Separate features and target
    features_df = final_df[feature_columns]
    target_series = final_df['target']

    # --- Calculate Target: Percentage Change --- #
    # Use the *original* non-shifted, non-scaled target column for calculation base
    # Ensure it exists before proceeding (should be guaranteed by including it in cols_to_keep)
    if raw_target_col not in final_df.columns:
         logger.error(f"Original target column '{raw_target_col}' needed for % change calculation not found in final DataFrame.")
         raise ValueError(f"Missing '{raw_target_col}' for target calculation.")

    # Calculate future price based on the shifted raw target
    future_price = final_df['target']
    # Use the *current* price (before shift) as the base for % change
    current_price = final_df[raw_target_col]

    # Calculate percentage change: (future - current) / current
    # Add small epsilon to prevent division by zero, although unlikely for prices
    epsilon = 1e-10
    target_pct_change = (future_price - current_price) / (current_price + epsilon)

    # Replace potential infinite values (if current_price was near zero) with 0 or NaN
    target_pct_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Forward fill any NaNs introduced by the pct_change calc or division by zero
    target_pct_change.fillna(method='ffill', inplace=True)
    # Drop any remaining NaNs (likely at the beginning)
    initial_target_len = len(target_pct_change)
    target_pct_change.dropna(inplace=True)
    if len(target_pct_change) < initial_target_len:
        logger.warning(f"Dropped {initial_target_len - len(target_pct_change)} rows due to NaNs in target % change calculation.")

    # --- Final Feature & Target Preparation --- #
    # Select only the feature columns *after* pct change calc is done
    features_df = final_df[feature_columns].copy()

    # Align features and the calculated target series using their common index
    common_index = features_df.index.intersection(target_pct_change.index)
    if len(common_index) < len(features_df) or len(common_index) < len(target_pct_change):
        logger.info(f"Aligning features ({len(features_df)}) and target ({len(target_pct_change)}) on common index after NaN handling. New length: {len(common_index)}")
        features_df = features_df.loc[common_index]
        target_pct_change = target_pct_change.loc[common_index]

    # Convert to numpy arrays
    features_final = features_df.values
    target_final = target_pct_change.values

    logger.info(f"Final features shape: {features_final.shape}")
    logger.info(f"Final target shape (percentage change): {target_final.shape}")

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
        # The features sequence
        feature_seq = features_final[i : i + sequence_length]

        # The target corresponds to the percentage change calculated for the *end* of the sequence
        # The `target_final` array is already aligned with `features_final`
        target_val = target_final[i + sequence_length - 1]

        # Append sequence and target
        X.append(feature_seq)
        y.append(target_val)

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

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, patience=10, monitor_metric="val_loss", monitor_mode="min", accuracy_threshold=0.01):
    """
    Trains the TCN model with learning rate scheduling.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: The optimizer (e.g., Adam).
        criterion: The loss function.
        scheduler: Learning rate scheduler (e.g., ReduceLROnPlateau).
        epochs (int): Maximum number of epochs to train.
        device: The device to train on (CPU or CUDA).
        patience (int): Patience for early stopping.
        monitor_metric (str): Validation metric to monitor for LR scheduling and early stopping.
        monitor_mode (str): Mode for the monitor_metric (e.g., 'min' for minimizing loss, 'max' for maximizing accuracy).
        accuracy_threshold (float): The relative threshold for custom accuracy calculation.

    Returns:
        dict: Training history containing train and validation losses.
    """
    model.to(device)
    # Initialize best score based on mode
    best_metric_score = float('inf') if monitor_mode == 'min' else float('-inf')
    patience_counter = 0
    # Initialize history with lists for new validation metrics
    history = {
        'train_loss': [], 'val_loss': [], 'lr': [],
        'val_mae': [], 'val_rmse': [], 'val_dir_acc': [], 'val_custom_acc': []
    }
    epsilon = 1e-10 # For safe division in accuracy metrics

    logger.info(f"Starting training for {epochs} epochs on device: {device}")
    logger.info(f"Monitoring '{monitor_metric}' ({monitor_mode} mode) for early stopping and best model saving.")

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
        epoch_val_preds = []
        epoch_val_targets = []
        avg_val_loss = float('nan') # Default to NaN
        val_mae = float('nan')
        val_rmse = float('nan')
        val_dir_acc = float('nan')
        val_custom_acc = float('nan')

        if len(val_loader) == 0:
             logger.warning(f"Epoch {epoch+1}: Validation loader is empty, skipping validation phase.")
             # Cannot calculate metrics or check for improvement
             current_metric_score = float('nan') # Assign NaN if no validation
        else:
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    epoch_val_loss += loss.item()

                    # Collect predictions and targets for metric calculation
                    epoch_val_preds.extend(outputs.cpu().numpy().flatten())
                    epoch_val_targets.extend(batch_targets.cpu().numpy().flatten())

            avg_val_loss = epoch_val_loss / len(val_loader)

            # Calculate validation metrics if predictions were collected
            if epoch_val_preds and epoch_val_targets:
                preds_np = np.array(epoch_val_preds)
                targets_np = np.array(epoch_val_targets)

                val_mae = np.mean(np.abs(preds_np - targets_np))
                val_rmse = np.sqrt(np.mean((preds_np - targets_np)**2))

                correct_direction = (np.sign(preds_np) == np.sign(targets_np))
                correct_direction[targets_np == 0] = (preds_np[targets_np == 0] == 0)
                val_dir_acc = np.mean(correct_direction)

                # Use the passed accuracy_threshold
                relative_error = np.abs(preds_np - targets_np) / (np.abs(targets_np) + epsilon)
                val_custom_acc = np.mean(relative_error < accuracy_threshold)
            else:
                 # Should not happen if val_loader is not empty, but handle defensively
                 logger.warning(f"Epoch {epoch+1}: No predictions/targets collected during validation despite non-empty loader.")

        # Store validation metrics in history
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_dir_acc'].append(val_dir_acc)
        history['val_custom_acc'].append(val_custom_acc)

        # Get the metric score for the current epoch
        # Use dictionary access for flexibility
        current_metric_score = history.get(monitor_metric, [float('nan')])[-1]

        # --- Logging & Scheduler/Early Stopping --- #
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        epoch_duration = time.time() - start_time

        # Format metrics for logging (handle NaNs)
        train_loss_str = f"{avg_train_loss:.6f}" if not np.isnan(avg_train_loss) else "N/A"
        val_loss_str = f"{avg_val_loss:.6f}" if not np.isnan(avg_val_loss) else "N/A"
        val_mae_str = f"{val_mae:.6f}" if not np.isnan(val_mae) else "N/A"
        val_rmse_str = f"{val_rmse:.6f}" if not np.isnan(val_rmse) else "N/A"
        val_dir_acc_str = f"{val_dir_acc:.2%}" if not np.isnan(val_dir_acc) else "N/A"
        val_custom_acc_str = f"{val_custom_acc:.2%}" if not np.isnan(val_custom_acc) else "N/A"


        log_msg = (
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {train_loss_str}, "
            f"Val Loss: {val_loss_str}, "
            f"Val MAE: {val_mae_str}, "
            f"Val RMSE: {val_rmse_str}, "
            f"Val Dir Acc: {val_dir_acc_str}, "
            f"Val Custom Acc: {val_custom_acc_str}, "
            f"LR: {current_lr:.1e}, "
            f"Duration: {epoch_duration:.2f}s"
        )
        logger.info(log_msg)

        # Learning rate scheduler step
        # Check if current_metric_score is a valid number for scheduler
        if not np.isnan(current_metric_score) and scheduler is not None:
             # Handle cases where metric might be infinite (e.g., initial val_loss)
             if np.isfinite(current_metric_score):
                  scheduler.step(current_metric_score)
                  new_lr = optimizer.param_groups[0]['lr']
                  if new_lr < current_lr:
                      logger.info(f"Learning rate reduced to {new_lr:.1e} based on {monitor_metric}")
             else:
                  logger.warning(f"Epoch {epoch+1}: Monitored metric '{monitor_metric}' is not finite ({current_metric_score}). Skipping scheduler step.")

        # Early stopping logic based on the monitored metric
        # Check if current_metric_score is a valid number for comparison
        if not np.isnan(current_metric_score):
             improvement = False
             if monitor_mode == 'min':
                 improvement = current_metric_score < best_metric_score
             else: # monitor_mode == 'max'
                 improvement = current_metric_score > best_metric_score

             if improvement:
                 best_metric_score = current_metric_score
                 patience_counter = 0
                 torch.save(model.state_dict(), 'best_tcn_model.pth')
                 logger.info(f"Monitored metric '{monitor_metric}' improved to {best_metric_score:.6f}. Saved best model.")
             else:
                 patience_counter += 1
                 logger.info(f"Monitored metric '{monitor_metric}' did not improve. Patience: {patience_counter}/{patience}")
                 if patience_counter >= patience:
                     logger.info(f"Early stopping triggered based on '{monitor_metric}'.")
                     break
        # If validation is skipped, only stop if training loader also becomes empty
        elif len(train_loader) == 0:
             logger.warning("Both train and validation loaders are empty. Stopping training.")
             break


    logger.info("Training finished.")
    # Load the best model weights back if they exist
    if os.path.exists('best_tcn_model.pth'):
         logger.info("Loading best model weights.")
         model.load_state_dict(torch.load('best_tcn_model.pth'))
    return history

def evaluate_model(model, test_loader, criterion, device, accuracy_threshold=0.01):
    """
    Evaluates the model on the test set and calculates various metrics.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test set.
        criterion: Loss function (e.g., MSELoss).
        device: Device to run evaluation on (CPU or CUDA).
        accuracy_threshold (float): The relative threshold for custom accuracy calculation.
                                     A prediction is considered accurate if
                                     `abs(pred - actual) / (abs(actual) + epsilon) < threshold`.

    Returns:
        tuple: (predictions, actuals, avg_test_loss, mae, rmse, directional_accuracy, custom_accuracy)
               Returns NaNs for metrics if evaluation cannot be performed.
    """
    # Check if test_loader is empty
    if len(test_loader) == 0:
        logger.warning("Test loader is empty. Skipping evaluation.")
        # Return empty lists and NaNs for all metrics
        return [], [], float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    model.to(device)
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    epsilon = 1e-10 # For safe division

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
    logger.info(f"Average Test Loss (MSE of % Change): {avg_test_loss:.6f}")

    # Calculate metrics if predictions/targets exist
    if not all_preds or not all_targets: # Check if lists are empty
         logger.warning("No predictions or targets collected during evaluation. Cannot calculate metrics.")
         mae = float('nan')
         rmse = float('nan')
         directional_accuracy = float('nan')
         custom_accuracy = float('nan')
    else:
         preds_np = np.array(all_preds)
         targets_np = np.array(all_targets)

         # MAE (Mean Absolute Error)
         mae = np.mean(np.abs(preds_np - targets_np))
         logger.info(f"MAE (% Change): {mae:.6f}")

         # RMSE (Root Mean Squared Error)
         rmse = np.sqrt(np.mean((preds_np - targets_np)**2))
         logger.info(f"RMSE (% Change): {rmse:.6f}")

         # Directional Accuracy
         correct_direction = (np.sign(preds_np) == np.sign(targets_np))
         # Handle cases where target is zero - sign is 0. If prediction is also 0, count as correct direction.
         # If prediction is non-zero but target is zero, it's incorrect direction for this metric's purpose.
         correct_direction[targets_np == 0] = (preds_np[targets_np == 0] == 0)
         directional_accuracy = np.mean(correct_direction)
         logger.info(f"Directional Accuracy: {directional_accuracy:.2%}")

         # Custom Accuracy (within threshold)
         relative_error = np.abs(preds_np - targets_np) / (np.abs(targets_np) + epsilon)
         custom_accuracy = np.mean(relative_error < accuracy_threshold)
         logger.info(f"Accuracy (within +/- {accuracy_threshold*100:.1f}% of actual % change): {custom_accuracy:.2%}")


    return all_preds, all_targets, avg_test_loss, mae, rmse, directional_accuracy, custom_accuracy

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
    plt.ylabel('Percentage Change')
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
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Initial learning rate for optimizer.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data to use for testing (if --test_data_path is not provided).")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training data to use for validation (if --val_data_path is not provided).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader (default: 0 recommended for Windows).")
    parser.add_argument("--accuracy_threshold", type=float, default=0.01, help="Relative threshold for custom accuracy metric (default: 0.01 for 1%).")
    parser.add_argument("--use_scheduler", action="store_true", help="Use ReduceLROnPlateau learning rate scheduler.")
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help="Factor by which the learning rate will be reduced (default: 0.1).")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="Patience for ReduceLROnPlateau scheduler (default: 5).")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-8, help="Minimum learning rate for scheduler (default: 1e-8).")
    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="val_loss",
        choices=['val_loss', 'val_mae', 'val_rmse', 'val_dir_acc', 'val_custom_acc'],
        help="Validation metric to monitor for LR scheduling and early stopping (default: val_loss)."
    )
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
            fit_scalers=None # Fit scalers on training data
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
                    fit_scalers=None
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
                X_test, y_test, _ = load_and_prepare_data(
                    args.test_data_path, args.sequence_length, args.target_col, args.prediction_steps,
                    fit_scalers=None
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
    num_workers = args.num_workers
    pin_memory = True if device == torch.device("cuda") else False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

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

    # --- Determine Monitoring Mode --- #
    if args.monitor_metric in ['val_dir_acc', 'val_custom_acc']:
        monitor_mode = 'max' # Maximize accuracy metrics
    else:
        monitor_mode = 'min' # Minimize loss/error metrics
    logger.info(f"Monitoring metric: {args.monitor_metric} (mode: {monitor_mode})")

    # --- Initialize Scheduler (Optional) --- #
    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=monitor_mode, # Use determined mode
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
            # verbose=True        # Removed: Older PyTorch versions don't support this
        )
        logger.info(f"Using ReduceLROnPlateau scheduler with factor={args.scheduler_factor}, patience={args.scheduler_patience}, min_lr={args.scheduler_min_lr}")
    else:
        logger.info("Learning rate scheduler is disabled.")

    # --- Train --- #
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Cannot train the model. Exiting.")
        sys.exit(1)

    logger.info("--- Starting Training ---")
    train_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
        monitor_mode=monitor_mode,
        accuracy_threshold=args.accuracy_threshold # Pass threshold
    )

    # --- Evaluate --- #
    logger.info("--- Starting Evaluation ---")
    predictions, actuals, test_loss, mae, rmse, dir_acc, custom_acc = evaluate_model(
        model, test_loader, criterion, device, args.accuracy_threshold
    )

    # Log overall evaluation results (already logged within evaluate_model)
    logger.info("--- Evaluation Summary ---")
    logger.info(f"Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Test MAE: {mae:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Test Directional Accuracy: {dir_acc:.2%}")
    logger.info(f"Test Custom Accuracy (+/- {args.accuracy_threshold*100:.1f}%): {custom_acc:.2%}")


    # --- Plot Results --- #
    # Check if predictions and actuals are numpy arrays before plotting
    if isinstance(predictions, np.ndarray) and isinstance(actuals, np.ndarray) and predictions.size > 0 and actuals.size > 0:
        logger.info("--- Plotting Results ---")
        plot_results(predictions, actuals)
    elif isinstance(predictions, list) and isinstance(actuals, list) and predictions and actuals:
         # Convert lists to numpy arrays if they are lists (fallback)
         logger.info("Converting prediction/actual lists to numpy arrays for plotting.")
         plot_results(np.array(predictions), np.array(actuals))
    else:
        logger.warning("Skipping plotting due to empty or invalid predictions/actuals.")


    logger.info("TCN Price Prediction Script Finished.")


if __name__ == "__main__":
    main() 