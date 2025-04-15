#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature normalization utility script.

This script provides utilities for normalizing features in a dataset,
with options to handle already normalized features and multi-key HDF5 files.
"""

import argparse
import json
import logging
import os
import sys
import glob
import pandas as pd
import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import DataLoader only if needed for non-HDF5, or keep for potential future use
# from rl_agent.data.data_loader import DataLoader

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feature_normalizer")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Normalize features in a dataset, supporting multi-key HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common file and directory options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data_path", type=str,
        help="Path to input data file (CSV or HDF5)"
    )
    input_group.add_argument(
        "--input_dir", type=str,
        help="Directory containing H5 files to process"
    )
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output_path", type=str,
        help="Path to save normalized data (CSV or HDF5)"
    )
    output_group.add_argument(
        "--output_dir", type=str,
        help="Directory to save normalized data files"
    )
    
    parser.add_argument(
        "--file_pattern", type=str, default="*.h5",
        help="File pattern for batch processing (e.g., '*.h5', 'train_*.h5')"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively search subdirectories for files (with --input_dir)"
    )
    
    # Normalization options
    parser.add_argument(
        "--method", type=str, default="minmax",
        choices=["minmax", "zscore", "robust"],
        help="Normalization method to use"
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Comma-separated list of features to normalize (normalize all if None)"
    )
    parser.add_argument(
        "--preserve_scaled", action="store_true",
        help="Preserve features already marked with '_scaled' suffix"
    )
    parser.add_argument(
        "--preserve_columns", type=str, default="close,open,high,low,volume",
        help="Essential columns to preserve in original form (comma-separated)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "--keep_originals", action="store_true",
        help="Keep original features alongside normalized ones (don't remove them)"
    )
    parser.add_argument(
        "--suffix", type=str, default="_normalized",
        help="Suffix to add to output filenames when processing a directory"
    )
    
    # Logging and display options
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Save logs to file"
    )
    
    return parser.parse_args()

def normalize_feature(
    data: pd.Series,
    method: str,
    feature_name: str
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Normalize a single feature.
    
    Args:
        data: Series containing the feature data
        method: Normalization method ('minmax', 'zscore', or 'robust')
        feature_name: Name of the feature (for logging)
        
    Returns:
        Tuple of (normalized data, normalization parameters)
    """
    params = {}
    normalized = data.copy()
    
    if method == "minmax":
        min_val = data.min()
        max_val = data.max()
        params["min"] = min_val
        params["max"] = max_val
        
        if max_val > min_val:  # Avoid division by zero
            normalized = (data - min_val) / (max_val - min_val)
        else:
            logger.warning(f"Feature '{feature_name}' has constant value {min_val}. Setting to 0.5.")
            normalized = pd.Series(0.5, index=data.index)
            
    elif method == "zscore":
        mean = data.mean()
        std = data.std()
        params["mean"] = mean
        params["std"] = std
        
        if std > 0:  # Avoid division by zero
            normalized = (data - mean) / std
        else:
            logger.warning(f"Feature '{feature_name}' has zero std. All values are {mean}. Setting to 0.")
            normalized = pd.Series(0.0, index=data.index)
            
    elif method == "robust":
        # Robust scaling using percentiles
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        median = data.median()
        params["q25"] = q25
        params["q75"] = q75
        params["median"] = median
        
        iqr = q75 - q25
        if iqr > 0:  # Avoid division by zero
            normalized = (data - median) / iqr
        else:
            logger.warning(f"Feature '{feature_name}' has zero IQR. Setting to 0.")
            normalized = pd.Series(0.0, index=data.index)
    
    return normalized, params

def normalize_dataset(
    data: pd.DataFrame,
    method: str = "minmax",
    features: Optional[List[str]] = None,
    preserve_scaled: bool = True,
    keep_originals: bool = False,
    preserve_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize a dataset.
    
    Args:
        data: DataFrame to normalize
        method: Normalization method ('minmax', 'zscore', or 'robust')
        features: List of features to normalize (all if None)
        preserve_scaled: Whether to preserve features with '_scaled' suffix
        keep_originals: Whether to keep original features (don't remove them)
        preserve_columns: Essential columns to preserve in original form
        
    Returns:
        Tuple of (normalized DataFrame, normalization parameters)
    """
    df = data.copy()
    normalization_params = {}
    
    # Set up preserve_columns as empty list if None
    preserve_columns = preserve_columns or []
    
    # Determine which features to normalize
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if features is not None:
        # Filter to only include requested features that exist in the data
        features_in_data = [f for f in features if f in numeric_columns]
        if len(features_in_data) == 0:
            logger.warning(f"None of the specified features {features} found in data columns: {numeric_columns}. Skipping normalization for this dataset.")
            return df, normalization_params
        target_features = features_in_data
        missing_features = [f for f in features if f not in numeric_columns]
        if missing_features:
            logger.warning(f"Specified features not found in data and will be ignored: {missing_features}")
    else:
        target_features = numeric_columns
    
    # Process each feature
    original_features_to_remove = []
    
    for feature in target_features:
        # Skip already scaled features if requested
        if preserve_scaled and feature.endswith("_scaled"):
            logger.info(f"Skipping already normalized feature: {feature}")
            continue
        
        # Skip non-numeric features (should be filtered already, but double-check)
        if feature not in numeric_columns:
            logger.warning(f"Skipping non-numeric feature: {feature}")
            continue
        
        # Skip features with all NaN values
        if df[feature].isnull().all():
            logger.warning(f"Skipping feature '{feature}' because all values are NaN.")
            continue
        
        # Normalize the feature
        logger.debug(f"Normalizing feature: {feature} using {method}")
        normalized_feature, params = normalize_feature(df[feature], method, feature)
        
        # Create new column name with '_scaled' suffix if it doesn't already have it
        new_name = feature if feature.endswith("_scaled") else f"{feature}_scaled"
        
        # Add the normalized feature to the DataFrame
        df[new_name] = normalized_feature
        
        # Store normalization parameters
        normalization_params[feature] = params
        
        # Mark original feature for removal if we created a new column
        # BUT never remove preserved columns
        if new_name != feature and not keep_originals and feature not in preserve_columns:
            original_features_to_remove.append(feature)
            logger.debug(f"Created normalized feature: {new_name}")
        elif feature in preserve_columns:
            logger.debug(f"Preserving essential column {feature} in original form")
    
    # Remove original features if requested (except preserved columns)
    unique_features_to_remove = list(set(original_features_to_remove))
    if len(unique_features_to_remove) > 0:
        logger.info(f"Removing {len(unique_features_to_remove)} original features: {unique_features_to_remove}")
        df = df.drop(columns=unique_features_to_remove)
    
    return df, normalization_params

def save_dataset(
    normalized_data_dict: Dict[str, pd.DataFrame],
    norm_params_dict: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str
):
    """
    Save multiple normalized datasets (timeframes) to a single HDF5 file.
    
    Args:
        normalized_data_dict: Dictionary where keys are HDF5 keys (e.g., '/15m')
                             and values are the normalized DataFrames.
        norm_params_dict: Dictionary where keys are HDF5 keys (e.g., '/15m')
                          and values are dictionaries of normalization parameters
                          for that timeframe.
        output_path: Path to save the HDF5 file.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine file format
    file_extension = os.path.splitext(output_path)[1].lower()
    
    if file_extension not in ['.h5', '.hdf5']:
        logger.error(f"Unsupported file format for multi-key saving: {file_extension}. Use .h5 or .hdf5.")
        # Attempt to save the first key to CSV as a fallback? Or just exit?
        if normalized_data_dict:
            first_key = next(iter(normalized_data_dict))
            logger.warning(f"Attempting to save data for key '{first_key}' to CSV instead.")
            csv_output_path = os.path.splitext(output_path)[0] + f"_{first_key.replace('/', '')}.csv"
            normalized_data_dict[first_key].to_csv(csv_output_path, index=True)
            # Save corresponding params
            if first_key in norm_params_dict:
                params_path = os.path.splitext(csv_output_path)[0] + "_norm_params.json"
                with open(params_path, 'w') as f:
                    clean_params = {feat: {k: float(v) for k, v in p.items()} for feat, p in norm_params_dict[first_key].items()}
                    json.dump(clean_params, f, indent=2)
                logger.info(f"Saved parameters for '{first_key}' to {params_path}")
            return # Exit after attempting fallback or if no data
    
    logger.info(f"Saving multiple normalized datasets to HDF5: {output_path}")
    
    try:
        # Use HDFStore for easier multi-key writing and metadata handling
        with pd.HDFStore(output_path, mode='w', complevel=9, complib='blosc') as store:
            # Save each normalized DataFrame under its original key
            for key, df in normalized_data_dict.items():
                logger.info(f"Saving normalized data for key: {key} ({df.shape})")
                store.put(key, df, format='table', data_columns=True)
            
            # Save normalization parameters in a structured way within metadata
            if norm_params_dict:
                # Convert numpy types to standard Python types for JSON compatibility
                serializable_params = {}
                for key, params_for_key in norm_params_dict.items():
                    serializable_params[key] = {}
                    for feature, feature_params in params_for_key.items():
                        serializable_params[key][feature] = {
                            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in feature_params.items()
                        }
                
                # Store the parameters dictionary as JSON string in metadata
                store.get_storer(next(iter(normalized_data_dict.keys()))).attrs.metadata = {
                    'normalization_parameters': json.dumps(serializable_params)
                }
                logger.info("Saved normalization parameters to HDF5 metadata.")
    
    except Exception as e:
        logger.error(f"Error saving multi-key HDF5 file {output_path}: {e}")
        # Clean up potentially corrupted file
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.info(f"Removed potentially corrupted file: {output_path}")

def process_file(
    input_path: str,
    output_path: str,
    method: str = "minmax",
    features: Optional[List[str]] = None,
    preserve_scaled: bool = True,
    force: bool = False,
    keep_originals: bool = False,
    preserve_columns: Optional[List[str]] = None
) -> bool:
    """
    Process a single file, handling multiple keys if it's HDF5.
    
    Args:
        input_path: Path to input file (CSV or HDF5)
        output_path: Path to output file (CSV or HDF5)
        method: Normalization method
        features: List of features to normalize
        preserve_scaled: Whether to preserve already scaled features
        force: Whether to overwrite existing output file
        keep_originals: Whether to keep original features
        preserve_columns: Essential columns to preserve in original form
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if output file exists
        if os.path.exists(output_path) and not force:
            logger.warning(f"Output file {output_path} already exists. Skipping. Use --force to overwrite.")
            return False
        
        file_extension = os.path.splitext(input_path)[1].lower()
        normalized_data_dict = {}
        norm_params_dict = {}
        
        if file_extension in ['.h5', '.hdf5']:
            logger.info(f"Processing HDF5 file: {input_path}")
            try:
                with pd.HDFStore(input_path, mode='r') as store:
                    keys = store.keys()
                    if not keys:
                        logger.warning(f"No keys found in HDF5 file: {input_path}. Skipping.")
                        return False
                    logger.info(f"Found keys: {keys}")
                    
                    for key in keys:
                        logger.info(f"--- Processing key: {key} ---")
                        data = store.get(key)
                        
                        if not isinstance(data, pd.DataFrame):
                            logger.warning(f"Object at key '{key}' is not a DataFrame. Skipping.")
                            continue
                        
                        logger.info(f"Normalizing data for key '{key}' using method: {method}")
                        normalized_data, norm_params = normalize_dataset(
                            data,
                            method=method,
                            features=features,
                            preserve_scaled=preserve_scaled,
                            keep_originals=keep_originals,
                            preserve_columns=preserve_columns
                        )
                        normalized_data_dict[key] = normalized_data
                        if norm_params: # Only add if normalization happened
                            norm_params_dict[key] = norm_params
                        
                        logger.info(f"Finished key '{key}'. Original shape: {data.shape}, Normalized shape: {normalized_data.shape}")
            
            except Exception as e:
                logger.error(f"Error reading HDF5 file {input_path}: {e}")
                return False
        
        elif file_extension == '.csv':
            logger.info(f"Processing CSV file: {input_path}")
            try:
                data = pd.read_csv(input_path, index_col=0, parse_dates=True) # Assuming index is date/time
                logger.info(f"Normalizing data from CSV using method: {method}")
                normalized_data, norm_params = normalize_dataset(
                    data,
                    method=method,
                    features=features,
                    preserve_scaled=preserve_scaled,
                    keep_originals=keep_originals,
                    preserve_columns=preserve_columns
                )
                # Use a default key for CSV output or handle differently?
                # For consistency, maybe save CSVs with key in name if output is HDF5?
                # Or save single key HDF5 if output is HDF5?
                # Let's assume if input is CSV, output should also be CSV or single-key HDF5.
                # Storing in dict with a default key '/data' for now.
                default_key = '/data'
                normalized_data_dict[default_key] = normalized_data
                if norm_params:
                    norm_params_dict[default_key] = norm_params
            
            except Exception as e:
                logger.error(f"Error reading or processing CSV file {input_path}: {e}")
                return False
        else:
            logger.error(f"Unsupported input file format: {file_extension}")
            return False
        
        # Save the processed data
        if not normalized_data_dict:
            logger.warning(f"No data was normalized for file {input_path}. No output generated.")
            return False
        
        # Decide how to save based on output path extension
        output_extension = os.path.splitext(output_path)[1].lower()
        
        if output_extension in ['.h5', '.hdf5']:
            # Save all processed keys to the single HDF5 file
            save_dataset(normalized_data_dict, norm_params_dict, output_path)
        elif output_extension == '.csv':
            if len(normalized_data_dict) > 1:
                logger.warning(f"Output format is CSV, but multiple datasets (keys) were processed from HDF5 input {input_path}.")
                logger.warning("Saving each dataset to a separate CSV file.")
                base_output_path, _ = os.path.splitext(output_path)
                for key, df in normalized_data_dict.items():
                    key_suffix = key.replace('/', '_').strip('_') # Make key filename-safe
                    specific_output_path = f"{base_output_path}_{key_suffix}.csv"
                    logger.info(f"Saving dataset for key '{key}' to {specific_output_path}")
                    df.to_csv(specific_output_path, index=True)
                    # Save corresponding params
                    if key in norm_params_dict:
                        params_path = os.path.splitext(specific_output_path)[0] + "_norm_params.json"
                        with open(params_path, 'w') as f:
                            clean_params = {feat: {k: float(v) for k, v in p.items()} for feat, p in norm_params_dict[key].items()}
                            json.dump(clean_params, f, indent=2)
                        logger.info(f"Saved parameters for '{key}' to {params_path}")
            
            elif normalized_data_dict:
                # Save the single processed dataset (from CSV input or single-key HDF5)
                key = next(iter(normalized_data_dict))
                logger.info(f"Saving normalized data to CSV: {output_path}")
                normalized_data_dict[key].to_csv(output_path, index=True)
                # Save corresponding params
                if key in norm_params_dict:
                    params_path = os.path.splitext(output_path)[0] + "_norm_params.json"
                    with open(params_path, 'w') as f:
                        clean_params = {feat: {k: float(v) for k, v in p.items()} for feat, p in norm_params_dict[key].items()}
                        json.dump(clean_params, f, indent=2)
                    logger.info(f"Saved normalization parameters to: {params_path}")
        else:
            logger.error(f"Unsupported output file format: {output_extension}")
            return False
        
        logger.info(f"Normalization complete for {input_path}. Output saved.")
        return True
    
    except Exception as e:
        logger.error(f"Unexpected error processing file {input_path}: {e}", exc_info=True)
        return False

def batch_process_directory(
    input_dir: str,
    output_dir: str,
    file_pattern: str = "*.h5",
    method: str = "minmax",
    features: Optional[List[str]] = None,
    preserve_scaled: bool = True,
    force: bool = False,
    keep_originals: bool = False,
    recursive: bool = False,
    suffix: str = "_normalized",
    preserve_columns: Optional[List[str]] = None
) -> Tuple[int, int]:
    """
    Process all files in a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        file_pattern: File pattern to match
        method: Normalization method
        features: List of features to normalize
        preserve_scaled: Whether to preserve already scaled features
        force: Whether to overwrite existing output files
        keep_originals: Whether to keep original features
        recursive: Whether to recursively search subdirectories
        suffix: Suffix to add to output filenames
        preserve_columns: Essential columns to preserve in original form
        
    Returns:
        Tuple of (number of files processed, number of files with errors)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all files matching the pattern
    if recursive:
        search_pattern = os.path.join(input_dir, "**", file_pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(search_pattern)
    
    # Filter out directories if pattern matches them
    files = [f for f in files if os.path.isfile(f)]
    
    logger.info(f"Found {len(files)} files matching pattern '{search_pattern}'")
    
    success_count = 0
    error_count = 0
    
    for input_file in files:
        # Determine output path
        rel_path = os.path.relpath(input_file, input_dir)
        
        # Add suffix to filename but preserve extension
        filename = os.path.basename(rel_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}{suffix}{ext}"
        
        # Create output path, preserving directory structure for recursive mode
        if recursive and os.path.dirname(rel_path): # Check if there is a relative dir
            rel_dir = os.path.dirname(rel_path)
            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)
            output_file = os.path.join(output_subdir, new_filename)
        else:
            output_file = os.path.join(output_dir, new_filename)
        
        logger.info(f"Processing {input_file} -> {output_file}")
        
        # Process the file
        success = process_file(
            input_path=input_file,
            output_path=output_file,
            method=method,
            features=features,
            preserve_scaled=preserve_scaled,
            force=force,
            keep_originals=keep_originals,
            preserve_columns=preserve_columns
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
            logger.error(f"Failed to process: {input_file}")
    
    return success_count, error_count

def configure_logging(verbose: bool, log_file: Optional[str] = None):
    """Configure logging based on command line arguments."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    configure_logging(args.verbose, args.log_file)
    
    # Process features list
    features = None
    if args.features:
        features = args.features.split(',')
        logger.info(f"Attempting to normalize specific features: {features}")
    
    # Process preserve_columns list
    preserve_columns = None
    if args.preserve_columns:
        preserve_columns = args.preserve_columns.split(',')
        logger.info(f"Preserving essential columns in original form: {preserve_columns}")
    
    # Single file mode
    if args.data_path:
        logger.info("Running in single file mode.")
        process_file(
            input_path=args.data_path,
            output_path=args.output_path,
            method=args.method,
            features=features,
            preserve_scaled=args.preserve_scaled,
            force=args.force,
            keep_originals=args.keep_originals,
            preserve_columns=preserve_columns
        )
    
    # Directory mode
    else: # args.input_dir must be set
        logger.info("Running in directory mode.")
        success_count, error_count = batch_process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            method=args.method,
            features=features,
            preserve_scaled=args.preserve_scaled,
            force=args.force,
            keep_originals=args.keep_originals,
            recursive=args.recursive,
            suffix=args.suffix,
            preserve_columns=preserve_columns
        )
        
        logger.info(f"Batch processing complete. Processed {success_count} files successfully, {error_count} with errors.")

if __name__ == "__main__":
    main() 