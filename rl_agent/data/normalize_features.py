#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature normalization utility script.

This script provides utilities for normalizing features in a dataset,
with options to handle already normalized features.
"""

import argparse
import logging
import os
import sys
import glob
import pandas as pd
import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_agent.data.data_loader import DataLoader

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("feature_normalizer")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Normalize features in a dataset",
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
        "--data_key", type=str, default=None,
        help="Key for HDF5 file (e.g., '/15m' for 15-minute data)"
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
        features = [f for f in features if f in numeric_columns]
        if len(features) == 0:
            logger.warning("None of the specified features found in data.")
            return df, normalization_params
        target_features = features
    else:
        target_features = numeric_columns
    
    # Process each feature
    original_features_to_remove = []
    
    for feature in target_features:
        # Skip already scaled features if requested
        if preserve_scaled and feature.endswith("_scaled"):
            logger.info(f"Skipping already normalized feature: {feature}")
            continue
        
        # Skip non-numeric features
        if feature not in numeric_columns:
            logger.warning(f"Skipping non-numeric feature: {feature}")
            continue
        
        # Normalize the feature
        logger.info(f"Normalizing feature: {feature} using {method}")
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
            logger.info(f"Created normalized feature: {new_name}")
        elif feature in preserve_columns:
            logger.info(f"Preserving essential column {feature} in original form")
    
    # Remove original features if requested (except preserved columns)
    if len(original_features_to_remove) > 0:
        logger.info(f"Removing {len(original_features_to_remove)} original features")
        df = df.drop(columns=original_features_to_remove)
    
    return df, normalization_params

def save_dataset(
    data: pd.DataFrame,
    output_path: str,
    data_key: Optional[str] = None,
    params: Optional[Dict[str, Dict[str, float]]] = None
):
    """
    Save a dataset to disk.
    
    Args:
        data: DataFrame to save
        output_path: Path to save to
        data_key: Key for HDF5 file
        params: Normalization parameters to save alongside data
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine file format
    file_extension = os.path.splitext(output_path)[1].lower()
    
    if file_extension == '.csv':
        logger.info(f"Saving normalized data to CSV: {output_path}")
        data.to_csv(output_path, index=True)
        
        # Save normalization parameters to a separate JSON file
        if params:
            params_path = os.path.splitext(output_path)[0] + "_norm_params.json"
            import json
            with open(params_path, 'w') as f:
                # Convert any numpy types to Python types
                clean_params = {}
                for feature, feature_params in params.items():
                    clean_params[feature] = {
                        k: float(v) for k, v in feature_params.items()
                    }
                json.dump(clean_params, f, indent=2)
            logger.info(f"Saved normalization parameters to: {params_path}")
            
    elif file_extension in ['.h5', '.hdf5']:
        logger.info(f"Saving normalized data to HDF5: {output_path}")
        
        # Determine the key to use
        key = data_key or '/data'
        if not key.startswith('/'):
            key = '/' + key
            
        data.to_hdf(output_path, key=key, mode='w')
        
        # Save normalization parameters inside the HDF5 file
        if params:
            try:
                with h5py.File(output_path, 'a') as f:
                    # Create a params group if it doesn't exist
                    if '/norm_params' not in f:
                        params_group = f.create_group('/norm_params')
                    else:
                        params_group = f['/norm_params']
                    
                    # Store parameters for each feature
                    for feature, feature_params in params.items():
                        # Create feature group
                        if feature in params_group:
                            feature_group = params_group[feature]
                        else:
                            feature_group = params_group.create_group(feature)
                            
                        # Store parameters
                        for param_name, param_value in feature_params.items():
                            if param_name in feature_group:
                                del feature_group[param_name]
                            feature_group.create_dataset(param_name, data=param_value)
                
                logger.info(f"Saved normalization parameters to HDF5 group: /norm_params")
            except Exception as e:
                logger.error(f"Error saving normalization parameters: {e}")
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        sys.exit(1)

def process_file(
    input_path: str,
    output_path: str,
    data_key: Optional[str] = None,
    method: str = "minmax",
    features: Optional[List[str]] = None,
    preserve_scaled: bool = True,
    force: bool = False,
    keep_originals: bool = False,
    preserve_columns: Optional[List[str]] = None
) -> bool:
    """
    Process a single file.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        data_key: Key for HDF5 file
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
            logger.warning(f"Output file already exists: {output_path}. Skipping. Use --force to overwrite.")
            return False
        
        # Load data
        logger.info(f"Loading data from: {input_path}")
        data_loader = DataLoader(data_path=input_path, data_key=data_key)
        data = data_loader.load_data()
        
        # Normalize data
        logger.info(f"Normalizing data using method: {method}")
        normalized_data, norm_params = normalize_dataset(
            data,
            method=method,
            features=features,
            preserve_scaled=preserve_scaled,
            keep_originals=keep_originals,
            preserve_columns=preserve_columns
        )
        
        # Save normalized data
        save_dataset(
            normalized_data,
            output_path,
            data_key,
            norm_params
        )
        
        logger.info(f"Normalization complete. Data saved to: {output_path}")
        logger.info(f"Original data shape: {data.shape}")
        logger.info(f"Normalized data shape: {normalized_data.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {e}")
        return False

def batch_process_directory(
    input_dir: str,
    output_dir: str,
    file_pattern: str = "*.h5",
    data_key: Optional[str] = None,
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
        data_key: Key for HDF5 files
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
        if recursive:
            rel_dir = os.path.dirname(rel_path)
            output_file = os.path.join(output_dir, rel_dir, new_filename)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            output_file = os.path.join(output_dir, new_filename)
        
        logger.info(f"Processing {input_file} -> {output_file}")
        
        # Process the file
        success = process_file(
            input_path=input_file,
            output_path=output_file,
            data_key=data_key,
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
        logger.info(f"Normalizing specific features: {features}")
    
    # Process preserve_columns list
    preserve_columns = None
    if args.preserve_columns:
        preserve_columns = args.preserve_columns.split(',')
        logger.info(f"Preserving essential columns in original form: {preserve_columns}")
    
    # Single file mode
    if args.data_path:
        process_file(
            input_path=args.data_path,
            output_path=args.output_path,
            data_key=args.data_key,
            method=args.method,
            features=features,
            preserve_scaled=args.preserve_scaled,
            force=args.force,
            keep_originals=args.keep_originals,
            preserve_columns=preserve_columns
        )
    
    # Directory mode
    else:
        success_count, error_count = batch_process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            data_key=args.data_key,
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