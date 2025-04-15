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
    
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to input data file (CSV or HDF5)"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save normalized data (CSV or HDF5)"
    )
    parser.add_argument(
        "--data_key", type=str, default=None,
        help="Key for HDF5 file (e.g., '/15m' for 15-minute data)"
    )
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
        "--force", action="store_true",
        help="Overwrite existing output file"
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
    preserve_scaled: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize a dataset.
    
    Args:
        data: DataFrame to normalize
        method: Normalization method ('minmax', 'zscore', or 'robust')
        features: List of features to normalize (all if None)
        preserve_scaled: Whether to preserve features with '_scaled' suffix
        
    Returns:
        Tuple of (normalized DataFrame, normalization parameters)
    """
    df = data.copy()
    normalization_params = {}
    
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
        
        # Remove the original feature if we created a new column
        if new_name != feature:
            logger.info(f"Created normalized feature: {new_name}")
    
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

def main():
    """Main function."""
    args = parse_args()
    
    # Check if output file exists
    if os.path.exists(args.output_path) and not args.force:
        logger.error(f"Output file already exists: {args.output_path}. Use --force to overwrite.")
        sys.exit(1)
    
    # Load data
    logger.info(f"Loading data from: {args.data_path}")
    data_loader = DataLoader(data_path=args.data_path, data_key=args.data_key)
    data = data_loader.load_data()
    
    # Parse features list
    features = None
    if args.features:
        features = args.features.split(',')
        logger.info(f"Normalizing specific features: {features}")
    
    # Normalize data
    logger.info(f"Normalizing using method: {args.method}")
    normalized_data, norm_params = normalize_dataset(
        data,
        method=args.method,
        features=features,
        preserve_scaled=args.preserve_scaled
    )
    
    # Save normalized data
    save_dataset(
        normalized_data,
        args.output_path,
        args.data_key,
        norm_params
    )
    
    logger.info(f"Normalization complete. Data saved to: {args.output_path}")
    logger.info(f"Original data shape: {data.shape}")
    logger.info(f"Normalized data shape: {normalized_data.shape}")
    
    # Show some samples of normalized features
    scaled_features = [col for col in normalized_data.columns if col.endswith('_scaled')]
    if scaled_features:
        logger.info(f"Normalized features sample (first 5 rows):")
        logger.info(normalized_data[scaled_features].head(5))

if __name__ == "__main__":
    main() 