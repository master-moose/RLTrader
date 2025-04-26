#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map string names to NumPy dtype objects
SUPPORTED_DTYPES = {
    'float32': np.float32,
    'float16': np.float16,
    # Add others here if needed, e.g., 'float64': np.float64
}

def convert_h5_to_float32(input_path: str, output_path: str, keys: list[str], target_dtype_str: str):
    """
    Reads DataFrames from an HDF5 file, converts float64 columns to the target float dtype,
    and saves them to a new HDF5 file.

    Args:
        input_path (str): Path to the input HDF5 file.
        output_path (str): Path to save the output HDF5 file.
        keys (list[str]): List of keys (datasets) to process within the HDF5 file.
        target_dtype_str (str): The target float dtype ('float32', 'float16').
    """
    if target_dtype_str not in SUPPORTED_DTYPES:
        logger.error(f"Unsupported target dtype: {target_dtype_str}. Supported: {list(SUPPORTED_DTYPES.keys())}")
        sys.exit(1)
    target_dtype = SUPPORTED_DTYPES[target_dtype_str]
    logger.info(f"Target conversion dtype: {target_dtype_str} ({target_dtype})")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Starting conversion from {input_path} to {output_path}")
    logger.info(f"Processing keys: {keys}")

    try:
        with pd.HDFStore(output_path, mode='w', complevel=9, complib='blosc') as store_out:
            with pd.HDFStore(input_path, mode='r') as store_in:
                # Check if specified keys exist
                available_keys = store_in.keys()
                valid_keys = [k for k in keys if f'/{k}' in available_keys] # HDFStore adds '/' prefix

                if not valid_keys:
                    logger.error(f"None of the specified keys {keys} found in {input_path}. Available keys: {[k.lstrip('/') for k in available_keys]}")
                    return # Exit function, don't create empty file

                for key in valid_keys:
                    logger.info(f"Processing key: {key}...")
                    try:
                        # Read the DataFrame for the current key
                        df = store_in.get(key)
                        logger.info(f"  Read DataFrame for key '{key}' with shape {df.shape}")

                        # Identify float64 columns
                        float64_cols = df.select_dtypes(include='float64').columns
                        if not float64_cols.empty:
                            logger.info(f"  Found {len(float64_cols)} float64 columns to convert to {target_dtype_str}.")
                            # Create a dictionary for dtype conversion
                            dtype_conversion = {col: target_dtype for col in float64_cols}
                            # Convert columns
                            df = df.astype(dtype_conversion)
                            logger.info(f"  Converted float64 columns to {target_dtype_str}.")
                        else:
                            logger.info(f"  No float64 columns found for key '{key}'.")

                        # Check dtypes after conversion (optional)
                        # logger.debug(f"Dtypes after conversion for key '{key}':\n{df.dtypes}")

                        # Save the converted DataFrame to the output store
                        # Use table format for better query support later if needed
                        store_out.put(key, df, format='table', data_columns=True)
                        logger.info(f"  Saved converted DataFrame for key '{key}' to {output_path}")

                    except Exception as e:
                        logger.error(f"Error processing key '{key}': {e}", exc_info=True)
                        # Continue to the next key even if one fails

    except Exception as e:
        logger.error(f"An error occurred during the HDF5 conversion process: {e}", exc_info=True)
        # Clean up potentially partially written output file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Removed partially created output file: {output_path}")
            except OSError as remove_err:
                logger.error(f"Failed to remove partial output file {output_path}: {remove_err}")

    logger.info(f"Conversion process finished for keys: {valid_keys}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert float64 columns in HDF5 datasets to float32.")
    parser.add_argument("input_h5", help="Path to the input HDF5 file.")
    parser.add_argument("output_h5", help="Path to save the output HDF5 file.")
    parser.add_argument(
        "--keys",
        nargs='+',
        default=['15m', '4h', '1d'],
        help="Space-separated list of keys (datasets) within the HDF5 file to process (default: 15m 4h 1d)."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='float32',
        choices=list(SUPPORTED_DTYPES.keys()),
        help="Target float data type for conversion (default: float32)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_h5_to_float32(args.input_h5, args.output_h5, args.keys, args.dtype) 