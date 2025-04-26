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

def convert_h5_to_float32(input_path: str, output_path: str, keys: list[str]):
    """
    Reads DataFrames from an HDF5 file, converts float64 columns to float32,
    and saves them to a new HDF5 file.

    Args:
        input_path (str): Path to the input HDF5 file.
        output_path (str): Path to save the output HDF5 file.
        keys (list[str]): List of keys (datasets) to process within the HDF5 file.
    """
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
                            logger.info(f"  Found {len(float64_cols)} float64 columns to convert.")
                            # Create a dictionary for dtype conversion
                            dtype_conversion = {col: np.float32 for col in float64_cols}
                            # Convert columns
                            df = df.astype(dtype_conversion)
                            logger.info(f"  Converted float64 columns to float32.")
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_h5_to_float32(args.input_h5, args.output_h5, args.keys) 