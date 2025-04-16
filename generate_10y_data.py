#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate 10 years of synthetic market data.
"""

import os
import sys
import argparse
import logging
from generate_data import generate_synthetic_data, setup_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('10y_data_generation.log')
    ]
)
logger = logging.getLogger('crypto_trading_model.data_generation_10y')

def main():
    """Generate 10 years of synthetic market data with improved stability."""
    parser = argparse.ArgumentParser(
        description="Generate 10 years of synthetic market data for cryptocurrency trading"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/synthetic_10y",
        help="Directory where the data will be saved (default: data/synthetic_10y)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="data_generation_config.json",
        help="Path to configuration file (default: data_generation_config.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Calculate number of samples for 10 years
    # 1 year = 365 days
    # 1 day = 24 hours = 96 fifteen-minute intervals
    # 10 years = 10 × 365 × 96 = 350,400 samples
    num_samples = 350400
    
    # Generate synthetic data
    logger.info(f"Generating {num_samples} samples (10 years of 15-minute data)")
    
    try:
        generate_synthetic_data(
            num_samples=num_samples,
            output_dir=args.output_dir,
            config_path=args.config
        )
        logger.info(f"Successfully generated 10 years of data in {args.output_dir}")
    except Exception as e:
        logger.error(f"Error generating data: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("10-year data generation complete!")

if __name__ == "__main__":
    main() 