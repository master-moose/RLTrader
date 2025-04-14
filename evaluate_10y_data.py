#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the generated 10-year synthetic market data.
"""

import os
import sys
import argparse
import logging
from evaluate_synthetic_data import (
    setup_output_dir,
    load_data,
    plot_price_series,
    plot_volume_series,
    plot_candlestick,
    calculate_descriptive_stats,
    analyze_return_distribution,
    analyze_autocorrelation,
    analyze_volatility,
    analyze_volume_price_relationship,
    analyze_indicators
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('10y_data_evaluation.log')
    ]
)
logger = logging.getLogger('synthetic_data_evaluation_10y')

def main():
    """Evaluate and visualize the generated 10-year synthetic market data."""
    parser = argparse.ArgumentParser(
        description="Evaluate the 10-year synthetic data for cryptocurrency trading model"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/synthetic_10y/synthetic_dataset.h5",
        help="Path to the HDF5 data file (default: data/synthetic_10y/synthetic_dataset.h5)"
    )
    parser.add_argument(
        "--timeframes", 
        type=str, 
        nargs='+',
        default=["15m", "4h", "1d"],
        help="Timeframes to analyze (default: 15m 4h 1d)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output/data_evaluation_10y",
        help="Directory to save evaluation results (default: output/data_evaluation_10y)"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=50000,
        help="Number of samples to use for analysis (default: 50000)"
    )
    parser.add_argument(
        "--candlestick_segment", 
        type=int, 
        default=200,
        help="Number of periods for candlestick chart (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Create base output directory
    setup_output_dir(args.output_dir)
    
    # Process each timeframe
    for timeframe in args.timeframes:
        logger.info(f"Evaluating {timeframe} timeframe data...")
        
        # Create timeframe-specific output directory
        tf_output_dir = os.path.join(args.output_dir, timeframe)
        setup_output_dir(tf_output_dir)
        
        try:
            # Load data
            df = load_data(args.data_path, timeframe, args.sample_size)
            
            # Generate visual plots
            plot_price_series(df, tf_output_dir)
            plot_volume_series(df, tf_output_dir)
            plot_candlestick(df, tf_output_dir, args.candlestick_segment)
            
            # Statistical analysis
            calculate_descriptive_stats(df, tf_output_dir)
            analyze_return_distribution(df, tf_output_dir)
            analyze_autocorrelation(df, tf_output_dir)
            analyze_volatility(df, tf_output_dir)
            analyze_volume_price_relationship(df, tf_output_dir)
            analyze_indicators(df, tf_output_dir)
            
            logger.info(f"Completed evaluation of {timeframe} timeframe data")
        except Exception as e:
            logger.error(f"Error evaluating {timeframe} timeframe: {str(e)}", exc_info=True)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 