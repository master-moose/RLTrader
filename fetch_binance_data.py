#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to fetch historical OHLCV data from Binance exchange for BTC/USDT 15m timeframe.
This script handles pagination, respects rate limits, and saves data to a CSV file.
"""

import os
import time
import datetime
import pandas as pd
import ccxt
from pathlib import Path


def fetch_binance_historical_data(symbol, timeframe, output_file):
    """
    Fetch historical OHLCV data from Binance and save to CSV.
    
    Args:
        symbol (str): The trading symbol (e.g., 'BTC/USDT')
        timeframe (str): The timeframe (e.g., '15m')
        output_file (str): Path to the output CSV file
    """
    # Initialize Binance exchange
    print(f"Connecting to Binance exchange...")
    exchange = ccxt.binance({
        'enableRateLimit': True,  # This enables built-in rate limiter
    })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching historical data for {symbol} with {timeframe} timeframe...")
    
    # Initialize variables for pagination
    all_ohlcv_data = []
    since = None  # Start from the most recent data and go backwards
    limit = 1000  # Number of candles per request (max for Binance is usually 1000)
    
    # Keep track of the earliest timestamp fetched
    earliest_timestamp = float('inf')
    
    # Fetch data in batches
    try:
        while True:
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                # If no data received, break the loop
                if not ohlcv or len(ohlcv) == 0:
                    print("No more data available.")
                    break
                
                # Append data to our list
                all_ohlcv_data.extend(ohlcv)
                
                # Print progress
                start_time = datetime.datetime.fromtimestamp(ohlcv[0][0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                end_time = datetime.datetime.fromtimestamp(ohlcv[-1][0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Fetched {len(ohlcv)} candles from {start_time} to {end_time}")
                
                # Update the earliest timestamp
                batch_earliest = ohlcv[0][0]
                if batch_earliest >= earliest_timestamp:
                    print("Reached overlapping data. Finishing...")
                    break
                
                earliest_timestamp = batch_earliest
                
                # Set 'since' parameter for the next iteration
                # Subtract 1ms to avoid getting duplicate candles
                since = ohlcv[0][0] - 1
                
                # Respect rate limits with an additional delay
                time.sleep(exchange.rateLimit / 1000 * 1.1)  # Add 10% buffer
                
            except ccxt.NetworkError as e:
                print(f"Network error occurred: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue
            except ccxt.ExchangeError as e:
                print(f"Exchange error occurred: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"Unexpected error occurred: {e}. Exiting...")
                break
                
        # Convert the list to a DataFrame
        if all_ohlcv_data:
            print(f"Total candles fetched: {len(all_ohlcv_data)}")
            
            # Create DataFrame
            df = pd.DataFrame(all_ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Sort by timestamp ascending (oldest first)
            df = df.sort_values('timestamp')
            
            # Remove duplicates if any
            df = df.drop_duplicates(subset='timestamp')
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            print(f"Data successfully saved to {output_file}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("No data was fetched. Please check your parameters.")
    
    except Exception as e:
        print(f"An error occurred during data processing: {e}")


if __name__ == "__main__":
    # Define parameters
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "15m"
    OUTPUT_FILE = "data/raw/btc_usdt_15m.csv"
    
    # Execute the function
    fetch_binance_historical_data(SYMBOL, TIMEFRAME, OUTPUT_FILE) 