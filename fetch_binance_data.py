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
    limit = 1000  # Number of candles per request (max for Binance is usually 1000)
    
    # Calculate timeframe in milliseconds for pagination
    timeframe_ms = 0
    if timeframe.endswith('m'):
        timeframe_ms = int(timeframe[:-1]) * 60 * 1000
    elif timeframe.endswith('h'):
        timeframe_ms = int(timeframe[:-1]) * 60 * 60 * 1000
    elif timeframe.endswith('d'):
        timeframe_ms = int(timeframe[:-1]) * 24 * 60 * 60 * 1000
    elif timeframe.endswith('w'):
        timeframe_ms = int(timeframe[:-1]) * 7 * 24 * 60 * 60 * 1000
    
    # Start with current time and go backwards
    end_time = exchange.milliseconds()  # Current timestamp in milliseconds
    
    # Track timestamps to avoid duplicate data
    processed_timestamps = set()
    
    # Counter for retry attempts
    retry_count = 0
    max_retries = 5
    
    # Fetch data in batches
    try:
        while True:
            try:
                # Calculate start time for this batch
                start_time = end_time - (limit * timeframe_ms)
                
                # Fetch OHLCV data
                print(f"Fetching data from {datetime.datetime.fromtimestamp(start_time/1000.0).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.datetime.fromtimestamp(end_time/1000.0).strftime('%Y-%m-%d %H:%M:%S')}")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_time, limit=limit, params={'endTime': end_time})
                
                # Reset retry counter on successful fetch
                retry_count = 0
                
                # If no data received or all data points already processed, we're done
                if not ohlcv or len(ohlcv) == 0:
                    print("No more data available.")
                    break
                
                # Filter out already processed timestamps to avoid duplicates
                new_data = [candle for candle in ohlcv if candle[0] not in processed_timestamps]
                
                # If we didn't get any new data, we're likely at the limit of available data
                if not new_data:
                    print("No new data found. Reached the limit of available historical data.")
                    break
                
                # Add new timestamps to the processed set
                for candle in new_data:
                    processed_timestamps.add(candle[0])
                
                # Append new data to our list
                all_ohlcv_data.extend(new_data)
                
                # Print progress
                start_time_str = datetime.datetime.fromtimestamp(min(candle[0] for candle in new_data) / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                end_time_str = datetime.datetime.fromtimestamp(max(candle[0] for candle in new_data) / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Fetched {len(new_data)} new candles from {start_time_str} to {end_time_str}")
                print(f"Total candles so far: {len(all_ohlcv_data)}")
                
                # Update end_time to fetch the next batch (going backwards in time)
                # Use the earliest timestamp from this batch minus 1ms
                earliest_timestamp = min(candle[0] for candle in ohlcv)
                end_time = earliest_timestamp - 1
                
                # Respect rate limits with an additional delay
                time.sleep(exchange.rateLimit / 1000 * 1.1)  # Add 10% buffer
                
            except ccxt.NetworkError as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Maximum retry attempts ({max_retries}) reached. Exiting...")
                    break
                
                wait_time = 10 * retry_count  # Increasing backoff
                print(f"Network error occurred: {e}. Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
                continue
                
            except ccxt.ExchangeError as e:
                retry_count += 1
                if retry_count > max_retries:
                    print(f"Maximum retry attempts ({max_retries}) reached. Exiting...")
                    break
                
                wait_time = 10 * retry_count  # Increasing backoff
                print(f"Exchange error occurred: {e}. Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
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
            print(f"Total unique candles saved: {len(df)}")
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