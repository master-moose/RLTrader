"""
Collects Limit Order Book (LOB) data for a specified symbol from Binance
and stores it directly in an HDF5 file in the format expected by the trading environment.

Modifications from original lob_collector.py:
- Calculates and stores actual bid/ask prices and sizes
- Uses same HDF5 schema as the one created by convert_csv_to_h5.py
- Filters to only include relevant data fields and control file size

Requirements:
  pip install ccxt pandas tables

Usage:
  python deepscalper_agent/lob_collector_v2.py
"""

import ccxt
import pandas as pd
import time
import datetime
import json
from pathlib import Path
import logging
import numpy as np

# --- Configuration ---
EXCHANGE_ID = 'binance'
SYMBOL = 'BTC/USDT'
# How many levels of bids/asks to include in the processed data
LOB_DEPTH = 10
# How many levels to fetch from exchange (fetch more, use LOB_DEPTH)
LOB_FETCH_DEPTH = 100  
# Fetch every second. Adjust based on needs and rate limits
FETCH_INTERVAL_SECONDS = 1.0
# Where to save the processed HDF5 files
OUTPUT_DIR = Path('data/lob_data')
# File naming convention - use date in filename
FILENAME_TEMPLATE = f"{EXCHANGE_ID}_{SYMBOL.replace('/', '')}_lob_{{date}}.h5"
# Key within the HDF5 file
HDF_KEY = 'lob_data'
# Maximum buffer size before writing to disk (to limit memory usage)
MAX_BUFFER_SIZE = 60  # Write every minute if collecting at 1Hz
# Maximum file size in bytes before starting a new daily file
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB per file
# Fallback values
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper Functions ---
def get_current_filename():
    """Generates the filename based on the current date."""
    today = datetime.date.today().strftime('%Y%m%d')
    return OUTPUT_DIR / FILENAME_TEMPLATE.format(date=today)

def get_next_filename(current_file):
    """Generate a numbered filename if the current file is too large."""
    base = current_file.stem
    parent = current_file.parent
    # If the filename already has a numbered suffix, increment it
    if "_part" in base:
        base_parts = base.split("_part")
        new_number = int(base_parts[1]) + 1
        new_name = f"{base_parts[0]}_part{new_number}{current_file.suffix}"
    else:
        # Otherwise, add a part number suffix
        new_name = f"{base}_part1{current_file.suffix}"
    return parent / new_name

def process_orderbook_data(order_book, timestamp_utc):
    """
    Process raw order book data into the format expected by the environment.
    
    Converts to the same format as convert_csv_to_h5.py:
    - timestamp (UTC)
    - ask_price_0...N-1
    - ask_size_0...N-1
    - bid_price_0...N-1
    - bid_size_0...N-1
    
    where N is LOB_DEPTH.
    """
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    
    # Ensure we have enough depth
    if len(bids) < LOB_DEPTH or len(asks) < LOB_DEPTH:
        logging.warning(f"Order book has insufficient depth. Bids: {len(bids)}, Asks: {len(asks)}")
        # Return None if we don't have enough data
        if len(bids) == 0 or len(asks) == 0:
            return None
    
    # Limit to the configured depth
    bids = bids[:LOB_DEPTH]
    asks = asks[:LOB_DEPTH]
    
    # Create a DataFrame with the timestamp
    data = {
        'timestamp': [timestamp_utc]
    }
    
    # Add bid/ask prices and sizes for each level
    for i, (price, size) in enumerate(bids):
        if i < LOB_DEPTH:
            data[f'bid_price_{i}'] = [float(price)]
            data[f'bid_size_{i}'] = [float(size)]
    
    for i, (price, size) in enumerate(asks):
        if i < LOB_DEPTH:
            data[f'ask_price_{i}'] = [float(price)]
            data[f'ask_size_{i}'] = [float(size)]
    
    # Fill any missing levels with NaN
    for i in range(LOB_DEPTH):
        if f'bid_price_{i}' not in data:
            data[f'bid_price_{i}'] = [np.nan]
        if f'bid_size_{i}' not in data:
            data[f'bid_size_{i}'] = [np.nan]
        if f'ask_price_{i}' not in data:
            data[f'ask_price_{i}'] = [np.nan]
        if f'ask_size_{i}' not in data:
            data[f'ask_size_{i}'] = [np.nan]
    
    # Create DataFrame
    return pd.DataFrame(data)


# --- Main Data Collection Logic ---
def collect_lob_data():
    """Fetches and stores LOB data continuously in the expected format."""
    logging.info(f"Initializing {EXCHANGE_ID} exchange connection...")
    try:
        exchange = getattr(ccxt, EXCHANGE_ID)({
            'enableRateLimit': True,  # Enable built-in rate limiter
        })
        exchange.load_markets()  # Pre-load markets to check symbol validity
        if SYMBOL not in exchange.markets:
            logging.error(f"Symbol {SYMBOL} not found on {EXCHANGE_ID}. Exiting.")
            return
        logging.info(f"Successfully connected to {EXCHANGE_ID}.")
    except Exception as e:
        logging.error(f"Error initializing exchange: {e}")
        return

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize data buffer
    data_buffer = []
    retries = 0
    current_file = get_current_filename()
    rows_processed = 0
    
    logging.info(f"Will store data to {current_file}")
    logging.info(f"LOB depth: {LOB_DEPTH} levels (fetching {LOB_FETCH_DEPTH} levels)")
    logging.info(f"Interval: {FETCH_INTERVAL_SECONDS} seconds")
    logging.info(f"Buffer size: {MAX_BUFFER_SIZE} records")
    logging.info(f"Max file size: {MAX_FILE_SIZE_BYTES/1024/1024:.2f} MB")

    while True:
        try:
            start_time = time.time()
            fetch_timestamp_utc = datetime.datetime.now(datetime.timezone.utc)

            # Fetch order book data with the configured depth
            order_book = exchange.fetch_order_book(SYMBOL, limit=LOB_FETCH_DEPTH)
            retries = 0  # Reset retries on successful fetch

            # Process the data into the required format
            processed_data = process_orderbook_data(order_book, fetch_timestamp_utc)
            
            if processed_data is not None:
                # Add to buffer
                data_buffer.append(processed_data)
                rows_processed += 1
                
                # Check if it's time to write to disk
                if len(data_buffer) >= MAX_BUFFER_SIZE:
                    combined_data = pd.concat(data_buffer)
                    
                    # Check if file is too large, if so create a new file
                    if current_file.exists() and current_file.stat().st_size >= MAX_FILE_SIZE_BYTES:
                        next_file = get_next_filename(current_file)
                        logging.info(f"Current file size reached {current_file.stat().st_size/1024/1024:.2f} MB. Starting new file: {next_file}")
                        current_file = next_file
                    
                    # Write buffer to disk
                    with pd.HDFStore(current_file, mode='a') as store:
                        store.append(HDF_KEY, combined_data, format='table', 
                                     data_columns=['timestamp'])
                    
                    logging.info(f"Stored {len(data_buffer)} LOB snapshots. Total rows: {rows_processed}. File size: {current_file.stat().st_size/1024/1024:.2f} MB")
                    # Clear buffer
                    data_buffer = []

            # Calculate sleep time, ensuring we adhere to the interval
            elapsed_time = time.time() - start_time
            sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed_time)
            time.sleep(sleep_time)

        except ccxt.RateLimitExceeded as e:
            logging.warning(f"Rate limit exceeded: {e}. Retrying after delay...")
            # Wait longer than default rate limit suggests
            time.sleep(exchange.rateLimit / 1000 * 1.5)
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            retries += 1
            logging.error(f"Network/Exchange error: {e}. Retry {retries}/{MAX_RETRIES}")
            if retries >= MAX_RETRIES:
                # Try to save remaining buffer before exiting
                if data_buffer:
                    try:
                        combined_data = pd.concat(data_buffer)
                        with pd.HDFStore(current_file, mode='a') as store:
                            store.append(HDF_KEY, combined_data, format='table',
                                     data_columns=['timestamp'])
                        logging.info(f"Saved remaining {len(data_buffer)} snapshots before exit.")
                    except Exception as e:
                        logging.error(f"Failed to save buffer before exit: {e}")
                logging.error("Max retries reached. Exiting.")
                break
            time.sleep(RETRY_DELAY_SECONDS * retries)  # Exponential backoff
        except KeyboardInterrupt:
            # Save any remaining data in the buffer before exiting
            if data_buffer:
                try:
                    combined_data = pd.concat(data_buffer)
                    with pd.HDFStore(current_file, mode='a') as store:
                        store.append(HDF_KEY, combined_data, format='table',
                                     data_columns=['timestamp'])
                    logging.info(f"Saved remaining {len(data_buffer)} snapshots before exit.")
                except Exception as e:
                    logging.error(f"Failed to save buffer before exit: {e}")
            logging.info("Keyboard interrupt received. Exiting.")
            break
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}. Attempting to continue...")
            # Attempt to continue after a delay for unexpected errors
            time.sleep(RETRY_DELAY_SECONDS)


if __name__ == "__main__":
    logging.info("Starting LOB data collector v2...")
    logging.info(f"Symbol: {SYMBOL}")
    logging.info("Press Ctrl+C to stop.")

    try:
        collect_lob_data()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Exiting gracefully.")
    finally:
        logging.info("LOB collector stopped.") 