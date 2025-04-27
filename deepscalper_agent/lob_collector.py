"""
Collects Limit Order Book (LOB) data for a specified symbol from Binance
and stores it incrementally in an HDF5 file using pandas HDFStore.

Requirements:
  pip install ccxt pandas tables

Usage:
  python deepscalper_agent/lob_collector.py
"""

import ccxt
import pandas as pd
import time
import datetime
import os
from pathlib import Path
import logging
import json

# --- Configuration ---
EXCHANGE_ID = 'binance'
SYMBOL = 'BTC/USDT'
# How many levels of bids/asks to fetch (Binance allows 5, 10, 20, 50, 100, 500, 1000)
# Higher limit = more data per snapshot but potentially slower fetch if not needed.
LOB_LIMIT = 100
FETCH_INTERVAL_SECONDS = 1.0  # Fetch every second. Adjust based on needs and rate limits.
OUTPUT_DIR = Path('data/lob_data')
FILENAME_TEMPLATE = f"{EXCHANGE_ID}_{SYMBOL.replace('/', '')}_lob_{{date}}.h5"
HDF_KEY = 'lob_data'  # Key within the HDF5 file
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

# --- Main Data Collection Logic ---
def collect_lob_data():
    """Fetches and stores LOB data continuously."""
    logging.info(f"Initializing {EXCHANGE_ID} exchange connection...")
    try:
        exchange = getattr(ccxt, EXCHANGE_ID)({
            'enableRateLimit': True,  # Enable built-in rate limiter
            # 'rateLimit': 1200, # Default Binance rate limit, enableRateLimit handles this
            # Increase timeout if needed for slower connections
            # 'timeout': 30000,
        })
        exchange.load_markets() # Pre-load markets to check symbol validity
        if SYMBOL not in exchange.markets:
            logging.error(f"Symbol {SYMBOL} not found on {EXCHANGE_ID}. Exiting.")
            return
        logging.info(f"Successfully connected to {EXCHANGE_ID}.")
    except Exception as e:
        logging.error(f"Error initializing exchange: {e}")
        return

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retries = 0
    while True:
        try:
            start_time = time.time()
            fetch_timestamp_utc = datetime.datetime.now(datetime.timezone.utc)

            # Fetch order book data
            order_book = exchange.fetch_order_book(SYMBOL, limit=LOB_LIMIT)
            retries = 0 # Reset retries on successful fetch

            # Prepare data for storage
            # Storing bids/asks as lists within the DataFrame.
            # This is simple but might not be the most HDF5-performant way.
            # Alternatives: Flatten lists, use multi-index, store in separate tables.
            bids_list = order_book.get('bids', [])
            asks_list = order_book.get('asks', [])
            data_to_store = pd.DataFrame({
                'timestamp_utc': [fetch_timestamp_utc],
                'lob_timestamp_ms': [order_book.get('timestamp')], # Exchange LOB timestamp
                'lob_nonce': [order_book.get('nonce')],         # Exchange LOB nonce
                'bids': [json.dumps(bids_list)], # Serialize to JSON string
                'asks': [json.dumps(asks_list)]  # Serialize to JSON string
            })
            # Convert timestamp to ensure compatibility
            data_to_store['timestamp_utc'] = pd.to_datetime(data_to_store['timestamp_utc'])

            # Determine current file and append data
            current_file = get_current_filename()
            logging.debug(f"Appending to {current_file}")
            with pd.HDFStore(current_file, mode='a') as store:
                store.append(HDF_KEY, data_to_store, format='table',
                             data_columns=['timestamp_utc']) # Index timestamp for faster queries

            logging.info(f"Stored LOB snapshot for {SYMBOL} at {fetch_timestamp_utc}")

            # Calculate sleep time, ensuring we adhere to the interval
            elapsed_time = time.time() - start_time
            sleep_time = max(0, FETCH_INTERVAL_SECONDS - elapsed_time)
            logging.debug(f"Fetch took {elapsed_time:.3f}s, sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)

        except ccxt.RateLimitExceeded as e:
            logging.warning(f"Rate limit exceeded: {e}. Retrying after delay...")
            time.sleep(exchange.rateLimit / 1000 * 1.5) # Wait longer than default
        except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
            retries += 1
            logging.error(f"Network/Exchange error: {e}. Retry {retries}/{MAX_RETRIES}")
            if retries >= MAX_RETRIES:
                logging.error("Max retries reached. Exiting.")
                break
            time.sleep(RETRY_DELAY_SECONDS * retries) # Exponential backoff
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}. Attempting to continue...")
            # Attempt to continue after a delay for unexpected errors
            time.sleep(RETRY_DELAY_SECONDS)


if __name__ == "__main__":
    logging.info("Starting LOB data collector...")
    logging.info(f"Symbol: {SYMBOL}")
    logging.info(f"Interval: {FETCH_INTERVAL_SECONDS} seconds")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info("Press Ctrl+C to stop.")

    # Check for dependencies
    try:
        import tables # noqa
    except ImportError:
        logging.error("Dependency 'tables' (pytables) not found.")
        logging.error("Please install it: pip install tables")
        exit(1)

    try:
        collect_lob_data()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Exiting gracefully.")
    finally:
        logging.info("LOB collector stopped.") 