# Binance Historical Data Fetcher

This project fetches historical OHLCV (Open, High, Low, Close, Volume) data from Binance for the BTC/USDT trading pair with a 15-minute timeframe.

## Features

- Connects to Binance exchange using the CCXT library
- Fetches historical OHLCV data for BTC/USDT with 15m timeframe
- Handles pagination to fetch as much historical data as possible
- Converts Unix timestamps to human-readable UTC datetime
- Saves data to CSV format with proper headers
- Includes error handling and progress reporting
- Respects Binance rate limits

## Prerequisites

- Python 3.6 or higher
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the data fetching script:

```bash
python fetch_binance_data.py
```

This will:
1. Connect to Binance
2. Fetch historical data for BTC/USDT with 15m timeframe
3. Create a `data/raw/` directory if it doesn't exist
4. Save the data to `data/raw/btc_usdt_15m.csv`

## Output

The script saves data to a CSV file with the following columns:
- timestamp (UTC datetime)
- open
- high
- low
- close
- volume

## Notes

- Binance has limitations on how far back you can fetch historical data
- The script respects Binance's rate limits to avoid being blocked
- If the script is interrupted, running it again will likely result in duplicate data, which is automatically cleaned 