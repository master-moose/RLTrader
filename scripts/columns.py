import pandas as pd
import sys

file_path = 'data/historic_norm/train_data_normalized.h5'
keys_to_check = ['/15m', '/4h', '/1d'] # Add/remove keys as needed

try:
    with pd.HDFStore(file_path, mode='r') as store:
        print(f"Checking file: {file_path}")
        all_keys = store.keys()
        print(f"Available keys: {all_keys}")

        for key in keys_to_check:
            if key in all_keys:
                print(f"\n--- Columns in key: {key} ---")
                # Load just the first row to get column names efficiently
                df_sample = pd.read_hdf(store, key, stop=1)
                print(df_sample.columns.tolist())
            else:
                print(f"\n--- Key '{key}' not found ---")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}", file=sys.stderr)
except Exception as e:
    print(f"An error occurred: {e}", file=sys.stderr)