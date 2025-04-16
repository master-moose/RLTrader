import pandas as pd
import os
import sys

def check_file(file_path):
    print(f"Checking file: {file_path}")
    try:
        if not os.path.exists(file_path):
            print(f"  File does not exist!")
            return

        store = pd.HDFStore(file_path, 'r')
        print(f"  Keys in file: {store.keys()}")
        
        for key in store.keys():
            df = store[key]
            print(f"  Group: {key}, Shape: {df.shape}")
            
            # Check for scaled columns
            scaled_cols = [col for col in df.columns if '_scaled' in col]
            print(f"  Has scaled columns? {len(scaled_cols) > 0} (Count: {len(scaled_cols)})")
            
            if len(scaled_cols) > 0:
                print(f"  First 5 scaled columns: {scaled_cols[:5]}")
                
            print(f"  First 5 regular columns: {df.columns[:5].tolist()}")
            print(f"  Total columns: {len(df.columns)}")
            
        store.close()
    except Exception as e:
        print(f"  Error: {e}")

def main():
    # Check historic data files
    print("CHECKING HISTORIC DATA FILES")
    historic_files = [
        'data/historic/train_data.h5',
        'data/historic/val_data.h5',
        'data/historic/test_data.h5',
        'data/historic/historic_dataset.h5'
    ]
    
    for file_path in historic_files:
        check_file(file_path)
        print("-" * 50)
    
    # Check synthetic data files
    print("\nCHECKING SYNTHETIC DATA FILES")
    synthetic_files = []
    
    # List all files in the synthetic directory
    synthetic_dir = 'data/synthetic'
    if os.path.exists(synthetic_dir):
        synthetic_files = [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir) if f.endswith('.h5')]
        
    if synthetic_files:
        for file_path in synthetic_files:
            check_file(file_path)
            print("-" * 50)
    else:
        print("No synthetic data files found in data/synthetic directory.")
        
        # Try looking for other potential synthetic data locations
        for dir_path in ['synthetic', 'data']:
            if os.path.exists(dir_path):
                h5_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.h5')]
                if h5_files:
                    print(f"Found potential synthetic data files in {dir_path}:")
                    for file_path in h5_files:
                        print(f"  {file_path}")
                        check_file(file_path)
                        print("-" * 50)

if __name__ == "__main__":
    main() 