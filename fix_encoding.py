#!/usr/bin/env python3

"""
Script to fix encoding issues in files
"""

import os

def fix_file_encoding(input_file, output_file=None):
    """Fix encoding issues in a file by creating a new UTF-8 encoded version"""
    if output_file is None:
        output_file = input_file + '.new'
    
    print(f"Reading from {input_file}")
    # Read with error handling
    with open(input_file, 'r', errors='ignore', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Writing to {output_file}")
    # Write with explicit UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if output_file != input_file + '.new':
        return
    
    print(f"Replacing {input_file} with {output_file}")
    # Replace the original file
    os.rename(output_file, input_file)
    print(f"File {input_file} has been rewritten with UTF-8 encoding")

if __name__ == '__main__':
    file_path = 'crypto_trading_model/lstm_lightning.py'
    fix_file_encoding(file_path) 