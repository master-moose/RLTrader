#!/usr/bin/env python3

"""
Script to fix binary encoding issues
"""

def fix_binary_encoding(input_file):
    """Fix encoding issues by reading in binary and removing null bytes"""
    print(f"Reading binary from {input_file}")
    
    # Read file in binary mode
    with open(input_file, 'rb') as f:
        content = f.read()
    
    # Remove null bytes
    content = content.replace(b'\x00', b'')
    
    print(f"Removing null bytes from {input_file}")
    
    # Write back to same file
    with open(input_file, 'wb') as f:
        f.write(content)
    
    print(f"File {input_file} has been fixed")

if __name__ == '__main__':
    file_path = 'crypto_trading_model/lstm_lightning.py'
    fix_binary_encoding(file_path) 