#!/usr/bin/env python3

"""
This script reads a file, removes any null bytes, and rewrites it with clean encoding.
"""

import sys

def clean_file(filename):
    """Clean a file of null bytes and rewrite with clean encoding."""
    print(f"Cleaning file: {filename}")
    try:
        # Read the file as binary to check for null bytes
        with open(filename, 'rb') as f:
            content = f.read()
        
        # Remove null bytes
        content = content.replace(b'\x00', b'')
        
        # Write content back to file with UTF-8 encoding
        with open(filename, 'wb') as f:
            f.write(content)
        
        print(f"File cleaned successfully: {filename}")
        return True
    except Exception as e:
        print(f"Error cleaning file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_encoding.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    success = clean_file(filename)
    sys.exit(0 if success else 1) 