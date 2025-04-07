#!/usr/bin/env python3

"""
Script to fix encoding issues in all Python files in the project
"""

import os
import sys

def fix_file_encoding(file_path):
    """Fix encoding issues in a file by removing null bytes"""
    try:
        # Read in binary mode to preserve content except null bytes
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Check if there are null bytes
        if b'\x00' in content:
            print(f"Removing null bytes from {file_path}")
            content = content.replace(b'\x00', b'')
            
            # Write back with clean content
            with open(file_path, 'wb') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    # Fix each file
    fixed_count = 0
    for file_path in python_files:
        if fix_file_encoding(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == "__main__":
    main() 