#!/usr/bin/env python3
import os
import sys

def fix_file(file_path):
    try:
        # Try to read the file, which may fail if it contains null bytes
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove null bytes
        content = content.replace('\0', '')
        
        # Write back the clean content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed: {file_path}")
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_directory(directory):
    success_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_file(file_path):
                    success_count += 1
                else:
                    failed_count += 1
    
    return success_count, failed_count

if __name__ == "__main__":
    directory = "crypto_trading_model"
    
    if not os.path.isdir(directory):
        print(f"Directory {directory} not found.")
        sys.exit(1)
    
    print(f"Processing Python files in {directory}...")
    success, failed = process_directory(directory)
    
    print(f"\nSummary:")
    print(f"Successfully fixed: {success} files")
    print(f"Failed to fix: {failed} files")
    
    if failed > 0:
        print("\nWarning: Some files could not be fixed. Check the errors above.")
        sys.exit(1)
    else:
        print("\nAll files processed successfully!")
        sys.exit(0) 