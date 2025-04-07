#!/usr/bin/env python3
import os
import sys
import codecs

def fix_encoding(file_path):
    try:
        # Try with different encodings
        encodings = ['utf-8', 'latin-1', 'ascii']
        content = None
        
        for encoding in encodings:
            try:
                with codecs.open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            # If all encodings fail, use binary mode and replace invalid bytes
            with open(file_path, 'rb') as f:
                binary_content = f.read()
                # Replace null bytes and other control characters
                binary_content = binary_content.replace(b'\x00', b'')
                for i in range(1, 32):
                    if i not in [9, 10, 13]:  # tab, newline, carriage return
                        binary_content = binary_content.replace(bytes([i]), b'')
                content = binary_content.decode('utf-8', errors='replace')
        
        # Write back with UTF-8 encoding
        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed encoding for: {file_path}")
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
                if fix_encoding(file_path):
                    success_count += 1
                else:
                    failed_count += 1
    
    return success_count, failed_count

if __name__ == "__main__":
    directory = "crypto_trading_model"
    
    if not os.path.isdir(directory):
        print(f"Directory {directory} not found.")
        sys.exit(1)
    
    print(f"Fixing encoding for Python files in {directory}...")
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