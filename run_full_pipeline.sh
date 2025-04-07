#!/bin/bash

# Script to run the full automated crypto trading pipeline
# This script handles:
# 1. Setting up the environment
# 2. Running the data generation
# 3. Training the model with cleaner logging
# 4. Saving the model

set -e  # Exit on error

echo "========== Starting Crypto Trading Pipeline =========="

# Create necessary directories
echo "Setting up directories..."
python -c "from crypto_trading_model.main import setup_directories; setup_directories()"

# Generate synthetic data if it doesn't exist
DATA_DIR="./data/synthetic"
if [ ! -f "$DATA_DIR/train_data.h5" ] || [ ! -f "$DATA_DIR/val_data.h5" ]; then
    echo "Generating synthetic data..."
    python -m crypto_trading_model.main --mode synthetic --config crypto_trading_model/config/synthetic_config.json
else
    echo "Synthetic data already exists, skipping generation."
fi

# Run the automated training with optimized settings
echo "Starting automated training..."
python run_automated_training.py --config crypto_trading_model/config/time_series_config.json

# Print completion message
echo "========== Pipeline Complete =========="
echo "The model has been trained and saved to output/time_series/"
echo "Check logs directory for training logs."

# Push changes to remote repository
echo "Committing and pushing changes..."
git add .
git commit -m "Completed automated training with optimized parameters"
git push

echo "Done!" 