#!/bin/bash

# Script to run the full automated crypto trading pipeline
# This script handles:
# 1. Setting up the environment
# 2. Running the data generation
# 3. Training the model with quality monitoring
# 4. Saving the model and quality assessment
# 5. Automatically retrying with better parameters if needed

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

# Run the automated training with quality monitoring
echo "Starting automated training with quality monitoring..."
python run_automated_training.py --config crypto_trading_model/config/time_series_config.json --min-accuracy 0.55

# Check the exit code from the training script
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully with a high-quality model!"
elif [ $TRAINING_EXIT_CODE -eq 2 ]; then
    echo "⚠️ Training completed but the model quality is questionable."
    echo "   Check the model_info.json file for details on quality issues."
    echo "   You might want to manually adjust parameters or training data."
else
    echo "❌ Training failed! Check the logs for errors."
    exit 1  # Exit with error
fi

# Print completion message
echo "========== Pipeline Complete =========="
echo "The model has been trained and saved to output/time_series/"
echo "Check logs directory for training logs and model_info.json for quality assessment."

# Push changes to remote repository
echo "Committing and pushing changes..."
git add .
git commit -m "Completed automated training with quality monitoring" || echo "No changes to commit"
git push || echo "Could not push to remote"

echo "Done!" 