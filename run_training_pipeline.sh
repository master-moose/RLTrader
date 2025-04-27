#!/bin/bash
# run_training_pipeline.sh - Complete pipeline for data collection and model training
set -e  # Exit on error

# Create directory structure if it doesn't exist
mkdir -p data/lob_data
mkdir -p models/dqn_lstm_scalper
mkdir -p logs/dqn_lstm_scalper
mkdir -p logs/dqn_lstm_scalper_continued

# Default values
COLLECT_DATA=false
COLLECT_HOURS=1
CONTINUE_TRAINING=false
TRAIN_FROM_SCRATCH=false
CHECKPOINT=""
TIMESTEPS=1000000
N_ENVS=4
OUTPUT_DIR="data/lob_data"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --collect)
      COLLECT_DATA=true
      shift
      ;;
    --hours)
      COLLECT_HOURS="$2"
      shift
      shift
      ;;
    --continue)
      CONTINUE_TRAINING=true
      shift
      ;;
    --train)
      TRAIN_FROM_SCRATCH=true
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift
      shift
      ;;
    --timesteps)
      TIMESTEPS="$2"
      shift
      shift
      ;;
    --envs)
      N_ENVS="$2"
      shift
      shift
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Function to display help message
display_help() {
  echo "Usage: ./run_training_pipeline.sh [options]"
  echo ""
  echo "Options:"
  echo "  --collect              Collect live data before training"
  echo "  --hours <hours>        Number of hours to collect data (default: 1)"
  echo "  --continue             Continue training from a checkpoint"
  echo "  --train                Train a new model from scratch"
  echo "  --checkpoint <path>    Path to checkpoint for continued training"
  echo "  --timesteps <steps>    Number of timesteps for training (default: 1000000)"
  echo "  --envs <num>           Number of parallel environments (default: 4)"
  echo "  --output <dir>         Output directory for collected data (default: data/lob_data)"
  echo ""
  echo "Example:"
  echo "  ./run_training_pipeline.sh --collect --hours 2 --continue --checkpoint models/dqn_lstm_scalper/dqn_lstm_scalper_final.zip --timesteps 500000"
}

# Show help if no arguments provided
if [[ $COLLECT_DATA == "false" && $CONTINUE_TRAINING == "false" && $TRAIN_FROM_SCRATCH == "false" ]]; then
  display_help
  exit 0
fi

echo "===== Deep Scalper Training Pipeline ====="
echo "Configuration:"
if [[ $COLLECT_DATA == "true" ]]; then
  echo "- Collecting data for $COLLECT_HOURS hours"
  echo "- Output directory: $OUTPUT_DIR"
fi

if [[ $CONTINUE_TRAINING == "true" ]]; then
  if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: Must specify a checkpoint path when using --continue"
    exit 1
  fi
  echo "- Continuing training from checkpoint: $CHECKPOINT"
  echo "- Training for $TIMESTEPS timesteps"
  echo "- Using $N_ENVS parallel environments"
fi

if [[ $TRAIN_FROM_SCRATCH == "true" ]]; then
  echo "- Training new model from scratch"
  echo "- Training for $TIMESTEPS timesteps"
  echo "- Using $N_ENVS parallel environments"
fi

echo ""
echo "Starting pipeline..."
echo ""

# 1. Data Collection
if [[ $COLLECT_DATA == "true" ]]; then
  echo "===== Step 1: Data Collection ====="
  echo "Collecting limit order book data for $COLLECT_HOURS hours..."
  
  # Calculate collection time in seconds
  COLLECTION_SECONDS=$((COLLECT_HOURS * 3600))
  
  # Run data collector with timeout
  timeout $COLLECTION_SECONDS python deepscalper_agent/lob_collector_v2.py || {
    if [[ $? -eq 124 ]]; then
      echo "Data collection completed as planned after $COLLECT_HOURS hours."
    else
      echo "Error during data collection. Check logs."
      exit 1
    fi
  }
  
  echo "Data collection completed."
  echo "Data saved to: $OUTPUT_DIR"
  echo ""
fi

# 2. Training
if [[ $TRAIN_FROM_SCRATCH == "true" ]]; then
  echo "===== Step 2: Training New Model ====="
  echo "Training new model from scratch for $TIMESTEPS timesteps..."
  
  # Set environment variables to override defaults
  export TOTAL_TIMESTEPS=$TIMESTEPS
  export N_ENVS=$N_ENVS
  export DATA_DIRECTORY=$OUTPUT_DIR
  
  # Run training
  python train_scalper.py
  
  echo "Training completed."
  echo "Model saved to: models/dqn_lstm_scalper"
  echo ""
fi

# 3. Continued Training
if [[ $CONTINUE_TRAINING == "true" ]]; then
  echo "===== Step 3: Continued Training ====="
  echo "Continuing training from checkpoint: $CHECKPOINT"
  echo "Training for $TIMESTEPS additional timesteps..."
  
  # Run continued training
  python train_scalper_continue.py \
    --load_model "$CHECKPOINT" \
    --total_timesteps "$TIMESTEPS" \
    --n_envs "$N_ENVS" \
    --data_dir "$OUTPUT_DIR"
  
  echo "Continued training completed."
  echo "Updated model saved to: models/dqn_lstm_scalper"
  echo ""
fi

echo "===== Pipeline Complete ====="
echo "Check logs directory for training metrics."
echo "Models are stored in models/dqn_lstm_scalper directory."

# Optional: Add timestamp to indicate when the pipeline was last run
echo "Last run: $(date)" > .last_training_run

# Commit changes if needed
echo "Would you like to commit changes to git? (y/n)"
read commit_answer
if [[ $commit_answer == "y" ]]; then
  git add .
  git commit -m "Run training pipeline: collect=${COLLECT_DATA}, continue=${CONTINUE_TRAINING}, train=${TRAIN_FROM_SCRATCH}, timesteps=${TIMESTEPS}"
  git push
  echo "Changes committed and pushed."
fi

echo "Done!" 