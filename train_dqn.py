#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Entry Point for Training RL Trading Agents

This script serves as the primary entry point for initiating the training
process for various reinforcement learning agents (DQN, PPO, A2C, SAC)
for cryptocurrency trading.

It utilizes the refactored components within the 'rl_agent' package.
"""

import os
import sys
import logging

# --- Robust Project Root Setup ---
def find_project_root(marker='.git'):
    """Find the project root directory by searching upwards for a marker."""
    path = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(path)
        if os.path.exists(os.path.join(path, marker)):
            return path
        if parent == path: # Reached filesystem root
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(os.path.join(script_dir, 'rl_agent')):
                 return script_dir
            # Fallback if marker not found but rl_agent exists in parent
            parent_dir = os.path.dirname(script_dir)
            if os.path.exists(os.path.join(parent_dir, 'rl_agent')):
                return parent_dir
            print("Warning: Project root marker not found. Using script directory.", file=sys.stderr)
            return script_dir
        path = parent

project_root = find_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Project Root Setup ---

# Set project root to path to allow importing rl_agent
# project_root = os.path.dirname(os.path.abspath(__file__)) # Old method removed
# if project_root not in sys.path: # Old method removed
#     sys.path.append(project_root) # Old method removed

# Import necessary components from the rl_agent package
try:
    # Make sure rl_agent and its submodules are importable
    # Adjust based on your final structure if rl_agent is not directly
    # in project root
    from rl_agent.train import main as run_agent_main
    from rl_agent.utils import setup_logger
except ImportError as e:
    print(f"Error importing rl_agent package: {e}")
    print(
        "Ensure the rl_agent directory is in your Python path or "
        "run this script from the project root."
    )
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Setup main logger
# Use the setup_logger from utils, ensuring it's configured correctly
logger = setup_logger(
    "train_dqn_entry", logging.INFO, log_filename="main_training.log"
)


def main():
    """Main function to parse arguments and run the training process."""
    # Arguments are parsed within run_agent_main
    logger.info("Starting main training/evaluation script...")

    # Initiate the process using the main function from rl_agent.train
    try:
        run_agent_main()
        logger.info("Main script finished successfully.")
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred in the main script: {e}",
            exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()