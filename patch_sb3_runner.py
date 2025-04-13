#!/usr/bin/env python
"""
Script to patch stable-baselines3 PPO implementation and run training.
This fixes the UserWarning about converting tensor with requires_grad=True to a scalar.
"""

import os
import sys
import importlib
import importlib.util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_patches():
    """Set up the necessary patches for Stable Baselines 3"""
    
    # Try to add the patch_sb3 directory to the Python path
    patch_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "patch_sb3"))
    
    if patch_path not in sys.path:
        logger.info(f"Adding patch_sb3 directory to Python path: {patch_path}")
        sys.path.insert(0, patch_path)
    
    # Try to import the patched modules
    try:
        # Import patched PPO
        from patch_sb3.stable_baselines3 import PPO
        
        # Replace the stable_baselines3 PPO with our patched version
        import stable_baselines3
        stable_baselines3.PPO = PPO
        logger.info("Successfully patched PPO")
        
        # Import patched A2C
        from patch_sb3.stable_baselines3 import A2C
        
        # Replace the stable_baselines3 A2C with our patched version
        stable_baselines3.A2C = A2C
        logger.info("Successfully patched A2C")
        
        # You can add more patches here as needed
        
        return True
    except ImportError as e:
        logger.warning(f"Failed to import patched modules: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error applying patches: {e}")
        return False

def run_training():
    """Run the original training script."""
    try:
        import train_dqn_agent
        logger.info("Imported train_dqn_agent module")
        train_dqn_agent.main()
    except Exception as e:
        logger.error(f"Error running training: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Apply patches
    success = setup_patches()
    
    if success:
        logger.info("Patches applied successfully")
        # Run the training
        run_training()
    else:
        logger.warning("Failed to apply some or all patches")
        sys.exit(1) 