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

def apply_patch():
    """Apply the patch to stable-baselines3 PPO implementation."""
    try:
        # Get the path to the original stable-baselines3 PPO module
        import stable_baselines3
        sb3_path = os.path.dirname(stable_baselines3.__file__)
        ppo_path = os.path.join(sb3_path, "ppo", "ppo.py")
        
        # Get the path to our patched version
        patch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_sb3")
        patched_ppo_path = os.path.join(patch_dir, "stable_baselines3", "ppo", "ppo.py")
        
        logger.info(f"Original SB3 PPO path: {ppo_path}")
        logger.info(f"Patched PPO path: {patched_ppo_path}")
        
        if not os.path.exists(patched_ppo_path):
            logger.error(f"Patched PPO file not found at: {patched_ppo_path}")
            return False
            
        # Read the patched version
        with open(patched_ppo_path, 'r') as f:
            patched_content = f.read()
            
        # Create a backup of the original file
        backup_path = ppo_path + ".backup"
        if not os.path.exists(backup_path):
            logger.info(f"Creating backup of original PPO file at: {backup_path}")
            import shutil
            shutil.copy2(ppo_path, backup_path)
        
        # Write the patched content to the original file
        with open(ppo_path, 'w') as f:
            f.write(patched_content)
            
        logger.info("Successfully patched stable-baselines3 PPO implementation!")
        
        # Reload the module to apply the patch
        if "stable_baselines3.ppo.ppo" in sys.modules:
            logger.info("Reloading stable_baselines3.ppo.ppo module")
            importlib.reload(sys.modules["stable_baselines3.ppo.ppo"])
        
        return True
    except Exception as e:
        logger.error(f"Error applying patch: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
    # Apply the patch
    if apply_patch():
        # Run the training
        run_training()
    else:
        logger.error("Failed to apply patch. Aborting.")
        sys.exit(1) 