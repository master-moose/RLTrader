"""Utility functions for the cryptocurrency trading model."""

import random
import numpy as np
import torch
import os


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional deterministic settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)