import torch
import random
import numpy as np

def set_seed(seed):
    """ 
    Fixing the seed is very important for reproducibility (it may slow the training)
    =======
    Args:
    =======
    @ seed : Value of seed to used fixed for the whole training.
    Example usage : set_seed(42) """
    torch.manual_seed(seed) # Sets seed for CPU operations
    np.random.seed(seed) # Sets seed for NumPy
    random.seed(seed) # Sets seed for python's random module

    # GPU specific settings
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # If using multiple GPUs
    torch.backends.cudnn.deterministic = True # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False # Disables non-deterministic optimizations

