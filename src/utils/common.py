"""Common utility functions."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, benchmark: bool = False) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value.
        benchmark: If True, enables CUDNN benchmark for speed; disables deterministic mode.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if benchmark:
        print(f"⚡ Benchmark mode ENABLED (Speed optimized). Deterministic mode DISABLED.")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print(f"🔒 Deterministic mode ENABLED (Reproducibility optimized). Benchmark mode DISABLED.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
