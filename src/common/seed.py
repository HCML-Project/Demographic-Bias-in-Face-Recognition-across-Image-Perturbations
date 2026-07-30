"""
Seed everything for reproducibility.
"""

import random

import numpy as np
import torch

from common.config import SEED


def seed_everything(seed: int = SEED) -> None:
    """
    Seed everything for reproducibility. Also sets PyTorch to use deterministic algorithms and disables benchmarking for reproducibility.

    Args:
        seed (int): The seed value to use for seeding. Default is SEED.
    """

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
