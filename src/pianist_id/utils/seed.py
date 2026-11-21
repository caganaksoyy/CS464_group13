"""One global seeding utility used by every script and notebook."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy and (if installed) PyTorch.

    Args:
        seed: the seed to use.
        deterministic: if True, ask cuDNN for deterministic kernels (slower). The
            default keeps ``cudnn.benchmark=True`` for speed; GPU results can then
            differ in the last digits between runs with the same seed.

    Returns:
        The seed, so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return seed


def make_rng(seed: int) -> np.random.Generator:
    """A NumPy Generator for code that should not touch the global random state."""
    return np.random.default_rng(seed)
