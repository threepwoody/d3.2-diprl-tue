"""
Helper utility functions for training scripts.
"""

import random

import numpy as np

import torch


def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


def format_number(n):
    """
    Format large numbers for display (e.g., 4000000 -> 4M).

    Args:
        n: Number to format

    Returns:
        Formatted string representation
    """

    if n < 1000:

        return str(int(n))

    elif n < 1000000:

        return f"{n / 1000:.0f}K" if n % 1000 == 0 else f"{n / 1000:.1f}K"

    elif n < 1000000000:

        return f"{n / 1000000:.0f}M" if n % 1000000 == 0 else f"{n / 1000000:.1f}M"

    else:

        return str(n)


def to_one_hot(arr, m):
    """
    Convert array to one-hot encoding.

    Args:
        arr: Input array of class indices
        m: Number of classes

    Returns:
        Flattened one-hot encoded array
    """

    arr = arr.astype(int)

    n = len(arr)

    one_hot = np.eye(m)[arr]

    return one_hot.reshape(-1)
