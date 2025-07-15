"""
Configuration utilities for LibContinual framework.
"""

import torch
from torch.utils.data import DataLoader


def create_seeded_dataloader(args, dataset, batch_size=32, shuffle=True, num_workers=0, non_verbose=False):
    """Create a seeded data loader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )