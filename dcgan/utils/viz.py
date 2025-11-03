#!/usr/bin/env python3
"""
Visualization Utility for saving generated image grids
"""

import torch
import torchvision.utils as vutils
from pathlib import Path


def save_image_grid(tensor, filename, nrow=8):
    """
    Saves a grid of images to disk.

    Args:
        tensor (Tensor): batch of images (B, C, H, W)
        filename (str): output filename
        nrow (int): number of images per row
    """

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Convert from [-1, 1] to [0, 1]
    grid = vutils.make_grid((tensor + 1) / 2, nrow=nrow)

    vutils.save_image(grid, filename)
    print(f"🖼️ Saved image grid: {filename}")
