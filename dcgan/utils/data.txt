#!/usr/bin/env python3
"""
MNIST Data Loader Utility
- Downloads MNIST only if not found locally
- Normalizes images to [-1, 1]
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loader(batch_size=128, data_dir="dcgan/data/mnist"):
    """
    Returns MNIST training DataLoader.

    Args:
        batch_size (int): batch size for training
        data_dir (str): directory where MNIST is stored

    Returns:
        DataLoader: PyTorch DataLoader for MNIST training set
    """

    # Create directory if missing
    os.makedirs(data_dir, exist_ok=True)

    # Transform: Resize → Tensor → Normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download only if missing
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=not os.path.exists(os.path.join(data_dir, "MNIST")),
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # 0 for Windows for safety
        pin_memory=True
    )

    return loader
