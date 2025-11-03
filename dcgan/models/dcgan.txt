#!/usr/bin/env python3
"""
DCGAN Models: Generator & Discriminator
- Generator upsamples noise into an image
- Discriminator classifies image as real or fake
"""

import torch
import torch.nn as nn


# -------- Helper Blocks -------- #

def conv_block(in_channels, out_channels, k=4, s=2, p=1, use_batchnorm=True):
    """Basic convolution block for Discriminator."""
    layers = [nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_channels, out_channels, k=4, s=2, p=1, use_batchnorm=True):
    """Transpose convolution block for Generator."""
    layers = [nn.ConvTranspose2d(in_channels, out_channels, k, s, p, bias=False)]
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# -------- Generator -------- #

class Generator(nn.Module):
    """
    Generator:
    Input: latent vector (noise)
    Output: 1x28x28 image
    """

    def __init__(self, latent_dim=100, base_channels=128, use_batchnorm=True):
        super().__init__()
        self.latent_dim = latent_dim

        # Output size target = 28x28 → Start from 7x7
        self.fc = nn.Linear(latent_dim, base_channels * 2 * 7 * 7, bias=False)
        self.bn0 = nn.BatchNorm1d(base_channels * 2 * 7 * 7) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(True)

        # Upsample 7x7 → 14x14 → 28x28
        self.deconv1 = deconv_block(base_channels * 2, base_channels, use_batchnorm=use_batchnorm)
        self.deconv2 = deconv_block(base_channels, base_channels // 2, use_batchnorm=use_batchnorm)

        # Final conv  → 1 channel (MNIST grayscale)
        self.final = nn.Conv2d(base_channels // 2, 1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()  # Output in [-1, 1]

    def forward(self, z):
        x = self.fc(z)
        x = self.bn0(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1, 7, 7)  # Reshape to feature map
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.final(x)

        return self.tanh(x)


# -------- Discriminator -------- #

class Discriminator(nn.Module):
    """
    Discriminator:
    Input: 1x28x28 image
    Output: Real (1) or Fake (0)
    """

    def __init__(self, base_channels=64, use_batchnorm=False):
        super().__init__()

        self.main = nn.Sequential(
            conv_block(1, base_channels, use_batchnorm=False),  # First block: no batchnorm
            conv_block(base_channels, base_channels * 2, use_batchnorm=use_batchnorm),
        )

        # Output: 1 value (real vs fake)
        self.final = nn.Conv2d(base_channels * 2, 1, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = self.main(x)
        x = self.final(x)
        return x.view(-1, 1)  # Flatten output
