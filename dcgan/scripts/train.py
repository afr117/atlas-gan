#!/usr/bin/env python3
"""
Trains 4 DCGAN experiments on MNIST (PyTorch)
- Auto-creates folder structure
- Auto-detects GPU (fallback to CPU)
- Downloads MNIST once to dcgan/data/mnist
- 5 epochs per experiment (fast run)
- Saves 2 preview images + 1 final image per experiment
- Saves only the final checkpoint per experiment
- AMP enabled only for experiment #4
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from datetime import datetime

from dcgan.models.dcgan import Generator, Discriminator
from dcgan.utils.data import get_data_loader
from dcgan.utils.viz import save_image_grid


# -----------------------------
# Utility Functions
# -----------------------------

def set_seed(seed: int = 42):
    """Ensure reproducible training."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_required_folders():
    """Create required project structure if missing."""
    folders = [
        "dcgan/data/mnist",
        "dcgan/logs",
        "dcgan/logs/checkpoints"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# -----------------------------
# Training for One Experiment
# -----------------------------

def train_experiment(exp_cfg: dict, device: torch.device):
    name = exp_cfg["name"]
    print(f"\n===== Starting Experiment: {name} =====")

    # Set seed for reproducibility
    set_seed(exp_cfg.get("seed", 42))

    # Data loader
    train_loader = get_data_loader(
        batch_size=exp_cfg["batch_size"],
        data_dir="dcgan/data/mnist"
    )

    # Create model
    G = Generator(
        latent_dim=exp_cfg["latent_dim"],
        base_channels=exp_cfg["G"]["base_channels"],
        use_batchnorm=exp_cfg["G"]["use_batchnorm"]
    ).to(device)

    D = Discriminator(
        base_channels=exp_cfg["D"]["base_channels"],
        use_batchnorm=exp_cfg["D"]["use_batchnorm"]
    ).to(device)

    # Optimizers
    g_opt = optim.Adam(G.parameters(), lr=exp_cfg["lr_G"], betas=(0.5, 0.999))
    d_opt = optim.Adam(D.parameters(), lr=exp_cfg["lr_D"], betas=(0.5, 0.999))

    # Mixed precision only for Exp #4
    use_amp = exp_cfg.get("use_amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Fixed noise for preview and final images
    fixed_noise = torch.randn(64, exp_cfg["latent_dim"], device=device)

    epochs = exp_cfg["epochs"]
    save_dir = f"dcgan/logs/{name}"
    ckpt_dir = f"{save_dir}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    preview_epochs = [1, 3]  # Save 2 preview images per experiment

    for epoch in range(1, epochs + 1):
        for real, _ in train_loader:
            real = real.to(device)
            bsz = real.size(0)

            # Train Discriminator
            noise = torch.randn(bsz, exp_cfg["latent_dim"], device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = G(noise)
                d_real = D(real)
                d_fake = D(fake.detach())

                real_loss = nn.functional.binary_cross_entropy_with_logits(
                    d_real, torch.ones_like(d_real)
                )
                fake_loss = nn.functional.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake)
                )
                d_loss = real_loss + fake_loss

            d_opt.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)

            # Train Generator
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_scores = D(fake)
                g_loss = nn.functional.binary_cross_entropy_with_logits(
                    fake_scores, torch.ones_like(fake_scores)
                )

            g_opt.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()

            step += 1

        print(f"[{name}] Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # Save preview images at chosen epochs
        if epoch in preview_epochs:
            with torch.no_grad():
                imgs = G(fixed_noise).cpu()
            save_image_grid(imgs, f"{save_dir}/preview_epoch{epoch}.png")

    # Save final image
    with torch.no_grad():
        final_imgs = G(fixed_noise).cpu()
    save_image_grid(final_imgs, f"{save_dir}/final.png")

    # Save final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"{name}_final.pt")
    torch.save({
        "G": G.state_dict(),
        "D": D.state_dict(),
        "latent_dim": exp_cfg["latent_dim"],
        "epoch": epochs,
        "timestamp": datetime.now().isoformat()
    }, ckpt_path)

    print(f"[{name}] ✅ Finished. Final checkpoint saved at: {ckpt_path}")


# -----------------------------
# Main — Run All Experiments
# -----------------------------

def main():
    make_required_folders()

    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 4 Experiments (Short training: 5 epochs)
    experiments = [
        {
            "name": "baseline",
            "seed": 42,
            "epochs": 5,
            "batch_size": 128,
            "latent_dim": 100,
            "lr_G": 2e-4,
            "lr_D": 2e-4,
            "G": {"base_channels": 128, "use_batchnorm": True},
            "D": {"base_channels": 64, "use_batchnorm": False},
            "use_amp": False
        },
        {
            "name": "arch_deeper",
            "seed": 123,
            "epochs": 5,
            "batch_size": 128,
            "latent_dim": 128,
            "lr_G": 2e-4,
            "lr_D": 2e-4,
            "G": {"base_channels": 256, "use_batchnorm": True},
            "D": {"base_channels": 128, "use_batchnorm": True},
            "use_amp": False
        },
        {
            "name": "hparam_tuned",
            "seed": 42,
            "epochs": 5,
            "batch_size": 64,
            "latent_dim": 100,
            "lr_G": 1e-4,
            "lr_D": 4e-4,
            "G": {"base_channels": 128, "use_batchnorm": True},
            "D": {"base_channels": 64, "use_batchnorm": False},
            "use_amp": False
        },
        {
            "name": "mixed_precision",
            "seed": 7,
            "epochs": 5,
            "batch_size": 256,
            "latent_dim": 100,
            "lr_G": 2e-4,
            "lr_D": 2e-4,
            "G": {"base_channels": 128, "use_batchnorm": True},
            "D": {"base_channels": 64, "use_batchnorm": False},
            "use_amp": True  # AMP ONLY here
        }
    ]

    # Run each experiment
    for exp_cfg in experiments:
        train_experiment(exp_cfg, device)

    print("\n✅ All experiments completed successfully.")


if __name__ == "__main__":
    main()
