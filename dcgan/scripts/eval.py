#!/usr/bin/env python3
"""
Evaluate a Trained DCGAN Model
- Loads a saved checkpoint
- Generates sample images
- Saves output to /dcgan/logs/<exp_name>/eval.png

Usage example:
    python -m dcgan.scripts.eval --ckpt dcgan/logs/baseline/checkpoints/baseline_final.pt
"""

import argparse
import torch
from pathlib import Path

from dcgan.models.dcgan import Generator
from dcgan.utils.viz import save_image_grid


def load_checkpoint(ckpt_path, device):
    """Load checkpoint and restore Generator."""
    ckpt = torch.load(ckpt_path, map_location=device)

    latent_dim = ckpt.get("latent_dim", 100)
    G = Generator(latent_dim=latent_dim).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    return G, latent_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--num", type=int, default=64,
                        help="Number of images to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    G, latent_dim = load_checkpoint(ckpt_path, device)
    noise = torch.randn(args.num, latent_dim, device=device)

    with torch.no_grad():
        imgs = G(noise).cpu()

    out_path = ckpt_path.parent.parent / "eval.png"
    save_image_grid(imgs, out_path)

    print(f"\n✅ Evaluation image saved to: {out_path}")


if __name__ == "__main__":
    main()
