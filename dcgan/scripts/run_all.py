#!/usr/bin/env python3
"""
Run All DCGAN Experiments (Shortcut Launcher)

This script simply imports and runs the main() function from train.py
so you don't need to call train.py manually.

Usage:
    python -m dcgan.scripts.run_all
"""

from dcgan.scripts.train import main


if __name__ == "__main__":
    print("\n🚀 Running all DCGAN experiments...\n")
    main()
