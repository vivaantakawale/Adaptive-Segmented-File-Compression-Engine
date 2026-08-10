#!/usr/bin/env python
"""Thin CLI wrapper around src.model.train
trains the algorithm-selection model from a labeled dataset produced by build_dataset.py
"""

from src.model.train import main

if __name__ == "__main__":
    main()
