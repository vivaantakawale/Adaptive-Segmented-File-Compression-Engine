#!/usr/bin/env python
"""Thin CLI wrapper around src.benchmark.compare
compares smart-zip against single algorithm baselines and brute force oracle on a set of files
"""

from src.benchmark.compare import main

if __name__ == "__main__":
    main()
