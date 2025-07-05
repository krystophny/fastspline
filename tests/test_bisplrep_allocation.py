"""Test bisplrep array allocation."""

import numpy as np
import sys
sys.path.insert(0, '../src')


def test_allocation():
    """Test array allocation logic."""
    # 10x10 grid
    m = 100
    kx = ky = 3
    
    # From bisplrep.py
    nxest = max(2*(kx+1), min(int(kx + np.sqrt(m/2)), m//2 + kx + 1))
    nyest = max(2*(ky+1), min(int(ky + np.sqrt(m/2)), m//2 + ky + 1))
    
    print(f"For m={m} points, kx=ky={kx}:")
    print(f"nxest = {nxest}")
    print(f"nyest = {nyest}")
    
    # Minimum needed for interpolation
    nx_min = int(np.sqrt(m) + kx + 1)
    print(f"\nMinimum needed for interpolation: {nx_min}")
    
    # Initial knots
    nx_init = 2 * (kx + 1)
    print(f"Initial knots: {nx_init}")
    
    # How many to add
    n_add = min(nx_min - nx_init, nxest - nx_init - 1)
    print(f"Knots to add: {n_add}")
    print(f"Final knots: {nx_init + n_add}")


if __name__ == "__main__":
    test_allocation()