"""Test knot addition logic."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import add_knot


def test_add_knot():
    """Test add_knot function."""
    # Initial knot vector
    t = np.zeros(20)
    n = 8
    k = 3
    
    # Set up initial knots [0,0,0,0,1,1,1,1]
    for i in range(4):
        t[i] = 0.0
        t[n-1-i] = 1.0
    
    print("Initial knot vector:")
    print(t[:n])
    
    # Try to add knot at 0.5
    print("\nAdding knot at 0.5...")
    n_new = add_knot(t, n, k, 0.5)
    
    print(f"New n: {n_new}")
    print("New knot vector:")
    print(t[:n_new])


def test_knot_addition_loop():
    """Test the knot addition loop."""
    m = 100
    kx = 3
    s = 0.0
    xb, xe = 0.0, 1.0
    nx = 8
    nxest = 16  # Increased
    
    # Initial knots
    tx = np.zeros(nxest)
    for i in range(kx+1):
        tx[i] = xb
        tx[nx-1-i] = xe
    
    print(f"Initial: nx={nx}, nxest={nxest}")
    print(f"Initial knots: {tx[:nx]}")
    
    # From bisplrep_qr_fit
    nx_min = int(np.sqrt(m) + kx + 1)
    print(f"\nnx_min for interpolation: {nx_min}")
    
    # Add interior knots uniformly
    if nx < nx_min and nx < nxest - 1:
        n_add = min(nx_min - nx, nxest - nx - 1)
        print(f"Need to add {n_add} knots")
        
        for i in range(n_add):
            new_knot = xb + (i + 1) * (xe - xb) / (n_add + 1)
            print(f"  Adding knot at {new_knot:.3f}")
            nx_old = nx
            nx = add_knot(tx, nx, kx, new_knot)
            if nx == nx_old:
                print(f"    Failed to add knot!")
            else:
                print(f"    Success, nx={nx}")
    
    print(f"\nFinal: nx={nx}")
    print(f"Final knots: {tx[:nx]}")


if __name__ == "__main__":
    test_add_knot()
    print("\n" + "="*50 + "\n")
    test_knot_addition_loop()