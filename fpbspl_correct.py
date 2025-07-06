#!/usr/bin/env python3
"""
Correct implementation of fpbspl following DIERCKX algorithm
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, boundscheck=False)
def fpbspl_correct(t, n, k, x, l):
    """
    Correct fpbspl implementation following DIERCKX algorithm.
    Evaluates the (k+1) non-zero B-splines at t[l] <= x < t[l+1]
    """
    h = np.zeros(k + 1, dtype=np.float64)
    h[0] = 1.0
    
    # Temporary storage
    hh = np.zeros(k, dtype=np.float64)
    
    for j in range(1, k + 1):
        # Save current h values
        for i in range(j):
            hh[i] = h[i]
        
        # Reset h[0]
        h[0] = 0.0
        
        # Recurrence relation
        for i in range(j):
            li = l + i + 1  # Note: Fortran 1-based to 0-based adjustment
            lj = li - j
            
            if lj >= 0 and li < n and t[li] != t[lj]:
                f = hh[i] / (t[li] - t[lj])
                h[i] = h[i] + f * (t[li] - x)
                h[i + 1] = f * (x - t[lj])
            else:
                # Handle boundary case
                if i == j - 1 and lj >= 0 and lj < n and li == n:
                    # Special case at right boundary
                    h[i + 1] = hh[i]
                else:
                    h[i + 1] = 0.0
    
    return h

# Test the implementation
if __name__ == "__main__":
    # Test linear B-splines
    tx = np.array([0., 0., 1., 1.])
    n = len(tx)
    k = 1
    
    print("Testing corrected fpbspl with knots [0, 0, 1, 1]")
    print("Expected: B_0(x) = 1-x, B_1(x) = x for x in [0,1]")
    print()
    
    test_cases = [
        (0.0, 1),   # x=0 at interval 1
        (0.5, 1),   # x=0.5 at interval 1
        (1.0, 2),   # x=1 at interval 2 (special case at boundary)
    ]
    
    for x, l in test_cases:
        h = fpbspl_correct(tx, n, k, x, l)
        print(f"x = {x}, l = {l}:")
        print(f"  B-spline values: {h}")
        print(f"  Expected: B_0={1-x:.3f}, B_1={x:.3f}")
        print()
    
    # Test cubic B-splines
    print("\nTesting cubic B-splines")
    tx_cubic = np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.])
    n_cubic = len(tx_cubic)
    k_cubic = 3
    
    x = 0.25
    l = 3  # Interval [0, 0.5]
    h = fpbspl_correct(tx_cubic, n_cubic, k_cubic, x, l)
    print(f"x = {x}, l = {l}:")
    print(f"  B-spline values: {h}")
    print(f"  Sum of B-splines: {np.sum(h)} (should be 1.0)")