#!/usr/bin/env python3
"""
Test proper interval selection for B-spline evaluation
"""

import numpy as np
from dierckx_cfunc import fpbspl_ultra

# For knots [0, 0, 1, 1] with degree k=1
tx = np.array([0., 0., 1., 1.])
n = 4
k = 1

print("Knots:", tx)
print(f"Valid interval indices: {k} to {n-k-1}")
print()

# The key insight is that for B-spline evaluation:
# - We need to find interval l such that t[l] <= x < t[l+1]
# - But at the right boundary, we use t[l] <= x <= t[l+1]

# For our knots:
# l=1: interval [0, 1) - contains x in [0, 1)
# l=2: interval [1, 1] - contains x = 1 (degenerate interval)

# But wait! For the degenerate interval, the standard B-spline
# evaluation fails. We need to handle this specially.

# Let's test what SciPy does
from scipy.interpolate import BSpline

# Create 1D B-spline
c_1d = np.array([1., 2.])  # Two coefficients for linear spline
spline = BSpline(tx, c_1d, k)

print("SciPy B-spline evaluation:")
for x in [0.0, 0.5, 0.999, 1.0]:
    y = spline(x)
    print(f"  B({x}) = {y}")

# The issue is that at x=1.0, we should use the last valid interval
# which is l=1, not l=2

print("\nCorrect interval selection:")
for x in [0.0, 0.5, 1.0]:
    # Find interval
    l = k
    while l < n - k - 1 and x > tx[l+1]:  # Note: > not >=
        l += 1
    
    # Special case for right boundary
    if x == tx[-1] and l == n - k - 1:
        l = n - k - 2  # Use last valid interval
    
    print(f"x={x}: interval l={l}")

print("\nTesting fpbspl_ultra with corrected intervals:")
for x in [0.0, 0.5, 1.0]:
    # Use interval l=1 for all points in [0,1]
    l = 1
    h = fpbspl_ultra(tx, n, k, x, l)
    print(f"x={x}, l={l}: B-splines = {h[:k+1]}")
    print(f"  Expected: B_0={1-x}, B_1={x}")