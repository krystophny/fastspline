#!/usr/bin/env python3
"""
Test fpbspl_ultra to see why it's giving wrong values
"""

import numpy as np
from dierckx_cfunc import fpbspl_ultra

# Test linear B-splines
tx = np.array([0., 0., 1., 1.])
n = len(tx)
k = 1

print("Testing linear B-splines with knots [0, 0, 1, 1]")
print("Expected: B_0(x) = 1-x, B_1(x) = x for x in [0,1]")
print()

test_points = [0.0, 0.25, 0.5, 0.75, 1.0]

for x in test_points:
    # Find interval
    l = k  # Start at k
    while l < n - k - 1 and x >= tx[l+1]:
        l += 1
    
    print(f"\nx = {x}:")
    print(f"  Interval search: l starts at {k}")
    print(f"  tx[l] = {tx[l]}, tx[l+1] = {tx[l+1] if l+1 < n else 'N/A'}")
    print(f"  Final l = {l}")
    
    # Evaluate B-splines
    h = fpbspl_ultra(tx, n, k, x, l)
    print(f"  B-spline values: {h[:k+1]}")
    print(f"  Expected: B_0={1-x:.3f}, B_1={x:.3f}")
    
# The issue is that for x=1.0, we need l=2 (not 1)
# because we want the interval [tx[2], tx[3]] = [1, 1]
print("\n=== ANALYSIS ===")
print("For x=1.0, the knot interval should be l=2")
print("This gives us the interval [tx[2], tx[3]] = [1, 1]")
print("But our search stops at l=1 because tx[2]=1 and x=1, so x >= tx[2] is false")

# Test with correct interval
print("\n=== TEST WITH CORRECT INTERVAL ===")
x = 1.0
l = 2  # Force correct interval
h = fpbspl_ultra(tx, n, k, x, l)
print(f"x = {x}, l = {l}")
print(f"B-spline values: {h[:k+1]}")
print(f"Expected: B_0=0, B_1=1")

# The problem is in the interval finding logic
# For the rightmost point, we need special handling