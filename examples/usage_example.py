#!/usr/bin/env python3
"""
Usage example for DIERCKX cfunc implementation
Shows how to use the ultra-optimized functions
"""

import numpy as np
import sys
sys.path.insert(0, '..')
from dierckx_cfunc import fpback_ultra, fpgivs_ultra, fprota_ultra, fprati_ultra, fpbspl_ultra

# Example 1: Backward substitution (fpback)
print("Example 1: Backward substitution (fpback)")
print("-" * 40)

n, k, nest = 5, 3, 8
# Create upper triangular banded matrix
a = np.zeros((nest, k), dtype=np.float64, order='F')
for i in range(n):
    a[i, 0] = 2.0 + 0.1 * i  # Diagonal
    for j in range(1, min(k, n-i)):
        a[i, j] = 0.1 / (j + 1)  # Upper bands

z = np.ones(n, dtype=np.float64)  # Right-hand side
c = np.zeros(n, dtype=np.float64)  # Solution vector

# Solve Ac = z
fpback_ultra(a, z, n, k, c, nest)
print(f"Solution vector c: {c}")

# Example 2: Givens rotation (fpgivs)
print("\n\nExample 2: Givens rotation (fpgivs)")
print("-" * 40)

piv, ww = 3.0, 4.0
dd, cos, sin = fpgivs_ultra(piv, ww)
print(f"Input: piv={piv}, ww={ww}")
print(f"Output: dd={dd}, cos={cos}, sin={sin}")
print(f"Verification: dd² = {dd**2:.6f}, piv² + ww² = {piv**2 + ww**2:.6f}")

# Example 3: Apply rotation (fprota)
print("\n\nExample 3: Apply rotation (fprota)")
print("-" * 40)

cos, sin = 0.8, 0.6  # Rotation parameters
a, b = 5.0, 3.0      # Original values
a_rot, b_rot = fprota_ultra(cos, sin, a, b)
print(f"Rotation: cos={cos}, sin={sin}")
print(f"Original: a={a}, b={b}")
print(f"Rotated: a'={a_rot}, b'={b_rot}")

# Example 4: Rational interpolation (fprati)
print("\n\nExample 4: Rational interpolation (fprati)")
print("-" * 40)

# Three points for rational interpolation
p1, f1 = 1.0, 2.0
p2, f2 = 2.0, 1.0
p3, f3 = 3.0, -1.0

result = fprati_ultra(p1, f1, p2, f2, p3, f3)
p, p1_new, f1_new, p2_new, f2_new = result
print(f"Input points: ({p1}, {f1}), ({p2}, {f2}), ({p3}, {f3})")
print(f"Interpolated value: p={p}")
print(f"Updated parameters: p1={p1_new}, f1={f1_new}, p2={p2_new}, f2={f2_new}")

# Example 5: B-spline evaluation (fpbspl)
print("\n\nExample 5: B-spline evaluation (fpbspl)")
print("-" * 40)

# Create knot vector for cubic spline (k=3)
k = 3
t = np.array([0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1.], dtype=np.float64)
n = len(t)
x = 0.5  # Evaluation point
l = 6    # Interval index where x lies

# Evaluate B-spline basis functions
h = fpbspl_ultra(t, n, k, x, l)
print(f"Knot vector: {t}")
print(f"Degree k={k}, evaluation point x={x}, interval l={l}")
print(f"B-spline basis values: {h[:k+1]}")
print(f"Sum of basis functions: {np.sum(h[:k+1]):.6f} (should be ≈1.0)")

print("\n✓ All examples completed successfully!")