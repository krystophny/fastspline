#!/usr/bin/env python3
"""
Analyze SciPy's bisplrep algorithm to understand correct implementation
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev

print("=== UNDERSTANDING SCIPY BISPLREP ===")

# Test different grid sizes and degrees
test_cases = [
    (2, 2, 1, 1),  # 2x2 grid, linear
    (3, 3, 1, 1),  # 3x3 grid, linear  
    (3, 3, 2, 2),  # 3x3 grid, quadratic
    (4, 4, 3, 3),  # 4x4 grid, cubic
]

for nx, ny, kx, ky in test_cases:
    print(f"\n--- Grid: {nx}x{ny}, Degree: kx={kx}, ky={ky} ---")
    
    # Create grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    m = len(x_flat)
    
    # Run SciPy bisplrep
    tck = bisplrep(x_flat, y_flat, z_flat, kx=kx, ky=ky, s=0)
    tx, ty, c, kx_out, ky_out = tck
    
    print(f"  Data points: {m}")
    print(f"  Knots x: {len(tx)} values: {tx}")
    print(f"  Knots y: {len(ty)} values: {ty}")
    print(f"  Coefficients: {len(c)} (should be (len(tx)-kx-1)*(len(ty)-ky-1) = {(len(tx)-kx-1)*(len(ty)-ky-1)})")
    
    # For interpolation (s=0), the relationship should be:
    # - Number of knots = number of coefficients + degree + 1
    # - For a rectangular grid, we expect specific knot placement
    
    ncx = len(tx) - kx - 1
    ncy = len(ty) - ky - 1
    print(f"  B-spline basis functions: {ncx} x {ncy} = {ncx*ncy}")
    
    # Check knot multiplicities at boundaries
    print(f"  Knot multiplicities:")
    print(f"    tx start: {np.sum(tx == tx[0])}, end: {np.sum(tx == tx[-1])}")  
    print(f"    ty start: {np.sum(ty == ty[0])}, end: {np.sum(ty == ty[-1])}")
    
    # For regular grids, analyze interior knot placement
    interior_tx = tx[kx:-kx]
    interior_ty = ty[ky:-ky]
    if len(interior_tx) > 0:
        print(f"  Interior knots x: {interior_tx}")
    if len(interior_ty) > 0:
        print(f"  Interior knots y: {interior_ty}")

print("\n=== KEY INSIGHTS ===")
print("1. For interpolation (s=0), SciPy places knots based on data distribution")
print("2. Boundary knots have multiplicity equal to degree+1")
print("3. Interior knots are placed to ensure the system is well-conditioned")
print("4. For a regular mxn grid with degree k, we don't necessarily get m+k+1 knots")

# Test with scattered data
print("\n=== SCATTERED DATA TEST ===")
np.random.seed(42)
n = 10
x = np.random.rand(n)
y = np.random.rand(n)
z = x + y

for kx, ky in [(1, 1), (2, 2), (3, 3)]:
    tck = bisplrep(x, y, z, kx=kx, ky=ky, s=0)
    tx, ty, c, _, _ = tck
    print(f"\nDegree ({kx}, {ky}):")
    print(f"  Knots: {len(tx)} x {len(ty)}")
    print(f"  Coefficients: {len(c)}")
    print(f"  tx: {tx}")
    print(f"  ty: {ty}")