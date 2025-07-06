#!/usr/bin/env python3
"""
Minimal test of bisplrep logic
"""

import numpy as np
from numba import njit
from dierckx_cfunc import fpbspl_ultra

@njit(fastmath=True)
def test_bisplrep_simple(x, y, z, kx=1, ky=1):
    """Minimal bisplrep implementation for testing"""
    m = len(x)
    
    # For 2x2 grid with linear splines, use fixed knots
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    nx = len(tx)
    ny = len(ty)
    ncx = nx - kx - 1
    ncy = ny - ky - 1
    
    print("Knots:")
    print("tx:", tx)
    print("ty:", ty)
    print("Basis functions:", ncx, "x", ncy, "=", ncx*ncy)
    
    # Build collocation matrix
    A = np.zeros((m, ncx * ncy))
    
    for i in range(m):
        xi, yi = x[i], y[i]
        
        # Find knot intervals
        lx = kx
        while lx < nx - kx - 1 and xi >= tx[lx+1]:
            lx += 1
            
        ly = ky  
        while ly < ny - ky - 1 and yi >= ty[ly+1]:
            ly += 1
        
        print("\nPoint", i, ":", "(", xi, ",", yi, ")")
        print("  Intervals: lx=", lx, ", ly=", ly)
        
        # Evaluate B-splines at this point
        hx = fpbspl_ultra(tx, nx, kx, xi, lx)
        hy = fpbspl_ultra(ty, ny, ky, yi, ly)
        
        print("  hx:", hx[:kx+1])
        print("  hy:", hy[:ky+1])
        
        # Tensor product B-splines
        for jx in range(kx + 1):
            ix = lx - kx + jx
            if 0 <= ix < ncx:
                for jy in range(ky + 1):
                    iy = ly - ky + jy
                    if 0 <= iy < ncy:
                        col_idx = ix * ncy + iy
                        A[i, col_idx] = hx[jx] * hy[jy]
                        if A[i, col_idx] != 0:
                            print("    A[", i, ",", col_idx, "] =", A[i, col_idx])
    
    print("\nMatrix A:")
    for i in range(m):
        print(" ", A[i])
    
    # Check matrix properties
    print("\nMatrix rank:", np.linalg.matrix_rank(A))
    
    # Try to solve
    if np.linalg.matrix_rank(A) == m:
        c = np.linalg.solve(A, z)
        print("Solution:", c)
        return tx, ty, c
    else:
        print("Matrix is singular!")
        return tx, ty, np.zeros(ncx * ncy)

# Test
x = np.array([0., 1., 0., 1.])
y = np.array([0., 0., 1., 1.])
z = np.array([1., 2., 2., 3.])

print("Testing minimal bisplrep implementation...")
tx, ty, c = test_bisplrep_simple(x, y, z)

# Compare with expected
print("\nExpected coefficients: [1, 2, 2, 3]")
print("Got coefficients:", c)