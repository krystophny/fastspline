#!/usr/bin/env python3
"""
Fixed bisplrep/bisplev implementation based on understanding of DIERCKX algorithm
"""

import numpy as np
from numba import njit, prange
from dierckx_cfunc import fpbspl_ultra

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def find_knot_interval(t, n, x):
    """Find the knot interval for x in knot vector t of length n"""
    # Binary search for efficiency
    left = 0
    right = n - 1
    
    while right - left > 1:
        mid = (left + right) // 2
        if x < t[mid]:
            right = mid
        else:
            left = mid
    
    return left

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def create_knots_for_interpolation(x_unique, kx):
    """
    Create knot vector for interpolation based on unique data values
    """
    n_unique = len(x_unique)
    
    if n_unique < kx + 1:
        raise ValueError("Not enough unique points for spline degree")
    
    # For interpolation, we need exactly n_unique basis functions
    # This means n_knots = n_unique + kx + 1
    n_knots = n_unique + kx + 1
    tx = np.zeros(n_knots)
    
    # Set boundary knots with proper multiplicity
    for i in range(kx + 1):
        tx[i] = x_unique[0]
        tx[n_knots - 1 - i] = x_unique[-1]
    
    # Interior knots - for interpolation, place at data points
    if n_unique > 2:
        # For degree 1: place knots at all interior data points
        # For degree 2+: use averaging/not-a-knot conditions
        if kx == 1:
            for i in range(1, n_unique - 1):
                tx[kx + i] = x_unique[i]
        else:
            # Use averaging for higher degrees
            n_interior = n_unique - 2
            if n_interior > 0:
                step = n_interior / (n_knots - 2*kx - 2 + 1)
                for i in range(n_knots - 2*kx - 2):
                    idx = min(int((i + 1) * step), n_interior - 1) 
                    tx[kx + 1 + i] = x_unique[1 + idx]
    
    return tx

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, parallel=True)
def bisplrep_fast(x, y, z, kx=3, ky=3, s=0.0):
    """
    Fast bivariate spline representation for regular or scattered data
    
    This is a simplified implementation focusing on interpolation (s=0)
    """
    m = len(x)
    
    # Get unique x and y values (approximately) to determine grid structure
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    # Simple unique value detection
    tol = 1e-10
    x_unique = [x_sorted[0]]
    for i in range(1, m):
        if x_sorted[i] - x_unique[-1] > tol:
            x_unique.append(x_sorted[i])
    x_unique = np.array(x_unique)
    
    y_unique = [y_sorted[0]]
    for i in range(1, m):
        if y_sorted[i] - y_unique[-1] > tol:
            y_unique.append(y_sorted[i])
    y_unique = np.array(y_unique)
    
    nx_unique = len(x_unique)
    ny_unique = len(y_unique)
    
    # Check if we have enough points
    min_points = (kx + 1) * (ky + 1)
    if m < min_points:
        raise ValueError(f"Need at least {min_points} points for degrees ({kx}, {ky})")
    
    # Create knot vectors
    tx = create_knots_for_interpolation(x_unique, kx)
    ty = create_knots_for_interpolation(y_unique, ky)
    
    # Number of B-spline coefficients
    ncx = len(tx) - kx - 1
    ncy = len(ty) - ky - 1
    n_coeff = ncx * ncy
    
    # For exact interpolation, we need n_coeff = m
    # If not, we need to adjust knots or use least squares
    if n_coeff != m and s == 0.0:
        # Adjust knot vectors for exact interpolation
        # This is a simplified approach - proper DIERCKX uses more sophisticated method
        if n_coeff < m:
            # Need more knots - add interior knots
            # This is where DIERCKX's fpsurf comes in
            pass  # For now, proceed with least squares
    
    # Build collocation matrix
    A = np.zeros((m, n_coeff))
    
    # Fill collocation matrix - can be parallelized
    for i in prange(m):
        xi, yi = x[i], y[i]
        
        # Find knot intervals
        lx = find_knot_interval(tx, len(tx), xi)
        ly = find_knot_interval(ty, len(ty), yi)
        
        # Ensure we're in valid range
        lx = max(kx, min(lx, len(tx) - kx - 2))
        ly = max(ky, min(ly, len(ty) - ky - 2))
        
        # Evaluate B-splines
        hx = fpbspl_ultra(tx, len(tx), kx, xi, lx)
        hy = fpbspl_ultra(ty, len(ty), ky, yi, ly)
        
        # Fill matrix with tensor product B-splines
        for jx in range(kx + 1):
            ix = lx - kx + jx
            if 0 <= ix < ncx:
                for jy in range(ky + 1):
                    iy = ly - ky + jy
                    if 0 <= iy < ncy:
                        col_idx = ix * ncy + iy
                        A[i, col_idx] = hx[jx] * hy[jy]
    
    # Solve the system
    if m == n_coeff:
        # Square system - direct solve
        c = np.linalg.solve(A, z)
    else:
        # Over/underdetermined - use normal equations
        # Solve A^T A c = A^T z
        AtA = A.T @ A
        Atz = A.T @ z
        
        # Add small regularization for stability
        reg = 1e-10
        for i in range(n_coeff):
            AtA[i, i] += reg
            
        c = np.linalg.solve(AtA, Atz)
    
    return tx, ty, c, kx, ky

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, parallel=True)
def bisplev_fast(x, y, tx, ty, c, kx, ky):
    """
    Fast evaluation of bivariate B-spline
    """
    # Handle both scalar and array inputs
    if x.ndim == 0:
        x = np.array([x])
    if y.ndim == 0:
        y = np.array([y])
    
    nx = len(x)
    ny = len(y)
    ntx = len(tx)
    nty = len(ty)
    ncx = ntx - kx - 1
    ncy = nty - ky - 1
    
    # Output array
    z = np.zeros((nx, ny))
    
    # Evaluate - can be parallelized
    for i in prange(nx):
        xi = x[i]
        
        # Find x interval
        lx = find_knot_interval(tx, ntx, xi)
        lx = max(kx, min(lx, ntx - kx - 2))
        
        # Evaluate x B-splines once
        hx = fpbspl_ultra(tx, ntx, kx, xi, lx)
        
        for j in range(ny):
            yj = y[j]
            
            # Find y interval
            ly = find_knot_interval(ty, nty, yj)
            ly = max(ky, min(ly, nty - ky - 2))
            
            # Evaluate y B-splines
            hy = fpbspl_ultra(ty, nty, ky, yj, ly)
            
            # Sum tensor product B-splines
            val = 0.0
            for jx in range(kx + 1):
                cx_idx = lx - kx + jx
                if 0 <= cx_idx < ncx:
                    for jy in range(ky + 1):
                        cy_idx = ly - ky + jy
                        if 0 <= cy_idx < ncy:
                            c_idx = cx_idx * ncy + cy_idx
                            if 0 <= c_idx < len(c):
                                val += hx[jx] * hy[jy] * c[c_idx]
            
            z[i, j] = val
    
    return z

# Test the fixed implementation
if __name__ == "__main__":
    import time
    from scipy.interpolate import bisplrep, bisplev
    
    print("=== TESTING FIXED IMPLEMENTATION ===")
    
    # Simple 2x2 test
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([1., 2., 2., 3.])
    
    print("\n--- 2x2 Grid Test ---")
    
    # SciPy reference
    tck_scipy = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx_scipy, ty_scipy, c_scipy, _, _ = tck_scipy
    
    # Our implementation
    tx_fast, ty_fast, c_fast, kx_out, ky_out = bisplrep_fast(x, y, z, kx=1, ky=1, s=0.0)
    
    print(f"SciPy knots x: {tx_scipy}")
    print(f"Fast  knots x: {tx_fast}")
    print(f"SciPy knots y: {ty_scipy}")
    print(f"Fast  knots y: {ty_fast}")
    
    # Evaluate
    x_eval = np.array([0.5])
    y_eval = np.array([0.5])
    z_scipy = bisplev(x_eval[0], y_eval[0], tck_scipy)
    z_fast = bisplev_fast(x_eval, y_eval, tx_fast, ty_fast, c_fast, 1, 1)[0, 0]
    
    print(f"\nEvaluation at (0.5, 0.5):")
    print(f"  Expected: 2.0")
    print(f"  SciPy: {z_scipy:.3f}")
    print(f"  Fast:  {z_fast:.3f}")
    
    # Larger test
    print("\n--- 10x10 Grid Performance Test ---")
    nx, ny = 10, 10
    x_grid = np.linspace(0, 1, nx)
    y_grid = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Time SciPy
    start = time.perf_counter()
    tck_scipy = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    scipy_time = time.perf_counter() - start
    
    # Time our implementation
    start = time.perf_counter()
    tx_fast, ty_fast, c_fast, _, _ = bisplrep_fast(x_flat, y_flat, z_flat, kx=3, ky=3, s=0.0)
    fast_time = time.perf_counter() - start
    
    print(f"bisplrep times:")
    print(f"  SciPy: {scipy_time*1000:.2f} ms")
    print(f"  Fast:  {fast_time*1000:.2f} ms")
    print(f"  Speedup: {scipy_time/fast_time:.2f}x")
    
    # Evaluate on test grid
    x_test = np.linspace(0, 1, 20)
    y_test = np.linspace(0, 1, 20)
    
    start = time.perf_counter()
    z_scipy = bisplev(x_test, y_test, tck_scipy)
    scipy_eval_time = time.perf_counter() - start
    
    start = time.perf_counter()
    z_fast = bisplev_fast(x_test, y_test, tx_fast, ty_fast, c_fast, 3, 3)
    fast_eval_time = time.perf_counter() - start
    
    print(f"\nbisplev times (20x20 grid):")
    print(f"  SciPy: {scipy_eval_time*1000:.2f} ms")
    print(f"  Fast:  {fast_eval_time*1000:.2f} ms")
    print(f"  Speedup: {scipy_eval_time/fast_eval_time:.2f}x")
    
    # Check accuracy
    max_error = np.max(np.abs(z_scipy - z_fast))
    print(f"\nMax error: {max_error:.2e}")