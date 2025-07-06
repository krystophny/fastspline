#!/usr/bin/env python3
"""
Optimized bisplrep/bisplev implementation
"""

import numpy as np
from numba import njit, prange
from dierckx_cfunc import fpbspl_ultra

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def find_unique_sorted(arr, tol=1e-10):
    """Find unique values in a sorted array"""
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=arr.dtype)
    
    unique = np.empty(n, dtype=arr.dtype)
    unique[0] = arr[0]
    count = 1
    
    for i in range(1, n):
        if arr[i] - unique[count-1] > tol:
            unique[count] = arr[i]
            count += 1
    
    return unique[:count]

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def find_interval_vectorized(t, n, k, x_arr):
    """Find intervals for array of x values"""
    m = len(x_arr)
    intervals = np.empty(m, dtype=np.int64)
    
    for i in range(m):
        x = x_arr[i]
        l = k
        # Binary search would be faster for large n
        while l < n - k - 1 and x > t[l+1]:
            l += 1
        # Handle right boundary
        if x == t[n-1] and l == n - k - 1:
            l = n - k - 2
        intervals[i] = l
    
    return intervals

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, parallel=True)
def bisplrep_fast(x, y, z, kx=3, ky=3, s=0.0):
    """
    Fast bivariate spline representation
    """
    m = len(x)
    
    # Sort and find unique values
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    x_unique = find_unique_sorted(x_sorted)
    y_unique = find_unique_sorted(y_sorted)
    
    nx_unique = len(x_unique)
    ny_unique = len(y_unique)
    
    # Determine number of knots
    if s == 0.0 and m == nx_unique * ny_unique:
        # Regular grid interpolation
        nx = nx_unique + kx + 1
        ny = ny_unique + ky + 1
    else:
        # Scattered data - need fewer basis functions than data points
        # Use approximation based on sqrt(m)
        n_basis = int(np.sqrt(m))
        nx = n_basis + kx + 1
        ny = n_basis + ky + 1
        
        # Ensure minimum knots
        nx = max(2*(kx+1), nx)
        ny = max(2*(ky+1), ny)
    
    # Create knot vectors
    tx = np.zeros(nx)
    ty = np.zeros(ny)
    
    # Set boundary knots
    xmin, xmax = x_unique[0], x_unique[-1]
    ymin, ymax = y_unique[0], y_unique[-1]
    
    for i in range(kx + 1):
        tx[i] = xmin
        tx[nx - kx - 1 + i] = xmax
    for i in range(ky + 1):
        ty[i] = ymin
        ty[ny - ky - 1 + i] = ymax
    
    # Set interior knots
    n_interior_x = nx - 2*(kx + 1)
    n_interior_y = ny - 2*(ky + 1)
    
    if n_interior_x > 0:
        if m == nx_unique * ny_unique and s == 0.0 and n_interior_x == nx_unique - 2:
            for i in range(n_interior_x):
                tx[kx + 1 + i] = x_unique[i + 1]
        else:
            dx = (xmax - xmin) / (n_interior_x + 1)
            for i in range(n_interior_x):
                tx[kx + 1 + i] = xmin + (i + 1) * dx
    
    if n_interior_y > 0:
        if m == nx_unique * ny_unique and s == 0.0 and n_interior_y == ny_unique - 2:
            for i in range(n_interior_y):
                ty[ky + 1 + i] = y_unique[i + 1]
        else:
            dy = (ymax - ymin) / (n_interior_y + 1)
            for i in range(n_interior_y):
                ty[ky + 1 + i] = ymin + (i + 1) * dy
    
    # Number of B-spline coefficients
    ncx = nx - kx - 1
    ncy = ny - ky - 1
    n_coeff = ncx * ncy
    
    # Pre-compute intervals for all data points
    lx_all = find_interval_vectorized(tx, nx, kx, x)
    ly_all = find_interval_vectorized(ty, ny, ky, y)
    
    # Build collocation matrix
    A = np.zeros((m, n_coeff))
    
    # Fill matrix - parallelize over data points
    for i in prange(m):
        lx = lx_all[i]
        ly = ly_all[i]
        
        # Evaluate B-splines
        hx = fpbspl_ultra(tx, nx, kx, x[i], lx)
        hy = fpbspl_ultra(ty, ny, ky, y[i], ly)
        
        # Tensor product
        for jx in range(kx + 1):
            ix = lx - kx + jx
            if 0 <= ix < ncx:
                for jy in range(ky + 1):
                    iy = ly - ky + jy
                    if 0 <= iy < ncy:
                        col_idx = ix * ncy + iy
                        A[i, col_idx] = hx[jx] * hy[jy]
    
    # Solve system
    if m == n_coeff:
        # Square system
        c = np.linalg.solve(A, z)
    else:
        # Use normal equations for speed (though less stable than QR)
        AtA = A.T @ A
        Atz = A.T @ z
        
        # Regularization
        reg = 1e-10 * np.trace(AtA) / n_coeff
        for i in range(n_coeff):
            AtA[i, i] += reg
            
        c = np.linalg.solve(AtA, Atz)
    
    return tx, ty, c, kx, ky

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, parallel=True)
def bisplev_fast(x, y, tx, ty, c, kx, ky):
    """
    Fast bivariate B-spline evaluation
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    nx = len(x)
    ny = len(y)
    ntx = len(tx)
    nty = len(ty)
    ncx = ntx - kx - 1
    ncy = nty - ky - 1
    
    # Pre-compute x intervals and B-splines
    lx_all = find_interval_vectorized(tx, ntx, kx, x)
    
    # Output
    z = np.zeros((nx, ny))
    
    # Parallelize over x
    for i in prange(nx):
        lx = lx_all[i]
        hx = fpbspl_ultra(tx, ntx, kx, x[i], lx)
        
        for j in range(ny):
            # Find y interval
            ly = ky
            while ly < nty - ky - 1 and y[j] > ty[ly+1]:
                ly += 1
            if y[j] == ty[nty-1] and ly == nty - ky - 1:
                ly = nty - ky - 2
            
            # Evaluate y B-splines
            hy = fpbspl_ultra(ty, nty, ky, y[j], ly)
            
            # Sum tensor products
            val = 0.0
            for jx in range(kx + 1):
                cx_idx = lx - kx + jx
                if 0 <= cx_idx < ncx:
                    for jy in range(ky + 1):
                        cy_idx = ly - ky + jy
                        if 0 <= cy_idx < ncy:
                            c_idx = cx_idx * ncy + cy_idx
                            val += hx[jx] * hy[jy] * c[c_idx]
            
            z[i, j] = val
    
    return z

if __name__ == "__main__":
    import time
    from scipy.interpolate import bisplrep, bisplev
    
    print("=== TESTING OPTIMIZED IMPLEMENTATION ===")
    
    # Test accuracy
    print("\n--- Accuracy Test ---")
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([1., 2., 2., 3.])
    
    tx, ty, c, _, _ = bisplrep_fast(x, y, z, kx=1, ky=1)
    print(f"Knots x: {tx}")
    print(f"Knots y: {ty}")
    print(f"Coefficients: {c}")
    
    # Evaluate
    x_eval = np.array([0.5])
    y_eval = np.array([0.5])
    z_eval = bisplev_fast(x_eval, y_eval, tx, ty, c, 1, 1)
    print(f"Evaluation at (0.5, 0.5): {z_eval[0,0]} (expected: 2.0)")
    
    # Performance test
    print("\n--- Performance Test ---")
    n = 100
    x_data = np.random.rand(n)
    y_data = np.random.rand(n)
    z_data = np.sin(2*np.pi*x_data) * np.cos(2*np.pi*y_data)
    
    # Warm up
    _ = bisplrep_fast(x_data, y_data, z_data, kx=3, ky=3)
    
    # Time
    start = time.perf_counter()
    for _ in range(10):
        tx, ty, c, _, _ = bisplrep_fast(x_data, y_data, z_data, kx=3, ky=3)
    fast_time = (time.perf_counter() - start) / 10
    
    # Compare with SciPy
    start = time.perf_counter()
    for _ in range(10):
        tck = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    scipy_time = (time.perf_counter() - start) / 10
    
    print(f"bisplrep times (n={n}):")
    print(f"  SciPy: {scipy_time*1000:.2f} ms")
    print(f"  Fast:  {fast_time*1000:.2f} ms")
    print(f"  Speedup: {scipy_time/fast_time:.2f}x")
    
    # Evaluation performance
    x_eval = np.linspace(0, 1, 50)
    y_eval = np.linspace(0, 1, 50)
    
    start = time.perf_counter()
    for _ in range(10):
        z_fast = bisplev_fast(x_eval, y_eval, tx, ty, c, 3, 3)
    fast_eval_time = (time.perf_counter() - start) / 10
    
    tck = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    start = time.perf_counter()
    for _ in range(10):
        z_scipy = bisplev(x_eval, y_eval, tck)
    scipy_eval_time = (time.perf_counter() - start) / 10
    
    print(f"\nbisplev times (50x50 grid):")
    print(f"  SciPy: {scipy_eval_time*1000:.2f} ms")
    print(f"  Fast:  {fast_eval_time*1000:.2f} ms")
    print(f"  Speedup: {scipy_eval_time/fast_eval_time:.2f}x")