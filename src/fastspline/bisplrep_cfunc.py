"""
Fast B-spline fitting implementation using Numba cfunc.
"""

import numpy as np
from numba import cfunc, types, njit
import numba

@njit(fastmath=True, cache=True)
def _generate_uniform_knots(x_min, x_max, n_interior, k):
    """Generate uniform knot vector."""
    # Total knots = n_interior + 2*(k+1)
    n_knots = n_interior + 2 * (k + 1)
    knots = np.zeros(n_knots)
    
    # Set boundary knots (multiplicity k+1)
    for i in range(k + 1):
        knots[i] = x_min
        knots[n_knots - 1 - i] = x_max
    
    # Set interior knots uniformly
    if n_interior > 0:
        dx = (x_max - x_min) / (n_interior + 1)
        for i in range(n_interior):
            knots[k + 1 + i] = x_min + (i + 1) * dx
    
    return knots


@njit(fastmath=True, cache=True)
def _evaluate_basis(x, knots, k, i):
    """Evaluate B-spline basis function N_{i,k} at point x."""
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
    
    # Cox-de Boor recursion
    left = 0.0
    right = 0.0
    
    # Left term
    denom_left = knots[i + k] - knots[i]
    if denom_left > 0:
        left = (x - knots[i]) / denom_left * _evaluate_basis(x, knots, k - 1, i)
    
    # Right term
    denom_right = knots[i + k + 1] - knots[i + 1]
    if denom_right > 0:
        right = (knots[i + k + 1] - x) / denom_right * _evaluate_basis(x, knots, k - 1, i + 1)
    
    return left + right


@njit(fastmath=True, cache=True)
def _build_collocation_matrix(x_data, y_data, tx, ty, kx, ky):
    """Build collocation matrix for least squares fitting."""
    n_data = len(x_data)
    nx = len(tx) - kx - 1  # Number of x basis functions
    ny = len(ty) - ky - 1  # Number of y basis functions
    n_coeffs = nx * ny
    
    # Build design matrix A where A @ c = z
    A = np.zeros((n_data, n_coeffs))
    
    for idx in range(n_data):
        x, y = x_data[idx], y_data[idx]
        
        # Find non-zero basis functions
        # For now, evaluate all (can optimize later)
        for i in range(nx):
            for j in range(ny):
                basis_x = _evaluate_basis(x, tx, kx, i)
                basis_y = _evaluate_basis(y, ty, ky, j)
                coeff_idx = i * ny + j
                A[idx, coeff_idx] = basis_x * basis_y
    
    return A


@njit(fastmath=True, cache=True)
def _solve_least_squares(A, b):
    """Solve least squares problem min ||Ax - b||^2 using normal equations."""
    # Form normal equations: A^T A x = A^T b
    AtA = A.T @ A
    Atb = A.T @ b
    
    # Add small regularization for numerical stability
    n = AtA.shape[0]
    for i in range(n):
        AtA[i, i] += 1e-10
    
    # Solve using Cholesky decomposition
    # For now, use simple LU decomposition (can optimize with Cholesky later)
    return np.linalg.solve(AtA, Atb)


@cfunc(types.int64(types.float64[:], types.float64[:], types.float64[:],
                   types.int64, types.int64, types.float64[:], types.float64[:],
                   types.float64[:]),
       nopython=True, fastmath=True, boundscheck=False)
def bisplrep(x_data, y_data, z_data, kx, ky, tx_out, ty_out, c_out):
    """
    Fast B-spline fitting for regular grids (simplified version).
    
    This is a simplified implementation that:
    - Assumes uniform knot spacing
    - Uses least squares fitting
    - No iterative knot placement
    - No smoothing parameter optimization
    
    Parameters:
    -----------
    x_data, y_data, z_data : Input data points
    kx, ky : Spline degrees
    tx_out, ty_out : Pre-allocated output knot vectors
    c_out : Pre-allocated output coefficient array
    
    Returns:
    --------
    Packed integer: (nx << 32) | ny
    """
    n_data = len(x_data)
    
    # Determine data bounds
    x_min, x_max = x_data[0], x_data[0]
    y_min, y_max = y_data[0], y_data[0]
    
    for i in range(n_data):
        if x_data[i] < x_min:
            x_min = x_data[i]
        if x_data[i] > x_max:
            x_max = x_data[i]
        if y_data[i] < y_min:
            y_min = y_data[i]
        if y_data[i] > y_max:
            y_max = y_data[i]
    
    # Add small margin to avoid boundary issues
    margin = 0.01
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    
    # Generate uniform knots
    # For simplicity, use sqrt(n_data) interior knots
    # Maximum interior knots based on array sizes
    max_nx = len(tx_out)
    max_ny = len(ty_out)
    n_interior_x = min(int(np.sqrt(n_data)) - kx - 1, max_nx - 2 * (kx + 1))
    n_interior_y = min(int(np.sqrt(n_data)) - ky - 1, max_ny - 2 * (ky + 1))
    
    # Ensure non-negative
    n_interior_x = max(0, n_interior_x)
    n_interior_y = max(0, n_interior_y)
    
    # Generate knot vectors
    for i in range(kx + 1):
        tx_out[i] = x_min
        tx_out[n_interior_x + kx + i + 1] = x_max
    
    for i in range(ky + 1):
        ty_out[i] = y_min
        ty_out[n_interior_y + ky + i + 1] = y_max
    
    # Interior knots
    if n_interior_x > 0:
        dx = (x_max - x_min) / (n_interior_x + 1)
        for i in range(n_interior_x):
            tx_out[kx + 1 + i] = x_min + (i + 1) * dx
    
    if n_interior_y > 0:
        dy = (y_max - y_min) / (n_interior_y + 1)
        for i in range(n_interior_y):
            ty_out[ky + 1 + i] = y_min + (i + 1) * dy
    
    # Calculate actual knot counts
    nx = n_interior_x + 2 * (kx + 1)
    ny = n_interior_y + 2 * (ky + 1)
    
    # Build collocation matrix
    n_basis_x = nx - kx - 1
    n_basis_y = ny - ky - 1
    n_coeffs = n_basis_x * n_basis_y
    
    # For cfunc, we need to build the matrix manually
    # Build design matrix A where A @ c = z
    A = np.zeros((n_data, n_coeffs))
    
    for idx in range(n_data):
        x, y = x_data[idx], y_data[idx]
        
        # Evaluate basis functions
        for i in range(n_basis_x):
            # Check if x is in support of basis function i
            if x >= tx_out[i] and x <= tx_out[i + kx + 1]:
                basis_x = _evaluate_basis(x, tx_out[:nx], kx, i)
                if basis_x > 0:
                    for j in range(n_basis_y):
                        # Check if y is in support of basis function j
                        if y >= ty_out[j] and y <= ty_out[j + ky + 1]:
                            basis_y = _evaluate_basis(y, ty_out[:ny], ky, j)
                            if basis_y > 0:
                                coeff_idx = i * n_basis_y + j
                                A[idx, coeff_idx] = basis_x * basis_y
    
    # Solve least squares: A @ c = z
    # Using normal equations: A^T A c = A^T z
    AtA = np.zeros((n_coeffs, n_coeffs))
    Atz = np.zeros(n_coeffs)
    
    # Compute A^T A and A^T z
    for i in range(n_coeffs):
        for j in range(n_coeffs):
            for k in range(n_data):
                AtA[i, j] += A[k, i] * A[k, j]
        
        for k in range(n_data):
            Atz[i] += A[k, i] * z_data[k]
    
    # Add regularization
    for i in range(n_coeffs):
        AtA[i, i] += 1e-8
    
    # Solve using Gaussian elimination (simplified)
    # For production, would use Cholesky or QR
    # Here we use a simple Gaussian elimination
    for i in range(n_coeffs):
        # Find pivot
        max_val = abs(AtA[i, i])
        max_row = i
        for k in range(i + 1, n_coeffs):
            if abs(AtA[k, i]) > max_val:
                max_val = abs(AtA[k, i])
                max_row = k
        
        # Swap rows
        if max_row != i:
            for j in range(n_coeffs):
                AtA[i, j], AtA[max_row, j] = AtA[max_row, j], AtA[i, j]
            Atz[i], Atz[max_row] = Atz[max_row], Atz[i]
        
        # Eliminate column
        for k in range(i + 1, n_coeffs):
            factor = AtA[k, i] / AtA[i, i]
            for j in range(i + 1, n_coeffs):
                AtA[k, j] -= factor * AtA[i, j]
            Atz[k] -= factor * Atz[i]
    
    # Back substitution
    for i in range(n_coeffs - 1, -1, -1):
        c_out[i] = Atz[i]
        for j in range(i + 1, n_coeffs):
            c_out[i] -= AtA[i, j] * c_out[j]
        c_out[i] /= AtA[i, i]
    
    # Return packed knot counts: (nx << 32) | ny
    return (nx << 32) | ny


# Test the implementation
if __name__ == "__main__":
    import time
    from scipy.interpolate import bisplrep as scipy_bisplrep
    
    # Generate test data
    np.random.seed(42)
    n_points = 100
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.exp(-(x**2 + y**2))
    
    # Allocate output arrays
    max_knots = 50
    tx = np.zeros(max_knots)
    ty = np.zeros(max_knots)
    c = np.zeros(max_knots * max_knots)
    nx = np.array([0], dtype=np.int64)
    ny = np.array([0], dtype=np.int64)
    
    # Test our implementation
    start = time.perf_counter()
    bisplrep(x, y, z, 3, 3, tx, ty, c)
    time_ours = (time.perf_counter() - start) * 1000
    
    # Test SciPy
    start = time.perf_counter()
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0)
    time_scipy = (time.perf_counter() - start) * 1000
    
    print(f"Our bisplrep: {time_ours:.2f}ms")
    print(f"SciPy bisplrep: {time_scipy:.2f}ms")
    print(f"Speedup: {time_scipy/time_ours:.2f}x")
    # Extract knot counts from packed return value
    result = bisplrep(x, y, z, 3, 3, tx, ty, c)
    nx_actual = (result >> 32) & 0xFFFFFFFF
    ny_actual = result & 0xFFFFFFFF
    print(f"Knots: nx={nx_actual}, ny={ny_actual}")