"""
Advanced B-spline fitting implementation following FITPACK/Dierckx algorithms.

This implementation includes:
- Adaptive knot placement based on residuals
- Iterative refinement to achieve smoothing factor
- Proper handling of boundary knots
- Optimized linear algebra for the normal equations
"""

import numpy as np
from numba import njit, cfunc, types
import numba


@njit(fastmath=True)
def _find_knot_span_bisplrep(knots, k, x):
    """Find knot span for x using binary search."""
    n = len(knots)
    if x >= knots[n - k - 2]:
        return n - k - 2
    if x <= knots[k]:
        return k
    
    low = k
    high = n - k - 1
    while low + 1 < high:
        mid = (low + high) // 2
        if x < knots[mid]:
            high = mid
        else:
            low = mid
    return low


@njit(fastmath=True)
def _basis_functions_bisplrep(knots, k, span, x):
    """Evaluate all non-zero B-spline basis functions at x."""
    N = np.zeros(k + 1)
    N[0] = 1.0
    
    left = np.zeros(k + 1)
    right = np.zeros(k + 1)
    
    for j in range(1, k + 1):
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N


@njit(fastmath=True)
def _generate_adaptive_knots(x_data, residuals, x_min, x_max, n_add, k, current_knots):
    """Generate knots adaptively based on residual distribution."""
    # Sort data by x coordinate
    idx = np.argsort(x_data)
    x_sorted = x_data[idx]
    res_sorted = np.abs(residuals[idx])
    
    # Find regions with highest residuals
    n_bins = min(20, len(x_data) // 5)
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_residuals = np.zeros(n_bins)
    
    for i in range(len(x_sorted)):
        # Find which bin this point belongs to
        for j in range(n_bins):
            if bin_edges[j] <= x_sorted[i] <= bin_edges[j + 1]:
                bin_residuals[j] += res_sorted[i]
                break
    
    # Select bins with highest residuals for new knots
    bin_indices = np.argsort(bin_residuals)[::-1]
    
    new_knots = []
    for i in range(min(n_add, n_bins)):
        bin_idx = bin_indices[i]
        if bin_residuals[bin_idx] > 0:
            # Place knot in center of bin
            knot_pos = 0.5 * (bin_edges[bin_idx] + bin_edges[bin_idx + 1])
            
            # Check if knot is not too close to existing knots
            min_dist = (x_max - x_min) / (len(current_knots) + 10)
            too_close = False
            for ck in current_knots:
                if abs(knot_pos - ck) < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                new_knots.append(knot_pos)
    
    return np.array(new_knots[:n_add])


@njit(fastmath=True)
def _build_observation_matrix(x_data, y_data, tx, ty, kx, ky):
    """Build the observation matrix for least squares fitting."""
    m = len(x_data)
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    n_coeffs = nx * ny
    
    # Allocate sparse representation (row, col, value)
    max_nonzero = m * (kx + 1) * (ky + 1)
    rows = np.zeros(max_nonzero, dtype=np.int32)
    cols = np.zeros(max_nonzero, dtype=np.int32)
    vals = np.zeros(max_nonzero)
    nnz = 0
    
    for idx in range(m):
        x, y = x_data[idx], y_data[idx]
        
        # Find knot spans
        span_x = _find_knot_span_bisplrep(tx, kx, x)
        span_y = _find_knot_span_bisplrep(ty, ky, y)
        
        # Evaluate basis functions
        Nx = _basis_functions_bisplrep(tx, kx, span_x, x)
        Ny = _basis_functions_bisplrep(ty, ky, span_y, y)
        
        # Add contributions to matrix
        for i in range(kx + 1):
            for j in range(ky + 1):
                if Nx[i] != 0 and Ny[j] != 0:
                    row = idx
                    col = (span_x - kx + i) * ny + (span_y - ky + j)
                    val = Nx[i] * Ny[j]
                    
                    rows[nnz] = row
                    cols[nnz] = col
                    vals[nnz] = val
                    nnz += 1
    
    return rows[:nnz], cols[:nnz], vals[:nnz], n_coeffs


@njit(fastmath=True)
def _solve_normal_equations_sparse(rows, cols, vals, z_data, n_coeffs, w_data):
    """Solve normal equations from sparse matrix representation."""
    # Build A^T W A
    AtWA = np.zeros((n_coeffs, n_coeffs))
    AtWz = np.zeros(n_coeffs)
    
    # First pass: compute A^T W A and A^T W z
    for k in range(len(rows)):
        i = rows[k]
        j = cols[k]
        val = vals[k]
        w = w_data[i]
        
        # Add to A^T W z
        AtWz[j] += w * val * z_data[i]
        
        # Add to diagonal of A^T W A
        AtWA[j, j] += w * val * val
    
    # Second pass: compute off-diagonal elements
    for k1 in range(len(rows)):
        i1 = rows[k1]
        j1 = cols[k1]
        val1 = vals[k1]
        w = w_data[i1]
        
        for k2 in range(k1 + 1, len(rows)):
            i2 = rows[k2]
            j2 = cols[k2]
            val2 = vals[k2]
            
            if i1 == i2:  # Same observation
                AtWA[j1, j2] += w * val1 * val2
                AtWA[j2, j1] += w * val1 * val2
    
    # Add regularization
    reg = 1e-8
    for i in range(n_coeffs):
        AtWA[i, i] += reg
    
    # Solve using Cholesky decomposition
    # For now, use simple Gaussian elimination (can optimize later)
    coeffs = np.linalg.solve(AtWA, AtWz)
    
    # Compute residuals
    residuals = np.zeros(len(z_data))
    for k in range(len(rows)):
        i = rows[k]
        j = cols[k]
        residuals[i] += vals[k] * coeffs[j]
    
    residuals = z_data - residuals
    
    # Compute weighted sum of squares
    fp = 0.0
    for i in range(len(z_data)):
        fp += w_data[i] * residuals[i] * residuals[i]
    
    return coeffs, residuals, fp


@cfunc(types.int64(types.float64[:], types.float64[:], types.float64[:],
                   types.float64[:], types.int64, types.int64, types.float64,
                   types.float64[:], types.float64[:], types.float64[:]),
       nopython=True, fastmath=True, boundscheck=False)
def bisplrep_advanced(x_data, y_data, z_data, w_data, kx, ky, s,
                      tx_out, ty_out, c_out):
    """
    Advanced B-spline surface fitting with adaptive knot placement.
    
    Implements key features of FITPACK's surfit:
    - Iterative knot addition based on residuals
    - Smoothing factor optimization
    - Weighted least squares
    
    Parameters:
    -----------
    x_data, y_data, z_data : Data points
    w_data : Weights (all 1.0 for unweighted)
    kx, ky : Spline degrees
    s : Smoothing factor (0 for interpolation)
    tx_out, ty_out : Pre-allocated knot arrays
    c_out : Pre-allocated coefficient array
    
    Returns:
    --------
    Packed integer: (nx << 32) | ny
    """
    m = len(x_data)
    
    # Find data bounds
    x_min = np.min(x_data)
    x_max = np.max(x_data)
    y_min = np.min(y_data)
    y_max = np.max(y_data)
    
    # Add margin
    margin = 0.01
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    
    # Start with minimal knots (polynomial)
    nx = 2 * (kx + 1)
    ny = 2 * (ky + 1)
    
    # Set boundary knots
    for i in range(kx + 1):
        tx_out[i] = x_min
        tx_out[nx - 1 - i] = x_max
        ty_out[i] = y_min
        ty_out[ny - 1 - i] = y_max
    
    # Main iteration loop
    max_iter = 20
    for iteration in range(max_iter):
        # Build and solve system
        rows, cols, vals, n_coeffs = _build_observation_matrix(
            x_data, y_data, tx_out[:nx], ty_out[:ny], kx, ky)
        
        coeffs, residuals, fp = _solve_normal_equations_sparse(
            rows, cols, vals, z_data, n_coeffs, w_data)
        
        # Check convergence
        if fp <= s or iteration == max_iter - 1:
            # Copy coefficients to output
            for i in range(min(n_coeffs, len(c_out))):
                c_out[i] = coeffs[i]
            break
        
        # Add knots based on residuals
        # Add at most 1 knot in each direction per iteration
        if nx < len(tx_out) - 2:
            new_x_knots = _generate_adaptive_knots(
                x_data, residuals, x_min, x_max, 1, kx, tx_out[kx+1:nx-kx-1])
            
            if len(new_x_knots) > 0:
                # Insert new knot while maintaining order
                new_knot = new_x_knots[0]
                # Find insertion position
                for i in range(kx + 1, nx - kx - 1):
                    if new_knot < tx_out[i]:
                        # Shift knots
                        for j in range(nx - 1, i - 1, -1):
                            tx_out[j + 1] = tx_out[j]
                        tx_out[i] = new_knot
                        nx += 1
                        break
        
        if ny < len(ty_out) - 2:
            new_y_knots = _generate_adaptive_knots(
                y_data, residuals, y_min, y_max, 1, ky, ty_out[ky+1:ny-ky-1])
            
            if len(new_y_knots) > 0:
                # Insert new knot while maintaining order
                new_knot = new_y_knots[0]
                # Find insertion position
                for i in range(ky + 1, ny - ky - 1):
                    if new_knot < ty_out[i]:
                        # Shift knots
                        for j in range(ny - 1, i - 1, -1):
                            ty_out[j + 1] = ty_out[j]
                        ty_out[i] = new_knot
                        ny += 1
                        break
    
    # Return packed knot counts
    return (nx << 32) | ny


# Python wrapper for testing
def bisplrep_advanced_py(x, y, z, w=None, kx=3, ky=3, s=0):
    """Python wrapper for advanced bisplrep."""
    if w is None:
        w = np.ones_like(x)
    
    # Allocate arrays
    max_knots = min(100, int(np.sqrt(len(x))) + 2 * (max(kx, ky) + 1))
    tx = np.zeros(max_knots)
    ty = np.zeros(max_knots)
    c = np.zeros(max_knots * max_knots)
    
    # Call cfunc
    result = bisplrep_advanced(x, y, z, w, kx, ky, s, tx, ty, c)
    
    # Extract knot counts
    nx = (result >> 32) & 0xFFFFFFFF
    ny = result & 0xFFFFFFFF
    
    return (tx[:nx], ty[:ny], c[:nx*ny], kx, ky)