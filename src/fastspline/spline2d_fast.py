"""Ultra-fast bisplev implementation focused on performance."""

import numpy as np
from numba import cfunc, types, njit


@njit(fastmath=True, cache=True)
def _find_span(t, k, x):
    """Find the knot span for x in knot vector t."""
    n = len(t) - k - 1
    if x >= t[n]:
        return n - 1
    if x <= t[k]:
        return k
    
    # Binary search
    low = k
    high = n
    mid = (low + high) // 2
    while x < t[mid] or x >= t[mid + 1]:
        if x < t[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


@njit(fastmath=True, cache=True)
def _basis_funs(i, x, k, t, N):
    """Compute non-zero basis functions."""
    N[0] = 1.0
    left = np.zeros(k + 1)
    right = np.zeros(k + 1)
    
    for j in range(1, k + 1):
        left[j] = x - t[i + 1 - j]
        right[j] = t[i + j] - x
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved


@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64), 
       nopython=True, fastmath=True, boundscheck=False)
def bisplev_scalar_fast(x, y, tx, ty, c, kx, ky):
    """
    Fast B-spline evaluation optimized for performance.
    """
    # Find spans
    span_x = _find_span(tx, kx, x)
    span_y = _find_span(ty, ky, y)
    
    # Compute basis functions
    Nx = np.zeros(kx + 1)
    Ny = np.zeros(ky + 1)
    _basis_funs(span_x, x, kx, tx, Nx)
    _basis_funs(span_y, y, ky, ty, Ny)
    
    # Compute tensor product
    result = 0.0
    my = len(ty) - ky - 1
    
    for i in range(kx + 1):
        for j in range(ky + 1):
            coeff_idx = (span_x - kx + i) * my + (span_y - ky + j)
            result += Nx[i] * Ny[j] * c[coeff_idx]
    
    return result


@cfunc(types.void(types.float64[:], types.float64[:], types.float64[:], types.float64[:],
                  types.float64[:], types.int64, types.int64, types.float64[:, :]), 
       nopython=True, fastmath=True, boundscheck=False)
def bisplev_fast(x, y, tx, ty, c, kx, ky, result):
    """
    Fast B-spline evaluation for arrays.
    """
    nx = len(x)
    ny = len(y)
    
    if nx == ny and nx > 0:
        # Same length - pointwise evaluation
        for i in range(nx):
            result.flat[i] = bisplev_scalar_fast(x[i], y[i], tx, ty, c, kx, ky)
    else:
        # Different lengths - meshgrid evaluation
        for i in range(nx):
            for j in range(ny):
                result[i, j] = bisplev_scalar_fast(x[i], y[j], tx, ty, c, kx, ky)