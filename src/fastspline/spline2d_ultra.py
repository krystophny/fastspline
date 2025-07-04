"""Ultra-fast bisplev implementation with aggressive optimizations."""

import numpy as np
from numba import cfunc, types, njit, prange
import math


@njit(fastmath=True, cache=True, inline='always')
def _find_span_fast(t, k, x):
    """Ultra-fast knot span search with branch prediction optimization."""
    n = len(t) - k - 1
    
    # Hot path optimization - most evaluations are in the middle
    if t[k] <= x < t[n]:
        # Binary search optimized for modern CPUs
        low = k
        high = n
        while high - low > 1:
            mid = (low + high) >> 1  # Bit shift for speed
            if x < t[mid]:
                high = mid
            else:
                low = mid
        return low
    
    # Edge cases
    if x >= t[n]:
        return n - 1
    return k


@njit(fastmath=True, cache=True, inline='always') 
def _cubic_basis_fast(span, x, t):
    """Specialized cubic basis functions with manual unrolling."""
    # Pre-compute all knot differences to avoid repeated array access
    t0 = t[span - 3]
    t1 = t[span - 2]  
    t2 = t[span - 1]
    t3 = t[span]
    t4 = t[span + 1]
    t5 = t[span + 2]
    t6 = t[span + 3]
    
    # De Boor's algorithm manually unrolled for k=3
    # Level 0
    N0, N1, N2, N3 = 1.0, 0.0, 0.0, 0.0
    
    # Level 1
    alpha = (x - t3) / (t4 - t3)
    N0 = (1.0 - alpha) * N0
    N1 = alpha * N0 + (1.0 - alpha) * N1
    N1 = alpha
    
    # Level 2  
    alpha0 = (x - t2) / (t4 - t2)
    alpha1 = (x - t3) / (t5 - t3)
    
    temp0 = N0
    N0 = (1.0 - alpha0) * N0
    N1 = alpha0 * temp0 + (1.0 - alpha1) * N1
    N2 = alpha1 * N1
    
    # Level 3
    alpha0 = (x - t1) / (t4 - t1)
    alpha1 = (x - t2) / (t5 - t2)
    alpha2 = (x - t3) / (t6 - t3)
    
    temp0, temp1 = N0, N1
    N0 = (1.0 - alpha0) * N0
    N1 = alpha0 * temp0 + (1.0 - alpha1) * N1
    N2 = alpha1 * temp1 + (1.0 - alpha2) * N2
    N3 = alpha2 * N2
    
    return N0, N1, N2, N3


@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64), 
       nopython=True, fastmath=True, boundscheck=False)
def bisplev_scalar_ultra(x, y, tx, ty, c, kx, ky):
    """Ultra-fast scalar evaluation optimized for cubic splines."""
    
    if kx == 3 and ky == 3:
        # Specialized cubic x cubic case - the most common
        span_x = _find_span_fast(tx, 3, x)
        span_y = _find_span_fast(ty, 3, y)
        
        # Get cubic basis functions
        Nx0, Nx1, Nx2, Nx3 = _cubic_basis_fast(span_x, x, tx)
        Ny0, Ny1, Ny2, Ny3 = _cubic_basis_fast(span_y, y, ty)
        
        # Tensor product with manual unrolling for maximum speed
        my = len(ty) - 4
        base_idx = (span_x - 3) * my + (span_y - 3)
        
        # Unrolled 4x4 tensor product
        result = 0.0
        result += Nx0 * (Ny0 * c[base_idx] + 
                        Ny1 * c[base_idx + 1] + 
                        Ny2 * c[base_idx + 2] + 
                        Ny3 * c[base_idx + 3])
        
        base_idx += my
        result += Nx1 * (Ny0 * c[base_idx] + 
                        Ny1 * c[base_idx + 1] + 
                        Ny2 * c[base_idx + 2] + 
                        Ny3 * c[base_idx + 3])
        
        base_idx += my  
        result += Nx2 * (Ny0 * c[base_idx] + 
                        Ny1 * c[base_idx + 1] + 
                        Ny2 * c[base_idx + 2] + 
                        Ny3 * c[base_idx + 3])
        
        base_idx += my
        result += Nx3 * (Ny0 * c[base_idx] + 
                        Ny1 * c[base_idx + 1] + 
                        Ny2 * c[base_idx + 2] + 
                        Ny3 * c[base_idx + 3])
        
        return result
    
    elif kx == 1 and ky == 1:
        # Linear case - ultra-fast bilinear interpolation
        span_x = _find_span_fast(tx, 1, x)
        span_y = _find_span_fast(ty, 1, y)
        
        # Bilinear weights
        wx = (x - tx[span_x]) / (tx[span_x + 1] - tx[span_x])
        wy = (y - ty[span_y]) / (ty[span_y + 1] - ty[span_y])
        
        # Coefficient indices
        my = len(ty) - 2
        idx = (span_x - 1) * my + (span_y - 1)
        
        # Bilinear interpolation
        c00 = c[idx]
        c01 = c[idx + 1] 
        c10 = c[idx + my]
        c11 = c[idx + my + 1]
        
        return (1.0 - wx) * (1.0 - wy) * c00 + \
               (1.0 - wx) * wy * c01 + \
               wx * (1.0 - wy) * c10 + \
               wx * wy * c11
    
    else:
        # General case - fall back to previous implementation
        span_x = _find_span_fast(tx, kx, x)
        span_y = _find_span_fast(ty, ky, y)
        
        # General basis functions
        Nx = np.zeros(kx + 1)
        Ny = np.zeros(ky + 1)
        
        # Simplified basis computation
        Nx[0] = 1.0
        for j in range(1, kx + 1):
            for r in range(j):
                if r == j - 1:
                    alpha = (x - tx[span_x - j + 1 + r]) / (tx[span_x + 1] - tx[span_x - j + 1 + r])
                    Nx[r] = (1.0 - alpha) * Nx[r]
                    Nx[r + 1] = alpha * Nx[r]
                else:
                    alpha1 = (x - tx[span_x - j + 1 + r]) / (tx[span_x + 1 + r - j + 1] - tx[span_x - j + 1 + r])
                    alpha2 = (tx[span_x + 1 + r] - x) / (tx[span_x + 1 + r] - tx[span_x + r - j + 1])
                    if r == 0:
                        Nx[r] = alpha2 * Nx[r]
                    else:
                        Nx[r] = alpha1 * Nx[r - 1] + alpha2 * Nx[r]
        
        Ny[0] = 1.0
        for j in range(1, ky + 1):
            for r in range(j):
                if r == j - 1:
                    alpha = (y - ty[span_y - j + 1 + r]) / (ty[span_y + 1] - ty[span_y - j + 1 + r])
                    Ny[r] = (1.0 - alpha) * Ny[r]
                    Ny[r + 1] = alpha * Ny[r]
                else:
                    alpha1 = (y - ty[span_y - j + 1 + r]) / (ty[span_y + 1 + r - j + 1] - ty[span_y - j + 1 + r])
                    alpha2 = (ty[span_y + 1 + r] - y) / (ty[span_y + 1 + r] - ty[span_y + r - j + 1])
                    if r == 0:
                        Ny[r] = alpha2 * Ny[r]
                    else:
                        Ny[r] = alpha1 * Ny[r - 1] + alpha2 * Ny[r]
        
        # Tensor product
        result = 0.0
        my = len(ty) - ky - 1
        for i in range(kx + 1):
            for j in range(ky + 1):
                coeff_idx = (span_x - kx + i) * my + (span_y - ky + j)
                result += Nx[i] * Ny[j] * c[coeff_idx]
        
        return result


@cfunc(types.void(types.float64[:], types.float64[:], types.float64[:], types.float64[:],
                  types.float64[:], types.int64, types.int64, types.float64[:, :]), 
       nopython=True, fastmath=True, boundscheck=False, parallel=True)
def bisplev_ultra(x, y, tx, ty, c, kx, ky, result):
    """Ultra-fast array evaluation with parallelization."""
    nx = len(x)
    ny = len(y)
    
    if nx == ny and nx > 0:
        # Same length - pointwise evaluation with parallelization
        for i in prange(nx):
            result.flat[i] = bisplev_scalar_ultra(x[i], y[i], tx, ty, c, kx, ky)
    else:
        # Different lengths - meshgrid evaluation with parallelization
        for i in prange(nx):
            for j in range(ny):  # Inner loop not parallelized to avoid overhead
                result[i, j] = bisplev_scalar_ultra(x[i], y[j], tx, ty, c, kx, ky)