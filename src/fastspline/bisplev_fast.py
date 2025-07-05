"""
Ultra-fast DIERCKX-compatible bisplev implementation with optimizations.
"""

import numpy as np
from numba import njit, prange, cfunc, types
from .bisplrep_dierckx import find_span, basis_funs


@njit(fastmath=True, cache=True, inline='always')
def bisplev_scalar_fast(x, y, tx, ty, c, kx, ky):
    """
    Evaluate 2D B-spline at a single point - optimized version.
    """
    # Find knot spans - optimized with bounds pre-computed
    nx = len(tx)
    ny = len(ty)
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    
    # Binary search for span - inline for speed
    # X direction
    if x >= tx[nk1x-1]:
        span_x = nk1x - 1
    elif x <= tx[kx]:
        span_x = kx
    else:
        low = kx
        high = nk1x
        span_x = (low + high) >> 1
        
        while x < tx[span_x] or x >= tx[span_x + 1]:
            if x < tx[span_x]:
                high = span_x
            else:
                low = span_x
            span_x = (low + high) >> 1
    
    # Y direction
    if y >= ty[nk1y-1]:
        span_y = nk1y - 1
    elif y <= ty[ky]:
        span_y = ky
    else:
        low = ky
        high = nk1y
        span_y = (low + high) >> 1
        
        while y < ty[span_y] or y >= ty[span_y + 1]:
            if y < ty[span_y]:
                high = span_y
            else:
                low = span_y
            span_y = (low + high) >> 1
    
    # Evaluate basis functions
    Nx = basis_funs(span_x, x, kx, tx)
    Ny = basis_funs(span_y, y, ky, ty)
    
    # Tensor product evaluation - optimized loop ordering
    result = 0.0
    base_x = span_x - kx
    base_y = span_y - ky
    
    # Special case for cubic (most common)
    if kx == 3 and ky == 3:
        # Unroll inner loop for cubic case
        for i in range(4):
            ix = base_x + i
            if 0 <= ix < nk1x:
                temp = 0.0
                coef_base = ix * nk1y
                for j in range(4):
                    iy = base_y + j
                    if 0 <= iy < nk1y:
                        temp += Ny[j] * c[coef_base + iy]
                result += Nx[i] * temp
    else:
        # General case
        for i in range(kx + 1):
            ix = base_x + i
            if 0 <= ix < nk1x:
                temp = 0.0
                coef_base = ix * nk1y
                for j in range(ky + 1):
                    iy = base_y + j
                    if 0 <= iy < nk1y:
                        temp += Ny[j] * c[coef_base + iy]
                result += Nx[i] * temp
    
    return result


@njit(parallel=True, fastmath=True, cache=True)
def bisplev_grid_fast(x, y, tx, ty, c, kx, ky, result):
    """
    Fast parallel B-spline evaluation on a grid.
    """
    nx = len(x)
    ny = len(y)
    
    # Parallel evaluation over x dimension
    for i in prange(nx):
        xi = x[i]
        for j in range(ny):
            result[i, j] = bisplev_scalar_fast(xi, y[j], tx, ty, c, kx, ky)


@cfunc(types.void(types.float64[:], types.float64[:], types.float64[:], types.float64[:],
                  types.float64[:], types.int64, types.int64, types.float64[:, :]), 
       nopython=True, fastmath=True)
def bisplev_fast_cfunc(x, y, tx, ty, c, kx, ky, result):
    """
    C-compatible fast evaluation function.
    """
    nx = len(x)
    ny = len(y)
    
    # Always do meshgrid evaluation for cfunc
    for i in range(nx):
        xi = x[i]
        for j in range(ny):
            result[i, j] = bisplev_scalar_fast(xi, y[j], tx, ty, c, kx, ky)


def bisplev_fast(x, y, tck, dx=0, dy=0):
    """
    Fast evaluation of bivariate B-spline.
    
    Optimized version with:
    - Inlined binary search
    - Loop unrolling for cubic case
    - Better cache usage
    - Parallel evaluation for grids
    """
    if dx != 0 or dy != 0:
        raise NotImplementedError("Derivatives not yet implemented")
    
    tx, ty, c, kx, ky = tck
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    x_is_scalar = x.ndim == 0
    y_is_scalar = y.ndim == 0
    
    if x_is_scalar:
        x = np.array([float(x)])
    if y_is_scalar:
        y = np.array([float(y)])
        
    x = x.ravel()
    y = y.ravel()
    
    # Allocate result - SciPy convention (ny, nx)
    result = np.empty((len(y), len(x)), dtype=np.float64)
    
    # Use parallel version for grids
    if len(x) * len(y) > 1000:  # Threshold for parallel
        result_temp = np.empty((len(x), len(y)), dtype=np.float64)
        bisplev_grid_fast(x, y, tx, ty, c, kx, ky, result_temp)
        result = result_temp.T
    else:
        # Use cfunc for smaller grids
        result_temp = np.empty((len(x), len(y)), dtype=np.float64)
        bisplev_fast_cfunc(x, y, tx, ty, c, kx, ky, result_temp)
        result = result_temp.T
    
    # Handle scalar output
    if x_is_scalar and y_is_scalar:
        return float(result.flat[0])
    
    return result