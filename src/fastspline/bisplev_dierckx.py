"""
DIERCKX-compatible bisplev implementation.

Handles the row-major coefficient ordering used by DIERCKX/SciPy.
"""

import numpy as np
from numba import njit, cfunc, types
from .bisplrep_dierckx import find_span, basis_funs


@njit(fastmath=True, cache=True)
def bisplev_scalar_dierckx(x, y, tx, ty, c, kx, ky):
    """
    Evaluate 2D B-spline at a single point with DIERCKX ordering.
    
    DIERCKX uses row-major ordering: c[i,j] -> c[(ny-ky-1)*i + j]
    """
    # Find knot spans
    nx = len(tx)
    ny = len(ty)
    span_x = find_span(nx, kx, x, tx)
    span_y = find_span(ny, ky, y, ty)
    
    # Evaluate basis functions
    Nx = basis_funs(span_x, x, kx, tx)
    Ny = basis_funs(span_y, y, ky, ty)
    
    # Tensor product with DIERCKX ordering
    result = 0.0
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    
    for i in range(kx + 1):
        ix = span_x - kx + i
        if 0 <= ix < nk1x:
            for j in range(ky + 1):
                iy = span_y - ky + j
                if 0 <= iy < nk1y:
                    # DIERCKX row-major ordering
                    coef_idx = ix * nk1y + iy
                    result += Nx[i] * Ny[j] * c[coef_idx]
    
    return result


@cfunc(types.void(types.float64[:], types.float64[:], types.float64[:], types.float64[:],
                  types.float64[:], types.int64, types.int64, types.float64[:, :]), 
       nopython=True, fastmath=True, boundscheck=False)
def bisplev_dierckx(x, y, tx, ty, c, kx, ky, result):
    """
    Fast B-spline evaluation with automatic meshgrid detection.
    DIERCKX-compatible coefficient ordering.
    
    If x and y are 1D arrays:
    - If same length: evaluates at points (x[i], y[i])
    - If different lengths: evaluates on meshgrid, result[i,j] = f(x[i], y[j])
    """
    nx = len(x)
    ny = len(y)
    
    if nx == ny and result.shape[0] == nx and result.ndim == 1:
        # Same length with 1D result array - pointwise evaluation
        for i in range(nx):
            result[i] = bisplev_scalar_dierckx(x[i], y[i], tx, ty, c, kx, ky)
    else:
        # Different lengths or 2D result - meshgrid evaluation
        for i in range(nx):
            for j in range(ny):
                result[i, j] = bisplev_scalar_dierckx(x[i], y[j], tx, ty, c, kx, ky)


def bisplev(x, y, tck, dx=0, dy=0):
    """
    Evaluate a bivariate B-spline and its derivatives.
    
    DIERCKX/SciPy-compatible interface.
    
    Parameters
    ----------
    x, y : array_like
        1-D arrays of coordinates.
    tck : tuple
        A tuple (tx, ty, c, kx, ky) containing the knot locations, coefficients,
        and degrees, as returned by bisplrep.
    dx, dy : int, optional
        The partial derivatives to be evaluated.
        
    Returns
    -------
    z : ndarray
        The interpolated values.
    """
    if dx != 0 or dy != 0:
        raise NotImplementedError("Derivatives not yet implemented")
    
    tx, ty, c, kx, ky = tck
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    x_is_scalar = x.ndim == 0
    y_is_scalar = y.ndim == 0
    
    # Convert to arrays
    if x_is_scalar:
        x = np.array([float(x)])
    if y_is_scalar:
        y = np.array([float(y)])
        
    x = x.ravel()
    y = y.ravel()
    
    # SciPy always does meshgrid evaluation for arrays
    # Allocate result - note SciPy returns shape (ny, nx)
    result = np.zeros((len(y), len(x)), dtype=np.float64)
    
    # Call low-level function - it expects (nx, ny) so we need to transpose
    result_temp = np.zeros((len(x), len(y)), dtype=np.float64)
    
    # Force meshgrid evaluation by passing different length indicator
    bisplev_dierckx(x, y, tx, ty, c, kx, ky, result_temp)
    
    # Transpose to match SciPy convention
    result = result_temp.T
    
    # Handle scalar output
    if x_is_scalar and y_is_scalar:
        return float(result.flat[0])
    
    return result