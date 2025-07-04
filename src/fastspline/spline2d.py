"""2D Spline interpolation with numba acceleration using cfunc for C interoperability."""

import numpy as np
from numba import cfunc, types, njit, prange
from typing import Tuple, Union, Optional


@cfunc(types.void(types.float64[:], types.int64, types.int64, types.float64, types.float64[:]),
       nopython=True, fastmath=True, boundscheck=False)
def _basis_functions(knots, k, i, x, N):
    """
    Evaluate all (k+1) non-zero B-spline basis functions at x.
    
    This uses the Cox-de Boor recursion formula, optimized for evaluation.
    """
    # Initialize zeroth degree
    N[0] = 1.0
    
    # Compute triangular table of basis functions
    for j in range(1, k + 1):
        left = x - knots[i + 1 - j]
        right = knots[i + j] - x
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right + left)
            N[r] = saved + right * temp
            saved = left * temp
            
            left = x - knots[i - r + 1 - j] if r < j - 1 else left
            right = knots[i + r + j + 1] - x if r < j - 1 else right
        
        N[j] = saved


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
def _bisplev_scalar(x, y, tx, ty, c, kx, ky):
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
def bisplev(x, y, tx, ty, c, kx, ky, result):
    """
    B-spline evaluation that handles both scalar and array inputs.
    
    If x and y are 1D arrays:
    - If same length: evaluates at points (x[i], y[i])
    - If different lengths: evaluates on meshgrid, result[i,j] = f(x[i], y[j])
    
    Result array must be pre-allocated with correct shape.
    """
    nx = len(x)
    ny = len(y)
    
    if nx == ny and nx > 0:
        # Same length - pointwise evaluation
        for i in range(nx):
            result.flat[i] = _bisplev_scalar(x[i], y[i], tx, ty, c, kx, ky)
    else:
        # Different lengths - meshgrid evaluation
        for i in range(nx):
            for j in range(ny):
                result[i, j] = _bisplev_scalar(x[i], y[j], tx, ty, c, kx, ky)


# For compatibility, also export the scalar version
bisplev_scalar = _bisplev_scalar


from .bisplrep import bisplrep


class Spline2D:
    """
    Fast 2D B-spline class optimized for scattered data.
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, 
                 z_data: np.ndarray, kx: int = 3, ky: int = 3, s: float = 0):
        """
        Initialize 2D spline interpolator from scattered data.
        
        Parameters
        ----------
        x_data, y_data : array_like
            1-D arrays of coordinates.
        z_data : array_like
            1-D array of data values.
        kx, ky : int, optional
            Degrees of the bivariate spline. Default is 3.
        s : float, optional
            Smoothing factor. Default is 0 (interpolating).
        """
        self.tck = bisplrep(x_data, y_data, z_data, kx=kx, ky=ky, s=s)
        
    def __call__(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate spline at given points."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        x_scalar = x.ndim == 0
        y_scalar = y.ndim == 0
        
        if x_scalar and y_scalar:
            # Scalar case
            tx, ty, c, kx, ky = self.tck
            return float(_bisplev_scalar(float(x), float(y), tx, ty, c, kx, ky))
        
        # Convert scalars to arrays
        if x_scalar:
            x = np.array([x])
        if y_scalar:
            y = np.array([y])
        
        # Flatten inputs
        x_flat = x.ravel()
        y_flat = y.ravel()
        
        # Determine output shape
        if len(x_flat) == len(y_flat):
            # Same length - pointwise
            result = np.zeros(len(x_flat), dtype=np.float64)
        else:
            # Different lengths - meshgrid
            result = np.zeros((len(x_flat), len(y_flat)), dtype=np.float64)
        
        # Evaluate
        tx, ty, c, kx, ky = self.tck
        bisplev(x_flat, y_flat, tx, ty, c, kx, ky, result)
        
        # Return with appropriate shape
        if x_scalar and y_scalar:
            return float(result.flat[0])
        elif len(x_flat) == len(y_flat):
            return result.reshape(x.shape)
        else:
            return result