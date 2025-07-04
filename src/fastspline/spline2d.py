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


@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64), 
       nopython=True, fastmath=True, boundscheck=False)
def _bisplev_scalar(x, y, tx, ty, c, kx, ky):
    """
    Scalar B-spline evaluation - evaluates at single point (x,y).
    """
    nx = len(tx)
    ny = len(ty)
    mx = nx - kx - 1
    my = ny - ky - 1
    
    # X direction knot span
    n_x = len(tx) - kx - 1
    if x >= tx[n_x]:
        span_x = n_x - 1
    elif x <= tx[kx]:
        span_x = kx
    else:
        # Binary search - bit shift optimization
        low = kx
        high = n_x
        while True:
            mid = (low + high) >> 1  # Bit shift instead of division
            if x < tx[mid]:
                high = mid
            elif x >= tx[mid + 1]:
                low = mid
            else:
                span_x = mid
                break
    
    # Y direction knot span  
    n_y = len(ty) - ky - 1
    if y >= ty[n_y]:
        span_y = n_y - 1
    elif y <= ty[ky]:
        span_y = ky
    else:
        # Binary search - bit shift optimization
        low = ky
        high = n_y
        while True:
            mid = (low + high) >> 1  # Bit shift instead of division
            if y < ty[mid]:
                high = mid
            elif y >= ty[mid + 1]:
                low = mid
            else:
                span_y = mid
                break
    
    # ULTRA-OPTIMIZED LINEAR CASE
    if kx == 1 and ky == 1:
        # Direct linear interpolation
        alpha_x = (x - tx[span_x]) / (tx[span_x + 1] - tx[span_x])
        alpha_y = (y - ty[span_y]) / (ty[span_y + 1] - ty[span_y])
        
        beta_x = 1.0 - alpha_x
        beta_y = 1.0 - alpha_y
        
        base_idx = (span_x - 1) * my + (span_y - 1)
        
        # Unrolled tensor product
        result = beta_x * beta_y * c[base_idx]
        result += beta_x * alpha_y * c[base_idx + 1]
        result += alpha_x * beta_y * c[base_idx + my]
        result += alpha_x * alpha_y * c[base_idx + my + 1]
        
        return result
    
    # Mixed linear/cubic cases
    elif kx == 1 and ky == 3:
        # Linear x Cubic case
        denom_x = tx[span_x + 1] - tx[span_x]
        alpha_x = (x - tx[span_x]) / denom_x
        Nx0 = 1.0 - alpha_x
        Nx1 = alpha_x
        
        # Use _basis_functions for cubic y
        Ny = np.zeros(4, dtype=np.float64)
        _basis_functions(ty, 3, span_y, y, Ny)
        
        # Tensor product: 2x4 = 8 terms
        idx_x = span_x - 1
        idx_y = span_y - 3
        
        result = 0.0
        for j in range(4):
            result += Nx0 * Ny[j] * c[idx_x * my + (idx_y + j)]
            result += Nx1 * Ny[j] * c[(idx_x + 1) * my + (idx_y + j)]
        
        return result
        
    elif kx == 3 and ky == 1:
        # Cubic x Linear case
        Nx = np.zeros(4, dtype=np.float64)
        _basis_functions(tx, 3, span_x, x, Nx)
        
        denom_y = ty[span_y + 1] - ty[span_y]
        alpha_y = (y - ty[span_y]) / denom_y
        Ny0 = 1.0 - alpha_y
        Ny1 = alpha_y
        
        # Tensor product: 4x2 = 8 terms
        idx_x = span_x - 3
        idx_y = span_y - 1
        
        result = 0.0
        for i in range(4):
            result += Nx[i] * Ny0 * c[(idx_x + i) * my + idx_y]
            result += Nx[i] * Ny1 * c[(idx_x + i) * my + (idx_y + 1)]
        
        return result
    
    # Inline cubic x cubic case
    elif kx == 3 and ky == 3:
        # Inline cubic basis functions for x
        Nx0, Nx1, Nx2, Nx3 = 1.0, 0.0, 0.0, 0.0
        
        # j=1
        left1_x = x - tx[span_x]
        right1_x = tx[span_x + 1] - x
        saved = 0.0
        temp = Nx0 / (right1_x + left1_x)
        Nx0 = saved + right1_x * temp
        saved = left1_x * temp
        Nx1 = saved
        
        # j=2
        left2_x = x - tx[span_x - 1]
        right2_x = tx[span_x + 2] - x
        saved = 0.0
        # r=0
        temp = Nx0 / (right1_x + left2_x)
        Nx0 = saved + right1_x * temp
        saved = left2_x * temp
        # r=1
        temp = Nx1 / (right2_x + left1_x)
        Nx1 = saved + right2_x * temp
        saved = left1_x * temp
        Nx2 = saved
        
        # j=3
        left3_x = x - tx[span_x - 2]
        right3_x = tx[span_x + 3] - x
        saved = 0.0
        # r=0
        temp = Nx0 / (right1_x + left3_x)
        Nx0 = saved + right1_x * temp
        saved = left3_x * temp
        # r=1
        temp = Nx1 / (right2_x + left2_x)
        Nx1 = saved + right2_x * temp
        saved = left2_x * temp
        # r=2
        temp = Nx2 / (right3_x + left1_x)
        Nx2 = saved + right3_x * temp
        saved = left1_x * temp
        Nx3 = saved
        
        # Inline cubic basis functions for y
        Ny0, Ny1, Ny2, Ny3 = 1.0, 0.0, 0.0, 0.0
        
        # j=1
        left1_y = y - ty[span_y]
        right1_y = ty[span_y + 1] - y
        saved = 0.0
        temp = Ny0 / (right1_y + left1_y)
        Ny0 = saved + right1_y * temp
        saved = left1_y * temp
        Ny1 = saved
        
        # j=2
        left2_y = y - ty[span_y - 1]
        right2_y = ty[span_y + 2] - y
        saved = 0.0
        # r=0
        temp = Ny0 / (right1_y + left2_y)
        Ny0 = saved + right1_y * temp
        saved = left2_y * temp
        # r=1
        temp = Ny1 / (right2_y + left1_y)
        Ny1 = saved + right2_y * temp
        saved = left1_y * temp
        Ny2 = saved
        
        # j=3
        left3_y = y - ty[span_y - 2]
        right3_y = ty[span_y + 3] - y
        saved = 0.0
        # r=0
        temp = Ny0 / (right1_y + left3_y)
        Ny0 = saved + right1_y * temp
        saved = left3_y * temp
        # r=1
        temp = Ny1 / (right2_y + left2_y)
        Ny1 = saved + right2_y * temp
        saved = left2_y * temp
        # r=2
        temp = Ny2 / (right3_y + left1_y)
        Ny2 = saved + right3_y * temp
        saved = left1_y * temp
        Ny3 = saved
        
        # Tensor product: 4x4 = 16 terms (unrolled for speed)
        idx_x = span_x - 3
        idx_y = span_y - 3
        
        result = 0.0
        result += Nx0 * Ny0 * c[idx_x * my + idx_y]
        result += Nx0 * Ny1 * c[idx_x * my + (idx_y + 1)]
        result += Nx0 * Ny2 * c[idx_x * my + (idx_y + 2)]
        result += Nx0 * Ny3 * c[idx_x * my + (idx_y + 3)]
        result += Nx1 * Ny0 * c[(idx_x + 1) * my + idx_y]
        result += Nx1 * Ny1 * c[(idx_x + 1) * my + (idx_y + 1)]
        result += Nx1 * Ny2 * c[(idx_x + 1) * my + (idx_y + 2)]
        result += Nx1 * Ny3 * c[(idx_x + 1) * my + (idx_y + 3)]
        result += Nx2 * Ny0 * c[(idx_x + 2) * my + idx_y]
        result += Nx2 * Ny1 * c[(idx_x + 2) * my + (idx_y + 1)]
        result += Nx2 * Ny2 * c[(idx_x + 2) * my + (idx_y + 2)]
        result += Nx2 * Ny3 * c[(idx_x + 2) * my + (idx_y + 3)]
        result += Nx3 * Ny0 * c[(idx_x + 3) * my + idx_y]
        result += Nx3 * Ny1 * c[(idx_x + 3) * my + (idx_y + 1)]
        result += Nx3 * Ny2 * c[(idx_x + 3) * my + (idx_y + 2)]
        result += Nx3 * Ny3 * c[(idx_x + 3) * my + (idx_y + 3)]
        
        return result
    
    # Fall back to general case
    Nx = np.zeros(kx + 1, dtype=np.float64)
    Ny = np.zeros(ky + 1, dtype=np.float64)
    
    _basis_functions(tx, kx, span_x, x, Nx)
    _basis_functions(ty, ky, span_y, y, Ny)
    
    # Compute tensor product
    result = 0.0
    for i in range(kx + 1):
        for j in range(ky + 1):
            coeff_idx = (span_x - kx + i) * my + (span_y - ky + j)
            if 0 <= coeff_idx < len(c):
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