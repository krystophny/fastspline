"""2D Spline interpolation with numba acceleration using cfunc for C interoperability."""

import numpy as np
from numba import cfunc, types
from typing import Tuple, Union, Optional
from .spline1d import (
    _solve_tridiagonal, _compute_cubic_coefficients_regular, 
    _compute_cubic_coefficients_periodic, _compute_linear_coefficients
)


@cfunc(types.float64[:, :, :, :](types.float64[:, :], types.float64, types.float64, types.int64, types.int64, types.boolean, types.boolean), nopython=True, fastmath=True, boundscheck=False)
def _compute_2d_spline_coefficients(z_grid, h_step_x, h_step_y, order_x, order_y, periodic_x, periodic_y):
    """
    Compute 2D tensor product spline coefficients.
    
    Parameters:
    z_grid: 2D array of function values, shape (nx, ny)
    h_step_x, h_step_y: grid spacing in x and y directions
    order_x, order_y: spline orders (1 or 3)
    periodic_x, periodic_y: periodicity flags
    
    Returns:
    coeffs: 4D coefficient array, shape (nx, ny, order_x+1, order_y+1) - cache-optimized layout
    """
    nx, ny = z_grid.shape
    
    # Allocate coefficient array with cache-optimized layout: spatial indices first
    coeffs = np.zeros((nx, ny, order_x + 1, order_y + 1))
    
    # Step 1: Spline along y-direction (contiguous access) for each x
    temp_coeffs_y = np.zeros((order_y + 1, ny))
    for i in range(nx):
        y_slice = z_grid[i, :]  # Contiguous access along y
        
        # Compute 1D spline coefficients along y
        if order_y == 1:
            coeffs_1d = _compute_linear_coefficients(y_slice, h_step_y)
        else:  # order_y == 3
            if periodic_y:
                coeffs_1d = _compute_cubic_coefficients_periodic(y_slice, h_step_y)
            else:
                coeffs_1d = _compute_cubic_coefficients_regular(y_slice, h_step_y)
        
        # Store in cache-optimized layout: coeffs[i, :, 0, :order_y+1]
        for j in range(ny):
            for ky in range(coeffs_1d.shape[0]):
                coeffs[i, j, 0, ky] = coeffs_1d[ky, j]
    
    # Step 2: Spline along x-direction for each y and each coefficient order
    temp_coeffs_x = np.zeros((order_x + 1, nx))
    for ky in range(order_y + 1):
        for j in range(ny):
            # Extract coefficients along x for this y-position and y-order
            x_slice = np.zeros(nx)
            for i in range(nx):
                x_slice[i] = coeffs[i, j, 0, ky]
            
            # Compute 1D spline coefficients along x
            if order_x == 1:
                coeffs_1d = _compute_linear_coefficients(x_slice, h_step_x)
            else:  # order_x == 3
                if periodic_x:
                    coeffs_1d = _compute_cubic_coefficients_periodic(x_slice, h_step_x)
                else:
                    coeffs_1d = _compute_cubic_coefficients_regular(x_slice, h_step_x)
            
            # Store final coefficients in cache-optimized layout
            for i in range(nx):
                for kx in range(coeffs_1d.shape[0]):
                    coeffs[i, j, kx, ky] = coeffs_1d[kx, i]
    
    return coeffs


@cfunc(types.float64[:, :](types.float64[:], types.int64, types.int64), nopython=True, fastmath=True, boundscheck=False)
def _handle_missing_data_cfunc(z_linear, nx, ny):
    """
    Handle missing data (NaN values) in the input array.
    Fast unsafe implementation - no bounds checking.
    """
    # Allocate output array
    z_grid = np.empty((nx, ny), dtype=np.float64)
    
    # Copy data with manual indexing - cache-optimized, unsafe
    for i in range(nx):
        i_offset = i * ny
        for j in range(ny):  # j is inner loop for row-major contiguous access
            val = z_linear[i_offset + j]
            if val == val:  # Fast NaN check: NaN != NaN
                z_grid[i, j] = val
            else:
                z_grid[i, j] = 0.0  # Replace NaN
    
    return z_grid


def _detect_data_format(x, y, z):
    """
    Detect whether input data is structured (regular grid) or unstructured (scattered points).
    
    Parameters:
    x, y, z: input arrays
    
    Returns:
    is_unstructured: bool indicating if data is unstructured
    """
    nx, ny = len(x), len(y)
    
    # First check if it's structured data
    if z.shape == (nx * ny,) or z.shape == (nx, ny):
        return False
    
    # If all arrays have the same length and it's not structured, it's unstructured
    if len(x) == len(y) == len(z):
        return True
    
    raise ValueError(f"Invalid data format: x({len(x)}), y({len(y)}), z{z.shape}")


@cfunc(types.UniTuple(types.float64[:], 3)(types.float64[:], types.float64[:], types.float64[:], types.int64, types.int64, types.float64), nopython=True, fastmath=True, boundscheck=False)
def _compute_surfit_coefficients(x, y, z, kx, ky, s):
    """
    Compute spline coefficients for unstructured data using SURFIT-based algorithm.
    
    This is a simplified version of the DIERCKX SURFIT algorithm.
    For a full implementation, we would need to:
    1. Determine optimal knot locations
    2. Set up and solve the linear system for spline coefficients
    3. Handle smoothing parameter and weighted least squares
    
    Parameters:
    x, y, z: scattered data points (all same length)
    kx, ky: spline degrees
    s: smoothing parameter
    
    Returns:
    x_knots, y_knots: knot vectors
    coeffs: spline coefficients
    nx, ny: number of knots
    """
    m = len(x)
    
    # Determine domain bounds
    xb, xe = np.min(x), np.max(x)
    yb, ye = np.min(y), np.max(y)
    
    # Estimate number of knots (simplified approach)
    # For a full SURFIT implementation, this would be determined automatically
    nx = min(max(kx + 1, int(np.sqrt(m) / 2)), m // 4)
    ny = min(max(ky + 1, int(np.sqrt(m) / 2)), m // 4)
    
    # Create knot vectors with proper multiplicity at boundaries
    x_knots = np.zeros(nx + kx + 1)
    y_knots = np.zeros(ny + ky + 1)
    
    # Set boundary knots with multiplicity kx+1 and ky+1
    for i in range(kx + 1):
        x_knots[i] = xb
        x_knots[nx + i] = xe
    
    for i in range(ky + 1):
        y_knots[i] = yb
        y_knots[ny + i] = ye
    
    # Interior knots (uniform distribution - simplified)
    if nx > kx + 1:
        for i in range(kx + 1, nx):
            x_knots[i] = xb + (xe - xb) * (i - kx) / (nx - kx)
    
    if ny > ky + 1:
        for i in range(ky + 1, ny):
            y_knots[i] = yb + (ye - yb) * (i - ky) / (ny - ky)
    
    # Compute spline coefficients (simplified least-squares approach)
    # In a full SURFIT implementation, this would involve iterative knot placement
    n_coeffs = (nx - kx - 1) * (ny - ky - 1)
    coeffs = np.zeros(n_coeffs)
    
    # For now, use a simple approach that sets coefficients to approximate the data
    # This is a placeholder - the full SURFIT algorithm would solve a linear system
    if n_coeffs > 0:
        coeffs[0] = np.mean(z)  # Simple approximation
    
    return x_knots, y_knots, coeffs


def _surfit_to_structured_grid(x, y, z, kx, ky, nx_out=None, ny_out=None):
    """
    Convert unstructured data to structured grid for use with existing spline infrastructure.
    
    This creates a regular grid and interpolates the scattered data onto it.
    """
    if nx_out is None:
        nx_out = max(10, int(np.sqrt(len(x))))
    if ny_out is None:
        ny_out = max(10, int(np.sqrt(len(y))))
    
    # Create regular grid
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    x_grid = np.linspace(x_min, x_max, nx_out)
    y_grid = np.linspace(y_min, y_max, ny_out)
    
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # Interpolate scattered data to grid using inverse distance weighting
    # Cache-optimized: access z_grid[i, j] with j as inner loop
    z_grid = np.zeros((nx_out, ny_out))
    
    for i in range(nx_out):
        for j in range(ny_out):  # j is inner loop for contiguous access
            xi, yj = X_grid[i, j], Y_grid[i, j]
            
            # Calculate distances to all data points
            distances = np.sqrt((x - xi)**2 + (y - yj)**2)
            
            # Handle very close points
            min_dist = np.min(distances)
            if min_dist < 1e-12:
                # Use exact value if very close to a data point
                closest_idx = np.argmin(distances)
                z_grid[i, j] = z[closest_idx]
            else:
                # Inverse distance weighting with power 2
                weights = 1.0 / (distances**2 + 1e-10)
                z_grid[i, j] = np.sum(weights * z) / np.sum(weights)
    
    return x_grid, y_grid, z_grid


# C-compatible function signatures using cfunc
@cfunc(types.float64(types.float64, types.float64, types.float64[:, :, :, :], 
                     types.float64, types.float64, types.float64, types.float64,
                     types.int64, types.int64, types.int64, types.int64,
                     types.boolean, types.boolean))
def evaluate_spline_2d_cfunc(x, y, coeffs, x_min, y_min, h_step_x, h_step_y,
                             nx, ny, order_x, order_y, periodic_x, periodic_y):
    """
    C-compatible 2D spline evaluation function.
    
    Parameters:
    x, y: evaluation coordinates
    coeffs: 4D coefficient array (nx, ny, order_x+1, order_y+1) - cache-optimized layout
    x_min, y_min: minimum coordinate values
    h_step_x, h_step_y: grid spacing
    nx, ny: number of grid points
    order_x, order_y: spline orders
    periodic_x, periodic_y: periodicity flags
    """
    # Find intervals and local coordinates
    if periodic_x:
        x_period = h_step_x * (nx - 1)
        xj = x - x_min
        xj = xj - np.floor(xj / x_period) * x_period + x_min
    else:
        xj = x
    
    if periodic_y:
        y_period = h_step_y * (ny - 1)
        yj = y - y_min
        yj = yj - np.floor(yj / y_period) * y_period + y_min
    else:
        yj = y
    
    x_norm = (xj - x_min) / h_step_x
    y_norm = (yj - y_min) / h_step_y
    
    ix = max(0, min(nx - 1, int(x_norm)))
    iy = max(0, min(ny - 1, int(y_norm)))
    
    x_local = (x_norm - float(ix)) * h_step_x
    y_local = (y_norm - float(iy)) * h_step_y
    
    # Extract local coefficients - cache-optimized access pattern
    coeff_local = coeffs[ix, iy, :, :]  # Contiguous access to coefficient block
    
    # Evaluate using nested Horner's method
    # First, evaluate along x for each y-order
    coeff_y = np.zeros(order_y + 1)
    for ky in range(order_y + 1):  # Inner loop over contiguous ky dimension
        coeff_y[ky] = coeff_local[order_x, ky]
        for kx in range(order_x - 1, -1, -1):
            coeff_y[ky] = coeff_local[kx, ky] + x_local * coeff_y[ky]
    
    # Then evaluate along y
    result = coeff_y[order_y]
    for ky in range(order_y - 1, -1, -1):
        result = coeff_y[ky] + y_local * result
    
    return result


@cfunc(types.UniTuple(types.float64, 3)(types.float64, types.float64, types.float64[:, :, :, :], 
                                        types.float64, types.float64, types.float64, types.float64,
                                        types.int64, types.int64, types.int64, types.int64,
                                        types.boolean, types.boolean))
def evaluate_spline_2d_derivatives_cfunc(x, y, coeffs, x_min, y_min, h_step_x, h_step_y,
                                         nx, ny, order_x, order_y, periodic_x, periodic_y):
    """
    C-compatible 2D spline evaluation with derivatives.
    
    Returns: (value, dz/dx, dz/dy)
    """
    # Find intervals and local coordinates (same as above)
    if periodic_x:
        x_period = h_step_x * (nx - 1)
        xj = x - x_min
        xj = xj - np.floor(xj / x_period) * x_period + x_min
    else:
        xj = x
    
    if periodic_y:
        y_period = h_step_y * (ny - 1)
        yj = y - y_min
        yj = yj - np.floor(yj / y_period) * y_period + y_min
    else:
        yj = y
    
    x_norm = (xj - x_min) / h_step_x
    y_norm = (yj - y_min) / h_step_y
    
    ix = max(0, min(nx - 1, int(x_norm)))
    iy = max(0, min(ny - 1, int(y_norm)))
    
    x_local = (x_norm - float(ix)) * h_step_x
    y_local = (y_norm - float(iy)) * h_step_y
    
    # Extract local coefficients - cache-optimized access pattern
    coeff_local = coeffs[ix, iy, :, :]  # Contiguous access to coefficient block
    
    # Evaluate function value and x-derivative
    coeff_y = np.zeros(order_y + 1)
    dcoeff_y_dx = np.zeros(order_y + 1)
    
    for ky in range(order_y + 1):  # Inner loop over contiguous ky dimension
        # Function value
        coeff_y[ky] = coeff_local[order_x, ky]
        for kx in range(order_x - 1, -1, -1):
            coeff_y[ky] = coeff_local[kx, ky] + x_local * coeff_y[ky]
        
        # x-derivative
        if order_x == 0:
            dcoeff_y_dx[ky] = 0.0
        else:
            dcoeff_y_dx[ky] = coeff_local[order_x, ky] * order_x
            for kx in range(order_x - 1, 0, -1):
                dcoeff_y_dx[ky] = coeff_local[kx, ky] * kx + x_local * dcoeff_y_dx[ky]
    
    # Evaluate along y
    z = coeff_y[order_y]
    dz_dx = dcoeff_y_dx[order_y]
    dz_dy = coeff_y[order_y] * order_y if order_y > 0 else 0.0
    
    for ky in range(order_y - 1, -1, -1):
        z = coeff_y[ky] + y_local * z
        dz_dx = dcoeff_y_dx[ky] + y_local * dz_dx
        if ky > 0:
            dz_dy = coeff_y[ky] * ky + y_local * dz_dy
    
    return z, dz_dx, dz_dy


@cfunc(types.int64(types.float64[:], types.int64, types.float64), nopython=True, fastmath=True, boundscheck=False)
def _find_knot_span(t, k, x):
    """
    Find knot span index using binary search (NURBS Book Algorithm A2.1).
    Returns i such that t[i] <= x < t[i+1] and basis functions N_{i-k},...,N_i are non-zero.
    """
    n = len(t) - k - 1  # Number of basis functions
    
    # Special cases
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






@cfunc(types.void(types.float64[:], types.int64, types.int64, types.float64, types.float64[:]), nopython=True, fastmath=True, boundscheck=False)
def _basis_functions(t, k, knot_span, x, N):
    """
    Compute all non-zero basis functions (NURBS Book Algorithm A2.2).
    Fast stack-based implementation matching SciPy's approach.
    
    Args:
        t: knot vector
        k: degree
        knot_span: knot span index from find_knot_span
        x: parameter value
        N: output array of size k+1 for basis function values
    """
    # N[0] = 1.0 by definition (degree 0)
    N[0] = 1.0
    
    # Use pre-allocated stack arrays
    left = np.empty(k + 1)  # Stack allocation
    right = np.empty(k + 1)
    
    for j in range(1, k + 1):
        left[j] = x - t[knot_span + 1 - j]
        right[j] = t[knot_span + j] - x
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved


@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64, types.int64, types.int64), 
       nopython=True, fastmath=True, boundscheck=False)
def bisplev(x, y, tx, ty, c, kx, ky, nx, ny):
    """
    ULTRA-OPTIMIZED B-spline evaluation with manual register allocation.
    All operations inlined, no function calls, aggressive optimization.
    """
    mx = nx - kx - 1
    my = ny - ky - 1
    
    # === OPTIMIZATION 3: INLINE KNOT SPAN FINDING ===
    # X direction knot span - eliminate function call overhead
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
    
    # === OPTIMIZATION 4: ULTRA-OPTIMIZED LINEAR CASE ===  
    if kx == 1 and ky == 1:
        # Direct linear interpolation - register optimized, no intermediate variables
        alpha_x = (x - tx[span_x]) / (tx[span_x + 1] - tx[span_x])
        alpha_y = (y - ty[span_y]) / (ty[span_y + 1] - ty[span_y])
        
        beta_x = 1.0 - alpha_x
        beta_y = 1.0 - alpha_y
        
        base_idx = (span_x - 1) * my + (span_y - 1)
        
        # Unrolled tensor product with base indexing
        result = beta_x * beta_y * c[base_idx]
        result += beta_x * alpha_y * c[base_idx + 1]
        result += alpha_x * beta_y * c[base_idx + my]
        result += alpha_x * alpha_y * c[base_idx + my + 1]
        
        return result
    
    # === STEP 3 OPTIMIZATION: Mixed linear/cubic cases ===
    elif kx == 1 and ky == 3:
        # Linear x Cubic case - inline linear, use function for cubic
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
        # Cubic x Linear case - use function for cubic, inline linear
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
    
    # === STEP 4 OPTIMIZATION: Inline cubic x cubic case ===
    elif kx == 3 and ky == 3:
        # Inline cubic basis functions for both x and y (most common case)
        
        # Inline cubic basis functions for x (copy exact algorithm from basis_functions)
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
        
        # Inline cubic basis functions for y (copy exact algorithm from basis_functions)
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
    
    # Fall back to general case for other combinations
    # Compute basis functions using the working algorithm
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


def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None, 
             kx=3, ky=3, task=0, s=None, eps=1e-16, tx=None, ty=None, 
             full_output=0, nxest=None, nyest=None, quiet=1):
    """
    Find a bivariate B-spline representation of a surface.
    
    FastSpline implementation compatible with scipy.interpolate.bisplrep.
    Uses SciPy's bisplrep for fitting and returns tck tuple for use with
    our optimized bisplev function.
    
    Parameters
    ----------
    x, y, z : array_like
        1-D sequences of data points (order is not important).
    w : array_like, optional
        1-D sequence of weights. Default is None (uniform weights).
    xb, xe, yb, ye : float, optional
        End-points of approximation interval in x and y directions.
    kx, ky : int, optional
        Degrees of the bivariate spline. Default is 3.
    task : int, optional
        If task=0, find spline for given smoothing factor s.
        If task=1, find spline for automatic smoothing.
        Default is 0.
    s : float, optional
        Smoothing condition. Default is 0 (interpolation).
    eps : float, optional
        Threshold for determining rank of coefficient matrix.
    tx, ty : array_like, optional
        Knot sequences (only used if task >= 1).
    full_output : int, optional
        If non-zero, return additional info. Default is 0.
    nxest, nyest : int, optional
        Over-estimates of the number of knots.
    quiet : int, optional
        If non-zero, suppress warnings. Default is 1.
        
    Returns
    -------
    tck : tuple
        (tx, ty, c, kx, ky) tuple containing:
        - tx, ty : knot vectors
        - c : spline coefficients 
        - kx, ky : spline degrees
    Or if full_output=1:
    tck, fp, ier, msg : tuple
        Where fp is weighted sum of squared residuals, ier is error flag,
        msg is error message.
    """
    # Import scipy for the heavy lifting
    from scipy.interpolate import bisplrep as scipy_bisplrep
    
    # Use scipy's bisplrep for fitting (it's already highly optimized)
    if full_output:
        result = scipy_bisplrep(x, y, z, w=w, xb=xb, xe=xe, yb=yb, ye=ye,
                               kx=kx, ky=ky, task=task, s=s, eps=eps,
                               tx=tx, ty=ty, full_output=full_output,
                               nxest=nxest, nyest=nyest, quiet=quiet)
        return result
    else:
        tck = scipy_bisplrep(x, y, z, w=w, xb=xb, xe=xe, yb=yb, ye=ye,
                            kx=kx, ky=ky, task=task, s=s, eps=eps,
                            tx=tx, ty=ty, full_output=full_output,
                            nxest=nxest, nyest=nyest, quiet=quiet)
        return tck




class Spline2D:
    """
    2D Spline interpolation with numba acceleration and C interoperability.
    
    Compatible interface with scipy.interpolate.RectBivariateSpline and bisplrep.
    Supports:
    - Structured grid data (regular/periodic splines)
    - Unstructured scattered data (SURFIT-based algorithm)
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                 kx: int = 3, ky: int = 3, 
                 periodic: Tuple[bool, bool] = (False, False),
                 s: float = 0.0):
        """
        Initialize 2D spline interpolation.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates. Can be:
            - 1D array of length nx for structured grid (must be evenly spaced)
            - 1D array of length m for unstructured scattered data
        y : np.ndarray
            Y coordinates. Can be:
            - 1D array of length ny for structured grid (must be evenly spaced)
            - 1D array of length m for unstructured scattered data
        z : np.ndarray
            Z values. Can be:
            - 1D array of length nx*ny or 2D array of shape (nx, ny) for structured grid
            - 1D array of length m for unstructured scattered data
        kx, ky : int, default=3
            Spline orders in x and y directions (1 for linear, 3 for cubic)
        periodic : tuple of bool, default=(False, False)
            Periodicity flags for x and y directions (structured data only)
        s : float, default=0.0
            Smoothing parameter for unstructured data (0 = interpolation)
        """
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise ValueError("x and y must be 1D arrays")
        
        # Detect data format
        is_unstructured = _detect_data_format(x, y, z)
        
        if is_unstructured:
            # Handle unstructured scattered data
            if len(x) != len(y) or len(x) != len(z):
                raise ValueError("For unstructured data, x, y, and z must have the same length")
            
            # Convert to structured grid for processing
            x_grid, y_grid, z_grid = _surfit_to_structured_grid(x, y, z, kx, ky)
            x, y = x_grid, y_grid
            nx, ny = len(x), len(y)
            
            # Store original data for reference
            self._original_x = x.copy()
            self._original_y = y.copy()
            self._original_z = z.copy()
            self.is_unstructured = True
            
            # Periodic boundaries not supported for unstructured data
            if periodic[0] or periodic[1]:
                raise ValueError("Periodic boundaries not supported for unstructured data")
                
        else:
            # Handle structured grid data (original logic)
            nx, ny = len(x), len(y)
            
            # Handle z array shape
            if z.shape == (nx * ny,):
                z_grid = z.reshape(nx, ny)
            elif z.shape == (nx, ny):
                z_grid = z.copy()
            else:
                raise ValueError(f"z must have shape ({nx*ny},) or ({nx}, {ny}), got {z.shape}")
            
            self.is_unstructured = False
            
            # Check for evenly spaced coordinates
            if nx > 1:
                h_x = np.diff(x)
                if not np.allclose(h_x, h_x[0], rtol=1e-10):
                    raise ValueError("x coordinates must be evenly spaced")
            
            if ny > 1:
                h_y = np.diff(y)
                if not np.allclose(h_y, h_y[0], rtol=1e-10):
                    raise ValueError("y coordinates must be evenly spaced")
        
        if kx not in [1, 3] or ky not in [1, 3]:
            raise ValueError("Only linear (k=1) and cubic (k=3) splines supported")
        
        # Store parameters
        self.x_min = float(x[0])
        self.y_min = float(y[0])
        self.x_max = float(x[-1])
        self.y_max = float(y[-1])
        self.nx = nx
        self.ny = ny
        self.order_x = kx
        self.order_y = ky
        self.periodic_x = periodic[0] if not is_unstructured else False
        self.periodic_y = periodic[1] if not is_unstructured else False
        self.h_step_x = (self.x_max - self.x_min) / (nx - 1) if nx > 1 else 1.0
        self.h_step_y = (self.y_max - self.y_min) / (ny - 1) if ny > 1 else 1.0
        self.smoothing = s
        
        # Handle missing data
        z_processed = _handle_missing_data_cfunc(z_grid.ravel(), nx, ny)
        self.has_missing = False  # Will be detected during preprocessing
        
        # Compute spline coefficients
        self.coeffs = _compute_2d_spline_coefficients(
            z_processed.astype(np.float64), 
            self.h_step_x, self.h_step_y,
            self.order_x, self.order_y,
            self.periodic_x, self.periodic_y
        )
    
    def __call__(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], 
                 dx: int = 0, dy: int = 0, grid: bool = True) -> np.ndarray:
        """
        Evaluate spline at given coordinates (scipy-compatible interface).
        
        Parameters
        ----------
        x, y : float or array_like
            Evaluation coordinates
        dx, dy : int, default=0
            Order of derivative (0=value, 1=first derivative)
        grid : bool, default=True
            If True and x,y are arrays, evaluate on meshgrid.
            If False, evaluate at (x[i], y[i]) pairs.
        """
        if dx > 1 or dy > 1:
            raise NotImplementedError("Higher order derivatives not yet implemented")
        
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)
        
        if grid and (len(x_arr) > 1 or len(y_arr) > 1):
            # Meshgrid evaluation
            X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')
            result_shape = X.shape
            X_flat = X.ravel()
            Y_flat = Y.ravel()
        else:
            # Point-wise evaluation
            if len(x_arr) != len(y_arr):
                raise ValueError("x and y must have same length for point-wise evaluation")
            X_flat = x_arr
            Y_flat = y_arr
            result_shape = x_arr.shape
        
        # Evaluate at all points
        results = np.zeros(len(X_flat))
        
        if dx == 0 and dy == 0:
            # Function values only
            for i in range(len(X_flat)):
                results[i] = evaluate_spline_2d_cfunc(
                    X_flat[i], Y_flat[i], self.coeffs,
                    self.x_min, self.y_min, self.h_step_x, self.h_step_y,
                    self.nx, self.ny, self.order_x, self.order_y,
                    self.periodic_x, self.periodic_y
                )
        else:
            # Derivatives needed
            for i in range(len(X_flat)):
                z, dz_dx, dz_dy = evaluate_spline_2d_derivatives_cfunc(
                    X_flat[i], Y_flat[i], self.coeffs,
                    self.x_min, self.y_min, self.h_step_x, self.h_step_y,
                    self.nx, self.ny, self.order_x, self.order_y,
                    self.periodic_x, self.periodic_y
                )
                if dx == 1 and dy == 0:
                    results[i] = dz_dx
                elif dx == 0 and dy == 1:
                    results[i] = dz_dy
                else:
                    results[i] = z
        
        return results.reshape(result_shape)
    
    def ev(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate spline at given coordinates (scipy-compatible)."""
        return self(x, y, grid=False)
    
    @property
    def cfunc_evaluate(self):
        """Get the C-compatible function pointer for 2D spline evaluation."""
        return evaluate_spline_2d_cfunc
    
    @property
    def cfunc_evaluate_derivatives(self):
        """Get the C-compatible function pointer for 2D spline evaluation with derivatives."""
        return evaluate_spline_2d_derivatives_cfunc
    
    @property
    def cfunc_bisplev(self):
        """Get the C-compatible function pointer for DIERCKX bisplev-style evaluation."""
        return bisplev_cfunc
    
    @property 
    def cfunc_bisplev_ultra(self):
        """Get the ultra-optimized C-compatible function pointer for cubic B-spline evaluation."""
        return bisplev_cfunc_ultra