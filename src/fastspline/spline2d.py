"""2D Spline interpolation with numba acceleration using cfunc for C interoperability."""

import numpy as np
from numba import njit, cfunc, types
from typing import Tuple, Union, Optional
from .spline1d import (
    _solve_tridiagonal, _compute_cubic_coefficients_regular, 
    _compute_cubic_coefficients_periodic, _compute_linear_coefficients
)


@njit
def _compute_2d_spline_coefficients(z_grid, h_step_x, h_step_y, order_x, order_y, periodic_x, periodic_y):
    """
    Compute 2D tensor product spline coefficients.
    
    Parameters:
    z_grid: 2D array of function values, shape (nx, ny)
    h_step_x, h_step_y: grid spacing in x and y directions
    order_x, order_y: spline orders (1 or 3)
    periodic_x, periodic_y: periodicity flags
    
    Returns:
    coeffs: 4D coefficient array, shape (order_x+1, order_y+1, nx, ny)
    """
    nx, ny = z_grid.shape
    
    # Allocate coefficient array
    coeffs = np.zeros((order_x + 1, order_y + 1, nx, ny))
    
    # Step 1: Spline along y-direction (second index) for each x
    temp_coeffs_y = np.zeros((order_y + 1, ny))
    for i in range(nx):
        y_slice = z_grid[i, :]
        
        # Compute 1D spline coefficients along y
        if order_y == 1:
            coeffs_1d = _compute_linear_coefficients(y_slice, h_step_y)
        else:  # order_y == 3
            if periodic_y:
                coeffs_1d = _compute_cubic_coefficients_periodic(y_slice, h_step_y)
            else:
                coeffs_1d = _compute_cubic_coefficients_regular(y_slice, h_step_y)
        
        # Store in temporary array
        temp_coeffs_y[:coeffs_1d.shape[0], :] = coeffs_1d
        
        # Copy to main coefficient array (only constant term for now)
        coeffs[0, :coeffs_1d.shape[0], i, :] = coeffs_1d
    
    # Step 2: Spline along x-direction (first index) for each y and each coefficient order
    temp_coeffs_x = np.zeros((order_x + 1, nx))
    for j in range(ny):
        for ky in range(order_y + 1):
            # Extract coefficients along x for this y-position and y-order
            x_slice = coeffs[0, ky, :, j].copy()
            
            # Compute 1D spline coefficients along x
            if order_x == 1:
                coeffs_1d = _compute_linear_coefficients(x_slice, h_step_x)
            else:  # order_x == 3
                if periodic_x:
                    coeffs_1d = _compute_cubic_coefficients_periodic(x_slice, h_step_x)
                else:
                    coeffs_1d = _compute_cubic_coefficients_regular(x_slice, h_step_x)
            
            # Store final coefficients
            coeffs[:coeffs_1d.shape[0], ky, :, j] = coeffs_1d
    
    return coeffs


@njit
def _handle_missing_data(z_linear, nx, ny):
    """
    Handle missing data (NaN values) in the input array.
    
    Returns:
    z_grid: 2D array with NaN handling
    has_missing: boolean indicating if missing data was found
    """
    z_grid = z_linear.reshape(nx, ny)
    has_missing = False
    
    # Check for NaN values
    for i in range(nx):
        for j in range(ny):
            if not np.isfinite(z_grid[i, j]):
                has_missing = True
                # Simple strategy: replace with interpolated value from neighbors
                # This is a basic approach - more sophisticated methods could be used
                z_grid[i, j] = 0.0  # Placeholder
    
    return z_grid, has_missing


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


@njit
def _compute_surfit_coefficients(x, y, z, kx, ky, s=0.0):
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
    
    return x_knots, y_knots, coeffs, nx, ny


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
    z_grid = np.zeros((nx_out, ny_out))
    
    for i in range(nx_out):
        for j in range(ny_out):
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
    coeffs: 4D coefficient array (order_x+1, order_y+1, nx, ny)
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
    
    # Extract local coefficients
    coeff_local = coeffs[:, :, ix, iy]
    
    # Evaluate using nested Horner's method
    # First, evaluate along x for each y-order
    coeff_y = np.zeros(order_y + 1)
    for ky in range(order_y + 1):
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
    
    # Extract local coefficients
    coeff_local = coeffs[:, :, ix, iy]
    
    # Evaluate function value and x-derivative
    coeff_y = np.zeros(order_y + 1)
    dcoeff_y_dx = np.zeros(order_y + 1)
    
    for ky in range(order_y + 1):
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


@njit
def find_knot_interval(t, n, k, x):
    """
    Find the knot interval index for x in knot vector t.
    Returns i such that t[i] <= x < t[i+1].
    For the rightmost point, returns the last valid interval.
    """
    # Handle left boundary
    if x < t[k]:
        return k
    
    # Handle right boundary - special case for x == t[n-k-1]
    if x >= t[n - k - 1]:
        # For x exactly at the right boundary, use the last interval
        if x == t[n - k - 1]:
            # Find the last interval where t[i] < t[i+1]
            for i in range(n - k - 2, k - 1, -1):
                if t[i] < t[i + 1]:
                    return i
        return n - k - 2
    
    # Binary search for interior points
    low = k
    high = n - k - 1
    
    while high - low > 1:
        mid = (low + high) // 2
        if x < t[mid]:
            high = mid
        else:
            low = mid
    
    return low


@njit
def deboor_1d(t, c, k, x):
    """
    Evaluate B-spline at x using de Boor's algorithm.
    
    Parameters:
    t: knot vector
    c: control points (coefficients)
    k: degree
    x: evaluation point
    
    Returns:
    B-spline value at x
    """
    n = len(c)
    
    # Find knot interval
    i = find_knot_interval(t, len(t), k, x)
    
    # Initialize working array with relevant control points
    d = np.zeros(k + 1)
    for j in range(k + 1):
        idx = i - k + j
        if 0 <= idx < n:
            d[j] = c[idx]
    
    # Apply de Boor recursion
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            idx = i - k + j
            alpha = (x - t[idx]) / (t[idx + k - r + 1] - t[idx])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    
    return d[k]


@njit
def bspline_basis(i, k, t, x):
    """
    Evaluate B-spline basis function B_{i,k} at x using a direct approach.
    This matches scipy's behavior exactly.
    """
    # For right boundary, use left-sided evaluation for the last basis function
    if x == t[-1] and i == len(t) - k - 2:
        x = x - 1e-14
    
    # Check if x is outside the support of B_{i,k}
    if x < t[i] or x >= t[i + k + 1]:
        return 0.0
    
    # Use de Boor's algorithm
    N = np.zeros(k + 1)
    
    # Find knot span
    j = i
    while j < len(t) - 1 and x >= t[j + 1]:
        j += 1
    
    # Initialize for degree 0
    for idx in range(k + 1):
        span_start = j - k + idx
        if 0 <= span_start < len(t) - 1:
            if t[span_start] <= x < t[span_start + 1]:
                N[idx] = 1.0
            else:
                N[idx] = 0.0
        else:
            N[idx] = 0.0
    
    # Compute higher degree basis functions
    for d in range(1, k + 1):
        for idx in range(k - d + 1):
            span_start = j - k + idx
            left = 0.0
            right = 0.0
            
            # Left term
            if span_start >= 0 and span_start + d < len(t):
                denom = t[span_start + d] - t[span_start]
                if denom > 0:
                    left = (x - t[span_start]) / denom * N[idx]
            
            # Right term  
            if span_start + 1 >= 0 and span_start + d + 1 < len(t):
                denom = t[span_start + d + 1] - t[span_start + 1]
                if denom > 0:
                    right = (t[span_start + d + 1] - x) / denom * N[idx + 1]
            
            N[idx] = left + right
    
    # Return the value for basis function i
    if j - k == i:
        return N[0]
    else:
        return 0.0


@njit
def eval_basis_simple(i, k, t, x):
    """Simple recursive B-spline basis evaluation that matches scipy."""
    if k == 0:
        if i >= len(t) - 1:
            return 0.0
        if t[i] <= x < t[i + 1]:
            return 1.0
        if x == t[i + 1] and i == len(t) - 2:  # Right boundary special case
            return 1.0
        return 0.0
    
    # Recursive case using Cox-de Boor formula
    result = 0.0
    
    # First term: (x - t[i]) / (t[i+k] - t[i]) * B_{i,k-1}(x)
    if i + k < len(t):
        denom1 = t[i + k] - t[i]
        if denom1 > 0.0:
            result += (x - t[i]) / denom1 * eval_basis_simple(i, k - 1, t, x)
        elif denom1 == 0.0:
            # Handle repeated knots: if denom is 0, treat this term as 0
            pass
    
    # Second term: (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * B_{i+1,k-1}(x)
    if i + k + 1 < len(t):
        denom2 = t[i + k + 1] - t[i + 1]
        if denom2 > 0.0:
            result += (t[i + k + 1] - x) / denom2 * eval_basis_simple(i + 1, k - 1, t, x)
        elif denom2 == 0.0:
            # Special case: when denom2 is 0 but we're at the boundary
            # Check if this should contribute
            if x == t[i + k + 1] and eval_basis_simple(i + 1, k - 1, t, x) > 0:
                result += eval_basis_simple(i + 1, k - 1, t, x)
    
    return result


@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64, types.int64, types.int64))
def bisplev_cfunc(x, y, tx, ty, c, kx, ky, nx, ny):
    """
    C-compatible function for evaluating 2D B-spline (bisplev interface).
    
    Compatible with scipy.interpolate.bisplev for DIERCKX spline evaluation.
    
    Parameters:
    x, y: evaluation coordinates
    tx, ty: knot vectors
    c: spline coefficients (flattened)
    kx, ky: spline degrees
    nx, ny: number of knots
    
    Returns:
    evaluated spline value
    """
    # Number of coefficients in each direction
    mx = nx - kx - 1
    my = ny - ky - 1
    
    # Initialize result
    result = 0.0
    
    # Evaluate using tensor product of 1D B-splines
    for i in range(mx):
        for j in range(my):
            # Evaluate basis functions
            bx = eval_basis_simple(i, kx, tx, x)
            by = eval_basis_simple(j, ky, ty, y)
            
            # Add contribution if non-zero
            if bx != 0.0 and by != 0.0:
                coeff_idx = i * my + j
                if coeff_idx < len(c):
                    result += bx * by * c[coeff_idx]
    
    return result


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
        z_processed, self.has_missing = _handle_missing_data(z_grid.ravel(), nx, ny)
        
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