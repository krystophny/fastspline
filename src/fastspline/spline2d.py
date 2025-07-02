"""2D Spline interpolation with numba acceleration using cfunc for C interoperability."""

import numpy as np
from numba import njit, cfunc, types
from typing import Tuple, Union
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


class Spline2D:
    """
    2D Spline interpolation with numba acceleration and C interoperability.
    
    Compatible interface with scipy.interpolate.RectBivariateSpline.
    Supports linear and cubic splines with regular or periodic boundary conditions.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                 kx: int = 3, ky: int = 3, 
                 periodic: Tuple[bool, bool] = (False, False)):
        """
        Initialize 2D spline interpolation.
        
        Parameters
        ----------
        x : np.ndarray, shape (nx,)
            X coordinates (must be evenly spaced)
        y : np.ndarray, shape (ny,)
            Y coordinates (must be evenly spaced)
        z : np.ndarray, shape (nx*ny,) or (nx, ny)
            Z values at grid points. If 1D, assumed to be in row-major order:
            z[i*ny + j] = f(x[i], y[j])
        kx, ky : int, default=3
            Spline orders in x and y directions (1 for linear, 3 for cubic)
        periodic : tuple of bool, default=(False, False)
            Periodicity flags for x and y directions
        """
        if len(x.shape) != 1 or len(y.shape) != 1:
            raise ValueError("x and y must be 1D arrays")
        
        nx, ny = len(x), len(y)
        
        # Handle z array shape
        if z.shape == (nx * ny,):
            z_grid = z.reshape(nx, ny)
        elif z.shape == (nx, ny):
            z_grid = z.copy()
        else:
            raise ValueError(f"z must have shape ({nx*ny},) or ({nx}, {ny}), got {z.shape}")
        
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
        self.periodic_x = periodic[0]
        self.periodic_y = periodic[1]
        self.h_step_x = (self.x_max - self.x_min) / (nx - 1) if nx > 1 else 1.0
        self.h_step_y = (self.y_max - self.y_min) / (ny - 1) if ny > 1 else 1.0
        
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