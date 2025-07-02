"""1D Spline interpolation with numba acceleration."""

import numpy as np
from numba import njit
from typing import Tuple


@njit
def _solve_tridiagonal(a, b, c, d):
    """
    Solve tridiagonal system Ax = d where A has diagonals a, b, c.
    Uses Thomas algorithm with forward elimination and back substitution.
    """
    n = len(d)
    x = np.zeros(n)
    
    # Forward elimination
    for i in range(1, n):
        m = a[i] / b[i-1]
        b[i] = b[i] - m * c[i-1]
        d[i] = d[i] - m * d[i-1]
    
    # Back substitution
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x


@njit
def _compute_cubic_coefficients_regular(y, h):
    """
    Compute cubic spline coefficients for regular (non-periodic) boundary conditions.
    Uses "not-a-knot" boundary conditions which preserve polynomial accuracy.
    """
    n = len(y)
    
    if n < 4:
        # For small number of points, use simpler approach
        coeffs = np.zeros((4, n))
        coeffs[0, :] = y
        
        for i in range(n-1):
            coeffs[1, i] = (y[i+1] - y[i]) / h
        
        if n > 2:
            coeffs[1, n-1] = coeffs[1, n-2]
        
        return coeffs
    
    # Set up tridiagonal system for not-a-knot spline
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    # Not-a-knot boundary conditions
    # First boundary condition: third derivative continuous at x_1
    b[0] = h
    c[0] = -h
    d[0] = 0.0
    
    # Last boundary condition: third derivative continuous at x_{n-2}
    a[n-1] = -h
    b[n-1] = h
    d[n-1] = 0.0
    
    # Interior points
    for i in range(1, n-1):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        d[i] = 6.0 * (y[i+1] - 2.0*y[i] + y[i-1]) / (h * h)
    
    # Solve for second derivatives
    s = _solve_tridiagonal(a, b, c, d)
    
    # Compute polynomial coefficients
    coeffs = np.zeros((4, n))
    coeffs[0, :] = y
    
    for i in range(n-1):
        coeffs[1, i] = (y[i+1] - y[i]) / h - h * (2.0*s[i] + s[i+1]) / 6.0
        coeffs[2, i] = s[i] / 2.0
        coeffs[3, i] = (s[i+1] - s[i]) / (6.0 * h)
    
    # Last interval
    coeffs[1, n-1] = coeffs[1, n-2]
    coeffs[2, n-1] = s[n-1] / 2.0
    coeffs[3, n-1] = 0.0
    
    return coeffs


@njit
def _compute_cubic_coefficients_periodic(y, h):
    """
    Compute cubic spline coefficients for periodic boundary conditions.
    """
    n = len(y)
    
    # Set up cyclic tridiagonal system
    a = np.full(n, h)
    b = np.full(n, 4.0 * h)
    c = np.full(n, h)
    d = np.zeros(n)
    
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        d[i] = 6.0 * (y[ip1] - 2.0*y[i] + y[im1])
    
    # Solve cyclic system (simplified approach)
    # For periodic case, we modify the first and last equations
    b[0] = 4.0 * h
    c[0] = 2.0 * h
    a[n-1] = 2.0 * h
    b[n-1] = 4.0 * h
    
    c_coeffs = _solve_tridiagonal(a, b, c, d)
    
    # Compute all coefficients
    coeffs = np.zeros((4, n))
    coeffs[0, :] = y
    
    for i in range(n):
        ip1 = (i + 1) % n
        coeffs[1, i] = (y[ip1] - y[i]) / h - h * (2.0*c_coeffs[i] + c_coeffs[ip1]) / 6.0
        coeffs[2, i] = c_coeffs[i] / 2.0
        coeffs[3, i] = (c_coeffs[ip1] - c_coeffs[i]) / (6.0 * h)
    
    return coeffs


@njit
def _compute_linear_coefficients(y, h):
    """Compute linear spline coefficients."""
    n = len(y)
    coeffs = np.zeros((2, n))
    coeffs[0, :] = y
    
    for i in range(n-1):
        coeffs[1, i] = (y[i+1] - y[i]) / h
    
    coeffs[1, n-1] = coeffs[1, n-2]
    return coeffs


@njit
def _evaluate_polynomial(coeffs, x_local, order):
    """Evaluate polynomial using Horner's method."""
    result = coeffs[order]
    for k in range(order-1, -1, -1):
        result = coeffs[k] + x_local * result
    return result


@njit
def _evaluate_polynomial_derivative(coeffs, x_local, order):
    """Evaluate polynomial and its derivative."""
    # Function value
    y = coeffs[order]
    for k in range(order-1, -1, -1):
        y = coeffs[k] + x_local * y
    
    # Derivative
    if order == 0:
        dy = 0.0
    else:
        dy = coeffs[order] * order
        for k in range(order-1, 0, -1):
            dy = coeffs[k] * k + x_local * dy
    
    return y, dy


@njit
def _evaluate_polynomial_second_derivative(coeffs, x_local, order):
    """Evaluate polynomial and its first and second derivatives."""
    # Function value
    y = coeffs[order]
    for k in range(order-1, -1, -1):
        y = coeffs[k] + x_local * y
    
    # First derivative
    if order == 0:
        dy = 0.0
    else:
        dy = coeffs[order] * order
        for k in range(order-1, 0, -1):
            dy = coeffs[k] * k + x_local * dy
    
    # Second derivative
    if order <= 1:
        d2y = 0.0
    else:
        d2y = coeffs[order] * order * (order - 1)
        for k in range(order-1, 1, -1):
            d2y = coeffs[k] * k * (k - 1) + x_local * d2y
    
    return y, dy, d2y


@njit
def _find_interval_and_local_coord(x, x_min, h_step, num_points, periodic):
    """Find interval index and local coordinate for interpolation."""
    if periodic:
        # Handle periodic case
        x_period = h_step * (num_points - 1)
        xj = x - x_min
        xj = xj - np.floor(xj / x_period) * x_period + x_min
    else:
        xj = x
    
    x_norm = (xj - x_min) / h_step
    interval_index = max(0, min(num_points - 1, int(x_norm)))
    x_local = (x_norm - float(interval_index)) * h_step
    
    return interval_index, x_local


class Spline1D:
    """
    1D Spline interpolation with numba acceleration.
    
    Supports linear and cubic splines with regular or periodic boundary conditions.
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, order: int = 3, periodic: bool = False):
        """
        Initialize 1D spline.
        
        Parameters
        ----------
        x : np.ndarray
            X coordinates (must be evenly spaced)
        y : np.ndarray
            Y values at x coordinates
        order : int, default=3
            Spline order (1 for linear, 3 for cubic)
        periodic : bool, default=False
            Whether to use periodic boundary conditions
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        
        if len(x) < 2:
            raise ValueError("Need at least 2 points")
        
        if order not in [1, 3]:
            raise ValueError("Only linear (order=1) and cubic (order=3) splines supported")
        
        if order == 3 and len(x) < 4:
            raise ValueError("Need at least 4 points for cubic spline")
        
        # Check if x is evenly spaced
        if len(x) > 2:
            h_steps = np.diff(x)
            if not np.allclose(h_steps, h_steps[0], rtol=1e-10):
                raise ValueError("x coordinates must be evenly spaced")
        
        self.x_min = float(x[0])
        self.x_max = float(x[-1])
        self.num_points = len(x)
        self.order = order
        self.periodic = periodic
        self.h_step = (self.x_max - self.x_min) / (self.num_points - 1)
        
        # Compute coefficients
        if order == 1:
            self.coeffs = _compute_linear_coefficients(y.astype(np.float64), self.h_step)
        elif order == 3:
            if periodic:
                self.coeffs = _compute_cubic_coefficients_periodic(y.astype(np.float64), self.h_step)
            else:
                self.coeffs = _compute_cubic_coefficients_regular(y.astype(np.float64), self.h_step)
    
    def evaluate(self, x: float) -> float:
        """Evaluate spline at given x coordinate."""
        interval_index, x_local = _find_interval_and_local_coord(
            x, self.x_min, self.h_step, self.num_points, self.periodic
        )
        
        coeff_local = self.coeffs[:, interval_index]
        return _evaluate_polynomial(coeff_local, x_local, self.order)
    
    def evaluate_with_derivative(self, x: float) -> Tuple[float, float]:
        """Evaluate spline and its derivative at given x coordinate."""
        interval_index, x_local = _find_interval_and_local_coord(
            x, self.x_min, self.h_step, self.num_points, self.periodic
        )
        
        coeff_local = self.coeffs[:, interval_index]
        return _evaluate_polynomial_derivative(coeff_local, x_local, self.order)
    
    def evaluate_with_second_derivative(self, x: float) -> Tuple[float, float, float]:
        """Evaluate spline and its first and second derivatives at given x coordinate."""
        interval_index, x_local = _find_interval_and_local_coord(
            x, self.x_min, self.h_step, self.num_points, self.periodic
        )
        
        coeff_local = self.coeffs[:, interval_index]
        return _evaluate_polynomial_second_derivative(coeff_local, x_local, self.order)