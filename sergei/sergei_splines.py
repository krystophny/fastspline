"""
Sergei's spline implementation in pure Numba cfunc
Based on interpolate.f90
"""

import numpy as np
from numba import cfunc, types
from numba.core import cgutils
from numba.core.typing import signature
import ctypes


# Type signatures for cfuncs
spl_reg_sig = types.void(
    types.int32,           # order
    types.int32,           # num_points
    types.float64,         # h_step
    types.CPointer(types.float64)  # coeff array (order+1, num_points)
)

spl_per_sig = types.void(
    types.int32,           # order
    types.int32,           # num_points  
    types.float64,         # h_step
    types.CPointer(types.float64)  # coeff array (order+1, num_points)
)

evaluate_1d_sig = types.void(
    types.int32,           # order
    types.int32,           # num_points
    types.int32,           # periodic (0 or 1)
    types.float64,         # x_min
    types.float64,         # h_step
    types.CPointer(types.float64),  # coeff array
    types.float64,         # x
    types.CPointer(types.float64)   # output y
)

evaluate_2d_sig = types.void(
    types.CPointer(types.int32),     # order array (2)
    types.CPointer(types.int32),     # num_points array (2)
    types.CPointer(types.int32),     # periodic array (2) - 0 or 1
    types.CPointer(types.float64),   # x_min array (2)
    types.CPointer(types.float64),   # h_step array (2)
    types.CPointer(types.float64),   # coeff array
    types.CPointer(types.float64),   # x array (2)
    types.CPointer(types.float64)    # output y
)


@cfunc(spl_reg_sig, nopython=True)
def spl_reg_cfunc(order, num_points, h_step, coeff):
    """
    Compute spline coefficients for regular (non-periodic) splines.
    This implements natural spline boundary conditions.
    """
    # For simplicity, implementing cubic splines (order=3) with natural boundary conditions
    # This is a placeholder - proper implementation would handle all orders
    
    if order == 3:
        # Cubic spline with natural boundary conditions
        # Set up tridiagonal system for second derivatives
        n = num_points
        
        # Working arrays - in real implementation these would be workspace
        # For now, using stack allocation pattern
        d = np.empty(n, dtype=np.float64)
        u = np.empty(n-1, dtype=np.float64)
        v = np.empty(n, dtype=np.float64)
        
        # Set up tridiagonal matrix
        for i in range(1, n-1):
            d[i] = 2.0
            u[i-1] = 0.5
            v[i] = 3.0 * (coeff[i+1] - coeff[i-1]) / h_step
        
        # Natural boundary conditions
        d[0] = 1.0
        d[n-1] = 1.0
        v[0] = 0.0
        v[n-1] = 0.0
        
        # Forward elimination
        for i in range(1, n):
            m = u[i-1] / d[i-1] if i < n-1 else 0.0
            d[i] = d[i] - m * 0.5
            v[i] = v[i] - m * v[i-1]
        
        # Back substitution to get second derivatives
        coeff[2*num_points + n-1] = v[n-1] / d[n-1]
        for i in range(n-2, -1, -1):
            coeff[2*num_points + i] = (v[i] - 0.5 * coeff[2*num_points + i+1]) / d[i]
        
        # Compute first derivatives
        for i in range(n-1):
            coeff[num_points + i] = (coeff[i+1] - coeff[i]) / h_step - \
                                   h_step * (2.0 * coeff[2*num_points + i] + \
                                           coeff[2*num_points + i+1]) / 6.0
        coeff[num_points + n-1] = coeff[num_points + n-2]
        
        # Third derivatives (constant within each interval)
        for i in range(n-1):
            coeff[3*num_points + i] = (coeff[2*num_points + i+1] - \
                                      coeff[2*num_points + i]) / h_step
        coeff[3*num_points + n-1] = 0.0


@cfunc(spl_per_sig, nopython=True)
def spl_per_cfunc(order, num_points, h_step, coeff):
    """
    Compute spline coefficients for periodic splines.
    """
    # For periodic splines, we enforce continuity at boundaries
    # This is a simplified implementation for cubic splines
    
    if order == 3:
        n = num_points
        
        # Ensure periodicity in function values
        coeff[n-1] = coeff[0]
        
        # Set up cyclic tridiagonal system
        d = np.empty(n, dtype=np.float64)
        u = np.empty(n, dtype=np.float64)
        v = np.empty(n, dtype=np.float64)
        
        # Fill tridiagonal matrix
        for i in range(n):
            d[i] = 4.0
            u[i] = 1.0
            v[i] = 3.0 * (coeff[(i+1)%n] - coeff[(i-1+n)%n]) / h_step
        
        # Solve cyclic tridiagonal system using Sherman-Morrison formula
        # This is simplified - full implementation would be more complex
        
        # For now, approximate with natural spline
        spl_reg_cfunc(order, num_points, h_step, coeff)


@cfunc(evaluate_1d_sig, nopython=True)
def evaluate_spline_1d_cfunc(order, num_points, periodic, x_min, h_step, 
                            coeff, x, y_out):
    """
    Evaluate 1D spline at point x.
    """
    # Handle periodic boundary conditions
    xj = x
    if periodic:
        period = h_step * (num_points - 1)
        xj = x - np.floor((x - x_min) / period) * period
    
    # Find interval
    x_norm = (xj - x_min) / h_step
    interval_index = int(x_norm)
    
    # Clamp to valid range
    if interval_index < 0:
        interval_index = 0
    elif interval_index >= num_points - 1:
        interval_index = num_points - 2
    
    # Local coordinate within interval
    x_local = (x_norm - interval_index) * h_step
    
    # Horner's method for polynomial evaluation
    y = coeff[order * num_points + interval_index]
    for k in range(order - 1, -1, -1):
        y = coeff[k * num_points + interval_index] + x_local * y
    
    y_out[0] = y


@cfunc(evaluate_2d_sig, nopython=True)
def evaluate_spline_2d_cfunc(order, num_points, periodic, x_min, h_step,
                            coeff, x, y_out):
    """
    Evaluate 2D spline at point (x[0], x[1]).
    """
    # Find intervals for both dimensions
    interval_index = np.empty(2, dtype=np.int32)
    x_local = np.empty(2, dtype=np.float64)
    
    for j in range(2):
        xj = x[j]
        if periodic[j]:
            period = h_step[j] * (num_points[j] - 1)
            xj = x[j] - np.floor((x[j] - x_min[j]) / period) * period
        
        x_norm = (xj - x_min[j]) / h_step[j]
        interval_index[j] = int(x_norm)
        
        # Clamp to valid range
        if interval_index[j] < 0:
            interval_index[j] = 0
        elif interval_index[j] >= num_points[j] - 1:
            interval_index[j] = num_points[j] - 2
        
        x_local[j] = (x_norm - interval_index[j]) * h_step[j]
    
    # Get local coefficients
    # coeff layout: (order[0]+1, order[1]+1, num_points[0], num_points[1])
    stride1 = (order[1] + 1) * num_points[0] * num_points[1]
    stride2 = num_points[0] * num_points[1]
    stride3 = num_points[1]
    
    base_idx = interval_index[0] * stride3 + interval_index[1]
    
    # Evaluate over second dimension first
    coeff_1d = np.empty(order[0] + 1, dtype=np.float64)
    
    for k1 in range(order[0] + 1):
        # Horner's method for dimension 2
        y_temp = coeff[k1 * stride1 + order[1] * stride2 + base_idx]
        for k2 in range(order[1] - 1, -1, -1):
            y_temp = coeff[k1 * stride1 + k2 * stride2 + base_idx] + x_local[1] * y_temp
        coeff_1d[k1] = y_temp
    
    # Evaluate over first dimension
    y = coeff_1d[order[0]]
    for k1 in range(order[0] - 1, -1, -1):
        y = coeff_1d[k1] + x_local[0] * y
    
    y_out[0] = y


# Python wrapper functions for easier use
def construct_spline_1d(x_min, x_max, y_values, order=3, periodic=False):
    """
    Construct 1D spline from data points.
    """
    num_points = len(y_values)
    h_step = (x_max - x_min) / (num_points - 1)
    
    # Allocate coefficient array
    coeff = np.zeros((order + 1, num_points), dtype=np.float64, order='C')
    coeff[0, :] = y_values
    
    # Compute spline coefficients
    coeff_ptr = coeff.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    if periodic:
        spl_per_cfunc(order, num_points, h_step, coeff_ptr)
    else:
        spl_reg_cfunc(order, num_points, h_step, coeff_ptr)
    
    return {
        'order': order,
        'num_points': num_points,
        'periodic': periodic,
        'x_min': x_min,
        'h_step': h_step,
        'coeff': coeff
    }


def evaluate_spline_1d(spline, x):
    """
    Evaluate 1D spline at point(s) x.
    """
    x = np.atleast_1d(x)
    y = np.zeros_like(x)
    
    coeff_ptr = spline['coeff'].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    for i, xi in enumerate(x):
        y_ptr = y[i:i+1].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        evaluate_spline_1d_cfunc(
            spline['order'], 
            spline['num_points'],
            1 if spline['periodic'] else 0,
            spline['x_min'],
            spline['h_step'],
            coeff_ptr,
            xi,
            y_ptr
        )
    
    return y if len(x) > 1 else y[0]


def construct_spline_2d(x_min, x_max, z_values, order=(3, 3), periodic=(False, False)):
    """
    Construct 2D spline from grid data.
    """
    num_points = z_values.shape
    h_step = np.array([(x_max[i] - x_min[i]) / (num_points[i] - 1) for i in range(2)])
    
    # Allocate coefficient array
    coeff = np.zeros((order[0] + 1, order[1] + 1, num_points[0], num_points[1]), 
                     dtype=np.float64, order='C')
    coeff[0, 0, :, :] = z_values
    
    # Compute spline coefficients in y-direction first
    for i in range(num_points[0]):
        for ky in range(order[1] + 1):
            spl_1d = construct_spline_1d(
                x_min[1], x_max[1], 
                coeff[0, ky, i, :],
                order[1], 
                periodic[1]
            )
            coeff[0, :, i, :] = spl_1d['coeff']
    
    # Then in x-direction
    for j in range(num_points[1]):
        for ky in range(order[1] + 1):
            for kx in range(order[0] + 1):
                spl_1d = construct_spline_1d(
                    x_min[0], x_max[0],
                    coeff[kx, ky, :, j],
                    order[0],
                    periodic[0]
                )
                coeff[:, ky, :, j] = spl_1d['coeff']
    
    return {
        'order': order,
        'num_points': num_points,
        'periodic': periodic,
        'x_min': x_min,
        'h_step': h_step,
        'coeff': coeff
    }


def evaluate_spline_2d(spline, x):
    """
    Evaluate 2D spline at point(s) x.
    x should be shape (n, 2) or (2,) for single point.
    """
    x = np.atleast_2d(x)
    if x.shape[1] != 2:
        x = x.T
    
    y = np.zeros(len(x))
    
    # Prepare arrays for cfunc
    order_arr = np.array(spline['order'], dtype=np.int32)
    num_points_arr = np.array(spline['num_points'], dtype=np.int32)
    periodic_arr = np.array([1 if p else 0 for p in spline['periodic']], dtype=np.int32)
    x_min_arr = np.array(spline['x_min'], dtype=np.float64)
    h_step_arr = np.array(spline['h_step'], dtype=np.float64)
    
    # Get pointers
    order_ptr = order_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    num_points_ptr = num_points_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    periodic_ptr = periodic_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    x_min_ptr = x_min_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    h_step_ptr = h_step_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    coeff_ptr = spline['coeff'].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    for i in range(len(x)):
        x_point = x[i].copy()
        x_ptr = x_point.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ptr = y[i:i+1].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        evaluate_spline_2d_cfunc(
            order_ptr, num_points_ptr, periodic_ptr,
            x_min_ptr, h_step_ptr, coeff_ptr,
            x_ptr, y_ptr
        )
    
    return y if len(x) > 1 else y[0]