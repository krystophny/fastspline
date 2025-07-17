#!/usr/bin/env python3
"""
Test cubic spline to see if the issue is specific to quintic
"""

import numpy as np
import sys
import os
import ctypes

# Add the fastspline source to the path
sys.path.insert(0, os.path.abspath('../../src'))

from fastspline.sergei_splines import get_cfunc_addresses

def test_cubic():
    """Test cubic spline construction"""
    
    # Get cfunc addresses
    cfunc_addr = get_cfunc_addresses()
    
    construct_1d = ctypes.CFUNCTYPE(
        None,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double)
    )(cfunc_addr['construct_splines_1d'])
    
    # Test parameters
    n = 10
    x_min = 0.0
    x_max = 1.0
    order = 3  # Cubic
    periodic = 0
    
    # Create test data
    x = np.linspace(x_min, x_max, n)
    y = np.sin(2.0 * np.pi * x)
    
    # Construct spline
    y_c = (ctypes.c_double * n)(*y)
    coeff_size = (order + 1) * n
    coeff_c = (ctypes.c_double * coeff_size)()
    
    construct_1d(x_min, x_max, y_c, n, order, periodic, coeff_c)
    
    # Convert coefficients to numpy array
    coeffs = np.array([coeff_c[i] for i in range(coeff_size)]).reshape(order + 1, n)
    
    print("Cubic spline coefficients:")
    for i in range(order + 1):
        print(f"  Row {i}: min={coeffs[i].min():.6f}, max={coeffs[i].max():.6f}")
    
    # Check for reasonable values
    max_coeff = np.abs(coeffs).max()
    print(f"Maximum absolute coefficient: {max_coeff:.6f}")
    
    if max_coeff > 1000:
        print("WARNING: Cubic spline has large coefficients too!")
        return False
    else:
        print("Cubic spline looks reasonable")
        return True

if __name__ == "__main__":
    test_cubic()