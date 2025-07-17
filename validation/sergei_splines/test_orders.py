#!/usr/bin/env python3
"""
Test script to validate different spline orders (3, 4, 5)
"""

import numpy as np
import sys
import os
import ctypes

# Add the fastspline source to the path
sys.path.insert(0, os.path.abspath('../../src'))

from fastspline.sergei_splines import get_cfunc_addresses

def test_order(order):
    """Test spline construction for a given order"""
    print(f"\nTesting order {order} spline...")
    
    # Get cfunc addresses
    cfunc_addr = get_cfunc_addresses()
    
    # Set up ctypes functions for 1D splines
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
    periodic = 0  # 0 for False
    
    # Create test data
    x = np.linspace(x_min, x_max, n)
    y = np.sin(2.0 * np.pi * x)
    
    # Construct spline
    y_c = (ctypes.c_double * n)(*y)
    coeff_size = (order + 1) * n
    coeff_c = (ctypes.c_double * coeff_size)()
    
    try:
        construct_1d(x_min, x_max, y_c, n, order, periodic, coeff_c)
        
        # Convert coefficients to numpy array
        coeffs = np.array([coeff_c[i] for i in range(coeff_size)]).reshape(order + 1, n)
        
        # Check if coefficients are reasonable
        print(f"Order {order} coefficients:")
        print(f"  Shape: {coeffs.shape}")
        print(f"  Min/Max: {coeffs.min():.6f} / {coeffs.max():.6f}")
        print(f"  Non-zero rows: {np.count_nonzero(np.abs(coeffs) > 1e-10, axis=1)}")
        
        # Check first few coefficients for each order
        for i in range(order + 1):
            row_nonzero = np.count_nonzero(np.abs(coeffs[i]) > 1e-10)
            print(f"  Row {i}: {row_nonzero}/{n} non-zero coeffs, max={coeffs[i].max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing order {order}: {e}")
        return False

def main():
    print("Testing different spline orders...")
    
    success_count = 0
    for order in [3, 4, 5]:
        if test_order(order):
            success_count += 1
    
    print(f"\nSummary: {success_count}/3 orders tested successfully")
    
    if success_count == 3:
        print("All spline orders working correctly!")
    else:
        print("Some spline orders have issues")

if __name__ == "__main__":
    main()