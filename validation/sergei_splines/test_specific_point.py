#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import fastspline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastspline.sergei_splines import construct_splines_2d_cfunc

def test_function(x, y):
    """Test function: sin(π*x) * cos(π*y)"""
    return np.sin(np.pi * x) * np.cos(np.pi * y)

def evaluate_splines_2d_debug(order, num_points, periodic, x_min, h_step, coeff, x, y_out):
    """Debug version of 2D evaluation"""
    
    # Extract parameters
    o1 = order[0]
    o2 = order[1]
    n1 = num_points[0]
    n2 = num_points[1]
    
    print(f"Debug evaluation at point ({x[0]:.3f}, {x[1]:.3f})")
    print(f"Orders: o1={o1}, o2={o2}")
    print(f"Grid size: n1={n1}, n2={n2}")
    
    # Find intervals for both dimensions
    x_norm_1 = (x[0] - x_min[0]) / h_step[0]
    interval_1 = int(x_norm_1)
    if interval_1 < 0:
        interval_1 = 0
    elif interval_1 >= n1 - 1:
        interval_1 = n1 - 2
    x_local_1 = (x_norm_1 - interval_1) * h_step[0]
    
    x_norm_2 = (x[1] - x_min[1]) / h_step[1]
    interval_2 = int(x_norm_2)
    if interval_2 < 0:
        interval_2 = 0
    elif interval_2 >= n2 - 1:
        interval_2 = n2 - 2
    x_local_2 = (x_norm_2 - interval_2) * h_step[1]
    
    print(f"Intervals: interval_1={interval_1}, interval_2={interval_2}")
    print(f"Local coords: x_local_1={x_local_1:.6f}, x_local_2={x_local_2:.6f}")
    
    # Extract coefficients and evaluate - matching Fortran approach exactly
    coeff_2 = [0.0] * (o2 + 1)
    print(f"Coefficients for this interval:")
    
    for k2 in range(o2 + 1):
        # Start with highest order k1 = o1
        idx = o1*(o2+1)*n1*n2 + k2*n1*n2 + interval_1*n2 + interval_2
        coeff_2[k2] = coeff[idx]
        print(f"  k1={o1}, k2={k2}: coeff[{idx}] = {coeff[idx]:.6f}")
        
        # Evaluate polynomial in x_local_1 using Horner's method
        for k1 in range(o1 - 1, -1, -1):
            idx = k1*(o2+1)*n1*n2 + k2*n1*n2 + interval_1*n2 + interval_2
            old_val = coeff_2[k2]
            coeff_2[k2] = coeff[idx] + x_local_1 * coeff_2[k2]
            print(f"    k1={k1}, k2={k2}: coeff[{idx}] = {coeff[idx]:.6f}, "
                  f"coeff_2[{k2}] = {coeff[idx]:.6f} + {x_local_1:.6f} * {old_val:.6f} = {coeff_2[k2]:.6f}")
    
    print(f"Final coeff_2 array: {coeff_2}")
    
    # Now evaluate along dimension 2 using coeff_2
    y = coeff_2[o2]
    print(f"Starting with y = coeff_2[{o2}] = {y:.6f}")
    
    for k2 in range(o2 - 1, -1, -1):
        old_y = y
        y = coeff_2[k2] + x_local_2 * y
        print(f"k2={k2}: y = {coeff_2[k2]:.6f} + {x_local_2:.6f} * {old_y:.6f} = {y:.6f}")
    
    print(f"Final result: {y:.6f}")
    y_out[0] = y

def test_specific_point():
    """Test the specific problem point"""
    
    # 8x8 grid as in clean_2d_comparison.py
    nx, ny = 8, 8
    x_min = np.array([0.0, 0.0])
    x_max = np.array([1.0, 1.0])
    order = 3
    orders_2d = np.array([order, order])
    periodic_2d = np.array([False, False])
    
    # Create data grid
    x_data = np.linspace(0, 1, nx)
    y_data = np.linspace(0, 1, ny)
    X_data, Y_data = np.meshgrid(x_data, y_data)
    Z_data = test_function(X_data, Y_data)
    
    print("Testing Specific Problem Point")
    print("=" * 40)
    print(f"Grid size: {nx}×{ny}")
    print(f"Function: sin(π*x) * cos(π*y)")
    print(f"Order: {order}")
    print()
    
    # Construct spline
    z_flat = Z_data.flatten()
    coeff_2d = np.zeros((order+1)**2 * nx * ny)
    workspace_y = np.zeros(nx * ny)
    workspace_coeff = np.zeros((order+1) * nx * ny)
    
    construct_splines_2d_cfunc(x_min, x_max, z_flat, 
                              np.array([nx, ny]), orders_2d, periodic_2d, 
                              coeff_2d, workspace_y, workspace_coeff)
    
    print("Construction completed")
    print()
    
    # Test the specific problem point (0.8, 0.3)
    h_step = np.array([1.0/(nx-1), 1.0/(ny-1)])
    x_test = np.array([0.8, 0.3])
    y_result = np.zeros(1)
    
    evaluate_splines_2d_debug(orders_2d, np.array([nx, ny]), periodic_2d, 
                             x_min, h_step, coeff_2d, x_test, y_result)
    
    expected = test_function(x_test[0], x_test[1])
    error = abs(y_result[0] - expected)
    
    print()
    print(f"Expected: {expected:.6f}")
    print(f"Got:      {y_result[0]:.6f}")
    print(f"Error:    {error:.6f}")
    
    if error > 0.1:
        print("⚠️  Large error detected!")
    else:
        print("✓ Reasonable error")

if __name__ == "__main__":
    test_specific_point()