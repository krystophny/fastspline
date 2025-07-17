#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import sys
import os

# Add the parent directory to the path so we can import fastspline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastspline.sergei_splines import construct_splines_2d_cfunc, evaluate_splines_2d_cfunc

def test_function(x, y):
    """Test function: sin(π*x) * cos(π*y)"""
    return np.sin(np.pi * x) * np.cos(np.pi * y)

def diagnose_2d_error():
    """Diagnose the 2D error issue"""
    
    # Data grid
    nx, ny = 8, 8
    x_data = np.linspace(0, 1, nx)
    y_data = np.linspace(0, 1, ny)
    X_data, Y_data = np.meshgrid(x_data, y_data)
    Z_data = test_function(X_data, Y_data)
    
    # Evaluation grid
    nx_eval, ny_eval = 21, 21  # Smaller first
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    Z_exact = test_function(X_eval, Y_eval)
    
    print("2D Error Diagnosis")
    print("=" * 40)
    print(f"Data grid: {nx}×{ny}")
    print(f"Evaluation grid: {nx_eval}×{ny_eval}")
    print(f"Function: sin(π*x) * cos(π*y)")
    print()
    
    # SciPy implementation
    try:
        scipy_spline = RectBivariateSpline(x_data, y_data, Z_data.T, kx=3, ky=3, s=0)
        Z_scipy = scipy_spline(x_eval, y_eval).T
        scipy_success = True
        print("✓ SciPy RectBivariateSpline - Success")
    except Exception as e:
        print(f"✗ SciPy failed: {e}")
        scipy_success = False
        Z_scipy = np.zeros_like(Z_exact)
    
    # FastSpline implementation
    try:
        order = 3
        coeff_2d = np.zeros((order+1)**2 * nx * ny)
        x_min = np.array([0.0, 0.0])
        x_max = np.array([1.0, 1.0])
        orders_2d = np.array([order, order])
        periodic_2d = np.array([False, False])
        
        # Construct
        z_flat = Z_data.flatten()
        workspace_y = np.zeros(nx * ny)
        workspace_coeff = np.zeros((order+1) * nx * ny)
        
        construct_splines_2d_cfunc(x_min, x_max, z_flat, 
                                  np.array([nx, ny]), orders_2d, periodic_2d, 
                                  coeff_2d, workspace_y, workspace_coeff)
        
        # Evaluate
        Z_fastspline = np.zeros_like(Z_exact)
        h_step = np.array([1.0/(nx-1), 1.0/(ny-1)])
        
        for i in range(nx_eval):
            for j in range(ny_eval):
                x_eval_point = np.array([x_eval[i], y_eval[j]])
                z_val = np.zeros(1)
                evaluate_splines_2d_cfunc(orders_2d, np.array([nx, ny]), periodic_2d, 
                                         x_min, h_step, coeff_2d, x_eval_point, z_val)
                Z_fastspline[j, i] = z_val[0]
        
        fastspline_success = True
        print("✓ FastSpline Sergei 2D - Success")
        
    except Exception as e:
        print(f"✗ FastSpline failed: {e}")
        fastspline_success = False
        Z_fastspline = np.zeros_like(Z_exact)
    
    # Compare results
    if scipy_success and fastspline_success:
        print()
        print("Error Analysis:")
        print("-" * 30)
        
        # SciPy vs exact
        error_scipy = np.abs(Z_scipy - Z_exact)
        rms_scipy = np.sqrt(np.mean(error_scipy**2))
        max_scipy = np.max(error_scipy)
        print(f"SciPy vs Exact:      RMS={rms_scipy:.2e}, Max={max_scipy:.2e}")
        
        # FastSpline vs exact
        error_fastspline = np.abs(Z_fastspline - Z_exact)
        rms_fastspline = np.sqrt(np.mean(error_fastspline**2))
        max_fastspline = np.max(error_fastspline)
        print(f"FastSpline vs Exact: RMS={rms_fastspline:.2e}, Max={max_fastspline:.2e}")
        
        # Methods vs each other
        error_methods = np.abs(Z_scipy - Z_fastspline)
        rms_methods = np.sqrt(np.mean(error_methods**2))
        max_methods = np.max(error_methods)
        print(f"Methods vs Each:     RMS={rms_methods:.2e}, Max={max_methods:.2e}")
        
        # Check specific points
        print()
        print("Specific Point Analysis:")
        print("-" * 30)
        
        test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
        for x_test, y_test in test_points:
            # Find closest indices
            i_closest = np.argmin(np.abs(x_eval - x_test))
            j_closest = np.argmin(np.abs(y_eval - y_test))
            
            exact_val = test_function(x_test, y_test)
            scipy_val = Z_scipy[j_closest, i_closest]
            fast_val = Z_fastspline[j_closest, i_closest]
            
            print(f"Point ({x_test:.3f},{y_test:.3f}):")
            print(f"  Exact:     {exact_val:.6f}")
            print(f"  SciPy:     {scipy_val:.6f} (error: {abs(scipy_val - exact_val):.2e})")
            print(f"  FastSpline: {fast_val:.6f} (error: {abs(fast_val - exact_val):.2e})")
            print(f"  Difference: {abs(scipy_val - fast_val):.2e}")
        
        # Check meshgrid indexing
        print()
        print("Meshgrid Indexing Check:")
        print("-" * 30)
        print(f"X_eval[0,0] = {X_eval[0,0]:.6f}, Y_eval[0,0] = {Y_eval[0,0]:.6f}")
        print(f"X_eval[0,1] = {X_eval[0,1]:.6f}, Y_eval[0,1] = {Y_eval[0,1]:.6f}")
        print(f"X_eval[1,0] = {X_eval[1,0]:.6f}, Y_eval[1,0] = {Y_eval[1,0]:.6f}")
        print(f"Z_exact[0,0] = {Z_exact[0,0]:.6f}")
        print(f"Z_scipy[0,0] = {Z_scipy[0,0]:.6f}")
        print(f"Z_fastspline[0,0] = {Z_fastspline[0,0]:.6f}")
        
        # Look at worst errors
        print()
        print("Worst Error Analysis:")
        print("-" * 30)
        
        worst_idx = np.unravel_index(np.argmax(error_fastspline), error_fastspline.shape)
        worst_j, worst_i = worst_idx
        worst_x = x_eval[worst_i]
        worst_y = y_eval[worst_j]
        
        print(f"Worst FastSpline error at ({worst_x:.3f},{worst_y:.3f}):")
        print(f"  Exact:      {Z_exact[worst_j, worst_i]:.6f}")
        print(f"  FastSpline: {Z_fastspline[worst_j, worst_i]:.6f}")
        print(f"  Error:      {error_fastspline[worst_j, worst_i]:.6f}")
        
        # Check if it's near boundaries
        print(f"  Grid position: i={worst_i}/{nx_eval-1}, j={worst_j}/{ny_eval-1}")
        print(f"  Near boundary: x={worst_x < 0.1 or worst_x > 0.9}, y={worst_y < 0.1 or worst_y > 0.9}")
        
    else:
        print("Cannot compare - one method failed")

if __name__ == "__main__":
    diagnose_2d_error()