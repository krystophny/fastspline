#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import RectBivariateSpline
import sys
import os

# Add the parent directory to the path so we can import fastspline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastspline.sergei_splines import construct_splines_2d_cfunc, evaluate_splines_2d_cfunc

def test_function(x, y):
    """Test function: sin(π*x) * cos(π*y)"""
    return np.sin(np.pi * x) * np.cos(np.pi * y)

def compare_exact_reproduction():
    """Reproduce the exact conditions from clean_2d_comparison.py"""
    
    # Exact same parameters as clean_2d_comparison.py
    nx, ny = 8, 8
    x_data = np.linspace(0, 1, nx)
    y_data = np.linspace(0, 1, ny)
    X_data, Y_data = np.meshgrid(x_data, y_data)
    Z_data = test_function(X_data, Y_data)
    
    # Evaluation grid
    nx_eval, ny_eval = 41, 41
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    Z_exact = test_function(X_eval, Y_eval)
    
    print("Exact Reproduction of clean_2d_comparison.py")
    print("=" * 50)
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
        
        # Evaluate - using exact same approach as clean_2d_comparison.py
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
    
    if scipy_success and fastspline_success:
        print()
        print("Results (exact reproduction):")
        print("-" * 40)
        
        # SciPy vs exact
        error_scipy = np.abs(Z_scipy - Z_exact)
        rms_scipy = np.sqrt(np.mean(error_scipy**2))
        max_scipy = np.max(error_scipy)
        print(f"SciPy:      RMS={rms_scipy:.2e}, Max={max_scipy:.2e}")
        
        # FastSpline vs exact
        error_fastspline = np.abs(Z_fastspline - Z_exact)
        rms_fastspline = np.sqrt(np.mean(error_fastspline**2))
        max_fastspline = np.max(error_fastspline)
        print(f"FastSpline: RMS={rms_fastspline:.2e}, Max={max_fastspline:.2e}")
        
        # Methods vs each other
        diff_rms = np.sqrt(np.mean((Z_scipy - Z_fastspline)**2))
        diff_max = np.max(np.abs(Z_scipy - Z_fastspline))
        print(f"Difference: RMS={diff_rms:.2e}, Max={diff_max:.2e}")
        
        # Check if this matches clean_2d_comparison.py results
        print()
        print("Comparison with clean_2d_comparison.py results:")
        print("Expected from clean_2d_comparison.py:")
        print("SciPy:      RMS=3.08e-04, Max=9.93e-04")
        print("FastSpline: RMS=7.06e-01, Max=1.00e+00")
        print()
        
        if abs(rms_scipy - 3.08e-04) < 1e-05:
            print("✓ SciPy results match clean_2d_comparison.py")
        else:
            print("⚠️  SciPy results don't match clean_2d_comparison.py")
        
        if abs(rms_fastspline - 7.06e-01) < 1e-02:
            print("✓ FastSpline results match clean_2d_comparison.py")
        else:
            print("⚠️  FastSpline results don't match clean_2d_comparison.py")
        
        # Find the worst error points
        print()
        print("Worst error analysis:")
        worst_idx = np.unravel_index(np.argmax(error_fastspline), error_fastspline.shape)
        worst_j, worst_i = worst_idx
        worst_x = x_eval[worst_i]
        worst_y = y_eval[worst_j]
        
        print(f"Worst FastSpline error at ({worst_x:.3f},{worst_y:.3f}):")
        print(f"  Exact:      {Z_exact[worst_j, worst_i]:.6f}")
        print(f"  FastSpline: {Z_fastspline[worst_j, worst_i]:.6f}")
        print(f"  SciPy:      {Z_scipy[worst_j, worst_i]:.6f}")
        print(f"  FastSpline Error: {error_fastspline[worst_j, worst_i]:.6f}")
        print(f"  SciPy Error:      {error_scipy[worst_j, worst_i]:.6f}")
        
        # Check if the error is consistent with cubic spline theory
        print()
        print("Theoretical analysis:")
        print(f"- For sin(π*x)*cos(π*y), the 4th derivatives are large")
        print(f"- Cubic spline error scales as h^4 * max(|f''''|)")
        print(f"- Grid spacing h = {1.0/(nx-1):.3f}")
        print(f"- So h^4 ≈ {(1.0/(nx-1))**4:.2e}")
        print(f"- The 4th derivative of sin(π*x)*cos(π*y) is π^4*sin(π*x)*cos(π*y)")
        print(f"- Max 4th derivative ≈ π^4 ≈ {np.pi**4:.1f}")
        print(f"- Expected error ≈ {(1.0/(nx-1))**4 * np.pi**4:.2e}")
        
        if rms_fastspline > 0.1:
            print("⚠️  FastSpline error is much higher than theoretical prediction")
            print("   This suggests a potential implementation issue")
        
    else:
        print("Cannot compare - one method failed")

if __name__ == "__main__":
    compare_exact_reproduction()