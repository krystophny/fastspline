#!/usr/bin/env python3
"""
Test the fixed quintic spline implementation
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_quintic_splines():
    """Test quintic splines with various n values"""
    print("Testing Fixed Quintic Spline Implementation")
    print("=" * 50)
    
    # Test different grid sizes
    test_sizes = [8, 10, 12, 16, 20]
    
    for n in test_sizes:
        print(f"\nTesting n = {n}")
        print("-" * 30)
        
        try:
            # Create test data
            x_min, x_max = 0.0, 1.0
            x = np.linspace(x_min, x_max, n)
            y = np.sin(2*np.pi*x)  # Simple periodic function
            
            # Allocate coefficient array for quintic (6 coefficient sets)
            coeff = np.zeros(6*n, dtype=np.float64)
            
            # Copy y values to a coefficients
            coeff[:n] = y
            
            # Construct spline  
            construct_splines_1d_cfunc(x_min, x_max, y, n, 5, False, coeff)
            
            print(f"✓ Construction succeeded")
            
            # Test evaluation at several points
            test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
            max_error = 0.0
            
            for x_test in test_points:
                # Evaluate spline
                h_step = (x_max - x_min) / (n - 1)
                y_out = np.zeros(1, dtype=np.float64)
                evaluate_splines_1d_cfunc(5, n, False, x_min, h_step, coeff, x_test, y_out)
                y_spline = y_out[0]
                
                # Exact value
                y_exact = np.sin(2*np.pi*x_test)
                error = abs(y_spline - y_exact)
                max_error = max(max_error, error)
                
                print(f"  x = {x_test:.1f}: spline = {y_spline:.6f}, exact = {y_exact:.6f}, error = {error:.2e}")
            
            print(f"  Max error: {max_error:.2e}")
            
            if max_error < 1e-2:  # Reasonable tolerance for quintic
                print(f"  ✓ SUCCESS: Quintic spline works for n={n}")
            else:
                print(f"  ⚠ WARNING: Large errors for n={n}")
                
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_quintic_splines()