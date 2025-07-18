#!/usr/bin/env python3
"""
Direct comparison of current Python implementation with Fortran
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_current_python_vs_fortran():
    """Test current Python implementation vs Fortran results"""
    print("Comparing Current Python vs Fortran Periodic Implementation")
    print("=" * 65)
    
    # Same test as Fortran: sin(2πx) on [0,1] with n=16
    n = 16
    x_min, x_max = 0.0, 1.0
    h = (x_max - x_min) / n  # Periodic spacing
    
    # Test data
    x = np.linspace(x_min, x_max, n, endpoint=False)
    y = np.sin(2*np.pi*x)
    
    print(f"Test setup: n={n}, h={h:.10f}")
    print(f"Domain: [{x_min}, {x_max}) (periodic)")
    print(f"Function: sin(2πx)")
    
    # Test current Python implementation
    print("\nTesting Current Python Implementation:")
    print("-" * 40)
    
    try:
        # Allocate coefficient array for cubic (4 sets)
        coeff = np.zeros(4*n, dtype=np.float64)
        
        # Construct splines using current implementation
        construct_splines_1d_cfunc(x_min, x_max, y, n, 3, True, coeff)  # periodic=True
        
        print("✓ Construction succeeded")
        
        # Test evaluation at key points
        test_points = [0.25, 0.5, 0.75, 0.95]
        expected_values = [1.0, 0.0, -1.0, -0.3090170]
        
        print("\nEvaluation tests:")
        for i, x_test in enumerate(test_points):
            y_out = np.zeros(1, dtype=np.float64)
            evaluate_splines_1d_cfunc(3, n, True, x_min, h, coeff, x_test, y_out)
            y_spline = y_out[0]
            y_exact = expected_values[i]
            error = abs(y_spline - y_exact)
            
            print(f"  x = {x_test:.3f}: spline = {y_spline:.8f}, exact = {y_exact:.8f}, error = {error:.2e}")
        
        # Test continuity at boundary
        print("\nBoundary continuity test:")
        
        # Evaluate at x = 0.0 (should be same as y[0])
        y_out_0 = np.zeros(1, dtype=np.float64)
        evaluate_splines_1d_cfunc(3, n, True, x_min, h, coeff, 0.0, y_out_0)
        f_0 = y_out_0[0]
        
        # Evaluate at x = 1.0 - eps (should approach same value)
        eps = 1e-10
        y_out_1 = np.zeros(1, dtype=np.float64)
        evaluate_splines_1d_cfunc(3, n, True, x_min, h, coeff, 1.0 - eps, y_out_1)
        f_1_minus = y_out_1[0]
        
        continuity_error = abs(f_0 - f_1_minus)
        print(f"  f(0) = {f_0:.12f}")
        print(f"  f(1-ε) = {f_1_minus:.12f}")
        print(f"  Continuity error = {continuity_error:.2e}")
        
        # Determine if we have floating point precision
        if continuity_error < 1e-14:
            print("\n✅ SUCCESS: Floating point precision achieved!")
            return True
        elif continuity_error < 1e-10:
            print("\n⚠️  CLOSE: Near floating point precision")
            return False
        else:
            print("\n❌ FAILURE: Poor continuity")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_coefficient_structure():
    """Analyze the coefficient structure from current implementation"""
    print("\n\nAnalyzing Coefficient Structure:")
    print("=" * 40)
    
    # Simple test case
    n = 4
    x_min, x_max = 0.0, 1.0
    h = (x_max - x_min) / n
    
    # Constant function - should give zero coefficients for b, c, d
    y = np.ones(n)
    coeff = np.zeros(4*n, dtype=np.float64)
    
    print(f"Test case: constant function (y=1) with n={n}")
    
    try:
        construct_splines_1d_cfunc(x_min, x_max, y, n, 3, True, coeff)
        
        print("Coefficients:")
        for i in range(n):
            a_i = coeff[i]
            b_i = coeff[n + i]
            c_i = coeff[2*n + i]
            d_i = coeff[3*n + i]
            print(f"  i={i}: a={a_i:.6f}, b={b_i:.6f}, c={c_i:.6f}, d={d_i:.6f}")
        
        # For constant function, b, c, d should all be zero
        b_max = np.max(np.abs(coeff[n:2*n]))
        c_max = np.max(np.abs(coeff[2*n:3*n]))
        d_max = np.max(np.abs(coeff[3*n:4*n]))
        
        print(f"\nMax coefficients: |b|={b_max:.2e}, |c|={c_max:.2e}, |d|={d_max:.2e}")
        
        if b_max < 1e-14 and c_max < 1e-14 and d_max < 1e-14:
            print("✅ Constant function test passed")
        else:
            print("❌ Constant function test failed")
            
    except Exception as e:
        print(f"✗ ERROR: {e}")

if __name__ == "__main__":
    success = test_current_python_vs_fortran()
    analyze_coefficient_structure()
    
    if success:
        print("\n" + "="*65)
        print("🎯 CONCLUSION: Current implementation achieves floating point precision!")
    else:
        print("\n" + "="*65)
        print("🔧 CONCLUSION: Implementation needs refinement for perfect precision")