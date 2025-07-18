#!/usr/bin/env python3
"""
Test periodic splines with truly periodic test functions
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_periodic_cubic():
    """Test periodic cubic with proper periodic sampling"""
    print("PERIODIC CUBIC SPLINE TEST")
    print("=" * 40)
    
    # Use truly periodic sampling: n points from [0, 2π) 
    # This avoids duplicate endpoints
    n = 16
    x = np.linspace(0, 2*np.pi, n, endpoint=False)  # [0, 2π) - no endpoint duplication
    y = np.sin(x)  # Periodic function
    h = 2*np.pi / n
    
    print(f"Domain: [0, 2π) with n={n} points")
    print(f"h = {h:.10f}")
    print(f"Periodic check: y[0]={y[0]:.6f}, y[n-1]={y[n-1]:.6f}, sin(2π)={np.sin(2*np.pi):.6f}")
    
    # Construct periodic cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
    
    # Test evaluation at many points within one period
    test_points = np.linspace(0, 2*np.pi, 100, endpoint=False)
    max_error = 0.0
    errors = []
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = np.sin(x_test)
        error = abs(y_spline - y_exact)
        errors.append(error)
        max_error = max(max_error, error)
    
    print(f"Max error over full period: {max_error:.2e}")
    print(f"RMS error: {np.sqrt(np.mean(np.array(errors)**2)):.2e}")
    
    # Test periodicity by evaluating beyond the domain
    print(f"\nPeriodicity test:")
    test_cases = [
        (0.5, 0.5 + 2*np.pi, "x vs x+2π"),
        (1.0, 1.0 + 2*np.pi, "x vs x+2π"),
        (1.5, 1.5 + 2*np.pi, "x vs x+2π"),
        (-0.5, -0.5 + 2*np.pi, "x vs x+2π"),
    ]
    
    max_periodicity_error = 0.0
    for x1, x2, desc in test_cases:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x1, y_out)
        y1 = y_out[0]
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x2, y_out)
        y2 = y_out[0]
        error = abs(y1 - y2)
        max_periodicity_error = max(max_periodicity_error, error)
        print(f"  {desc}: f({x1:.1f})={y1:.8f}, f({x2:.1f})={y2:.8f}, diff={error:.2e}")
    
    print(f"Max periodicity error: {max_periodicity_error:.2e}")
    
    return max_error, max_periodicity_error

def test_periodic_quartic():
    """Test periodic quartic spline"""
    print(f"\n\nPERIODIC QUARTIC SPLINE TEST")
    print("=" * 40)
    
    # Same setup as cubic
    n = 16
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = np.sin(x)
    h = 2*np.pi / n
    
    print(f"Domain: [0, 2π) with n={n} points")
    print(f"Testing if periodic quartic construction works...")
    
    # Try to construct periodic quartic spline
    try:
        coeff = np.zeros(5*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 4, True, coeff)
        print("✓ Periodic quartic construction succeeded")
        
        # Test evaluation at a few points
        test_points = [0.5, 1.0, 1.5, 2.0]
        max_error = 0.0
        
        y_out = np.zeros(1)
        for x_test in test_points:
            evaluate_splines_1d_cfunc(4, n, True, 0.0, h, coeff, x_test, y_out)
            y_spline = y_out[0]
            y_exact = np.sin(x_test)
            error = abs(y_spline - y_exact)
            max_error = max(max_error, error)
            print(f"  x={x_test:.1f}: spline={y_spline:.6f}, exact={y_exact:.6f}, error={error:.2e}")
        
        print(f"Max error: {max_error:.2e}")
        return max_error
        
    except Exception as e:
        print(f"✗ Periodic quartic construction failed: {e}")
        return float('inf')

def test_different_periodic_functions():
    """Test with different periodic functions"""
    print(f"\n\nTESTING DIFFERENT PERIODIC FUNCTIONS")
    print("=" * 50)
    
    n = 16
    h = 2*np.pi / n
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    test_functions = [
        ("sin(x)", lambda x: np.sin(x)),
        ("cos(x)", lambda x: np.cos(x)),
        ("sin(2x)", lambda x: np.sin(2*x)),
        ("sin(x) + 0.5*cos(2x)", lambda x: np.sin(x) + 0.5*np.cos(2*x)),
    ]
    
    for name, func in test_functions:
        print(f"\nTesting {name}:")
        y = func(x)
        
        # Test cubic
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
        
        # Evaluate at test points
        test_points = np.linspace(0, 2*np.pi, 50, endpoint=False)
        max_error = 0.0
        
        y_out = np.zeros(1)
        for x_test in test_points:
            evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x_test, y_out)
            y_spline = y_out[0]
            y_exact = func(x_test)
            error = abs(y_spline - y_exact)
            max_error = max(max_error, error)
        
        print(f"  Cubic max error: {max_error:.2e}")

def main():
    """Main test routine"""
    cubic_max_error, cubic_periodicity_error = test_periodic_cubic()
    quartic_max_error = test_periodic_quartic()
    test_different_periodic_functions()
    
    print(f"\n" + "=" * 50)
    print("SUMMARY OF PERIODIC TESTS")
    print("=" * 50)
    print(f"Cubic spline max error:        {cubic_max_error:.2e}")
    print(f"Cubic periodicity error:       {cubic_periodicity_error:.2e}")
    print(f"Quartic spline max error:      {quartic_max_error:.2e}")
    
    if cubic_max_error < 1e-10 and cubic_periodicity_error < 1e-14:
        print("✅ Cubic periodic splines achieve excellent precision")
    else:
        print("⚠️  Cubic periodic splines need improvement")
    
    if quartic_max_error < 1e-10:
        print("✅ Quartic periodic splines achieve excellent precision")
    elif quartic_max_error == float('inf'):
        print("❌ Quartic periodic splines not implemented")
    else:
        print("⚠️  Quartic periodic splines need improvement")

if __name__ == "__main__":
    main()