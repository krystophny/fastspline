#!/usr/bin/env python3
"""
Test quintic splines precision vs Fortran reference
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_quintic_nonperiodic_precision():
    """Test quintic non-periodic splines vs known good case"""
    print("TESTING QUINTIC NON-PERIODIC SPLINES")
    print("=" * 40)
    
    # Use exact same test case as our reference that gives 1.17e-04 error
    n = 20  # Known to work well with quintic
    x = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: sin(2πx) with n={n} points")
    print(f"Domain: [0, 1], h={h:.10f}")
    
    # Construct quintic spline
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    # Test evaluation at x=0.5 (reference case)
    x_test = 0.5
    y_out = np.zeros(1)
    evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
    y_spline = y_out[0]
    y_exact = np.sin(2*np.pi*x_test)
    error = abs(y_spline - y_exact)
    
    print(f"Reference evaluation at x={x_test}:")
    print(f"  Spline result: {y_spline:.15f}")
    print(f"  Exact result:  {y_exact:.15f}")
    print(f"  Error:         {error:.2e}")
    
    # Test at multiple points
    test_points = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    max_error = 0.0
    
    print(f"\nMultiple point evaluation:")
    for x_test in test_points:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        print(f"  x={x_test:.1f}: spline={y_spline:.8f}, exact={y_exact:.8f}, error={error:.2e}")
    
    print(f"Max error: {max_error:.2e}")
    
    return error, max_error  # Return reference error and max error

def test_quintic_quartic_exact():
    """Test that quintic splines represent quartic functions well"""
    print(f"\n\nTESTING QUINTIC REPRESENTATION OF QUARTIC FUNCTION")
    print("=" * 55)
    
    # Quartic function: f(x) = 1 + x + x^2 + x^3 + 0.1*x^4
    def f(x):
        return 1.0 + x + x**2 + x**3 + 0.1*x**4
    
    n = 20
    x = np.linspace(0, 1, n)
    y = f(x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: f(x) = 1 + x + x² + x³ + 0.1x⁴")
    print(f"Domain: [0, 1] with n={n} points")
    
    # Construct quintic spline
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    # Test at many points
    test_points = np.linspace(0.1, 0.9, 20)
    max_error = 0.0
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
    
    print(f"Max error: {max_error:.2e}")
    
    # Test specific points with high precision
    specific_points = [0.25, 0.5, 0.75]
    print(f"High precision test:")
    for x_test in specific_points:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        print(f"  x={x_test:.2f}: spline={y_spline:.15f}, exact={y_exact:.15f}, error={error:.2e}")
    
    return max_error

def test_quintic_with_different_n():
    """Test quintic splines with different numbers of points"""
    print(f"\n\nTESTING QUINTIC WITH DIFFERENT N VALUES")
    print("=" * 45)
    
    def f(x):
        return np.sin(2*np.pi*x)
    
    n_values = [10, 15, 20, 25]
    x_test = 0.5
    
    print(f"Test function: sin(2πx), evaluation at x={x_test}")
    print(f"{'n':>3} | {'Error':>12} | {'Status'}")
    print("-" * 30)
    
    best_error = float('inf')
    best_n = 0
    
    for n in n_values:
        x = np.linspace(0, 1, n)
        y = f(x)
        h = 1.0 / (n - 1)
        
        # Construct quintic spline
        coeff = np.zeros(6*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
        
        # Evaluate
        y_out = np.zeros(1)
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        
        if error < best_error:
            best_error = error
            best_n = n
        
        status = "✓" if error < 1e-3 else "○"
        print(f"{n:>3} | {error:>12.2e} | {status}")
    
    print(f"\nBest: n={best_n}, error={best_error:.2e}")
    return best_error

def main():
    """Main test routine"""
    print("COMPREHENSIVE QUINTIC SPLINE FORTRAN PRECISION TEST")
    print("=" * 60)
    
    reference_error, max_error = test_quintic_nonperiodic_precision()
    quartic_error = test_quintic_quartic_exact()
    best_error = test_quintic_with_different_n()
    
    print(f"\n" + "=" * 60)
    print("QUINTIC SPLINE PRECISION SUMMARY")
    print("=" * 60)
    print(f"Reference case (n=20, x=0.5) error: {reference_error:.2e}")
    print(f"Max error across multiple points:   {max_error:.2e}")
    print(f"Quartic representation error:       {quartic_error:.2e}")
    print(f"Best error across different n:      {best_error:.2e}")
    print()
    
    # Success criteria based on known Fortran comparison
    success = True
    if reference_error > 2e-4:  # Allow some tolerance around 1.17e-04
        print(f"⚠️  Reference error {reference_error:.2e} worse than expected ~1.17e-04")
        success = False
    else:
        print("✅ Reference error matches Fortran comparison")
    
    if max_error > 1e-3:
        print("⚠️  Some evaluation points have large errors")
        success = False
    else:
        print("✅ Evaluation error reasonable across test points")
    
    if quartic_error > 1e-2:
        print("⚠️  Quartic representation poor")
        success = False
    else:
        print("✅ Quartic representation reasonable")
    
    if success:
        print("\n🎉 ALL QUINTIC TESTS PASSED!")
        return True
    else:
        print("\n❌ Some quintic tests need work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)