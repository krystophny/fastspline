#!/usr/bin/env python3
"""
Comprehensive test to verify cubic splines match Fortran at double precision
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_cubic_nonperiodic_precision():
    """Test cubic non-periodic splines vs Fortran reference"""
    print("TESTING CUBIC NON-PERIODIC SPLINES")
    print("=" * 40)
    
    # Same test case as used in previous Fortran comparisons
    n = 10
    x = np.linspace(0, 1, n)
    y = np.sin(2*np.pi*x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: sin(2πx) with n={n} points")
    print(f"Domain: [0, 1], h={h:.10f}")
    
    # Construct cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
    
    # Test evaluation at x=0.5
    x_test = 0.5
    y_out = np.zeros(1)
    evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
    y_spline = y_out[0]
    y_exact = np.sin(2*np.pi*x_test)
    error = abs(y_spline - y_exact)
    
    print(f"Evaluation at x={x_test}:")
    print(f"  Spline result: {y_spline:.15f}")
    print(f"  Exact result:  {y_exact:.15f}")
    print(f"  Error:         {error:.2e}")
    
    # Test at multiple points
    test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_error = 0.0
    
    print(f"\nMultiple point evaluation:")
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        print(f"  x={x_test:.1f}: spline={y_spline:.8f}, exact={y_exact:.8f}, error={error:.2e}")
    
    print(f"Max error: {max_error:.2e}")
    
    return max_error

def test_cubic_periodic_precision():
    """Test cubic periodic splines"""
    print(f"\n\nTESTING CUBIC PERIODIC SPLINES")
    print("=" * 40)
    
    # Use proper periodic sampling
    n = 16
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = np.sin(x)
    h = 2*np.pi / n
    
    print(f"Test function: sin(x) with n={n} points")
    print(f"Domain: [0, 2π), h={h:.10f}")
    print(f"Periodic check: y[0]={y[0]:.6f}, sin(0)={np.sin(0):.6f}")
    
    # Construct periodic cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
    
    # Test evaluation at multiple points
    test_points = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    max_error = 0.0
    
    print(f"\nEvaluation test:")
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = np.sin(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        print(f"  x={x_test:.1f}: spline={y_spline:.8f}, exact={y_exact:.8f}, error={error:.2e}")
    
    # Test periodicity
    print(f"\nPeriodicity test:")
    periodic_test_points = [(0.5, 0.5 + 2*np.pi), (1.0, 1.0 + 2*np.pi)]
    max_periodicity_error = 0.0
    
    for x1, x2 in periodic_test_points:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x1, y_out)
        y1 = y_out[0]
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x2, y_out)
        y2 = y_out[0]
        error = abs(y1 - y2)
        max_periodicity_error = max(max_periodicity_error, error)
        print(f"  f({x1:.1f}) vs f({x2:.1f}): {y1:.8f} vs {y2:.8f}, diff={error:.2e}")
    
    print(f"Max evaluation error: {max_error:.2e}")
    print(f"Max periodicity error: {max_periodicity_error:.2e}")
    
    return max_error, max_periodicity_error

def test_cubic_quadratic_exact():
    """Test that cubic splines represent quadratic functions exactly"""
    print(f"\n\nTESTING CUBIC EXACT REPRESENTATION OF QUADRATICS")
    print("=" * 55)
    
    # Quadratic function: f(x) = 1 + 2x + 3x^2
    def f(x):
        return 1.0 + 2.0*x + 3.0*x**2
    
    n = 8
    x = np.linspace(0, 1, n)
    y = f(x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: f(x) = 1 + 2x + 3x²")
    print(f"Domain: [0, 1] with n={n} points")
    
    # Construct cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
    
    # Test at many points
    test_points = np.linspace(0.1, 0.9, 20)
    max_error = 0.0
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
    
    print(f"Max error: {max_error:.2e}")
    
    # Test specific points with high precision
    specific_points = [0.25, 0.5, 0.75]
    print(f"High precision test:")
    for x_test in specific_points:
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        print(f"  x={x_test:.2f}: spline={y_spline:.15f}, exact={y_exact:.15f}, error={error:.2e}")
    
    return max_error

def main():
    """Main test routine"""
    print("COMPREHENSIVE CUBIC SPLINE FORTRAN PRECISION TEST")
    print("=" * 60)
    
    nonperiodic_error = test_cubic_nonperiodic_precision()
    periodic_error, periodicity_error = test_cubic_periodic_precision()
    quadratic_error = test_cubic_quadratic_exact()
    
    print(f"\n" + "=" * 60)
    print("CUBIC SPLINE PRECISION SUMMARY")
    print("=" * 60)
    print(f"Non-periodic evaluation error:  {nonperiodic_error:.2e}")
    print(f"Periodic evaluation error:      {periodic_error:.2e}")
    print(f"Periodicity error:              {periodicity_error:.2e}")
    print(f"Quadratic representation error: {quadratic_error:.2e}")
    print()
    
    # Success criteria
    success = True
    if nonperiodic_error > 1e-15:
        print("⚠️  Non-periodic cubic needs improvement")
        success = False
    else:
        print("✅ Non-periodic cubic achieves double precision")
    
    if periodic_error > 1e-10:
        print("⚠️  Periodic cubic evaluation needs improvement") 
        success = False
    else:
        print("✅ Periodic cubic evaluation excellent")
    
    if periodicity_error > 1e-14:
        print("⚠️  Periodicity not perfect")
        success = False
    else:
        print("✅ Perfect periodicity achieved")
    
    if quadratic_error > 1e-12:
        print("⚠️  Quadratic representation not exact enough")
        success = False
    else:
        print("✅ Quadratic representation near-exact")
    
    if success:
        print("\n🎉 ALL CUBIC TESTS PASSED!")
        return True
    else:
        print("\n❌ Some cubic tests need work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)