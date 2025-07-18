#!/usr/bin/env python3
"""
Test splines with functions they can represent exactly (within numerical precision)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_cubic_with_quadratic():
    """Cubic splines should represent quadratic functions exactly"""
    print("CUBIC SPLINES WITH QUADRATIC FUNCTION (SHOULD BE EXACT)")
    print("=" * 60)
    
    # Quadratic function: f(x) = 2 + 3x + x^2
    def f(x):
        return 2.0 + 3.0*x + x**2
    
    n = 10
    x = np.linspace(0, 1, n)
    y = f(x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: f(x) = 2 + 3x + x²")
    print(f"Domain: [0, 1] with n={n} points")
    
    # Construct cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
    
    # Test at many points
    test_points = np.linspace(0.1, 0.9, 50)
    max_error = 0.0
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
    
    print(f"Max error: {max_error:.2e}")
    
    # Test specific points
    print("Detailed test:")
    specific_points = [0.25, 0.5, 0.75]
    for x_test in specific_points:
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        print(f"  x={x_test:.2f}: spline={y_spline:.15f}, exact={y_exact:.15f}, error={error:.2e}")
    
    return max_error

def test_quartic_with_cubic():
    """Quartic splines should represent cubic functions exactly"""
    print(f"\n\nQUARTIC SPLINES WITH CUBIC FUNCTION (SHOULD BE EXACT)")
    print("=" * 60)
    
    # Cubic function: f(x) = 1 + 2x + 3x^2 + x^3
    def f(x):
        return 1.0 + 2.0*x + 3.0*x**2 + x**3
    
    n = 10
    x = np.linspace(0, 1, n)
    y = f(x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: f(x) = 1 + 2x + 3x² + x³")
    print(f"Domain: [0, 1] with n={n} points")
    
    # Construct quartic spline
    coeff = np.zeros(5*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 4, False, coeff)
    
    # Test at many points
    test_points = np.linspace(0.1, 0.9, 50)
    max_error = 0.0
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(4, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
    
    print(f"Max error: {max_error:.2e}")
    
    # Test specific points with full precision
    print("Detailed test:")
    specific_points = [0.25, 0.5, 0.75]
    for x_test in specific_points:
        evaluate_splines_1d_cfunc(4, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        print(f"  x={x_test:.2f}: spline={y_spline:.15f}, exact={y_exact:.15f}, error={error:.2e}")
    
    return max_error

def test_quintic_with_quartic():
    """Quintic splines should represent quartic functions exactly"""
    print(f"\n\nQUINTIC SPLINES WITH QUARTIC FUNCTION (SHOULD BE EXACT)")
    print("=" * 60)
    
    # Quartic function: f(x) = 1 + x + x^2 + x^3 + 0.1*x^4
    def f(x):
        return 1.0 + x + x**2 + x**3 + 0.1*x**4
    
    n = 20  # Use more points for quintic as it was noted to work better with n=20
    x = np.linspace(0, 1, n)
    y = f(x)
    h = 1.0 / (n - 1)
    
    print(f"Test function: f(x) = 1 + x + x² + x³ + 0.1x⁴")
    print(f"Domain: [0, 1] with n={n} points")
    
    # Construct quintic spline
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    # Test at many points
    test_points = np.linspace(0.1, 0.9, 50)
    max_error = 0.0
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
    
    print(f"Max error: {max_error:.2e}")
    
    # Test specific points
    print("Detailed test:")
    specific_points = [0.25, 0.5, 0.75]
    for x_test in specific_points:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        print(f"  x={x_test:.2f}: spline={y_spline:.15f}, exact={y_exact:.15f}, error={error:.2e}")
    
    return max_error

def main():
    """Main test routine"""
    cubic_error = test_cubic_with_quadratic()
    quartic_error = test_quartic_with_cubic()
    quintic_error = test_quintic_with_quartic()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY OF EXACT REPRESENTATION TESTS")
    print("=" * 60)
    print(f"Cubic splines with quadratic function:  {cubic_error:.2e}")
    print(f"Quartic splines with cubic function:   {quartic_error:.2e}")
    print(f"Quintic splines with quartic function: {quintic_error:.2e}")
    print()
    
    if cubic_error < 1e-14:
        print("✅ Cubic splines achieve double precision for quadratic functions")
    else:
        print("⚠️  Cubic splines not achieving double precision")
    
    if quartic_error < 1e-14:
        print("✅ Quartic splines achieve double precision for cubic functions")
    else:
        print("⚠️  Quartic splines not achieving double precision")
    
    if quintic_error < 1e-14:
        print("✅ Quintic splines achieve double precision for quartic functions")
    else:
        print("⚠️  Quintic splines not achieving double precision")

if __name__ == "__main__":
    main()