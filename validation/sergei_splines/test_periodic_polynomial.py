#!/usr/bin/env python3
"""
Test periodic splines with polynomial functions that should be exactly representable
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def create_periodic_polynomial():
    """Create a polynomial that is exactly periodic over [0, 2π]"""
    # For a function to be periodic over [0, 2π], we need f(0) = f(2π)
    # A simple periodic polynomial is a combination of Fourier basis functions
    # Let's use: f(x) = a + b*cos(x) + c*sin(x) + d*cos(2x) + e*sin(2x)
    # This is exactly periodic and should be well-approximated by splines
    
    def f(x):
        return 1.0 + 0.5*np.cos(x) + 0.3*np.sin(x) + 0.2*np.cos(2*x)
    
    return f

def test_cubic_with_periodic_polynomial():
    """Test cubic splines with periodic polynomial"""
    print("CUBIC SPLINES WITH PERIODIC POLYNOMIAL")
    print("=" * 50)
    
    # Use more points for better approximation
    n = 32
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    f = create_periodic_polynomial()
    y = f(x)
    h = 2*np.pi / n
    
    print(f"Test function: f(x) = 1 + 0.5*cos(x) + 0.3*sin(x) + 0.2*cos(2x)")
    print(f"Domain: [0, 2π) with n={n} points")
    print(f"Periodicity check: f(0)={f(0):.8f}, f(2π)={f(2*np.pi):.8f}")
    
    # Construct periodic cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
    
    # Test evaluation at many points
    test_points = np.linspace(0, 2*np.pi, 200, endpoint=False)
    max_error = 0.0
    errors = []
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        errors.append(error)
        max_error = max(max_error, error)
    
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    
    print(f"Max error: {max_error:.2e}")
    print(f"RMS error: {rms_error:.2e}")
    
    # Test specific points
    print(f"\nDetailed evaluation test:")
    test_specific = [0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
    for x_test in test_specific:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        print(f"  x={x_test:.3f}: spline={y_spline:.8f}, exact={y_exact:.8f}, error={error:.2e}")
    
    return max_error, rms_error

def test_quartic_with_periodic_polynomial():
    """Test quartic splines with periodic polynomial"""
    print(f"\n\nQUARTIC SPLINES WITH PERIODIC POLYNOMIAL")
    print("=" * 50)
    
    n = 32
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    f = create_periodic_polynomial()
    y = f(x)
    h = 2*np.pi / n
    
    print(f"Test function: f(x) = 1 + 0.5*cos(x) + 0.3*sin(x) + 0.2*cos(2x)")
    print(f"Domain: [0, 2π) with n={n} points")
    
    try:
        # Construct periodic quartic spline
        coeff = np.zeros(5*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 4, True, coeff)
        print("✓ Periodic quartic construction succeeded")
        
        # Test evaluation
        test_points = np.linspace(0, 2*np.pi, 200, endpoint=False)
        max_error = 0.0
        errors = []
        
        y_out = np.zeros(1)
        for x_test in test_points:
            evaluate_splines_1d_cfunc(4, n, True, 0.0, h, coeff, x_test, y_out)
            y_spline = y_out[0]
            y_exact = f(x_test)
            error = abs(y_spline - y_exact)
            errors.append(error)
            max_error = max(max_error, error)
        
        rms_error = np.sqrt(np.mean(np.array(errors)**2))
        
        print(f"Max error: {max_error:.2e}")
        print(f"RMS error: {rms_error:.2e}")
        
        return max_error, rms_error
        
    except Exception as e:
        print(f"✗ Periodic quartic failed: {e}")
        return float('inf'), float('inf')

def test_simple_periodic_cubic():
    """Test with a cubic polynomial that should be exactly representable"""
    print(f"\n\nCUBIC POLYNOMIAL TEST (SHOULD BE EXACT)")
    print("=" * 50)
    
    # A cubic function that's periodic: f(x) = 0.1*(x-π)^3 adjusted to be periodic
    # Actually, let's use a simpler approach: a cosine which is smooth
    def f(x):
        return np.cos(x)  # This should be very well approximated
    
    n = 16
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = f(x)
    h = 2*np.pi / n
    
    print(f"Test function: f(x) = cos(x)")
    print(f"Domain: [0, 2π) with n={n} points")
    
    # Construct periodic cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
    
    # Test evaluation at many points
    test_points = np.linspace(0, 2*np.pi, 100, endpoint=False)
    max_error = 0.0
    
    y_out = np.zeros(1)
    for x_test in test_points:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = f(x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
    
    print(f"Max error: {max_error:.2e}")
    
    # Test periodicity
    print(f"\nPeriodicity test:")
    test_pairs = [(0.5, 0.5 + 2*np.pi), (1.0, 1.0 + 2*np.pi)]
    max_periodicity_error = 0.0
    
    for x1, x2 in test_pairs:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x1, y_out)
        y1 = y_out[0]
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x2, y_out)
        y2 = y_out[0]
        error = abs(y1 - y2)
        max_periodicity_error = max(max_periodicity_error, error)
        print(f"  f({x1:.1f}) vs f({x2:.1f}): {y1:.8f} vs {y2:.8f}, diff={error:.2e}")
    
    print(f"Max periodicity error: {max_periodicity_error:.2e}")
    
    return max_error, max_periodicity_error

def main():
    """Main test routine"""
    cubic_max, cubic_rms = test_cubic_with_periodic_polynomial()
    quartic_max, quartic_rms = test_quartic_with_periodic_polynomial()
    simple_max, simple_periodicity = test_simple_periodic_cubic()
    
    print(f"\n" + "=" * 60)
    print("SUMMARY OF PERIODIC POLYNOMIAL TESTS")
    print("=" * 60)
    print(f"Cubic (complex periodic function):")
    print(f"  Max error: {cubic_max:.2e}, RMS error: {cubic_rms:.2e}")
    print(f"Quartic (complex periodic function):")
    print(f"  Max error: {quartic_max:.2e}, RMS error: {quartic_rms:.2e}")
    print(f"Cubic (simple cosine):")
    print(f"  Max error: {simple_max:.2e}, Periodicity: {simple_periodicity:.2e}")
    
    if simple_periodicity < 1e-14:
        print("✅ Perfect periodicity achieved")
    else:
        print("⚠️  Periodicity needs improvement")
    
    if simple_max < 1e-6:
        print("✅ Excellent approximation quality")
    else:
        print("⚠️  Approximation quality could be better")

if __name__ == "__main__":
    main()