#!/usr/bin/env python3
"""
Compare Sergei splines with SciPy splines on realistic test cases
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

try:
    from scipy.interpolate import splrep, splev, CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: SciPy not available, skipping SciPy comparisons")

def test_cubic_vs_scipy_smooth_function():
    """Compare cubic splines on smooth function"""
    if not HAS_SCIPY:
        return None, None
    
    print("COMPARING CUBIC SPLINES WITH SCIPY (SMOOTH FUNCTION)")
    print("=" * 55)
    
    # Test function: exponential decay with oscillation
    def f(x):
        return np.exp(-x) * np.cos(4*x) + 0.1*x
    
    n = 20
    x_data = np.linspace(0, 2, n)
    y_data = f(x_data)
    h = 2.0 / (n - 1)
    
    print(f"Test function: exp(-x)*cos(4x) + 0.1x")
    print(f"Domain: [0, 2] with n={n} points")
    
    # Sergei cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2.0, y_data, n, 3, False, coeff)
    
    # SciPy cubic spline
    scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
    
    # Test evaluation at many points
    x_test = np.linspace(0.1, 1.9, 50)
    sergei_errors = []
    scipy_errors = []
    
    y_out = np.zeros(1)
    for x in x_test:
        # Sergei evaluation
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x, y_out)
        y_sergei = y_out[0]
        
        # SciPy evaluation
        y_scipy = scipy_spline(x)
        
        # Exact value
        y_exact = f(x)
        
        sergei_error = abs(y_sergei - y_exact)
        scipy_error = abs(y_scipy - y_exact)
        
        sergei_errors.append(sergei_error)
        scipy_errors.append(scipy_error)
    
    sergei_max = np.max(sergei_errors)
    scipy_max = np.max(scipy_errors)
    sergei_rms = np.sqrt(np.mean(np.array(sergei_errors)**2))
    scipy_rms = np.sqrt(np.mean(np.array(scipy_errors)**2))
    
    print(f"Results:")
    print(f"  Sergei cubic - Max error: {sergei_max:.2e}, RMS error: {sergei_rms:.2e}")
    print(f"  SciPy cubic  - Max error: {scipy_max:.2e}, RMS error: {scipy_rms:.2e}")
    print(f"  Ratio (Sergei/SciPy) - Max: {sergei_max/scipy_max:.2f}, RMS: {sergei_rms/scipy_rms:.2f}")
    
    return sergei_max, scipy_max

def test_cubic_vs_scipy_noisy_data():
    """Compare cubic splines on noisy data"""
    if not HAS_SCIPY:
        return None, None
    
    print(f"\n\nCOMPARING CUBIC SPLINES WITH SCIPY (NOISY DATA)")
    print("=" * 50)
    
    # Test function: polynomial with added noise
    np.random.seed(42)  # Reproducible results
    def f(x):
        return 2*x**3 - 3*x**2 + x + 1
    
    n = 15
    x_data = np.linspace(0, 2, n)
    y_clean = f(x_data)
    noise = 0.05 * np.random.randn(n)
    y_data = y_clean + noise
    h = 2.0 / (n - 1)
    
    print(f"Test function: 2x³ - 3x² + x + 1 with 5% noise")
    print(f"Domain: [0, 2] with n={n} points")
    
    # Sergei cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2.0, y_data, n, 3, False, coeff)
    
    # SciPy cubic spline
    scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
    
    # Test evaluation - compare smoothness rather than exact function match
    x_test = np.linspace(0.1, 1.9, 50)
    sergei_vals = []
    scipy_vals = []
    
    y_out = np.zeros(1)
    for x in x_test:
        # Sergei evaluation
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x, y_out)
        sergei_vals.append(y_out[0])
        
        # SciPy evaluation
        scipy_vals.append(scipy_spline(x))
    
    # Compare spline similarity rather than absolute accuracy
    spline_diff = np.array(sergei_vals) - np.array(scipy_vals)
    max_diff = np.max(np.abs(spline_diff))
    rms_diff = np.sqrt(np.mean(spline_diff**2))
    
    print(f"Spline comparison:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  RMS difference: {rms_diff:.2e}")
    
    return max_diff, rms_diff

def test_quintic_vs_high_order_scipy():
    """Compare quintic splines with high-order SciPy interpolation"""
    if not HAS_SCIPY:
        return None
    
    print(f"\n\nCOMPARING QUINTIC SPLINES (HIGH-ORDER TEST)")
    print("=" * 50)
    
    # Test function that benefits from high-order interpolation
    def f(x):
        return np.sin(x) * np.exp(-0.5*x) + 0.1*x**3
    
    n = 12
    x_data = np.linspace(0, 3, n)
    y_data = f(x_data)
    h = 3.0 / (n - 1)
    
    print(f"Test function: sin(x)*exp(-0.5x) + 0.1x³")
    print(f"Domain: [0, 3] with n={n} points")
    
    # Sergei quintic spline
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 3.0, y_data, n, 5, False, coeff)
    
    # SciPy high-order spline (degree 5)
    from scipy.interpolate import splrep, splev
    tck = splrep(x_data, y_data, k=5)  # degree 5 spline
    
    # Test evaluation
    x_test = np.linspace(0.2, 2.8, 30)
    sergei_errors = []
    scipy_errors = []
    
    y_out = np.zeros(1)
    for x in x_test:
        # Sergei evaluation
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x, y_out)
        y_sergei = y_out[0]
        
        # SciPy evaluation
        y_scipy = splev(x, tck)
        
        # Exact value
        y_exact = f(x)
        
        sergei_error = abs(y_sergei - y_exact)
        scipy_error = abs(y_scipy - y_exact)
        
        sergei_errors.append(sergei_error)
        scipy_errors.append(scipy_error)
    
    sergei_max = np.max(sergei_errors)
    scipy_max = np.max(scipy_errors)
    sergei_rms = np.sqrt(np.mean(np.array(sergei_errors)**2))
    scipy_rms = np.sqrt(np.mean(np.array(scipy_errors)**2))
    
    print(f"Results:")
    print(f"  Sergei quintic - Max error: {sergei_max:.2e}, RMS error: {sergei_rms:.2e}")
    print(f"  SciPy degree-5 - Max error: {scipy_max:.2e}, RMS error: {scipy_rms:.2e}")
    print(f"  Ratio (Sergei/SciPy) - Max: {sergei_max/scipy_max:.2f}, RMS: {sergei_rms/scipy_rms:.2f}")
    
    return sergei_max

def test_periodic_cubic():
    """Test periodic cubic splines on periodic function"""
    print(f"\n\nTESTING PERIODIC CUBIC SPLINES")
    print("=" * 40)
    
    # Periodic function
    def f(x):
        return np.cos(x) + 0.3*np.sin(3*x)
    
    n = 20
    x_data = np.linspace(0, 2*np.pi, n, endpoint=False)
    y_data = f(x_data)
    h = 2*np.pi / n
    
    print(f"Test function: cos(x) + 0.3*sin(3x)")
    print(f"Domain: [0, 2π) with n={n} points")
    print(f"Periodicity: f(0)={f(0):.6f}, f(2π)={f(2*np.pi):.6f}")
    
    # Sergei periodic cubic spline
    coeff = np.zeros(4*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 2*np.pi, y_data, n, 3, True, coeff)
    
    # Test evaluation and periodicity
    test_points = [0.5, 1.0, 2.0, 3.0, 5.0]
    max_error = 0.0
    max_periodicity_error = 0.0
    
    y_out = np.zeros(1)
    print(f"\nEvaluation test:")
    for x in test_points:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x, y_out)
        y_spline = y_out[0]
        y_exact = f(x)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        print(f"  x={x:.1f}: spline={y_spline:.6f}, exact={y_exact:.6f}, error={error:.2e}")
    
    # Test periodicity
    print(f"\nPeriodicity test:")
    periodic_pairs = [(0.5, 0.5 + 2*np.pi), (1.5, 1.5 + 2*np.pi)]
    for x1, x2 in periodic_pairs:
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x1, y_out)
        y1 = y_out[0]
        evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x2, y_out)
        y2 = y_out[0]
        error = abs(y1 - y2)
        max_periodicity_error = max(max_periodicity_error, error)
        print(f"  f({x1:.1f}) vs f({x2:.1f}): {y1:.8f} vs {y2:.8f}, diff={error:.2e}")
    
    print(f"\nMax evaluation error: {max_error:.2e}")
    print(f"Max periodicity error: {max_periodicity_error:.2e}")
    
    return max_error, max_periodicity_error

def main():
    """Main test routine"""
    print("COMPREHENSIVE SCIPY COMPARISON TESTS")
    print("=" * 50)
    
    if not HAS_SCIPY:
        print("SciPy not available - skipping comparisons")
        return False
    
    # Run tests
    sergei_smooth, scipy_smooth = test_cubic_vs_scipy_smooth_function()
    noisy_diff_max, noisy_diff_rms = test_cubic_vs_scipy_noisy_data()
    quintic_error = test_quintic_vs_high_order_scipy()
    periodic_error, periodicity_error = test_periodic_cubic()
    
    print(f"\n" + "=" * 60)
    print("SCIPY COMPARISON SUMMARY")
    print("=" * 60)
    
    if sergei_smooth and scipy_smooth:
        print(f"Smooth function - Sergei: {sergei_smooth:.2e}, SciPy: {scipy_smooth:.2e}")
        ratio = sergei_smooth / scipy_smooth
        if ratio < 3.0:
            print("✅ Sergei cubic competitive with SciPy cubic")
        else:
            print("⚠️  Sergei cubic significantly worse than SciPy")
    
    if noisy_diff_max:
        print(f"Noisy data - Max difference: {noisy_diff_max:.2e}, RMS: {noisy_diff_rms:.2e}")
        if noisy_diff_max < 0.1:
            print("✅ Sergei and SciPy cubic give similar results on noisy data")
        else:
            print("⚠️  Sergei and SciPy cubic differ significantly")
    
    if quintic_error:
        print(f"Quintic spline - Max error: {quintic_error:.2e}")
        if quintic_error < 1e-3:
            print("✅ Quintic splines achieve good accuracy")
        else:
            print("⚠️  Quintic splines need improvement")
    
    print(f"Periodic cubic - Evaluation: {periodic_error:.2e}, Periodicity: {periodicity_error:.2e}")
    if periodic_error < 1e-2 and periodicity_error < 1e-14:
        print("✅ Periodic cubic splines work well")
    else:
        print("⚠️  Periodic cubic splines need improvement")
    
    # Overall assessment
    success_count = 0
    total_tests = 4
    
    if sergei_smooth and scipy_smooth and sergei_smooth/scipy_smooth < 3.0:
        success_count += 1
    if noisy_diff_max and noisy_diff_max < 0.1:
        success_count += 1
    if quintic_error and quintic_error < 1e-3:
        success_count += 1
    if periodic_error < 1e-2 and periodicity_error < 1e-14:
        success_count += 1
    
    print(f"\nOverall: {success_count}/{total_tests} tests passed")
    
    if success_count >= 3:
        print("🎉 SCIPY COMPARISON SUCCESSFUL!")
        return True
    else:
        print("❌ Some comparisons need improvement")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)