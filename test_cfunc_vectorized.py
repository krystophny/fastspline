#!/usr/bin/env python3
"""Test the optimized cfunc vectorized bisplev implementation."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_cfunc_vectorized

def test_cfunc_accuracy():
    """Test vectorized cfunc accuracy vs scipy."""
    print("Testing vectorized cfunc accuracy...")
    
    # Generate test data
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 100)
    y_data = np.random.uniform(-1, 1, 100)
    z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data)
    
    # Create splines
    tck_scipy = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    tck_fast = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    
    # Test points
    n_test = 50
    x_test = np.random.uniform(-0.8, 0.8, n_test)
    y_test = np.random.uniform(-0.8, 0.8, n_test)
    
    # Method 1: SciPy individual calls
    scipy_results = np.array([scipy_bisplev(x, y, tck_scipy) for x, y in zip(x_test, y_test)])
    
    # Method 2: FastSpline bisplev (uses vectorized cfunc internally)
    fast_results = bisplev(x_test, y_test, tck_fast)
    
    # Method 3: Direct vectorized cfunc call
    tx, ty, c, kx, ky = tck_scipy
    cfunc_results = np.zeros_like(x_test)
    bisplev_cfunc_vectorized(x_test, y_test, tx, ty, c, kx, ky, cfunc_results)
    
    # Compare accuracy
    diff_fast = np.abs(scipy_results - fast_results)
    diff_cfunc = np.abs(scipy_results - cfunc_results)
    
    print(f"FastSpline vs SciPy - Max diff: {np.max(diff_fast):.2e}, RMS: {np.sqrt(np.mean(diff_fast**2)):.2e}")
    print(f"Cfunc vs SciPy - Max diff: {np.max(diff_cfunc):.2e}, RMS: {np.sqrt(np.mean(diff_cfunc**2)):.2e}")
    
    # Verify cfunc and bisplev give same results
    diff_internal = np.abs(fast_results - cfunc_results)
    print(f"FastSpline vs Cfunc - Max diff: {np.max(diff_internal):.2e} (should be ~0)")
    
    if np.max(diff_fast) < 1e-12 and np.max(diff_cfunc) < 1e-12:
        print("✓ Accuracy test PASSED")
        return True
    else:
        print("✗ Accuracy test FAILED")
        return False

def test_cfunc_performance():
    """Test vectorized cfunc performance."""
    print("\nTesting vectorized cfunc performance...")
    
    # Generate larger test case
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 500)
    y_data = np.random.uniform(-1, 1, 500)
    z_data = x_data**2 + y_data**2
    
    tck = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    tx, ty, c, kx, ky = tck
    
    # Large evaluation set
    n_eval = 10000
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    # Time scipy individual calls
    start = time.perf_counter()
    scipy_results = np.array([scipy_bisplev(x, y, tck) for x, y in zip(x_eval, y_eval)])
    time_scipy = (time.perf_counter() - start) * 1000
    
    # Time FastSpline bisplev (uses vectorized cfunc)
    start = time.perf_counter()
    fast_results = bisplev(x_eval, y_eval, tck)
    time_fast = (time.perf_counter() - start) * 1000
    
    # Time direct vectorized cfunc
    cfunc_results = np.zeros_like(x_eval)
    start = time.perf_counter()
    bisplev_cfunc_vectorized(x_eval, y_eval, tx, ty, c, kx, ky, cfunc_results)
    time_cfunc = (time.perf_counter() - start) * 1000
    
    print(f"Performance comparison ({n_eval} evaluations):")
    print(f"  SciPy (individual):  {time_scipy:6.1f}ms")
    print(f"  FastSpline bisplev:  {time_fast:6.1f}ms  (speedup: {time_scipy/time_fast:.2f}x)")
    print(f"  Direct cfunc:        {time_cfunc:6.1f}ms  (speedup: {time_scipy/time_cfunc:.2f}x)")
    print(f"  Cfunc vs bisplev:    {time_cfunc/time_fast:.2f}x {'faster' if time_cfunc < time_fast else 'slower'}")

def test_c_interop_example():
    """Show how to use the cfunc from C-like interface."""
    print("\nC interoperability example:")
    print("# The bisplev_cfunc_vectorized can be called from C/Fortran/Julia:")
    print("""
// C code example:
void evaluate_spline(double* x_points, double* y_points, int n_points,
                     double* tx, int nx, double* ty, int ny, 
                     double* coeffs, int kx, int ky,
                     double* results) {
    // Call the Numba-compiled cfunc directly
    bisplev_cfunc_vectorized(x_points, y_points, tx, ty, coeffs, kx, ky, results);
}
""")
    
    # Show memory layout requirements
    print("Memory layout requirements:")
    print("- All arrays must be contiguous C-order")
    print("- x_points, y_points, results must have same length")
    print("- No bounds checking - caller must ensure valid indices")

if __name__ == "__main__":
    print("Testing Optimized Vectorized C-Function (cfunc)")
    print("=" * 60)
    
    accuracy_ok = test_cfunc_accuracy()
    if accuracy_ok:
        test_cfunc_performance()
        test_c_interop_example()
    else:
        print("Skipping performance tests due to accuracy issues.")