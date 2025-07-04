#!/usr/bin/env python3
"""Test the bisplev array evaluation implementation."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_scalar

def test_array_evaluation_accuracy():
    """Test array evaluation accuracy vs scipy."""
    print("Testing array evaluation accuracy...")
    
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
    
    # Method 2: FastSpline scalar calls
    tx, ty, c, kx, ky = tck_fast
    scalar_results = np.array([bisplev_scalar(x, y, tx, ty, c, kx, ky) for x, y in zip(x_test, y_test)])
    
    # Method 3: Array evaluation with bisplev (pointwise since same length)
    array_results = np.zeros(n_test)
    bisplev(x_test, y_test, tx, ty, c, kx, ky, array_results)
    
    # Compare accuracy
    diff_scalar = np.abs(scipy_results - scalar_results)
    diff_array = np.abs(scipy_results - array_results)
    
    print(f"Scalar vs SciPy - Max diff: {np.max(diff_scalar):.2e}, RMS: {np.sqrt(np.mean(diff_scalar**2)):.2e}")
    print(f"Array vs SciPy - Max diff: {np.max(diff_array):.2e}, RMS: {np.sqrt(np.mean(diff_array**2)):.2e}")
    
    # Verify scalar and array give same results
    diff_internal = np.abs(scalar_results - array_results)
    print(f"Scalar vs Array evaluation - Max diff: {np.max(diff_internal):.2e} (should be ~0)")
    
    if np.max(diff_scalar) < 1e-10 and np.max(diff_array) < 1e-10:
        print("✓ Accuracy test PASSED")
        return True
    else:
        print("✗ Accuracy test FAILED")
        return False

def test_meshgrid_evaluation():
    """Test meshgrid evaluation mode."""
    print("\nTesting meshgrid evaluation...")
    
    # Create simple test surface
    x_data = np.linspace(-1, 1, 20)
    y_data = np.linspace(-1, 1, 20)
    xx, yy = np.meshgrid(x_data, y_data, indexing='ij')
    z_data = xx.ravel()**2 + yy.ravel()**2
    
    # Fit spline
    tck = bisplrep(xx.ravel(), yy.ravel(), z_data, kx=3, ky=3, s=0)
    tx, ty, c, kx, ky = tck
    
    # Test on smaller grid
    x_eval = np.linspace(-0.8, 0.8, 10)
    y_eval = np.linspace(-0.8, 0.8, 15)
    
    # Meshgrid evaluation (different lengths trigger meshgrid mode)
    result_grid = np.zeros((10, 15))
    bisplev(x_eval, y_eval, tx, ty, c, kx, ky, result_grid)
    
    # Verify against manual evaluation
    for i in range(10):
        for j in range(15):
            expected = bisplev_scalar(x_eval[i], y_eval[j], tx, ty, c, kx, ky)
            if abs(result_grid[i, j] - expected) > 1e-12:
                print(f"✗ Meshgrid mismatch at ({i},{j}): {result_grid[i,j]} vs {expected}")
                return False
    
    print("✓ Meshgrid evaluation test PASSED")
    return True

def test_performance():
    """Test array evaluation performance."""
    print("\nTesting array evaluation performance...")
    
    # Generate larger test case
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 200)
    y_data = np.random.uniform(-1, 1, 200)
    z_data = x_data**2 + y_data**2
    
    tck = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    tx, ty, c, kx, ky = tck
    
    # Large evaluation set
    n_eval = 10000
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    # Time scipy individual calls
    start = time.perf_counter()
    scipy_results = np.array([scipy_bisplev(x, y, tck) for x, y in zip(x_eval, y_eval)])
    time_scipy = (time.perf_counter() - start) * 1000
    
    # Time scalar evaluation
    start = time.perf_counter()
    scalar_results = np.array([bisplev_scalar(x, y, tx, ty, c, kx, ky) for x, y in zip(x_eval, y_eval)])
    time_scalar = (time.perf_counter() - start) * 1000
    
    # Time array evaluation
    array_results = np.zeros(n_eval)
    start = time.perf_counter()
    bisplev(x_eval, y_eval, tx, ty, c, kx, ky, array_results)
    time_array = (time.perf_counter() - start) * 1000
    
    print(f"Performance comparison ({n_eval} evaluations):")
    print(f"  SciPy (individual):    {time_scipy:6.1f}ms")
    print(f"  FastSpline scalar:     {time_scalar:6.1f}ms  (speedup: {time_scipy/time_scalar:.2f}x)")
    print(f"  FastSpline array:      {time_array:6.1f}ms  (speedup: {time_scipy/time_array:.2f}x)")

def test_c_interop_example():
    """Show how to use the cfunc from C-like interface."""
    print("\nC interoperability example:")
    print("# The bisplev_scalar cfunc can be called from C/Fortran/Julia:")
    print("""
// C code example:
double evaluate_spline_point(double x, double y, 
                           double* tx, double* ty, double* coeffs, 
                           int kx, int ky) {
    // Call the Numba-compiled cfunc directly
    return bisplev_scalar(x, y, tx, ty, coeffs, kx, ky);
}
""")
    
    # Show how to get the function address
    print("\nGetting function address for C interop:")
    print(">>> from fastspline import bisplev_scalar")
    print(">>> func_address = bisplev_scalar.address")
    print(f">>> print(hex(func_address))  # Example: {hex(bisplev_scalar.address)}")

if __name__ == "__main__":
    print("Testing Array Evaluation and C-Function Interface")
    print("=" * 60)
    
    accuracy_ok = test_array_evaluation_accuracy()
    meshgrid_ok = test_meshgrid_evaluation()
    
    if accuracy_ok and meshgrid_ok:
        test_performance()
        test_c_interop_example()
    else:
        print("Skipping performance tests due to accuracy issues.")