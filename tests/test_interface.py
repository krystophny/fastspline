#!/usr/bin/env python3
"""Test the scipy-compatible bisplrep/bisplev interface."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_scalar

def test_interface_compatibility():
    """Test that our interface works the same as scipy's."""
    print("Testing scipy-compatible interface...")
    
    # Generate test data
    np.random.seed(42)
    x = np.random.uniform(-1, 1, 100)
    y = np.random.uniform(-1, 1, 100)
    z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x) * np.sin(np.pi * y)
    
    # Test bisplrep
    print("Testing bisplrep...")
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0)
    tck_fast = bisplrep(x, y, z, kx=3, ky=3, s=0)
    
    print(f"SciPy tck: tx={len(tck_scipy[0])}, ty={len(tck_scipy[1])}, c={len(tck_scipy[2])}")
    print(f"FastSpline tck: tx={len(tck_fast[0])}, ty={len(tck_fast[1])}, c={len(tck_fast[2])}")
    
    # Test bisplev with scalar inputs
    print("\nTesting bisplev with scalar inputs...")
    x_test, y_test = 0.5, 0.5
    result_scipy = scipy_bisplev(x_test, y_test, tck_scipy)
    
    # FastSpline uses bisplev_scalar for single point evaluation
    tx, ty, c, kx, ky = tck_fast
    result_fast = bisplev_scalar(x_test, y_test, tx, ty, c, kx, ky)
    
    print(f"SciPy result: {result_scipy}")
    print(f"FastSpline result: {result_fast}")
    print(f"Difference: {abs(result_scipy - result_fast):.2e}")
    
    # Test bisplev with array inputs (scipy requires special handling)
    print("\nTesting bisplev with individual points...")
    x_points = [0.0, 0.5, -0.5]
    y_points = [0.0, 0.5, -0.5]
    
    results_scipy = []
    results_fast = []
    
    for x_pt, y_pt in zip(x_points, y_points):
        results_scipy.append(scipy_bisplev(x_pt, y_pt, tck_scipy))
        results_fast.append(bisplev_scalar(x_pt, y_pt, tx, ty, c, kx, ky))
    
    results_scipy = np.array(results_scipy)
    results_fast = np.array(results_fast)
    
    print(f"SciPy results: {results_scipy}")
    print(f"FastSpline results: {results_fast}")
    print(f"Max difference: {np.max(np.abs(results_scipy - results_fast)):.2e}")
    
    # Test our array interface
    print("\nTesting FastSpline array interface...")
    x_test = np.array([0.0, 0.5, -0.5])
    y_test = np.array([0.0, 0.5, -0.5])
    # Pre-allocate result array for bisplev
    results_fast_array = np.zeros(3)
    bisplev(x_test, y_test, tx, ty, c, kx, ky, results_fast_array)
    
    print(f"FastSpline array results: {results_fast_array}")
    print(f"Matches individual calls: {np.allclose(results_fast, results_fast_array)}")
    
    # Test with linear splines
    print("\nTesting with linear splines...")
    tck_scipy_linear = scipy_bisplrep(x, y, z, kx=1, ky=1, s=0)
    tck_fast_linear = bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    result_scipy_linear = scipy_bisplev(0.25, 0.25, tck_scipy_linear)
    tx_lin, ty_lin, c_lin, kx_lin, ky_lin = tck_fast_linear
    result_fast_linear = bisplev_scalar(0.25, 0.25, tx_lin, ty_lin, c_lin, kx_lin, ky_lin)
    
    print(f"Linear SciPy: {result_scipy_linear}")
    print(f"Linear FastSpline: {result_fast_linear}")
    print(f"Linear difference: {abs(result_scipy_linear - result_fast_linear):.2e}")
    
    print("\n✓ Interface compatibility test completed!")

if __name__ == "__main__":
    test_interface_compatibility()