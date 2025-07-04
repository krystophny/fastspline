#!/usr/bin/env python3
"""Test and benchmark the bisplrep cfunc implementation."""

import numpy as np
import time
import sys
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev

# Import our cfunc
sys.path.insert(0, 'src')
from fastspline import bisplrep, bisplev_scalar as bisplev


def test_simple_surface():
    """Test with a simple polynomial surface."""
    print("Testing simple polynomial surface...")
    
    # Create a simple polynomial surface
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    xx, yy = np.meshgrid(x, y)
    x_data = xx.ravel()
    y_data = yy.ravel()
    z_data = x_data**2 + y_data**2  # Simple paraboloid
    
    # Call bisplrep with Python interface
    tck = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    tx, ty, c, kx, ky = tck
    
    print(f"  Knot counts: nx={len(tx)}, ny={len(ty)}")
    
    # Test evaluation at a few points
    test_points = [(0.0, 0.0), (0.5, 0.5), (-0.5, 0.5)]
    for x_test, y_test in test_points:
        z_eval = bisplev(x_test, y_test, tx, ty, c, kx, ky)
        z_true = x_test**2 + y_test**2
        print(f"  Point ({x_test:4.1f}, {y_test:4.1f}): eval={z_eval:.6f}, true={z_true:.6f}, diff={abs(z_eval-z_true):.2e}")
    
    return True


def test_against_scipy():
    """Compare our implementation against SciPy."""
    print("\nComparing against SciPy...")
    
    # Generate test data
    np.random.seed(42)
    n_points = 100
    x_data = np.random.uniform(-1, 1, n_points)
    y_data = np.random.uniform(-1, 1, n_points)
    z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data)
    
    # SciPy bisplrep
    start = time.perf_counter()
    tck_scipy = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    time_scipy = (time.perf_counter() - start) * 1000
    tx_scipy, ty_scipy, c_scipy, kx_scipy, ky_scipy = tck_scipy
    
    # Our bisplrep (currently using SciPy)
    start = time.perf_counter()
    tck_ours = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    time_ours = (time.perf_counter() - start) * 1000
    tx_ours, ty_ours, c_ours, kx_ours, ky_ours = tck_ours
    
    print(f"  SciPy time: {time_scipy:.2f}ms")
    print(f"  Our time: {time_ours:.2f}ms")
    print(f"  Ratio: {time_ours/time_scipy:.2f}x")
    print(f"  SciPy knots: nx={len(tx_scipy)}, ny={len(ty_scipy)}")
    print(f"  Our knots: nx={len(tx_ours)}, ny={len(ty_ours)}")
    
    # Compare evaluations
    n_test = 50
    np.random.seed(123)
    x_test = np.random.uniform(-0.8, 0.8, n_test)
    y_test = np.random.uniform(-0.8, 0.8, n_test)
    
    diffs = []
    for i in range(n_test):
        z_scipy = scipy_bisplev(x_test[i], y_test[i], tck_scipy)
        z_ours = bisplev(x_test[i], y_test[i], tx_ours, ty_ours, c_ours, kx_ours, ky_ours)
        diffs.append(abs(z_scipy - z_ours))
    
    diffs = np.array(diffs)
    print(f"  Max difference: {np.max(diffs):.2e}")
    print(f"  Mean difference: {np.mean(diffs):.2e}")
    print(f"  RMS difference: {np.sqrt(np.mean(diffs**2)):.2e}")
    
    return np.max(diffs) < 0.1  # Allow some difference due to different algorithms


def benchmark_performance():
    """Benchmark performance for different problem sizes."""
    print("\nPerformance Benchmark")
    print("=" * 60)
    print(f"{'N Points':<10} {'SciPy (ms)':<12} {'Ours (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for n_points in [50, 100, 200, 500, 1000]:
        # Generate data
        np.random.seed(42)
        x_data = np.random.uniform(-1, 1, n_points)
        y_data = np.random.uniform(-1, 1, n_points)
        z_data = np.sin(2*x_data) * np.cos(2*y_data) + 0.1*np.random.randn(n_points)
        
        # Time SciPy
        start = time.perf_counter()
        try:
            tck_scipy = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
            time_scipy = (time.perf_counter() - start) * 1000
        except Exception as e:
            print(f"{n_points:<10} {'Failed':<12} {'-':<12} {'-':<10}")
            continue
        
        # Time ours
        max_knots = 100
        tx = np.zeros(max_knots)
        ty = np.zeros(max_knots)
        c = np.zeros(max_knots * max_knots)
        
        start = time.perf_counter()
        result = bisplrep(x_data, y_data, z_data, 3, 3, tx, ty, c)
        time_ours = (time.perf_counter() - start) * 1000
        
        speedup = time_scipy / time_ours
        print(f"{n_points:<10} {time_scipy:<12.2f} {time_ours:<12.2f} {speedup:<10.2f}x")


def test_edge_cases():
    """Test edge cases and robustness."""
    print("\nTesting edge cases...")
    
    # Test with minimal data
    x_data = np.array([0.0, 1.0, 0.0, 1.0])
    y_data = np.array([0.0, 0.0, 1.0, 1.0])
    z_data = np.array([0.0, 1.0, 1.0, 2.0])
    
    tx = np.zeros(20)
    ty = np.zeros(20)
    c = np.zeros(400)
    
    try:
        result = bisplrep(x_data, y_data, z_data, 1, 1, tx, ty, c)
        nx = (result >> 32) & 0xFFFFFFFF
        ny = result & 0xFFFFFFFF
        print(f"  Minimal data test passed: nx={nx}, ny={ny}")
    except Exception as e:
        print(f"  Minimal data test failed: {e}")
    
    # Test with noisy data
    np.random.seed(42)
    n = 200
    x_data = np.random.uniform(-2, 2, n)
    y_data = np.random.uniform(-2, 2, n)
    z_data = np.sin(x_data) * np.cos(y_data) + 0.5*np.random.randn(n)
    
    tx = np.zeros(50)
    ty = np.zeros(50)
    c = np.zeros(2500)
    
    try:
        result = bisplrep(x_data, y_data, z_data, 3, 3, tx, ty, c)
        nx = (result >> 32) & 0xFFFFFFFF
        ny = result & 0xFFFFFFFF
        print(f"  Noisy data test passed: nx={nx}, ny={ny}")
    except Exception as e:
        print(f"  Noisy data test failed: {e}")


def main():
    """Run all tests and benchmarks."""
    print("Testing bisplrep cfunc implementation")
    print("=" * 60)
    
    # Run tests
    test_simple_surface()
    test_against_scipy()
    test_edge_cases()
    
    # Run benchmarks
    benchmark_performance()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()