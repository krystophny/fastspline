#!/usr/bin/env python3
"""Simple validation and performance test for bisplrep/bisplev."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline.bisplrep_cfunc import bisplrep
from fastspline import bisplev


def validate_simple_case():
    """Validate with a simple polynomial surface."""
    print("1. Simple Polynomial Validation")
    print("-" * 40)
    
    # Create simple test data (paraboloid)
    x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1])
    y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
    z = x**2 + y**2
    
    # Fit with our bisplrep
    tx = np.zeros(20)
    ty = np.zeros(20)
    c = np.zeros(400)
    
    result = bisplrep(x, y, z, 2, 2, tx, ty, c)
    nx = (result >> 32) & 0xFFFFFFFF
    ny = result & 0xFFFFFFFF
    
    print(f"Knots: nx={nx}, ny={ny}")
    
    # Test evaluation
    test_pts = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    print("\nEvaluation test:")
    for xt, yt in test_pts:
        z_eval = bisplev(xt, yt, tx[:nx], ty[:ny], c[:nx*ny], 2, 2)
        z_true = xt**2 + yt**2
        print(f"  ({xt}, {yt}): eval={z_eval:.4f}, true={z_true:.4f}, diff={abs(z_eval-z_true):.2e}")


def benchmark_performance():
    """Quick performance benchmark."""
    print("\n2. Performance Benchmark")
    print("-" * 40)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.exp(-(x**2 + y**2))
    
    # Time SciPy
    start = time.perf_counter()
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0)
    time_scipy_fit = (time.perf_counter() - start) * 1000
    
    # Time our implementation
    tx = np.zeros(50)
    ty = np.zeros(50)
    c = np.zeros(2500)
    
    start = time.perf_counter()
    result = bisplrep(x, y, z, 3, 3, tx, ty, c)
    time_our_fit = (time.perf_counter() - start) * 1000
    
    nx = (result >> 32) & 0xFFFFFFFF
    ny = result & 0xFFFFFFFF
    
    print(f"Fitting time:")
    print(f"  SciPy: {time_scipy_fit:.2f}ms (nx={len(tck_scipy[0])}, ny={len(tck_scipy[1])})")
    print(f"  Ours:  {time_our_fit:.2f}ms (nx={nx}, ny={ny})")
    print(f"  Speedup: {time_scipy_fit/time_our_fit:.2f}x")
    
    # Evaluation benchmark
    n_eval = 1000
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    # Time SciPy evaluation
    start = time.perf_counter()
    for i in range(n_eval):
        _ = scipy_bisplev(x_eval[i], y_eval[i], tck_scipy)
    time_scipy_eval = (time.perf_counter() - start) * 1000
    
    # Time our evaluation
    tx_scipy, ty_scipy, c_scipy, kx, ky = tck_scipy
    start = time.perf_counter()
    for i in range(n_eval):
        _ = bisplev(x_eval[i], y_eval[i], tx_scipy, ty_scipy, c_scipy, kx, ky)
    time_our_eval = (time.perf_counter() - start) * 1000
    
    print(f"\nEvaluation time ({n_eval} points):")
    print(f"  SciPy: {time_scipy_eval:.2f}ms")
    print(f"  Ours:  {time_our_eval:.2f}ms")
    print(f"  Speedup: {time_scipy_eval/time_our_eval:.2f}x")


def validate_accuracy():
    """Validate accuracy against SciPy."""
    print("\n3. Accuracy Validation")
    print("-" * 40)
    
    # Generate smooth test surface
    np.random.seed(42)
    n = 200
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.sin(2*x) * np.cos(2*y)
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0)
    
    # Test on regular grid
    n_test = 20
    x_test = np.linspace(-0.9, 0.9, n_test)
    y_test = np.linspace(-0.9, 0.9, n_test)
    
    diffs = []
    for i in range(n_test):
        for j in range(n_test):
            z_scipy = scipy_bisplev(x_test[i], y_test[j], tck_scipy)
            z_ours = bisplev(x_test[i], y_test[j], 
                            tck_scipy[0], tck_scipy[1], tck_scipy[2], 3, 3)
            diffs.append(abs(z_scipy - z_ours))
    
    diffs = np.array(diffs)
    print(f"Comparison with SciPy (using SciPy's knots):")
    print(f"  Max difference:  {np.max(diffs):.2e}")
    print(f"  Mean difference: {np.mean(diffs):.2e}")
    print(f"  RMS difference:  {np.sqrt(np.mean(diffs**2)):.2e}")
    
    if np.max(diffs) < 1e-10:
        print("  ✓ Perfect agreement with SciPy!")
    elif np.max(diffs) < 1e-6:
        print("  ✓ Excellent agreement with SciPy")
    else:
        print("  ⚠ Some differences from SciPy")


if __name__ == "__main__":
    print("FastSpline bisplrep/bisplev Validation")
    print("=" * 40)
    
    validate_simple_case()
    benchmark_performance()
    validate_accuracy()
    
    print("\nValidation complete!")