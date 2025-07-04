#!/usr/bin/env python3
"""Test meshgrid performance for FastSpline vs SciPy."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep as fast_bisplrep, bisplev as fast_bisplev

# Generate test data
np.random.seed(42)
n = 500
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
z = np.exp(-(x**2 + y**2))

# Fit with both
tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3)
tck_fast = fast_bisplrep(x, y, z, kx=3, ky=3)

print("Meshgrid Evaluation Performance Test")
print("=" * 60)

# Test different grid sizes
grid_sizes = [10, 20, 32, 50, 64, 100]

print(f"{'Grid Size':<12} {'SciPy (ms)':<15} {'FastSpline (ms)':<18} {'Speedup':<10}")
print("-" * 60)

for size in grid_sizes:
    x_grid = np.linspace(-0.8, 0.8, size)
    y_grid = np.linspace(-0.8, 0.8, size)
    n_points = size * size
    
    # Warm up JIT
    if size == grid_sizes[0]:
        _ = fast_bisplev(x_grid[:2], y_grid[:2], tck_fast)
    
    # SciPy
    start = time.perf_counter()
    result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
    scipy_time = (time.perf_counter() - start) * 1000
    
    # FastSpline
    start = time.perf_counter()
    result_fast = fast_bisplev(x_grid, y_grid, tck_fast)
    fast_time = (time.perf_counter() - start) * 1000
    
    speedup = scipy_time / fast_time
    
    print(f"{size}x{size:<8} {scipy_time:<15.2f} {fast_time:<18.2f} {speedup:<10.2f}x")
    
    # Verify results match
    max_diff = np.max(np.abs(result_scipy - result_fast))
    if max_diff > 1e-14:
        print(f"  WARNING: Max difference = {max_diff:.2e}")

print("\nNote: FastSpline uses parallel evaluation with numba")
print("      SciPy uses optimized Fortran code")