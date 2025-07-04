#!/usr/bin/env python3
"""Test meshgrid performance for FastSpline vs SciPy."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep as fast_bisplrep, bisplev as fast_bisplev, bisplev_scalar

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

tx, ty, c, kx, ky = tck_fast

for size in grid_sizes:
    x_grid = np.linspace(-0.8, 0.8, size)
    y_grid = np.linspace(-0.8, 0.8, size)
    n_points = size * size
    
    # Warm up JIT on first iteration
    if size == grid_sizes[0]:
        warmup_result = np.zeros((2, 2))
        fast_bisplev(x_grid[:2], y_grid[:2], tx, ty, c, kx, ky, warmup_result)
    
    # SciPy
    start = time.perf_counter()
    result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
    scipy_time = (time.perf_counter() - start) * 1000
    
    # FastSpline - pre-allocate result array
    result_fast = np.zeros((size, size))
    start = time.perf_counter()
    fast_bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_fast)
    fast_time = (time.perf_counter() - start) * 1000
    
    speedup = scipy_time / fast_time
    
    print(f"{size}x{size:<8} {scipy_time:<15.2f} {fast_time:<18.2f} {speedup:<10.2f}x")
    
    # Verify accuracy for first grid
    if size == grid_sizes[0]:
        max_diff = np.max(np.abs(result_scipy - result_fast))
        print(f"  Max difference: {max_diff:.2e}")

# Additional test: compare with manual scalar evaluation
print("\nComparison with scalar evaluation (10x10 grid):")
size = 10
x_grid = np.linspace(-0.8, 0.8, size)
y_grid = np.linspace(-0.8, 0.8, size)

# Manual scalar evaluation
start = time.perf_counter()
result_scalar = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        result_scalar[i, j] = bisplev_scalar(x_grid[i], y_grid[j], tx, ty, c, kx, ky)
scalar_time = (time.perf_counter() - start) * 1000

# Array evaluation
result_array = np.zeros((size, size))
start = time.perf_counter()
fast_bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_array)
array_time = (time.perf_counter() - start) * 1000

print(f"  Scalar evaluation: {scalar_time:.2f} ms")
print(f"  Array evaluation:  {array_time:.2f} ms")
print(f"  Array speedup:     {scalar_time/array_time:.2f}x")
print(f"  Max difference:    {np.max(np.abs(result_scalar - result_array)):.2e}")