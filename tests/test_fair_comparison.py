#!/usr/bin/env python3
"""Fair performance comparison between FastSpline and SciPy bisplev."""

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

print("Fair Performance Comparison: FastSpline vs SciPy bisplev")
print("=" * 60)

# Test 1: Single point evaluation
print("\n1. Single point evaluation (1000 times):")
x_test, y_test = 0.5, 0.5

start = time.perf_counter()
for _ in range(1000):
    result = scipy_bisplev(x_test, y_test, tck_scipy)
scipy_single = (time.perf_counter() - start) * 1000

# Use bisplev_scalar for single point evaluation
tx, ty, c, kx, ky = tck_fast
start = time.perf_counter()
for _ in range(1000):
    result = bisplev_scalar(x_test, y_test, tx, ty, c, kx, ky)
fast_single = (time.perf_counter() - start) * 1000

print(f"  SciPy: {scipy_single:.2f} ms ({scipy_single/1000:.3f} ms per eval)")
print(f"  FastSpline: {fast_single:.2f} ms ({fast_single/1000:.3f} ms per eval)")
print(f"  Speedup: {scipy_single/fast_single:.1f}x")

# Test 2: Grid evaluation (scipy's strength)
print("\n2. Grid evaluation (32x32 = 1024 points):")
x_grid = np.linspace(-0.8, 0.8, 32)
y_grid = np.linspace(-0.8, 0.8, 32)

start = time.perf_counter()
result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
scipy_grid = (time.perf_counter() - start) * 1000

# FastSpline needs pre-allocated result array for meshgrid
result_fast = np.zeros((32, 32))
start = time.perf_counter()
fast_bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_fast)
fast_grid = (time.perf_counter() - start) * 1000

print(f"  SciPy: {scipy_grid:.2f} ms ({scipy_grid/1024:.3f} ms per point)")
print(f"  FastSpline: {fast_grid:.2f} ms ({fast_grid/1024:.3f} ms per point)")
print(f"  Speedup: {scipy_grid/fast_grid:.1f}x")

# Test 3: Scattered points (real-world use case)
print("\n3. Scattered points evaluation (1000 random points):")
x_scatter = np.random.uniform(-0.8, 0.8, 1000)
y_scatter = np.random.uniform(-0.8, 0.8, 1000)

# SciPy - must evaluate one by one for scattered points
start = time.perf_counter()
result_scipy_scatter = np.array([scipy_bisplev(x_scatter[i], y_scatter[i], tck_scipy) 
                        for i in range(1000)])
scipy_scatter = (time.perf_counter() - start) * 1000

# FastSpline - can handle arrays directly (pointwise since same length)
result_fast_scatter = np.zeros(1000)
start = time.perf_counter()
fast_bisplev(x_scatter, y_scatter, tx, ty, c, kx, ky, result_fast_scatter)
fast_scatter = (time.perf_counter() - start) * 1000

print(f"  SciPy: {scipy_scatter:.2f} ms ({scipy_scatter/1000:.3f} ms per point)")
print(f"  FastSpline: {fast_scatter:.2f} ms ({fast_scatter/1000:.3f} ms per point)")
print(f"  Speedup: {scipy_scatter/fast_scatter:.1f}x")

# Verify accuracy
print("\n4. Accuracy verification:")
# For grid comparison, need to evaluate on same points
grid_diff = []
for i in range(32):
    for j in range(32):
        scipy_val = result_scipy[i, j]
        fast_val = result_fast[i, j]
        grid_diff.append(abs(scipy_val - fast_val))

print(f"  Max difference (grid): {max(grid_diff):.2e}")
print(f"  Max difference (scattered): {np.max(np.abs(result_scipy_scatter - result_fast_scatter)):.2e}")