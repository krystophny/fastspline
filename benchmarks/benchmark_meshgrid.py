#!/usr/bin/env python3
"""Benchmark showing FastSpline's meshgrid advantages."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_scalar

# Generate test data
np.random.seed(42)
n = 200
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x)

# Fit splines
print("Fitting splines...")
tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3)
tck_fast = bisplrep(x, y, z, kx=3, ky=3)
tx, ty, c, kx, ky = tck_fast

print("\nMeshgrid Evaluation Benchmark")
print("=" * 60)

# Test different grid sizes
grid_sizes = [10, 20, 32, 50, 64]

print(f"{'Size':<8} {'SciPy (ms)':<12} {'FastSpline (ms)':<15} {'Speedup':<10} {'API'}")
print("-" * 60)

for size in grid_sizes:
    x_grid = np.linspace(-0.8, 0.8, size)
    y_grid = np.linspace(-0.8, 0.8, size)
    
    # SciPy meshgrid evaluation
    start = time.perf_counter()
    result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
    scipy_time = (time.perf_counter() - start) * 1000
    
    # FastSpline automatic meshgrid
    result_fast = np.zeros((size, size))
    start = time.perf_counter()
    bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_fast)
    fast_time = (time.perf_counter() - start) * 1000
    
    speedup = scipy_time / fast_time
    api_note = "Automatic"
    
    print(f"{size}x{size:<3} {scipy_time:<12.2f} {fast_time:<15.2f} {speedup:<10.2f}x {api_note}")

print("\nKey Advantages of FastSpline:")
print("- Automatic meshgrid detection (same length = pointwise, different = meshgrid)")
print("- C-compatible cfunc interface")  
print("- No Python overhead for array operations")
print("- Pre-allocated result arrays for memory efficiency")

# Demonstrate the API difference
print("\nAPI Comparison Example:")
print("=" * 60)

x_eval = np.array([0.1, 0.2, 0.3])
y_eval = np.array([0.1, 0.2])  # Different length -> meshgrid

print("For meshgrid evaluation of 3x2 = 6 points:")
print()
print("SciPy:")
print("  result = scipy_bisplev(x_eval, y_eval, tck)")
print("  # Automatically handles meshgrid")
print()
print("FastSpline:")
print("  result = np.zeros((3, 2))")
print("  bisplev(x_eval, y_eval, tx, ty, c, kx, ky, result)")
print("  # Automatic meshgrid detection + pre-allocated result")

# Show the actual result to verify correctness
result_scipy_demo = scipy_bisplev(x_eval, y_eval, tck_scipy)
result_fast_demo = np.zeros((3, 2))
bisplev(x_eval, y_eval, tx, ty, c, kx, ky, result_fast_demo)

print("\nResults match:", np.allclose(result_scipy_demo, result_fast_demo, rtol=1e-10))
print(f"Max difference: {np.max(np.abs(result_scipy_demo - result_fast_demo)):.2e}")