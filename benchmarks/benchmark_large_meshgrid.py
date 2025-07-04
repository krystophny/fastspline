#!/usr/bin/env python3
"""Benchmark FastSpline vs SciPy for large meshgrid evaluations."""

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

# Warmup compilation
print("Warming up FastSpline compilation...")
x_warmup = np.linspace(-0.5, 0.5, 5)
y_warmup = np.linspace(-0.5, 0.5, 5)
result_warmup = np.zeros((5, 5))
bisplev(x_warmup, y_warmup, tx, ty, c, kx, ky, result_warmup)

print("\nLarge Meshgrid Evaluation Benchmark")
print("=" * 70)

# Test progressively larger grids
grid_sizes = [50, 100, 150, 200, 250, 300]

print(f"{'Grid Size':<12} {'Points':<10} {'SciPy (ms)':<12} {'FastSpline (ms)':<15} {'Speedup':<10}")
print("-" * 70)

for size in grid_sizes:
    x_grid = np.linspace(-0.8, 0.8, size)
    y_grid = np.linspace(-0.8, 0.8, size)
    n_points = size * size
    
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
    
    print(f"{size}x{size:<7} {n_points:<10} {scipy_time:<12.2f} {fast_time:<15.2f} {speedup:<10.2f}x")
    
    # Verify accuracy for first grid
    if size == grid_sizes[0]:
        max_diff = np.max(np.abs(result_scipy - result_fast))
        print(f"  Accuracy check: Max difference = {max_diff:.2e}")

print("\nScattered Points vs Meshgrid Comparison")
print("=" * 70)

# Compare scattered vs meshgrid evaluation
n_scattered = 10000
x_scattered = np.random.uniform(-0.8, 0.8, n_scattered)
y_scattered = np.random.uniform(-0.8, 0.8, n_scattered)

# Equivalent meshgrid size
grid_size = int(np.sqrt(n_scattered))  # 100x100 = 10000 points
x_grid = np.linspace(-0.8, 0.8, grid_size)
y_grid = np.linspace(-0.8, 0.8, grid_size)

print(f"Evaluating {n_scattered} points:")
print(f"{'Method':<20} {'Time (ms)':<12} {'Points/sec':<15}")
print("-" * 50)

# SciPy scattered (must evaluate one by one)
start = time.perf_counter()
result_scipy_scattered = np.array([scipy_bisplev(x_scattered[i], y_scattered[i], tck_scipy) 
                                 for i in range(n_scattered)])
scipy_scattered_time = (time.perf_counter() - start) * 1000

# FastSpline scattered (pointwise mode)
result_fast_scattered = np.zeros(n_scattered)
start = time.perf_counter()
bisplev(x_scattered, y_scattered, tx, ty, c, kx, ky, result_fast_scattered)
fast_scattered_time = (time.perf_counter() - start) * 1000

# FastSpline meshgrid 
result_fast_grid = np.zeros((grid_size, grid_size))
start = time.perf_counter()
bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_fast_grid)
fast_grid_time = (time.perf_counter() - start) * 1000

print(f"{'SciPy scattered':<20} {scipy_scattered_time:<12.1f} {n_scattered/(scipy_scattered_time/1000):<15.0f}")
print(f"{'FastSpline scattered':<20} {fast_scattered_time:<12.1f} {n_scattered/(fast_scattered_time/1000):<15.0f}")
print(f"{'FastSpline meshgrid':<20} {fast_grid_time:<12.1f} {n_scattered/(fast_grid_time/1000):<15.0f}")

scattered_speedup = scipy_scattered_time / fast_scattered_time
grid_speedup = scipy_scattered_time / fast_grid_time

print(f"\nSpeedup vs SciPy scattered:")
print(f"  FastSpline scattered: {scattered_speedup:.1f}x")
print(f"  FastSpline meshgrid:  {grid_speedup:.1f}x")

print("\nMemory Usage Comparison")
print("=" * 40)
print(f"SciPy result:      {result_scipy.nbytes / 1024:.1f} KB")
print(f"FastSpline result: {result_fast.nbytes / 1024:.1f} KB")
print(f"Memory efficiency: Same (pre-allocated arrays)")

print("\nKey FastSpline Advantages:")
print("- Automatic detection: same length → pointwise, different → meshgrid")
print("- C-compatible cfunc interface for interoperability")
print("- Memory efficient with pre-allocated result arrays")
print("- Single function handles both evaluation modes seamlessly")