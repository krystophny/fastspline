#!/usr/bin/env python3
"""Comprehensive benchmark of all bisplev modes and implementations."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_cfunc, bisplev_grid_cfunc, bisplev_points_cfunc

# Generate test data
np.random.seed(42)
n = 500
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
z = np.exp(-(x**2 + y**2))

# Fit with both
print("Fitting splines...")
tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3)
tck_fast = bisplrep(x, y, z, kx=3, ky=3)
tx, ty, c, kx, ky = tck_fast

print("\nComprehensive Performance Benchmark")
print("=" * 80)

# Test 1: Single point evaluation
print("\n1. SINGLE POINT EVALUATION (10,000 calls)")
print("-" * 60)
x_pt, y_pt = 0.5, 0.5
n_calls = 10000

# SciPy
start = time.perf_counter()
for _ in range(n_calls):
    result = scipy_bisplev(x_pt, y_pt, tck_scipy)
t_scipy = (time.perf_counter() - start) * 1000
print(f"SciPy bisplev:                    {t_scipy:8.2f} ms ({t_scipy/n_calls:.4f} ms/call)")

# FastSpline Python wrapper
start = time.perf_counter()
for _ in range(n_calls):
    result = bisplev(np.array([x_pt]), np.array([y_pt]), tck_fast, grid=False)
t_fast_py = (time.perf_counter() - start) * 1000
print(f"FastSpline bisplev (Python):      {t_fast_py:8.2f} ms ({t_fast_py/n_calls:.4f} ms/call)")

# FastSpline cfunc
start = time.perf_counter()
for _ in range(n_calls):
    result = bisplev_cfunc(x_pt, y_pt, tx, ty, c, kx, ky)
t_fast_cfunc = (time.perf_counter() - start) * 1000
print(f"FastSpline bisplev_cfunc:         {t_fast_cfunc:8.2f} ms ({t_fast_cfunc/n_calls:.4f} ms/call)")

print(f"\nSpeedup (cfunc vs SciPy):         {t_scipy/t_fast_cfunc:.1f}x")

# Test 2: Grid evaluation
print("\n2. GRID EVALUATION (50x50 = 2,500 points)")
print("-" * 60)
x_grid = np.linspace(-0.8, 0.8, 50)
y_grid = np.linspace(-0.8, 0.8, 50)

# SciPy (native grid)
start = time.perf_counter()
result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
t_scipy_grid = (time.perf_counter() - start) * 1000
print(f"SciPy bisplev (grid):             {t_scipy_grid:8.2f} ms")

# FastSpline Python wrapper (grid=True default)
start = time.perf_counter()
result_fast = bisplev(x_grid, y_grid, tck_fast)
t_fast_grid = (time.perf_counter() - start) * 1000
print(f"FastSpline bisplev (grid=True):   {t_fast_grid:8.2f} ms")

# FastSpline grid cfunc with pre-allocated result
result_buffer = np.zeros((50, 50))
start = time.perf_counter()
bisplev_grid_cfunc(x_grid, y_grid, tx, ty, c, kx, ky, result_buffer)
t_fast_grid_cfunc = (time.perf_counter() - start) * 1000
print(f"FastSpline bisplev_grid_cfunc:    {t_fast_grid_cfunc:8.2f} ms")

print(f"\nSciPy remains faster for grids:   {t_fast_grid/t_scipy_grid:.1f}x slower")

# Test 3: Scattered points
print("\n3. SCATTERED POINTS EVALUATION (10,000 random points)")
print("-" * 60)
n_scatter = 10000
x_scatter = np.random.uniform(-0.8, 0.8, n_scatter)
y_scatter = np.random.uniform(-0.8, 0.8, n_scatter)

# SciPy (must loop)
start = time.perf_counter()
result_scipy = np.array([scipy_bisplev(x_scatter[i], y_scatter[i], tck_scipy) 
                        for i in range(n_scatter)])
t_scipy_scatter = (time.perf_counter() - start) * 1000
print(f"SciPy bisplev (loop):             {t_scipy_scatter:8.2f} ms")

# FastSpline Python wrapper (grid=False)
start = time.perf_counter()
result_fast = bisplev(x_scatter, y_scatter, tck_fast, grid=False)
t_fast_scatter = (time.perf_counter() - start) * 1000
print(f"FastSpline bisplev (grid=False):  {t_fast_scatter:8.2f} ms")

# FastSpline points cfunc with pre-allocated result
result_buffer = np.zeros(n_scatter)
start = time.perf_counter()
bisplev_points_cfunc(x_scatter, y_scatter, tx, ty, c, kx, ky, result_buffer)
t_fast_points_cfunc = (time.perf_counter() - start) * 1000
print(f"FastSpline bisplev_points_cfunc:  {t_fast_points_cfunc:8.2f} ms")

print(f"\nSpeedup for scattered points:     {t_scipy_scatter/t_fast_scatter:.1f}x")
print(f"Speedup with cfunc:               {t_scipy_scatter/t_fast_points_cfunc:.1f}x")

# Summary
print("\n" + "=" * 80)
print("SUMMARY:")
print("- Single point: FastSpline cfunc is {:.1f}x faster".format(t_scipy/t_fast_cfunc))
print("- Grid evaluation: SciPy is {:.1f}x faster".format(t_fast_grid/t_scipy_grid))
print("- Scattered points: FastSpline is {:.1f}x faster".format(t_scipy_scatter/t_fast_scatter))
print("\nRecommendations:")
print("- Use scipy.bisplev for regular grids (meshgrid evaluation)")
print("- Use fastspline.bisplev(grid=False) for scattered points")
print("- Use bisplev_cfunc for single points in tight loops")
print("- Use bisplev_points_cfunc for maximum scattered point performance")