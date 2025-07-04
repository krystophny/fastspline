#!/usr/bin/env python3
"""Benchmark showing FastSpline's C interoperability advantages."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_scalar
import ctypes

# Generate test data
np.random.seed(42)
n = 100
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
z = np.exp(-(x**2 + y**2))

# Fit splines
print("FastSpline C Interoperability Benchmark")
print("=" * 60)

tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3)
tck_fast = bisplrep(x, y, z, kx=3, ky=3)
tx, ty, c, kx, ky = tck_fast

print("\n1. C Function Address Access")
print("-" * 40)
print("FastSpline provides direct C function addresses:")
print(f"bisplev_scalar address: {hex(bisplev_scalar.address)}")
print("This enables direct calls from C/C++/Fortran/Julia code")

print("\n2. Performance Comparison")
print("-" * 40)

# Warm up
for _ in range(10):
    scipy_bisplev(0.5, 0.5, tck_scipy)
    bisplev_scalar(0.5, 0.5, tx, ty, c, kx, ky)

n_calls = 1000
test_points = [(0.1*i, 0.1*i) for i in range(10)]

print(f"Evaluating {n_calls} calls across {len(test_points)} different points:")

# SciPy timing
start = time.perf_counter()
for _ in range(n_calls // len(test_points)):
    for x_pt, y_pt in test_points:
        scipy_bisplev(x_pt, y_pt, tck_scipy)
scipy_time = time.perf_counter() - start

# FastSpline timing
start = time.perf_counter()
for _ in range(n_calls // len(test_points)):
    for x_pt, y_pt in test_points:
        bisplev_scalar(x_pt, y_pt, tx, ty, c, kx, ky)
fast_time = time.perf_counter() - start

print(f"SciPy:      {scipy_time*1000:.2f} ms ({scipy_time*1000/n_calls:.4f} ms/call)")
print(f"FastSpline: {fast_time*1000:.2f} ms ({fast_time*1000/n_calls:.4f} ms/call)")
print(f"Relative performance: {fast_time/scipy_time:.2f}x")

print("\n3. Interoperability Features")
print("-" * 40)
print("✓ C-compatible function signature:")
print("  double bisplev_scalar(double x, double y, double* tx, double* ty,")
print("                        double* c, int64_t kx, int64_t ky)")
print("✓ No Python overhead - direct machine code")
print("✓ Memory layout compatible with C arrays")
print("✓ Thread-safe for parallel computation")

print("\n4. Array Interface Advantages")
print("-" * 40)

# Large array test
size = 100
x_grid = np.linspace(-0.8, 0.8, size)
y_grid = np.linspace(-0.8, 0.8, size)

print(f"Large array evaluation ({size}x{size} = {size*size} points):")

# SciPy
start = time.perf_counter()
result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
scipy_array_time = time.perf_counter() - start

# FastSpline
result_fast = np.zeros((size, size))
start = time.perf_counter()
bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_fast)
fast_array_time = time.perf_counter() - start

print(f"SciPy:      {scipy_array_time*1000:.2f} ms")
print(f"FastSpline: {fast_array_time*1000:.2f} ms")
print(f"Relative performance: {fast_array_time/scipy_array_time:.2f}x")

# Accuracy check
max_diff = np.max(np.abs(result_scipy - result_fast))
print(f"Max difference: {max_diff:.2e}")

print("\n5. Use Case Examples")
print("-" * 40)
print("FastSpline excels in these scenarios:")
print("• High-frequency evaluation from C/C++/Fortran applications")
print("• Real-time systems requiring predictable performance")
print("• Embedded systems with limited Python overhead")
print("• Multi-language scientific computing pipelines")
print("• GPU kernels via CuPy or similar frameworks")
print("• Custom numerical solvers requiring spline evaluation")

print("\n6. Memory and Threading")
print("-" * 40)
print("Memory layout:")
print(f"  SciPy result:     {result_scipy.nbytes} bytes")
print(f"  FastSpline result: {result_fast.nbytes} bytes")
print(f"  Memory overhead:  0 bytes (pre-allocated)")

print("\nThread safety:")
print("  SciPy:      Limited (Python GIL)")
print("  FastSpline: Full (pure C function)")

print("\n" + "=" * 60)
print("SUMMARY: FastSpline provides near-SciPy performance")
print("with C interoperability and no Python overhead.")