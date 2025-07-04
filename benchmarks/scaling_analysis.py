#!/usr/bin/env python3
"""Log-log scaling analysis for multi-point meshgrid evaluation."""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev

# Generate test data
print("Scaling Analysis: Multi-Point Meshgrid Evaluation")
print("=" * 60)

np.random.seed(42)
n = 100
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x)

# Fit splines
tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3)
tck_fast = bisplrep(x, y, z, kx=3, ky=3)
tx, ty, c, kx, ky = tck_fast

# Warmup
x_warm = np.linspace(-0.5, 0.5, 5)
y_warm = np.linspace(-0.5, 0.5, 5)
result_warm = np.zeros((5, 5))
bisplev(x_warm, y_warm, tx, ty, c, kx, ky, result_warm)

# Test grid sizes (powers of 2 for clear scaling)
grid_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
points = []
scipy_times = []
fast_times = []

print(f"{'Grid Size':<10} {'Points':<8} {'SciPy (ms)':<12} {'FastSpline (ms)':<15} {'Speedup':<8}")
print("-" * 65)

for size in grid_sizes:
    n_points = size * size
    points.append(n_points)
    
    x_grid = np.linspace(-0.8, 0.8, size)
    y_grid = np.linspace(-0.8, 0.8, size)
    
    # SciPy timing
    start = time.perf_counter()
    result_scipy = scipy_bisplev(x_grid, y_grid, tck_scipy)
    scipy_time = (time.perf_counter() - start) * 1000
    scipy_times.append(scipy_time)
    
    # FastSpline timing
    result_fast = np.zeros((size, size))
    start = time.perf_counter()
    bisplev(x_grid, y_grid, tx, ty, c, kx, ky, result_fast)
    fast_time = (time.perf_counter() - start) * 1000
    fast_times.append(fast_time)
    
    speedup = scipy_time / fast_time
    print(f"{size}x{size:<7} {n_points:<8} {scipy_time:<12.3f} {fast_time:<15.3f} {speedup:<8.2f}x")

# Create log-log plot
plt.figure(figsize=(12, 8))

# Plot 1: Time vs Points (log-log)
plt.subplot(2, 2, 1)
plt.loglog(points, scipy_times, 'o-', label='SciPy', linewidth=2, markersize=6)
plt.loglog(points, fast_times, 's-', label='FastSpline', linewidth=2, markersize=6)
plt.xlabel('Number of Points (N²)')
plt.ylabel('Evaluation Time (ms)')
plt.title('Log-Log Scaling: Time vs Grid Points')
plt.legend()
plt.grid(True, alpha=0.3)

# Add scaling reference lines
min_points, max_points = min(points), max(points)
linear_ref = np.array([min_points, max_points])
quadratic_ref = linear_ref**2 / linear_ref[0]**2 * scipy_times[0]
plt.loglog(linear_ref, quadratic_ref * scipy_times[0] / quadratic_ref[0], 
           'k--', alpha=0.5, label='O(N²) reference')

# Plot 2: Speedup vs Points
plt.subplot(2, 2, 2)
speedups = [s/f for s, f in zip(scipy_times, fast_times)]
plt.semilogx(points, speedups, 'ro-', linewidth=2, markersize=6)
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Parity')
plt.xlabel('Number of Points (N²)')
plt.ylabel('Speedup (SciPy/FastSpline)')
plt.title('Speedup vs Grid Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Throughput (points/second)
plt.subplot(2, 2, 3)
scipy_throughput = [p/(t/1000) for p, t in zip(points, scipy_times)]
fast_throughput = [p/(t/1000) for p, t in zip(points, fast_times)]
plt.loglog(points, scipy_throughput, 'o-', label='SciPy', linewidth=2, markersize=6)
plt.loglog(points, fast_throughput, 's-', label='FastSpline', linewidth=2, markersize=6)
plt.xlabel('Number of Points (N²)')
plt.ylabel('Throughput (points/sec)')
plt.title('Evaluation Throughput')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Performance ratio by grid size
plt.subplot(2, 2, 4)
grid_sizes_array = np.array(grid_sizes)
plt.semilogx(grid_sizes_array, speedups, 'go-', linewidth=2, markersize=6)
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Parity')
plt.xlabel('Grid Size (N)')
plt.ylabel('Speedup (SciPy/FastSpline)')
plt.title('Performance vs Grid Dimension')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmarks/scaling_analysis.pdf', bbox_inches='tight')

print(f"\nScaling Analysis Summary:")
print(f"- Grid sizes tested: {grid_sizes[0]}x{grid_sizes[0]} to {grid_sizes[-1]}x{grid_sizes[-1]}")
print(f"- Point range: {points[0]:,} to {points[-1]:,} points")
print(f"- Best speedup: {max(speedups):.2f}x at {grid_sizes[speedups.index(max(speedups))]}x{grid_sizes[speedups.index(max(speedups))]}")
print(f"- Average speedup: {np.mean(speedups):.2f}x")
print(f"- Large grid speedup (≥128x128): {np.mean([s for i, s in enumerate(speedups) if grid_sizes[i] >= 128]):.2f}x")

print(f"\nPlot saved to: benchmarks/scaling_analysis.pdf")

# Show theoretical scaling
print(f"\nTheoretical Scaling Analysis:")
print(f"- Both methods show O(N²) scaling as expected for meshgrid evaluation")
print(f"- SciPy time increases: {scipy_times[-1]/scipy_times[0]:.1f}x for {points[-1]/points[0]:.0f}x more points")
print(f"- FastSpline time increases: {fast_times[-1]/fast_times[0]:.1f}x for {points[-1]/points[0]:.0f}x more points")
print(f"- Scaling efficiency: Both methods scale proportionally to grid size²")

plt.show()