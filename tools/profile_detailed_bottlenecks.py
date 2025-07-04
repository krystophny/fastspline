#!/usr/bin/env python3
"""Detailed profiling to identify performance bottlenecks in bisplev."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev

def profile_individual_components():
    """Profile each component of the evaluation pipeline."""
    print("Detailed Component Profiling")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 1000)
    y_data = np.random.uniform(-1, 1, 1000)
    z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data)
    
    # Create splines
    tck_k1 = scipy_bisplrep(x_data, y_data, z_data, kx=1, ky=1, s=0.01)
    tck_k3 = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    
    # Test points
    n_eval = 10000
    np.random.seed(123)
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    # Profile k=1 (linear)
    print("\n1. Linear Splines (k=1) Analysis:")
    tx, ty, c = tck_k1[0], tck_k1[1], tck_k1[2]
    nx, ny = len(tx), len(ty)
    
    # Time knot span finding only
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = _find_knot_span(tx, 1, x_eval[i])
        span_y = _find_knot_span(ty, 1, y_eval[i])
    time_knot_k1 = (time.perf_counter() - start) * 1000
    
    # Time basis functions only
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = _find_knot_span(tx, 1, x_eval[i])
        span_y = _find_knot_span(ty, 1, y_eval[i])
        Nx = np.zeros(2, dtype=np.float64)
        Ny = np.zeros(2, dtype=np.float64)
        _basis_functions(tx, 1, span_x, x_eval[i], Nx)
        _basis_functions(ty, 1, span_y, y_eval[i], Ny)
    time_basis_k1 = (time.perf_counter() - start) * 1000
    
    # Time full evaluation
    start = time.perf_counter()
    for i in range(n_eval):
        result = _bisplev_cfunc(x_eval[i], y_eval[i], tx, ty, c, 1, 1, nx, ny)
    time_full_k1 = (time.perf_counter() - start) * 1000
    
    # Time scipy for comparison
    start = time.perf_counter()
    for i in range(n_eval):
        result = scipy_bisplev(x_eval[i], y_eval[i], tck_k1)
    time_scipy_k1 = (time.perf_counter() - start) * 1000
    
    print(f"  Knot span finding:     {time_knot_k1:6.1f}ms")
    print(f"  + Basis functions:     {time_basis_k1:6.1f}ms")
    print(f"  Full bisplev_cfunc:    {time_full_k1:6.1f}ms")
    print(f"  SciPy reference:       {time_scipy_k1:6.1f}ms")
    print(f"  Evaluation overhead:   {time_full_k1 - time_basis_k1:6.1f}ms")
    print(f"  SciPy speedup:         {time_full_k1/time_scipy_k1:.2f}x")
    
    # Profile k=3 (cubic)
    print("\n2. Cubic Splines (k=3) Analysis:")
    tx, ty, c = tck_k3[0], tck_k3[1], tck_k3[2]
    nx, ny = len(tx), len(ty)
    
    # Time knot span finding only
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = _find_knot_span(tx, 3, x_eval[i])
        span_y = _find_knot_span(ty, 3, y_eval[i])
    time_knot_k3 = (time.perf_counter() - start) * 1000
    
    # Time basis functions only
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = _find_knot_span(tx, 3, x_eval[i])
        span_y = _find_knot_span(ty, 3, y_eval[i])
        Nx = np.zeros(4, dtype=np.float64)
        Ny = np.zeros(4, dtype=np.float64)
        _basis_functions(tx, 3, span_x, x_eval[i], Nx)
        _basis_functions(ty, 3, span_y, y_eval[i], Ny)
    time_basis_k3 = (time.perf_counter() - start) * 1000
    
    # Time full evaluation
    start = time.perf_counter()
    for i in range(n_eval):
        result = _bisplev_cfunc(x_eval[i], y_eval[i], tx, ty, c, 3, 3, nx, ny)
    time_full_k3 = (time.perf_counter() - start) * 1000
    
    # Time scipy for comparison
    start = time.perf_counter()
    for i in range(n_eval):
        result = scipy_bisplev(x_eval[i], y_eval[i], tck_k3)
    time_scipy_k3 = (time.perf_counter() - start) * 1000
    
    print(f"  Knot span finding:     {time_knot_k3:6.1f}ms")
    print(f"  + Basis functions:     {time_basis_k3:6.1f}ms")
    print(f"  Full bisplev_cfunc:    {time_full_k3:6.1f}ms")
    print(f"  SciPy reference:       {time_scipy_k3:6.1f}ms")
    print(f"  Evaluation overhead:   {time_full_k3 - time_basis_k3:6.1f}ms")
    print(f"  SciPy speedup:         {time_full_k3/time_scipy_k3:.2f}x")
    
    # Analysis summary
    print("\n3. Bottleneck Analysis:")
    print("Linear splines (k=1):")
    print(f"  Knot span: {time_knot_k1/time_full_k1*100:.1f}% of total time")
    print(f"  Basis funcs: {(time_basis_k1-time_knot_k1)/time_full_k1*100:.1f}% of total time") 
    print(f"  Evaluation: {(time_full_k1-time_basis_k1)/time_full_k1*100:.1f}% of total time")
    
    print("Cubic splines (k=3):")
    print(f"  Knot span: {time_knot_k3/time_full_k3*100:.1f}% of total time")
    print(f"  Basis funcs: {(time_basis_k3-time_knot_k3)/time_full_k3*100:.1f}% of total time")
    print(f"  Evaluation: {(time_full_k3-time_basis_k3)/time_full_k3*100:.1f}% of total time")
    
    return {
        'k1': {'knot': time_knot_k1, 'basis': time_basis_k1, 'full': time_full_k1, 'scipy': time_scipy_k1},
        'k3': {'knot': time_knot_k3, 'basis': time_basis_k3, 'full': time_full_k3, 'scipy': time_scipy_k3}
    }

def benchmark_quick(label, n_points=50):
    """Quick benchmark for testing optimizations."""
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 100)
    y_data = np.random.uniform(-1, 1, 100)
    z_data = np.exp(-(x_data**2 + y_data**2))
    
    tck = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    
    x_eval = np.random.uniform(-0.8, 0.8, n_points)
    y_eval = np.random.uniform(-0.8, 0.8, n_points)
    
    # Time scipy
    start = time.perf_counter()
    scipy_results = [scipy_bisplev(x, y, tck) for x, y in zip(x_eval, y_eval)]
    time_scipy = (time.perf_counter() - start) * 1000
    
    # Time fastspline
    start = time.perf_counter()
    # Direct cfunc evaluation for comparison
    tx, ty, c, kx, ky = tck
    fast_results = np.array([bisplev(x, y, tx, ty, c, kx, ky, len(tx), len(ty)) 
                            for x, y in zip(x_eval, y_eval)])
    time_fast = (time.perf_counter() - start) * 1000
    
    # Check accuracy
    diff = np.abs(np.array(scipy_results) - fast_results)
    max_diff = np.max(diff)
    
    print(f"{label}: SciPy={time_scipy:.1f}ms, FastSpline={time_fast:.1f}ms, "
          f"Speedup={time_scipy/time_fast:.2f}x, MaxDiff={max_diff:.2e}")
    
    return time_scipy, time_fast, max_diff

if __name__ == "__main__":
    print("FastSpline Ultra-Optimized Performance Test")
    print("=" * 50)
    
    # Test different sizes
    for n_points in [50, 100, 500, 1000]:
        print(f"\n{n_points} point benchmark:")
        benchmark_quick(f"n={n_points}", n_points)