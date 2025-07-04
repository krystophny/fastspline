#!/usr/bin/env python3
"""Profile specific bottlenecks in bisplev_cfunc."""

import numpy as np
import time
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline.spline2d import bisplev_cfunc, find_knot_span, basis_functions

def profile_components():
    """Profile individual components of bisplev_cfunc."""
    
    # Generate test data
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 1000)
    y_data = np.random.uniform(-1, 1, 1000)
    z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data) * np.sin(np.pi * y_data)
    
    # Create splines
    tck_k1 = scipy_bisplrep(x_data, y_data, z_data, kx=1, ky=1, s=0)
    tck_k3 = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
    
    # Test points
    n_eval = 10000
    np.random.seed(123)
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    print("Profiling bisplev_cfunc components")
    print("=" * 50)
    
    # Profile k=1 components
    print("\nLinear splines (k=1):")
    tx, ty, c = tck_k1[0], tck_k1[1], tck_k1[2]
    nx, ny = len(tx), len(ty)
    
    # Time find_knot_span
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = find_knot_span(tx, 1, x_eval[i])
        span_y = find_knot_span(ty, 1, y_eval[i])
    time_knot_span = (time.perf_counter() - start) * 1000
    
    # Time basis_functions  
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = find_knot_span(tx, 1, x_eval[i])
        span_y = find_knot_span(ty, 1, y_eval[i])
        Nx = np.zeros(2, dtype=np.float64)
        Ny = np.zeros(2, dtype=np.float64)
        basis_functions(tx, 1, span_x, x_eval[i], Nx)
        basis_functions(ty, 1, span_y, y_eval[i], Ny)
    time_basis = (time.perf_counter() - start) * 1000
    
    # Time full bisplev_cfunc
    start = time.perf_counter()
    for i in range(n_eval):
        result = bisplev_cfunc(x_eval[i], y_eval[i], tx, ty, c, 1, 1, nx, ny)
    time_full = (time.perf_counter() - start) * 1000
    
    # Time scipy for comparison
    start = time.perf_counter()
    for i in range(n_eval):
        result = scipy_bisplev(x_eval[i], y_eval[i], tck_k1)
    time_scipy = (time.perf_counter() - start) * 1000
    
    print(f"  Knot span finding:    {time_knot_span:6.1f}ms")
    print(f"  + Basis functions:    {time_basis:6.1f}ms")
    print(f"  Full bisplev_cfunc:   {time_full:6.1f}ms")
    print(f"  SciPy reference:      {time_scipy:6.1f}ms")
    print(f"  Overhead:             {time_full - time_basis:6.1f}ms")
    
    # Profile k=3 components
    print("\nCubic splines (k=3):")
    tx, ty, c = tck_k3[0], tck_k3[1], tck_k3[2]
    nx, ny = len(tx), len(ty)
    
    # Time find_knot_span
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = find_knot_span(tx, 3, x_eval[i])
        span_y = find_knot_span(ty, 3, y_eval[i])
    time_knot_span = (time.perf_counter() - start) * 1000
    
    # Time basis_functions  
    start = time.perf_counter()
    for i in range(n_eval):
        span_x = find_knot_span(tx, 3, x_eval[i])
        span_y = find_knot_span(ty, 3, y_eval[i])
        Nx = np.zeros(4, dtype=np.float64)
        Ny = np.zeros(4, dtype=np.float64)
        basis_functions(tx, 3, span_x, x_eval[i], Nx)
        basis_functions(ty, 3, span_y, y_eval[i], Ny)
    time_basis = (time.perf_counter() - start) * 1000
    
    # Time full bisplev_cfunc
    start = time.perf_counter()
    for i in range(n_eval):
        result = bisplev_cfunc(x_eval[i], y_eval[i], tx, ty, c, 3, 3, nx, ny)
    time_full = (time.perf_counter() - start) * 1000
    
    # Time scipy for comparison
    start = time.perf_counter()
    for i in range(n_eval):
        result = scipy_bisplev(x_eval[i], y_eval[i], tck_k3)
    time_scipy = (time.perf_counter() - start) * 1000
    
    print(f"  Knot span finding:    {time_knot_span:6.1f}ms")
    print(f"  + Basis functions:    {time_basis:6.1f}ms")
    print(f"  Full bisplev_cfunc:   {time_full:6.1f}ms")
    print(f"  SciPy reference:      {time_scipy:6.1f}ms")
    print(f"  Overhead:             {time_full - time_basis:6.1f}ms")

if __name__ == "__main__":
    profile_components()