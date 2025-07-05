"""Benchmark different bisplev implementations."""

import numpy as np
import time
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline import bisplrep
from fastspline.bisplev_dierckx import bisplev as bisplev_dierckx
from fastspline.bisplev_fast import bisplev_fast


def benchmark_bisplev():
    """Compare bisplev implementations."""
    print("Bisplev Performance Comparison")
    print("=" * 60)
    
    # Create test surface
    x = np.linspace(0, 2*np.pi, 30)
    y = np.linspace(0, 2*np.pi, 30)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx) * np.cos(yy) + 0.1 * xx * yy
    
    # Fit spline
    print("Fitting spline...")
    tck = bisplrep(xx.ravel(), yy.ravel(), z.ravel(), s=0)
    tck_scipy = interpolate.bisplrep(xx.ravel(), yy.ravel(), z.ravel(), s=0)
    
    # Test different evaluation grid sizes
    eval_sizes = [10, 50, 100, 200, 300, 400, 500]
    
    print("\nEvaluation Performance (time in milliseconds):")
    print(f"{'Grid Size':>10} {'SciPy':>10} {'DIERCKX':>10} {'Fast':>10} {'Speedup':>10}")
    print("-" * 60)
    
    for n in eval_sizes:
        x_eval = np.linspace(0, 2*np.pi, n)
        y_eval = np.linspace(0, 2*np.pi, n)
        
        # Warm up
        _ = interpolate.bisplev(x_eval, y_eval, tck_scipy)
        _ = bisplev_dierckx(x_eval, y_eval, tck)
        _ = bisplev_fast(x_eval, y_eval, tck)
        
        # Time SciPy
        n_iter = max(1, 100 // n)
        start = time.time()
        for _ in range(n_iter):
            z_scipy = interpolate.bisplev(x_eval, y_eval, tck_scipy)
        time_scipy = 1000 * (time.time() - start) / n_iter
        
        # Time DIERCKX
        start = time.time()
        for _ in range(n_iter):
            z_dierckx = bisplev_dierckx(x_eval, y_eval, tck)
        time_dierckx = 1000 * (time.time() - start) / n_iter
        
        # Time Fast
        start = time.time()
        for _ in range(n_iter):
            z_fast = bisplev_fast(x_eval, y_eval, tck)
        time_fast = 1000 * (time.time() - start) / n_iter
        
        speedup = time_scipy / time_fast if time_fast > 0 else 0
        
        print(f"{n:>10}x{n:<4} {time_scipy:>10.2f} {time_dierckx:>10.2f} "
              f"{time_fast:>10.2f} {speedup:>10.1f}x")
        
        # Verify accuracy
        max_diff = np.abs(z_fast - z_scipy).max()
        if max_diff > 1e-10:
            print(f"  WARNING: Max difference = {max_diff:.2e}")
    
    # Test scalar evaluation
    print("\n\nScalar Evaluation Performance (microseconds per call):")
    print(f"{'Method':>10} {'Time':>10}")
    print("-" * 30)
    
    x_scalar = 1.5
    y_scalar = 2.5
    n_scalar = 10000
    
    # SciPy
    start = time.time()
    for _ in range(n_scalar):
        _ = interpolate.bisplev(x_scalar, y_scalar, tck_scipy)
    time_scipy_scalar = 1e6 * (time.time() - start) / n_scalar
    
    # DIERCKX
    start = time.time()
    for _ in range(n_scalar):
        _ = bisplev_dierckx(x_scalar, y_scalar, tck)
    time_dierckx_scalar = 1e6 * (time.time() - start) / n_scalar
    
    # Fast
    start = time.time()
    for _ in range(n_scalar):
        _ = bisplev_fast(x_scalar, y_scalar, tck)
    time_fast_scalar = 1e6 * (time.time() - start) / n_scalar
    
    print(f"{'SciPy':>10} {time_scipy_scalar:>10.2f}")
    print(f"{'DIERCKX':>10} {time_dierckx_scalar:>10.2f}")
    print(f"{'Fast':>10} {time_fast_scalar:>10.2f}")
    print(f"\nSpeedup over SciPy: {time_scipy_scalar/time_fast_scalar:.1f}x")


if __name__ == "__main__":
    benchmark_bisplev()