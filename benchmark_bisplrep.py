"""Benchmark bisplrep implementation against SciPy."""

import numpy as np
import time
from scipy import interpolate
from src.fastspline import bisplrep
from src.fastspline.wrappers import bisplev


def benchmark_bisplrep():
    """Benchmark bisplrep fitting and evaluation."""
    print("FastSpline bisplrep Benchmark")
    print("=" * 50)
    
    # Test different grid sizes
    sizes = [5, 10, 20, 30, 50]
    
    for n in sizes:
        print(f"\n{n}x{n} grid ({n*n} points):")
        
        # Generate test data
        x = np.linspace(0, 2*np.pi, n)
        y = np.linspace(0, 2*np.pi, n)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) * np.cos(yy)
        
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        z_flat = z.ravel()
        
        # Benchmark fitting
        # Warm up
        _ = bisplrep(x_flat, y_flat, z_flat, s=0.01)
        _ = interpolate.bisplrep(x_flat, y_flat, z_flat, s=0.01)
        
        # Time FastSpline
        start = time.time()
        tck_fast = bisplrep(x_flat, y_flat, z_flat, s=0.01)
        time_fast = time.time() - start
        
        # Time SciPy
        start = time.time()
        tck_scipy = interpolate.bisplrep(x_flat, y_flat, z_flat, s=0.01)
        time_scipy = time.time() - start
        
        print(f"  Fitting:")
        print(f"    FastSpline: {time_fast:.3f}s")
        print(f"    SciPy:      {time_scipy:.3f}s")
        print(f"    Speedup:    {time_scipy/time_fast:.1f}x")
        
        # Benchmark evaluation
        x_eval = np.linspace(0.5, 5.5, 100)
        y_eval = np.linspace(0.5, 5.5, 100)
        
        # Warm up
        _ = bisplev(x_eval, y_eval, tck_fast)
        _ = interpolate.bisplev(x_eval, y_eval, tck_scipy)
        
        # Time evaluation
        start = time.time()
        z_fast = bisplev(x_eval, y_eval, tck_fast)
        eval_time_fast = time.time() - start
        
        start = time.time()
        z_scipy = interpolate.bisplev(x_eval, y_eval, tck_scipy)
        eval_time_scipy = time.time() - start
        
        print(f"  Evaluation (100x100 grid):")
        print(f"    FastSpline: {eval_time_fast:.3f}s")
        print(f"    SciPy:      {eval_time_scipy:.3f}s")
        print(f"    Speedup:    {eval_time_scipy/eval_time_fast:.1f}x")
        
        # Check accuracy
        error = np.abs(z_fast - z_scipy).mean()
        print(f"  Mean difference: {error:.2e}")


def benchmark_interpolation_accuracy():
    """Test interpolation accuracy."""
    print("\n\nInterpolation Accuracy Test")
    print("=" * 50)
    
    # Test on known function
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    
    # Polynomial that splines can represent exactly
    z = 1 + 2*xx + 3*yy + 4*xx*yy + 5*xx**2 + 6*yy**2
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    # Fit with both methods
    tck_fast = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    tck_scipy = interpolate.bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    
    # Evaluate at test points
    x_test = np.linspace(0.1, 0.9, 20)
    y_test = np.linspace(0.1, 0.9, 20)
    xx_test, yy_test = np.meshgrid(x_test, y_test)
    z_true = 1 + 2*xx_test + 3*yy_test + 4*xx_test*yy_test + 5*xx_test**2 + 6*yy_test**2
    
    z_fast = bisplev(x_test, y_test, tck_fast)
    z_scipy = interpolate.bisplev(x_test, y_test, tck_scipy)
    
    error_fast = np.abs(z_fast - z_true).max()
    error_scipy = np.abs(z_scipy - z_true).max()
    
    print(f"Maximum error on polynomial surface:")
    print(f"  FastSpline: {error_fast:.2e}")
    print(f"  SciPy:      {error_scipy:.2e}")


if __name__ == "__main__":
    benchmark_bisplrep()
    benchmark_interpolation_accuracy()