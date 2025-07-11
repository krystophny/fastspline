#!/usr/bin/env python3
"""
Compare runtime performance of scipy bisplrep/bisplev vs direct f2py wrapper calls.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import dfitpack

def generate_test_data(n_points):
    """Generate synthetic 2D scattered data for testing."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.sin(np.sqrt(x**2 + y**2)) + 0.1 * np.random.randn(n_points)
    return x, y, z

def bisplrep_scipy(x, y, z, kx=3, ky=3, s=0):
    """Standard scipy bisplrep call."""
    return bisplrep(x, y, z, kx=kx, ky=ky, s=s)

def bisplrep_f2py(x, y, z, kx=3, ky=3, s=0):
    """Direct f2py wrapper call using dfitpack.surfit."""
    m = len(x)
    w = np.ones(m, dtype=np.float64)
    
    # Determine bounding box
    xb, xe = x.min(), x.max()
    yb, ye = y.min(), y.max()
    
    # Estimate knot counts
    nxest = int(kx + np.sqrt(m/2)) + 4
    nyest = int(ky + np.sqrt(m/2)) + 4
    
    # Set task and eps
    task = 0
    eps = 1e-16
    
    # Initialize arrays for knots
    tx = np.zeros(nxest, dtype=np.float64)
    ty = np.zeros(nyest, dtype=np.float64)
    
    # Calculate required workspace
    u = nxest - kx - 1
    v = nyest - ky - 1
    km = max(kx, ky) + 1
    ne = max(nxest, nyest)
    bx = kx*v + ky + 1
    by = ky*u + kx + 1
    b1 = bx if bx > by else by
    b2 = bx + v - ky if bx > by else by + u - kx
    
    lwrk1 = u*v*(2 + b1 + b2) + 2*(u + v + km*(m + ne) + ne - kx - ky) + b2 + 1
    lwrk2 = u*v*(b2 + 1) + b2
    
    # Call the Fortran routine directly using dfitpack
    nx, tx, ny, ty, c, fp, wrk1, ier = dfitpack.surfit(
        task, x, y, z, w, xb, xe, yb, ye, kx, ky, s, nxest, nyest, 
        eps, lwrk1, lwrk2
    )
    
    if ier > 10:
        raise ValueError(f"Error in surfit: ier={ier}")
    
    # Trim arrays to actual size
    tx = tx[:nx]
    ty = ty[:ny]
    c = c[:(nx-kx-1)*(ny-ky-1)]
    
    return [tx, ty, c, kx, ky]

def bisplev_scipy(x, y, tck):
    """Standard scipy bisplev call."""
    return bisplev(x, y, tck)

def bisplev_f2py(x, y, tck):
    """Direct f2py wrapper call mimicking bisplev."""
    tx, ty, c, kx, ky = tck
    
    # Ensure arrays are properly formatted
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    # Call bispev directly
    z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
    
    if ier != 0:
        raise ValueError(f"Error in bispev: ier={ier}")
    
    return z

def benchmark_function(func, *args, n_runs=10):
    """Benchmark a function with multiple runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times), result

def main():
    # Test with different data sizes
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    
    # Storage for results
    bisplrep_scipy_times = []
    bisplrep_f2py_times = []
    bisplev_scipy_times = []
    bisplev_f2py_times = []
    
    bisplrep_scipy_stds = []
    bisplrep_f2py_stds = []
    bisplev_scipy_stds = []
    bisplev_f2py_stds = []
    
    print("Benchmarking bisplrep and bisplev...")
    print("-" * 60)
    
    for n_points in sizes:
        print(f"\nTesting with {n_points} points...")
        
        # Generate test data
        x, y, z = generate_test_data(n_points)
        
        # Benchmark bisplrep
        mean_time_scipy, std_time_scipy, tck_scipy = benchmark_function(
            bisplrep_scipy, x, y, z, n_runs=5
        )
        mean_time_f2py, std_time_f2py, tck_f2py = benchmark_function(
            bisplrep_f2py, x, y, z, n_runs=5
        )
        
        bisplrep_scipy_times.append(mean_time_scipy)
        bisplrep_f2py_times.append(mean_time_f2py)
        bisplrep_scipy_stds.append(std_time_scipy)
        bisplrep_f2py_stds.append(std_time_f2py)
        
        print(f"  bisplrep scipy: {mean_time_scipy:.4f} ± {std_time_scipy:.4f} s")
        print(f"  bisplrep f2py:  {mean_time_f2py:.4f} ± {std_time_f2py:.4f} s")
        print(f"  Speedup: {mean_time_scipy/mean_time_f2py:.2f}x")
        
        # Generate evaluation grid
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        
        # Benchmark bisplev
        mean_time_scipy, std_time_scipy, zi_scipy = benchmark_function(
            bisplev_scipy, xi, yi, tck_scipy, n_runs=10
        )
        mean_time_f2py, std_time_f2py, zi_f2py = benchmark_function(
            bisplev_f2py, xi, yi, tck_f2py, n_runs=10
        )
        
        bisplev_scipy_times.append(mean_time_scipy)
        bisplev_f2py_times.append(mean_time_f2py)
        bisplev_scipy_stds.append(std_time_scipy)
        bisplev_f2py_stds.append(std_time_f2py)
        
        print(f"  bisplev scipy: {mean_time_scipy:.4f} ± {std_time_scipy:.4f} s")
        print(f"  bisplev f2py:  {mean_time_f2py:.4f} ± {std_time_f2py:.4f} s")
        print(f"  Speedup: {mean_time_scipy/mean_time_f2py:.2f}x")
        
        # Verify results match
        if np.allclose(zi_scipy, zi_f2py, rtol=1e-10):
            print("  ✓ Results match!")
        else:
            max_diff = np.max(np.abs(zi_scipy - zi_f2py))
            print(f"  ✗ Results differ! Max difference: {max_diff}")
    
    # Create performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot bisplrep performance
    ax1.errorbar(sizes, bisplrep_scipy_times, yerr=bisplrep_scipy_stds, 
                 marker='o', label='scipy interface', capsize=5)
    ax1.errorbar(sizes, bisplrep_f2py_times, yerr=bisplrep_f2py_stds,
                 marker='s', label='f2py direct', capsize=5)
    ax1.set_xlabel('Number of data points')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('bisplrep Performance Comparison')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot bisplev performance
    ax2.errorbar(sizes, bisplev_scipy_times, yerr=bisplev_scipy_stds,
                 marker='o', label='scipy interface', capsize=5)
    ax2.errorbar(sizes, bisplev_f2py_times, yerr=bisplev_f2py_stds,
                 marker='s', label='f2py direct', capsize=5)
    ax2.set_xlabel('Number of original data points')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('bisplev Performance Comparison\n(50x50 evaluation grid)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('scipy_vs_f2py_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'scipy_vs_f2py_performance.png'")
    
    # Calculate and display average speedups
    avg_bisplrep_speedup = np.mean(np.array(bisplrep_scipy_times) / np.array(bisplrep_f2py_times))
    avg_bisplev_speedup = np.mean(np.array(bisplev_scipy_times) / np.array(bisplev_f2py_times))
    
    print(f"\nAverage speedups:")
    print(f"  bisplrep: {avg_bisplrep_speedup:.2f}x")
    print(f"  bisplev:  {avg_bisplev_speedup:.2f}x")

if __name__ == "__main__":
    main()