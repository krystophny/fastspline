#!/usr/bin/env python3
"""
Compare runtime performance of scipy bisplrep/bisplev vs direct f2py wrapper calls.
This version uses the internal _fitpack module directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import bisplrep, bisplev

# Import the internal fitpack module
try:
    from scipy.interpolate import _fitpack
except ImportError:
    print("Cannot import _fitpack directly")
    _fitpack = None

def generate_test_data(n_points):
    """Generate synthetic 2D scattered data for testing."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.sin(np.sqrt(x**2 + y**2)) + 0.1 * np.random.randn(n_points)
    return x, y, z

def bisplrep_scipy(x, y, z, kx=3, ky=3, s=None):
    """Standard scipy bisplrep call."""
    if s is None:
        # Use automatic smoothing
        s = len(x)
    return bisplrep(x, y, z, kx=kx, ky=ky, s=s)

def bisplev_scipy(x, y, tck):
    """Standard scipy bisplev call."""
    return bisplev(x, y, tck)

def bispev_direct(tx, ty, c, kx, ky, x, y):
    """Direct call to bispev f2py wrapper."""
    if _fitpack is None:
        return None
        
    # Ensure proper array formats
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    x = np.atleast_1d(x).astype(np.float64)
    y = np.atleast_1d(y).astype(np.float64)
    
    # Calculate dimensions
    nx = len(tx)
    ny = len(ty)
    
    # Prepare work array
    lwrk = nx + ny + 2*(x.size + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    
    # Call the f2py wrapper directly
    z, ier = _fitpack._bispev(tx, ty, c, kx, ky, x, y, wrk)
    
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
    sizes = [100, 500, 1000, 2000, 5000]
    
    # Storage for results
    bisplrep_times = []
    bisplev_scipy_times = []
    bisplev_direct_times = []
    
    bisplrep_stds = []
    bisplev_scipy_stds = []
    bisplev_direct_stds = []
    
    print("Benchmarking bisplrep and bisplev...")
    print("-" * 60)
    
    # First check if we can access _fitpack
    if _fitpack is None:
        print("Cannot access _fitpack module directly.")
        print("Running alternative comparison...")
        
    for n_points in sizes:
        print(f"\nTesting with {n_points} points...")
        
        # Generate test data
        x, y, z = generate_test_data(n_points)
        
        # Benchmark bisplrep (only scipy version available)
        mean_time, std_time, tck = benchmark_function(
            bisplrep_scipy, x, y, z, n_runs=5
        )
        
        bisplrep_times.append(mean_time)
        bisplrep_stds.append(std_time)
        
        print(f"  bisplrep scipy: {mean_time:.4f} ± {std_time:.4f} s")
        
        # Generate evaluation grid
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        
        # Benchmark bisplev scipy version
        mean_time_scipy, std_time_scipy, zi_scipy = benchmark_function(
            bisplev_scipy, xi, yi, tck, n_runs=10
        )
        
        bisplev_scipy_times.append(mean_time_scipy)
        bisplev_scipy_stds.append(std_time_scipy)
        
        print(f"  bisplev scipy: {mean_time_scipy:.4f} ± {std_time_scipy:.4f} s")
        
        # Try direct bispev if available
        if _fitpack is not None and hasattr(_fitpack, '_bispev'):
            tx, ty, c, kx, ky = tck
            mean_time_direct, std_time_direct, zi_direct = benchmark_function(
                bispev_direct, tx, ty, c, kx, ky, xi, yi, n_runs=10
            )
            
            bisplev_direct_times.append(mean_time_direct)
            bisplev_direct_stds.append(std_time_direct)
            
            print(f"  bisplev f2py direct: {mean_time_direct:.4f} ± {std_time_direct:.4f} s")
            print(f"  Speedup: {mean_time_scipy/mean_time_direct:.2f}x")
            
            # Verify results match
            if zi_direct is not None and np.allclose(zi_scipy, zi_direct, rtol=1e-10):
                print("  ✓ Results match!")
            else:
                if zi_direct is not None:
                    max_diff = np.max(np.abs(zi_scipy - zi_direct))
                    print(f"  ✗ Results differ! Max difference: {max_diff}")
        else:
            # Use scipy times as placeholder
            bisplev_direct_times.append(mean_time_scipy)
            bisplev_direct_stds.append(std_time_scipy)
    
    # Create performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot bisplrep performance
    ax1.errorbar(sizes, bisplrep_times, yerr=bisplrep_stds, 
                 marker='o', label='bisplrep (scipy)', capsize=5)
    ax1.set_xlabel('Number of data points')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('bisplrep Performance')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot bisplev performance comparison
    ax2.errorbar(sizes, bisplev_scipy_times, yerr=bisplev_scipy_stds,
                 marker='o', label='bisplev (scipy interface)', capsize=5)
    if _fitpack is not None and hasattr(_fitpack, '_bispev'):
        ax2.errorbar(sizes, bisplev_direct_times, yerr=bisplev_direct_stds,
                     marker='s', label='bispev (f2py direct)', capsize=5)
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
    
    # Calculate overhead analysis
    print("\n" + "="*60)
    print("Performance Analysis Summary:")
    print("="*60)
    
    if _fitpack is not None and hasattr(_fitpack, '_bispev') and len(bisplev_direct_times) == len(sizes):
        # Calculate average overhead
        overheads = [scipy_t / direct_t for scipy_t, direct_t in 
                     zip(bisplev_scipy_times, bisplev_direct_times)]
        avg_overhead = np.mean(overheads)
        
        print(f"\nAverage scipy interface overhead for bisplev: {avg_overhead:.2f}x")
        print("\nDetailed overhead by data size:")
        for size, overhead in zip(sizes, overheads):
            print(f"  {size} points: {overhead:.2f}x overhead")
    else:
        print("\nDirect f2py wrapper comparison not available.")
        print("This might be due to changes in scipy's internal structure.")
        
    # Estimate the overhead from function call analysis
    print("\n" + "-"*60)
    print("Note: The scipy interface adds overhead through:")
    print("- Input validation and type checking")
    print("- Array conversion and ensuring correct dtypes")
    print("- Error handling and result formatting")
    print("- Python function call overhead")
    print("\nDirect f2py calls bypass most of this overhead.")

if __name__ == "__main__":
    main()