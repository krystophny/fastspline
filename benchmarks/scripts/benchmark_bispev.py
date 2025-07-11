"""
Performance benchmarks comparing C wrapper with scipy.interpolate.bisplev.
"""
import numpy as np
import time
from scipy.interpolate import bisplrep, bisplev
from bispev_ctypes import bispev as bispev_c


def generate_test_data(n_data=20, n_eval=50):
    """Generate test data for benchmarking."""
    # Create data grid
    x_data = np.linspace(0, 1, n_data)
    y_data = np.linspace(0, 1, n_data)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = np.sin(2*np.pi*x_grid) * np.cos(2*np.pi*y_grid)
    
    # Fit spline
    tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01)
    
    # Create evaluation points
    x_eval = np.linspace(0.05, 0.95, n_eval)
    y_eval = np.linspace(0.05, 0.95, n_eval)
    
    return tck, x_eval, y_eval


def benchmark_scipy(tck, x_eval, y_eval, n_runs=100):
    """Benchmark scipy's bisplev."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        z = bisplev(x_eval, y_eval, tck)
        end = time.perf_counter()
        times.append(end - start)
    return np.array(times)


def benchmark_c_wrapper(tck, x_eval, y_eval, n_runs=100):
    """Benchmark C wrapper."""
    tx, ty, c, kx, ky = tck
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        z = bispev_c(tx, ty, c, kx, ky, x_eval, y_eval)
        end = time.perf_counter()
        times.append(end - start)
    return np.array(times)


def benchmark_c_wrapper_preallocated(tck, x_eval, y_eval, n_runs=100):
    """Benchmark C wrapper with pre-allocated arrays."""
    tx, ty, c, kx, ky = tck
    
    # Pre-allocate arrays
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    x_eval = np.ascontiguousarray(x_eval, dtype=np.float64)
    y_eval = np.ascontiguousarray(y_eval, dtype=np.float64)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        z = bispev_c(tx, ty, c, kx, ky, x_eval, y_eval)
        end = time.perf_counter()
        times.append(end - start)
    return np.array(times)


def print_results(name, times):
    """Print benchmark results."""
    mean_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    print(f"{name:30s}: {mean_time:8.3f} ± {std_time:6.3f} ms (min: {min_time:6.3f} ms)")


def main():
    print("DIERCKX bispev Performance Benchmarks")
    print("=" * 70)
    
    # Test different grid sizes
    grid_sizes = [(10, 10), (20, 20), (50, 50), (100, 100)]
    
    for n_data, n_eval in grid_sizes:
        print(f"\nData grid: {n_data}x{n_data}, Eval grid: {n_eval}x{n_eval}")
        print("-" * 70)
        
        # Generate test data
        tck, x_eval, y_eval = generate_test_data(n_data, n_eval)
        
        # Warm-up runs
        for _ in range(5):
            bisplev(x_eval, y_eval, tck)
            bispev_c(*tck, x_eval, y_eval)
        
        # Run benchmarks
        scipy_times = benchmark_scipy(tck, x_eval, y_eval, n_runs=100)
        c_times = benchmark_c_wrapper(tck, x_eval, y_eval, n_runs=100)
        c_prealloc_times = benchmark_c_wrapper_preallocated(tck, x_eval, y_eval, n_runs=100)
        
        # Print results
        print_results("scipy.interpolate.bisplev", scipy_times)
        print_results("C wrapper (ctypes)", c_times)
        print_results("C wrapper (pre-allocated)", c_prealloc_times)
        
        # Calculate speedup
        scipy_mean = np.mean(scipy_times)
        c_mean = np.mean(c_times)
        c_prealloc_mean = np.mean(c_prealloc_times)
        
        print(f"\nSpeedup vs scipy:")
        print(f"  C wrapper:              {scipy_mean/c_mean:.2f}x")
        print(f"  C wrapper (pre-alloc):  {scipy_mean/c_prealloc_mean:.2f}x")
        
        # Calculate overhead
        overhead = (c_mean - scipy_mean) / scipy_mean * 100
        overhead_prealloc = (c_prealloc_mean - scipy_mean) / scipy_mean * 100
        
        print(f"\nOverhead vs scipy:")
        print(f"  C wrapper:              {overhead:+.1f}%")
        print(f"  C wrapper (pre-alloc):  {overhead_prealloc:+.1f}%")


if __name__ == "__main__":
    main()