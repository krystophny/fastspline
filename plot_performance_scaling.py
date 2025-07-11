"""
Generate performance scaling plots comparing scipy, C wrapper, and pre-allocated variants.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import bisplrep, bisplev
from bispev_ctypes import bispev as bispev_c
import warnings
warnings.filterwarnings('ignore')


def generate_test_data(n_data=20):
    """Generate test spline data."""
    x_data = np.linspace(0, 1, n_data)
    y_data = np.linspace(0, 1, n_data)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = np.sin(2*np.pi*x_grid) * np.cos(2*np.pi*y_grid)
    
    tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01, quiet=1)
    return tck


def benchmark_variant(func, *args, n_runs=50, warmup=5):
    """Benchmark a function with warmup runs."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Time
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.array(times)


def run_benchmarks():
    """Run benchmarks for different grid sizes."""
    # Grid sizes to test
    eval_sizes = [5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
    
    # Results storage
    scipy_times = []
    c_wrapper_times = []
    c_prealloc_times = []
    
    # Fixed data grid size
    tck = generate_test_data(n_data=30)
    tx, ty, c, kx, ky = tck
    
    # Pre-allocate arrays for the pre-allocated variant
    tx_pre = np.ascontiguousarray(tx, dtype=np.float64)
    ty_pre = np.ascontiguousarray(ty, dtype=np.float64)
    c_pre = np.ascontiguousarray(c, dtype=np.float64)
    
    print("Running benchmarks...")
    for n_eval in eval_sizes:
        print(f"  Eval grid size: {n_eval}x{n_eval}")
        
        # Generate evaluation points
        x_eval = np.linspace(0.05, 0.95, n_eval)
        y_eval = np.linspace(0.05, 0.95, n_eval)
        x_eval_pre = np.ascontiguousarray(x_eval, dtype=np.float64)
        y_eval_pre = np.ascontiguousarray(y_eval, dtype=np.float64)
        
        # Benchmark scipy
        times = benchmark_variant(bisplev, x_eval, y_eval, tck)
        scipy_times.append(np.median(times))
        
        # Benchmark C wrapper
        times = benchmark_variant(bispev_c, tx, ty, c, kx, ky, x_eval, y_eval)
        c_wrapper_times.append(np.median(times))
        
        # Benchmark C wrapper with pre-allocated arrays
        times = benchmark_variant(bispev_c, tx_pre, ty_pre, c_pre, kx, ky, 
                                x_eval_pre, y_eval_pre)
        c_prealloc_times.append(np.median(times))
    
    return eval_sizes, scipy_times, c_wrapper_times, c_prealloc_times


def plot_results(eval_sizes, scipy_times, c_wrapper_times, c_prealloc_times):
    """Create performance scaling plots."""
    # Convert to arrays and to milliseconds
    eval_sizes = np.array(eval_sizes)
    scipy_times = np.array(scipy_times) * 1000
    c_wrapper_times = np.array(c_wrapper_times) * 1000
    c_prealloc_times = np.array(c_prealloc_times) * 1000
    
    # Total number of evaluation points
    n_points = eval_sizes ** 2
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Absolute times
    ax1.loglog(n_points, scipy_times, 'o-', label='scipy.bisplev', linewidth=2, markersize=8)
    ax1.loglog(n_points, c_wrapper_times, 's-', label='C wrapper (ctypes)', linewidth=2, markersize=8)
    ax1.loglog(n_points, c_prealloc_times, '^-', label='C wrapper (pre-alloc)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of evaluation points')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Absolute Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative performance (speedup)
    speedup_c = scipy_times / c_wrapper_times
    speedup_prealloc = scipy_times / c_prealloc_times
    
    ax2.semilogx(n_points, speedup_c, 's-', label='C wrapper (ctypes)', linewidth=2, markersize=8)
    ax2.semilogx(n_points, speedup_prealloc, '^-', label='C wrapper (pre-alloc)', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of evaluation points')
    ax2.set_ylabel('Speedup vs scipy')
    ax2.set_title('Relative Performance (Speedup)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    # Plot 3: Overhead percentage
    overhead_c = (c_wrapper_times - scipy_times) / scipy_times * 100
    overhead_prealloc = (c_prealloc_times - scipy_times) / scipy_times * 100
    
    ax3.semilogx(n_points, overhead_c, 's-', label='C wrapper (ctypes)', linewidth=2, markersize=8)
    ax3.semilogx(n_points, overhead_prealloc, '^-', label='C wrapper (pre-alloc)', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of evaluation points')
    ax3.set_ylabel('Overhead (%)')
    ax3.set_title('Overhead vs scipy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bispev_performance_scaling.png', dpi=150, bbox_inches='tight')
    plt.savefig('bispev_performance_scaling.pdf', bbox_inches='tight')
    print("\nPlots saved as bispev_performance_scaling.png and .pdf")
    
    # Create a second plot showing time per point
    fig2, ax4 = plt.subplots(figsize=(8, 6))
    
    time_per_point_scipy = scipy_times / n_points * 1e6  # microseconds
    time_per_point_c = c_wrapper_times / n_points * 1e6
    time_per_point_prealloc = c_prealloc_times / n_points * 1e6
    
    ax4.loglog(n_points, time_per_point_scipy, 'o-', label='scipy.bisplev', linewidth=2, markersize=8)
    ax4.loglog(n_points, time_per_point_c, 's-', label='C wrapper (ctypes)', linewidth=2, markersize=8)
    ax4.loglog(n_points, time_per_point_prealloc, '^-', label='C wrapper (pre-alloc)', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of evaluation points')
    ax4.set_ylabel('Time per point (μs)')
    ax4.set_title('Amortized Cost per Evaluation Point')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bispev_time_per_point.png', dpi=150, bbox_inches='tight')
    plt.savefig('bispev_time_per_point.pdf', bbox_inches='tight')
    print("Time per point plots saved as bispev_time_per_point.png and .pdf")
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print("-" * 60)
    print(f"{'Grid Size':<12} {'scipy (ms)':<12} {'C wrap (ms)':<12} {'Overhead %':<12}")
    print("-" * 60)
    for i, size in enumerate(eval_sizes):
        print(f"{size}x{size:<9} {scipy_times[i]:<12.3f} {c_wrapper_times[i]:<12.3f} {overhead_c[i]:<12.1f}")


if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("matplotlib not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    
    # Run benchmarks and create plots
    eval_sizes, scipy_times, c_wrapper_times, c_prealloc_times = run_benchmarks()
    plot_results(eval_sizes, scipy_times, c_wrapper_times, c_prealloc_times)