"""
Generate performance scaling plots comparing scipy, Fortran wrapper, and Numba cfunc.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import ctypes
import sys
from scipy.interpolate import bisplrep, bisplev
from bispev_ctypes import bispev as bispev_fortran
sys.path.insert(0, 'numba_implementation')
from bispev_numba import bispev_cfunc_address
import warnings
warnings.filterwarnings('ignore')


def create_bispev_numba_ctypes():
    """Create ctypes wrapper for Numba bispev."""
    return ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    )(bispev_cfunc_address)


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
    fortran_times = []
    numba_times = []
    
    # Create Numba wrapper
    bispev_numba = create_bispev_numba_ctypes()
    
    print("Running benchmarks...")
    for n_eval in eval_sizes:
        print(f"  Eval grid size: {n_eval}x{n_eval}")
        
        # Generate test data
        n_data = 30
        x_data = np.linspace(0, 1, n_data)
        y_data = np.linspace(0, 1, n_data)
        x_grid, y_grid = np.meshgrid(x_data, y_data)
        z_data = np.sin(2*np.pi*x_grid) * np.cos(2*np.pi*y_grid)
        
        tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01, quiet=1)
        tx, ty, c, kx, ky = tck
        
        # Evaluation points
        x_eval = np.linspace(0.05, 0.95, n_eval)
        y_eval = np.linspace(0.05, 0.95, n_eval)
        
        # Ensure arrays are contiguous
        tx = np.ascontiguousarray(tx, dtype=np.float64)
        ty = np.ascontiguousarray(ty, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        x_eval = np.ascontiguousarray(x_eval, dtype=np.float64)
        y_eval = np.ascontiguousarray(y_eval, dtype=np.float64)
        
        # Benchmark scipy
        times = benchmark_variant(bisplev, x_eval, y_eval, tck)
        scipy_times.append(np.median(times))
        
        # Benchmark Fortran wrapper
        times = benchmark_variant(bispev_fortran, tx, ty, c, kx, ky, x_eval, y_eval)
        fortran_times.append(np.median(times))
        
        # Benchmark Numba cfunc
        def run_numba():
            mx = len(x_eval)
            my = len(y_eval)
            z = np.zeros(mx * my, dtype=np.float64)
            lwrk = mx * (kx + 1) + my * (ky + 1)
            wrk = np.zeros(lwrk, dtype=np.float64)
            kwrk = mx + my
            iwrk = np.zeros(kwrk, dtype=np.int32)
            ier = np.array([0], dtype=np.int32)
            
            bispev_numba(
                tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(tx),
                ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(ty),
                c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                kx, ky,
                x_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
                y_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
                z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
                iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), kwrk,
                ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )
        
        times = benchmark_variant(run_numba)
        numba_times.append(np.median(times))
    
    return eval_sizes, scipy_times, fortran_times, numba_times


def plot_results(eval_sizes, scipy_times, fortran_times, numba_times):
    """Create performance scaling plots."""
    # Convert to arrays and to milliseconds
    eval_sizes = np.array(eval_sizes)
    scipy_times = np.array(scipy_times) * 1000
    fortran_times = np.array(fortran_times) * 1000
    numba_times = np.array(numba_times) * 1000
    
    # Total number of evaluation points
    n_points = eval_sizes ** 2
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Absolute times
    ax1.loglog(n_points, scipy_times, 'o-', label='scipy.bisplev', linewidth=2, markersize=8)
    ax1.loglog(n_points, fortran_times, 's-', label='Fortran wrapper', linewidth=2, markersize=8)
    ax1.loglog(n_points, numba_times, '^-', label='Numba cfunc', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of evaluation points')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Absolute Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative performance (speedup/slowdown)
    slowdown_fortran = fortran_times / scipy_times
    slowdown_numba = numba_times / scipy_times
    
    ax2.semilogx(n_points, slowdown_fortran, 's-', label='Fortran wrapper', linewidth=2, markersize=8)
    ax2.semilogx(n_points, slowdown_numba, '^-', label='Numba cfunc', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='scipy baseline')
    ax2.set_xlabel('Number of evaluation points')
    ax2.set_ylabel('Relative time vs scipy')
    ax2.set_title('Performance Relative to scipy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 8)
    
    # Plot 3: Overhead percentage
    overhead_fortran = (fortran_times - scipy_times) / scipy_times * 100
    overhead_numba = (numba_times - scipy_times) / scipy_times * 100
    
    ax3.semilogx(n_points, overhead_fortran, 's-', label='Fortran wrapper', linewidth=2, markersize=8)
    ax3.semilogx(n_points, overhead_numba, '^-', label='Numba cfunc', linewidth=2, markersize=8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of evaluation points')
    ax3.set_ylabel('Overhead (%)')
    ax3.set_title('Overhead vs scipy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numba_performance_scaling.png', dpi=150, bbox_inches='tight')
    plt.savefig('numba_performance_scaling.pdf', bbox_inches='tight')
    print("\nPlots saved as numba_performance_scaling.png and .pdf")
    
    # Create a second plot comparing Fortran vs Numba directly
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Direct comparison: Numba vs Fortran
    ratio = numba_times / fortran_times
    ax4.semilogx(n_points, ratio, 'o-', color='purple', linewidth=2, markersize=8)
    ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Number of evaluation points')
    ax4.set_ylabel('Numba time / Fortran time')
    ax4.set_title('Numba Performance Relative to Fortran')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.8, 1.4)
    
    # Time per point
    time_per_point_scipy = scipy_times / n_points * 1e6  # microseconds
    time_per_point_fortran = fortran_times / n_points * 1e6
    time_per_point_numba = numba_times / n_points * 1e6
    
    ax5.loglog(n_points, time_per_point_scipy, 'o-', label='scipy.bisplev', linewidth=2, markersize=8)
    ax5.loglog(n_points, time_per_point_fortran, 's-', label='Fortran wrapper', linewidth=2, markersize=8)
    ax5.loglog(n_points, time_per_point_numba, '^-', label='Numba cfunc', linewidth=2, markersize=8)
    ax5.set_xlabel('Number of evaluation points')
    ax5.set_ylabel('Time per point (μs)')
    ax5.set_title('Amortized Cost per Evaluation Point')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numba_fortran_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('numba_fortran_comparison.pdf', bbox_inches='tight')
    print("Comparison plots saved as numba_fortran_comparison.png and .pdf")
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print("-" * 70)
    print(f"{'Grid Size':<12} {'scipy (ms)':<12} {'Fortran (ms)':<15} {'Numba (ms)':<12} {'Numba/Fortran':<15}")
    print("-" * 70)
    for i, size in enumerate(eval_sizes):
        ratio = numba_times[i] / fortran_times[i]
        print(f"{size}x{size:<9} {scipy_times[i]:<12.3f} {fortran_times[i]:<15.3f} {numba_times[i]:<12.3f} {ratio:<15.3f}")


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
    eval_sizes, scipy_times, fortran_times, numba_times = run_benchmarks()
    plot_results(eval_sizes, scipy_times, fortran_times, numba_times)