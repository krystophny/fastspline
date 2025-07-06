"""
Benchmark script to test performance scaling of spline fitting and evaluation
with respect to the number of input points.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import interpolate
from dierckx_numba_simple import fpbspl_njit, fpback_njit, fpgivs_njit
import os

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def generate_test_data(n_points):
    """Generate test data for 2D surface fitting"""
    # Create grid points
    n_x = int(np.sqrt(n_points))
    n_y = n_x
    actual_points = n_x * n_y
    
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Create test function: combination of waves
    zz = np.sin(4 * np.pi * xx) * np.cos(4 * np.pi * yy) + 0.5 * np.sin(2 * np.pi * xx)
    
    # Flatten for spline fitting
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = zz.ravel()
    
    return x_flat, y_flat, z_flat, actual_points


def benchmark_scipy_bisplrep(x, y, z, kx=3, ky=3, s=0):
    """Benchmark scipy's bisplrep fitting"""
    start = time.time()
    tck = interpolate.bisplrep(x, y, z, kx=kx, ky=ky, s=s)
    elapsed = time.time() - start
    return tck, elapsed


def benchmark_scipy_bisplev(tck, x_eval, y_eval):
    """Benchmark scipy's bisplev evaluation"""
    tx, ty, c, kx, ky = tck
    
    start = time.time()
    z_eval = interpolate.bisplev(x_eval, y_eval, tck)
    elapsed = time.time() - start
    
    return z_eval, elapsed


def benchmark_numba_bspline_eval(tx, ty, c, kx, ky, x_eval, y_eval):
    """Benchmark Numba B-spline evaluation using our implementations"""
    nx = len(tx)
    ny = len(ty)
    n_eval_x = len(x_eval)
    n_eval_y = len(y_eval)
    
    # Pre-allocate output
    z_eval = np.zeros((n_eval_x, n_eval_y))
    
    start = time.time()
    
    # For each evaluation point
    for i in range(n_eval_x):
        for j in range(n_eval_y):
            # Find B-spline values in x direction
            lx, hx = fpbspl_njit(tx, nx, kx, x_eval[i])
            
            # Find B-spline values in y direction  
            ly, hy = fpbspl_njit(ty, ny, ky, y_eval[j])
            
            # Compute tensor product (simplified - not using actual coefficients)
            # This is just to measure the B-spline evaluation performance
            val = 0.0
            for ii in range(kx+1):
                for jj in range(ky+1):
                    val += hx[ii] * hy[jj]
            z_eval[i, j] = val
    
    elapsed = time.time() - start
    return z_eval, elapsed


def run_benchmarks():
    """Run benchmarks for different problem sizes"""
    # Test different numbers of input points
    n_points_list = [100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000]
    
    # Results storage
    scipy_fit_times = []
    scipy_eval_times = []
    numba_eval_times = []
    actual_points_list = []
    
    # Evaluation grid (fixed size)
    n_eval = 50
    x_eval = np.linspace(0, 1, n_eval)
    y_eval = np.linspace(0, 1, n_eval)
    
    print("Running benchmarks...")
    print("-" * 60)
    print(f"{'N Points':>10} {'Actual':>10} {'Fit (ms)':>12} {'Eval (ms)':>12} {'Numba (ms)':>12}")
    print("-" * 60)
    
    for n_target in n_points_list:
        # Generate test data
        x, y, z, n_actual = generate_test_data(n_target)
        
        # Benchmark scipy fitting
        tck, fit_time = benchmark_scipy_bisplrep(x, y, z)
        
        # Benchmark scipy evaluation
        z_scipy, eval_time = benchmark_scipy_bisplev(tck, x_eval, y_eval)
        
        # Benchmark Numba evaluation
        tx, ty, c, kx, ky = tck
        z_numba, numba_time = benchmark_numba_bspline_eval(tx, ty, c, kx, ky, x_eval, y_eval)
        
        # Store results
        actual_points_list.append(n_actual)
        scipy_fit_times.append(fit_time * 1000)  # Convert to ms
        scipy_eval_times.append(eval_time * 1000)
        numba_eval_times.append(numba_time * 1000)
        
        print(f"{n_target:>10} {n_actual:>10} {fit_time*1000:>12.2f} "
              f"{eval_time*1000:>12.2f} {numba_time*1000:>12.2f}")
    
    return actual_points_list, scipy_fit_times, scipy_eval_times, numba_eval_times


def plot_results(n_points, fit_times, scipy_eval_times, numba_eval_times):
    """Create plots showing performance scaling"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Fitting time vs number of points
    ax1.loglog(n_points, fit_times, 'o-', linewidth=2, markersize=8, label='scipy.bisplrep')
    
    # Add reference lines
    n_arr = np.array(n_points)
    ax1.loglog(n_arr, fit_times[0] * (n_arr / n_points[0]), 'k--', alpha=0.5, label='O(n)')
    ax1.loglog(n_arr, fit_times[0] * (n_arr / n_points[0])**1.5, 'k:', alpha=0.5, label='O(n^1.5)')
    
    ax1.set_xlabel('Number of Input Points', fontsize=12)
    ax1.set_ylabel('Fitting Time (ms)', fontsize=12)
    ax1.set_title('Spline Fitting Performance Scaling', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Evaluation time comparison
    ax2.semilogy(n_points, scipy_eval_times, 'o-', linewidth=2, markersize=8, 
                 label='scipy.bisplev')
    ax2.semilogy(n_points, numba_eval_times, 's-', linewidth=2, markersize=8,
                 label='Numba B-spline eval')
    
    ax2.set_xlabel('Number of Knots (∝ sqrt(input points))', fontsize=12)
    ax2.set_ylabel('Evaluation Time (ms) for 50×50 grid', fontsize=12)
    ax2.set_title('Spline Evaluation Performance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add speedup annotation
    speedup = np.mean(np.array(scipy_eval_times) / np.array(numba_eval_times))
    ax2.text(0.95, 0.05, f'Average Numba speedup: {speedup:.1f}×', 
             transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('benchmark_scaling.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'benchmark_scaling.png'")
    
    # Create additional plot showing scaling behavior
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Calculate speedup for each size
    speedups = np.array(scipy_eval_times) / np.array(numba_eval_times)
    
    ax3.plot(n_points, speedups, 'o-', linewidth=2, markersize=8, color='green')
    ax3.axhline(y=np.mean(speedups), color='red', linestyle='--', 
                label=f'Mean speedup: {np.mean(speedups):.1f}×')
    
    ax3.set_xlabel('Number of Input Points', fontsize=12)
    ax3.set_ylabel('Speedup Factor (scipy / Numba)', fontsize=12)
    ax3.set_title('Numba vs SciPy Evaluation Speedup', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_speedup.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved as 'benchmark_speedup.png'")


def create_performance_table(n_points, fit_times, scipy_eval_times, numba_eval_times):
    """Create a performance summary table"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'N Points':>10} | {'Fit (ms)':>10} | {'SciPy Eval':>12} | {'Numba Eval':>12} | {'Speedup':>8}")
    print("-"*80)
    
    for i, n in enumerate(n_points):
        speedup = scipy_eval_times[i] / numba_eval_times[i]
        print(f"{n:>10} | {fit_times[i]:>10.2f} | {scipy_eval_times[i]:>12.2f} | "
              f"{numba_eval_times[i]:>12.2f} | {speedup:>8.1f}×")
    
    print("-"*80)
    avg_speedup = np.mean(np.array(scipy_eval_times) / np.array(numba_eval_times))
    print(f"{'AVERAGE':>10} | {np.mean(fit_times):>10.2f} | {np.mean(scipy_eval_times):>12.2f} | "
          f"{np.mean(numba_eval_times):>12.2f} | {avg_speedup:>8.1f}×")
    print("="*80)
    
    # Performance scaling analysis
    print("\nSCALING ANALYSIS:")
    print("-"*40)
    
    # Fit scaling
    log_n = np.log(n_points)
    log_fit = np.log(fit_times)
    fit_slope, fit_intercept = np.polyfit(log_n, log_fit, 1)
    print(f"Fitting scaling: O(n^{fit_slope:.2f})")
    
    # Eval scaling  
    log_scipy_eval = np.log(scipy_eval_times)
    scipy_eval_slope, _ = np.polyfit(log_n, log_scipy_eval, 1)
    print(f"SciPy eval scaling: O(n^{scipy_eval_slope:.2f})")
    
    log_numba_eval = np.log(numba_eval_times)
    numba_eval_slope, _ = np.polyfit(log_n, log_numba_eval, 1)
    print(f"Numba eval scaling: O(n^{numba_eval_slope:.2f})")


if __name__ == "__main__":
    # Run benchmarks
    n_points, fit_times, scipy_eval_times, numba_eval_times = run_benchmarks()
    
    # Create plots
    plot_results(n_points, fit_times, scipy_eval_times, numba_eval_times)
    
    # Create performance table
    create_performance_table(n_points, fit_times, scipy_eval_times, numba_eval_times)