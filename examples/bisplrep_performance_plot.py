"""Visualize bisplrep performance comparison between FastSpline and SciPy."""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline import bisplrep, bisplev


def benchmark_sizes():
    """Benchmark different grid sizes and return timing data."""
    sizes = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    n_points = [s*s for s in sizes]
    
    times_fast_fit = []
    times_scipy_fit = []
    times_fast_eval = []
    times_scipy_eval = []
    speedups_fit = []
    speedups_eval = []
    
    # Evaluation grid
    x_eval = np.linspace(0, 2*np.pi, 100)
    y_eval = np.linspace(0, 2*np.pi, 100)
    
    for n in sizes:
        print(f"Benchmarking {n}x{n} grid...")
        
        # Generate test data
        x = np.linspace(0, 2*np.pi, n)
        y = np.linspace(0, 2*np.pi, n)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(xx) * np.cos(yy)
        
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        z_flat = z.ravel()
        
        # Time fitting - use more iterations for small sizes
        n_iter = max(1, 100 // n)
        
        # FastSpline fitting
        start = time.time()
        for _ in range(n_iter):
            tck_fast = bisplrep(x_flat, y_flat, z_flat, s=0)
        time_fast_fit = (time.time() - start) / n_iter
        times_fast_fit.append(time_fast_fit)
        
        # SciPy fitting
        start = time.time()
        for _ in range(n_iter):
            tck_scipy = interpolate.bisplrep(x_flat, y_flat, z_flat, s=0)
        time_scipy_fit = (time.time() - start) / n_iter
        times_scipy_fit.append(time_scipy_fit)
        
        speedups_fit.append(time_scipy_fit / time_fast_fit if time_fast_fit > 0 else 0)
        
        # Time evaluation
        n_eval_iter = 10
        
        # FastSpline evaluation
        start = time.time()
        for _ in range(n_eval_iter):
            z_fast = bisplev(x_eval, y_eval, tck_fast)
        time_fast_eval = (time.time() - start) / n_eval_iter
        times_fast_eval.append(time_fast_eval)
        
        # SciPy evaluation
        start = time.time()
        for _ in range(n_eval_iter):
            z_scipy = interpolate.bisplev(x_eval, y_eval, tck_scipy)
        time_scipy_eval = (time.time() - start) / n_eval_iter
        times_scipy_eval.append(time_scipy_eval)
        
        speedups_eval.append(time_scipy_eval / time_fast_eval if time_fast_eval > 0 else 0)
    
    return {
        'sizes': sizes,
        'n_points': n_points,
        'times_fast_fit': times_fast_fit,
        'times_scipy_fit': times_scipy_fit,
        'times_fast_eval': times_fast_eval,
        'times_scipy_eval': times_scipy_eval,
        'speedups_fit': speedups_fit,
        'speedups_eval': speedups_eval
    }


def plot_performance(data):
    """Create performance visualization plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Fitting time comparison
    ax1.loglog(data['n_points'], data['times_fast_fit'], 'b-o', label='FastSpline', markersize=6)
    ax1.loglog(data['n_points'], data['times_scipy_fit'], 'r-s', label='SciPy', markersize=6)
    ax1.set_xlabel('Number of data points')
    ax1.set_ylabel('Fitting time (seconds)')
    ax1.set_title('Bisplrep Fitting Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Fitting speedup
    ax2.semilogx(data['n_points'], data['speedups_fit'], 'g-^', markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of data points')
    ax2.set_ylabel('Speedup factor')
    ax2.set_title('FastSpline Fitting Speedup over SciPy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(data['speedups_fit']) * 1.1)
    
    # Add speedup annotations for key points
    for i in [4, 7, 10]:  # 25x25, 50x50, 80x80
        if i < len(data['n_points']):
            ax2.annotate(f'{data["speedups_fit"][i]:.1f}x',
                        xy=(data['n_points'][i], data['speedups_fit'][i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
    
    # Plot 3: Evaluation time comparison
    ax3.loglog(data['n_points'], data['times_fast_eval'], 'b-o', label='FastSpline', markersize=6)
    ax3.loglog(data['n_points'], data['times_scipy_eval'], 'r-s', label='SciPy', markersize=6)
    ax3.set_xlabel('Number of fitting points')
    ax3.set_ylabel('Evaluation time (seconds)')
    ax3.set_title('Bisplev Evaluation Performance (100x100 grid)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Evaluation speedup
    ax4.semilogx(data['n_points'], data['speedups_eval'], 'g-^', markersize=8)
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Number of fitting points')
    ax4.set_ylabel('Speedup factor')
    ax4.set_title('FastSpline Evaluation Speedup over SciPy')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('FastSpline vs SciPy Performance Comparison for 2D B-splines', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig('bisplrep_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print("=" * 50)
    print(f"Fitting speedup range: {min(data['speedups_fit']):.1f}x - {max(data['speedups_fit']):.1f}x")
    print(f"Average fitting speedup: {np.mean(data['speedups_fit']):.1f}x")
    print(f"Evaluation speedup range: {min(data['speedups_eval']):.1f}x - {max(data['speedups_eval']):.1f}x")
    print(f"Average evaluation speedup: {np.mean(data['speedups_eval']):.1f}x")


if __name__ == "__main__":
    print("Running bisplrep performance benchmarks...")
    print("This may take a minute...\n")
    
    data = benchmark_sizes()
    plot_performance(data)