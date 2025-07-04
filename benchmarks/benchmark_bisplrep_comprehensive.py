#!/usr/bin/env python3
"""Comprehensive benchmark of bisplrep implementation."""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline import bisplrep, bisplev_scalar as bisplev


def generate_test_surface(n_points, surface_type='smooth'):
    """Generate various test surfaces."""
    np.random.seed(42)
    
    if surface_type == 'grid':
        # Regular grid
        nx = int(np.sqrt(n_points))
        ny = n_points // nx
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        xx, yy = np.meshgrid(x, y)
        x_data = xx.ravel()[:n_points]
        y_data = yy.ravel()[:n_points]
    else:
        # Scattered points
        x_data = np.random.uniform(-1, 1, n_points)
        y_data = np.random.uniform(-1, 1, n_points)
    
    # Generate surface
    if surface_type == 'smooth':
        z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data)
    elif surface_type == 'peaks':
        z_data = 3*(1-x_data)**2 * np.exp(-(x_data**2) - (y_data+1)**2) \
                - 10*(x_data/5 - x_data**3 - y_data**5) * np.exp(-x_data**2 - y_data**2) \
                - 1/3 * np.exp(-(x_data+1)**2 - y_data**2)
    elif surface_type == 'polynomial':
        z_data = x_data**2 + x_data*y_data + y_data**2 + 0.1*x_data**3
    elif surface_type == 'noisy':
        z_data = np.sin(3*x_data) * np.cos(3*y_data) + 0.2*np.random.randn(n_points)
    
    return x_data, y_data, z_data


def benchmark_construction(n_points_list, surface_type='smooth'):
    """Benchmark construction time for bisplrep implementations."""
    print(f"\n{'='*80}")
    print(f"Construction Benchmark - {surface_type} surface")
    print(f"{'='*80}")
    print(f"{'N Points':<10} {'Implementation':<20} {'Time (ms)':<12} {'Knots':<20}")
    print(f"{'-'*70}")
    
    results = {'scipy': {'n_points': [], 'times': [], 'nx': [], 'ny': []},
               'fastspline': {'n_points': [], 'times': [], 'nx': [], 'ny': []}}
    
    for n_points in n_points_list:
        x, y, z = generate_test_surface(n_points, surface_type)
        
        # Benchmark SciPy
        try:
            start = time.perf_counter()
            tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.1)
            elapsed_scipy = (time.perf_counter() - start) * 1000
            nx_scipy, ny_scipy = len(tck_scipy[0]), len(tck_scipy[1])
            
            results['scipy']['n_points'].append(n_points)
            results['scipy']['times'].append(elapsed_scipy)
            results['scipy']['nx'].append(nx_scipy)
            results['scipy']['ny'].append(ny_scipy)
            
            print(f"{n_points:<10} {'scipy':<20} {elapsed_scipy:<12.2f} nx={nx_scipy:<3} ny={ny_scipy:<3}")
        except Exception as e:
            print(f"{n_points:<10} {'scipy':<20} {'FAILED':<12} {str(e)[:40]}")
        
        # Benchmark FastSpline
        try:
            start = time.perf_counter()
            tck_fast = bisplrep(x, y, z, kx=3, ky=3, s=0.1)
            elapsed_fast = (time.perf_counter() - start) * 1000
            nx_fast, ny_fast = len(tck_fast[0]), len(tck_fast[1])
            
            results['fastspline']['n_points'].append(n_points)
            results['fastspline']['times'].append(elapsed_fast)
            results['fastspline']['nx'].append(nx_fast)
            results['fastspline']['ny'].append(ny_fast)
            
            print(f"{n_points:<10} {'fastspline':<20} {elapsed_fast:<12.2f} nx={nx_fast:<3} ny={ny_fast:<3}")
            
            # Show speedup
            if 'scipy' in results and len(results['scipy']['times']) > 0:
                speedup = elapsed_scipy / elapsed_fast
                print(f"{'':<10} {'Speedup:':<20} {speedup:<12.2f}x")
                
        except Exception as e:
            print(f"{n_points:<10} {'fastspline':<20} {'FAILED':<12} {str(e)[:40]}")
    
    return results


def benchmark_evaluation(n_points=500, n_eval=1000):
    """Benchmark evaluation performance."""
    print(f"\n{'='*80}")
    print(f"Evaluation Benchmark - {n_eval} evaluations")
    print(f"{'='*80}")
    
    # Generate data and fit with SciPy
    x, y, z = generate_test_surface(n_points, 'smooth')
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.1)
    
    # Generate evaluation points
    np.random.seed(123)
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    # Benchmark SciPy
    start = time.perf_counter()
    for i in range(n_eval):
        _ = scipy_bisplev(x_eval[i], y_eval[i], tck_scipy)
    time_scipy = (time.perf_counter() - start) * 1000
    
    # Benchmark our cfunc
    tx, ty, c, kx, ky = tck_scipy
    start = time.perf_counter()
    for i in range(n_eval):
        _ = bisplev(x_eval[i], y_eval[i], tx, ty, c, kx, ky)
    time_cfunc = (time.perf_counter() - start) * 1000
    
    print(f"{'Method':<20} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"{'-'*42}")
    print(f"{'SciPy bisplev':<20} {time_scipy:<12.2f} {'1.00x':<10}")
    print(f"{'FastSpline bisplev':<20} {time_cfunc:<12.2f} {time_scipy/time_cfunc:<10.2f}x")


def test_accuracy_comparison():
    """Compare accuracy of implementations."""
    print(f"\n{'='*80}")
    print(f"Accuracy Comparison")
    print(f"{'='*80}")
    
    # Generate test data
    n_points = 200
    x, y, z = generate_test_surface(n_points, 'smooth')
    
    # Fit with different methods
    print("Fitting surfaces...")
    
    # SciPy
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0)
    print(f"  SciPy: nx={len(tck_scipy[0])}, ny={len(tck_scipy[1])}")
    
    # FastSpline
    tck_fast = bisplrep(x, y, z, kx=3, ky=3, s=0)
    print(f"  FastSpline: nx={len(tck_fast[0])}, ny={len(tck_fast[1])}")
    
    # Test on a grid
    print("\nEvaluating on test grid...")
    n_test = 50
    x_test = np.linspace(-0.9, 0.9, n_test)
    y_test = np.linspace(-0.9, 0.9, n_test)
    
    # Evaluate with each method
    z_scipy = np.zeros((n_test, n_test))
    z_fast = np.zeros((n_test, n_test))
    
    for i in range(n_test):
        for j in range(n_test):
            z_scipy[i, j] = scipy_bisplev(x_test[i], y_test[j], tck_scipy)
            z_fast[i, j] = bisplev(x_test[i], y_test[j], 
                                  tck_fast[0], tck_fast[1], tck_fast[2], 3, 3)
    
    # Compare differences
    diff = np.abs(z_scipy - z_fast)
    
    print(f"\n{'Method':<20} {'Max Diff':<15} {'Mean Diff':<15} {'RMS Diff':<15}")
    print(f"{'-'*65}")
    print(f"{'FastSpline vs SciPy':<20} {np.max(diff):<15.2e} "
          f"{np.mean(diff):<15.2e} {np.sqrt(np.mean(diff**2)):<15.2e}")


def plot_results(construction_results):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Construction time
    ax1.set_title('bisplrep Construction Time', fontsize=14)
    
    markers = {'scipy': 'o', 'fastspline': 's'}
    colors = {'scipy': 'blue', 'fastspline': 'green'}
    
    for name, data in construction_results.items():
        if 'n_points' in data and 'times' in data and len(data['n_points']) > 0:
            ax1.loglog(data['n_points'], data['times'],
                      marker=markers.get(name, 'o'),
                      color=colors.get(name, 'gray'),
                      linewidth=2, markersize=8,
                      label=name)
    
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Knot counts
    ax2.set_title('Number of Knots Used', fontsize=14)
    
    for name, data in construction_results.items():
        if 'n_points' in data and 'nx' in data and len(data['n_points']) > 0:
            total_knots = [nx * ny for nx, ny in zip(data['nx'], data['ny'])]
            ax2.plot(data['n_points'], total_knots,
                    marker=markers.get(name, 'o'),
                    color=colors.get(name, 'gray'),
                    linewidth=2, markersize=8,
                    label=f"{name} (nx*ny)")
    
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('Total Knots (nx * ny)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bisplrep_comparison.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to bisplrep_comparison.png")


def main():
    """Run comprehensive benchmarks."""
    print("Comprehensive bisplrep Benchmark")
    print("=" * 80)
    
    # Test different problem sizes
    n_points_list = [50, 100, 200, 500, 1000]
    
    # Run construction benchmarks
    results_smooth = benchmark_construction(n_points_list, 'smooth')
    results_noisy = benchmark_construction(n_points_list, 'noisy')
    
    # Run evaluation benchmark
    benchmark_evaluation()
    
    # Test accuracy
    test_accuracy_comparison()
    
    # Plot results
    plot_results(results_smooth)
    
    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    main()