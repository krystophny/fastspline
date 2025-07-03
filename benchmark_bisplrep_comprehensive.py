#!/usr/bin/env python3
"""Comprehensive benchmark of bisplrep implementations."""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline.bisplrep_cfunc import bisplrep as bisplrep_simple
from fastspline.bisplrep_advanced import bisplrep_advanced, bisplrep_advanced_py
from fastspline import bisplev


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


def benchmark_construction(implementations, n_points_list, surface_type='smooth'):
    """Benchmark construction time for different implementations."""
    print(f"\n{'='*80}")
    print(f"Construction Benchmark - {surface_type} surface")
    print(f"{'='*80}")
    print(f"{'N Points':<10} {'Implementation':<20} {'Time (ms)':<12} {'Knots':<20} {'FP':<12}")
    print(f"{'-'*80}")
    
    results = {}
    
    for n_points in n_points_list:
        x, y, z = generate_test_surface(n_points, surface_type)
        w = np.ones_like(x)
        
        for name, impl in implementations.items():
            try:
                if name == 'scipy':
                    start = time.perf_counter()
                    tck = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.1)
                    elapsed = (time.perf_counter() - start) * 1000
                    nx, ny = len(tck[0]), len(tck[1])
                    fp = 0  # SciPy doesn't return this directly
                    
                elif name == 'simple':
                    # Allocate arrays
                    max_knots = 50
                    tx = np.zeros(max_knots)
                    ty = np.zeros(max_knots)
                    c = np.zeros(max_knots * max_knots)
                    
                    start = time.perf_counter()
                    result = bisplrep_simple(x, y, z, 3, 3, tx, ty, c)
                    elapsed = (time.perf_counter() - start) * 1000
                    
                    nx = (result >> 32) & 0xFFFFFFFF
                    ny = result & 0xFFFFFFFF
                    fp = 0  # Simple version doesn't compute fp
                    
                elif name == 'advanced':
                    # Allocate arrays
                    max_knots = 50
                    tx = np.zeros(max_knots)
                    ty = np.zeros(max_knots)
                    c = np.zeros(max_knots * max_knots)
                    
                    start = time.perf_counter()
                    result = bisplrep_advanced(x, y, z, w, 3, 3, 0.1, tx, ty, c)
                    elapsed = (time.perf_counter() - start) * 1000
                    
                    nx = (result >> 32) & 0xFFFFFFFF
                    ny = result & 0xFFFFFFFF
                    fp = 0  # Would need to compute separately
                
                elif name == 'advanced_py':
                    start = time.perf_counter()
                    tck = bisplrep_advanced_py(x, y, z, w, kx=3, ky=3, s=0.1)
                    elapsed = (time.perf_counter() - start) * 1000
                    nx, ny = len(tck[0]), len(tck[1])
                    fp = 0
                
                # Store results
                if name not in results:
                    results[name] = {'n_points': [], 'times': [], 'nx': [], 'ny': []}
                
                results[name]['n_points'].append(n_points)
                results[name]['times'].append(elapsed)
                results[name]['nx'].append(nx)
                results[name]['ny'].append(ny)
                
                print(f"{n_points:<10} {name:<20} {elapsed:<12.2f} nx={nx:<3} ny={ny:<3} {fp:<12.2e}")
                
            except Exception as e:
                print(f"{n_points:<10} {name:<20} {'FAILED':<12} {str(e)[:40]}")
    
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
    """Compare accuracy of different implementations."""
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
    
    # Simple cfunc
    tx_simple = np.zeros(50)
    ty_simple = np.zeros(50)
    c_simple = np.zeros(2500)
    result = bisplrep_simple(x, y, z, 3, 3, tx_simple, ty_simple, c_simple)
    nx_simple = (result >> 32) & 0xFFFFFFFF
    ny_simple = result & 0xFFFFFFFF
    print(f"  Simple: nx={nx_simple}, ny={ny_simple}")
    
    # Advanced cfunc
    w = np.ones_like(x)
    tx_adv = np.zeros(50)
    ty_adv = np.zeros(50)
    c_adv = np.zeros(2500)
    result = bisplrep_advanced(x, y, z, w, 3, 3, 0, tx_adv, ty_adv, c_adv)
    nx_adv = (result >> 32) & 0xFFFFFFFF
    ny_adv = result & 0xFFFFFFFF
    print(f"  Advanced: nx={nx_adv}, ny={ny_adv}")
    
    # Test on a grid
    print("\nEvaluating on test grid...")
    n_test = 50
    x_test = np.linspace(-0.9, 0.9, n_test)
    y_test = np.linspace(-0.9, 0.9, n_test)
    
    # Evaluate with each method
    z_scipy = np.zeros((n_test, n_test))
    z_simple = np.zeros((n_test, n_test))
    z_advanced = np.zeros((n_test, n_test))
    
    for i in range(n_test):
        for j in range(n_test):
            z_scipy[i, j] = scipy_bisplev(x_test[i], y_test[j], tck_scipy)
            z_simple[i, j] = bisplev(x_test[i], y_test[j], 
                                    tx_simple[:nx_simple], ty_simple[:ny_simple],
                                    c_simple[:nx_simple*ny_simple], 3, 3)
            z_advanced[i, j] = bisplev(x_test[i], y_test[j],
                                      tx_adv[:nx_adv], ty_adv[:ny_adv],
                                      c_adv[:nx_adv*ny_adv], 3, 3)
    
    # Compare differences
    diff_simple = np.abs(z_scipy - z_simple)
    diff_advanced = np.abs(z_scipy - z_advanced)
    
    print(f"\n{'Method':<20} {'Max Diff':<15} {'Mean Diff':<15} {'RMS Diff':<15}")
    print(f"{'-'*65}")
    print(f"{'Simple vs SciPy':<20} {np.max(diff_simple):<15.2e} "
          f"{np.mean(diff_simple):<15.2e} {np.sqrt(np.mean(diff_simple**2)):<15.2e}")
    print(f"{'Advanced vs SciPy':<20} {np.max(diff_advanced):<15.2e} "
          f"{np.mean(diff_advanced):<15.2e} {np.sqrt(np.mean(diff_advanced**2)):<15.2e}")


def plot_results(construction_results):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Construction time
    ax1.set_title('bisplrep Construction Time', fontsize=14)
    
    markers = {'scipy': 'o', 'simple': 's', 'advanced': '^', 'advanced_py': 'D'}
    colors = {'scipy': 'blue', 'simple': 'green', 'advanced': 'red', 'advanced_py': 'orange'}
    
    for name, data in construction_results.items():
        if 'n_points' in data and 'times' in data:
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
        if 'n_points' in data and 'nx' in data:
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
    
    # Define implementations to test
    implementations = {
        'scipy': None,
        'simple': bisplrep_simple,
        'advanced': bisplrep_advanced,
        'advanced_py': bisplrep_advanced_py
    }
    
    # Test different problem sizes
    n_points_list = [50, 100, 200, 500, 1000]
    
    # Run construction benchmarks
    results_smooth = benchmark_construction(implementations, n_points_list, 'smooth')
    results_noisy = benchmark_construction(implementations, n_points_list, 'noisy')
    
    # Run evaluation benchmark
    benchmark_evaluation()
    
    # Test accuracy
    test_accuracy_comparison()
    
    # Plot results
    plot_results(results_smooth)
    
    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    main()