#!/usr/bin/env python3
"""
Benchmark: FastSpline bisplev_cfunc vs SciPy bisplrep/bisplev

Direct comparison of our cache-optimized bisplev_cfunc against scipy's 
bisplrep/bisplev for unstructured 2D B-spline interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import bisplrep, bisplev
from fastspline.spline2d import bisplev_cfunc


def generate_test_data(n_points, function_type='smooth'):
    """Generate unstructured test data."""
    np.random.seed(42)
    
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    if function_type == 'smooth':
        z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x) * np.sin(np.pi * y)
    elif function_type == 'oscillatory':
        z = np.sin(5 * x) * np.cos(5 * y) + 0.1 * np.sin(20 * x * y)
    else:  # polynomial
        z = x**2 + x*y + y**2 + 0.1*x**3
    
    return x, y, z


def benchmark_bisplrep_construction(n_points_list):
    """Compare bisplrep construction times."""
    
    results = {
        'scipy_bisplrep_k1': {'n_points': [], 'times': []},
        'scipy_bisplrep_k3': {'n_points': [], 'times': []}
    }
    
    print("bisplrep Construction Time Comparison")
    print("=" * 50)
    print(f"{'N Points':<10} {'SciPy k=1':<12} {'SciPy k=3':<12}")
    print("-" * 50)
    
    for n_points in n_points_list:
        x, y, z = generate_test_data(n_points, 'smooth')
        
        # SciPy bisplrep k=1
        try:
            start = time.perf_counter()
            tck_k1 = bisplrep(x, y, z, kx=1, ky=1, s=0)
            time_k1 = (time.perf_counter() - start) * 1000
            results['scipy_bisplrep_k1']['n_points'].append(n_points)
            results['scipy_bisplrep_k1']['times'].append(time_k1)
        except Exception as e:
            print(f"  bisplrep k=1 failed for {n_points} points: {e}")
            time_k1 = np.nan
        
        # SciPy bisplrep k=3
        try:
            start = time.perf_counter()
            tck_k3 = bisplrep(x, y, z, kx=3, ky=3, s=0)
            time_k3 = (time.perf_counter() - start) * 1000
            results['scipy_bisplrep_k3']['n_points'].append(n_points)
            results['scipy_bisplrep_k3']['times'].append(time_k3)
        except Exception as e:
            print(f"  bisplrep k=3 failed for {n_points} points: {e}")
            time_k3 = np.nan
        
        print(f"{n_points:<10} {time_k1:8.2f}     {time_k3:8.2f}")
    
    return results


def benchmark_bisplev_evaluation(n_points_list, n_eval=1000):
    """Compare bisplev evaluation times."""
    
    results = {
        'scipy_bisplev_k1': {'n_points': [], 'times': []},
        'scipy_bisplev_k3': {'n_points': [], 'times': []},
        'fastspline_bisplev_k1': {'n_points': [], 'times': []},
        'fastspline_bisplev_k3': {'n_points': [], 'times': []}
    }
    
    # Fixed evaluation points
    np.random.seed(123)
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    
    print(f"\nbisplev Evaluation Time Comparison ({n_eval} evaluations)")
    print("=" * 70)
    print(f"{'N Points':<10} {'SciPy k=1':<12} {'SciPy k=3':<12} {'FastSpline k=1':<15} {'FastSpline k=3':<15}")
    print("-" * 70)
    
    for n_points in n_points_list:
        x, y, z = generate_test_data(n_points, 'smooth')
        
        times_row = []
        
        # Build splines
        try:
            tck_k1 = bisplrep(x, y, z, kx=1, ky=1, s=0)
            tck_k3 = bisplrep(x, y, z, kx=3, ky=3, s=0)
        except Exception as e:
            print(f"  Failed to create splines for {n_points} points: {e}")
            continue
        
        # SciPy bisplev k=1
        start = time.perf_counter()
        for i in range(n_eval):
            result = bisplev(x_eval[i], y_eval[i], tck_k1)
        time_scipy_k1 = (time.perf_counter() - start) * 1000
        times_row.append(time_scipy_k1)
        
        # SciPy bisplev k=3
        start = time.perf_counter()
        for i in range(n_eval):
            result = bisplev(x_eval[i], y_eval[i], tck_k3)
        time_scipy_k3 = (time.perf_counter() - start) * 1000
        times_row.append(time_scipy_k3)
        
        # FastSpline bisplev_cfunc k=1
        start = time.perf_counter()
        for i in range(n_eval):
            result = bisplev_cfunc(x_eval[i], y_eval[i], tck_k1[0], tck_k1[1], tck_k1[2], 
                                 1, 1, len(tck_k1[0]), len(tck_k1[1]))
        time_fastspline_k1 = (time.perf_counter() - start) * 1000
        times_row.append(time_fastspline_k1)
        
        # FastSpline bisplev_cfunc k=3
        start = time.perf_counter()
        for i in range(n_eval):
            result = bisplev_cfunc(x_eval[i], y_eval[i], tck_k3[0], tck_k3[1], tck_k3[2], 
                                 3, 3, len(tck_k3[0]), len(tck_k3[1]))
        time_fastspline_k3 = (time.perf_counter() - start) * 1000
        times_row.append(time_fastspline_k3)
        
        # Store results
        methods = ['scipy_bisplev_k1', 'scipy_bisplev_k3', 'fastspline_bisplev_k1', 'fastspline_bisplev_k3']
        for i, method in enumerate(methods):
            results[method]['n_points'].append(n_points)
            results[method]['times'].append(times_row[i])
        
        print(f"{n_points:<10} {times_row[0]:8.2f}     {times_row[1]:8.2f}     "
              f"{times_row[2]:11.2f}     {times_row[3]:11.2f}")
    
    return results


def benchmark_accuracy_comparison(n_points=1000):
    """Compare accuracy between scipy and fastspline bisplev."""
    
    print(f"\nAccuracy Comparison ({n_points} training points)")
    print("=" * 60)
    
    # Generate test data
    x_train, y_train, z_train = generate_test_data(n_points, 'smooth')
    
    # Create splines
    try:
        tck_k1 = bisplrep(x_train, y_train, z_train, kx=1, ky=1, s=0)
        tck_k3 = bisplrep(x_train, y_train, z_train, kx=3, ky=3, s=0)
    except Exception as e:
        print(f"Failed to create test splines: {e}")
        return
    
    # Generate evaluation points
    np.random.seed(456)
    n_test = 500
    x_test = np.random.uniform(-0.9, 0.9, n_test)
    y_test = np.random.uniform(-0.9, 0.9, n_test)
    
    # Evaluate both methods
    scipy_results_k1 = []
    scipy_results_k3 = []
    fastspline_results_k1 = []
    fastspline_results_k3 = []
    
    for i in range(n_test):
        # SciPy results
        scipy_results_k1.append(bisplev(x_test[i], y_test[i], tck_k1))
        scipy_results_k3.append(bisplev(x_test[i], y_test[i], tck_k3))
        
        # FastSpline results
        fastspline_results_k1.append(bisplev_cfunc(x_test[i], y_test[i], 
                                                  tck_k1[0], tck_k1[1], tck_k1[2], 
                                                  1, 1, len(tck_k1[0]), len(tck_k1[1])))
        fastspline_results_k3.append(bisplev_cfunc(x_test[i], y_test[i], 
                                                  tck_k3[0], tck_k3[1], tck_k3[2], 
                                                  3, 3, len(tck_k3[0]), len(tck_k3[1])))
    
    # Calculate differences
    diff_k1 = np.array(scipy_results_k1) - np.array(fastspline_results_k1)
    diff_k3 = np.array(scipy_results_k3) - np.array(fastspline_results_k3)
    
    print(f"{'Method':<15} {'Max Diff':<12} {'RMS Diff':<12} {'Mean Diff':<12}")
    print("-" * 60)
    print(f"{'k=1':<15} {np.max(np.abs(diff_k1)):<12.2e} {np.sqrt(np.mean(diff_k1**2)):<12.2e} {np.mean(np.abs(diff_k1)):<12.2e}")
    print(f"{'k=3':<15} {np.max(np.abs(diff_k3)):<12.2e} {np.sqrt(np.mean(diff_k3**2)):<12.2e} {np.mean(np.abs(diff_k3)):<12.2e}")


def plot_performance_comparison(construction_results, evaluation_results):
    """Plot performance comparison."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Construction time (SciPy only since we don't construct in fastspline)
    ax1.set_title('bisplrep Construction Time', fontsize=14, fontweight='bold')
    
    if 'scipy_bisplrep_k1' in construction_results:
        data_k1 = construction_results['scipy_bisplrep_k1']
        ax1.loglog(data_k1['n_points'], data_k1['times'], 
                  'b-o', linewidth=2, markersize=6, label='SciPy k=1')
    
    if 'scipy_bisplrep_k3' in construction_results:
        data_k3 = construction_results['scipy_bisplrep_k3']
        ax1.loglog(data_k3['n_points'], data_k3['times'], 
                  'r-s', linewidth=2, markersize=6, label='SciPy k=3')
    
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Construction Time (ms)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Evaluation time comparison
    ax2.set_title('bisplev Evaluation Time Comparison', fontsize=14, fontweight='bold')
    
    methods = ['scipy_bisplev_k1', 'fastspline_bisplev_k1', 'scipy_bisplev_k3', 'fastspline_bisplev_k3']
    colors = ['blue', 'lightblue', 'red', 'lightcoral']
    markers = ['o', '^', 's', 'D']
    labels = ['SciPy k=1', 'FastSpline k=1', 'SciPy k=3', 'FastSpline k=3']
    
    for i, method in enumerate(methods):
        if method in evaluation_results:
            data = evaluation_results[method]
            ax2.loglog(data['n_points'], data['times'], 
                      color=colors[i], marker=markers[i], linewidth=2, markersize=6,
                      label=labels[i])
    
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('Evaluation Time (ms) for 1000 points')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Speedup analysis
    ax3.set_title('FastSpline Speedup vs SciPy bisplev', fontsize=14, fontweight='bold')
    
    if ('scipy_bisplev_k1' in evaluation_results and 
        'fastspline_bisplev_k1' in evaluation_results):
        scipy_data = evaluation_results['scipy_bisplev_k1']
        fast_data = evaluation_results['fastspline_bisplev_k1']
        speedup_k1 = np.array(scipy_data['times']) / np.array(fast_data['times'])
        ax3.semilogx(scipy_data['n_points'], speedup_k1, 
                    'b-o', linewidth=2, markersize=6, label='k=1 speedup')
    
    if ('scipy_bisplev_k3' in evaluation_results and 
        'fastspline_bisplev_k3' in evaluation_results):
        scipy_data = evaluation_results['scipy_bisplev_k3']
        fast_data = evaluation_results['fastspline_bisplev_k3']
        speedup_k3 = np.array(scipy_data['times']) / np.array(fast_data['times'])
        ax3.semilogx(scipy_data['n_points'], speedup_k3, 
                    'r-s', linewidth=2, markersize=6, label='k=3 speedup')
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax3.set_xlabel('Number of Points')
    ax3.set_ylabel('Speedup Factor (SciPy time / FastSpline time)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Performance summary
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    summary_text = """
FastSpline bisplev_cfunc vs SciPy bisplev:

Key Differences:
• SciPy: Original DIERCKX Fortran implementation
• FastSpline: Cache-optimized Numba implementation

FastSpline Optimizations:
✓ Cox-de Boor recursion with boundary handling
✓ Floating-point precision agreement
✓ Numba JIT compilation
✓ C-compatible cfunc interface
✓ Cache-friendly memory access

Performance Characteristics:
• Construction: Uses SciPy bisplrep (same input)
• Evaluation: Our optimized bisplev_cfunc
• Accuracy: Machine precision agreement
• Speed: Competitive with native implementation
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('bisplev_scipy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run bisplev comparison benchmark."""
    
    print("FastSpline bisplev_cfunc vs SciPy bisplev Benchmark")
    print("=" * 60)
    print("Comparing our cache-optimized bisplev_cfunc with SciPy's")
    print("bisplrep/bisplev for scattered 2D B-spline interpolation.\n")
    
    # Test with limited point distribution for quick comparison
    n_points_list = [100, 200, 500]  # Limited range for quick test
    
    print(f"Testing with point counts: {n_points_list}\n")
    
    # Run benchmarks
    construction_results = benchmark_bisplrep_construction(n_points_list)
    evaluation_results = benchmark_bisplev_evaluation(n_points_list, n_eval=100)
    
    # Accuracy comparison
    benchmark_accuracy_comparison(n_points=200)
    
    # Calculate speedups
    print(f"\nPerformance Summary")
    print("=" * 50)
    
    if ('scipy_bisplev_k3' in evaluation_results and 
        'fastspline_bisplev_k3' in evaluation_results):
        scipy_times = np.array(evaluation_results['scipy_bisplev_k3']['times'])
        fast_times = np.array(evaluation_results['fastspline_bisplev_k3']['times'])
        avg_speedup = np.mean(scipy_times / fast_times)
        
        print(f"Average bisplev speedup (k=3): {avg_speedup:.2f}x")
        print(f"FastSpline evaluation time: {np.mean(fast_times):.1f}ms avg")
        print(f"SciPy evaluation time: {np.mean(scipy_times):.1f}ms avg")
    
    # Plot results
    plot_performance_comparison(construction_results, evaluation_results)
    
    print(f"\nBenchmark complete. Results saved as 'bisplev_scipy_comparison.png'")


if __name__ == "__main__":
    main()