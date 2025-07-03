#!/usr/bin/env python3
"""
Comparative benchmark: FastSpline vs SciPy for unstructured 2D interpolation.

This script compares performance and accuracy between FastSpline's cache-optimized
implementation and SciPy's various interpolation methods for scattered data.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import griddata, bisplrep, bisplev, LinearNDInterpolator, CloughTocher2DInterpolator
from fastspline.spline2d import Spline2D


def generate_test_data(n_points, function_type='smooth', noise_level=0.0):
    """Generate test data for benchmarking."""
    np.random.seed(42)
    
    # Generate scattered points
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    if function_type == 'smooth':
        z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x) * np.sin(np.pi * y)
    elif function_type == 'oscillatory':
        z = np.sin(5 * x) * np.cos(5 * y) + 0.1 * np.sin(20 * x * y)
    elif function_type == 'peaks':
        # MATLAB peaks function
        z = (3 * (1-x)**2 * np.exp(-(x**2) - (y+1)**2) - 
             10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2) - 
             1/3 * np.exp(-(x+1)**2 - y**2))
    else:  # polynomial
        z = x**2 + x*y + y**2 + 0.1*x**3
    
    # Add noise if specified
    if noise_level > 0:
        z += np.random.normal(0, noise_level * np.std(z), len(z))
    
    return x, y, z


def benchmark_construction_scipy_vs_fastspline(n_points_list):
    """Compare construction times between scipy methods and fastspline."""
    
    results = {
        'fastspline_linear': {'n_points': [], 'times': []},
        'fastspline_cubic': {'n_points': [], 'times': []},
        'scipy_bisplrep_linear': {'n_points': [], 'times': []},
        'scipy_bisplrep_cubic': {'n_points': [], 'times': []},
        'scipy_linear_nd': {'n_points': [], 'times': []},
        'scipy_clough_tocher': {'n_points': [], 'times': []}
    }
    
    print("Construction Time Comparison: FastSpline vs SciPy")
    print("=" * 80)
    print(f"{'N Points':<8} {'FastSpline':<12} {'FastSpline':<12} {'SciPy':<12} {'SciPy':<12} {'Linear':<10} {'Clough':<10}")
    print(f"{'':8} {'Linear':<12} {'Cubic':<12} {'bisplrep k1':<12} {'bisplrep k3':<12} {'ND':<10} {'Tocher':<10}")
    print("-" * 80)
    
    for n_points in n_points_list:
        x, y, z = generate_test_data(n_points, 'smooth')
        
        times_row = []
        
        # FastSpline Linear
        start = time.perf_counter()
        fs_linear = Spline2D(x, y, z, kx=1, ky=1)
        times_row.append((time.perf_counter() - start) * 1000)
        results['fastspline_linear']['n_points'].append(n_points)
        results['fastspline_linear']['times'].append(times_row[-1])
        
        # FastSpline Cubic
        start = time.perf_counter()
        fs_cubic = Spline2D(x, y, z, kx=3, ky=3)
        times_row.append((time.perf_counter() - start) * 1000)
        results['fastspline_cubic']['n_points'].append(n_points)
        results['fastspline_cubic']['times'].append(times_row[-1])
        
        # SciPy Linear (using griddata for construction timing)
        start = time.perf_counter()
        # We'll use LinearNDInterpolator for fair comparison
        scipy_linear = LinearNDInterpolator(np.column_stack((x, y)), z)
        times_row.append((time.perf_counter() - start) * 1000)
        results['scipy_linear']['n_points'].append(n_points)
        results['scipy_linear']['times'].append(times_row[-1])
        
        # SciPy Cubic (using CloughTocher2DInterpolator)
        start = time.perf_counter()
        scipy_cubic = CloughTocher2DInterpolator(np.column_stack((x, y)), z)
        times_row.append((time.perf_counter() - start) * 1000)
        results['scipy_cubic']['n_points'].append(n_points)
        results['scipy_cubic']['times'].append(times_row[-1])
        
        # RBF Linear
        start = time.perf_counter()
        rbf_linear = RBFInterpolator(np.column_stack((x, y)), z, kernel='linear')
        times_row.append((time.perf_counter() - start) * 1000)
        results['scipy_rbf_linear']['n_points'].append(n_points)
        results['scipy_rbf_linear']['times'].append(times_row[-1])
        
        # RBF Multiquadric (with epsilon parameter)
        start = time.perf_counter()
        rbf_multiquadric = RBFInterpolator(np.column_stack((x, y)), z, kernel='multiquadric', epsilon=1.0)
        times_row.append((time.perf_counter() - start) * 1000)
        results['scipy_rbf_multiquadric']['n_points'].append(n_points)
        results['scipy_rbf_multiquadric']['times'].append(times_row[-1])
        
        # Clough-Tocher (redundant but for completeness)
        times_row.append(times_row[3])  # Same as scipy_cubic
        results['scipy_clough_tocher']['n_points'].append(n_points)
        results['scipy_clough_tocher']['times'].append(times_row[-1])
        
        print(f"{n_points:<8} {times_row[0]:8.2f}     {times_row[1]:8.2f}     "
              f"{times_row[2]:8.2f}   {times_row[3]:8.2f}   {times_row[4]:8.2f}   "
              f"{times_row[5]:8.2f}     {times_row[6]:8.2f}")
    
    return results


def benchmark_evaluation_scipy_vs_fastspline(n_points_list, n_eval=1000):
    """Compare evaluation times between scipy methods and fastspline."""
    
    eval_results = {
        'fastspline_linear': {'n_points': [], 'times': []},
        'fastspline_cubic': {'n_points': [], 'times': []},
        'scipy_griddata_linear': {'n_points': [], 'times': []},
        'scipy_griddata_cubic': {'n_points': [], 'times': []},
        'scipy_linear_interp': {'n_points': [], 'times': []},
        'scipy_clough_tocher': {'n_points': [], 'times': []},
        'scipy_rbf_linear': {'n_points': [], 'times': []},
    }
    
    # Fixed evaluation points
    np.random.seed(123)
    x_eval = np.random.uniform(-0.8, 0.8, n_eval)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval)
    eval_points = np.column_stack((x_eval, y_eval))
    
    print(f"\nEvaluation Time Comparison ({n_eval} evaluations)")
    print("=" * 80)
    print(f"{'N Points':<8} {'FastSpline':<12} {'FastSpline':<12} {'GridData':<10} {'GridData':<10} {'Linear':<10} {'Clough':<10} {'RBF':<10}")
    print(f"{'':8} {'Linear':<12} {'Cubic':<12} {'Linear':<10} {'Cubic':<10} {'Interp':<10} {'Tocher':<10} {'Linear':<10}")
    print("-" * 90)
    
    for n_points in n_points_list:
        x, y, z = generate_test_data(n_points, 'smooth')
        points = np.column_stack((x, y))
        
        times_row = []
        
        # Build interpolators first
        fs_linear = Spline2D(x, y, z, kx=1, ky=1)
        fs_cubic = Spline2D(x, y, z, kx=3, ky=3)
        scipy_linear = LinearNDInterpolator(points, z)
        scipy_clough = CloughTocher2DInterpolator(points, z)
        rbf_linear = RBFInterpolator(points, z, kernel='linear')
        
        # FastSpline Linear
        start = time.perf_counter()
        result = fs_linear(x_eval, y_eval, grid=False)
        times_row.append((time.perf_counter() - start) * 1000)
        
        # FastSpline Cubic
        start = time.perf_counter()
        result = fs_cubic(x_eval, y_eval, grid=False)
        times_row.append((time.perf_counter() - start) * 1000)
        
        # SciPy griddata linear
        start = time.perf_counter()
        result = griddata(points, z, eval_points, method='linear')
        times_row.append((time.perf_counter() - start) * 1000)
        
        # SciPy griddata cubic
        start = time.perf_counter()
        result = griddata(points, z, eval_points, method='cubic')
        times_row.append((time.perf_counter() - start) * 1000)
        
        # SciPy LinearNDInterpolator
        start = time.perf_counter()
        result = scipy_linear(eval_points)
        times_row.append((time.perf_counter() - start) * 1000)
        
        # SciPy CloughTocher2DInterpolator
        start = time.perf_counter()
        result = scipy_clough(eval_points)
        times_row.append((time.perf_counter() - start) * 1000)
        
        # RBF Linear
        start = time.perf_counter()
        result = rbf_linear(eval_points)
        times_row.append((time.perf_counter() - start) * 1000)
        
        # Store results
        methods = ['fastspline_linear', 'fastspline_cubic', 'scipy_griddata_linear', 
                  'scipy_griddata_cubic', 'scipy_linear_interp', 'scipy_clough_tocher', 'scipy_rbf_linear']
        for i, method in enumerate(methods):
            eval_results[method]['n_points'].append(n_points)
            eval_results[method]['times'].append(times_row[i])
        
        print(f"{n_points:<8} {times_row[0]:8.2f}     {times_row[1]:8.2f}     "
              f"{times_row[2]:8.2f}   {times_row[3]:8.2f}   {times_row[4]:8.2f}   "
              f"{times_row[5]:8.2f}   {times_row[6]:8.2f}")
    
    return eval_results


def benchmark_accuracy_comparison(n_points=1000):
    """Compare interpolation accuracy between methods."""
    
    print(f"\nAccuracy Comparison ({n_points} training points)")
    print("=" * 60)
    
    # Generate test data with known analytical function
    x_train, y_train, z_train = generate_test_data(n_points, 'smooth')
    
    # Generate fine evaluation grid
    x_eval = np.linspace(-0.9, 0.9, 50)
    y_eval = np.linspace(-0.9, 0.9, 50)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    Z_true = np.exp(-(X_eval**2 + Y_eval**2)) * np.cos(np.pi * X_eval) * np.sin(np.pi * Y_eval)
    
    eval_points_flat = np.column_stack((X_eval.ravel(), Y_eval.ravel()))
    z_true_flat = Z_true.ravel()
    
    methods = {}
    
    # FastSpline methods
    fs_linear = Spline2D(x_train, y_train, z_train, kx=1, ky=1)
    fs_cubic = Spline2D(x_train, y_train, z_train, kx=3, ky=3)
    
    methods['FastSpline Linear'] = fs_linear(X_eval.ravel(), Y_eval.ravel(), grid=False)
    methods['FastSpline Cubic'] = fs_cubic(X_eval.ravel(), Y_eval.ravel(), grid=False)
    
    # SciPy methods
    train_points = np.column_stack((x_train, y_train))
    
    methods['SciPy Linear'] = griddata(train_points, z_train, eval_points_flat, method='linear')
    methods['SciPy Cubic'] = griddata(train_points, z_train, eval_points_flat, method='cubic')
    
    scipy_linear = LinearNDInterpolator(train_points, z_train)
    methods['LinearNDInterpolator'] = scipy_linear(eval_points_flat)
    
    clough_tocher = CloughTocher2DInterpolator(train_points, z_train)
    methods['Clough-Tocher'] = clough_tocher(eval_points_flat)
    
    rbf_linear = RBFInterpolator(train_points, z_train, kernel='linear')
    methods['RBF Linear'] = rbf_linear(eval_points_flat)
    
    # Calculate errors
    print(f"{'Method':<20} {'RMS Error':<12} {'Max Error':<12} {'Valid Points':<12}")
    print("-" * 60)
    
    for name, z_pred in methods.items():
        # Handle NaN values
        valid_mask = np.isfinite(z_pred)
        if np.sum(valid_mask) > 0:
            rms_error = np.sqrt(np.mean((z_pred[valid_mask] - z_true_flat[valid_mask])**2))
            max_error = np.max(np.abs(z_pred[valid_mask] - z_true_flat[valid_mask]))
            valid_points = np.sum(valid_mask)
        else:
            rms_error = np.inf
            max_error = np.inf
            valid_points = 0
        
        print(f"{name:<20} {rms_error:<12.6f} {max_error:<12.6f} {valid_points:<12}")


def plot_performance_comparison(construction_results, evaluation_results):
    """Plot performance comparison results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Construction time comparison
    ax1.set_title('Construction Time Comparison', fontsize=14, fontweight='bold')
    
    # Select key methods for clarity
    key_methods = ['fastspline_cubic', 'scipy_cubic', 'scipy_rbf_linear', 'scipy_clough_tocher']
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    labels = ['FastSpline Cubic', 'SciPy Cubic', 'RBF Linear', 'Clough-Tocher']
    
    for i, method in enumerate(key_methods):
        if method in construction_results:
            data = construction_results[method]
            ax1.loglog(data['n_points'], data['times'], 
                      color=colors[i], marker=markers[i], linewidth=2, markersize=6,
                      label=labels[i])
    
    ax1.set_xlabel('Number of Training Points')
    ax1.set_ylabel('Construction Time (ms)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Evaluation time comparison
    ax2.set_title('Evaluation Time Comparison', fontsize=14, fontweight='bold')
    
    eval_methods = ['fastspline_cubic', 'scipy_griddata_cubic', 'scipy_clough_tocher', 'scipy_rbf_linear']
    eval_labels = ['FastSpline Cubic', 'GridData Cubic', 'Clough-Tocher', 'RBF Linear']
    
    for i, method in enumerate(eval_methods):
        if method in evaluation_results:
            data = evaluation_results[method]
            ax2.loglog(data['n_points'], data['times'], 
                      color=colors[i], marker=markers[i], linewidth=2, markersize=6,
                      label=eval_labels[i])
    
    ax2.set_xlabel('Number of Training Points')
    ax2.set_ylabel('Evaluation Time (ms) for 1000 points')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Speedup analysis
    ax3.set_title('FastSpline Speedup vs SciPy', fontsize=14, fontweight='bold')
    
    if 'fastspline_cubic' in construction_results and 'scipy_cubic' in construction_results:
        fs_data = construction_results['fastspline_cubic']
        scipy_data = construction_results['scipy_cubic']
        
        speedup_construction = np.array(scipy_data['times']) / np.array(fs_data['times'])
        ax3.semilogx(fs_data['n_points'], speedup_construction, 
                    color='blue', marker='o', linewidth=2, markersize=6,
                    label='Construction Time')
    
    if 'fastspline_cubic' in evaluation_results and 'scipy_clough_tocher' in evaluation_results:
        fs_eval = evaluation_results['fastspline_cubic']
        scipy_eval = evaluation_results['scipy_clough_tocher']
        
        speedup_evaluation = np.array(scipy_eval['times']) / np.array(fs_eval['times'])
        ax3.semilogx(fs_eval['n_points'], speedup_evaluation, 
                    color='red', marker='s', linewidth=2, markersize=6,
                    label='Evaluation Time')
    
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax3.set_xlabel('Number of Training Points')
    ax3.set_ylabel('Speedup Factor (SciPy time / FastSpline time)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Memory efficiency comparison
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    summary_text = """
FastSpline Advantages:
✓ Cache-optimized memory layout
✓ Consistent O(n) scaling
✓ Fast evaluation (independent of training size)
✓ Numba JIT compilation
✓ C-compatible interface

SciPy Methods Comparison:
• GridData: Simple but rebuilds each time
• Clough-Tocher: Good accuracy, slower
• RBF: Flexible kernels, memory intensive
• LinearND: Fast for simple interpolation

Cache Optimizations:
• Spatial indices outermost: (nx, ny, kx, ky)
• Contiguous coefficient access
• Row-major memory access patterns
• Efficient nested loop ordering
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('scipy_fastspline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run comprehensive comparison benchmark."""
    
    print("FastSpline vs SciPy: Unstructured Data Interpolation Benchmark")
    print("=" * 70)
    print("Comparing cache-optimized FastSpline with SciPy's interpolation methods")
    print("for scattered 2D data interpolation.\n")
    
    # Test with logarithmic point distribution
    n_points_list = np.logspace(1.5, 3.5, 8).astype(int)  # ~32 to ~3162 points
    n_points_list = np.unique(n_points_list)
    
    print(f"Testing with point counts: {n_points_list}\n")
    
    # Run benchmarks
    construction_results = benchmark_construction_scipy_vs_fastspline(n_points_list)
    evaluation_results = benchmark_evaluation_scipy_vs_fastspline(n_points_list, n_eval=1000)
    
    # Accuracy comparison
    benchmark_accuracy_comparison(n_points=1000)
    
    # Performance analysis
    print(f"\nPerformance Analysis")
    print("=" * 50)
    
    # Calculate average speedups
    if len(construction_results['fastspline_cubic']['times']) > 0:
        fs_times = np.array(construction_results['fastspline_cubic']['times'])
        scipy_times = np.array(construction_results['scipy_cubic']['times'])
        avg_speedup_construction = np.mean(scipy_times / fs_times)
        
        print(f"Average construction speedup vs SciPy cubic: {avg_speedup_construction:.1f}x")
    
    if len(evaluation_results['fastspline_cubic']['times']) > 0:
        fs_eval_times = np.array(evaluation_results['fastspline_cubic']['times'])
        scipy_eval_times = np.array(evaluation_results['scipy_clough_tocher']['times'])
        avg_speedup_eval = np.mean(scipy_eval_times / fs_eval_times)
        
        print(f"Average evaluation speedup vs Clough-Tocher: {avg_speedup_eval:.1f}x")
    
    # Plot results
    plot_performance_comparison(construction_results, evaluation_results)
    
    print(f"\nBenchmark complete. Results saved as 'scipy_fastspline_comparison.png'")


if __name__ == "__main__":
    main()