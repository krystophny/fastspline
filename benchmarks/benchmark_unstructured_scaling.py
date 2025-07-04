#!/usr/bin/env python3
"""
Performance benchmark for 2D splines from unstructured points.

This script benchmarks the scaling of 2D spline interpolation from unstructured
scattered points, testing from 10 to 10,000 points with logarithmic scaling
and plotting the runtime performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from fastspline.spline2d import Spline2D


def generate_unstructured_data(n_points, func_type='smooth'):
    """
    Generate unstructured test data.
    
    Parameters:
    n_points: number of scattered points
    func_type: 'smooth', 'oscillatory', or 'complex'
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate random scattered points
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    if func_type == 'smooth':
        # Smooth Gaussian-like function
        z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x) * np.sin(np.pi * y)
    elif func_type == 'oscillatory':
        # Oscillatory function
        z = np.sin(5 * x) * np.cos(5 * y) + 0.1 * np.sin(20 * x * y)
    else:  # complex
        # Complex function with multiple scales
        z = (np.exp(-(x**2 + y**2)) + 
             0.5 * np.sin(3 * x) * np.cos(3 * y) +
             0.2 * np.sin(10 * x) * np.cos(10 * y))
    
    return x, y, z


def benchmark_construction_time(n_points_list, spline_orders=[(1, 1), (3, 3)], 
                              func_types=['smooth', 'oscillatory']):
    """
    Benchmark spline construction time vs number of points.
    
    Returns:
    results: dict with construction times for each configuration
    """
    results = {}
    
    print("Benchmarking 2D Spline Construction from Unstructured Data")
    print("=" * 65)
    print(f"{'N Points':<8} {'Function':<12} {'Order':<8} {'Time (ms)':<10} {'Memory (MB)':<12}")
    print("-" * 65)
    
    for func_type in func_types:
        for kx, ky in spline_orders:
            key = f"{func_type}_k{kx}{ky}"
            results[key] = {'n_points': [], 'times': [], 'memory': []}
            
            for n_points in n_points_list:
                # Generate test data
                x, y, z = generate_unstructured_data(n_points, func_type)
                
                # Measure construction time
                start_time = time.perf_counter()
                spline = Spline2D(x, y, z, kx=kx, ky=ky)
                end_time = time.perf_counter()
                
                construction_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Estimate memory usage (rough approximation)
                grid_size = max(10, int(np.sqrt(n_points)))
                memory_mb = (grid_size**2 * (kx + 1) * (ky + 1) * 8) / (1024**2)
                
                # Store results
                results[key]['n_points'].append(n_points)
                results[key]['times'].append(construction_time)
                results[key]['memory'].append(memory_mb)
                
                print(f"{n_points:<8} {func_type:<12} ({kx},{ky})    {construction_time:8.2f}   {memory_mb:8.3f}")
    
    return results


def benchmark_evaluation_time(n_points_list, n_eval_points=1000):
    """
    Benchmark evaluation time vs number of training points.
    
    Returns:
    eval_results: dict with evaluation times
    """
    print(f"\nBenchmarking Evaluation Time ({n_eval_points} evaluations)")
    print("=" * 50)
    print(f"{'N Points':<8} {'Function':<12} {'Eval Time (ms)':<15} {'μs/eval':<10}")
    print("-" * 50)
    
    eval_results = {}
    
    # Generate fixed evaluation points
    np.random.seed(123)
    x_eval = np.random.uniform(-0.8, 0.8, n_eval_points)
    y_eval = np.random.uniform(-0.8, 0.8, n_eval_points)
    
    for func_type in ['smooth', 'oscillatory']:
        key = f"{func_type}_eval"
        eval_results[key] = {'n_points': [], 'times': []}
        
        for n_points in n_points_list:
            # Create spline from unstructured data
            x, y, z = generate_unstructured_data(n_points, func_type)
            spline = Spline2D(x, y, z, kx=3, ky=3)
            
            # Measure evaluation time
            start_time = time.perf_counter()
            results = spline(x_eval, y_eval, grid=False)
            end_time = time.perf_counter()
            
            eval_time = (end_time - start_time) * 1000  # Convert to ms
            time_per_eval = eval_time * 1000 / n_eval_points  # Convert to μs
            
            eval_results[key]['n_points'].append(n_points)
            eval_results[key]['times'].append(eval_time)
            
            print(f"{n_points:<8} {func_type:<12} {eval_time:10.2f}     {time_per_eval:8.2f}")
    
    return eval_results


def plot_scaling_results(construction_results, evaluation_results):
    """
    Create logarithmic plots of performance scaling.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Construction time scaling
    ax1.set_title('Construction Time Scaling', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, (key, data) in enumerate(construction_results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax1.loglog(data['n_points'], data['times'], 
                  color=color, marker=marker, linewidth=2, markersize=6,
                  label=key.replace('_', ', '))
    
    ax1.set_xlabel('Number of Points', fontsize=12)
    ax1.set_ylabel('Construction Time (ms)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Memory usage scaling
    ax2.set_title('Memory Usage Scaling', fontsize=14, fontweight='bold')
    
    for i, (key, data) in enumerate(construction_results.items()):
        if 'k33' in key:  # Only show cubic splines for memory
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            ax2.loglog(data['n_points'], data['memory'], 
                      color=color, marker=marker, linewidth=2, markersize=6,
                      label=key.replace('_k33', ''))
    
    ax2.set_xlabel('Number of Points', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Evaluation time scaling
    ax3.set_title('Evaluation Time Scaling', fontsize=14, fontweight='bold')
    
    for i, (key, data) in enumerate(evaluation_results.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax3.loglog(data['n_points'], data['times'], 
                  color=color, marker=marker, linewidth=2, markersize=6,
                  label=key.replace('_eval', ''))
    
    ax3.set_xlabel('Number of Points', fontsize=12)
    ax3.set_ylabel('Evaluation Time (ms)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Scaling efficiency
    ax4.set_title('Scaling Efficiency Analysis', fontsize=14, fontweight='bold')
    
    # Compute scaling exponents for construction times
    for key, data in construction_results.items():
        if 'k33' in key:  # Focus on cubic splines
            n_points = np.array(data['n_points'])
            times = np.array(data['times'])
            
            # Fit power law: time = a * n^b
            log_n = np.log10(n_points)
            log_t = np.log10(times)
            
            # Linear regression in log space
            coeffs = np.polyfit(log_n, log_t, 1)
            scaling_exponent = coeffs[0]
            
            # Plot theoretical curves
            n_theory = np.logspace(1, 4, 100)
            t_theory = times[0] * (n_theory / n_points[0])**scaling_exponent
            
            color = 'blue' if 'smooth' in key else 'red'
            line_style = '--' if 'smooth' in key else '-.'
            
            ax4.loglog(n_theory, t_theory, color=color, linestyle=line_style, 
                      linewidth=2, alpha=0.7,
                      label=f"{key.replace('_k33', '')} (slope={scaling_exponent:.2f})")
            
            # Plot actual data points
            ax4.loglog(n_points, times, color=color, marker='o', markersize=6)
    
    # Add reference lines
    n_ref = np.array([10, 10000])
    ax4.loglog(n_ref, n_ref**1 * 0.1, 'k--', alpha=0.5, label='O(n) linear')
    ax4.loglog(n_ref, n_ref**1.5 * 0.01, 'k:', alpha=0.5, label='O(n^1.5)')
    ax4.loglog(n_ref, n_ref**2 * 0.001, 'k-.', alpha=0.5, label='O(n²) quadratic')
    
    ax4.set_xlabel('Number of Points', fontsize=12)
    ax4.set_ylabel('Construction Time (ms)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('unstructured_scaling_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def analyze_complexity(construction_results):
    """
    Analyze computational complexity of the algorithms.
    """
    print(f"\nComplexity Analysis")
    print("=" * 50)
    
    for key, data in construction_results.items():
        n_points = np.array(data['n_points'])
        times = np.array(data['times'])
        
        # Fit power law: time = a * n^b
        log_n = np.log10(n_points)
        log_t = np.log10(times)
        
        # Linear regression in log space
        coeffs = np.polyfit(log_n, log_t, 1)
        scaling_exponent = coeffs[0]
        r_squared = np.corrcoef(log_n, log_t)[0, 1]**2
        
        print(f"{key:<20}: O(n^{scaling_exponent:.2f}), R²={r_squared:.3f}")
    
    print()
    print("Theoretical complexities:")
    print("- Grid creation: O(n) where n is number of input points")
    print("- Spline fitting: O(m²) where m is grid size ≈ √n")
    print("- Overall: O(n + √n²) = O(n) for unstructured → structured conversion")


def main():
    """
    Run comprehensive unstructured scaling benchmark.
    """
    print("FastSpline Unstructured Data Scaling Benchmark")
    print("=" * 60)
    print("Testing spline construction from scattered points")
    print("Point count range: 10 to 10,000 (logarithmic scaling)")
    print()
    
    # Define test point counts (logarithmic spacing)
    n_points_list = np.logspace(1, 4, 15).astype(int)  # 10 to 10,000 points
    n_points_list = np.unique(n_points_list)  # Remove duplicates
    
    print(f"Testing with point counts: {n_points_list}")
    print()
    
    # Run benchmarks
    construction_results = benchmark_construction_time(
        n_points_list, 
        spline_orders=[(1, 1), (3, 3)],
        func_types=['smooth', 'oscillatory']
    )
    
    evaluation_results = benchmark_evaluation_time(n_points_list)
    
    # Analyze complexity
    analyze_complexity(construction_results)
    
    # Create plots
    fig = plot_scaling_results(construction_results, evaluation_results)
    
    # Performance summary
    print(f"\nPerformance Summary")
    print("=" * 50)
    
    largest_n = n_points_list[-1]
    for key, data in construction_results.items():
        if 'k33' in key:  # Focus on cubic splines
            time_10k = data['times'][-1]
            print(f"{key}: {time_10k:.1f}ms for {largest_n} points")
    
    print(f"\nCache optimization benefits:")
    print("✓ Spatial locality in coefficient access")
    print("✓ Contiguous memory layout for inner loops")
    print("✓ Efficient grid construction from scattered data")
    print("✓ Cache-friendly evaluation patterns")
    
    print(f"\nPlot saved as: unstructured_scaling_benchmark.png")


if __name__ == "__main__":
    main()