#!/usr/bin/env python3
"""
Performance benchmark: SciPy bisplrep/bisplev vs cfunc implementation
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep, bisplev
import sys
sys.path.insert(0, '..')

from dierckx_cfunc import bisplrep_cfunc, bisplev_cfunc

def generate_surface_data(n_points):
    """Generate random surface data"""
    # Random points in [-2, 2] x [-2, 2]
    x = np.random.uniform(-2, 2, n_points)
    y = np.random.uniform(-2, 2, n_points)
    
    # Test function: Franke's function (commonly used for testing)
    term1 = 0.75 * np.exp(-(9*x-2)**2/4 - (9*y-2)**2/4)
    term2 = 0.75 * np.exp(-(9*x+1)**2/49 - (9*y+1)/10)
    term3 = 0.5 * np.exp(-(9*x-7)**2/4 - (9*y-3)**2/4)
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)
    z = term1 + term2 + term3 + term4
    
    return x, y, z

def time_function(func, *args, iterations=10, warmup=2):
    """Time a function with proper warmup"""
    # Warmup
    for _ in range(warmup):
        try:
            func(*args)
        except:
            pass
    
    # Actual timing
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = func(*args)
            end = time.perf_counter()
            times.append(end - start)
        except:
            return float('inf'), None
    
    return np.median(times), result

def benchmark_bisplrep():
    """Benchmark bisplrep construction performance"""
    print("\n" + "=" * 60)
    print("BENCHMARKING BISPLREP (SPLINE CONSTRUCTION)")
    print("=" * 60)
    
    # Different problem sizes
    n_points_list = [25, 50, 100, 200, 400, 800]
    degrees = [(1,1), (2,2), (3,3)]
    
    results = {}
    
    for kx, ky in degrees:
        print(f"\nDegree (kx={kx}, ky={ky}):")
        print("-" * 40)
        
        scipy_times = []
        cfunc_times = []
        speedups = []
        
        for n in n_points_list:
            # Generate data
            x, y, z = generate_surface_data(n)
            
            # Time SciPy
            scipy_time, tck_scipy = time_function(bisplrep, x, y, z, kx, ky, 0)
            
            # Time cfunc
            cfunc_time, tck_cfunc = time_function(bisplrep_cfunc, x, y, z, kx, ky, 0.0)
            
            if scipy_time != float('inf') and cfunc_time != float('inf'):
                speedup = scipy_time / cfunc_time
                scipy_times.append(scipy_time * 1000)  # Convert to ms
                cfunc_times.append(cfunc_time * 1000)
                speedups.append(speedup)
                
                print(f"  n={n:4d}: SciPy {scipy_time*1000:6.1f}ms, "
                      f"cfunc {cfunc_time*1000:6.1f}ms, Speedup: {speedup:.2f}×")
            else:
                print(f"  n={n:4d}: Failed")
                scipy_times.append(np.nan)
                cfunc_times.append(np.nan)
                speedups.append(np.nan)
        
        results[(kx, ky)] = {
            'n_points': n_points_list,
            'scipy': scipy_times,
            'cfunc': cfunc_times,
            'speedups': speedups
        }
    
    return results

def benchmark_bisplev():
    """Benchmark bisplev evaluation performance"""
    print("\n" + "=" * 60)
    print("BENCHMARKING BISPLEV (SPLINE EVALUATION)")
    print("=" * 60)
    
    # Create spline from medium-sized dataset
    x_data, y_data, z_data = generate_surface_data(100)
    
    # Different evaluation grid sizes
    grid_sizes = [10, 20, 40, 80, 160]
    
    print("\nConstructing reference splines...")
    tck_scipy = bisplrep(x_data, y_data, z_data, s=0)
    tx, ty, c, kx, ky = bisplrep_cfunc(x_data, y_data, z_data, s=0.0)
    
    print("\nEvaluation performance:")
    print("-" * 40)
    
    scipy_times = []
    cfunc_times = []
    speedups = []
    total_points = []
    
    for n in grid_sizes:
        # Create evaluation grid
        x_eval = np.linspace(-2, 2, n)
        y_eval = np.linspace(-2, 2, n)
        total_pts = n * n
        total_points.append(total_pts)
        
        # Time SciPy
        scipy_time, _ = time_function(bisplev, x_eval, y_eval, tck_scipy, iterations=100)
        
        # Time cfunc
        cfunc_time, _ = time_function(bisplev_cfunc, x_eval, y_eval, tx, ty, c, kx, ky, iterations=100)
        
        speedup = scipy_time / cfunc_time
        scipy_times.append(scipy_time * 1000)
        cfunc_times.append(cfunc_time * 1000)
        speedups.append(speedup)
        
        print(f"  {n}×{n} grid ({total_pts:5d} pts): "
              f"SciPy {scipy_time*1000:6.2f}ms, "
              f"cfunc {cfunc_time*1000:6.2f}ms, "
              f"Speedup: {speedup:.2f}×")
    
    return {
        'grid_sizes': grid_sizes,
        'total_points': total_points,
        'scipy': scipy_times,
        'cfunc': cfunc_times,
        'speedups': speedups
    }

def create_performance_plots(bisplrep_results, bisplev_results):
    """Create comprehensive performance visualization"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. bisplrep scaling for different degrees
    ax1 = plt.subplot(2, 3, 1)
    for (kx, ky), data in bisplrep_results.items():
        mask = ~np.isnan(data['scipy'])
        ax1.loglog(np.array(data['n_points'])[mask], 
                  np.array(data['scipy'])[mask], 
                  'o-', label=f'SciPy (k={kx},{ky})', linewidth=2)
        ax1.loglog(np.array(data['n_points'])[mask], 
                  np.array(data['cfunc'])[mask], 
                  '^--', label=f'cfunc (k={kx},{ky})', linewidth=2)
    
    ax1.set_xlabel('Number of Data Points')
    ax1.set_ylabel('Construction Time (ms)')
    ax1.set_title('bisplrep Performance Scaling', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # 2. bisplrep speedup
    ax2 = plt.subplot(2, 3, 2)
    for (kx, ky), data in bisplrep_results.items():
        mask = ~np.isnan(data['speedups'])
        ax2.semilogx(np.array(data['n_points'])[mask], 
                    np.array(data['speedups'])[mask], 
                    's-', label=f'k={kx},{ky}', linewidth=2, markersize=8)
    
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Data Points')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('bisplrep Speedup Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. bisplev evaluation performance
    ax3 = plt.subplot(2, 3, 3)
    ax3.loglog(bisplev_results['total_points'], bisplev_results['scipy'], 
              'r-o', label='SciPy', linewidth=2, markersize=8)
    ax3.loglog(bisplev_results['total_points'], bisplev_results['cfunc'], 
              'g-^', label='cfunc', linewidth=2, markersize=8)
    ax3.set_xlabel('Total Evaluation Points')
    ax3.set_ylabel('Evaluation Time (ms)')
    ax3.set_title('bisplev Performance Scaling', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. bisplev speedup
    ax4 = plt.subplot(2, 3, 4)
    ax4.semilogx(bisplev_results['total_points'], bisplev_results['speedups'], 
                'b-s', linewidth=2, markersize=8)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Total Evaluation Points')
    ax4.set_ylabel('Speedup Factor')
    ax4.set_title('bisplev Speedup Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(bisplev_results['speedups']) * 1.2)
    
    # 5. Construction time comparison
    ax5 = plt.subplot(2, 3, 5)
    degrees = list(bisplrep_results.keys())
    avg_speedups = []
    for deg in degrees:
        mask = ~np.isnan(bisplrep_results[deg]['speedups'])
        avg_speedups.append(np.mean(np.array(bisplrep_results[deg]['speedups'])[mask]))
    
    x = np.arange(len(degrees))
    bars = ax5.bar(x, avg_speedups, alpha=0.8, color=['blue', 'green', 'orange'])
    ax5.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'k={kx},{ky}' for kx, ky in degrees])
    ax5.set_ylabel('Average Speedup')
    ax5.set_title('bisplrep Average Speedup by Degree', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate overall statistics
    bisplrep_speedups = []
    for data in bisplrep_results.values():
        mask = ~np.isnan(data['speedups'])
        bisplrep_speedups.extend(np.array(data['speedups'])[mask])
    
    summary_text = "PERFORMANCE SUMMARY\n\n"
    summary_text += "bisplrep (construction):\n"
    summary_text += f"  Avg speedup: {np.mean(bisplrep_speedups):.2f}×\n"
    summary_text += f"  Max speedup: {np.max(bisplrep_speedups):.2f}×\n"
    summary_text += f"  Min speedup: {np.min(bisplrep_speedups):.2f}×\n\n"
    
    summary_text += "bisplev (evaluation):\n"
    summary_text += f"  Avg speedup: {np.mean(bisplev_results['speedups']):.2f}×\n"
    summary_text += f"  Max speedup: {np.max(bisplev_results['speedups']):.2f}×\n"
    summary_text += f"  Min speedup: {np.min(bisplev_results['speedups']):.2f}×\n"
    
    ax6.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
            fontfamily='monospace', transform=ax6.transAxes)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.suptitle('SciPy vs cfunc Bivariate Spline Performance Analysis', 
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plt.savefig('bisplrep_bisplev_performance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Performance plot saved as 'bisplrep_bisplev_performance.png'")

def main():
    """Run comprehensive benchmark"""
    print("SCIPY VS CFUNC BIVARIATE SPLINE BENCHMARK")
    print("=" * 60)
    
    # Warmup
    print("\nWarming up functions...")
    x, y, z = generate_surface_data(25)
    bisplrep(x, y, z, s=0)
    bisplrep_cfunc(x, y, z, s=0.0)
    
    # Run benchmarks
    bisplrep_results = benchmark_bisplrep()
    bisplev_results = benchmark_bisplev()
    
    # Create plots
    create_performance_plots(bisplrep_results, bisplev_results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    
    # Calculate overall performance
    all_speedups = []
    for data in bisplrep_results.values():
        mask = ~np.isnan(data['speedups'])
        all_speedups.extend(np.array(data['speedups'])[mask])
    all_speedups.extend(bisplev_results['speedups'])
    
    print(f"\nOverall average speedup: {np.mean(all_speedups):.2f}×")
    
    if np.mean(all_speedups) > 1.0:
        print("✓ cfunc implementation is faster than SciPy on average!")
    else:
        print("⚠ SciPy is faster on average for these problem sizes")
    
    print("\n✓ All benchmarks complete!")

if __name__ == "__main__":
    main()