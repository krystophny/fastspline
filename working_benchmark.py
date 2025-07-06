#!/usr/bin/env python3
"""
Working benchmark for DIERCKX vs Numba implementation
Compares surface fitting and evaluation performance
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from dierckx_wrapper import bisplrep_dierckx, bisplev_dierckx

def create_test_data(n_points):
    """Create test data for surface fitting"""
    np.random.seed(42)
    x = np.random.uniform(0.1, 0.9, n_points)
    y = np.random.uniform(0.1, 0.9, n_points)
    z = np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + 0.1*np.random.randn(n_points)
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)

def benchmark_fitting(sizes):
    """Benchmark surface fitting performance"""
    print("SURFACE FITTING BENCHMARK")
    print("=" * 50)
    
    results = []
    
    for n in sizes:
        x, y, z = create_test_data(n)
        
        # Benchmark Numba implementation
        start_time = time.time()
        for _ in range(5):  # Multiple iterations for better timing
            try:
                tck_numba = bisplrep_dierckx(x, y, z, kx=3, ky=3, s=0.1)
            except:
                tck_numba = None
        numba_time = (time.time() - start_time) / 5 * 1000  # ms
        
        results.append((n, numba_time, tck_numba is not None))
        
        if tck_numba is not None:
            print(f"  {n:4d} points: {numba_time:6.2f} ms - SUCCESS")
        else:
            print(f"  {n:4d} points: {numba_time:6.2f} ms - FAILED")
    
    return results

def benchmark_evaluation(n_eval=1000):
    """Benchmark surface evaluation performance"""
    print("\nSURFACE EVALUATION BENCHMARK")
    print("=" * 50)
    
    # Create test surface
    x, y, z = create_test_data(100)
    tck = bisplrep_dierckx(x, y, z, kx=3, ky=3, s=0.1)
    
    # Create evaluation points
    np.random.seed(123)
    x_eval = np.random.uniform(0.2, 0.8, n_eval)
    y_eval = np.random.uniform(0.2, 0.8, n_eval)
    
    # Benchmark evaluation
    start_time = time.time()
    for _ in range(10):  # Multiple iterations
        z_eval = bisplev_dierckx(x_eval, y_eval, tck)
    eval_time = (time.time() - start_time) / 10 * 1000  # ms
    
    print(f"  {n_eval} evaluations: {eval_time:.2f} ms")
    print(f"  Average per evaluation: {eval_time/n_eval*1000:.2f} μs")
    
    return eval_time

def run_memory_test():
    """Test memory usage with large datasets"""
    print("\nMEMORY USAGE TEST")
    print("=" * 50)
    
    sizes = [1000, 5000, 10000]
    
    for n in sizes:
        x, y, z = create_test_data(n)
        
        try:
            start_time = time.time()
            tck = bisplrep_dierckx(x, y, z, kx=3, ky=3, s=0.1)
            fit_time = (time.time() - start_time) * 1000
            
            # Test evaluation on a grid
            xi = np.linspace(0.1, 0.9, 50)
            yi = np.linspace(0.1, 0.9, 50)
            start_time = time.time()
            zi = bisplev_dierckx(xi, yi, tck)
            eval_time = (time.time() - start_time) * 1000
            
            print(f"  {n:5d} points: Fit {fit_time:6.1f}ms, Eval {eval_time:5.1f}ms - SUCCESS")
            
        except Exception as e:
            print(f"  {n:5d} points: FAILED - {str(e)[:50]}")

def create_performance_plot(fitting_results):
    """Create performance visualization"""
    sizes = [r[0] for r in fitting_results if r[2]]  # Only successful fits
    times = [r[1] for r in fitting_results if r[2]]
    
    if len(sizes) < 3:
        print("Not enough data points for plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Fitting performance
    plt.subplot(1, 2, 1)
    plt.loglog(sizes, times, 'bo-', label='Numba Implementation')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Fitting Time (ms)')
    plt.title('Surface Fitting Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Scaling analysis
    plt.subplot(1, 2, 2)
    if len(sizes) >= 3:
        # Fit power law: t = a * n^b
        log_sizes = np.log(sizes)
        log_times = np.log(times)
        coeffs = np.polyfit(log_sizes, log_times, 1)
        scaling_exp = coeffs[0]
        
        plt.loglog(sizes, times, 'bo', label='Measured')
        fit_line = np.exp(coeffs[1]) * np.array(sizes) ** scaling_exp
        plt.loglog(sizes, fit_line, 'r--', label=f'O(n^{scaling_exp:.2f})')
        plt.xlabel('Number of Data Points')
        plt.ylabel('Fitting Time (ms)')
        plt.title(f'Scaling Analysis\nComplexity: O(n^{scaling_exp:.2f})')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('examples/current_performance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Performance plot saved as 'examples/current_performance.png'")

def main():
    """Run comprehensive benchmark suite"""
    print("FASTSPLINE NUMBA IMPLEMENTATION BENCHMARK")
    print("=" * 60)
    
    # Test different data sizes
    sizes = [50, 100, 200, 500, 1000, 2000]
    
    # Run benchmarks
    fitting_results = benchmark_fitting(sizes)
    eval_time = benchmark_evaluation(1000)
    run_memory_test()
    
    # Performance summary
    print("\nPERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful_fits = [r for r in fitting_results if r[2]]
    if successful_fits:
        avg_time = np.mean([r[1] for r in successful_fits])
        min_time = min([r[1] for r in successful_fits])
        max_time = max([r[1] for r in successful_fits])
        
        print(f"Fitting Performance:")
        print(f"  • Average time: {avg_time:.1f} ms")
        print(f"  • Range: {min_time:.1f} - {max_time:.1f} ms")
        print(f"  • Success rate: {len(successful_fits)}/{len(fitting_results)} ({100*len(successful_fits)/len(fitting_results):.0f}%)")
    
    print(f"\nEvaluation Performance:")
    print(f"  • 1000 evaluations: {eval_time:.1f} ms")
    print(f"  • Per evaluation: {eval_time/1000*1000:.1f} μs")
    
    # Create visualization
    create_performance_plot(fitting_results)
    
    print("\n✓ Benchmark complete!")
    
    return successful_fits, eval_time

if __name__ == "__main__":
    main()