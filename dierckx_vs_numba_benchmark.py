#!/usr/bin/env python3
"""
Comprehensive benchmark: DIERCKX f2py vs Numba implementation
Measures speedup for all core functions with scaling analysis
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from dierckx_numba_optimized import (
    fpback_njit, fpgivs_njit, fprota_njit, fprati_njit, fpbspl_njit,
    warmup_optimized_functions
)

# Import corrected DIERCKX f2py wrapper
import dierckx_f2py

def time_function(func, *args, iterations=1000, warmup=10):
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
            return float('inf')  # Failed
    
    return np.median(times) * 1e6  # microseconds

def benchmark_fpback_scaling():
    """Benchmark fpback with different matrix sizes"""
    print("Benchmarking fpback (backward substitution)...")
    
    sizes = [10, 20, 50, 100, 200]
    dierckx_times = []
    numba_times = []
    speedups = []
    
    for n in sizes:
        k = min(5, n//3)
        nest = n + 10
        
        # Create test data
        a = np.zeros((nest, k), dtype=np.float32, order='F')
        for i in range(n):
            a[i, 0] = 2.0 + 0.1 * i
            for j in range(1, min(k, n-i)):
                a[i, j] = 0.5 / (j + 1)
        
        z = np.random.randn(n).astype(np.float32)
        
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fpback, a, z, n, k, nest, iterations=100)
        
        # Numba timing
        c_numba = np.zeros(n, dtype=np.float64)
        numba_time = time_function(fpback_njit, a.astype(np.float64), z.astype(np.float64), 
                                  n, k, c_numba, nest, iterations=100)
        
        speedup = dierckx_time / numba_time if numba_time > 0 and dierckx_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        numba_times.append(numba_time)
        speedups.append(speedup)
        
        print(f"  n={n:3d}: DIERCKX {dierckx_time:6.1f}μs, Numba {numba_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return sizes, dierckx_times, numba_times, speedups

def benchmark_fpgivs():
    """Benchmark fpgivs (Givens rotations)"""
    print("\nBenchmarking fpgivs (Givens rotations)...")
    
    test_cases = [(3.0, 4.0), (1.0, 0.0), (0.0, 1.0), (-2.0, 3.0), (1e5, 1e-5)]
    dierckx_times = []
    numba_times = []
    speedups = []
    
    for piv, ww in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fpgivs, piv, ww, iterations=10000)
        
        # Numba timing
        numba_time = time_function(fpgivs_njit, piv, ww, iterations=10000)
        
        speedup = dierckx_time / numba_time if numba_time > 0 and dierckx_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        numba_times.append(numba_time)
        speedups.append(speedup)
        
        print(f"  ({piv:g},{ww:g}): DIERCKX {dierckx_time:6.1f}μs, Numba {numba_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return test_cases, dierckx_times, numba_times, speedups

def benchmark_fprota():
    """Benchmark fprota (apply rotation)"""
    print("\nBenchmarking fprota (apply rotation)...")
    
    test_cases = [(1.0, 0.0, 3.0, 4.0), (0.707, 0.707, 1.0, 1.0), (0.8, -0.6, 5.0, -3.0)]
    dierckx_times = []
    numba_times = []
    speedups = []
    
    for cos, sin, a, b in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fprota, cos, sin, a, b, iterations=10000)
        
        # Numba timing
        numba_time = time_function(fprota_njit, cos, sin, a, b, iterations=10000)
        
        speedup = dierckx_time / numba_time if numba_time > 0 and dierckx_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        numba_times.append(numba_time)
        speedups.append(speedup)
        
        print(f"  rotation: DIERCKX {dierckx_time:6.1f}μs, Numba {numba_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return test_cases, dierckx_times, numba_times, speedups

def benchmark_fprati():
    """Benchmark fprati (rational interpolation)"""
    print("\nBenchmarking fprati (rational interpolation)...")
    
    test_cases = [(1.0, 2.0, 2.0, 1.0, 3.0, -1.0), (0.1, 1.1, 0.5, 0.5, 0.9, -0.1)]
    dierckx_times = []
    numba_times = []
    speedups = []
    
    for p1, f1, p2, f2, p3, f3 in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fprati, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        # Numba timing
        numba_time = time_function(fprati_njit, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        speedup = dierckx_time / numba_time if numba_time > 0 and dierckx_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        numba_times.append(numba_time)
        speedups.append(speedup)
        
        print(f"  case: DIERCKX {dierckx_time:6.1f}μs, Numba {numba_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return test_cases, dierckx_times, numba_times, speedups

def benchmark_fpbspl_scaling():
    """Benchmark fpbspl with different spline degrees"""
    print("\nBenchmarking fpbspl (B-spline evaluation)...")
    
    degrees = [1, 2, 3, 4, 5]
    dierckx_times = []
    numba_times = []
    speedups = []
    
    for k in degrees:
        n = 2*k + 10
        t = np.concatenate([np.zeros(k+1), np.linspace(0, 1, n-2*k-2), np.ones(k+1)]).astype(np.float32)
        x = 0.5
        l = k + 2
        
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fpbspl, t, k, x, l, iterations=1000)
        
        # Numba timing
        numba_time = time_function(fpbspl_njit, t.astype(np.float64), n, k, x, l, iterations=1000)
        
        speedup = dierckx_time / numba_time if numba_time > 0 and dierckx_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        numba_times.append(numba_time)
        speedups.append(speedup)
        
        print(f"  k={k}: DIERCKX {dierckx_time:6.1f}μs, Numba {numba_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return degrees, dierckx_times, numba_times, speedups

def create_speedup_plots(results):
    """Create comprehensive speedup visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. fpback scaling speedup
    sizes, _, _, speedups = results['fpback']
    valid_data = [(s, sp) for s, sp in zip(sizes, speedups) if sp > 0]
    if valid_data:
        sizes_v, speedups_v = zip(*valid_data)
        ax1.semilogx(sizes_v, speedups_v, 'bo-', linewidth=2, markersize=8, label='Numba vs DIERCKX')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax1.set_xlabel('Matrix Size (n)')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('fpback Speedup vs Problem Size', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, max(speedups_v) * 1.1 if speedups_v else 2)
    
    # 2. fpbspl scaling speedup
    degrees, _, _, speedups = results['fpbspl']
    valid_data = [(d, sp) for d, sp in zip(degrees, speedups) if sp > 0]
    if valid_data:
        degrees_v, speedups_v = zip(*valid_data)
        ax2.plot(degrees_v, speedups_v, 'go-', linewidth=2, markersize=8, label='Numba vs DIERCKX')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax2.set_xlabel('Spline Degree (k)')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('fpbspl Speedup vs Spline Degree', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(degrees)
        ax2.set_ylim(0, max(speedups_v) * 1.1 if speedups_v else 2)
    
    # 3. Function speedup comparison
    functions = []
    avg_speedups = []
    
    for func_name, (_, _, _, speedups) in results.items():
        valid_speedups = [s for s in speedups if s > 0 and s != float('inf')]
        if valid_speedups:
            functions.append(func_name)
            avg_speedups.append(np.mean(valid_speedups))
    
    if functions:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(functions)]
        bars = ax3.bar(functions, avg_speedups, color=colors, alpha=0.8, edgecolor='black')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax3.set_ylabel('Average Speedup Factor')
        ax3.set_title('Average Speedup by Function', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()
        
        # Add value labels on bars
        for bar, speedup in zip(bars, avg_speedups):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speedup:.1f}×', ha='center', va='bottom', fontweight='bold')
    
    # 4. Execution time comparison
    func_names = []
    dierckx_avg = []
    numba_avg = []
    
    for func_name, (_, dierckx_times, numba_times, _) in results.items():
        valid_d = [t for t in dierckx_times if t > 0 and t != float('inf')]
        valid_n = [t for t in numba_times if t > 0 and t != float('inf')]
        
        if valid_d and valid_n:
            func_names.append(func_name)
            dierckx_avg.append(np.mean(valid_d))
            numba_avg.append(np.mean(valid_n))
    
    if func_names:
        x = np.arange(len(func_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, dierckx_avg, width, label='DIERCKX f2py', alpha=0.8, color='red')
        bars2 = ax4.bar(x + width/2, numba_avg, width, label='Numba', alpha=0.8, color='blue')
        
        ax4.set_xlabel('Function')
        ax4.set_ylabel('Execution Time (μs)')
        ax4.set_title('Execution Time Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(func_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_yscale('log')
    
    plt.suptitle('DIERCKX vs Numba Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plt.savefig('examples/dierckx_vs_numba_speedup.png', dpi=300, bbox_inches='tight')
    print("\n✓ Speedup plot saved as 'examples/dierckx_vs_numba_speedup.png'")
    
    return fig

def main():
    """Run comprehensive DIERCKX vs Numba benchmark"""
    print("DIERCKX vs NUMBA COMPREHENSIVE BENCHMARK")
    print("=" * 60)
    
    results = {}
    
    # Run all benchmarks
    results['fpback'] = benchmark_fpback_scaling()
    results['fpgivs'] = benchmark_fpgivs()
    results['fprota'] = benchmark_fprota()
    results['fprati'] = benchmark_fprati()
    results['fpbspl'] = benchmark_fpbspl_scaling()
    
    # Create plots
    create_speedup_plots(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for func_name, (_, _, _, speedups) in results.items():
        valid_speedups = [s for s in speedups if s > 0 and s != float('inf')]
        if valid_speedups:
            avg_speedup = np.mean(valid_speedups)
            max_speedup = max(valid_speedups)
            min_speedup = min(valid_speedups)
            print(f"{func_name:8s}: Avg {avg_speedup:4.1f}×, Range {min_speedup:4.1f}× - {max_speedup:4.1f}×")
        else:
            print(f"{func_name:8s}: No valid measurements")
    
    print("\n✓ Comprehensive benchmark complete!")
    return results

if __name__ == "__main__":
    main()