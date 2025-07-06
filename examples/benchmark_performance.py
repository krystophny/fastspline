#!/usr/bin/env python3
"""
Performance benchmark: DIERCKX f2py vs Numba cfunc implementation
Generates scaling plots and speedup analysis
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

# Import cfunc implementations
from dierckx_cfunc import (
    fpback_ultra as fpback_cfunc, fpgivs_ultra as fpgivs_cfunc, 
    fprota_ultra as fprota_cfunc, fprati_ultra as fprati_cfunc,
    warmup_ultra_functions as warmup_functions
)

# Import DIERCKX f2py reference
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
            return float('inf')
    
    return np.median(times) * 1e6  # microseconds

def benchmark_fpback():
    """Benchmark fpback with different matrix sizes"""
    print("Benchmarking fpback (backward substitution)...")
    
    sizes = [10, 20, 50, 100, 200, 500]
    dierckx_times = []
    cfunc_times = []
    speedups = []
    
    for n in sizes:
        k = min(5, n//3)
        nest = n + 10
        
        # Create test data
        a = np.zeros((nest, k), dtype=np.float64, order='F')
        for i in range(n):
            a[i, 0] = 2.0 + 0.1 * i
            for j in range(1, min(k, n-i)):
                a[i, j] = 0.5 / (j + 1)
        
        z = np.random.randn(n).astype(np.float64)
        
        # DIERCKX f2py timing
        a32 = a.astype(np.float32, order='F')
        z32 = z.astype(np.float32)
        dierckx_time = time_function(dierckx_f2py.fpback, a32, z32, n, k, nest, iterations=100)
        
        # cfunc timing
        c_cfunc = np.zeros(n, dtype=np.float64)
        cfunc_time = time_function(fpback_cfunc, a, z, n, k, c_cfunc, nest, iterations=100)
        
        speedup = dierckx_time / cfunc_time if cfunc_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        cfunc_times.append(cfunc_time)
        speedups.append(speedup)
        
        print(f"  n={n:3d}: DIERCKX {dierckx_time:6.1f}μs, cfunc {cfunc_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return sizes, dierckx_times, cfunc_times, speedups

def create_performance_plots(results):
    """Create performance visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. fpback scaling performance
    sizes, dierckx_times, cfunc_times, speedups = results
    
    ax1.loglog(sizes, dierckx_times, 'r-o', label='DIERCKX f2py', linewidth=2, markersize=8)
    ax1.loglog(sizes, cfunc_times, 'g-^', label='Numba cfunc', linewidth=2, markersize=8)
    ax1.set_xlabel('Problem Size (n)')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('fpback Performance Scaling', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. fpback speedup
    ax2.semilogx(sizes, speedups, 'b-s', label='cfunc vs DIERCKX', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Problem Size (n)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('fpback Speedup Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, max(speedups) * 1.1 if speedups else 2)
    
    # 3. Time comparison bar chart
    x = np.arange(len(sizes))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, dierckx_times, width, label='DIERCKX f2py', alpha=0.8, color='red')
    bars2 = ax3.bar(x + width/2, cfunc_times, width, label='Numba cfunc', alpha=0.8, color='green')
    
    ax3.set_xlabel('Problem Size')
    ax3.set_ylabel('Execution Time (μs)')
    ax3.set_title('Execution Time Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'n={s}' for s in sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # 4. Summary statistics
    avg_speedup = np.mean(speedups)
    max_speedup = max(speedups)
    min_speedup = min(speedups)
    
    ax4.text(0.1, 0.8, f'Average Speedup: {avg_speedup:.2f}×', fontsize=16, fontweight='bold')
    ax4.text(0.1, 0.6, f'Maximum Speedup: {max_speedup:.2f}×', fontsize=16)
    ax4.text(0.1, 0.4, f'Minimum Speedup: {min_speedup:.2f}×', fontsize=16)
    ax4.text(0.1, 0.2, f'Best Performance: n≥{sizes[speedups.index(max_speedup)]}', fontsize=16)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Performance Summary', fontweight='bold', fontsize=18)
    
    plt.suptitle('DIERCKX vs Numba cfunc Performance Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plt.savefig('dierckx_cfunc_performance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Performance plot saved as 'dierckx_cfunc_performance.png'")
    
    return fig

def main():
    """Run performance benchmark"""
    print("DIERCKX vs NUMBA CFUNC PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Warmup
    warmup_functions()
    
    # Run benchmarks
    results = benchmark_fpback()
    
    # Create plots
    create_performance_plots(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    sizes, dierckx_times, cfunc_times, speedups = results
    avg_speedup = np.mean(speedups)
    max_speedup = max(speedups)
    
    print(f"fpback average speedup: {avg_speedup:.2f}×")
    print(f"fpback maximum speedup: {max_speedup:.2f}× at n={sizes[speedups.index(max_speedup)]}")
    
    if avg_speedup > 1.0:
        print("\n✓ Numba cfunc is faster than DIERCKX f2py on average!")
    else:
        print("\n⚠ DIERCKX f2py is faster on average for these problem sizes")
    
    print("\n✓ Benchmark complete!")
    
    return results

if __name__ == "__main__":
    main()