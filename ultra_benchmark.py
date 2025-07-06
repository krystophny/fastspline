#!/usr/bin/env python3
"""
Ultra-optimized DIERCKX benchmark: cfunc vs DIERCKX f2py
Focus on working functions and maximum performance measurement
"""

import numpy as np
import time
import matplotlib.pyplot as plt

# Import ultra-optimized cfunc implementations
from dierckx_numba_ultra import (
    fpback_ultra, fpgivs_ultra, fprota_ultra, fprati_ultra,
    warmup_ultra_functions
)

# Import previous optimized implementations
from dierckx_numba_optimized import (
    fpback_njit as fpback_opt, fpgivs_njit as fpgivs_opt, 
    fprota_njit as fprota_opt, fprati_njit as fprati_opt
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
            return float('inf')  # Failed
    
    return np.median(times) * 1e6  # microseconds

def benchmark_fpback_scaling():
    """Benchmark fpback implementations across problem sizes"""
    print("Benchmarking fpback (backward substitution) - scaling analysis...")
    
    sizes = [10, 20, 50, 100, 200, 500]
    dierckx_times = []
    optimized_times = []
    ultra_times = []
    
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
        
        # Optimized Numba timing
        c_opt = np.zeros(n, dtype=np.float64)
        optimized_time = time_function(fpback_opt, a, z, n, k, c_opt, nest, iterations=100)
        
        # Ultra-optimized cfunc timing
        c_ultra = np.zeros(n, dtype=np.float64)
        ultra_time = time_function(fpback_ultra, a, z, n, k, c_ultra, nest, iterations=100)
        
        dierckx_times.append(dierckx_time)
        optimized_times.append(optimized_time)
        ultra_times.append(ultra_time)
        
        opt_speedup = dierckx_time / optimized_time if optimized_time > 0 else 0
        ultra_speedup = dierckx_time / ultra_time if ultra_time > 0 else 0
        
        print(f"  n={n:3d}: DIERCKX {dierckx_time:6.1f}μs, Opt {optimized_time:6.1f}μs ({opt_speedup:.2f}×), Ultra {ultra_time:6.1f}μs ({ultra_speedup:.2f}×)")
    
    return sizes, dierckx_times, optimized_times, ultra_times

def benchmark_simple_functions():
    """Benchmark simple functions for overhead analysis"""
    print("\nBenchmarking simple functions (fpgivs, fprota, fprati)...")
    
    results = {}
    
    # fpgivs benchmark
    print("  fpgivs (Givens rotation):")
    test_cases = [(3.0, 4.0), (1.0, 0.0), (0.0, 1.0)]
    dierckx_times = []
    optimized_times = []
    ultra_times = []
    
    for piv, ww in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fpgivs, piv, ww, iterations=10000)
        
        # Optimized timing
        optimized_time = time_function(fpgivs_opt, piv, ww, iterations=10000)
        
        # Ultra timing
        ultra_time = time_function(fpgivs_ultra, piv, ww, iterations=10000)
        
        dierckx_times.append(dierckx_time)
        optimized_times.append(optimized_time)
        ultra_times.append(ultra_time)
        
        opt_speedup = dierckx_time / optimized_time if optimized_time > 0 else 0
        ultra_speedup = dierckx_time / ultra_time if ultra_time > 0 else 0
        
        print(f"    ({piv},{ww}): DIERCKX {dierckx_time:.1f}μs, Opt {optimized_time:.1f}μs ({opt_speedup:.2f}×), Ultra {ultra_time:.1f}μs ({ultra_speedup:.2f}×)")
    
    results['fpgivs'] = (test_cases, dierckx_times, optimized_times, ultra_times)
    
    # fprota benchmark
    print("  fprota (apply rotation):")
    test_cases = [(0.8, 0.6, 1.0, 2.0), (1.0, 0.0, 3.0, 4.0), (0.707, 0.707, 1.0, 1.0)]
    dierckx_times = []
    optimized_times = []
    ultra_times = []
    
    for cos, sin, a, b in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fprota, cos, sin, a, b, iterations=10000)
        
        # Optimized timing
        optimized_time = time_function(fprota_opt, cos, sin, a, b, iterations=10000)
        
        # Ultra timing
        ultra_time = time_function(fprota_ultra, cos, sin, a, b, iterations=10000)
        
        dierckx_times.append(dierckx_time)
        optimized_times.append(optimized_time)
        ultra_times.append(ultra_time)
        
        opt_speedup = dierckx_time / optimized_time if optimized_time > 0 else 0
        ultra_speedup = dierckx_time / ultra_time if ultra_time > 0 else 0
        
        print(f"    rotation: DIERCKX {dierckx_time:.1f}μs, Opt {optimized_time:.1f}μs ({opt_speedup:.2f}×), Ultra {ultra_time:.1f}μs ({ultra_speedup:.2f}×)")
    
    results['fprota'] = (test_cases, dierckx_times, optimized_times, ultra_times)
    
    # fprati benchmark
    print("  fprati (rational interpolation):")
    test_cases = [(1.0, 2.0, 2.0, 1.0, 3.0, -1.0), (0.1, 1.1, 0.5, 0.5, 0.9, -0.1)]
    dierckx_times = []
    optimized_times = []
    ultra_times = []
    
    for p1, f1, p2, f2, p3, f3 in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fprati, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        # Optimized timing
        optimized_time = time_function(fprati_opt, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        # Ultra timing
        ultra_time = time_function(fprati_ultra, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        dierckx_times.append(dierckx_time)
        optimized_times.append(optimized_time)
        ultra_times.append(ultra_time)
        
        opt_speedup = dierckx_time / optimized_time if optimized_time > 0 else 0
        ultra_speedup = dierckx_time / ultra_time if ultra_time > 0 else 0
        
        print(f"    case: DIERCKX {dierckx_time:.1f}μs, Opt {optimized_time:.1f}μs ({opt_speedup:.2f}×), Ultra {ultra_time:.1f}μs ({ultra_speedup:.2f}×)")
    
    results['fprati'] = (test_cases, dierckx_times, optimized_times, ultra_times)
    
    return results

def create_ultra_performance_plot(fpback_data, simple_data):
    """Create comprehensive performance comparison plots"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. fpback scaling performance
    sizes, dierckx_times, opt_times, ultra_times = fpback_data
    
    ax1.loglog(sizes, dierckx_times, 'r-o', label='DIERCKX f2py', linewidth=2, markersize=8)
    ax1.loglog(sizes, opt_times, 'b-s', label='Optimized Numba', linewidth=2, markersize=8)
    ax1.loglog(sizes, ultra_times, 'g-^', label='Ultra cfunc', linewidth=2, markersize=8)
    ax1.set_xlabel('Problem Size (n)')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('fpback Performance Scaling', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. fpback speedup vs problem size
    opt_speedups = [d/o if o > 0 else 0 for d, o in zip(dierckx_times, opt_times)]
    ultra_speedups = [d/u if u > 0 else 0 for d, u in zip(dierckx_times, ultra_times)]
    
    ax2.semilogx(sizes, opt_speedups, 'b-s', label='Optimized vs DIERCKX', linewidth=2, markersize=8)
    ax2.semilogx(sizes, ultra_speedups, 'g-^', label='Ultra vs DIERCKX', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Problem Size (n)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('fpback Speedup Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Function overhead comparison
    functions = ['fpgivs', 'fprota', 'fprati']
    dierckx_avg = []
    opt_avg = []
    ultra_avg = []
    
    for func_name in functions:
        _, d_times, o_times, u_times = simple_data[func_name]
        dierckx_avg.append(np.mean(d_times))
        opt_avg.append(np.mean(o_times))
        ultra_avg.append(np.mean(u_times))
    
    x = np.arange(len(functions))
    width = 0.25
    
    bars1 = ax3.bar(x - width, dierckx_avg, width, label='DIERCKX f2py', alpha=0.8, color='red')
    bars2 = ax3.bar(x, opt_avg, width, label='Optimized', alpha=0.8, color='blue')
    bars3 = ax3.bar(x + width, ultra_avg, width, label='Ultra cfunc', alpha=0.8, color='green')
    
    ax3.set_xlabel('Function')
    ax3.set_ylabel('Execution Time (μs)')
    ax3.set_title('Function Overhead Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(functions)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # 4. Speedup summary
    speedup_data = []
    labels = []
    
    # fpback speedups (average)
    avg_opt_speedup = np.mean([s for s in opt_speedups if s > 0])
    avg_ultra_speedup = np.mean([s for s in ultra_speedups if s > 0])
    speedup_data.extend([avg_opt_speedup, avg_ultra_speedup])
    labels.extend(['fpback\n(Optimized)', 'fpback\n(Ultra)'])
    
    # Simple function speedups
    for func_name in functions:
        _, d_times, o_times, u_times = simple_data[func_name]
        opt_speedup = np.mean([d/o if o > 0 else 0 for d, o in zip(d_times, o_times)])
        ultra_speedup = np.mean([d/u if u > 0 else 0 for d, u in zip(d_times, u_times)])
        speedup_data.extend([opt_speedup, ultra_speedup])
        labels.extend([f'{func_name}\n(Optimized)', f'{func_name}\n(Ultra)'])
    
    colors = ['blue', 'green'] * 4
    bars = ax4.bar(range(len(speedup_data)), speedup_data, color=colors, alpha=0.8)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax4.set_xlabel('Implementation')
    ax4.set_ylabel('Speedup Factor')
    ax4.set_title('Overall Speedup Summary', fontweight='bold')
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    # Add value labels on bars
    for bar, speedup in zip(bars, speedup_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speedup:.1f}×', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Ultra-Optimized DIERCKX Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    plt.savefig('examples/ultra_dierckx_performance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Ultra performance plot saved as 'examples/ultra_dierckx_performance.png'")
    
    return fig

def main():
    """Run ultra-optimized DIERCKX benchmark"""
    print("ULTRA-OPTIMIZED DIERCKX BENCHMARK")
    print("=" * 60)
    print("Comparing: DIERCKX f2py vs Optimized Numba vs Ultra cfunc")
    print("=" * 60)
    
    # Warmup
    warmup_ultra_functions()
    
    # Run benchmarks
    fpback_data = benchmark_fpback_scaling()
    simple_data = benchmark_simple_functions()
    
    # Create performance plots
    create_ultra_performance_plot(fpback_data, simple_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("ULTRA BENCHMARK SUMMARY")
    print("=" * 60)
    
    sizes, dierckx_times, opt_times, ultra_times = fpback_data
    
    # fpback summary
    opt_speedups = [d/o if o > 0 else 0 for d, o in zip(dierckx_times, opt_times)]
    ultra_speedups = [d/u if u > 0 else 0 for d, u in zip(dierckx_times, ultra_times)]
    
    avg_opt_speedup = np.mean([s for s in opt_speedups if s > 0])
    avg_ultra_speedup = np.mean([s for s in ultra_speedups if s > 0])
    max_opt_speedup = max(opt_speedups)
    max_ultra_speedup = max(ultra_speedups)
    
    print(f"fpback (backward substitution):")
    print(f"  Optimized Numba: {avg_opt_speedup:.2f}× average, {max_opt_speedup:.2f}× maximum speedup")
    print(f"  Ultra cfunc:     {avg_ultra_speedup:.2f}× average, {max_ultra_speedup:.2f}× maximum speedup")
    
    # Simple functions summary
    for func_name in ['fpgivs', 'fprota', 'fprati']:
        _, d_times, o_times, u_times = simple_data[func_name]
        opt_speedup = np.mean([d/o if o > 0 else 0 for d, o in zip(d_times, o_times)])
        ultra_speedup = np.mean([d/u if u > 0 else 0 for d, u in zip(d_times, u_times)])
        
        print(f"{func_name}: Opt {opt_speedup:.2f}×, Ultra {ultra_speedup:.2f}× speedup")
    
    print("\n✓ Ultra-optimized benchmark complete!")
    
    return fpback_data, simple_data

if __name__ == "__main__":
    main()