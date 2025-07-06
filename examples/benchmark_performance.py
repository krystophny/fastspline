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
    fpbspl_ultra as fpbspl_cfunc,
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
    print("\n1. Benchmarking fpback (backward substitution)...")
    print("-" * 60)
    
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
    
    return {'sizes': sizes, 'dierckx': dierckx_times, 'cfunc': cfunc_times, 'speedups': speedups}

def benchmark_fpgivs():
    """Benchmark fpgivs for various inputs"""
    print("\n2. Benchmarking fpgivs (Givens rotation)...")
    print("-" * 60)
    
    test_cases = [
        (3.0, 4.0), (1.0, 0.0), (0.0, 1.0), (-2.0, 3.0),
        (1e-10, 1.0), (1.0, 1e-10), (100.0, 0.01)
    ]
    
    dierckx_times = []
    cfunc_times = []
    speedups = []
    
    for piv, ww in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fpgivs, piv, ww, iterations=10000)
        
        # cfunc timing
        cfunc_time = time_function(fpgivs_cfunc, piv, ww, iterations=10000)
        
        speedup = dierckx_time / cfunc_time if cfunc_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        cfunc_times.append(cfunc_time)
        speedups.append(speedup)
        
        print(f"  ({piv:g},{ww:g}): DIERCKX {dierckx_time:6.1f}μs, cfunc {cfunc_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return {'cases': test_cases, 'dierckx': dierckx_times, 'cfunc': cfunc_times, 'speedups': speedups}

def benchmark_fprota():
    """Benchmark fprota for various rotations"""
    print("\n3. Benchmarking fprota (apply rotation)...")
    print("-" * 60)
    
    test_cases = [
        (0.8, 0.6, 1.0, 2.0), (1.0, 0.0, 3.0, 4.0),
        (0.707, 0.707, 1.0, 1.0), (0.6, -0.8, 5.0, -3.0)
    ]
    
    dierckx_times = []
    cfunc_times = []
    speedups = []
    
    for cos, sin, a, b in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fprota, cos, sin, a, b, iterations=10000)
        
        # cfunc timing
        cfunc_time = time_function(fprota_cfunc, cos, sin, a, b, iterations=10000)
        
        speedup = dierckx_time / cfunc_time if cfunc_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        cfunc_times.append(cfunc_time)
        speedups.append(speedup)
        
        print(f"  rotation: DIERCKX {dierckx_time:6.1f}μs, cfunc {cfunc_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return {'cases': test_cases, 'dierckx': dierckx_times, 'cfunc': cfunc_times, 'speedups': speedups}

def benchmark_fprati():
    """Benchmark fprati for rational interpolation"""
    print("\n4. Benchmarking fprati (rational interpolation)...")
    print("-" * 60)
    
    test_cases = [
        (1.0, 2.0, 2.0, 1.0, 3.0, -1.0),
        (0.1, 1.1, 0.5, 0.5, 0.9, -0.1),
        (10.0, 100.0, 5.0, 25.0, 1.0, 1.0)
    ]
    
    dierckx_times = []
    cfunc_times = []
    speedups = []
    
    for p1, f1, p2, f2, p3, f3 in test_cases:
        # DIERCKX f2py timing
        dierckx_time = time_function(dierckx_f2py.fprati, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        # cfunc timing
        cfunc_time = time_function(fprati_cfunc, p1, f1, p2, f2, p3, f3, iterations=10000)
        
        speedup = dierckx_time / cfunc_time if cfunc_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        cfunc_times.append(cfunc_time)
        speedups.append(speedup)
        
        print(f"  case: DIERCKX {dierckx_time:6.1f}μs, cfunc {cfunc_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return {'cases': test_cases, 'dierckx': dierckx_times, 'cfunc': cfunc_times, 'speedups': speedups}

def benchmark_fpbspl():
    """Benchmark fpbspl for different spline degrees"""
    print("\n5. Benchmarking fpbspl (B-spline evaluation)...")
    print("-" * 60)
    
    degrees = [1, 2, 3, 4, 5]
    dierckx_times = []
    cfunc_times = []
    speedups = []
    
    for k in degrees:
        # Create knot vector
        t = np.concatenate([np.zeros(k+1), np.linspace(0, 1, 8), np.ones(k+1)]).astype(np.float64)
        n = len(t)
        x = 0.5
        l = k + 4  # Middle interval
        
        # DIERCKX f2py timing
        t32 = t.astype(np.float32)
        dierckx_time = time_function(dierckx_f2py.fpbspl, t32, k, x, l, iterations=5000)
        
        # cfunc timing
        cfunc_time = time_function(fpbspl_cfunc, t, n, k, x, l, iterations=5000)
        
        speedup = dierckx_time / cfunc_time if cfunc_time > 0 else 0
        
        dierckx_times.append(dierckx_time)
        cfunc_times.append(cfunc_time)
        speedups.append(speedup)
        
        print(f"  k={k}: DIERCKX {dierckx_time:6.1f}μs, cfunc {cfunc_time:6.1f}μs, Speedup: {speedup:.2f}×")
    
    return {'degrees': degrees, 'dierckx': dierckx_times, 'cfunc': cfunc_times, 'speedups': speedups}

def create_all_performance_plots(fpback_results, fpgivs_results, fprota_results, 
                                fprati_results, fpbspl_results):
    """Create comprehensive performance visualization for all functions"""
    
    # Create a figure with 3x2 subplots
    fig = plt.figure(figsize=(18, 20))
    
    # 1. fpback scaling performance
    ax1 = plt.subplot(3, 2, 1)
    sizes = fpback_results['sizes']
    ax1.loglog(sizes, fpback_results['dierckx'], 'r-o', label='DIERCKX f2py', linewidth=2, markersize=8)
    ax1.loglog(sizes, fpback_results['cfunc'], 'g-^', label='Numba cfunc', linewidth=2, markersize=8)
    ax1.set_xlabel('Problem Size (n)')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.set_title('fpback Performance Scaling', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. fpback speedup
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogx(sizes, fpback_results['speedups'], 'b-s', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Problem Size (n)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('fpback Speedup Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, max(fpback_results['speedups']) * 1.1)
    
    # 3. All functions speedup comparison
    ax3 = plt.subplot(3, 2, 3)
    functions = ['fpback', 'fpgivs', 'fprota', 'fprati', 'fpbspl']
    avg_speedups = [
        np.mean(fpback_results['speedups']),
        np.mean(fpgivs_results['speedups']),
        np.mean(fprota_results['speedups']),
        np.mean(fprati_results['speedups']),
        np.mean(fpbspl_results['speedups'])
    ]
    max_speedups = [
        max(fpback_results['speedups']),
        max(fpgivs_results['speedups']),
        max(fprota_results['speedups']),
        max(fprati_results['speedups']),
        max(fpbspl_results['speedups'])
    ]
    
    x = np.arange(len(functions))
    width = 0.35
    bars1 = ax3.bar(x - width/2, avg_speedups, width, label='Average', alpha=0.8, color='blue')
    bars2 = ax3.bar(x + width/2, max_speedups, width, label='Maximum', alpha=0.8, color='green')
    
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Function')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('All Functions Speedup Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(functions)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. fpbspl degree comparison
    ax4 = plt.subplot(3, 2, 4)
    degrees = fpbspl_results['degrees']
    ax4.plot(degrees, fpbspl_results['dierckx'], 'r-o', label='DIERCKX f2py', linewidth=2, markersize=8)
    ax4.plot(degrees, fpbspl_results['cfunc'], 'g-^', label='Numba cfunc', linewidth=2, markersize=8)
    ax4.set_xlabel('B-spline Degree (k)')
    ax4.set_ylabel('Execution Time (μs)')
    ax4.set_title('fpbspl Performance vs Degree', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Execution time ranges
    ax5 = plt.subplot(3, 2, 5)
    func_names = []
    dierckx_ranges = []
    cfunc_ranges = []
    
    for name, results in [("fpback", fpback_results), ("fpgivs", fpgivs_results), 
                         ("fprota", fprota_results), ("fprati", fprati_results), 
                         ("fpbspl", fpbspl_results)]:
        func_names.append(name)
        dierckx_ranges.append((min(results['dierckx']), max(results['dierckx'])))
        cfunc_ranges.append((min(results['cfunc']), max(results['cfunc'])))
    
    y_pos = np.arange(len(func_names))
    for i, (name, d_range, c_range) in enumerate(zip(func_names, dierckx_ranges, cfunc_ranges)):
        ax5.barh(i - 0.2, d_range[1] - d_range[0], 0.4, left=d_range[0], 
                color='red', alpha=0.7, label='DIERCKX' if i == 0 else '')
        ax5.barh(i + 0.2, c_range[1] - c_range[0], 0.4, left=c_range[0], 
                color='green', alpha=0.7, label='cfunc' if i == 0 else '')
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(func_names)
    ax5.set_xlabel('Execution Time Range (μs)')
    ax5.set_title('Execution Time Ranges by Function', fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.legend()
    
    # 6. Summary statistics
    ax6 = plt.subplot(3, 2, 6)
    summary_text = "PERFORMANCE SUMMARY\n\n"
    
    for name, results in [("fpback", fpback_results), ("fpgivs", fpgivs_results), 
                         ("fprota", fprota_results), ("fprati", fprati_results), 
                         ("fpbspl", fpbspl_results)]:
        avg = np.mean(results['speedups'])
        max_val = max(results['speedups'])
        summary_text += f"{name}:\n"
        summary_text += f"  Avg: {avg:.2f}×  Max: {max_val:.2f}×\n"
    
    overall_avg = np.mean([np.mean(r['speedups']) for r in 
                          [fpback_results, fpgivs_results, fprota_results, fprati_results, fpbspl_results]])
    summary_text += f"\nOverall Average: {overall_avg:.2f}×"
    
    ax6.text(0.1, 0.9, summary_text, fontsize=14, verticalalignment='top', 
            fontfamily='monospace', transform=ax6.transAxes)
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    
    plt.suptitle('DIERCKX vs Numba cfunc Comprehensive Performance Analysis', 
                fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save plot
    plt.savefig('dierckx_cfunc_performance_all.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive performance plot saved as 'dierckx_cfunc_performance_all.png'")
    
    return fig

def main():
    """Run performance benchmark"""
    print("DIERCKX vs NUMBA CFUNC PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Warmup
    warmup_functions()
    
    # Run all benchmarks
    fpback_results = benchmark_fpback()
    fpgivs_results = benchmark_fpgivs()
    fprota_results = benchmark_fprota()
    fprati_results = benchmark_fprati()
    fpbspl_results = benchmark_fpbspl()
    
    # Create comprehensive plots
    create_all_performance_plots(fpback_results, fpgivs_results, fprota_results, 
                                fprati_results, fpbspl_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # fpback summary
    sizes, dierckx_times, cfunc_times, speedups = fpback_results['sizes'], fpback_results['dierckx'], fpback_results['cfunc'], fpback_results['speedups']
    avg_speedup = np.mean(speedups)
    max_speedup = max(speedups)
    print(f"\nfpback:")
    print(f"  Average speedup: {avg_speedup:.2f}×")
    print(f"  Maximum speedup: {max_speedup:.2f}× at n={sizes[speedups.index(max_speedup)]}")
    
    # Other functions summary
    for name, results in [("fpgivs", fpgivs_results), ("fprota", fprota_results), 
                         ("fprati", fprati_results), ("fpbspl", fpbspl_results)]:
        speedups = results['speedups']
        print(f"\n{name}:")
        print(f"  Average speedup: {np.mean(speedups):.2f}×")
        print(f"  Maximum speedup: {max(speedups):.2f}×")
    
    print("\n✓ Benchmark complete!")
    
    return {
        'fpback': fpback_results,
        'fpgivs': fpgivs_results,
        'fprota': fprota_results,
        'fprati': fprati_results,
        'fpbspl': fpbspl_results
    }

if __name__ == "__main__":
    main()