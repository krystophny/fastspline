#!/usr/bin/env python3
"""
Comprehensive benchmark of all DIERCKX cfunc implementations
Generates scaling and speedup plots for all functions
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '..')

from dierckx_cfunc import (
    fpback_ultra, fpgivs_ultra, fprota_ultra, fprati_ultra, fpbspl_ultra,
    bisplrep_cfunc, bisplev_cfunc, bisplrep_cfunc_auto, warmup_ultra_functions
)
from scipy.interpolate import bisplrep, bisplev
import dierckx_f2py


def benchmark_low_level_functions():
    """Benchmark low-level DIERCKX functions"""
    print("\n" + "="*60)
    print("BENCHMARKING LOW-LEVEL DIERCKX FUNCTIONS")
    print("="*60)
    
    results = {}
    
    # 1. fpback - backward substitution
    print("\n1. fpback (backward substitution)")
    print("-"*40)
    sizes = [10, 20, 50, 100, 200, 500]
    fpback_times = {'dierckx': [], 'cfunc': [], 'sizes': sizes}
    
    for n in sizes:
        k = 3
        a = np.zeros((n, k), dtype=np.float64, order='F')
        for i in range(n):
            a[i, 0] = 2.0 + 0.1 * i
            for j in range(1, min(k, n-i)):
                a[i, j] = 0.1 / (j + 1)
        z = np.ones(n, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        
        # DIERCKX
        times = []
        for _ in range(100):
            c_copy = c.copy()
            start = time.perf_counter()
            dierckx_f2py.fpback(a, z, n, k, c_copy)
            times.append(time.perf_counter() - start)
        dierckx_time = np.median(times)
        fpback_times['dierckx'].append(dierckx_time)
        
        # cfunc
        times = []
        for _ in range(100):
            c_copy = c.copy()
            start = time.perf_counter()
            fpback_ultra(a, z, n, k, c_copy, n)
            times.append(time.perf_counter() - start)
        cfunc_time = np.median(times)
        fpback_times['cfunc'].append(cfunc_time)
        
        speedup = dierckx_time / cfunc_time
        print(f"  n={n:3d}: DIERCKX {dierckx_time*1e6:6.1f}μs, cfunc {cfunc_time*1e6:6.1f}μs, Speedup: {speedup:.2f}×")
    
    results['fpback'] = fpback_times
    
    # 2. fpgivs - Givens rotation
    print("\n2. fpgivs (Givens rotation)")
    print("-"*40)
    test_cases = [(3, 4), (1, 0), (0, 1), (-2, 3), (1e-10, 1), (1, 1e-10), (100, 0.01)]
    fpgivs_times = {'dierckx': [], 'cfunc': []}
    
    for piv, ww in test_cases:
        # DIERCKX
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = dierckx_f2py.fpgivs(piv, ww)
            times.append(time.perf_counter() - start)
        dierckx_time = np.median(times)
        fpgivs_times['dierckx'].append(dierckx_time)
        
        # cfunc
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = fpgivs_ultra(piv, ww)
            times.append(time.perf_counter() - start)
        cfunc_time = np.median(times)
        fpgivs_times['cfunc'].append(cfunc_time)
        
        speedup = dierckx_time / cfunc_time
        print(f"  ({piv},{ww}): DIERCKX {dierckx_time*1e6:6.1f}μs, cfunc {cfunc_time*1e6:6.1f}μs, Speedup: {speedup:.2f}×")
    
    results['fpgivs'] = fpgivs_times
    
    # 3. fpbspl - B-spline evaluation  
    print("\n3. fpbspl (B-spline evaluation)")
    print("-"*40)
    degrees = [1, 2, 3, 4, 5]
    fpbspl_times = {'dierckx': [], 'cfunc': [], 'degrees': degrees}
    
    for k in degrees:
        n = 11
        t = np.linspace(0, 1, n, dtype=np.float64)
        x = 0.5
        l = 5
        h = np.zeros(6, dtype=np.float64)
        
        # DIERCKX
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            dierckx_f2py.fpbspl(t, n, k, x, l, h)
            times.append(time.perf_counter() - start)
        dierckx_time = np.median(times)
        fpbspl_times['dierckx'].append(dierckx_time)
        
        # cfunc
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = fpbspl_ultra(t, n, k, x, l)
            times.append(time.perf_counter() - start)
        cfunc_time = np.median(times)
        fpbspl_times['cfunc'].append(cfunc_time)
        
        speedup = dierckx_time / cfunc_time
        print(f"  k={k}: DIERCKX {dierckx_time*1e6:6.1f}μs, cfunc {cfunc_time*1e6:6.1f}μs, Speedup: {speedup:.2f}×")
    
    results['fpbspl'] = fpbspl_times
    
    return results


def benchmark_bisplrep_bisplev():
    """Benchmark bivariate spline functions"""
    print("\n" + "="*60)
    print("BENCHMARKING BIVARIATE SPLINE FUNCTIONS")
    print("="*60)
    
    results = {}
    
    # Test different data sizes
    sizes = [16, 25, 36, 49, 64, 81, 100, 144, 196]
    degrees = [(1, 1), (2, 2), (3, 3)]
    
    for kx, ky in degrees:
        print(f"\nDegree (kx={kx}, ky={ky}):")
        print("-"*40)
        
        bisplrep_times = {'scipy': [], 'cfunc': [], 'sizes': []}
        bisplev_times = {'scipy': [], 'cfunc': [], 'sizes': []}
        
        for n in sizes:
            # Create regular grid data
            n_side = int(np.sqrt(n))
            x_1d = np.linspace(0, 1, n_side)
            y_1d = np.linspace(0, 1, n_side)
            X, Y = np.meshgrid(x_1d, y_1d)
            x = X.flatten()
            y = Y.flatten()
            z = x**2 + y**2 + 0.1*np.sin(4*np.pi*x)*np.cos(4*np.pi*y)
            
            try:
                # Benchmark bisplrep
                # SciPy
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    tck_scipy = bisplrep(x, y, z, kx=kx, ky=ky, s=0)
                    times.append(time.perf_counter() - start)
                scipy_time = np.median(times)
                
                # cfunc
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    tx, ty, c, _, _ = bisplrep_cfunc_auto(x, y, z, kx=kx, ky=ky, s=0.0)
                    times.append(time.perf_counter() - start)
                cfunc_time = np.median(times)
                
                speedup = scipy_time / cfunc_time
                print(f"  n={n:3d}: bisplrep - SciPy {scipy_time*1000:6.2f}ms, cfunc {cfunc_time*1000:6.2f}ms, Speedup: {speedup:5.2f}×")
                
                bisplrep_times['scipy'].append(scipy_time)
                bisplrep_times['cfunc'].append(cfunc_time)
                bisplrep_times['sizes'].append(n)
                
                # Benchmark bisplev
                # Create evaluation grid
                x_eval = np.linspace(0.1, 0.9, 20)
                y_eval = np.linspace(0.1, 0.9, 20)
                
                # SciPy
                times = []
                for _ in range(100):
                    start = time.perf_counter()
                    _ = bisplev(x_eval, y_eval, tck_scipy)
                    times.append(time.perf_counter() - start)
                scipy_time = np.median(times)
                
                # cfunc
                times = []
                for _ in range(100):
                    start = time.perf_counter()
                    _ = bisplev_cfunc(x_eval, y_eval, tx, ty, c, kx, ky)
                    times.append(time.perf_counter() - start)
                cfunc_time = np.median(times)
                
                speedup = scipy_time / cfunc_time
                print(f"           bisplev  - SciPy {scipy_time*1000:6.2f}ms, cfunc {cfunc_time*1000:6.2f}ms, Speedup: {speedup:5.2f}×")
                
                bisplev_times['scipy'].append(scipy_time)
                bisplev_times['cfunc'].append(cfunc_time)
                bisplev_times['sizes'].append(n)
                
            except Exception as e:
                print(f"  n={n:3d}: Failed - {str(e)}")
        
        key = f'bisplrep_k{kx}'
        results[key] = bisplrep_times
        key = f'bisplev_k{kx}'
        results[key] = bisplev_times
    
    # Test scattered data
    print("\nScattered Data Performance:")
    print("-"*40)
    
    sizes = [25, 50, 100, 200, 400]
    scattered_times = {'scipy': [], 'cfunc': [], 'sizes': []}
    
    for n in sizes:
        np.random.seed(42)
        x = np.random.uniform(0, 1, n)
        y = np.random.uniform(0, 1, n)
        z = x**2 + y**2
        
        try:
            # SciPy
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = bisplrep(x, y, z, s=0)
                times.append(time.perf_counter() - start)
            scipy_time = np.median(times)
            
            # cfunc  
            times = []
            for _ in range(5):
                start = time.perf_counter()
                _ = bisplrep_cfunc_auto(x, y, z, s=0.0)
                times.append(time.perf_counter() - start)
            cfunc_time = np.median(times)
            
            speedup = scipy_time / cfunc_time
            print(f"  n={n:3d}: SciPy {scipy_time*1000:6.2f}ms, cfunc {cfunc_time*1000:6.2f}ms, Speedup: {speedup:5.2f}×")
            
            scattered_times['scipy'].append(scipy_time)
            scattered_times['cfunc'].append(cfunc_time)
            scattered_times['sizes'].append(n)
            
        except Exception as e:
            print(f"  n={n:3d}: Failed - {str(e)}")
    
    results['scattered'] = scattered_times
    
    return results


def plot_results(low_level_results, high_level_results):
    """Create comprehensive performance plots"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('DIERCKX cfunc Performance: Scaling and Speedup Analysis', fontsize=16, fontweight='bold')
    
    # Low-level functions
    # fpback
    ax1 = plt.subplot(3, 3, 1)
    data = low_level_results['fpback']
    ax1.loglog(data['sizes'], np.array(data['dierckx'])*1e6, 'b-o', label='DIERCKX', linewidth=2)
    ax1.loglog(data['sizes'], np.array(data['cfunc'])*1e6, 'r-s', label='Numba cfunc', linewidth=2)
    ax1.set_xlabel('Problem Size (n)')
    ax1.set_ylabel('Time (μs)')
    ax1.set_title('fpback (Backward Substitution)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    speedup = np.array(data['dierckx']) / np.array(data['cfunc'])
    ax2.plot(data['sizes'], speedup, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Problem Size (n)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('fpback Speedup (DIERCKX/cfunc)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(2, max(speedup)*1.1))
    
    # fpbspl
    ax3 = plt.subplot(3, 3, 3)
    data = low_level_results['fpbspl']
    x_pos = np.arange(len(data['degrees']))
    width = 0.35
    ax3.bar(x_pos - width/2, np.array(data['dierckx'])*1e6, width, label='DIERCKX', color='blue', alpha=0.7)
    ax3.bar(x_pos + width/2, np.array(data['cfunc'])*1e6, width, label='Numba cfunc', color='red', alpha=0.7)
    ax3.set_xlabel('Degree (k)')
    ax3.set_ylabel('Time (μs)')
    ax3.set_title('fpbspl (B-spline Evaluation)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(data['degrees'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # bisplrep k=3
    ax4 = plt.subplot(3, 3, 4)
    if 'bisplrep_k3' in high_level_results:
        data = high_level_results['bisplrep_k3']
        ax4.loglog(data['sizes'], np.array(data['scipy'])*1000, 'b-o', label='SciPy', linewidth=2)
        ax4.loglog(data['sizes'], np.array(data['cfunc'])*1000, 'r-s', label='Numba cfunc', linewidth=2)
        ax4.set_xlabel('Number of Data Points')
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('bisplrep (k=3) - Regular Grid')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    ax5 = plt.subplot(3, 3, 5)
    if 'bisplrep_k3' in high_level_results:
        data = high_level_results['bisplrep_k3']
        speedup = np.array(data['scipy']) / np.array(data['cfunc'])
        ax5.semilogx(data['sizes'], speedup, 'g-o', linewidth=2, markersize=8)
        ax5.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Number of Data Points')
        ax5.set_ylabel('Speedup')
        ax5.set_title('bisplrep Speedup (SciPy/cfunc)')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, max(2, max(speedup)*1.1))
    
    # bisplev k=3
    ax6 = plt.subplot(3, 3, 6)
    if 'bisplev_k3' in high_level_results:
        data = high_level_results['bisplev_k3']
        ax6.loglog(data['sizes'], np.array(data['scipy'])*1000, 'b-o', label='SciPy', linewidth=2)
        ax6.loglog(data['sizes'], np.array(data['cfunc'])*1000, 'r-s', label='Numba cfunc', linewidth=2)
        ax6.set_xlabel('Number of Knots')
        ax6.set_ylabel('Time (ms)')
        ax6.set_title('bisplev (k=3) - 20×20 Evaluation')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    # Scattered data
    ax7 = plt.subplot(3, 3, 7)
    if 'scattered' in high_level_results:
        data = high_level_results['scattered']
        ax7.loglog(data['sizes'], np.array(data['scipy'])*1000, 'b-o', label='SciPy', linewidth=2)
        ax7.loglog(data['sizes'], np.array(data['cfunc'])*1000, 'r-s', label='Numba cfunc', linewidth=2)
        ax7.set_xlabel('Number of Data Points')
        ax7.set_ylabel('Time (ms)')
        ax7.set_title('bisplrep - Scattered Data')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    ax8 = plt.subplot(3, 3, 8)
    if 'scattered' in high_level_results:
        data = high_level_results['scattered']
        speedup = np.array(data['scipy']) / np.array(data['cfunc'])
        ax8.semilogx(data['sizes'], speedup, 'g-o', linewidth=2, markersize=8)
        ax8.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Number of Data Points')
        ax8.set_ylabel('Speedup')
        ax8.set_title('Scattered Data Speedup')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, max(2, max(speedup)*1.1))
    
    # Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = "Performance Summary:\n\n"
    summary_text += "Low-level functions:\n"
    summary_text += f"• fpback: up to {max(np.array(low_level_results['fpback']['dierckx']) / np.array(low_level_results['fpback']['cfunc'])):.1f}× speedup\n"
    summary_text += f"• fpbspl: {np.mean(np.array(low_level_results['fpbspl']['dierckx']) / np.array(low_level_results['fpbspl']['cfunc'])):.1f}× avg speedup\n\n"
    
    summary_text += "High-level functions:\n"
    if 'bisplrep_k3' in high_level_results:
        speedup = np.array(high_level_results['bisplrep_k3']['scipy']) / np.array(high_level_results['bisplrep_k3']['cfunc'])
        summary_text += f"• bisplrep: up to {max(speedup):.1f}× speedup\n"
    if 'bisplev_k3' in high_level_results:
        speedup = np.array(high_level_results['bisplev_k3']['scipy']) / np.array(high_level_results['bisplev_k3']['cfunc'])
        summary_text += f"• bisplev: up to {max(speedup):.1f}× speedup\n"
    if 'scattered' in high_level_results:
        speedup = np.array(high_level_results['scattered']['scipy']) / np.array(high_level_results['scattered']['cfunc'])
        summary_text += f"• scattered: up to {max(speedup):.1f}× speedup\n"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dierckx_cfunc_scaling_speedup.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comprehensive scaling and speedup plot saved as 'dierckx_cfunc_scaling_speedup.png'")


def main():
    print("COMPREHENSIVE DIERCKX CFUNC BENCHMARK")
    print("="*60)
    
    # Warmup
    print("\nWarming up functions...")
    warmup_ultra_functions()
    
    # Run benchmarks
    low_level_results = benchmark_low_level_functions()
    high_level_results = benchmark_bisplrep_bisplev()
    
    # Generate plots
    plot_results(low_level_results, high_level_results)
    
    print("\n✓ All benchmarks complete!")


if __name__ == "__main__":
    main()