#!/usr/bin/env python3
"""
Compare runtime performance of scipy bisplev vs DIRECT dfitpack.bispev calls.
This script actually calls the Fortran wrappers directly, bypassing scipy's interface.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import dfitpack

def generate_test_data(n_points):
    """Generate synthetic 2D scattered data for testing."""
    np.random.seed(42)
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.sin(np.sqrt(x**2 + y**2)) + 0.1 * np.random.randn(n_points)
    return x, y, z

def bisplev_scipy(x, y, tck):
    """Standard scipy bisplev call - includes all validation and overhead."""
    return bisplev(x, y, tck)

def bisplev_dfitpack_direct(x, y, tck):
    """Direct call to dfitpack.bispev - bypasses scipy's validation and overhead."""
    tx, ty, c, kx, ky = tck
    
    # Minimal conversion - dfitpack expects arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Direct Fortran call
    z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
    
    if ier == 10:
        raise ValueError("Invalid input data")
    
    return z

def bisplev_dfitpack_minimal(x, y, tx, ty, c, kx, ky):
    """Ultra-minimal direct call - pre-unpacked tck, no conversions."""
    # Direct Fortran call with absolutely minimal overhead
    z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)
    return z

def benchmark_function(func, *args, n_runs=10, warmup=2):
    """Benchmark a function with warmup runs."""
    # Warmup runs
    for _ in range(warmup):
        func(*args)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times), result

def main():
    print("="*60)
    print("DIRECT scipy vs dfitpack.bispev comparison")
    print("="*60)
    
    # Verify we're using the right functions
    print(f"\nVerifying dfitpack.bispev exists: {hasattr(dfitpack, 'bispev')}")
    print(f"dfitpack module: {dfitpack}")
    
    # Test with different data sizes
    sizes = [100, 500, 1000, 2000, 5000]
    
    # Storage for results
    bisplev_scipy_times = []
    bisplev_direct_times = []
    bisplev_minimal_times = []
    
    bisplev_scipy_stds = []
    bisplev_direct_stds = []
    bisplev_minimal_stds = []
    
    print("\nBenchmarking scipy.bisplev vs dfitpack.bispev...")
    print("-" * 60)
    
    for n_points in sizes:
        print(f"\nTesting with {n_points} data points...")
        
        # Generate test data and fit spline
        x, y, z = generate_test_data(n_points)
        tck = bisplrep(x, y, z, s=n_points)  # Use smoothing
        tx, ty, c, kx, ky = tck
        
        # Generate evaluation grid
        xi = np.linspace(x.min(), x.max(), 50).astype(np.float64)
        yi = np.linspace(y.min(), y.max(), 50).astype(np.float64)
        
        # Benchmark 1: Standard scipy bisplev
        mean_scipy, std_scipy, zi_scipy = benchmark_function(
            bisplev_scipy, xi, yi, tck, n_runs=50
        )
        bisplev_scipy_times.append(mean_scipy)
        bisplev_scipy_stds.append(std_scipy)
        print(f"  scipy.bisplev:        {mean_scipy*1e6:.2f} ± {std_scipy*1e6:.2f} µs")
        
        # Benchmark 2: Direct dfitpack with minimal scipy overhead
        mean_direct, std_direct, zi_direct = benchmark_function(
            bisplev_dfitpack_direct, xi, yi, tck, n_runs=50
        )
        bisplev_direct_times.append(mean_direct)
        bisplev_direct_stds.append(std_direct)
        print(f"  dfitpack.bispev:      {mean_direct*1e6:.2f} ± {std_direct*1e6:.2f} µs")
        
        # Benchmark 3: Ultra-minimal with pre-unpacked tck
        mean_minimal, std_minimal, zi_minimal = benchmark_function(
            bisplev_dfitpack_minimal, xi, yi, tx, ty, c, kx, ky, n_runs=50
        )
        bisplev_minimal_times.append(mean_minimal)
        bisplev_minimal_stds.append(std_minimal)
        print(f"  dfitpack (minimal):   {mean_minimal*1e6:.2f} ± {std_minimal*1e6:.2f} µs")
        
        # Calculate speedups
        speedup_direct = mean_scipy / mean_direct
        speedup_minimal = mean_scipy / mean_minimal
        print(f"  Speedup (direct):     {speedup_direct:.2f}x")
        print(f"  Speedup (minimal):    {speedup_minimal:.2f}x")
        
        # Verify results match
        if np.allclose(zi_scipy, zi_direct, rtol=1e-10):
            print("  ✓ Results match exactly!")
        else:
            max_diff = np.max(np.abs(zi_scipy - zi_direct))
            print(f"  ✗ Results differ! Max difference: {max_diff}")
    
    # Create detailed performance plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Absolute times
    ax1.errorbar(sizes, np.array(bisplev_scipy_times)*1e6, 
                 yerr=np.array(bisplev_scipy_stds)*1e6,
                 marker='o', label='scipy.bisplev', capsize=5, linewidth=2)
    ax1.errorbar(sizes, np.array(bisplev_direct_times)*1e6, 
                 yerr=np.array(bisplev_direct_stds)*1e6,
                 marker='s', label='dfitpack.bispev (direct)', capsize=5, linewidth=2)
    ax1.errorbar(sizes, np.array(bisplev_minimal_times)*1e6, 
                 yerr=np.array(bisplev_minimal_stds)*1e6,
                 marker='^', label='dfitpack.bispev (minimal)', capsize=5, linewidth=2)
    ax1.set_xlabel('Number of original data points')
    ax1.set_ylabel('Time (µs)')
    ax1.set_title('Evaluation Time Comparison (50x50 grid)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Speedup factors
    speedups_direct = np.array(bisplev_scipy_times) / np.array(bisplev_direct_times)
    speedups_minimal = np.array(bisplev_scipy_times) / np.array(bisplev_minimal_times)
    
    ax2.plot(sizes, speedups_direct, marker='s', label='Direct call speedup', linewidth=2)
    ax2.plot(sizes, speedups_minimal, marker='^', label='Minimal call speedup', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of original data points')
    ax2.set_ylabel('Speedup factor')
    ax2.set_title('Speedup: dfitpack.bispev vs scipy.bisplev')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Overhead breakdown
    overhead_scipy = np.array(bisplev_scipy_times) - np.array(bisplev_minimal_times)
    overhead_direct = np.array(bisplev_direct_times) - np.array(bisplev_minimal_times)
    
    ax3.bar(range(len(sizes)), overhead_scipy * 1e6, width=0.4, 
            label='scipy overhead', alpha=0.7)
    ax3.bar(np.arange(len(sizes)) + 0.4, overhead_direct * 1e6, width=0.4,
            label='direct call overhead', alpha=0.7)
    ax3.set_xticks(range(len(sizes)))
    ax3.set_xticklabels([str(s) for s in sizes])
    ax3.set_xlabel('Number of original data points')
    ax3.set_ylabel('Overhead (µs)')
    ax3.set_title('Absolute Overhead vs Minimal Call')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # Plot 4: Summary text
    ax4.text(0.5, 0.95, 'Performance Analysis Summary', 
             horizontalalignment='center', verticalalignment='top',
             transform=ax4.transAxes, fontsize=14, weight='bold')
    
    avg_speedup_direct = np.mean(speedups_direct)
    avg_speedup_minimal = np.mean(speedups_minimal)
    avg_overhead_scipy = np.mean(overhead_scipy) * 1e6
    avg_overhead_direct = np.mean(overhead_direct) * 1e6
    
    summary_text = f"""
Average Performance Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Direct call speedup:   {avg_speedup_direct:.2f}x
• Minimal call speedup:  {avg_speedup_minimal:.2f}x
• Scipy overhead:        {avg_overhead_scipy:.1f} µs
• Direct overhead:       {avg_overhead_direct:.1f} µs

Scipy bisplev overhead includes:
• Input validation & type checking
• Array shape handling
• Error message formatting  
• Python function call stack

Direct dfitpack.bispev benefits:
• Bypasses all Python validation
• Direct Fortran array access
• Minimal error handling
• No shape manipulation
    """
    
    ax4.text(0.05, 0.85, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', family='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('scipy_vs_dfitpack_direct_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'scipy_vs_dfitpack_direct_comparison.png'")
    
    # Final summary
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print(f"Average speedup from direct dfitpack.bispev: {avg_speedup_direct:.2f}x")
    print(f"Average speedup from minimal calls: {avg_speedup_minimal:.2f}x")
    print(f"\nThis demonstrates ACTUAL direct f2py/Fortran wrapper performance")
    print("compared to scipy's Python interface layer.")

if __name__ == "__main__":
    main()