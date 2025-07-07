#!/usr/bin/env python3
"""
Compare scipy bisplev vs the ACTUAL compiled Fortran extension _dfitpack.
This bypasses ALL Python layers and accesses the compiled .so directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import _dfitpack  # The actual compiled extension!

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

def bisplev_compiled_fortran(x, y, tck):
    """Direct call to compiled Fortran extension - true f2py wrapper."""
    tx, ty, c, kx, ky = tck
    
    # Ensure arrays are contiguous and correct dtype
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    # Direct call to compiled Fortran
    z, ier = _dfitpack.bispev(tx, ty, c, kx, ky, x, y)
    
    if ier == 10:
        raise ValueError("Invalid input data")
    
    return z

def bisplev_ultra_minimal(x, y, tx, ty, c, kx, ky):
    """Ultra-minimal - no checks, pre-prepared arrays."""
    # Absolute minimal overhead - direct fortran call
    z, ier = _dfitpack.bispev(tx, ty, c, kx, ky, x, y)
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

def measure_overhead_components():
    """Measure individual overhead components."""
    # Test arrays
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    
    n_timing = 1000
    
    # Time array operations
    start = time.perf_counter()
    for _ in range(n_timing):
        np.ascontiguousarray(x, dtype=np.float64)
        np.ascontiguousarray(y, dtype=np.float64)
    array_time = (time.perf_counter() - start) / n_timing
    
    # Time tuple unpacking
    tck = (x, y, x, 3, 3)  # Dummy tck
    start = time.perf_counter()
    for _ in range(n_timing):
        tx, ty, c, kx, ky = tck
    unpack_time = (time.perf_counter() - start) / n_timing
    
    return array_time * 1e6, unpack_time * 1e6

def main():
    print("="*60)
    print("scipy.bisplev vs COMPILED FORTRAN EXTENSION comparison")
    print("="*60)
    
    # Verify we're using the actual compiled extension
    print(f"\n_dfitpack module: {_dfitpack}")
    print(f"_dfitpack.bispev type: {type(_dfitpack.bispev)}")
    print(f"Module file: {_dfitpack.__file__}")
    
    # Measure overhead components
    array_overhead, unpack_overhead = measure_overhead_components()
    print(f"\nOverhead measurements:")
    print(f"  Array conversion: {array_overhead:.2f} µs")
    print(f"  Tuple unpacking: {unpack_overhead:.3f} µs")
    
    # Test with different data sizes
    sizes = [100, 500, 1000, 2000, 5000]
    
    # Storage for results
    bisplev_scipy_times = []
    bisplev_fortran_times = []
    bisplev_minimal_times = []
    
    bisplev_scipy_stds = []
    bisplev_fortran_stds = []
    bisplev_minimal_stds = []
    
    print("\nBenchmarking...")
    print("-" * 60)
    
    for n_points in sizes:
        print(f"\nTesting with {n_points} data points...")
        
        # Generate test data and fit spline
        x, y, z = generate_test_data(n_points)
        tck = bisplrep(x, y, z, s=n_points)  # Use smoothing
        tx, ty, c, kx, ky = tck
        
        # Generate evaluation grid - ensure correct format
        xi = np.ascontiguousarray(np.linspace(x.min(), x.max(), 50), dtype=np.float64)
        yi = np.ascontiguousarray(np.linspace(y.min(), y.max(), 50), dtype=np.float64)
        
        # Benchmark 1: Standard scipy bisplev
        mean_scipy, std_scipy, zi_scipy = benchmark_function(
            bisplev_scipy, xi, yi, tck, n_runs=100
        )
        bisplev_scipy_times.append(mean_scipy)
        bisplev_scipy_stds.append(std_scipy)
        print(f"  scipy.bisplev:          {mean_scipy*1e6:.2f} ± {std_scipy*1e6:.2f} µs")
        
        # Benchmark 2: Direct compiled Fortran
        mean_fortran, std_fortran, zi_fortran = benchmark_function(
            bisplev_compiled_fortran, xi, yi, tck, n_runs=100
        )
        bisplev_fortran_times.append(mean_fortran)
        bisplev_fortran_stds.append(std_fortran)
        print(f"  _dfitpack.bispev:       {mean_fortran*1e6:.2f} ± {std_fortran*1e6:.2f} µs")
        
        # Benchmark 3: Ultra-minimal
        mean_minimal, std_minimal, zi_minimal = benchmark_function(
            bisplev_ultra_minimal, xi, yi, tx, ty, c, kx, ky, n_runs=100
        )
        bisplev_minimal_times.append(mean_minimal)
        bisplev_minimal_stds.append(std_minimal)
        print(f"  Fortran (minimal):      {mean_minimal*1e6:.2f} ± {std_minimal*1e6:.2f} µs")
        
        # Calculate overhead
        overhead_scipy = (mean_scipy - mean_minimal) * 1e6
        overhead_fortran = (mean_fortran - mean_minimal) * 1e6
        print(f"  Scipy overhead:         {overhead_scipy:.2f} µs")
        print(f"  Direct call overhead:   {overhead_fortran:.2f} µs")
        
        # Verify results match
        if np.allclose(zi_scipy, zi_fortran, rtol=1e-10):
            print("  ✓ Results match exactly!")
        else:
            max_diff = np.max(np.abs(zi_scipy - zi_fortran))
            print(f"  ✗ Results differ! Max difference: {max_diff}")
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Absolute times with error bars
    ax1.errorbar(sizes, np.array(bisplev_scipy_times)*1e6, 
                 yerr=np.array(bisplev_scipy_stds)*1e6,
                 marker='o', label='scipy.bisplev', capsize=5, linewidth=2)
    ax1.errorbar(sizes, np.array(bisplev_fortran_times)*1e6, 
                 yerr=np.array(bisplev_fortran_stds)*1e6,
                 marker='s', label='_dfitpack.bispev', capsize=5, linewidth=2)
    ax1.errorbar(sizes, np.array(bisplev_minimal_times)*1e6, 
                 yerr=np.array(bisplev_minimal_stds)*1e6,
                 marker='^', label='Minimal Fortran', capsize=5, linewidth=2)
    ax1.set_xlabel('Number of original data points')
    ax1.set_ylabel('Time (µs)')
    ax1.set_title('Evaluation Time: scipy vs Direct Fortran (50x50 grid)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Overhead breakdown
    overhead_scipy = (np.array(bisplev_scipy_times) - np.array(bisplev_minimal_times)) * 1e6
    overhead_fortran = (np.array(bisplev_fortran_times) - np.array(bisplev_minimal_times)) * 1e6
    
    width = 0.35
    x_pos = np.arange(len(sizes))
    ax2.bar(x_pos - width/2, overhead_scipy, width, label='scipy overhead', alpha=0.7)
    ax2.bar(x_pos + width/2, overhead_fortran, width, label='direct overhead', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(s) for s in sizes])
    ax2.set_xlabel('Number of original data points')
    ax2.set_ylabel('Overhead (µs)')
    ax2.set_title('Absolute Overhead vs Minimal Fortran Call')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Plot 3: Percentage overhead
    pct_overhead_scipy = overhead_scipy / (np.array(bisplev_minimal_times) * 1e6) * 100
    pct_overhead_fortran = overhead_fortran / (np.array(bisplev_minimal_times) * 1e6) * 100
    
    ax3.plot(sizes, pct_overhead_scipy, marker='o', label='scipy overhead %', linewidth=2)
    ax3.plot(sizes, pct_overhead_fortran, marker='s', label='direct overhead %', linewidth=2)
    ax3.set_xlabel('Number of original data points')
    ax3.set_ylabel('Overhead (%)')
    ax3.set_title('Relative Overhead Percentage')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Summary
    ax4.text(0.5, 0.95, 'REAL F2PY Performance Analysis', 
             horizontalalignment='center', verticalalignment='top',
             transform=ax4.transAxes, fontsize=14, weight='bold')
    
    avg_overhead_scipy = np.mean(overhead_scipy)
    avg_overhead_fortran = np.mean(overhead_fortran)
    avg_pct_scipy = np.mean(pct_overhead_scipy)
    
    summary_text = f"""
Scipy bisplev overhead breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Array conversion:      ~{array_overhead:.1f} µs
• Tuple unpacking:       ~{unpack_overhead:.2f} µs  
• Input validation:      ~{avg_overhead_scipy - avg_overhead_fortran - array_overhead:.1f} µs
• Total scipy overhead:  ~{avg_overhead_scipy:.1f} µs ({avg_pct_scipy:.0f}%)

Direct _dfitpack.bispev benefits:
• Bypasses scipy's validation layer
• Direct access to compiled Fortran
• Minimal Python overhead
• Average overhead: ~{avg_overhead_fortran:.1f} µs

Conclusion:
The scipy interface adds ~{avg_overhead_scipy:.0f} µs overhead
which is ~{avg_pct_scipy:.0f}% of the total execution time.
For this use case, the overhead is minimal.
    """
    
    ax4.text(0.05, 0.85, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', family='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('scipy_vs_real_fortran_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'scipy_vs_real_fortran_comparison.png'")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print("="*60)
    print(f"Average scipy overhead: {avg_overhead_scipy:.1f} µs ({avg_pct_scipy:.0f}%)")
    print(f"Average direct overhead: {avg_overhead_fortran:.1f} µs")
    print(f"\nThis comparison uses the ACTUAL compiled Fortran extension")
    print("(_dfitpack.bispev) bypassing all Python wrapper layers.")

if __name__ == "__main__":
    main()