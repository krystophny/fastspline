"""
Focused benchmark comparing B-spline basis function evaluation only.
This tests the core computational kernel without the overhead of full surface evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dierckx_numba_simple import fpbspl_njit

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')


def benchmark_numpy_bspline(t, k, x_points, iterations=1000):
    """Benchmark naive NumPy B-spline evaluation"""
    n = len(t)
    n_eval = len(x_points)
    
    def eval_bspline_numpy(t, k, x):
        """Evaluate B-splines using NumPy (not optimized)"""
        n = len(t)
        # Find knot span
        l = k
        while l < n - k - 1 and x >= t[l + 1]:
            l = l + 1
            
        # Initialize B-splines
        h = np.zeros(k + 1)
        h[0] = 1.0
        
        # Build B-splines using de Boor recursion
        for j in range(1, k + 1):
            h_prev = h.copy()
            h[0] = 0.0
            for i in range(j):
                li = l + i + 1
                lj = li - j
                denom = t[li-1] - t[lj-1]
                if abs(denom) > 1e-14:
                    f = h_prev[i] / denom
                    h[i] = h[i] + f * (t[li-1] - x)
                    h[i+1] = f * (x - t[lj-1])
        
        return l, h
    
    start = time.time()
    for _ in range(iterations):
        for x in x_points:
            l, h = eval_bspline_numpy(t, k, x)
    elapsed = time.time() - start
    
    return elapsed / (iterations * n_eval)


def benchmark_numba_bspline(t, k, x_points, iterations=1000):
    """Benchmark Numba JIT-compiled B-spline evaluation"""
    n = len(t)
    n_eval = len(x_points)
    
    # Warmup
    for _ in range(10):
        for x in x_points[:5]:
            l, h = fpbspl_njit(t, n, k, x)
    
    start = time.time()
    for _ in range(iterations):
        for x in x_points:
            l, h = fpbspl_njit(t, n, k, x)
    elapsed = time.time() - start
    
    return elapsed / (iterations * n_eval)


def run_bspline_benchmarks():
    """Run B-spline basis function benchmarks"""
    print("B-SPLINE BASIS FUNCTION BENCHMARKS")
    print("=" * 60)
    
    # Test parameters
    degrees = [1, 2, 3, 4, 5]  # Linear to quintic
    n_knots_list = [10, 20, 50, 100, 200, 500]
    n_eval_points = 100
    iterations = 1000
    
    results = {
        'numpy': {},
        'numba': {},
        'speedup': {}
    }
    
    for k in degrees:
        results['numpy'][k] = []
        results['numba'][k] = []
        results['speedup'][k] = []
        
        print(f"\nDegree k = {k}:")
        print("-" * 40)
        print(f"{'N Knots':>8} {'NumPy (μs)':>12} {'Numba (μs)':>12} {'Speedup':>10}")
        print("-" * 40)
        
        for n_knots in n_knots_list:
            # Create uniform knot vector
            t = np.linspace(0, 1, n_knots)
            
            # Evaluation points
            x_points = np.linspace(t[k], t[-k-1], n_eval_points)
            
            # Benchmark
            numpy_time = benchmark_numpy_bspline(t, k, x_points, iterations) * 1e6  # to μs
            numba_time = benchmark_numba_bspline(t, k, x_points, iterations) * 1e6
            speedup = numpy_time / numba_time
            
            results['numpy'][k].append(numpy_time)
            results['numba'][k].append(numba_time)
            results['speedup'][k].append(speedup)
            
            print(f"{n_knots:>8} {numpy_time:>12.3f} {numba_time:>12.3f} {speedup:>10.1f}×")
    
    return results, n_knots_list


def plot_bspline_results(results, n_knots_list):
    """Create plots for B-spline benchmarks"""
    degrees = sorted(results['numpy'].keys())
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Execution time vs number of knots
    for k in degrees:
        ax1.loglog(n_knots_list, results['numpy'][k], 'o-', label=f'NumPy (k={k})')
        ax1.loglog(n_knots_list, results['numba'][k], 's--', label=f'Numba (k={k})')
    
    ax1.set_xlabel('Number of Knots', fontsize=12)
    ax1.set_ylabel('Time per Evaluation (μs)', fontsize=12)
    ax1.set_title('B-spline Evaluation Performance', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs number of knots
    for k in degrees:
        ax2.semilogx(n_knots_list, results['speedup'][k], 'o-', label=f'Degree k={k}', linewidth=2)
    
    ax2.set_xlabel('Number of Knots', fontsize=12)
    ax2.set_ylabel('Speedup Factor (NumPy/Numba)', fontsize=12)
    ax2.set_title('Numba Speedup for B-spline Evaluation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Time vs degree for fixed knot count
    fixed_knot_idx = 3  # Use 100 knots
    numpy_times = [results['numpy'][k][fixed_knot_idx] for k in degrees]
    numba_times = [results['numba'][k][fixed_knot_idx] for k in degrees]
    
    x = np.arange(len(degrees))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, numpy_times, width, label='NumPy', alpha=0.8)
    bars2 = ax3.bar(x + width/2, numba_times, width, label='Numba', alpha=0.8)
    
    ax3.set_xlabel('B-spline Degree', fontsize=12)
    ax3.set_ylabel('Time per Evaluation (μs)', fontsize=12)
    ax3.set_title(f'Performance vs Degree (n_knots={n_knots_list[fixed_knot_idx]})', 
                  fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(degrees)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add speedup labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        speedup = bar1.get_height() / bar2.get_height()
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(),
                f'{speedup:.1f}×', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('examples/benchmark_bspline_basis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'examples/benchmark_bspline_basis.png'")


def create_performance_summary(results, n_knots_list):
    """Create performance summary"""
    print("\n\nPERFORMANCE SUMMARY")
    print("=" * 80)
    
    degrees = sorted(results['numpy'].keys())
    
    # Average speedups by degree
    print("\nAverage Speedup by Degree:")
    print("-" * 40)
    for k in degrees:
        avg_speedup = np.mean(results['speedup'][k])
        min_speedup = np.min(results['speedup'][k])
        max_speedup = np.max(results['speedup'][k])
        print(f"Degree {k}: {avg_speedup:>6.1f}× (range: {min_speedup:.1f}× - {max_speedup:.1f}×)")
    
    # Overall statistics
    all_speedups = []
    for k in degrees:
        all_speedups.extend(results['speedup'][k])
    
    print("\nOverall Statistics:")
    print("-" * 40)
    print(f"Mean speedup:   {np.mean(all_speedups):>6.1f}×")
    print(f"Median speedup: {np.median(all_speedups):>6.1f}×")
    print(f"Min speedup:    {np.min(all_speedups):>6.1f}×")
    print(f"Max speedup:    {np.max(all_speedups):>6.1f}×")
    
    # Performance scaling
    print("\nScaling Analysis:")
    print("-" * 40)
    
    # Check scaling with number of knots
    for k in degrees[:3]:  # Just show first 3 degrees
        log_n = np.log(n_knots_list)
        log_time = np.log(results['numba'][k])
        slope, _ = np.polyfit(log_n, log_time, 1)
        print(f"Numba scaling for degree {k}: O(n^{slope:.2f})")


def benchmark_single_call():
    """Benchmark a single B-spline evaluation to show raw performance"""
    print("\nSINGLE CALL PERFORMANCE TEST")
    print("=" * 40)
    
    k = 3  # Cubic
    n = 20  # 20 knots
    t = np.linspace(0, 1, n)
    x = 0.5
    
    # Time single calls
    iterations = 100000
    
    # NumPy baseline (no JIT)
    def bspline_eval_simple(t, k, x):
        n = len(t)
        l = k
        while l < n - k - 1 and x >= t[l + 1]:
            l = l + 1
        return l
    
    start = time.time()
    for _ in range(iterations):
        l = bspline_eval_simple(t, k, x)
    numpy_time = (time.time() - start) / iterations * 1e9  # nanoseconds
    
    # Numba
    start = time.time()
    for _ in range(iterations):
        l, h = fpbspl_njit(t, n, k, x)
    numba_time = (time.time() - start) / iterations * 1e9
    
    print(f"Cubic B-spline evaluation at single point:")
    print(f"  NumPy (knot finding only): {numpy_time:>8.0f} ns")
    print(f"  Numba (full evaluation):   {numba_time:>8.0f} ns")
    print(f"  Ratio:                     {numpy_time/numba_time:>8.1f}×")


if __name__ == "__main__":
    # Create output directory
    os.makedirs('examples', exist_ok=True)
    
    # Run benchmarks
    results, n_knots_list = run_bspline_benchmarks()
    
    # Create plots
    plot_bspline_results(results, n_knots_list)
    
    # Create summary
    create_performance_summary(results, n_knots_list)
    
    # Single call benchmark
    benchmark_single_call()
    
    print("\n✓ B-spline benchmark complete! Check examples/ folder for plots.")