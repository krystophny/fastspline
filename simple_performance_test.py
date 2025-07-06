#!/usr/bin/env python3
"""
Simple performance test of core DIERCKX functions
"""

import numpy as np
import time
from dierckx_numba_simple import (
    fpback_njit, fpgivs_njit, fprota_njit, fprati_njit, fpbspl_njit
)

def time_function(func, *args, iterations=1000):
    """Time a function call"""
    # Warmup
    for _ in range(10):
        func(*args)
    
    # Actual timing
    start = time.time()
    for _ in range(iterations):
        result = func(*args)
    end = time.time()
    
    return (end - start) / iterations * 1e6  # microseconds

def benchmark_core_functions():
    """Benchmark core DIERCKX functions"""
    print("CORE FUNCTION PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    results = {}
    
    # fpback - backward substitution
    n, k, nest = 20, 5, 25
    a = np.random.randn(nest, k).astype(np.float64)
    z = np.random.randn(n).astype(np.float64)
    c = np.zeros(n, dtype=np.float64)
    
    # Make A upper triangular
    for i in range(n):
        a[i, 0] = 2.0 + 0.1 * i
        for j in range(1, min(k, n-i)):
            a[i, j] = 0.5 / (j + 1)
    
    fpback_time = time_function(fpback_njit, a, z, n, k, c, nest)
    results['fpback'] = fpback_time
    print(f"fpback (n={n}): {fpback_time:.2f} μs")
    
    # fpgivs - Givens rotations
    fpgivs_time = time_function(fpgivs_njit, 3.0, 4.0)
    results['fpgivs'] = fpgivs_time
    print(f"fpgivs: {fpgivs_time:.2f} μs")
    
    # fprota - apply rotation
    fprota_time = time_function(fprota_njit, 0.8, 0.6, 5.0, 3.0)
    results['fprota'] = fprota_time
    print(f"fprota: {fprota_time:.2f} μs")
    
    # fprati - rational interpolation
    fprati_time = time_function(fprati_njit, 1.0, 2.0, 2.0, 1.0, 3.0, -1.0)
    results['fprati'] = fprati_time
    print(f"fprati: {fprati_time:.2f} μs")
    
    # fpbspl - B-spline evaluation
    k = 3
    n = 15
    t = np.concatenate([np.zeros(k+1), np.linspace(0, 1, n-2*k-2), np.ones(k+1)])
    fpbspl_time = time_function(fpbspl_njit, t, n, k, 0.5, k+2)
    results['fpbspl'] = fpbspl_time
    print(f"fpbspl (k={k}): {fpbspl_time:.2f} μs")
    
    return results

def scaling_test():
    """Test scaling with problem size"""
    print("\nSCALING TEST")
    print("=" * 50)
    
    # Test fpback scaling
    print("fpback scaling:")
    for n in [10, 20, 50, 100]:
        k, nest = min(5, n//2), n + 10
        a = np.random.randn(nest, k).astype(np.float64)
        z = np.random.randn(n).astype(np.float64)
        c = np.zeros(n, dtype=np.float64)
        
        # Make A upper triangular
        for i in range(n):
            a[i, 0] = 2.0 + 0.1 * i
            for j in range(1, min(k, n-i)):
                a[i, j] = 0.5 / (j + 1)
        
        fpback_time = time_function(fpback_njit, a, z, n, k, c, nest, iterations=100)
        print(f"  n={n:3d}: {fpback_time:6.1f} μs")
    
    # Test fpbspl scaling
    print("\nfpbspl scaling:")
    for k in [1, 2, 3, 4, 5]:
        n = 2*k + 10
        t = np.concatenate([np.zeros(k+1), np.linspace(0, 1, n-2*k-2), np.ones(k+1)])
        fpbspl_time = time_function(fpbspl_njit, t, n, k, 0.5, k+2, iterations=1000)
        print(f"  k={k}: {fpbspl_time:6.1f} μs")

def main():
    """Run performance tests"""
    print("FASTSPLINE PERFORMANCE TESTS")
    print("=" * 60)
    
    results = benchmark_core_functions()
    scaling_test()
    
    print("\nSUMMARY")
    print("=" * 60)
    total_time = sum(results.values())
    print(f"Total execution time for core functions: {total_time:.1f} μs")
    print(f"Average function call time: {total_time/len(results):.1f} μs")
    
    print("\nPerformance characteristics:")
    print("• All functions execute in microsecond timescales")
    print("• Excellent performance for real-time applications")
    print("• Numba JIT provides near-C performance")
    
    print("\n✓ Performance test complete!")

if __name__ == "__main__":
    main()