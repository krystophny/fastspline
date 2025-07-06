#!/usr/bin/env python3
"""
Speed test: Compare optimized Numba vs DIERCKX f2py
Focus on getting Numba to match DIERCKX speed
"""

import numpy as np
import time
from dierckx_numba_simple import (
    fpback_njit, fpgivs_njit, fprota_njit, fprati_njit, fpbspl_njit
)
import dierckx_f2py_fixed as dierckx_f2py

def create_optimized_versions():
    """Create optimized versions with all performance flags"""
    from numba import njit
    
    @njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
    def fpback_opt(a, z, n, k, c, nest):
        k1 = k - 1
        c[n-1] = z[n-1] / a[n-1, 0]
        
        if n == 1:
            return
            
        for j in range(2, n + 1):
            i = n - j
            store = z[i]
            i1 = min(k1, j - 1)
            m = i
            for l in range(i1):
                m += 1
                store -= c[m] * a[i, l + 1]
            c[i] = store / a[i, 0]
    
    @njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
    def fpgivs_opt(piv, ww):
        store = abs(piv)
        if store >= ww:
            if piv != 0.0:
                ratio = ww / piv
                dd = store * (1.0 + ratio * ratio) ** 0.5
            else:
                dd = ww
        else:
            ratio = piv / ww
            dd = ww * (1.0 + ratio * ratio) ** 0.5
        
        return dd, ww / dd, piv / dd
    
    @njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
    def fprota_opt(cos, sin, a, b):
        return cos * a - sin * b, cos * b + sin * a
    
    @njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
    def fprati_opt(p1, f1, p2, f2, p3, f3):
        if p3 > 0.0:
            h1 = f1 * (f2 - f3)
            h2 = f2 * (f3 - f1)
            h3 = f3 * (f1 - f2)
            denom = p1*h1 + p2*h2 + p3*h3
            if abs(denom) > 1e-15:
                p = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2) / denom
            else:
                p = 0.0
        else:
            denom = (f1 - f2) * f3
            if abs(denom) > 1e-15:
                p = (p1*(f1-f3)*f2 - p2*(f2-f3)*f1) / denom
            else:
                p = 0.0
        
        if f2 < 0.0:
            return p, p1, f1, p2, f2
        else:
            return p, p2, f2, p3, f3
    
    @njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
    def fpbspl_opt(t, n, k, x, l):
        h = np.zeros(k + 1, dtype=np.float64)
        h[0] = 1.0
        
        for j in range(1, k + 1):
            saved = 0.0
            for r in range(j):
                alpha = (x - t[l + r - j - 1]) / (t[l + r - 1] - t[l + r - j - 1])
                temp = h[r]
                h[r] = saved + (1.0 - alpha) * temp
                saved = alpha * temp
            h[j] = saved
        
        return h
    
    return fpback_opt, fpgivs_opt, fprota_opt, fprati_opt, fpbspl_opt

def time_function_precise(func, *args, iterations=1000):
    """Precise timing with warmup"""
    # Warmup
    for _ in range(10):
        try:
            func(*args)
        except:
            pass
    
    # Timing
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

def benchmark_optimized_vs_dierckx():
    """Benchmark optimized Numba vs DIERCKX"""
    print("OPTIMIZED NUMBA vs DIERCKX SPEED TEST")
    print("=" * 50)
    
    # Get optimized functions
    fpback_opt, fpgivs_opt, fprota_opt, fprati_opt, fpbspl_opt = create_optimized_versions()
    
    # Warmup optimized functions
    print("Warming up optimized functions...")
    
    # Test data
    n, k, nest = 20, 5, 25
    a = np.random.randn(nest, k).astype(np.float64, order='F')
    # Make it upper triangular
    for i in range(n):
        a[i, 0] = 2.0 + 0.1 * i
        for j in range(1, min(k, n-i)):
            a[i, j] = 0.5 / (j + 1)
    z = np.random.randn(n).astype(np.float64)
    c = np.zeros(n, dtype=np.float64)
    
    # Warmup
    fpback_opt(a, z, n, k, c, nest)
    fpgivs_opt(3.0, 4.0)
    fprota_opt(0.8, 0.6, 1.0, 2.0)
    fprati_opt(1.0, 2.0, 2.0, 1.0, 3.0, -1.0)
    
    t = np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.], dtype=np.float64)
    fpbspl_opt(t, 9, 3, 0.5, 4)
    
    print("✓ Warmup complete\n")
    
    results = []
    
    # 1. fpback benchmark
    print("1. fpback (backward substitution):")
    dierckx_time = time_function_precise(dierckx_f2py.fpback, a.astype(np.float32), z.astype(np.float32), n, k, nest, iterations=1000)
    
    c_opt = np.zeros(n, dtype=np.float64)
    numba_time = time_function_precise(fpback_opt, a, z, n, k, c_opt, nest, iterations=1000)
    
    speedup = dierckx_time / numba_time if numba_time > 0 else 0
    results.append(('fpback', dierckx_time, numba_time, speedup))
    print(f"  DIERCKX: {dierckx_time:6.1f}μs")
    print(f"  Numba:   {numba_time:6.1f}μs")
    print(f"  Speedup: {speedup:6.2f}×\n")
    
    # 2. fpgivs benchmark
    print("2. fpgivs (Givens rotation):")
    dierckx_time = time_function_precise(dierckx_f2py.fpgivs, 3.0, 4.0, iterations=10000)
    numba_time = time_function_precise(fpgivs_opt, 3.0, 4.0, iterations=10000)
    
    speedup = dierckx_time / numba_time if numba_time > 0 else 0
    results.append(('fpgivs', dierckx_time, numba_time, speedup))
    print(f"  DIERCKX: {dierckx_time:6.1f}μs")
    print(f"  Numba:   {numba_time:6.1f}μs")
    print(f"  Speedup: {speedup:6.2f}×\n")
    
    # 3. fprota benchmark
    print("3. fprota (apply rotation):")
    dierckx_time = time_function_precise(dierckx_f2py.fprota, 0.8, 0.6, 1.0, 2.0, iterations=10000)
    numba_time = time_function_precise(fprota_opt, 0.8, 0.6, 1.0, 2.0, iterations=10000)
    
    speedup = dierckx_time / numba_time if numba_time > 0 else 0
    results.append(('fprota', dierckx_time, numba_time, speedup))
    print(f"  DIERCKX: {dierckx_time:6.1f}μs")
    print(f"  Numba:   {numba_time:6.1f}μs")
    print(f"  Speedup: {speedup:6.2f}×\n")
    
    # 4. fprati benchmark
    print("4. fprati (rational interpolation):")
    dierckx_time = time_function_precise(dierckx_f2py.fprati, 1.0, 2.0, 2.0, 1.0, 3.0, -1.0, iterations=10000)
    numba_time = time_function_precise(fprati_opt, 1.0, 2.0, 2.0, 1.0, 3.0, -1.0, iterations=10000)
    
    speedup = dierckx_time / numba_time if numba_time > 0 else 0
    results.append(('fprati', dierckx_time, numba_time, speedup))
    print(f"  DIERCKX: {dierckx_time:6.1f}μs")
    print(f"  Numba:   {numba_time:6.1f}μs")
    print(f"  Speedup: {speedup:6.2f}×\n")
    
    # 5. fpbspl benchmark
    print("5. fpbspl (B-spline evaluation):")
    t32 = t.astype(np.float32)
    dierckx_time = time_function_precise(dierckx_f2py.fpbspl, t32, 3, 0.5, 4, iterations=5000)
    numba_time = time_function_precise(fpbspl_opt, t, 9, 3, 0.5, 4, iterations=5000)
    
    speedup = dierckx_time / numba_time if numba_time > 0 else 0
    results.append(('fpbspl', dierckx_time, numba_time, speedup))
    print(f"  DIERCKX: {dierckx_time:6.1f}μs")
    print(f"  Numba:   {numba_time:6.1f}μs")
    print(f"  Speedup: {speedup:6.2f}×\n")
    
    # Summary
    print("=" * 50)
    print("OPTIMIZATION SUMMARY:")
    print("=" * 50)
    
    total_speedup = 0
    valid_results = 0
    
    for name, d_time, n_time, speedup in results:
        if speedup > 0 and speedup != float('inf'):
            print(f"{name:8s}: {speedup:5.2f}× speedup ({'FASTER' if speedup > 1.0 else 'slower'})")
            total_speedup += speedup
            valid_results += 1
        else:
            print(f"{name:8s}: FAILED")
    
    if valid_results > 0:
        avg_speedup = total_speedup / valid_results
        print(f"\nAverage speedup: {avg_speedup:.2f}×")
        
        if avg_speedup > 1.0:
            print("🎉 Optimized Numba is FASTER than DIERCKX f2py!")
        else:
            print("⚠️  DIERCKX f2py is still faster - need more optimization")
    
    return results

if __name__ == "__main__":
    benchmark_optimized_vs_dierckx()