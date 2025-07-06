"""
Test the simplified DIERCKX Numba implementations.
"""

import numpy as np
import time
from dierckx_numba_simple import (fpback_njit, fpgivs_njit, fprota_njit, 
                                 fprati_njit, fpdisc_njit, fprank_njit,
                                 fporde_njit, fpbspl_njit)


def test_all_functions():
    """Test all DIERCKX functions"""
    
    print("Testing DIERCKX Numba implementations (simplified version)...")
    
    # Test fpback
    print("\n1. Testing fpback:")
    n = 4
    k = 3
    nest = 10
    
    a = np.zeros((nest, k), dtype=np.float64, order='F')
    for i in range(n):
        a[i, 0] = 2.0  # diagonal
    for i in range(n-1):
        a[i, 1] = 1.0  # first super-diagonal
    for i in range(n-2):
        a[i, 2] = 0.5  # second super-diagonal
        
    z = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    
    fpback_njit(a, z, n, k, c, nest)
    
    print(f"  Solution c: {c}")
    print(f"  Expected: [0.25 0.25 0.5  2.  ]")
    
    # Test fpgivs
    print("\n2. Testing fpgivs:")
    piv = 3.0
    ww = 4.0
    ww_out, cos, sin = fpgivs_njit(piv, ww)
    print(f"  Output: ww={ww_out}, cos={cos}, sin={sin}")
    print(f"  Check: cos^2 + sin^2 = {cos**2 + sin**2}")
    
    # Test fprota
    print("\n3. Testing fprota:")
    a_in = 5.0
    b_in = 12.0
    a_out, b_out = fprota_njit(cos, sin, a_in, b_in)
    print(f"  Output: a={a_out}, b={b_out}")
    
    # Test fprati
    print("\n4. Testing fprati:")
    p1, f1 = 0.0, 1.0
    p2, f2 = 0.5, -0.5
    p3, f3 = 1.0, -2.0
    p, p1_out, f1_out, p3_out, f3_out = fprati_njit(p1, f1, p2, f2, p3, f3)
    print(f"  Output: p={p}")
    
    # Test fpdisc
    print("\n5. Testing fpdisc:")
    n = 8
    k2 = 4
    nest = 10
    t = np.linspace(0.0, 1.0, n)
    b = np.zeros((nest, k2), dtype=np.float64, order='F')
    fpdisc_njit(t, n, k2, b, nest)
    print(f"  First few rows of b:")
    print(b[:3, :])
    
    # Test fprank
    print("\n6. Testing fprank:")
    n = 6
    m = 4
    na = n
    tol = 1e-10
    
    a = np.zeros((na, m), dtype=np.float64, order='F')
    a[0, 0] = 2.0
    a[0, 1] = 1.0
    a[1, 0] = 1e-12  # Small diagonal
    a[1, 1] = 1.0
    a[2, 0] = 2.0
    a[2, 1] = 0.5
    a[3, 0] = 1e-11  # Small diagonal
    a[3, 1] = 0.3
    a[4, 0] = 1.5
    a[5, 0] = 2.5
    
    f = np.array([1.0, 0.5, 2.0, 0.1, 1.5, 2.5], dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    aa = np.zeros((n, m), dtype=np.float64, order='F')
    ff = np.zeros(n, dtype=np.float64)
    h = np.zeros(m, dtype=np.float64)
    
    sq, rank = fprank_njit(a, f, n, m, na, tol, c, aa, ff, h)
    
    print(f"  Computed rank: {rank} (expected: {n-2})")
    print(f"  Sum of squared residuals: {sq}")
    
    # Test fporde
    print("\n7. Testing fporde:")
    m = 10
    kx = 3
    ky = 3
    nx = 8
    ny = 8
    tx = np.linspace(0.0, 1.0, nx)
    ty = np.linspace(0.0, 1.0, ny)
    
    np.random.seed(42)
    x = np.random.uniform(0.1, 0.9, m)
    y = np.random.uniform(0.1, 0.9, m)
    
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    nyy = nk1y - ky
    nreg = (nk1x - kx) * nyy
    
    nummer = np.zeros(m, dtype=np.int32)
    index = np.zeros(nreg, dtype=np.int32)
    
    fporde_njit(x, y, m, kx, ky, tx, nx, ty, ny, nummer, index, nreg)
    
    print(f"  Number of panels: {nreg}")
    print(f"  Panels with data: {np.sum(index > 0)}")
    
    # Test fpbspl
    print("\n8. Testing fpbspl:")
    k = 3
    n = 10
    t = np.linspace(0.0, 1.0, n)
    x = 0.45
    
    l, h = fpbspl_njit(t, n, k, x)
    
    print(f"  Knot interval l: {l}")
    print(f"  B-spline values: {h}")
    print(f"  Sum of B-splines: {np.sum(h)} (should be 1.0)")


def benchmark_functions():
    """Benchmark the implementations"""
    print("\n\nPerformance Benchmarks:")
    print("-" * 60)
    
    # Setup for fpback
    n = 100
    k = 5
    nest = 200
    a = np.random.randn(nest, k).astype(np.float64, order='F')
    for i in range(n):
        a[i, 0] = max(abs(a[i, 0]), 0.1)
    z = np.random.randn(n).astype(np.float64)
    c = np.zeros(n, dtype=np.float64)
    
    # Warmup and benchmark
    for _ in range(10):
        fpback_njit(a, z, n, k, c, nest)
    
    start = time.time()
    iterations = 10000
    for _ in range(iterations):
        fpback_njit(a, z, n, k, c, nest)
    elapsed = time.time() - start
    
    print(f"fpback: {elapsed*1000/iterations:.3f} ms per call")
    
    # Benchmark fpgivs
    start = time.time()
    iterations = 100000
    for _ in range(iterations):
        ww, cos, sin = fpgivs_njit(3.0, 4.0)
    elapsed = time.time() - start
    
    print(f"fpgivs: {elapsed*1000000/iterations:.3f} μs per call")
    
    # Benchmark fpbspl
    k = 3
    n = 20
    t = np.linspace(0.0, 1.0, n)
    
    start = time.time()
    iterations = 100000
    for _ in range(iterations):
        l, h = fpbspl_njit(t, n, k, 0.5)
    elapsed = time.time() - start
    
    print(f"fpbspl: {elapsed*1000000/iterations:.3f} μs per call")


if __name__ == "__main__":
    test_all_functions()
    benchmark_functions()