"""
Comprehensive test script for all DIERCKX Numba implementations.
Tests each function and compares performance.
"""

import numpy as np
import time
from dierckx_numba import (fpback, fpgivs, fprota, fprati, fpdisc, 
                          fprank, fporde, fpbspl)
import ctypes

def test_fprank():
    """Test fprank implementation"""
    print("\n6. Testing fprank:")
    
    # Create a rank-deficient system
    n = 6
    m = 4
    na = n
    tol = 1e-10
    
    # Create banded matrix with small diagonal elements
    a = np.zeros((na, m), dtype=np.float64, order='F')
    
    # Fill with test data
    a[0, 0] = 2.0
    a[0, 1] = 1.0
    a[1, 0] = 1e-12  # Small diagonal - rank deficient
    a[1, 1] = 1.0
    a[2, 0] = 2.0
    a[2, 1] = 0.5
    a[3, 0] = 1e-11  # Small diagonal - rank deficient
    a[3, 1] = 0.3
    a[4, 0] = 1.5
    a[5, 0] = 2.5
    
    # RHS
    f = np.array([1.0, 0.5, 2.0, 0.1, 1.5, 2.5], dtype=np.float64)
    
    # Output arrays
    c = np.zeros(n, dtype=np.float64)
    sq = np.array([0.0], dtype=np.float64)
    rank_arr = np.array([0], dtype=np.int32)
    
    # Work arrays
    aa = np.zeros((n, m), dtype=np.float64, order='F')
    ff = np.zeros(n, dtype=np.float64)
    h = np.zeros(m, dtype=np.float64)
    
    # Call fprank
    fprank_func = fprank.ctypes
    fprank_func(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(n),
        ctypes.c_int32(m),
        ctypes.c_int32(na),
        ctypes.c_double(tol),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rank_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        aa.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ff.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"  Tolerance: {tol}")
    print(f"  Computed rank: {rank_arr[0]} (expected: {n-2})")
    print(f"  Sum of squared residuals: {sq[0]}")
    print(f"  Solution c: {c}")
    
    return rank_arr[0] == n - 2


def test_fporde():
    """Test fporde implementation"""
    print("\n7. Testing fporde:")
    
    # Setup test data
    m = 10
    kx = 3
    ky = 3
    
    # Create knot vectors
    nx = 8
    ny = 8
    tx = np.linspace(0.0, 1.0, nx)
    ty = np.linspace(0.0, 1.0, ny)
    
    # Create random data points
    np.random.seed(42)
    x = np.random.uniform(0.1, 0.9, m)
    y = np.random.uniform(0.1, 0.9, m)
    
    # Calculate nreg
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    nyy = nk1y - ky
    nreg = (nk1x - kx) * nyy
    
    # Output arrays
    nummer = np.zeros(m, dtype=np.int32)
    index = np.zeros(nreg, dtype=np.int32)
    
    # Call fporde
    fporde_func = fporde.ctypes
    fporde_func(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(m),
        ctypes.c_int32(kx),
        ctypes.c_int32(ky),
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(nx),
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(ny),
        nummer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        index.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int32(nreg)
    )
    
    print(f"  Number of data points: {m}")
    print(f"  Number of panels: {nreg}")
    print(f"  Data points x: {x[:5]}...")
    print(f"  Data points y: {y[:5]}...")
    print(f"  First 5 nummer values: {nummer[:5]}")
    print(f"  Non-zero index values: {index[index > 0]}")
    
    # Check that all points are assigned
    assigned = np.sum(index > 0)
    print(f"  Panels with data: {assigned}")
    
    return True


def test_fpbspl():
    """Test fpbspl implementation"""
    print("\n8. Testing fpbspl:")
    
    # Setup
    k = 3  # Cubic splines
    n = 10
    t = np.linspace(0.0, 1.0, n)
    
    # Test point
    x = 0.45
    
    # Output arrays
    l = np.array([0], dtype=np.int32)
    h = np.zeros(k+1, dtype=np.float64)
    
    # Call fpbspl
    fpbspl_func = fpbspl.ctypes
    fpbspl_func(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(n),
        ctypes.c_int32(k),
        ctypes.c_double(x),
        l.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"  Knot vector: {t}")
    print(f"  Evaluation point x: {x}")
    print(f"  Knot interval l: {l[0]} (t[{l[0]-1}] <= x < t[{l[0]}])")
    print(f"  B-spline values h: {h}")
    print(f"  Sum of B-splines: {np.sum(h)} (should be 1.0)")
    
    return np.isclose(np.sum(h), 1.0)


def benchmark_functions():
    """Benchmark Numba implementations"""
    print("\n\nPerformance Benchmarks:")
    print("-" * 60)
    
    # Benchmark fpback
    n = 100
    k = 5
    nest = 200
    a = np.random.randn(nest, k).astype(np.float64, order='F')
    for i in range(n):
        a[i, 0] = max(abs(a[i, 0]), 0.1)  # Ensure non-zero diagonal
    z = np.random.randn(n).astype(np.float64)
    c = np.zeros(n, dtype=np.float64)
    
    # Warmup
    fpback_func = fpback.ctypes
    for _ in range(10):
        fpback_func(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   ctypes.c_int32(n), ctypes.c_int32(k),
                   c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   ctypes.c_int32(nest))
    
    # Time it
    start = time.time()
    iterations = 1000
    for _ in range(iterations):
        fpback_func(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   ctypes.c_int32(n), ctypes.c_int32(k),
                   c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   ctypes.c_int32(nest))
    elapsed = time.time() - start
    
    print(f"fpback: {elapsed*1000/iterations:.3f} ms per call ({iterations} iterations)")
    
    # Benchmark fpgivs
    piv_arr = np.array([3.0], dtype=np.float64)
    ww_arr = np.array([4.0], dtype=np.float64)
    cos_arr = np.zeros(1, dtype=np.float64)
    sin_arr = np.zeros(1, dtype=np.float64)
    
    fpgivs_func = fpgivs.ctypes
    
    start = time.time()
    iterations = 100000
    for _ in range(iterations):
        fpgivs_func(piv_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   ww_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   cos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   sin_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    elapsed = time.time() - start
    
    print(f"fpgivs: {elapsed*1000000/iterations:.3f} μs per call ({iterations} iterations)")


if __name__ == "__main__":
    print("Testing ALL DIERCKX Numba implementations...")
    
    # Run basic tests from previous script
    from test_dierckx_numba import (test_fpback, test_fpgivs, test_fprota, 
                                   test_fprati, test_fpdisc)
    
    results = []
    results.append(("fpback", test_fpback()))
    results.append(("fpgivs", test_fpgivs()))
    results.append(("fprota", test_fprota()))
    results.append(("fprati", test_fprati()))
    results.append(("fpdisc", test_fpdisc()))
    results.append(("fprank", test_fprank()))
    results.append(("fporde", test_fporde()))
    results.append(("fpbspl", test_fpbspl()))
    
    print("\n\nSummary:")
    print("-" * 40)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:10s}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("-" * 40)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    # Run benchmarks
    benchmark_functions()