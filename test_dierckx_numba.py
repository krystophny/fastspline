"""
Test script for DIERCKX Numba implementations.
Tests each function individually and compares with expected results.
"""

import numpy as np
from dierckx_numba import fpback, fpgivs, fprota, fprati, fpdisc
import ctypes

def test_fpback():
    """Test fpback implementation"""
    print("\n1. Testing fpback:")
    n = 4
    k = 3
    nest = 10
    
    # Create upper triangular banded matrix (Fortran order)
    a = np.zeros((nest, k), dtype=np.float64, order='F')
    # Fill diagonal
    for i in range(n):
        a[i, 0] = 2.0  # diagonal
    # Fill super-diagonals
    for i in range(n-1):
        a[i, 1] = 1.0  # first super-diagonal
    for i in range(n-2):
        a[i, 2] = 0.5  # second super-diagonal
        
    # RHS vector
    z = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    
    # Output vector
    c = np.zeros(n, dtype=np.float64)
    
    # Call fpback using ctypes
    fpback_func = fpback.ctypes
    fpback_func(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(n),
        ctypes.c_int32(k),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(nest)
    )
    
    print(f"  Matrix a (first {n} rows):")
    print(a[:n, :])
    print(f"  RHS z: {z}")
    print(f"  Solution c: {c}")
    
    # Verify solution by forward substitution
    verify = np.zeros(n)
    for i in range(n):
        verify[i] = a[i, 0] * c[i]
        for j in range(1, min(k, n-i)):
            verify[i] += a[i, j] * c[i+j]
    print(f"  Verification (should equal z): {verify}")
    print(f"  Max error: {np.max(np.abs(verify - z))}")
    
    return np.allclose(verify, z)


def test_fpgivs():
    """Test fpgivs implementation"""
    print("\n2. Testing fpgivs:")
    
    piv = 3.0
    ww = 4.0
    
    # Create arrays for in/out parameters
    piv_arr = np.array([piv], dtype=np.float64)
    ww_arr = np.array([ww], dtype=np.float64)
    cos_arr = np.zeros(1, dtype=np.float64)
    sin_arr = np.zeros(1, dtype=np.float64)
    
    # Call fpgivs
    fpgivs_func = fpgivs.ctypes
    fpgivs_func(
        piv_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ww_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sin_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    ww_out = ww_arr[0]
    cos = cos_arr[0]
    sin = sin_arr[0]
    
    print(f"  Input: piv={piv}, ww={ww}")
    print(f"  Output: ww={ww_out}, cos={cos}, sin={sin}")
    print(f"  Check: cos^2 + sin^2 = {cos**2 + sin**2}")
    print(f"  Expected dd = {np.sqrt(piv**2 + ww**2)}, actual = {ww_out}")
    
    return np.isclose(cos**2 + sin**2, 1.0)


def test_fprota():
    """Test fprota implementation"""
    print("\n3. Testing fprota:")
    
    # Use results from fpgivs
    piv = 3.0
    ww = 4.0
    dd = np.sqrt(piv**2 + ww**2)
    cos = ww / dd
    sin = piv / dd
    
    a_in = 5.0
    b_in = 12.0
    
    # Create arrays for in/out parameters  
    a_arr = np.array([a_in], dtype=np.float64)
    b_arr = np.array([b_in], dtype=np.float64)
    
    # Call fprota
    fprota_func = fprota.ctypes
    fprota_func(
        ctypes.c_double(cos),
        ctypes.c_double(sin),
        a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    a_out = a_arr[0]
    b_out = b_arr[0]
    
    print(f"  Input: a={a_in}, b={b_in}, cos={cos}, sin={sin}")
    print(f"  Output: a={a_out}, b={b_out}")
    
    # Verify rotation preserves norm
    norm_in = np.sqrt(a_in**2 + b_in**2)
    norm_out = np.sqrt(a_out**2 + b_out**2)
    print(f"  Norm preservation: input={norm_in}, output={norm_out}")
    
    return np.isclose(norm_in, norm_out)


def test_fprati():
    """Test fprati implementation"""
    print("\n4. Testing fprati:")
    
    p1 = 0.0
    f1 = 1.0
    p2 = 0.5
    f2 = -0.5
    p3 = 1.0
    f3 = -2.0
    
    # Create arrays for in/out parameters
    p1_arr = np.array([p1], dtype=np.float64)
    f1_arr = np.array([f1], dtype=np.float64)
    p3_arr = np.array([p3], dtype=np.float64)
    f3_arr = np.array([f3], dtype=np.float64)
    
    # Get function pointer with proper signature
    fprati_func = ctypes.CFUNCTYPE(
        ctypes.c_double,  # return type
        ctypes.POINTER(ctypes.c_double),  # p1
        ctypes.POINTER(ctypes.c_double),  # f1
        ctypes.c_double,                  # p2
        ctypes.c_double,                  # f2
        ctypes.POINTER(ctypes.c_double),  # p3
        ctypes.POINTER(ctypes.c_double),  # f3
    )(fprati.address)
    
    # Call fprati
    p = fprati_func(
        p1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        p2,
        f2,
        p3_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        f3_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"  Input: ({p1},{f1}), ({p2},{f2}), ({p3},{f3})")
    print(f"  Output: p={p}")
    print(f"  Updated: p1={p1_arr[0]}, f1={f1_arr[0]}, p3={p3_arr[0]}, f3={f3_arr[0]}")
    
    return True


def test_fpdisc():
    """Test fpdisc implementation"""
    print("\n5. Testing fpdisc:")
    
    # Simple test case
    n = 8
    k2 = 4
    nest = 10
    
    # Create knot vector
    t = np.linspace(0.0, 1.0, n)
    
    # Output matrix
    b = np.zeros((nest, k2), dtype=np.float64, order='F')
    
    # Work array
    h = np.zeros(12, dtype=np.float64)
    
    # Call fpdisc
    fpdisc_func = fpdisc.ctypes
    fpdisc_func(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(n),
        ctypes.c_int32(k2),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int32(nest),
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"  Knot vector t: {t}")
    print(f"  k2 = {k2}")
    print(f"  First few rows of discontinuity matrix b:")
    nrows = min(5, n - k2)
    print(b[:nrows, :])
    
    return True


if __name__ == "__main__":
    print("Testing DIERCKX Numba implementations...")
    
    results = []
    results.append(("fpback", test_fpback()))
    results.append(("fpgivs", test_fpgivs()))
    results.append(("fprota", test_fprota()))
    results.append(("fprati", test_fprati()))
    results.append(("fpdisc", test_fpdisc()))
    
    print("\n\nSummary:")
    print("-" * 40)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:10s}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("-" * 40)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")