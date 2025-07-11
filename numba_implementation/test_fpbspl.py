"""
Test fpbspl cfunc implementation against Fortran.
"""
import numpy as np
import ctypes
from fpbspl_numba import fpbspl_cfunc_address


def test_fpbspl_cfunc():
    """Test fpbspl cfunc with direct ctypes call."""
    print("Testing fpbspl cfunc implementation...")
    
    # Create ctypes function from cfunc address
    fpbspl_c = ctypes.CFUNCTYPE(
        None,  # return type (void)
        ctypes.POINTER(ctypes.c_double),  # t
        ctypes.c_int,                      # n
        ctypes.c_int,                      # k
        ctypes.c_double,                   # x
        ctypes.c_int,                      # l
        ctypes.POINTER(ctypes.c_double),  # h
    )(fpbspl_cfunc_address)
    
    # Test case: cubic B-spline (k=3)
    k = 3
    n = 8
    t = np.array([0., 0., 0., 0., 1., 1., 1., 1.], dtype=np.float64)
    x = 0.5
    l = 4  # knot interval for x=0.5 (1-based)
    h = np.zeros(k + 1, dtype=np.float64)
    
    # Call cfunc
    fpbspl_c(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n,
        k,
        x,
        l,
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"Input: k={k}, x={x}, l={l}")
    print(f"Knots: {t}")
    print(f"B-spline values: {h}")
    
    # For cubic B-splines at x=0.5 with uniform knots [0,0,0,0,1,1,1,1]
    # We expect the Bernstein polynomials: [0.125, 0.375, 0.375, 0.125]
    expected = np.array([0.125, 0.375, 0.375, 0.125])
    
    if np.allclose(h, expected, rtol=1e-14):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
        print(f"Expected: {expected}")
        print(f"Max error: {np.max(np.abs(h - expected))}")
    
    return h


def test_fpbspl_against_scipy():
    """Test against scipy's B-spline evaluation if available."""
    try:
        from scipy.interpolate import BSpline
        print("\nTesting against scipy...")
        
        # Create a simple B-spline
        k = 3
        t = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
        c = np.array([1., 0., 0., 0.])  # Coefficients for first B-spline
        
        # Evaluate at x=0.5
        x = 0.5
        l = 4
        h = np.zeros(k + 1, dtype=np.float64)
        
        # Call our cfunc
        fpbspl_c = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        )(fpbspl_cfunc_address)
        
        fpbspl_c(
            t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(t),
            k,
            x,
            l,
            h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        # Compare with scipy
        spl = BSpline(t, c, k, extrapolate=False)
        scipy_val = spl(x)
        our_val = h[0]  # First B-spline value
        
        print(f"Our first B-spline value: {our_val}")
        print(f"Scipy evaluation with c=[1,0,0,0]: {scipy_val}")
        print(f"Match: {np.isclose(our_val, scipy_val)}")
        
    except ImportError:
        print("\nScipy not available for comparison")


def test_fpbspl_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")
    
    fpbspl_c = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    )(fpbspl_cfunc_address)
    
    # Test 1: Linear B-spline (k=1)
    k = 1
    n = 4
    t = np.array([0., 0.5, 1., 1.5], dtype=np.float64)
    x = 0.25
    l = 1  # interval [0, 0.5]
    h = np.zeros(k + 1, dtype=np.float64)
    
    fpbspl_c(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n, k, x, l,
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"Linear B-spline at x={x}: {h}")
    print(f"Sum (should be 1.0): {np.sum(h)}")
    
    # Test 2: Quintic B-spline (k=5)
    k = 5
    n = 12
    t = np.concatenate([np.zeros(6), np.ones(6)]).astype(np.float64)
    x = 0.3
    l = 6  # Only valid interval
    h = np.zeros(k + 1, dtype=np.float64)
    
    fpbspl_c(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n, k, x, l,
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    print(f"\nQuintic B-spline at x={x}: {h}")
    print(f"Sum (should be 1.0): {np.sum(h)}")


if __name__ == "__main__":
    test_fpbspl_cfunc()
    test_fpbspl_against_scipy()
    test_fpbspl_edge_cases()