"""
Test bispev cfunc implementation against scipy.
"""
import numpy as np
import ctypes
import sys
sys.path.insert(0, '..')
from scipy.interpolate import bisplrep, bisplev
from bispev_ctypes import bispev as bispev_fortran
sys.path.pop(0)
from bispev_numba import bispev_cfunc_address


def test_bispev_simple():
    """Test bispev with a simple constant spline."""
    print("Testing Numba bispev cfunc implementation...")
    
    # Create ctypes function
    bispev_c = ctypes.CFUNCTYPE(
        None,  # return type
        ctypes.POINTER(ctypes.c_double),  # tx
        ctypes.c_int,                      # nx
        ctypes.POINTER(ctypes.c_double),  # ty
        ctypes.c_int,                      # ny
        ctypes.POINTER(ctypes.c_double),  # c
        ctypes.c_int,                      # kx
        ctypes.c_int,                      # ky
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.c_int,                      # mx
        ctypes.POINTER(ctypes.c_double),  # y
        ctypes.c_int,                      # my
        ctypes.POINTER(ctypes.c_double),  # z
        ctypes.POINTER(ctypes.c_double),  # wrk
        ctypes.c_int,                      # lwrk
        ctypes.POINTER(ctypes.c_int),     # iwrk
        ctypes.c_int,                      # kwrk
        ctypes.POINTER(ctypes.c_int),     # ier
    )(bispev_cfunc_address)
    
    # Simple test: constant spline
    kx = ky = 3
    nx = ny = 8
    tx = ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.], dtype=np.float64)
    
    # Coefficients for constant value 2.0
    nc = (nx - kx - 1) * (ny - ky - 1)  # 4 * 4 = 16
    c = np.full(nc, 2.0, dtype=np.float64)
    
    # Evaluation points
    mx = my = 3
    x = y = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    
    # Output and work arrays
    z = np.zeros(mx * my, dtype=np.float64)
    lwrk = mx * (kx + 1) + my * (ky + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    kwrk = mx + my
    iwrk = np.zeros(kwrk, dtype=np.int32)
    ier = np.array([0], dtype=np.int32)
    
    # Call bispev
    bispev_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        kx, ky,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
        iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), kwrk,
        ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Error flag: {ier[0]}")
    print(f"Output z (should all be 2.0):")
    print(z.reshape((mx, my), order='F'))
    print(f"All values close to 2.0: {np.allclose(z, 2.0, rtol=1e-14)}")
    
    return z


def test_bispev_against_scipy():
    """Test against scipy's bisplev."""
    print("\nTesting Numba bispev against scipy...")
    
    # Create test data
    kx = ky = 3
    x_data = np.linspace(0, 1, 10)
    y_data = np.linspace(0, 1, 10)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = np.sin(2 * np.pi * x_grid) * np.cos(2 * np.pi * y_grid)
    
    # Fit spline with scipy
    tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01, quiet=1)
    tx, ty, c, kx_out, ky_out = tck
    
    # Test points
    x_eval = np.array([0.3, 0.5, 0.7], dtype=np.float64)
    y_eval = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    
    # Call scipy
    z_scipy = bisplev(x_eval, y_eval, tck)
    
    # Call our Numba implementation
    bispev_c = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    )(bispev_cfunc_address)
    
    mx = len(x_eval)
    my = len(y_eval)
    z_numba = np.zeros(mx * my, dtype=np.float64)
    lwrk = mx * (kx_out + 1) + my * (ky_out + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    kwrk = mx + my
    iwrk = np.zeros(kwrk, dtype=np.int32)
    ier = np.array([0], dtype=np.int32)
    
    # Convert scipy arrays
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    
    bispev_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(tx),
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(ty),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        kx_out, ky_out,
        x_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
        y_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
        z_numba.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
        iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), kwrk,
        ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    # Reshape to match scipy output
    z_numba_2d = z_numba.reshape((my, mx), order='F').T
    
    print(f"Error flag: {ier[0]}")
    print(f"Scipy shape: {z_scipy.shape}")
    print(f"Scipy values:\n{z_scipy}")
    print(f"\nNumba shape: {z_numba_2d.shape}")
    print(f"Numba values:\n{z_numba_2d}")
    print(f"\nMax difference: {np.max(np.abs(z_scipy - z_numba_2d))}")
    
    if np.allclose(z_scipy, z_numba_2d, rtol=1e-12):
        print("✓ Results match scipy!")


def test_bispev_against_fortran():
    """Test Numba implementation against Fortran wrapper."""
    print("\nTesting Numba bispev against Fortran wrapper...")
    
    # Create test spline
    kx = ky = 3
    tx = ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    c = np.random.randn(16)
    
    # Test points
    x = np.linspace(0.1, 0.9, 5)
    y = np.linspace(0.1, 0.9, 5)
    
    # Call Fortran wrapper
    z_fortran = bispev_fortran(tx, ty, c, kx, ky, x, y)
    
    # Call Numba implementation
    bispev_c = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    )(bispev_cfunc_address)
    
    mx = len(x)
    my = len(y)
    z_numba = np.zeros(mx * my, dtype=np.float64)
    lwrk = mx * (kx + 1) + my * (ky + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    kwrk = mx + my
    iwrk = np.zeros(kwrk, dtype=np.int32)
    ier = np.array([0], dtype=np.int32)
    
    # Ensure arrays are contiguous
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    bispev_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(tx),
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(ty),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        kx, ky,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
        z_numba.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
        iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), kwrk,
        ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    z_numba_2d = z_numba.reshape((my, mx), order='F').T
    
    print(f"Fortran result shape: {z_fortran.shape}")
    print(f"Numba result shape: {z_numba_2d.shape}")
    print(f"Max difference: {np.max(np.abs(z_fortran - z_numba_2d))}")
    
    if np.allclose(z_fortran, z_numba_2d, rtol=1e-14, atol=1e-14):
        print("✓ Numba matches Fortran exactly!")
    else:
        print("✗ Mismatch between Numba and Fortran")
        idx = np.unravel_index(np.argmax(np.abs(z_fortran - z_numba_2d)), z_fortran.shape)
        print(f"  Largest diff at {idx}: Fortran={z_fortran[idx]}, Numba={z_numba_2d[idx]}")


def test_error_handling():
    """Test error handling in bispev."""
    print("\nTesting error handling...")
    
    bispev_c = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    )(bispev_cfunc_address)
    
    # Test 1: Insufficient workspace
    kx = ky = 3
    tx = ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.], dtype=np.float64)
    c = np.ones(16, dtype=np.float64)
    x = y = np.array([0.5], dtype=np.float64)
    z = np.zeros(1, dtype=np.float64)
    wrk = np.zeros(1, dtype=np.float64)  # Too small!
    iwrk = np.zeros(2, dtype=np.int32)
    ier = np.array([0], dtype=np.int32)
    
    bispev_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 8,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 8,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        3, 3,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 2,
        ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Test 1 - Insufficient workspace: ier = {ier[0]} (expected 10)")
    
    # Test 2: Unsorted x array
    x_bad = np.array([0.5, 0.3, 0.7], dtype=np.float64)
    y = np.array([0.5], dtype=np.float64)
    z = np.zeros(3, dtype=np.float64)
    wrk = np.zeros(100, dtype=np.float64)
    iwrk = np.zeros(4, dtype=np.int32)
    ier[0] = 0
    
    bispev_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 8,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 8,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        3, 3,
        x_bad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 3,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1,
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 100,
        iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), 4,
        ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Test 2 - Unsorted x array: ier = {ier[0]} (expected 10)")


if __name__ == "__main__":
    test_bispev_simple()
    test_bispev_against_scipy()
    test_bispev_against_fortran()
    test_error_handling()