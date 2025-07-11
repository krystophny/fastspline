"""
Test fpbisp cfunc implementation.
"""
import numpy as np
import ctypes
from fpbisp_numba import fpbisp_cfunc_address


def test_fpbisp_simple():
    """Test fpbisp with a simple constant spline."""
    print("Testing fpbisp cfunc implementation...")
    
    # Create ctypes function
    fpbisp_c = ctypes.CFUNCTYPE(
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
        ctypes.POINTER(ctypes.c_double),  # wx
        ctypes.POINTER(ctypes.c_double),  # wy
        ctypes.POINTER(ctypes.c_int),     # lx
        ctypes.POINTER(ctypes.c_int),     # ly
    )(fpbisp_cfunc_address)
    
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
    wx = np.zeros(mx * (kx + 1), dtype=np.float64)
    wy = np.zeros(my * (ky + 1), dtype=np.float64)
    lx = np.zeros(mx, dtype=np.int32)
    ly = np.zeros(my, dtype=np.int32)
    
    # Call fpbisp
    fpbisp_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        kx, ky,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ly.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    # Reshape output
    z_grid = z.reshape((mx, my), order='F')
    
    print(f"Evaluation points x: {x}")
    print(f"Evaluation points y: {y}")
    print(f"Output z (should all be 2.0):")
    print(z_grid)
    print(f"All values close to 2.0: {np.allclose(z, 2.0, rtol=1e-14)}")
    
    return z_grid


def test_fpbisp_linear():
    """Test with a linear function z = x + y."""
    print("\nTesting linear function z = x + y...")
    
    fpbisp_c = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    )(fpbisp_cfunc_address)
    
    # Create spline for z = x + y
    kx = ky = 1  # Linear
    nx = ny = 4
    tx = ty = np.array([0., 0., 1., 1.], dtype=np.float64)
    
    # For linear B-splines representing z = x + y
    # We need coefficients at the corners: (0,0), (0,1), (1,0), (1,1)
    # z values: 0, 1, 1, 2
    c = np.array([0., 1., 1., 2.], dtype=np.float64)
    
    # Test points
    mx = my = 5
    x = y = np.linspace(0.1, 0.9, 5, dtype=np.float64)
    
    # Arrays
    z = np.zeros(mx * my, dtype=np.float64)
    wx = np.zeros(mx * (kx + 1), dtype=np.float64)
    wy = np.zeros(my * (ky + 1), dtype=np.float64)
    lx = np.zeros(mx, dtype=np.int32)
    ly = np.zeros(my, dtype=np.int32)
    
    # Call
    fpbisp_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        kx, ky,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ly.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    # Check against expected values
    z_grid = z.reshape((mx, my), order='F')
    expected = np.zeros((mx, my))
    for i in range(mx):
        for j in range(my):
            expected[i, j] = x[i] + y[j]
    
    print(f"Output z:")
    print(z_grid)
    print(f"Expected (x + y):")
    print(expected)
    print(f"Max error: {np.max(np.abs(z_grid - expected))}")


def test_fpbisp_against_scipy():
    """Test against scipy's bisplev."""
    try:
        from scipy.interpolate import bisplrep, bisplev
        print("\nTesting against scipy.interpolate.bisplev...")
        
        # Create test data
        kx = ky = 3
        x_data = np.linspace(0, 1, 10)
        y_data = np.linspace(0, 1, 10)
        x_grid, y_grid = np.meshgrid(x_data, y_data)
        z_data = np.sin(2 * np.pi * x_grid) * np.cos(2 * np.pi * y_grid)
        
        # Fit spline with scipy
        tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01)
        tx, ty, c, kx_out, ky_out = tck
        
        # Test points
        x_eval = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        y_eval = np.array([0.2, 0.5, 0.8], dtype=np.float64)
        
        # Call scipy
        z_scipy = bisplev(x_eval, y_eval, tck)
        
        # Call our implementation
        fpbisp_c = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        )(fpbisp_cfunc_address)
        
        mx = len(x_eval)
        my = len(y_eval)
        z = np.zeros(mx * my, dtype=np.float64)
        wx = np.zeros(mx * (kx_out + 1), dtype=np.float64)
        wy = np.zeros(my * (ky_out + 1), dtype=np.float64)
        lx = np.zeros(mx, dtype=np.int32)
        ly = np.zeros(my, dtype=np.int32)
        
        # Convert scipy arrays to ensure they're contiguous
        tx = np.ascontiguousarray(tx, dtype=np.float64)
        ty = np.ascontiguousarray(ty, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)
        
        fpbisp_c(
            tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(tx),
            ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(ty),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            kx_out, ky_out,
            x_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
            y_eval.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            wx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            wy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ly.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        
        z_ours = z.reshape((mx, my), order='F')
        
        print(f"Scipy result shape: {z_scipy.shape}")
        print(f"Scipy values:\n{z_scipy}")
        print(f"\nOur result shape: {z_ours.shape}")
        print(f"Our values:\n{z_ours}")
        print(f"\nMax difference: {np.max(np.abs(z_scipy - z_ours))}")
        
        if np.allclose(z_scipy, z_ours, rtol=1e-12):
            print("✓ Results match scipy!")
        
    except ImportError:
        print("\nScipy not available for comparison")


if __name__ == "__main__":
    test_fpbisp_simple()
    test_fpbisp_linear()
    test_fpbisp_against_scipy()