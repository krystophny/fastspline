#!/usr/bin/env python3
"""
Compare bispev vs parder for function evaluation (nux=0, nuy=0).
"""
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from fastspline.numba_implementation.bispev_numba import bispev_cfunc
from fastspline.numba_implementation.parder import call_parder_safe
import ctypes

def compare_function_evaluation():
    """Compare bispev_cfunc vs parder_cfunc for function evaluation"""
    print("=== Comparing Function Evaluation ===")
    
    # Create simple constant function
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.ones_like(X)  # Constant = 1
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Scipy reference
    z_scipy = bisplev(xi, yi, tck)
    print(f"Scipy bisplev: {z_scipy:.6f}")
    
    # Our bispev_cfunc
    try:
        # Setup for bispev_cfunc call
        nx, ny = len(tx), len(ty)
        mx, my = len(xi), len(yi)
        kx, ky = 3, 3
        
        tx_arr = np.asarray(tx, dtype=np.float64)
        ty_arr = np.asarray(ty, dtype=np.float64)
        c_arr = np.asarray(c, dtype=np.float64)
        xi_arr = np.asarray(xi, dtype=np.float64)
        yi_arr = np.asarray(yi, dtype=np.float64)
        
        z_bispev = np.zeros(mx * my, dtype=np.float64)
        
        lwrk = kx + ky + 2
        wrk = np.zeros(lwrk, dtype=np.float64)
        iwrk = np.zeros(kx + ky, dtype=np.int32)
        ier = np.zeros(1, dtype=np.int32)
        
        bispev_cfunc(
            tx_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nx,
            ty_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ny,
            c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            kx, ky,
            xi_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            mx,
            yi_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            my,
            z_bispev.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            lwrk,
            iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )
        
        print(f"Bispev cfunc: {z_bispev[0]:.6f}, ier={ier[0]}")
        
    except Exception as e:
        print(f"Bispev cfunc failed: {e}")
    
    # Our parder call
    try:
        z_parder, ier_parder = call_parder_safe(tx, ty, c, 3, 3, 0, 0, xi, yi)
        print(f"Parder cfunc: {z_parder[0]:.6f}, ier={ier_parder}")
    except Exception as e:
        print(f"Parder cfunc failed: {e}")

if __name__ == "__main__":
    compare_function_evaluation()