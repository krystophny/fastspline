"""
Test suite for parder_numba.py implementation.
Validates against scipy.interpolate.dfitpack.parder.
"""
import numpy as np
from scipy.interpolate import dfitpack, bisplrep
from parder_fixed import parder_fixed_cfunc_address
import ctypes

def test_parder_basic():
    """Test basic parder functionality against scipy dfitpack."""
    # Create simple test data - linear function
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X + Y  # Simple linear function
    
    # Create spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0)
    tx, ty, c = tck[0], tck[1], tck[2]
    kx, ky = 3, 3
    
    # Test single point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Test various derivatives
    test_cases = [
        (0, 0),  # no derivative - should be 1.0
        (1, 0),  # dx=1 - should be 1.0
        (0, 1),  # dy=1 - should be 1.0
        (1, 1),  # dx=1, dy=1 - should be 0.0
    ]
    
    for nux, nuy in test_cases:
        print(f"\nTesting derivative order ({nux}, {nuy}):")
        
        # scipy reference
        z_ref, ier_ref = dfitpack.parder(tx, ty, c, kx, ky, nux, nuy, xi, yi)
        
        # Numba implementation
        nx, ny = len(tx), len(ty)
        mx, my = len(xi), len(yi)
        
        # Setup arrays
        c_arr = np.asarray(c, dtype=np.float64)
        z_numba = np.zeros(mx * my, dtype=np.float64)
        
        # Workspace
        lwrk = (kx + 1 - nux) * mx + (ky + 1 - nuy) * my
        wrk = np.zeros(lwrk, dtype=np.float64)
        iwrk = np.zeros(mx + my, dtype=np.int32)
        ier = np.zeros(1, dtype=np.int32)
        
        # Call Numba cfunc
        parder_func = ctypes.CFUNCTYPE(None, 
                                       ctypes.POINTER(ctypes.c_double),  # tx
                                       ctypes.c_int32,                   # nx
                                       ctypes.POINTER(ctypes.c_double),  # ty
                                       ctypes.c_int32,                   # ny
                                       ctypes.POINTER(ctypes.c_double),  # c
                                       ctypes.c_int32,                   # kx
                                       ctypes.c_int32,                   # ky
                                       ctypes.c_int32,                   # nux
                                       ctypes.c_int32,                   # nuy
                                       ctypes.POINTER(ctypes.c_double),  # x
                                       ctypes.c_int32,                   # mx
                                       ctypes.POINTER(ctypes.c_double),  # y
                                       ctypes.c_int32,                   # my
                                       ctypes.POINTER(ctypes.c_double),  # z
                                       ctypes.POINTER(ctypes.c_double),  # wrk
                                       ctypes.c_int32,                   # lwrk
                                       ctypes.POINTER(ctypes.c_int32),   # iwrk
                                       ctypes.c_int32,                   # kwrk
                                       ctypes.POINTER(ctypes.c_int32)    # ier
                                       )(parder_fixed_cfunc_address)
        
        parder_func(tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx, 
                   ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny, 
                   c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), kx, ky, nux, nuy, 
                   xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx, 
                   yi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my, 
                   z_numba.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                   wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk, 
                   iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), mx + my, 
                   ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        
        # Compare results
        z_numba_2d = z_numba.reshape(mx, my)
        max_diff = np.max(np.abs(z_ref - z_numba_2d))
        
        print(f"  Error flag: scipy={ier_ref}, numba={ier[0]}")
        print(f"  Max difference: {max_diff:.2e}")
        
        if max_diff < 1e-14:
            print("  ✓ EXACT equality!")
        else:
            print("  ✗ Not exactly equal")
            # Show some values for debugging
            print(f"  scipy[0,0]: {z_ref[0,0]}")
            print(f"  numba[0,0]: {z_numba_2d[0,0]}")

if __name__ == "__main__":
    test_parder_basic()