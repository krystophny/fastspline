#!/usr/bin/env python3
"""
Test script to compare cfunc parder implementations against dfitpack.parder
"""
import numpy as np
import warnings
import ctypes
from scipy.interpolate import bisplrep, dfitpack

# Import the cfunc implementations
import sys
sys.path.append('fastspline/numba_implementation')
from parder_fixed import parder_fixed_cfunc_address
from parder_numba import parder_cfunc_address

def test_cfunc_vs_dfitpack():
    """Compare cfunc implementations with dfitpack.parder"""
    
    print('=== CFUNC vs DFITPACK COMPARISON ===')
    
    # Create test data
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2  # Known function
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test points
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Test derivatives
    derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    
    print(f'Testing at point (0.5, 0.5):')
    print('=' * 60)
    
    for nux, nuy in derivatives:
        print(f'\\nDerivative order ({nux}, {nuy}):')
        print('-' * 30)
        
        # dfitpack reference
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            z_dfitpack, ier_dfitpack = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
        
        print(f'dfitpack result: {z_dfitpack[0,0]:.10f} (ier={ier_dfitpack})')
        
        # Test cfunc implementations
        for cfunc_name, cfunc_addr in [('parder_fixed', parder_fixed_cfunc_address), 
                                       ('parder_numba', parder_cfunc_address)]:
            try:
                # Setup arrays for cfunc call
                nx, ny = len(tx), len(ty)
                mx, my = len(xi), len(yi)
                
                c_arr = np.asarray(c, dtype=np.float64)
                z_cfunc = np.zeros(mx * my, dtype=np.float64)
                
                # Workspace
                lwrk = (3 + 1 - nux) * mx + (3 + 1 - nuy) * my
                wrk = np.zeros(lwrk, dtype=np.float64)
                iwrk = np.zeros(mx + my, dtype=np.int32)
                ier = np.zeros(1, dtype=np.int32)
                
                # Create cfunc
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
                                               )(cfunc_addr)
                
                # Call cfunc
                parder_func(tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx, 
                           ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny, 
                           c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 3, 3, nux, nuy, 
                           xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx, 
                           yi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my, 
                           z_cfunc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
                           wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk, 
                           iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), mx + my, 
                           ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                
                # Compare results
                z_cfunc_val = z_cfunc[0]
                diff = abs(z_dfitpack[0,0] - z_cfunc_val)
                
                print(f'{cfunc_name} result: {z_cfunc_val:.10f} (ier={ier[0]})')
                print(f'  Difference: {diff:.2e}')
                
                if diff < 1e-14:
                    print(f'  ✓ EXACT match!')
                elif diff < 1e-10:
                    print(f'  ✓ Very close')
                else:
                    print(f'  ✗ MISMATCH!')
                
            except Exception as e:
                print(f'{cfunc_name} error: {e}')

if __name__ == '__main__':
    test_cfunc_vs_dfitpack()