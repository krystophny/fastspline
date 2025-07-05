"""
Optimized DIERCKX-compatible bisplrep with cfunc interface.
"""

import numpy as np
from numba import njit, cfunc, types
from .bisplrep_dierckx import bisplrep_dierckx_core


@cfunc(types.int64(types.float64[:], types.float64[:], types.float64[:],
                   types.float64[:], types.int64, types.int64, types.float64,
                   types.float64[:], types.float64[:], types.float64[:]),
       nopython=True, fastmath=True, boundscheck=False)
def bisplrep_cfunc(x, y, z, w, kx, ky, s, tx_out, ty_out, c_out):
    """
    FITPACK-compatible B-spline surface fitting using QR decomposition.
    
    C-compatible interface for maximum performance.
    
    Parameters:
    -----------
    x, y, z : Data points
    w : Weights
    kx, ky : Spline degrees
    s : Smoothing factor
    tx_out, ty_out : Pre-allocated knot arrays
    c_out : Pre-allocated coefficient array
    
    Returns:
    --------
    Packed integer: (nx << 32) | ny
    """
    nxest = len(tx_out)
    nyest = len(ty_out)
    
    nx, ny, fp = bisplrep_dierckx_core(x, y, z, w, kx, ky, s, nxest, nyest,
                                       tx_out, ty_out, c_out)
    
    return (nx << 32) | ny


def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
             kx=3, ky=3, task=0, s=0, eps=1e-16, tx=None, ty=None,
             nxest=None, nyest=None, wrk=None, lwrk1=None, lwrk2=None):
    """
    Find a bivariate B-spline representation of a surface.
    
    Optimized implementation using cfunc for performance.
    """
    # Convert inputs
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    
    m = len(x)
    
    if w is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
    
    # Estimate knot array sizes
    if nxest is None:
        # For interpolation, ensure we have enough knots
        if s == 0:
            nxest = max(2*(kx+1), int(np.sqrt(m) + kx + 1) + 2)
        else:
            nxest = max(2*(kx+1), min(int(kx + np.sqrt(m)), m//2 + kx + 1))
    if nyest is None:
        if s == 0:
            nyest = max(2*(ky+1), int(np.sqrt(m) + ky + 1) + 2)
        else:
            nyest = max(2*(ky+1), min(int(ky + np.sqrt(m)), m//2 + ky + 1))
    
    # Allocate arrays
    tx_arr = np.zeros(nxest)
    ty_arr = np.zeros(nyest)
    c_arr = np.zeros(nxest * nyest)
    
    # Call cfunc
    packed = bisplrep_cfunc(x, y, z, w, kx, ky, s, tx_arr, ty_arr, c_arr)
    
    # Unpack result
    nx = (packed >> 32) & 0xFFFFFFFF
    ny = packed & 0xFFFFFFFF
    
    # Extract results
    tx_out = tx_arr[:nx].copy()
    ty_out = ty_arr[:ny].copy()
    c_out = c_arr[:(nx-kx-1)*(ny-ky-1)].copy()
    
    return (tx_out, ty_out, c_out, kx, ky)