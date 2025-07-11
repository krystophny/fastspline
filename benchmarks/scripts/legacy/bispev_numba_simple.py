"""
Simplified Numba integration example for DIERCKX bispev.
This demonstrates how to call the C wrapper from Numba-compiled code.
"""
import ctypes
import numpy as np
from numba import njit, types
from numba.core import cgutils
from numba.core.typing.ctypes_utils import to_ctypes
from pathlib import Path
import os

# Load the shared library
lib_path = Path(__file__).parent / "libbispev.so"
if not lib_path.exists():
    raise RuntimeError(f"Could not find libbispev.so. Please run 'make' first.")

lib = ctypes.CDLL(str(lib_path))

# Define the C function prototype
bispev_c = lib.bispev_c
bispev_c.argtypes = [
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
]
bispev_c.restype = ctypes.c_int

# Get the function address
bispev_c_addr = ctypes.cast(bispev_c, ctypes.c_void_p).value


@njit
def call_bispev_from_numba(tx, ty, c, kx, ky, x, y, z, wrk, iwrk, ier):
    """
    Low-level function to call bispev from Numba.
    All arrays must be pre-allocated with correct sizes.
    """
    nx = len(tx)
    ny = len(ty)
    mx = len(x)
    my = len(y)
    lwrk = len(wrk)
    kwrk = len(iwrk)
    
    # Use Numba's ctypes support to call the function
    import numba.core.typing.ctypes_utils as ctypes_utils
    from numba.core import types, cgutils
    from numba.core.pythonapi import make_arg_tuple
    
    # This is a placeholder - in practice you'd use numba's intrinsics
    # or cfunc for proper integration
    pass


def bispev_python_wrapper(tx, ty, c, kx, ky, x, y):
    """
    Python wrapper that calls the C function via ctypes.
    This can be used from regular Python code or as a reference.
    """
    # Convert inputs
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    nx = len(tx)
    ny = len(ty)
    mx = len(x)
    my = len(y)
    
    # Allocate output and workspace
    z = np.zeros(mx * my, dtype=np.float64)
    lwrk = mx * (kx + 1) + my * (ky + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    kwrk = mx + my
    iwrk = np.zeros(kwrk, dtype=np.int32)
    ier = np.array([0], dtype=np.int32)
    
    # Call C function
    result = bispev_c(
        tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        nx,
        ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ny,
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        kx,
        ky,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mx,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        my,
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        lwrk,
        iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        kwrk,
        ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    if ier[0] != 0:
        raise RuntimeError(f"bispev returned error code {ier[0]}")
    
    return z.reshape((my, mx), order='F').T


def demonstrate_usage():
    """Demonstrate the usage of the wrapper."""
    print("Demonstrating bispev wrapper usage...")
    
    # Create test spline
    kx = ky = 3
    tx = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    c = np.ones(16)
    
    # Evaluation points
    x = np.linspace(0.1, 0.9, 10)
    y = np.linspace(0.1, 0.9, 10)
    
    # Call wrapper
    z = bispev_python_wrapper(tx, ty, c, kx, ky, x, y)
    
    print(f"Output shape: {z.shape}")
    print(f"Output range: [{z.min():.6f}, {z.max():.6f}]")
    
    # Benchmark
    import time
    n_runs = 100
    
    start = time.perf_counter()
    for _ in range(n_runs):
        z = bispev_python_wrapper(tx, ty, c, kx, ky, x, y)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000
    print(f"\nAverage time per call: {avg_time:.3f} ms")
    
    return z


if __name__ == "__main__":
    demonstrate_usage()