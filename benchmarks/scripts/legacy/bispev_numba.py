"""
Numba cfunc integration for the DIERCKX bispev C wrapper.
"""
import ctypes
import numpy as np
from numba import cfunc, njit, types
from numba.core import cgutils
from numba.core.typing import signature
from pathlib import Path
import os

# Load the shared library
lib_path = Path(__file__).parent / "libbispev.so"
if not lib_path.exists():
    for alt_path in ["./libbispev.so", "/usr/local/lib/libbispev.so"]:
        if os.path.exists(alt_path):
            lib_path = alt_path
            break
    else:
        raise RuntimeError(f"Could not find libbispev.so. Please run 'make' first.")

lib = ctypes.CDLL(str(lib_path))

# Get the function pointer
bispev_c_ptr = lib.bispev_c

# Define the cfunc signature for bispev_c
# int bispev_c(const double* tx, int nx, const double* ty, int ny,
#              const double* c, int kx, int ky,
#              const double* x, int mx, const double* y, int my,
#              double* z, double* wrk, int lwrk, int* iwrk, int kwrk,
#              int* ier);

bispev_c_sig = types.int32(
    types.CPointer(types.float64),  # tx
    types.int32,                     # nx
    types.CPointer(types.float64),  # ty
    types.int32,                     # ny
    types.CPointer(types.float64),  # c
    types.int32,                     # kx
    types.int32,                     # ky
    types.CPointer(types.float64),  # x
    types.int32,                     # mx
    types.CPointer(types.float64),  # y
    types.int32,                     # my
    types.CPointer(types.float64),  # z
    types.CPointer(types.float64),  # wrk
    types.int32,                     # lwrk
    types.CPointer(types.int32),    # iwrk
    types.int32,                     # kwrk
    types.CPointer(types.int32),    # ier
)

# Create the cfunc
bispev_cfunc = cfunc(bispev_c_sig, nopython=True)(bispev_c_ptr)


@njit
def bispev_numba(tx, ty, c, kx, ky, x, y):
    """
    Numba-compatible wrapper for bispev evaluation.
    
    Parameters
    ----------
    tx : array
        Knots in x-direction
    ty : array
        Knots in y-direction
    c : array
        B-spline coefficients
    kx : int
        Degree of spline in x-direction
    ky : int
        Degree of spline in y-direction
    x : array
        X-coordinates at which to evaluate
    y : array
        Y-coordinates at which to evaluate
        
    Returns
    -------
    z : array
        Values of the spline at (x[i], y[j]) for all i,j
        Shape is (len(x), len(y))
    """
    nx = len(tx)
    ny = len(ty)
    mx = len(x)
    my = len(y)
    
    # Ensure arrays are contiguous
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    # Allocate output and workspace
    z = np.zeros(mx * my, dtype=np.float64)
    lwrk = mx * (kx + 1) + my * (ky + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    kwrk = mx + my
    iwrk = np.zeros(kwrk, dtype=np.int32)
    
    # Error flag
    ier = np.zeros(1, dtype=np.int32)
    
    # Get data pointers using numba's intrinsic
    from numba.core import cgutils
    
    # Call the C function through cfunc
    result = bispev_cfunc(
        tx.ctypes,
        nx,
        ty.ctypes,
        ny,
        c.ctypes,
        kx,
        ky,
        x.ctypes,
        mx,
        y.ctypes,
        my,
        z.ctypes,
        wrk.ctypes,
        lwrk,
        iwrk.ctypes,
        kwrk,
        ier.ctypes
    )
    
    if ier[0] != 0:
        # In numba, we can't raise exceptions easily, so return NaN array
        return np.full((mx, my), np.nan)
    
    # Reshape output
    return z.reshape((my, mx)).T


@njit
def evaluate_spline_grid(tx, ty, c, kx, ky, x_min, x_max, y_min, y_max, nx, ny):
    """
    Example of using bispev in a numba-jitted function to evaluate on a grid.
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    
    # Call our numba wrapper
    z = bispev_numba(tx, ty, c, kx, ky, x, y)
    
    return x, y, z


def test_numba_integration():
    """Test the Numba integration."""
    print("Testing Numba cfunc integration...")
    
    # Create test data
    kx = ky = 3
    tx = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    c = np.ones(16)
    
    # Test points
    x = np.linspace(0.1, 0.9, 5)
    y = np.linspace(0.1, 0.9, 5)
    
    # Call through numba
    z = bispev_numba(tx, ty, c, kx, ky, x, y)
    print(f"Output shape: {z.shape}")
    print(f"Output values:\n{z}")
    
    # Test the grid evaluation function
    x_grid, y_grid, z_grid = evaluate_spline_grid(
        tx, ty, c, kx, ky, 0.1, 0.9, 0.1, 0.9, 10, 10
    )
    print(f"\nGrid evaluation shape: {z_grid.shape}")
    print(f"Grid min/max: {z_grid.min():.6f} / {z_grid.max():.6f}")
    
    # Benchmark
    import time
    
    # Compile
    _ = bispev_numba(tx, ty, c, kx, ky, x, y)
    
    # Time
    n_runs = 1000
    start = time.perf_counter()
    for _ in range(n_runs):
        z = bispev_numba(tx, ty, c, kx, ky, x, y)
    end = time.perf_counter()
    
    avg_time = (end - start) / n_runs * 1000  # ms
    print(f"\nNumba cfunc average time: {avg_time:.3f} ms")


if __name__ == "__main__":
    test_numba_integration()