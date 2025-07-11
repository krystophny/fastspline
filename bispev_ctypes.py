"""
Python ctypes interface for the DIERCKX bispev C wrapper.
"""
import ctypes
import numpy as np
from pathlib import Path
import os

# Load the shared library
lib_path = Path(__file__).parent / "libbispev.so"
if not lib_path.exists():
    # Try alternative locations
    for alt_path in ["./libbispev.so", "/usr/local/lib/libbispev.so"]:
        if os.path.exists(alt_path):
            lib_path = alt_path
            break
    else:
        raise RuntimeError(f"Could not find libbispev.so. Please run 'make' first.")

lib = ctypes.CDLL(str(lib_path))

# Define the C function signature
# int bispev_c(const double* tx, int nx, const double* ty, int ny,
#              const double* c, int kx, int ky,
#              const double* x, int mx, const double* y, int my,
#              double* z, double* wrk, int lwrk, int* iwrk, int kwrk,
#              int* ier);

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


def bispev(tx, ty, c, kx, ky, x, y):
    """
    Evaluate a bivariate B-spline.
    
    Parameters
    ----------
    tx : array_like
        Knots in x-direction
    ty : array_like
        Knots in y-direction
    c : array_like
        B-spline coefficients
    kx : int
        Degree of spline in x-direction
    ky : int
        Degree of spline in y-direction
    x : array_like
        X-coordinates at which to evaluate
    y : array_like
        Y-coordinates at which to evaluate
        
    Returns
    -------
    z : ndarray
        Values of the spline at (x[i], y[j]) for all i,j
        Shape is (len(x), len(y))
    """
    # Convert inputs to numpy arrays
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    
    # Get dimensions
    nx = len(tx)
    ny = len(ty)
    mx = len(x)
    my = len(y)
    
    # Validate input
    expected_c_len = (nx - kx - 1) * (ny - ky - 1)
    if len(c) != expected_c_len:
        raise ValueError(f"c has wrong length: expected {expected_c_len}, got {len(c)}")
    
    # Allocate output and workspace
    z = np.zeros(mx * my, dtype=np.float64)
    lwrk = mx * (kx + 1) + my * (ky + 1)
    wrk = np.zeros(lwrk, dtype=np.float64)
    kwrk = mx + my
    iwrk = np.zeros(kwrk, dtype=np.int32)
    
    # Error flag
    ier = ctypes.c_int(0)
    
    # Call the C function
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
        ctypes.byref(ier)
    )
    
    if ier.value != 0:
        raise RuntimeError(f"bispev returned error code {ier.value}")
    
    # Reshape output to 2D array
    # scipy's bisplev returns shape (my, mx) - row corresponds to y, column to x
    return z.reshape((my, mx), order='F').T


if __name__ == "__main__":
    # Simple test
    print("Testing bispev ctypes wrapper...")
    
    # Create a simple test case
    kx = ky = 3
    tx = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    c = np.ones((len(tx)-kx-1) * (len(ty)-ky-1))
    
    x = np.linspace(0.1, 0.9, 5)
    y = np.linspace(0.1, 0.9, 5)
    
    z = bispev(tx, ty, c, kx, ky, x, y)
    print(f"Output shape: {z.shape}")
    print(f"Output values:\n{z}")