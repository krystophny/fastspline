"""
Simple, correct cfunc implementation of parder.
Based on direct understanding of the algorithm.
"""
import numpy as np
from numba import cfunc, types, njit
import ctypes


@njit
def find_knot_interval(t, n, x, k):
    """Find knot interval for point x in knot vector t"""
    # Find interval such that t[i] <= x < t[i+1]
    left = k
    right = n - k - 1
    
    while left < right:
        mid = (left + right) // 2
        if x < t[mid]:
            right = mid
        else:
            left = mid + 1
    
    # Handle edge case where x equals last knot
    if left > 0 and x == t[left]:
        left -= 1
    
    return left


@njit
def compute_bspline_basis(t, n, k, x, nu, interval):
    """Compute B-spline basis functions and derivatives"""
    # Array to store basis functions
    basis = np.zeros(k + 1 - nu, dtype=np.float64)
    
    if nu == 0:
        # No derivative - compute standard B-spline basis
        # Use simplified de Boor algorithm
        basis[0] = 1.0
        
        for j in range(1, k + 1):
            saved = 0.0
            for r in range(j):
                alpha = (x - t[interval + r]) / (t[interval + r + j] - t[interval + r])
                temp = basis[r]
                basis[r] = saved + (1.0 - alpha) * temp
                saved = alpha * temp
            basis[j] = saved
    else:
        # Derivative case - use recursive formula
        # First compute basis functions of degree k-nu
        temp_basis = np.zeros(k + 1, dtype=np.float64)
        temp_basis[0] = 1.0
        
        for j in range(1, k + 1):
            saved = 0.0
            for r in range(j):
                if t[interval + r + j] != t[interval + r]:
                    alpha = (x - t[interval + r]) / (t[interval + r + j] - t[interval + r])
                    temp = temp_basis[r]
                    temp_basis[r] = saved + (1.0 - alpha) * temp
                    saved = alpha * temp
                else:
                    temp_basis[r] = saved
                    saved = 0.0
            temp_basis[j] = saved
        
        # Apply derivative operator nu times
        for deriv_order in range(nu):
            current_k = k - deriv_order
            for i in range(current_k):
                left_knot = interval + i
                right_knot = interval + i + current_k
                
                if t[right_knot] != t[left_knot]:
                    factor = current_k / (t[right_knot] - t[left_knot])
                    if i == 0:
                        temp_basis[i] = -factor * temp_basis[i]
                    else:
                        temp_basis[i] = factor * (temp_basis[i] - temp_basis[i-1])
                else:
                    temp_basis[i] = 0.0
        
        # Copy result
        for i in range(k + 1 - nu):
            basis[i] = temp_basis[i]
    
    return basis


@cfunc(types.void(
    types.CPointer(types.float64),  # tx
    types.int32,                     # nx
    types.CPointer(types.float64),  # ty
    types.int32,                     # ny
    types.CPointer(types.float64),  # c
    types.int32,                     # kx
    types.int32,                     # ky
    types.int32,                     # nux
    types.int32,                     # nuy
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
), nopython=True)
def parder_simple_cfunc(tx, nx, ty, ny, c, kx, ky, nux, nuy, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    Simple, correct implementation of bivariate spline derivatives.
    """
    # Basic validation
    ier[0] = 0
    
    if nux < 0 or nux >= kx or nuy < 0 or nuy >= ky:
        ier[0] = 10
        return
    
    if mx < 1 or my < 1:
        ier[0] = 10
        return
    
    # Main computation
    nkx1 = nx - kx - 1
    nky1 = ny - ky - 1
    
    m = 0
    for i in range(mx):
        for j in range(my):
            xi = x[i]
            yj = y[j]
            
            # Find knot intervals
            ix = find_knot_interval(tx, nx, xi, kx)
            iy = find_knot_interval(ty, ny, yj, ky)
            
            # Compute basis functions
            bx = compute_bspline_basis(tx, nx, kx, xi, nux, ix)
            by = compute_bspline_basis(ty, ny, ky, yj, nuy, iy)
            
            # Compute tensor product
            result = 0.0
            for p in range(kx + 1 - nux):
                for q in range(ky + 1 - nuy):
                    # Index into coefficient array
                    coeff_idx = (ix - kx + p) * nky1 + (iy - ky + q)
                    if 0 <= coeff_idx < nkx1 * nky1:
                        result += c[coeff_idx] * bx[p] * by[q]
            
            z[m] = result
            m += 1


# Export address
parder_simple_cfunc_address = parder_simple_cfunc.address


def test_simple_parder():
    """Test the simple parder implementation"""
    print("=== TESTING SIMPLE PARDER ===")
    
    # Create test data
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2
    
    from scipy.interpolate import bisplrep, dfitpack
    import warnings
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Test function value
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        z_scipy, ier_scipy = dfitpack.parder(tx, ty, c, 3, 3, 0, 0, xi, yi)
    
    print(f"Function value test:")
    print(f"  scipy: {z_scipy[0,0]:.6f} (ier={ier_scipy})")
    
    # Test cfunc
    try:
        nx, ny = len(tx), len(ty)
        mx, my = len(xi), len(yi)
        
        c_arr = np.asarray(c, dtype=np.float64)
        z_cfunc = np.zeros(mx * my, dtype=np.float64)
        
        # Dummy workspace
        wrk = np.zeros(100, dtype=np.float64)
        iwrk = np.zeros(10, dtype=np.int32)
        ier = np.zeros(1, dtype=np.int32)
        
        # Call cfunc
        parder_func = ctypes.CFUNCTYPE(None, 
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int32,
                                       ctypes.c_int32,
                                       ctypes.c_int32,
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_int32),
                                       ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_int32)
                                       )(parder_simple_cfunc_address)
        
        parder_func(tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
                   ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
                   c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 3, 3, 0, 0,
                   xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
                   yi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
                   z_cfunc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 100,
                   iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 10,
                   ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        
        print(f"  cfunc: {z_cfunc[0]:.6f} (ier={ier[0]})")
        print(f"  diff:  {abs(z_scipy[0,0] - z_cfunc[0]):.2e}")
        
    except Exception as e:
        print(f"  cfunc error: {e}")


if __name__ == "__main__":
    test_simple_parder()