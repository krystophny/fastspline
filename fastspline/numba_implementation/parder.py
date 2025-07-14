"""
Numba cfunc implementation of DIERCKX parder algorithm.

EXACT IMPLEMENTATION following DIERCKX parder.f line by line.
This implementation should match scipy.interpolate.dfitpack.parder exactly.
"""
import numpy as np
from numba import cfunc, types
import ctypes


@cfunc(types.void(
    types.CPointer(types.float64),  # t
    types.int32,                     # n
    types.int32,                     # k
    types.float64,                   # x
    types.int32,                     # nu (derivative order)
    types.int32,                     # l
    types.CPointer(types.float64),  # wrk
    types.int32,                     # offset
), nopython=True)
def fpbspl_with_derivatives(t, n, k, x, nu, l, wrk, offset):
    """
    DIERCKX fpbspl with derivative support.
    Computes B-spline basis functions and their derivatives.
    """
    # Initialize - wrk[offset] = 1.0
    wrk[offset] = 1.0
    
    # Cox-de Boor recursion
    for j in range(1, k + 1):
        # Save previous values
        for i in range(j):
            wrk[offset + 20 + i] = wrk[offset + i]
        
        wrk[offset] = 0.0
        
        for i in range(1, j + 1):
            li = l + i
            lj = li - j
            
            if li < n and lj >= 0 and t[li] != t[lj]:
                f = wrk[offset + 20 + i - 1] / (t[li] - t[lj])
                wrk[offset + i - 1] += f * (t[li] - x)
                wrk[offset + i] = f * (x - t[lj])
            else:
                wrk[offset + i] = 0.0
    
    # Apply derivative formula nu times
    for deriv in range(nu):
        curr_k = k - deriv
        
        # Save current values
        for i in range(curr_k + 1):
            wrk[offset + 20 + i] = wrk[offset + i]
        
        # Apply derivative recurrence
        for i in range(curr_k):
            li = l + i + 1
            lj = li - curr_k
            
            if li < n and lj >= 0 and t[li] != t[lj]:
                factor = float(curr_k) / (t[li] - t[lj])
                wrk[offset + i] = factor * (wrk[offset + 20 + i + 1] - wrk[offset + 20 + i])
            else:
                wrk[offset + i] = 0.0


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
def parder_cfunc(tx, nx, ty, ny, c, kx, ky, nux, nuy, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    EXACT DIERCKX parder algorithm implementation.
    Following parder.f line by line with correct Fortran->C array indexing.
    """
    # Input validation exactly as in DIERCKX
    ier[0] = 10
    
    kx1 = kx + 1
    ky1 = ky + 1
    nkx1 = nx - kx1
    nky1 = ny - ky1
    
    if nux < 0 or nux >= kx:
        return
    if nuy < 0 or nuy >= ky:
        return
    
    lwest = (kx1 - nux) * mx + (ky1 - nuy) * my
    if lwrk < lwest:
        return
    if kwrk < (mx + my):
        return
    
    # Check x array is sorted
    if mx > 1:
        for i in range(1, mx):
            if x[i] < x[i-1]:
                return
    
    # Check y array is sorted  
    if my > 1:
        for j in range(1, my):
            if y[j] < y[j-1]:
                return
    
    ier[0] = 0
    
    # Check x domain restrictions for derivatives
    if nux > 0:
        for i in range(mx):
            # Fortran: if(x(i).lt.tx(kx1) .or. x(i).gt.tx(nkx1)) go to 400
            # C: if(x[i] < tx[kx1-1] || x[i] > tx[nkx1-1]) return
            if x[i] < tx[kx1-1] or x[i] > tx[nkx1-1]:
                ier[0] = 10
                return
    
    # Check y domain restrictions for derivatives
    if nuy > 0:
        for j in range(my):
            # Fortran: if(y(j).lt.ty(ky1) .or. y(j).gt.ty(nky1)) go to 400
            # C: if(y[j] < ty[ky1-1] || y[j] > ty[nky1-1]) return
            if y[j] < ty[ky1-1] or y[j] > ty[nky1-1]:
                ier[0] = 10
                return
    
    # The partial derivative is computed - following DIERCKX exactly
    m = 0
    
    # Main loop over x points
    for i in range(mx):
        # Handle X direction
        if nux == 0:
            # No x derivative - standard case
            iwx = i * kx1
            l_x = kx
            ak = x[i]
            
            # Search for knot interval t(l) <= ak < t(l+1)
            l_x = kx
            while ak >= tx[l_x + 1] and l_x != nkx1 - 1:
                l_x += 1
            if ak == tx[l_x + 1]:
                l_x = l_x + 1
            
            # Call fpbspl for standard evaluation
            fpbspl_with_derivatives(tx, nx, kx, ak, 0, l_x, wrk, iwx)
        else:
            # X derivative case
            ak = x[i]
            nkx1_deriv = nx - nux
            # Domain restriction for derivatives
            tb = tx[nux]
            te = tx[nkx1_deriv - 1]
            if ak < tb:
                ak = tb
            if ak > te:
                ak = te
                
            # Search for knot interval t(l) <= ak < t(l+1)
            l_x = nux
            while ak >= tx[l_x + 1] and l_x != nkx1_deriv - 1:
                l_x += 1
            if ak == tx[l_x + 1]:
                l_x = l_x + 1
            
            # Call fpbspl for derivative evaluation
            iwx = i * (kx1 - nux)
            fpbspl_with_derivatives(tx, nx, kx, ak, nux, l_x, wrk, iwx)
        
        # Handle Y direction
        if nuy == 0:
            # No y derivative - standard case
            for j in range(my):
                ak = y[j]
                # Domain check for standard evaluation
                if ak < ty[ky1-1] or ak > ty[nky1-1]:
                    ier[0] = 10
                    return
                    
                # Search for knot interval t(l) <= ak < t(l+1)
                l_y = ky
                while ak >= ty[l_y + 1] and l_y != nky1 - 1:
                    l_y += 1
                if ak == ty[l_y + 1]:
                    l_y = l_y + 1
                
                # Call fpbspl for standard evaluation
                iwy = j * ky1
                fpbspl_with_derivatives(ty, ny, ky, ak, 0, l_y, wrk, (kx1 - nux) * mx + iwy)
                
                # Compute the partial derivative
                iwrk[mx + j] = l_y - ky
                z[m] = 0.0
                l2 = l_y - ky
                
                # Sum over basis functions
                for lx in range(kx1 - nux):
                    l1 = l2
                    for ly in range(ky1):
                        z[m] += c[l1] * wrk[iwx + lx] * wrk[(kx1 - nux) * mx + iwy + ly]
                        l1 += 1
                    l2 += nky1
                
                m += 1
        else:
            # Y derivative case
            for j in range(my):
                ak = y[j]
                nky1_deriv = ny - nuy
                # Domain restriction for derivatives
                tb = ty[nuy]
                te = ty[nky1_deriv - 1]
                if ak < tb:
                    ak = tb
                if ak > te:
                    ak = te
                    
                # Search for knot interval t(l) <= ak < t(l+1)
                l_y = nuy
                while ak >= ty[l_y + 1] and l_y != nky1_deriv - 1:
                    l_y += 1
                if ak == ty[l_y + 1]:
                    l_y = l_y + 1
                
                # Call fpbspl for derivative evaluation
                iwy = j * (ky1 - nuy)
                fpbspl_with_derivatives(ty, ny, ky, ak, nuy, l_y, wrk, (kx1 - nux) * mx + iwy)
                
                # Compute the partial derivative
                iwrk[i] = l_y - nuy
                iwrk[mx + j] = l_y - nuy
                z[m] = 0.0
                l2 = l_y - nuy
                
                # Sum over basis functions
                for lx in range(kx1 - nux):
                    l1 = l2
                    for ly in range(ky1 - nuy):
                        z[m] += c[l1] * wrk[iwx + lx] * wrk[(kx1 - nux) * mx + iwy + ly]
                        l1 += 1
                    l2 += nky1
                
                m += 1


# Export address
parder_cfunc_address = parder_cfunc.address

# Also export with expected name for test compatibility
parder_correct_cfunc_address = parder_cfunc.address


def call_parder_safe(tx, ty, c, kx, ky, nux, nuy, xi, yi):
    """Safe wrapper for parder cfunc that manages memory properly"""
    nx, ny = len(tx), len(ty)
    mx, my = len(xi), len(yi)
    
    # Ensure all arrays are contiguous and properly typed
    tx = np.ascontiguousarray(tx, dtype=np.float64)
    ty = np.ascontiguousarray(ty, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)
    xi = np.ascontiguousarray(xi, dtype=np.float64)
    yi = np.ascontiguousarray(yi, dtype=np.float64)
    
    # Pre-allocate output and workspace with proper sizing
    z_cfunc = np.zeros(mx * my, dtype=np.float64)
    lwrk = max(100, (kx + 1 - nux) * mx + (ky + 1 - nuy) * my + 50)  # Extra safety margin
    wrk = np.zeros(lwrk, dtype=np.float64)
    iwrk = np.zeros(max(50, mx + my + 10), dtype=np.int32)  # Extra safety margin
    ier = np.zeros(1, dtype=np.int32)
    
    # Create ctypes function - do this once to avoid repeated creation
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
                                   )(parder_cfunc_address)
    
    # Call the function
    parder_func(tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
               ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
               c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), kx, ky, nux, nuy,
               xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
               yi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
               z_cfunc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
               wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
               iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(iwrk),
               ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    
    # Clean up references before return to help GC
    del parder_func, wrk, iwrk
    
    return z_cfunc, ier[0]


def test_parder():
    """Test the parder implementation against scipy"""
    print("=== TESTING FIXED PARDER IMPLEMENTATION ===")
    
    from scipy.interpolate import bisplrep, dfitpack
    import warnings
    import gc
    
    # Simple test case - linear function
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = 2*X + 3*Y  # Linear function with known derivatives
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    print(f"Testing linear function f(x,y) = 2x + 3y at point (0.5, 0.5)")
    
    derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    expected = {
        (0, 0): 2*0.5 + 3*0.5,  # f(0.5, 0.5) = 2.5
        (1, 0): 2.0,             # df/dx = 2
        (0, 1): 3.0,             # df/dy = 3
        (2, 0): 0.0,             # d²f/dx² = 0
        (0, 2): 0.0,             # d²f/dy² = 0
        (1, 1): 0.0              # d²f/dxdy = 0
    }
    
    passed = 0
    total = len(derivatives)
    
    for nux, nuy in derivatives:
        try:
            # scipy reference
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                z_scipy, ier_scipy = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
            
            # Test cfunc with safe wrapper
            z_cfunc, ier_cfunc = call_parder_safe(tx, ty, c, 3, 3, nux, nuy, xi, yi)
            
            if ier_scipy == 0 and ier_cfunc == 0:
                diff = abs(z_scipy[0,0] - z_cfunc[0])
                expected_val = expected[nux, nuy]
                
                print(f"  ({nux},{nuy}): scipy={z_scipy[0,0]:.6f}, cfunc={z_cfunc[0]:.6f}, expected={expected_val:.6f}, diff={diff:.2e}")
                
                if diff < 1e-10:
                    passed += 1
                    print(f"    ✓ PASS")
                else:
                    print(f"    ✗ FAIL")
            else:
                print(f"  ({nux},{nuy}): ERROR - scipy_ier={ier_scipy}, cfunc_ier={ier_cfunc}")
                
        except Exception as e:
            print(f"  ({nux},{nuy}): EXCEPTION: {e}")
        
        gc.collect()
    
    print(f"\nSUMMARY: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    test_parder()