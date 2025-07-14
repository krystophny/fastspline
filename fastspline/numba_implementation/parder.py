"""
Fully inlined cfunc implementation of parder matching scipy exactly.
All operations are inlined into a single cfunc without any external function calls.
"""
import numpy as np
from numba import cfunc, types
import ctypes


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
def parder_inline_cfunc(tx, nx, ty, ny, c, kx, ky, nux, nuy, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    Fully inlined implementation of parder based on working version.
    """
    # Input validation
    ier[0] = 10
    kx1 = kx + 1
    ky1 = ky + 1
    nkx1 = nx - kx1
    nky1 = ny - ky1
    
    # Validate derivative orders
    if nux < 0 or nux >= kx:
        return
    if nuy < 0 or nuy >= ky:
        return
    
    # Validate workspace
    lwest = (kx1 - nux) * mx + (ky1 - nuy) * my
    if lwrk < lwest:
        return
    if kwrk < (mx + my):
        return
    
    # Validate array sizes
    if mx < 1:
        return
    if my < 1:
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
    
    # All checks passed
    ier[0] = 0
    
    # Main computation
    m = 0
    h = np.zeros(20, dtype=np.float64)
    hh = np.zeros(20, dtype=np.float64)
    
    for i in range(mx):
        # X direction processing
        if nux == 0:
            # No derivative case
            ak = x[i]
            if ak < tx[kx] or ak > tx[nkx1]:
                ier[0] = 10
                return
            
            # Find knot interval
            l = kx
            l1 = l + 1
            while not (ak < tx[l1] or l == nkx1):
                l = l1
                l1 = l + 1
            
            if l1 < nx and ak == tx[l1]:
                l = l1
            
            # Inline fpbspl for x direction
            h[0] = 1.0
            
            # Cox-de Boor recurrence
            for j in range(1, kx + 1):
                # Save current values
                for ii in range(j):
                    hh[ii] = h[ii]
                
                h[0] = 0.0
                
                for ii in range(1, j + 1):
                    li = l + ii
                    lj = li - j
                    
                    if li >= nx or lj < 0:
                        h[ii] = 0.0
                    elif tx[li] == tx[lj]:
                        h[ii] = 0.0
                    else:
                        f = hh[ii-1] / (tx[li] - tx[lj])
                        h[ii-1] = h[ii-1] + f * (tx[li] - ak)
                        h[ii] = f * (ak - tx[lj])
            
            # Store in workspace
            iwx = i * kx1
            for j in range(kx1):
                wrk[iwx + j] = h[j]
            
            iwrk[i] = l - kx
            
        else:
            # Derivative case
            ak = x[i]
            nkx1_local = nx - nux
            tb = tx[nux]
            te = tx[nkx1_local-1]
            
            if ak < tb:
                ak = tb
            if ak > te:
                ak = te
            
            # Find knot interval
            l = nux
            l1 = l + 1
            while not (ak < tx[l1] or l == nkx1_local):
                l = l1
                l1 = l + 1
            
            if ak == tx[l1]:
                l = l1
            
            # Inline fpbspl with derivative for x direction
            h[0] = 1.0
            
            # Cox-de Boor recurrence
            for j in range(1, kx + 1):
                # Save current values
                for ii in range(j):
                    hh[ii] = h[ii]
                
                h[0] = 0.0
                
                for ii in range(1, j + 1):
                    li = l + ii
                    lj = li - j
                    
                    if li >= nx or lj < 0:
                        h[ii] = 0.0
                    elif tx[li] == tx[lj]:
                        h[ii] = 0.0
                    else:
                        f = hh[ii-1] / (tx[li] - tx[lj])
                        h[ii-1] = h[ii-1] + f * (tx[li] - ak)
                        h[ii] = f * (ak - tx[lj])
            
            # Apply derivative formula nux times
            for deriv_iter in range(nux):
                current_k = kx - deriv_iter
                
                # Save current values
                for ii in range(current_k + 1):
                    hh[ii] = h[ii]
                
                # Apply derivative recurrence
                for ii in range(current_k):
                    li = l + ii + 1
                    lj = li - current_k
                    
                    if li >= nx or lj < 0:
                        h[ii] = 0.0
                    elif tx[li] == tx[lj]:
                        h[ii] = 0.0
                    else:
                        factor = float(current_k) / (tx[li] - tx[lj])
                        h[ii] = factor * (hh[ii+1] - hh[ii])
            
            # Store in workspace
            iwx = i * (kx1 - nux)
            for j in range(kx1 - nux):
                wrk[iwx + j] = h[j]
            
            iwrk[i] = l - nux
        
        # Y direction processing
        if nuy == 0:
            # No y derivative
            for j in range(my):
                ak = y[j]
                
                if ak < ty[ky] or ak > ty[nky1]:
                    ier[0] = 10
                    return
                
                # Find knot interval
                l = ky
                l1 = l + 1
                while not (ak < ty[l1] or l == nky1):
                    l = l1
                    l1 = l + 1
                
                if ak == ty[l1]:
                    l = l1
                
                # Inline fpbspl for y direction
                h[0] = 1.0
                
                # Cox-de Boor recurrence
                for jj in range(1, ky + 1):
                    # Save current values
                    for ii in range(jj):
                        hh[ii] = h[ii]
                    
                    h[0] = 0.0
                    
                    for ii in range(1, jj + 1):
                        li = l + ii
                        lj = li - jj
                        
                        if li >= ny or lj < 0:
                            h[ii] = 0.0
                        elif ty[li] == ty[lj]:
                            h[ii] = 0.0
                        else:
                            f = hh[ii-1] / (ty[li] - ty[lj])
                            h[ii-1] = h[ii-1] + f * (ty[li] - ak)
                            h[ii] = f * (ak - ty[lj])
                
                # Store in workspace
                iwy = mx * (kx1 - nux) + j * ky1
                for jj in range(ky1):
                    wrk[iwy + jj] = h[jj]
                
                iwrk[mx + j] = l - ky
                
                # Compute tensor product
                z[m] = 0.0
                l2 = iwrk[i] * nky1 + iwrk[mx + j]
                
                for lx in range(kx1 - nux):
                    l1 = l2
                    wx = wrk[i * (kx1 - nux) + lx]
                    for ly in range(ky1):
                        wy = wrk[mx * (kx1 - nux) + j * ky1 + ly]
                        z[m] = z[m] + c[l1] * wx * wy
                        l1 = l1 + 1
                    l2 = l2 + nky1
                
                m = m + 1
        else:
            # Y derivative case
            for j in range(my):
                ak = y[j]
                nky1_local = ny - nuy
                tb = ty[nuy]
                te = ty[nky1_local-1]
                
                if ak < tb:
                    ak = tb
                if ak > te:
                    ak = te
                
                # Find knot interval
                l = nuy
                l1 = l + 1
                while not (ak < ty[l1] or l == nky1_local):
                    l = l1
                    l1 = l + 1
                
                if ak == ty[l1]:
                    l = l1
                
                # Inline fpbspl with derivative for y direction
                h[0] = 1.0
                
                # Cox-de Boor recurrence
                for jj in range(1, ky + 1):
                    # Save current values
                    for ii in range(jj):
                        hh[ii] = h[ii]
                    
                    h[0] = 0.0
                    
                    for ii in range(1, jj + 1):
                        li = l + ii
                        lj = li - jj
                        
                        if li >= ny or lj < 0:
                            h[ii] = 0.0
                        elif ty[li] == ty[lj]:
                            h[ii] = 0.0
                        else:
                            f = hh[ii-1] / (ty[li] - ty[lj])
                            h[ii-1] = h[ii-1] + f * (ty[li] - ak)
                            h[ii] = f * (ak - ty[lj])
                
                # Apply derivative formula nuy times
                for deriv_iter in range(nuy):
                    current_k = ky - deriv_iter
                    
                    # Save current values
                    for ii in range(current_k + 1):
                        hh[ii] = h[ii]
                    
                    # Apply derivative recurrence
                    for ii in range(current_k):
                        li = l + ii + 1
                        lj = li - current_k
                        
                        if li >= ny or lj < 0:
                            h[ii] = 0.0
                        elif ty[li] == ty[lj]:
                            h[ii] = 0.0
                        else:
                            factor = float(current_k) / (ty[li] - ty[lj])
                            h[ii] = factor * (hh[ii+1] - hh[ii])
                
                # Store in workspace
                iwy = mx * (kx1 - nux) + j * (ky1 - nuy)
                for jj in range(ky1 - nuy):
                    wrk[iwy + jj] = h[jj]
                
                iwrk[mx + j] = l - nuy
                
                # Compute tensor product
                z[m] = 0.0
                l2 = iwrk[i] * nky1 + iwrk[mx + j]
                
                for lx in range(kx1 - nux):
                    l1 = l2
                    wx = wrk[i * (kx1 - nux) + lx]
                    for ly in range(ky1 - nuy):
                        wy = wrk[mx * (kx1 - nux) + j * (ky1 - nuy) + ly]
                        z[m] = z[m] + c[l1] * wx * wy
                        l1 = l1 + 1
                    l2 = l2 + nky1
                
                m = m + 1


# Export address
parder_cfunc_address = parder_inline_cfunc.address

# Also export with expected name for test compatibility
parder_correct_cfunc_address = parder_inline_cfunc.address


def test_parder():
    """Test the parder implementation"""
    print("=== TESTING INLINE PARDER IMPLEMENTATION ===")
    
    from scipy.interpolate import bisplrep, dfitpack
    import warnings
    
    # Create test data
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Test derivatives
    derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    
    for nux, nuy in derivatives:
        print(f"\nTesting derivative ({nux}, {nuy}):")
        
        # scipy reference
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            z_scipy, ier_scipy = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
        
        print(f"  scipy: {z_scipy[0,0]:.10f} (ier={ier_scipy})")
        
        # Test cfunc
        try:
            nx, ny = len(tx), len(ty)
            mx, my = len(xi), len(yi)
            
            c_arr = np.asarray(c, dtype=np.float64)
            z_cfunc = np.zeros(mx * my, dtype=np.float64)
            
            # Workspace
            lwrk = (3 + 1 - nux) * mx + (3 + 1 - nuy) * my
            wrk = np.zeros(lwrk, dtype=np.float64)
            iwrk = np.zeros(mx + my, dtype=np.int32)
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
                                           )(parder_inline_cfunc_address)
            
            parder_func(tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
                       ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
                       c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 3, 3, nux, nuy,
                       xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
                       yi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
                       z_cfunc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
                       iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), mx + my,
                       ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            
            print(f"  cfunc: {z_cfunc[0]:.10f} (ier={ier[0]})")
            
            diff = abs(z_scipy[0,0] - z_cfunc[0])
            print(f"  diff:  {diff:.2e}")
            
            if diff < 1e-14:
                print("  ✓ EXACT MATCH!")
            elif diff < 1e-10:
                print("  ✓ Very close")
            else:
                print("  ✗ MISMATCH!")
                
        except Exception as e:
            print(f"  ✗ cfunc error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_parder()