"""
Real cfunc implementation of DIERCKX parder algorithm.
Inlines all operations into a single cfunc without external function calls.
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
def parder_cfunc(tx, nx, ty, ny, c, kx, ky, nux, nuy, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    Real implementation of DIERCKX parder algorithm.
    """
    # Input validation
    ier[0] = 10
    
    if nux < 0 or nux >= kx or nuy < 0 or nuy >= ky:
        return
    if mx < 1 or my < 1:
        return
        
    # Check workspace
    lwest = (kx + 1 - nux) * mx + (ky + 1 - nuy) * my
    if lwrk < lwest:
        return
    if kwrk < (mx + my):
        return
    
    ier[0] = 0
    
    # Main computation following original DIERCKX parder algorithm structure
    kx1 = kx + 1
    ky1 = ky + 1
    nkx1 = nx - kx1
    nky1 = ny - ky1
    
    m = 0
    # Follow the EXACT structure of parder.f - compute x basis for all points first
    for i in range(mx):
        l = kx1
        l1 = l + 1
        if nux == 0:
            ak = x[i]
            tb = tx[kx1-1]  # tx(kx1) in Fortran
            te = tx[nkx1]   # tx(nkx1+1) in Fortran  
            if ak < tb:
                ak = tb
            if ak > te:
                ak = te
        else:
            ak = x[i]
            nkx1_temp = nx - nux
            tb = tx[nux]     # tx(nux+1) in Fortran
            te = tx[nkx1_temp-1]  # tx(nkx1) in Fortran
            if ak < tb:
                ak = tb
            if ak > te:
                ak = te
            
        # Search for knot interval following DIERCKX exactly
        if nux == 0:
            l = kx
            l1 = l + 1
            while ak >= tx[l1] and l < nkx1:
                l = l1
                l1 = l + 1
        else:
            l = nux
            l1 = l + 1
            while ak >= tx[l1] and l < (nx - nux):
                l = l1 
                l1 = l + 1
            if ak == tx[l1]:
                l = l1
        
        # Inline fpbspl for x 
        if nux == 0:
            iwx = i * kx1
        else:
            iwx = i * (kx1 - nux)
            
        wrk[iwx] = 1.0
        
        # B-spline computation
        for j in range(1, kx + 1):
            for ii in range(j):
                wrk[lwrk - 20 + ii] = wrk[iwx + ii]
            
            wrk[iwx] = 0.0
            
            for ii in range(1, j + 1):
                li = l + ii
                lj = li - j
                
                if li < nx and lj >= 0 and tx[li] != tx[lj]:
                    f = wrk[lwrk - 20 + ii - 1] / (tx[li] - tx[lj])
                    wrk[iwx + ii - 1] = wrk[iwx + ii - 1] + f * (tx[li] - ak)
                    wrk[iwx + ii] = f * (ak - tx[lj])
                else:
                    wrk[iwx + ii] = 0.0
        
        # Apply x derivatives
        if nux > 0:
            for deriv in range(nux):
                current_k = kx - deriv
                
                # Save current values
                for ii in range(current_k + 1):
                    wrk[lwrk - 20 + ii] = wrk[iwx + ii]
                
                # Apply derivative recurrence
                for ii in range(current_k):
                    li = l + ii + 1
                    lj = li - current_k
                    
                    if li < nx and lj >= 0 and tx[li] != tx[lj]:
                        factor = float(current_k) / (tx[li] - tx[lj])
                        wrk[iwx + ii] = factor * (wrk[lwrk - 20 + ii + 1] - wrk[lwrk - 20 + ii])
                    else:
                        wrk[iwx + ii] = 0.0
        
        if nux == 0:
            iwrk[i] = l - kx
        else:
            iwrk[i] = l - nux
    
    # Now the main computation loops
    for i in range(mx):
        if nuy == 0:
            # Case 1: nuy=0 (lines 130-200 in parder.f)
            for j in range(my):
                l = ky1
                l1 = l + 1
                ak = y[j]
                if ak < ty[ky1-1] or ak > ty[nky1]:
                    ier[0] = 10
                    return
                    
                # Search for knot interval
                l = ky
                l1 = l + 1
                while ak >= ty[l1] and l < nky1:
                    l = l1
                    l1 = l + 1
                if ak == ty[l1]:
                    l = l1
                    
                # Y basis computation
                iwy = mx * (kx1 - nux) + j * ky1
                wrk[iwy] = 1.0
                
                for jj in range(1, ky + 1):
                    for ii in range(jj):
                        wrk[lwrk - 20 + ii] = wrk[iwy + ii]
                    
                    wrk[iwy] = 0.0
                    
                    for ii in range(1, jj + 1):
                        li = l + ii
                        lj = li - jj
                        
                        if li < ny and lj >= 0 and ty[li] != ty[lj]:
                            f = wrk[lwrk - 20 + ii - 1] / (ty[li] - ty[lj])
                            wrk[iwy + ii - 1] = wrk[iwy + ii - 1] + f * (ty[li] - ak)
                            wrk[iwy + ii] = f * (ak - ty[lj])
                        else:
                            wrk[iwy + ii] = 0.0
                
                iwrk[mx + j] = l - ky
                m += 1
                z[m-1] = 0.0
                l2 = iwrk[i] * nky1 + iwrk[mx + j]
                
                for lx in range(kx1 - nux):
                    l1 = l2
                    for ly in range(ky1):
                        l1 += 1
                        z[m-1] += c[l1 - 1] * wrk[i * (kx1 - nux) + lx] * wrk[iwy + ly]
                    l2 += nky1
        else:
            # Case 2: nuy>0 (lines 100-120 in parder.f)
            for j in range(my):
                l = ky1
                l1 = l + 1
                ak = y[j]
                nky1_temp = ny - nuy
                tb = ty[nuy]     # ty(nuy+1) in Fortran
                te = ty[nky1_temp-1]  # ty(nky1) in Fortran
                if ak < tb:
                    ak = tb
                if ak > te:
                    ak = te
                    
                # Search for knot interval
                l = nuy
                l1 = l + 1
                while ak >= ty[l1] and l < nky1_temp:
                    l = l1
                    l1 = l + 1
                if ak == ty[l1]:
                    l = l1
                    
                # Y basis computation with derivatives
                iwy = mx * (kx1 - nux) + j * (ky1 - nuy)
                wrk[iwy] = 1.0
                
                for jj in range(1, ky + 1):
                    for ii in range(jj):
                        wrk[lwrk - 20 + ii] = wrk[iwy + ii]
                    
                    wrk[iwy] = 0.0
                    
                    for ii in range(1, jj + 1):
                        li = l + ii
                        lj = li - jj
                        
                        if li < ny and lj >= 0 and ty[li] != ty[lj]:
                            f = wrk[lwrk - 20 + ii - 1] / (ty[li] - ty[lj])
                            wrk[iwy + ii - 1] = wrk[iwy + ii - 1] + f * (ty[li] - ak)
                            wrk[iwy + ii] = f * (ak - ty[lj])
                        else:
                            wrk[iwy + ii] = 0.0
                
                # Apply y derivatives
                for deriv in range(nuy):
                    current_k = ky - deriv
                    
                    for ii in range(current_k + 1):
                        wrk[lwrk - 20 + ii] = wrk[iwy + ii]
                    
                    for ii in range(current_k):
                        li = l + ii + 1
                        lj = li - current_k
                        
                        if li < ny and lj >= 0 and ty[li] != ty[lj]:
                            factor = float(current_k) / (ty[li] - ty[lj])
                            wrk[iwy + ii] = factor * (wrk[lwrk - 20 + ii + 1] - wrk[lwrk - 20 + ii])
                        else:
                            wrk[iwy + ii] = 0.0

                iwrk[mx + j] = l - nuy
                m += 1
                z[m-1] = 0.0
                l2 = l - nuy  # Direct from Fortran line 161: l2 = l-nuy
                
                for lx in range(kx1 - nux):
                    l1 = l2
                    for ly in range(ky1 - nuy):
                        l1 += 1
                        z[m-1] += c[l1 - 1] * wrk[i * (kx1 - nux) + lx] * wrk[iwy + ly]
                    l2 += (ny - ky1)  # From line 167: l2 = l2+nky1




# Export address
parder_cfunc_address = parder_cfunc.address

# Also export with expected name for test compatibility
parder_correct_cfunc_address = parder_cfunc.address


def test_parder():
    """Test the parder implementation against scipy"""
    print("=== TESTING REAL PARDER IMPLEMENTATION ===")
    
    from scipy.interpolate import bisplrep, dfitpack
    import warnings
    
    # Test with multiple different functions and points
    test_cases = [
        # Test case 1: Quadratic function
        {
            'name': 'Quadratic f(x,y) = x² + y²',
            'func': lambda X, Y: X**2 + Y**2,
            'points': [(0.5, 0.5), (0.2, 0.8), (0.7, 0.3)]
        },
        # Test case 2: Linear function  
        {
            'name': 'Linear f(x,y) = 2x + 3y',
            'func': lambda X, Y: 2*X + 3*Y,
            'points': [(0.4, 0.6), (0.1, 0.9)]
        },
        # Test case 3: Product function
        {
            'name': 'Product f(x,y) = xy',
            'func': lambda X, Y: X * Y,
            'points': [(0.3, 0.7), (0.6, 0.4)]
        }
    ]
    
    derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    total_tests = 0
    passed_tests = 0
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        # Create test data
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = test_case['func'](X, Y)
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        for xi_val, yi_val in test_case['points']:
            xi = np.array([xi_val])
            yi = np.array([yi_val])
            
            print(f"\nPoint ({xi_val}, {yi_val}):")
            
            for nux, nuy in derivatives:
                total_tests += 1
                
                # scipy reference
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    z_scipy, ier_scipy = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                
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
                                                   )(parder_cfunc_address)
                    
                    parder_func(tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), nx,
                               ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ny,
                               c_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 3, 3, nux, nuy,
                               xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), mx,
                               yi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), my,
                               z_cfunc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               wrk.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), lwrk,
                               iwrk.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), mx + my,
                               ier.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
                    
                    diff = abs(z_scipy[0,0] - z_cfunc[0])
                    
                    if diff < 1e-10:
                        print(f"  ({nux},{nuy}): ✓ PASS (diff: {diff:.2e})")
                        passed_tests += 1
                    else:
                        print(f"  ({nux},{nuy}): ✗ FAIL (diff: {diff:.2e}) scipy: {z_scipy[0,0]:.10f} cfunc: {z_cfunc[0]:.10f}")
                        
                except Exception as e:
                    print(f"  ({nux},{nuy}): ✗ ERROR: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    test_parder()