"""
Numba cfunc implementation of DIERCKX bispev routine with derivative support.

bispev evaluates a bivariate spline and its derivatives on a rectangular grid.
"""
import numpy as np
from numba import cfunc, types


# Define the cfunc signature for bispev with derivative support
bispev_sig = types.void(
    types.CPointer(types.float64),  # tx - x knots
    types.int32,                     # nx - number of x knots
    types.CPointer(types.float64),  # ty - y knots
    types.int32,                     # ny - number of y knots
    types.CPointer(types.float64),  # c - coefficients
    types.int32,                     # kx - x degree
    types.int32,                     # ky - y degree
    types.int32,                     # nux - x derivative order
    types.int32,                     # nuy - y derivative order
    types.CPointer(types.float64),  # x - x evaluation points
    types.int32,                     # mx - number of x points
    types.CPointer(types.float64),  # y - y evaluation points
    types.int32,                     # my - number of y points
    types.CPointer(types.float64),  # z - output values
    types.CPointer(types.float64),  # wrk - workspace
    types.int32,                     # lwrk - workspace size
    types.CPointer(types.int32),    # iwrk - integer workspace
    types.int32,                     # kwrk - integer workspace size
    types.CPointer(types.int32),    # ier - error flag
)


@cfunc(bispev_sig, nopython=True, fastmath=True)
def bispev_cfunc(tx, nx, ty, ny, c, kx, ky, nux, nuy, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    Evaluate bivariate spline and its derivatives on a rectangular grid.
    
    If nux=0 and nuy=0, this is equivalent to standard bispev.
    Otherwise, it computes partial derivatives using parder algorithm.
    """
    # Input validation
    ier[0] = 10
    
    # Calculate workspace requirements
    if nux == 0 and nuy == 0:
        lwest = (kx + 1) * mx + (ky + 1) * my
    else:
        lwest = (kx + 1 - nux) * mx + (ky + 1 - nuy) * my
    
    if lwrk < lwest:
        return
    if kwrk < (mx + my):
        return
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
        for i in range(1, my):
            if y[i] < y[i-1]:
                return
    
    # Additional validation for derivatives
    if nux < 0 or nux >= kx:
        return
    if nuy < 0 or nuy >= ky:
        return
    
    # All checks passed
    ier[0] = 0
    
    # Temporary arrays for B-spline evaluation
    h = np.zeros(20, dtype=np.float64)
    hh = np.zeros(19, dtype=np.float64)
    
    if nux == 0 and nuy == 0:
        # Standard bispev evaluation
        # Set up work array pointers
        iw = mx * (kx + 1)
        
        # Local variables for fpbisp
        kx1 = kx + 1
        nkx1 = nx - kx1
        tb_x = tx[kx]
        te_x = tx[nkx1]
        
        # Evaluate B-splines in x-direction
        l = kx
        l1 = l + 1
        
        for i in range(mx):
            arg = x[i]
            if arg < tb_x:
                arg = tb_x
            if arg > te_x:
                arg = te_x
                
            # Find knot interval
            while arg >= tx[l1] and l < nkx1 - 1:
                l = l1
                l1 = l + 1
                
            # Inline fpbspl algorithm for x direction
            h[0] = 1.0
            
            for j in range(1, kx + 1):
                for ii in range(j):
                    hh[ii] = h[ii]
                
                h[0] = 0.0
                
                for ii in range(1, j + 1):
                    li = (l + 1) + ii
                    lj = li - j
                    
                    if tx[li-1] == tx[lj-1]:
                        h[ii] = 0.0
                    else:
                        f = hh[ii-1] / (tx[li-1] - tx[lj-1])
                        h[ii-1] = h[ii-1] + f * (tx[li-1] - arg)
                        h[ii] = f * (arg - tx[lj-1])
            
            # Store interval index (0-based) in iwrk
            iwrk[i] = l - kx
            
            # Copy B-spline values to wrk (wx part)
            for j in range(kx1):
                wrk[i * kx1 + j] = h[j]
        
        # Evaluate B-splines in y-direction
        ky1 = ky + 1
        nky1 = ny - ky1
        tb_y = ty[ky]
        te_y = ty[nky1]
        
        l = ky
        l1 = l + 1
        
        for i in range(my):
            arg = y[i]
            if arg < tb_y:
                arg = tb_y
            if arg > te_y:
                arg = te_y
                
            while arg >= ty[l1] and l < nky1 - 1:
                l = l1
                l1 = l + 1
                
            # Inline fpbspl algorithm for y direction
            h[0] = 1.0
            
            for j in range(1, ky + 1):
                for ii in range(j):
                    hh[ii] = h[ii]
                
                h[0] = 0.0
                
                for ii in range(1, j + 1):
                    li = (l + 1) + ii
                    lj = li - j
                    
                    if ty[li-1] == ty[lj-1]:
                        h[ii] = 0.0
                    else:
                        f = hh[ii-1] / (ty[li-1] - ty[lj-1])
                        h[ii-1] = h[ii-1] + f * (ty[li-1] - arg)
                        h[ii] = f * (arg - ty[lj-1])
            
            # Store interval index in iwrk
            iwrk[mx + i] = l - ky
            
            # Copy B-spline values to wrk (wy part)
            for j in range(ky1):
                wrk[iw + i * ky1 + j] = h[j]
        
        # Evaluate tensor product
        m = 0
        
        # Temporary array for x B-spline values
        hx = np.zeros(6, dtype=np.float64)
        
        for i in range(mx):
            l_base = iwrk[i] * nky1
            
            # Copy x B-spline values for this point
            for i1 in range(kx1):
                hx[i1] = wrk[i * kx1 + i1]
                
            for j in range(my):
                l1 = l_base + iwrk[mx + j]
                sp = 0.0
                
                # Tensor product sum
                for i1 in range(kx1):
                    l2 = l1
                    for j1 in range(ky1):
                        sp = sp + c[l2] * hx[i1] * wrk[iw + j * ky1 + j1]
                        l2 = l2 + 1
                    l1 = l1 + nky1
                    
                z[m] = sp
                m = m + 1
    
    else:
        # Derivative evaluation using parder algorithm
        # This is a simplified version - full implementation would inline parder algorithm
        kx1 = kx + 1
        ky1 = ky + 1
        nkx1 = nx - kx1
        nky1 = ny - ky1
        
        # Main computation
        m = 0
        
        for i in range(mx):
            # X-direction B-spline evaluation with derivatives
            ak = x[i]
            
            if nux == 0:
                # No derivative - clamp to domain
                if ak < tx[kx]:
                    ak = tx[kx]
                if ak > tx[nkx1]:
                    ak = tx[nkx1]
                
                # Find knot interval
                l = kx
                while l < nkx1 - 1 and ak >= tx[l + 1]:
                    l += 1
            else:
                # Derivative case - clamp to derivative domain
                nkx1_deriv = nx - nux
                if ak < tx[nux]:
                    ak = tx[nux]
                if ak > tx[nkx1_deriv]:
                    ak = tx[nkx1_deriv]
                
                # Find knot interval
                l = nux
                while l < nkx1_deriv - 1 and ak >= tx[l + 1]:
                    l += 1
            
            # Compute B-spline basis with derivatives
            h[0] = 1.0
            
            # Cox-de Boor recurrence
            for j in range(1, kx + 1):
                for ii in range(j):
                    hh[ii] = h[ii]
                
                h[0] = 0.0
                
                for ii in range(1, j + 1):
                    li = l + ii
                    lj = li - j
                    
                    if tx[li] == tx[lj]:
                        h[ii] = 0.0
                    else:
                        f = hh[ii-1] / (tx[li] - tx[lj])
                        h[ii-1] = h[ii-1] + f * (tx[li] - ak)
                        h[ii] = f * (ak - tx[lj])
            
            # Apply derivative formula nux times
            for deriv_order in range(nux):
                current_k = kx - deriv_order
                
                # Save current values
                for ii in range(current_k + 1):
                    hh[ii] = h[ii]
                
                # Apply derivative recurrence
                for ii in range(current_k):
                    li = l + ii + 1
                    lj = li - current_k
                    
                    if tx[li] != tx[lj]:
                        factor = current_k / (tx[li] - tx[lj])
                        h[ii] = factor * (hh[ii+1] - hh[ii])
                    else:
                        h[ii] = 0.0
            
            # Store x B-spline values
            iwx = i * (kx1 - nux)
            for j in range(kx1 - nux):
                wrk[iwx + j] = h[j]
            
            iwrk[i] = l - (nux if nux > 0 else kx)
            
            # Process each y point for this x
            for j in range(my):
                # Y-direction B-spline evaluation with derivatives
                ak = y[j]
                
                if nuy == 0:
                    # No derivative - clamp to domain
                    if ak < ty[ky]:
                        ak = ty[ky]
                    if ak > ty[nky1]:
                        ak = ty[nky1]
                    
                    # Find knot interval
                    l = ky
                    while l < nky1 - 1 and ak >= ty[l + 1]:
                        l += 1
                else:
                    # Derivative case - clamp to derivative domain
                    nky1_deriv = ny - nuy
                    if ak < ty[nuy]:
                        ak = ty[nuy]
                    if ak > ty[nky1_deriv]:
                        ak = ty[nky1_deriv]
                    
                    # Find knot interval
                    l = nuy
                    while l < nky1_deriv - 1 and ak >= ty[l + 1]:
                        l += 1
                
                # Compute B-spline basis with derivatives
                h[0] = 1.0
                
                # Cox-de Boor recurrence
                for jj in range(1, ky + 1):
                    for ii in range(jj):
                        hh[ii] = h[ii]
                    
                    h[0] = 0.0
                    
                    for ii in range(1, jj + 1):
                        li = l + ii
                        lj = li - jj
                        
                        if ty[li] == ty[lj]:
                            h[ii] = 0.0
                        else:
                            f = hh[ii-1] / (ty[li] - ty[lj])
                            h[ii-1] = h[ii-1] + f * (ty[li] - ak)
                            h[ii] = f * (ak - ty[lj])
                
                # Apply derivative formula nuy times
                for deriv_order in range(nuy):
                    current_k = ky - deriv_order
                    
                    # Save current values
                    for ii in range(current_k + 1):
                        hh[ii] = h[ii]
                    
                    # Apply derivative recurrence
                    for ii in range(current_k):
                        li = l + ii + 1
                        lj = li - current_k
                        
                        if ty[li] != ty[lj]:
                            factor = current_k / (ty[li] - ty[lj])
                            h[ii] = factor * (hh[ii+1] - hh[ii])
                        else:
                            h[ii] = 0.0
                
                # Store y B-spline values
                iwy = mx * (kx1 - nux) + j * (ky1 - nuy)
                for jj in range(ky1 - nuy):
                    wrk[iwy + jj] = h[jj]
                
                iwrk[mx + j] = l - (nuy if nuy > 0 else ky)
                
                # Compute tensor product
                z[m] = 0.0
                l2 = iwrk[i] * nky1 + iwrk[mx + j]
                
                for lx in range(kx1 - nux):
                    l1 = l2
                    wx = wrk[i * (kx1 - nux) + lx]
                    for ly in range(ky1 - nuy):
                        wy = wrk[mx * (kx1 - nux) + j * (ky1 - nuy) + ly]
                        z[m] += c[l1] * wx * wy
                        l1 += 1
                    l2 += nky1
                
                m += 1


# Export the cfunc address
bispev_cfunc_address = bispev_cfunc.address