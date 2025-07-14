"""
Numba cfunc implementation of DIERCKX parder routine.

parder evaluates partial derivatives of a bivariate spline on a rectangular grid.
Direct port from parder.f with 0-based indexing.
"""
import numpy as np
from numba import cfunc, types


# Define the cfunc signature for parder
parder_sig = types.void(
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
    types.CPointer(types.float64),  # z - output derivative values
    types.CPointer(types.float64),  # wrk - workspace
    types.int32,                     # lwrk - workspace size
    types.CPointer(types.int32),    # iwrk - integer workspace
    types.int32,                     # kwrk - integer workspace size
    types.CPointer(types.int32),    # ier - error flag
)


@cfunc(parder_sig, nopython=True, fastmath=True)
def parder_cfunc(tx, nx, ty, ny, c, kx, ky, nux, nuy, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    Evaluate partial derivatives of bivariate spline on a rectangular grid.
    
    Direct translation from Fortran parder.f with 0-based indexing.
    Inlines fpbspl with derivative computation.
    """
    # Input validation (direct port from Fortran)
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
    
    # Check workspace size
    lwest = (kx1 - nux) * mx + (ky1 - nuy) * my
    if lwrk < lwest:
        return
    if kwrk < (mx + my):
        return
    
    # Check array sizes
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
    
    # Temporary arrays for B-spline evaluation
    h = np.zeros(20, dtype=np.float64)
    hh = np.zeros(19, dtype=np.float64)
    
    # Main computation
    m = 0
    
    # Process each x point
    for i in range(mx):
        # X-direction B-spline evaluation
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
        
        # Inline fpbspl with derivative computation for x direction
        # First compute standard B-spline values
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
                
                if tx[li] == tx[lj]:
                    h[ii] = 0.0
                else:
                    f = hh[ii-1] / (tx[li] - tx[lj])
                    h[ii-1] = h[ii-1] + f * (tx[li] - ak)
                    h[ii] = f * (ak - tx[lj])
        
        # Apply derivative formula nux times
        for deriv_order in range(nux):
            # Apply derivative formula: d/dx B_{i,k}(x) = k * (B_{i,k-1}(x)/(t_{i+k}-t_i) - B_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
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
            # Y-direction B-spline evaluation
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
            
            # Inline fpbspl with derivative computation for y direction
            # First compute standard B-spline values
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
                    
                    if ty[li] == ty[lj]:
                        h[ii] = 0.0
                    else:
                        f = hh[ii-1] / (ty[li] - ty[lj])
                        h[ii-1] = h[ii-1] + f * (ty[li] - ak)
                        h[ii] = f * (ak - ty[lj])
            
            # Apply derivative formula nuy times
            for deriv_order in range(nuy):
                # Apply derivative formula
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
parder_cfunc_address = parder_cfunc.address