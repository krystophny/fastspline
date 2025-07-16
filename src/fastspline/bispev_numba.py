"""
Numba cfunc implementation of DIERCKX bispev routine.

bispev evaluates a bivariate spline on a rectangular grid.
"""
import numpy as np
from numba import cfunc, types


# Define the cfunc signature for bispev (function evaluation only)
bispev_sig = types.void(
    types.CPointer(types.float64),  # tx - x knots
    types.int32,                     # nx - number of x knots
    types.CPointer(types.float64),  # ty - y knots
    types.int32,                     # ny - number of y knots
    types.CPointer(types.float64),  # c - coefficients
    types.int32,                     # kx - x degree
    types.int32,                     # ky - y degree
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
def bispev_cfunc(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier):
    """
    Evaluate bivariate spline on a rectangular grid.
    
    This is the standard bispev function evaluation only.
    For derivatives, use the separate parder function.
    """
    # Input validation
    ier[0] = 10
    
    # Calculate workspace requirements
    lwest = (kx + 1) * mx + (ky + 1) * my
    
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
    
    # All checks passed
    ier[0] = 0
    
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
        # Use workspace for h array starting at end of main workspace
        h_start = lwrk - 40  # Reserve 40 elements at end for temp arrays
        hh_start = h_start + 20
        
        wrk[h_start] = 1.0
        for ii in range(1, 20):
            wrk[h_start + ii] = 0.0
        
        for j in range(1, kx + 1):
            for ii in range(j):
                wrk[hh_start + ii] = wrk[h_start + ii]
            
            wrk[h_start] = 0.0
            
            for ii in range(1, j + 1):
                li = (l + 1) + ii
                lj = li - j
                
                if tx[li-1] == tx[lj-1]:
                    wrk[h_start + ii] = 0.0
                else:
                    f = wrk[hh_start + ii-1] / (tx[li-1] - tx[lj-1])
                    wrk[h_start + ii-1] = wrk[h_start + ii-1] + f * (tx[li-1] - arg)
                    wrk[h_start + ii] = f * (arg - tx[lj-1])
        
        # Store interval index (0-based) in iwrk
        iwrk[i] = l - kx
        
        # Copy B-spline values to wrk (wx part)
        for j in range(kx1):
            wrk[i * kx1 + j] = wrk[h_start + j]
        
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
        # Use workspace for h array (reusing same area as x direction)
        wrk[h_start] = 1.0
        for ii in range(1, 20):
            wrk[h_start + ii] = 0.0
        
        for j in range(1, ky + 1):
            for ii in range(j):
                wrk[hh_start + ii] = wrk[h_start + ii]
            
            wrk[h_start] = 0.0
            
            for ii in range(1, j + 1):
                li = (l + 1) + ii
                lj = li - j
                
                if ty[li-1] == ty[lj-1]:
                    wrk[h_start + ii] = 0.0
                else:
                    f = wrk[hh_start + ii-1] / (ty[li-1] - ty[lj-1])
                    wrk[h_start + ii-1] = wrk[h_start + ii-1] + f * (ty[li-1] - arg)
                    wrk[h_start + ii] = f * (arg - ty[lj-1])
        
        # Store interval index in iwrk
        iwrk[mx + i] = l - ky
        
        # Copy B-spline values to wrk (wy part)
        for j in range(ky1):
            wrk[iw + i * ky1 + j] = wrk[h_start + j]
    
    # Evaluate tensor product
    m = 0
    
    # Use workspace for x B-spline values (at end of workspace)
    hx_start = lwrk - 6
    
    for i in range(mx):
        l_base = iwrk[i] * nky1
        
        # Copy x B-spline values for this point
        for i1 in range(kx1):
            wrk[hx_start + i1] = wrk[i * kx1 + i1]
            
        for j in range(my):
            l1 = l_base + iwrk[mx + j]
            sp = 0.0
            
            # Tensor product sum
            for i1 in range(kx1):
                l2 = l1
                for j1 in range(ky1):
                    sp = sp + c[l2] * wrk[hx_start + i1] * wrk[iw + j * ky1 + j1]
                    l2 = l2 + 1
                l1 = l1 + nky1
                
            z[m] = sp
            m = m + 1


# Export the cfunc address
bispev_cfunc_address = bispev_cfunc.address