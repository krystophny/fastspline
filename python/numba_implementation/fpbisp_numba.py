"""
Numba cfunc implementation of DIERCKX fpbisp routine.

fpbisp evaluates a bivariate spline on a rectangular grid using tensor products.
"""
import numpy as np
from numba import cfunc, types


# Define the cfunc signature for fpbisp
# void fpbisp(double* tx, int nx, double* ty, int ny, double* c, 
#             int kx, int ky, double* x, int mx, double* y, int my,
#             double* z, double* wx, double* wy, int* lx, int* ly)
fpbisp_sig = types.void(
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
    types.CPointer(types.float64),  # wx - work array for x B-splines
    types.CPointer(types.float64),  # wy - work array for y B-splines
    types.CPointer(types.int32),    # lx - x interval indices
    types.CPointer(types.int32),    # ly - y interval indices
)


@cfunc(fpbisp_sig, nopython=True, fastmath=True)
def fpbisp_cfunc(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, wx, wy, lx, ly):
    """
    Evaluate bivariate spline on a grid.
    
    Direct translation from Fortran fpbisp.f with 0-based indexing.
    Includes fpbspl algorithm inline to avoid function pointer issues.
    """
    # Local variables
    kx1 = kx + 1
    nkx1 = nx - kx1
    tb = tx[kx]  # Fortran: tx(kx1) with kx1=kx+1
    te = tx[nkx1]  # Fortran: tx(nkx1+1)
    
    # Temporary arrays for B-spline evaluation
    h = np.zeros(20, dtype=np.float64)
    hh = np.zeros(19, dtype=np.float64)
    
    # Evaluate B-splines in x-direction
    # Fortran loop: do 40 i=1,mx
    l = kx  # Start at kx (0-based), Fortran starts at kx1 (1-based)
    l1 = l + 1
    
    for i in range(mx):
        arg = x[i]
        if arg < tb:
            arg = tb
        if arg > te:
            arg = te
            
        # Find knot interval - Fortran: 10 if(arg.lt.tx(l1) .or. l.eq.nkx1) go to 20
        while arg >= tx[l1] and l < nkx1 - 1:
            l = l1
            l1 = l + 1
            
        # Inline fpbspl algorithm for x direction
        # Initialize h[0] = 1.0
        h[0] = 1.0
        
        # Main loop for B-spline evaluation
        for j in range(1, kx + 1):
            # Copy current h values to hh
            for ii in range(j):
                hh[ii] = h[ii]
            
            # Reset h[0]
            h[0] = 0.0
            
            # Inner loop
            for ii in range(1, j + 1):
                # Calculate indices (l is 0-based here, but we add 1 for compatibility)
                li = (l + 1) + ii
                lj = li - j
                
                # Check for knot multiplicity
                if tx[li-1] == tx[lj-1]:
                    h[ii] = 0.0
                else:
                    f = hh[ii-1] / (tx[li-1] - tx[lj-1])
                    h[ii-1] = h[ii-1] + f * (tx[li-1] - arg)
                    h[ii] = f * (arg - tx[lj-1])
        
        # Store interval index (0-based)
        lx[i] = l - kx
        
        # Copy B-spline values to wx
        # wx is conceptually wx[mx][kx+1] but stored as 1D
        for j in range(kx1):
            wx[i * kx1 + j] = h[j]
    
    # Evaluate B-splines in y-direction
    ky1 = ky + 1
    nky1 = ny - ky1
    tb = ty[ky]
    te = ty[nky1]
    
    l = ky
    l1 = l + 1
    
    for i in range(my):
        arg = y[i]
        if arg < tb:
            arg = tb
        if arg > te:
            arg = te
            
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
        
        ly[i] = l - ky
        
        for j in range(ky1):
            wy[i * ky1 + j] = h[j]
    
    # Evaluate tensor product
    m = 0
    
    # Temporary array for x B-spline values
    hx = np.zeros(6, dtype=np.float64)
    
    for i in range(mx):
        l_base = lx[i] * nky1
        
        # Copy x B-spline values for this point
        for i1 in range(kx1):
            hx[i1] = wx[i * kx1 + i1]
            
        for j in range(my):
            l1 = l_base + ly[j]
            sp = 0.0
            
            # Tensor product sum
            for i1 in range(kx1):
                l2 = l1
                for j1 in range(ky1):
                    # c is 0-based, but l2 is already 0-based
                    sp = sp + c[l2] * hx[i1] * wy[j * ky1 + j1]
                    l2 = l2 + 1
                l1 = l1 + nky1
                
            z[m] = sp
            m = m + 1


# Export the cfunc address
fpbisp_cfunc_address = fpbisp_cfunc.address