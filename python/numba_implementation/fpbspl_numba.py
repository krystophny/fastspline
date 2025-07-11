"""
Numba cfunc implementation of DIERCKX fpbspl routine.

fpbspl evaluates the (k+1) non-zero b-splines of degree k at t(l) <= x < t(l+1)
using the stable recurrence relation of de Boor and Cox.
"""
import numpy as np
from numba import cfunc, types
from numba.core import cgutils


# Define the cfunc signature for fpbspl
# void fpbspl(double* t, int n, int k, double x, int l, double* h)
fpbspl_sig = types.void(
    types.CPointer(types.float64),  # t - knot vector
    types.int32,                     # n - number of knots
    types.int32,                     # k - degree
    types.float64,                   # x - evaluation point
    types.int32,                     # l - knot interval index
    types.CPointer(types.float64),  # h - output array for b-spline values
)


@cfunc(fpbspl_sig, nopython=True, fastmath=True)
def fpbspl_cfunc(t, n, k, x, l, h):
    """
    Evaluate the (k+1) non-zero B-splines at x.
    
    Direct translation from Fortran fpbspl.f with 0-based indexing.
    
    Parameters:
    - t: knot vector (0-indexed array of length n)
    - n: number of knots
    - k: degree of B-splines
    - x: evaluation point, t[l-1] <= x < t[l] (using Fortran's 1-based l)
    - l: knot interval index (1-based as in Fortran)
    - h: output array for B-spline values (length k+1)
    
    Note: In Fortran, arrays are 1-indexed and l is 1-based.
    We keep l as 1-based for compatibility but adjust array access.
    """
    # Constants
    one = 1.0
    
    # Local arrays - in Fortran these are h(20) and hh(19)
    # We only need k+1 elements, but allocate statically for cfunc
    hh = np.zeros(19, dtype=np.float64)
    
    # Initialize h[0] = 1.0 (Fortran: h(1) = one)
    h[0] = one
    
    # Main loop - Fortran: do 20 j=1,k
    for j in range(1, k + 1):
        # Copy current h values to hh
        # Fortran: do 10 i=1,j; hh(i) = h(i)
        for i in range(j):
            hh[i] = h[i]
        
        # Reset h[0]
        h[0] = 0.0
        
        # Inner loop - Fortran: do 20 i=1,j
        for i in range(1, j + 1):
            # Calculate indices (keeping l as 1-based)
            # Fortran: li = l+i, lj = li-j
            li = l + i
            lj = li - j
            
            # Check for knot multiplicity
            # Array access: t is 0-indexed, but li/lj are 1-based
            if t[li-1] == t[lj-1]:
                h[i] = 0.0
            else:
                # Fortran: f = hh(i)/(t(li)-t(lj))
                f = hh[i-1] / (t[li-1] - t[lj-1])
                # Fortran: h(i) = h(i)+f*(t(li)-x)
                h[i-1] = h[i-1] + f * (t[li-1] - x)
                # Fortran: h(i+1) = f*(x-t(lj))
                h[i] = f * (x - t[lj-1])


# Export the cfunc address for use in other cfuncs
fpbspl_cfunc_address = fpbspl_cfunc.address