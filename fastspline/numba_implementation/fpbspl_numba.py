"""
Numba cfunc implementation of DIERCKX fpbspl routine.

fpbspl evaluates the (k+1) non-zero b-splines of degree k at t(l) <= x < t(l+1)
using the stable recurrence relation of de Boor and Cox.
"""
import numpy as np
from numba import cfunc, types
from numba.core import cgutils


# Define the cfunc signature for fpbspl with derivative parameter
# Allow both pointer and array inputs for ease of use
fpbspl_sig_ptr = types.void(
    types.CPointer(types.float64),  # t - knot vector
    types.int32,                     # n - number of knots
    types.int32,                     # k - degree
    types.float64,                   # x - evaluation point
    types.int32,                     # nux - derivative order
    types.int32,                     # l - knot interval index
    types.CPointer(types.float64),  # h - output array for b-spline values
)

fpbspl_sig_arr = types.void(
    types.CPointer(types.float64),        # t - knot vector
    types.int32,                           # n - number of knots
    types.int32,                           # k - degree
    types.float64,                         # x - evaluation point
    types.int32,                           # nux - derivative order
    types.int32,                           # l - knot interval index
    types.Array(types.float64, 1, 'C'),  # h - output array
)


@cfunc(fpbspl_sig_arr, nopython=True, fastmath=True)
def fpbspl_cfunc(t, n, k, x, nux, l, h):
    """
    Evaluate the (k+1-nux) non-zero B-splines or derivatives at x.
    
    DIERCKX fpbspl with derivative parameter nux.
    
    Parameters:
    - t: knot vector (0-indexed array of length n)
    - n: number of knots
    - k: degree of B-splines
    - x: evaluation point, t[l-1] <= x < t[l] (using Fortran's 1-based l)
    - nux: derivative order (0 for function values, >0 for derivatives)
    - l: knot interval index (1-based as in Fortran)
    - h: output array for B-spline values (length k+1-nux)
    
    For nux=0: standard B-spline evaluation (k+1 values)
    For nux>0: derivative evaluation (k+1-nux values)
    """
    # Constants
    one = 1.0
    
    # Local arrays - in Fortran these are h(20) and hh(19)
    # We only need k+1 elements, but allocate statically for cfunc
    hh = np.zeros(19, dtype=np.float64)
    
    # For derivatives: effective degree is k-nux
    k_eff = k - nux
    
    # Initialize h[0] = 1.0 (Fortran: h(1) = one)
    h[0] = one
    
    # Main loop - Fortran: do 20 j=1,k_eff
    for j in range(1, k_eff + 1):
        # Copy current h values to hh
        # Fortran: do 10 i=1,j; hh(i) = h(i)
        for i in range(j):
            hh[i] = h[i]
        
        # Reset h[0]
        h[0] = 0.0
        
        # Inner loop - Fortran: do 20 i=1,j
        for i in range(1, j + 1):
            # Calculate indices (keeping l as 1-based)
            # For derivatives, the knot interval calculation is different
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
    
    # For derivatives nux > 0, apply derivative scaling factor
    if nux > 0:
        fac = 1.0
        for i in range(nux):
            fac *= (k - i)
        for i in range(k_eff + 1):
            h[i] *= fac


# Export the cfunc address for use in other cfuncs
fpbspl_cfunc_address = fpbspl_cfunc.address