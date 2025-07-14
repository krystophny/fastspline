"""
Numba cfunc implementation of DIERCKX fpbspl routine with derivative support.

This is the extended fpbspl that handles derivatives as used in parder.f
"""
import numpy as np
from numba import cfunc, types


# Define the cfunc signature for fpbspl with derivative support
# void fpbspl_derivative(double* t, int n, int k, double x, int nu, int l, double* h)
fpbspl_derivative_sig = types.void(
    types.CPointer(types.float64),  # t - knot vector
    types.int32,                     # n - number of knots
    types.int32,                     # k - degree
    types.float64,                   # x - evaluation point
    types.int32,                     # nu - derivative order
    types.int32,                     # l - knot interval index
    types.CPointer(types.float64),  # h - output array for b-spline values
)


@cfunc(fpbspl_derivative_sig, nopython=True, fastmath=True)
def fpbspl_derivative_cfunc(t, n, k, x, nu, l, h):
    """
    Evaluate the B-spline basis functions and their derivatives.
    
    Direct implementation of the Cox-de Boor recurrence relation
    with derivative computation.
    
    Parameters:
    - t: knot vector (0-indexed array of length n)
    - n: number of knots
    - k: degree of B-splines
    - x: evaluation point
    - nu: derivative order (0 for function value, 1 for first derivative, etc.)
    - l: knot interval index (0-based)
    - h: output array for B-spline values (length k+1-nu)
    """
    # Local arrays for intermediate computation
    hh = np.zeros(20, dtype=np.float64)
    
    # First compute the standard B-spline values
    h[0] = 1.0
    
    # Cox-de Boor recurrence for function values
    for j in range(1, k + 1):
        # Save current values
        for i in range(j):
            hh[i] = h[i]
        
        h[0] = 0.0
        
        for i in range(1, j + 1):
            li = l + i
            lj = li - j
            
            if t[li] == t[lj]:
                h[i] = 0.0
            else:
                f = hh[i-1] / (t[li] - t[lj])
                h[i-1] = h[i-1] + f * (t[li] - x)
                h[i] = f * (x - t[lj])
    
    # Apply derivative formula nu times
    if nu > 0:
        for deriv_order in range(nu):
            # Current degree after deriv_order derivatives
            current_k = k - deriv_order
            
            # Apply derivative formula: d/dx B_{i,k}(x) = k * (B_{i,k-1}(x)/(t_{i+k}-t_i) - B_{i+1,k-1}(x)/(t_{i+k+1}-t_{i+1}))
            for i in range(current_k):
                li = l + i + 1
                lj = li - current_k
                
                if t[li] != t[lj]:
                    factor = current_k / (t[li] - t[lj])
                    if i > 0:
                        h[i] = factor * (h[i] - h[i-1])
                    else:
                        h[i] = -factor * h[i]
                else:
                    h[i] = 0.0
            
            # The last element should be handled differently
            if current_k > 0:
                li = l + current_k
                lj = l
                if t[li] != t[lj]:
                    factor = current_k / (t[li] - t[lj])
                    h[current_k-1] = factor * h[current_k-1]


# Export the cfunc address
fpbspl_derivative_cfunc_address = fpbspl_derivative_cfunc.address