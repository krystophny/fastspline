#!/usr/bin/env python3
"""
Fix the general quintic spline algorithm based on Fortran implementation
"""

import numpy as np

def quintic_spline_fortran_style(n, h, y):
    """
    Implement quintic spline following the Fortran algorithm exactly
    """
    # Constants
    rhop = 13.0 + np.sqrt(105.0)  # ~23.247
    rhom = 13.0 - np.sqrt(105.0)  # ~2.753
    
    # Initialize coefficient arrays
    a = y.copy()  # a coefficients are the y values
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    e = np.zeros(n)
    f = np.zeros(n)
    
    # Boundary condition matrices (3x3 systems)
    # First system for odd derivatives
    a11, a12, a13 = 1.0, 1.0/4.0, 1.0/16.0
    a21, a22, a23 = 3.0, 27.0/4.0, 9.0*27.0/16.0
    a31, a32, a33 = 5.0, 125.0/4.0, 5.0**5/16.0
    
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    
    # Compute boundary values for b, d, f at beginning
    b1 = a[3] - a[2]
    b2 = a[4] - a[1]
    b3 = a[5] - a[0]
    
    bbeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    dbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    fbeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    
    # Boundary values at end
    b1 = a[n-3] - a[n-4]
    b2 = a[n-2] - a[n-5]
    b3 = a[n-1] - a[n-6]
    
    bend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    dend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    fend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    
    # Second system for even derivatives
    a11, a12, a13 = 2.0, 1.0/2.0, 1.0/8.0
    a21, a22, a23 = 2.0, 9.0/2.0, 81.0/8.0
    a31, a32, a33 = 2.0, 25.0/2.0, 625.0/8.0
    
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    
    # Beginning values for a, c, e
    b1 = a[3] + a[2]
    b2 = a[4] + a[1]
    b3 = a[5] + a[0]
    
    abeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    cbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    ebeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    
    # End values
    b1 = a[n-3] + a[n-4]
    b2 = a[n-2] + a[n-5]
    b3 = a[n-1] + a[n-6]
    
    aend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    cend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    eend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    
    # Forward elimination arrays
    alp = np.zeros(n)
    bet = np.zeros(n)
    gam = np.zeros(n)
    
    # First forward elimination
    alp[0] = 0.0
    bet[0] = ebeg*(2.0 + rhom) - 5.0*fbeg*(3.0 + 1.5*rhom)
    
    for i in range(n-4):
        ip1 = i + 1
        alp[ip1] = -1.0 / (rhop + alp[i])
        bet[ip1] = alp[ip1] * (bet[i] - 5.0*(a[i+4] - 4.0*a[i+3] + 6.0*a[i+2] - 4.0*a[ip1] + a[i]))
    
    # Back substitution for gamma
    gam[n-3] = eend*(2.0 + rhom) + 5.0*fend*(3.0 + 1.5*rhom)
    for i in range(n-4, -1, -1):
        gam[i] = gam[i+1]*alp[i] + bet[i]
    
    # Second forward elimination
    alp[0] = 0.0
    bet[0] = ebeg - 2.5*5.0*fbeg
    
    for i in range(n-2):
        ip1 = i + 1
        alp[ip1] = -1.0 / (rhom + alp[i])
        bet[ip1] = alp[ip1] * (bet[i] - gam[i])
    
    # Back substitution for e
    e[n-1] = eend + 2.5*5.0*fend
    e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
    f[n-2] = (e[n-1] - e[n-2]) / 5.0
    e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
    f[n-3] = (e[n-2] - e[n-3]) / 5.0
    d[n-3] = dend + 1.5*4.0*eend + 1.5**2*10.0*fend
    
    # Complete back substitution
    for i in range(n-4, -1, -1):
        e[i] = e[i+1]*alp[i] + bet[i]
        f[i] = (e[i+1] - e[i]) / 5.0
        d[i] = (a[i+3] - 3.0*a[i+2] + 3.0*a[i+1] - a[i])/6.0 - \
               (e[i+3] + 27.0*e[i+2] + 93.0*e[i+1] + 59.0*e[i])/30.0
        c[i] = 0.5*(a[i+2] + a[i]) - a[i+1] - 0.5*d[i+1] - 2.5*d[i] - \
               0.1*(e[i+2] + 18.0*e[i+1] + 31.0*e[i])
        b[i] = a[i+1] - a[i] - c[i] - d[i] - 0.2*(4.0*e[i] + e[i+1])
    
    # Handle last few points
    for i in range(n-3, n):
        b[i] = b[i-1] + 2.0*c[i-1] + 3.0*d[i-1] + 4.0*e[i-1] + 5.0*f[i-1]
        c[i] = c[i-1] + 3.0*d[i-1] + 6.0*e[i-1] + 10.0*f[i-1]
        d[i] = d[i-1] + 4.0*e[i-1] + 10.0*f[i-1]
        if i != n-1:
            f[i] = a[i+1] - a[i] - b[i] - c[i] - d[i] - e[i]
    f[n-1] = f[n-2]
    
    # Scale by h factors
    fac = 1.0 / h
    b *= fac
    fac = fac / h
    c *= fac
    fac = fac / h
    d *= fac
    fac = fac / h
    e *= fac
    fac = fac / h
    f *= fac
    
    return a, b, c, d, e, f

def test_fortran_algorithm():
    """Test the Fortran-style algorithm"""
    print("Testing Fortran-style Quintic Algorithm")
    print("=" * 50)
    
    # Test with n=8
    n = 8
    x_min, x_max = 0.0, 1.0
    h = (x_max - x_min) / (n - 1)
    x = np.linspace(x_min, x_max, n)
    y = np.sin(np.pi * x)
    
    print(f"Test with n={n}, h={h:.6f}")
    
    # Apply Fortran algorithm
    a, b, c, d, e, f = quintic_spline_fortran_style(n, h, y)
    
    # Test evaluation at x=0.5
    x_test = 0.5
    idx = int(x_test / h)
    x_local = (x_test - idx * h) / h
    
    # Evaluate polynomial
    y_spline = a[idx] + x_local*(b[idx] + x_local*(c[idx] + x_local*(d[idx] + x_local*(e[idx] + x_local*f[idx]))))
    y_exact = np.sin(np.pi * x_test)
    error = abs(y_spline - y_exact)
    
    print(f"\nEvaluation at x={x_test}:")
    print(f"  Exact: {y_exact:.10f}")
    print(f"  Spline: {y_spline:.10f}")
    print(f"  Error: {error:.4e}")
    
    # Print coefficients
    print(f"\nCoefficients at interval {idx}:")
    print(f"  a[{idx}] = {a[idx]:.10f}")
    print(f"  b[{idx}] = {b[idx]:.10f}")
    print(f"  c[{idx}] = {c[idx]:.10f}")
    print(f"  d[{idx}] = {d[idx]:.10f}")
    print(f"  e[{idx}] = {e[idx]:.10f}")
    print(f"  f[{idx}] = {f[idx]:.10f}")
    
    # Test multiple points
    print(f"\nTesting multiple points:")
    test_points = [0.1, 0.25, 0.5, 0.75, 0.9]
    max_error = 0.0
    
    for xt in test_points:
        idx = int(xt / h)
        if idx >= n-1:
            idx = n-2
        x_local = (xt - idx * h) / h
        
        y_spline = a[idx] + x_local*(b[idx] + x_local*(c[idx] + x_local*(d[idx] + x_local*(e[idx] + x_local*f[idx]))))
        y_exact = np.sin(np.pi * xt)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        
        print(f"  x={xt:.2f}: exact={y_exact:.6f}, spline={y_spline:.6f}, error={error:.4e}")
    
    print(f"\nMax error: {max_error:.4e}")
    
    return max_error < 1e-6

if __name__ == "__main__":
    success = test_fortran_algorithm()
    
    print("\n" + "="*50)
    if success:
        print("SUCCESS: Fortran-style quintic algorithm works correctly!")
        print("Now we need to integrate this into the Numba implementation.")
    else:
        print("FAILURE: Algorithm needs more debugging.")
        print("Check boundary conditions and coefficient calculations.")