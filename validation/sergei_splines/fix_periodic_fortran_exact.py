#!/usr/bin/env python3
"""
Exact implementation of Fortran splper periodic cubic spline algorithm
"""

import numpy as np
from numba import njit

@njit
def spfper_exact(n, amx1, amx2, amx3):
    """
    Exact implementation of Fortran spfper helper routine
    """
    n1 = n - 1
    
    # Initial values - exactly as in Fortran
    amx1[0] = 2.0
    amx2[0] = 0.5
    amx3[0] = 0.5
    amx1[1] = np.sqrt(15.0) / 2.0
    amx2[1] = 1.0 / amx1[1]
    amx3[1] = -0.25 / amx1[1]
    beta = 3.75
    
    # Loop - exactly as in Fortran
    for i in range(2, n1):
        i1 = i - 1
        beta = 4.0 - 1.0 / beta
        amx1[i] = np.sqrt(beta)
        amx2[i] = 1.0 / amx1[i]
        amx3[i] = -amx3[i1] / amx1[i] / amx1[i1]
    
    # Final adjustments - exactly as in Fortran
    amx3[n1-1] = amx3[n1-1] + 1.0 / amx1[n1-1]
    amx2[n1-1] = amx3[n1-1]
    
    ss = 0.0
    for i in range(n1):
        ss = ss + amx3[i] * amx3[i]
    amx1[n-1] = np.sqrt(4.0 - ss)

@njit
def splper_exact(n, h, y, bi, ci, di):
    """
    Exact implementation of Fortran splper periodic cubic spline
    """
    # Allocate arrays
    bmx = np.zeros(n)
    yl = np.zeros(n)
    amx1 = np.zeros(n)
    amx2 = np.zeros(n)
    amx3 = np.zeros(n)
    
    # Initialize bmx[0] (unused sentinel)
    bmx[0] = 1e30
    
    # Set up indices - exactly as in Fortran
    nmx = n - 1
    n1 = nmx - 1
    n2 = nmx - 2
    psi = 3.0 / h / h
    
    # Call spfper to set up matrix
    spfper_exact(n, amx1, amx2, amx3)
    
    # Set up right-hand side - exactly as in Fortran
    # bmx(nmx) = (y(nmx+1)-2*y(nmx)+y(nmx-1))*psi
    # Note: y(nmx+1) in Fortran is y[nmx] in Python (wraps to y[0])
    bmx[nmx-1] = (y[0] - 2.0*y[nmx-1] + y[nmx-2]) * psi
    
    # bmx(1) = (y(2)-y(1)-y(nmx+1)+y(nmx))*psi  
    # Note: y(nmx+1) wraps to y(1) in Fortran = y[0] in Python
    bmx[0] = (y[1] - y[0] - y[0] + y[nmx-1]) * psi
    
    # DO i = 3,nmx: bmx(i-1) = (y(i)-2*y(i-1)+y(i-2))*psi
    for i in range(2, nmx):  # i = 3,nmx in Fortran maps to i = 2,nmx-1 in Python
        bmx[i-1] = (y[i] - 2.0*y[i-1] + y[i-2]) * psi
    
    # Forward elimination - exactly as in Fortran
    yl[0] = bmx[0] / amx1[0]
    for i in range(1, n1):  # i = 2,n1 in Fortran
        i1 = i - 1
        yl[i] = (bmx[i] - yl[i1]*amx2[i1]) / amx1[i]
    
    # Sum calculation - exactly as in Fortran
    ss = 0.0
    for i in range(n1):  # i = 1,n1 in Fortran
        ss = ss + yl[i] * amx3[i]
    yl[nmx-1] = (bmx[nmx-1] - ss) / amx1[nmx-1]
    
    # Back substitution - exactly as in Fortran
    bmx[nmx-1] = yl[nmx-1] / amx1[nmx-1]
    bmx[n1-1] = (yl[n1-1] - amx2[n1-1]*bmx[nmx-1]) / amx1[n1-1]
    for i in range(n2-1, -1, -1):  # i = n2,1,-1 in Fortran
        bmx[i] = (yl[i] - amx3[i]*bmx[nmx-1] - amx2[i]*bmx[i+1]) / amx1[i]
    
    # Copy c coefficients - exactly as in Fortran
    for i in range(nmx):  # i = 1,nmx in Fortran
        ci[i] = bmx[i]
    
    # Calculate b and d coefficients - exactly as in Fortran
    for i in range(n1):  # i = 1,n1 in Fortran
        bi[i] = (y[i+1] - y[i]) / h - h * (ci[i+1] + 2.0*ci[i]) / 3.0
        di[i] = (ci[i+1] - ci[i]) / h / 3.0
    
    # Periodic boundary calculations - exactly as in Fortran
    # bi(nmx) = (y(n)-y(n-1))/h-h*(ci(1)+2*ci(nmx))/3
    # Note: y(n) in Fortran wraps to y[0] in Python, ci(1) is ci[0] in Python
    bi[nmx-1] = (y[0] - y[nmx-1]) / h - h * (ci[0] + 2.0*ci[nmx-1]) / 3.0
    di[nmx-1] = (ci[0] - ci[nmx-1]) / h / 3.0
    
    # Fix periodicity boundary - exactly as in Fortran
    bi[n-1] = bi[0]
    ci[n-1] = ci[0]
    di[n-1] = di[0]

def test_fortran_exact_periodic():
    """Test the exact Fortran implementation"""
    print("Testing Exact Fortran Periodic Cubic Spline Implementation")
    print("=" * 60)
    
    # Test with same data as validation
    n = 16
    x_min, x_max = 0.0, 1.0
    h = (x_max - x_min) / n  # Periodic: h = L/n
    
    # Test function: sin(2πx)
    x = np.linspace(x_min, x_max, n, endpoint=False)
    y = np.sin(2*np.pi*x)
    
    print(f"Test function: sin(2πx) on [{x_min}, {x_max})")
    print(f"Grid: n = {n} points (periodic)")
    print(f"Step size: h = {h:.10f}")
    
    # Allocate coefficient arrays
    bi = np.zeros(n)
    ci = np.zeros(n)
    di = np.zeros(n)
    
    # Construct spline using exact Fortran algorithm
    splper_exact(n, h, y, bi, ci, di)
    
    # Test continuity at boundaries
    print("\nBoundary Continuity Tests:")
    print("-" * 40)
    
    # Function value continuity
    f_0 = y[0]  # a[0]
    f_2pi_left = y[n-1] + bi[n-1] + ci[n-1] + di[n-1]  # evaluate at right boundary
    print(f"f(0) = {f_0:.12f}")
    print(f"f(1-) = {f_2pi_left:.12f}")
    print(f"Function difference: {abs(f_0 - f_2pi_left):.2e}")
    
    # First derivative continuity
    df_0 = bi[0]
    df_2pi_left = bi[n-1] + 2*ci[n-1] + 3*di[n-1]
    print(f"\nf'(0) = {df_0:.12f}")
    print(f"f'(1-) = {df_2pi_left:.12f}")
    print(f"First derivative difference: {abs(df_0 - df_2pi_left):.2e}")
    
    # Second derivative continuity
    d2f_0 = 2*ci[0]
    d2f_2pi_left = 2*ci[n-1] + 6*di[n-1]
    print(f"\nf''(0) = {d2f_0:.12f}")
    print(f"f''(1-) = {d2f_2pi_left:.12f}")
    print(f"Second derivative difference: {abs(d2f_0 - d2f_2pi_left):.2e}")
    
    # Test accuracy at various points
    print("\n\nAccuracy Tests:")
    print("-" * 40)
    
    test_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    max_error = 0.0
    
    for x_test in test_x:
        # Find interval and evaluate
        i = int(x_test / h)
        if i >= n:
            i = n - 1
        t = (x_test - i*h) / h
        
        # Cubic polynomial evaluation: a + t*(b + t*(c + t*d))
        y_spline = y[i] + t*(bi[i] + t*(ci[i] + t*di[i]))
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_spline - y_exact)
        max_error = max(max_error, error)
        
        print(f"x = {x_test:.3f}: exact = {y_exact:.8f}, "
              f"spline = {y_spline:.8f}, error = {error:.2e}")
    
    print(f"\nMax error: {max_error:.2e}")
    
    # Check floating point precision
    print("\n" + "="*60)
    cont_f = abs(f_0 - f_2pi_left)
    cont_df = abs(df_0 - df_2pi_left)
    cont_d2f = abs(d2f_0 - d2f_2pi_left)
    
    tolerance = 1e-14  # Machine precision tolerance
    
    if cont_f < tolerance and cont_df < tolerance and cont_d2f < tolerance:
        print("✅ SUCCESS: Floating point precision achieved!")
        print(f"   Function continuity: {cont_f:.2e}")
        print(f"   Derivative continuity: {cont_df:.2e}")
        print(f"   Second derivative continuity: {cont_d2f:.2e}")
        return True
    else:
        print("❌ FAILURE: Not at floating point precision")
        print(f"   Function continuity: {cont_f:.2e} (target: < {tolerance:.0e})")
        print(f"   Derivative continuity: {cont_df:.2e} (target: < {tolerance:.0e})")
        print(f"   Second derivative continuity: {cont_d2f:.2e} (target: < {tolerance:.0e})")
        return False

if __name__ == "__main__":
    test_fortran_exact_periodic()