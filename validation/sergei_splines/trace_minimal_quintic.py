#!/usr/bin/env python3
"""
Trace quintic algorithm for minimal test case
"""

import numpy as np

def trace_minimal_case():
    """Trace algorithm for n=6 points (minimum for quintic)"""
    print("MINIMAL QUINTIC TRACE (n=6)")
    print("=" * 50)
    
    n = 6
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4  # Should give constant e coefficients
    
    print(f"Test: x^4 with n={n} points")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"h = {h:.6f}")
    
    # Copy y values
    coeff = np.copy(y).astype(np.float64)
    
    # Constants
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    print(f"\nConstants: rhop={rhop:.6f}, rhom={rhom:.6f}")
    
    # First boundary system
    a11 = 1.0
    a12 = 1.0/4.0
    a13 = 1.0/16.0
    a21 = 3.0
    a22 = 27.0/4.0
    a23 = 9.0*27.0/16.0
    a31 = 5.0
    a32 = 125.0/4.0
    a33 = 5.0**5/16.0
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    
    print(f"\nFirst system det={det:.6f}")
    
    # Beginning boundary
    b1 = coeff[3] - coeff[2]  # y[3] - y[2]
    b2 = coeff[4] - coeff[1]  # y[4] - y[1]
    b3 = coeff[5] - coeff[0]  # y[5] - y[0]
    
    print(f"\nBeginning differences:")
    print(f"b1 = y[3]-y[2] = {coeff[3]:.6f} - {coeff[2]:.6f} = {b1:.6f}")
    print(f"b2 = y[4]-y[1] = {coeff[4]:.6f} - {coeff[1]:.6f} = {b2:.6f}")
    print(f"b3 = y[5]-y[0] = {coeff[5]:.6f} - {coeff[0]:.6f} = {b3:.6f}")
    
    bbeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    dbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    fbeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    
    print(f"\nBoundary values: bbeg={bbeg:.6f}, dbeg={dbeg:.6f}, fbeg={fbeg:.6e}")
    
    # For x^4, the 5th derivative is 0, so fbeg should be very small
    print(f"\nFor x^4: 5th derivative = 0, so fbeg should be ~0")
    print(f"Actual fbeg = {fbeg:.2e}")
    
    # Working arrays
    alp = np.zeros(n, dtype=np.float64)
    bet = np.zeros(n, dtype=np.float64)
    gam = np.zeros(n, dtype=np.float64)
    
    # First elimination
    alp[0] = 0.0
    bet[0] = ebeg*(2.0 + rhom) - 5.0*fbeg*(3.0 + 1.5*rhom)  # Wait, ebeg not defined yet!
    
    print(f"\nERROR: ebeg is used before it's calculated!")
    print(f"This is a critical bug in the algorithm ordering")

def check_algorithm_order():
    """Check the correct order of calculations"""
    print("\n" + "=" * 50)
    print("ALGORITHM ORDER CHECK")
    print("=" * 50)
    
    print("Looking at Fortran lines 82-83:")
    print("  alp(1)=0.0d0")
    print("  bet(1)=ebeg*(2.d0+rhom)-5.d0*fbeg*(3.d0+1.5d0*rhom) !gamma1")
    print("\nebeg is calculated at line 69, fbeg at line 41")
    print("So ebeg IS defined when used at line 83")
    print("\nMy implementation must have ebeg calculated before first elimination")

if __name__ == "__main__":
    trace_minimal_case()
    check_algorithm_order()