#!/usr/bin/env python3
"""
Trace intermediate values to find the bug
"""

import numpy as np

def trace_quintic_manually():
    """Manually trace quintic algorithm for n=6"""
    print("MANUAL TRACE OF QUINTIC ALGORITHM")
    print("=" * 50)
    
    n = 6
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4
    
    print(f"n = {n}, h = {h:.6f}")
    print(f"y = {y}")
    
    coeff = np.copy(y).astype(np.float64)
    
    # Constants
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    
    # Working arrays
    alp = np.zeros(n, dtype=np.float64)
    bet = np.zeros(n, dtype=np.float64)
    gam = np.zeros(n, dtype=np.float64)
    
    # Skip boundary calculations for brevity...
    # Assume we have ebeg, eend, fbeg, fend
    
    # For x^4 with proper boundary conditions:
    # fbeg ≈ 0, fend ≈ 0 (5th derivative = 0)
    # ebeg and eend relate to 4th derivative
    
    # Let's use approximate values
    ebeg = 0.0016  # Approximate
    eend = 0.0016  # Approximate
    fbeg = 0.0
    fend = 0.0
    
    print(f"\nBoundary values:")
    print(f"ebeg={ebeg:.6f}, eend={eend:.6f}")
    print(f"fbeg={fbeg:.6f}, fend={fend:.6f}")
    
    # First elimination
    print("\nFirst elimination:")
    alp[0] = 0.0
    bet[0] = ebeg*(2.0 + rhom) - 5.0*fbeg*(3.0 + 1.5*rhom)
    print(f"alp[0]={alp[0]:.6f}, bet[0]={bet[0]:.6f}")
    
    # do i=1,n-4 means i=1,2 for n=6
    # In Python: indices 0,1
    for i in range(n-4):  # i = 0, 1
        ip1 = i + 1
        alp[ip1] = -1.0 / (rhop + alp[i])
        bet[ip1] = alp[ip1] * (bet[i] - 5.0*(coeff[i+4] - 4.0*coeff[i+3] + 6.0*coeff[i+2] - 4.0*coeff[ip1] + coeff[i]))
        print(f"i={i}: alp[{ip1}]={alp[ip1]:.6f}, bet[{ip1}]={bet[ip1]:.6f}")
    
    # Back substitution
    print("\nBack substitution for gamma:")
    gam[n-2] = eend*(2.0 + rhom) + 5.0*fend*(3.0 + 1.5*rhom)
    print(f"gam[{n-2}]={gam[n-2]:.6f}")
    
    # This is the critical loop - check indices carefully
    for i in range(n-4, -1, -1):  # i = 2, 1, 0
        gam[i] = gam[i+1]*alp[i] + bet[i]
        print(f"i={i}: gam[{i}]={gam[i]:.6f} = {gam[i+1]:.6f}*{alp[i]:.6f} + {bet[i]:.6f}")
    
    # Second elimination
    print("\nSecond elimination:")
    alp[0] = 0.0
    bet[0] = ebeg - 2.5*5.0*fbeg
    print(f"alp[0]={alp[0]:.6f}, bet[0]={bet[0]:.6f}")
    
    for i in range(n-2):  # i = 0, 1, 2, 3
        ip1 = i + 1
        alp[ip1] = -1.0 / (rhom + alp[i])
        bet[ip1] = alp[ip1] * (bet[i] - gam[i])
        print(f"i={i}: alp[{ip1}]={alp[ip1]:.6f}, bet[{ip1}]={bet[ip1]:.6f}")
    
    # Calculate e values
    print("\nCalculating e values:")
    e = np.zeros(n, dtype=np.float64)
    
    e[n-1] = eend + 2.5*5.0*fend
    print(f"e[{n-1}]={e[n-1]:.6f}")
    
    e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
    print(f"e[{n-2}]={e[n-2]:.6f}")
    
    e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
    print(f"e[{n-3}]={e[n-3]:.6f}")
    
    # Main loop
    for i in range(n-4, -1, -1):  # i = 2, 1, 0
        e[i] = e[i+1]*alp[i] + bet[i]
        print(f"i={i}: e[{i}]={e[i]:.6f}")
    
    print(f"\nFinal e values: {e}")
    print(f"Should all be ~1.0 for x^4")

if __name__ == "__main__":
    trace_quintic_manually()