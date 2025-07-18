#!/usr/bin/env python3
"""
Trace e coefficient calculation in detail
"""

import numpy as np

def trace_e_coefficients():
    """Manually trace e coefficient calculation"""
    print("TRACING E COEFFICIENT CALCULATION")
    print("=" * 50)
    
    # Simple case n=6, x^4
    n = 6
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4
    
    # Constants
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    
    print(f"n={n}, h={h:.6f}")
    print(f"rhop={rhop:.6f}, rhom={rhom:.6f}")
    
    # For this test, use known boundary values
    ebeg = 0.0016  # From earlier calculation
    eend = 0.0016
    fbeg = 0.0
    fend = 0.0
    
    # Second elimination to get bet values for e calculation
    alp = np.zeros(n, dtype=np.float64)
    bet = np.zeros(n, dtype=np.float64)
    
    alp[0] = 0.0
    bet[0] = ebeg - 2.5*5.0*fbeg
    print(f"\nSecond elimination:")
    print(f"alp[0]={alp[0]:.6f}, bet[0]={bet[0]:.6f}")
    
    # Need gam values from first elimination
    # For simplicity, assume gam values are approximately correct
    gam = np.array([0.007605, 0.007591, 0.007933, 0.007933, 0.007605, 0.0])
    
    # Second elimination loop
    print("\nSecond elimination loop:")
    
    # Original Python code uses: for i in range(1, n-1)
    # This gives i = 1,2,3,4
    for i in range(1, n-1):
        ip1 = i + 1
        alp[ip1-1] = -1.0 / (rhom + alp[i-1])
        bet[ip1-1] = alp[ip1-1] * (bet[i-1] - gam[i-1])
        print(f"i={i}: alp[{ip1-1}]={alp[ip1-1]:.6f}, bet[{ip1-1}]={bet[ip1-1]:.6f}")
    
    # Calculate e values
    print("\nCalculating e values:")
    e = np.zeros(n, dtype=np.float64)
    
    e[n-1] = eend + 2.5*5.0*fend
    print(f"e[{n-1}] = eend + 12.5*fend = {e[n-1]:.6f}")
    
    e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
    print(f"e[{n-2}] = e[{n-1}]*alp[{n-2}] + bet[{n-2}] = {e[n-2]:.6f}")
    
    e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
    print(f"e[{n-3}] = e[{n-2}]*alp[{n-3}] + bet[{n-3}] = {e[n-3]:.6f}")
    
    # Main backward loop
    print("\nBackward loop for e:")
    # Original: for i in range(n-3, 0, -1)
    # This gives i = 3,2,1
    for i in range(n-3, 0, -1):
        e[i-1] = e[i]*alp[i-1] + bet[i-1]
        print(f"i={i}: e[{i-1}] = e[{i}]*alp[{i-1}] + bet[{i-1}] = {e[i-1]:.6f}")
    
    print(f"\nFinal e values: {e}")
    print(f"After scaling by 1/h^4 = {1/h**4:.2f}:")
    print(f"Scaled e values: {e * (1/h**4)}")
    
    # The scaled values should all be ~1.0 for x^4

if __name__ == "__main__":
    trace_e_coefficients()