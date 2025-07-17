#!/usr/bin/env python3
"""
Continue systematic comparison: second elimination phase
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))

def continue_comparison():
    # Test parameters
    n = 10
    x_min = 0.0
    x_max = 1.0
    h_step = (x_max - x_min) / (n - 1)
    
    # Create test data
    x = np.linspace(x_min, x_max, n)
    a = np.sin(2.0 * np.pi * x)
    
    # Constants
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    
    # Recompute boundary conditions (from previous comparison)
    # First system
    a11 = 1.0; a12 = 1.0/4.0; a13 = 1.0/16.0
    a21 = 3.0; a22 = 27.0/4.0; a23 = 9.0*27.0/16.0
    a31 = 5.0; a32 = 125.0/4.0; a33 = 5.0**5/16.0
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    
    b1 = a[3] - a[2]; b2 = a[4] - a[1]; b3 = a[5] - a[0]
    bbeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32)/det
    dbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3)/det
    fbeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32)/det
    
    b1 = a[n-3] - a[n-4]; b2 = a[n-2] - a[n-5]; b3 = a[n-1] - a[n-6]
    bend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32)/det
    dend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3)/det
    fend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32)/det
    
    # Second system
    a11 = 2.0; a12 = 1.0/2.0; a13 = 1.0/8.0
    a21 = 2.0; a22 = 9.0/2.0; a23 = 81.0/8.0
    a31 = 2.0; a32 = 25.0/2.0; a33 = 625.0/8.0
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    
    b1 = a[3] + a[2]; b2 = a[4] + a[1]; b3 = a[5] + a[0]
    abeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32)/det
    cbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3)/det
    ebeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32)/det
    
    b1 = a[n-3] + a[n-4]; b2 = a[n-2] + a[n-5]; b3 = a[n-1] + a[n-6]
    aend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32)/det
    cend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3)/det
    eend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32)/det
    
    # Compute correct gamma values
    alp = np.zeros(n)
    bet = np.zeros(n)
    gam = np.zeros(n)
    
    # First elimination for gamma
    alp[0] = 0.0
    bet[0] = ebeg*(2.0+rhom) - 5.0*fbeg*(3.0+1.5*rhom)
    
    for i in range(0, n-4):
        ip1 = i + 1
        alp[ip1] = -1.0/(rhop + alp[i])
        if i+4 < n:
            fifth_diff = a[i+4] - 4.0*a[i+3] + 6.0*a[i+2] - 4.0*a[ip1] + a[i]
        else:
            fifth_diff = 0.0
        bet[ip1] = alp[ip1] * (bet[i] - 5.0*fifth_diff)
    
    # Back substitution for gamma
    gam[n-2] = eend*(2.0+rhom) + 5.0*fend*(3.0+1.5*rhom)
    for i in range(n-3, -1, -1):
        gam[i] = gam[i+1] * alp[i] + bet[i]
    
    print(f"=== SECOND ELIMINATION (E COEFFICIENTS) ===")
    
    # Fortran: alp(1)=0.0d0, bet(1)=ebeg-2.5d0*5.d0*fbeg
    # Python: alp[0]=0.0, bet[0]=ebeg-2.5*5.0*fbeg
    alp[0] = 0.0
    bet[0] = ebeg - 2.5*5.0*fbeg
    print(f"Initial values:")
    print(f"  alp[0] = {alp[0]}")
    print(f"  bet[0] = {bet[0]}")
    
    # Fortran: do i=1,n-2; ip1=i+1; alp(ip1)=-1.d0/(rhom+alp(i)); bet(ip1)=alp(ip1)*(bet(i)-gam(i))
    # Python: for i in range(0, n-2): ip1=i+1; alp[ip1]=-1.0/(rhom+alp[i]); bet[ip1]=alp[ip1]*(bet[i]-gam[i])
    print(f"Forward elimination loop i=0 to {n-3}:")
    for i in range(0, n-2):
        ip1 = i + 1
        alp[ip1] = -1.0/(rhom + alp[i])
        bet[ip1] = alp[ip1] * (bet[i] - gam[i])
        print(f"  i={i}: alp[{ip1}]={alp[ip1]:.12f}, bet[{ip1}]={bet[ip1]:.12f}, gam[{i}]={gam[i]:.12f}")
    
    # Initialize e and f arrays
    e = np.zeros(n)
    f = np.zeros(n)
    
    # Fortran: e(n)=eend+2.5d0*5.d0*fend
    # Python: e[n-1]=eend+2.5*5.0*fend
    e[n-1] = eend + 2.5*5.0*fend
    print(f"\\nBoundary condition:")
    print(f"  e[{n-1}] = eend + 2.5*5.0*fend = {eend} + 2.5*5.0*{fend} = {e[n-1]}")
    
    # Fortran: e(n-1)=e(n)*alp(n-1)+bet(n-1)
    # Python: e[n-2]=e[n-1]*alp[n-2]+bet[n-2]
    e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
    print(f"  e[{n-2}] = e[{n-1}]*alp[{n-2}] + bet[{n-2}] = {e[n-1]}*{alp[n-2]} + {bet[n-2]} = {e[n-2]}")
    
    # Fortran: f(n-1)=(e(n)-e(n-1))/5.d0
    # Python: f[n-2]=(e[n-1]-e[n-2])/5.0
    f[n-2] = (e[n-1] - e[n-2])/5.0
    print(f"  f[{n-2}] = (e[{n-1}] - e[{n-2}])/5.0 = ({e[n-1]} - {e[n-2]})/5.0 = {f[n-2]}")
    
    # Fortran: e(n-2)=e(n-1)*alp(n-2)+bet(n-2)
    # Python: e[n-3]=e[n-2]*alp[n-3]+bet[n-3]
    e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
    print(f"  e[{n-3}] = e[{n-2}]*alp[{n-3}] + bet[{n-3}] = {e[n-2]}*{alp[n-3]} + {bet[n-3]} = {e[n-3]}")
    
    # Fortran: f(n-2)=(e(n-1)-e(n-2))/5.d0
    # Python: f[n-3]=(e[n-2]-e[n-3])/5.0
    f[n-3] = (e[n-2] - e[n-3])/5.0
    print(f"  f[{n-3}] = (e[{n-2}] - e[{n-3}])/5.0 = ({e[n-2]} - {e[n-3]})/5.0 = {f[n-3]}")
    
    # Fortran: d(n-2)=dend+1.5d0*4.d0*eend+1.5d0**2*10.d0*fend
    # Python: (no d calculation here, just for reference)
    d_n_minus_2 = dend + 1.5*4.0*eend + 1.5**2*10.0*fend
    print(f"  d[{n-3}] = dend + 1.5*4.0*eend + 1.5^2*10.0*fend = {dend} + 1.5*4.0*{eend} + 1.5^2*10.0*{fend} = {d_n_minus_2}")
    
    # Main backward loop
    print(f"\\nMain backward loop:")
    # Fortran: do i=n-3,1,-1
    # With n=10, 1-based: i=7,6,5,4,3,2,1
    # Python 0-based: i=6,5,4,3,2,1,0
    for i in range(n-4, -1, -1):
        # Fortran: e(i)=e(i+1)*alp(i)+bet(i)
        # Python: e[i]=e[i+1]*alp[i]+bet[i]
        e[i] = e[i+1]*alp[i] + bet[i]
        
        # Fortran: f(i)=(e(i+1)-e(i))/5.d0
        # Python: f[i]=(e[i+1]-e[i])/5.0
        f[i] = (e[i+1] - e[i])/5.0
        
        print(f"  i={i}: e[{i}]={e[i]:.12f}, f[{i}]={f[i]:.12f}")
    
    print(f"\\nFinal e coefficients (before scaling):")
    for i in range(n):
        print(f"  e[{i}] = {e[i]:.12f}")
    
    print(f"\\nFinal f coefficients (before scaling):")
    for i in range(n):
        print(f"  f[{i}] = {f[i]:.12f}")
    
    # Apply scaling
    print(f"\\nApplying scaling:")
    fac = 1.0/h_step
    print(f"  Initial fac = 1/h_step = {fac}")
    
    # Scale e coefficients (4th power)
    for i in range(4):
        fac /= h_step
    print(f"  fac for e (4th power) = {fac}")
    
    e_scaled = e * fac
    print(f"  e coefficients after scaling:")
    for i in range(n):
        print(f"    e[{i}] = {e_scaled[i]:.12f}")
    
    # Scale f coefficients (5th power)
    fac /= h_step
    print(f"  fac for f (5th power) = {fac}")
    
    f_scaled = f * fac
    print(f"  f coefficients after scaling:")
    for i in range(n):
        print(f"    f[{i}] = {f_scaled[i]:.12f}")
    
    # Compare with expected
    expected_f = [-9.215304710619, -26.239525644771, 20.250410797674, -81.156412065059, 
                  -70.663660507931, -81.156412065050, 20.250410797658, -26.239525644656, 
                  -9.215304713004, -9.215304713004]
    
    print(f"\\nComparison with expected f:")
    for i in range(n):
        diff = abs(f_scaled[i] - expected_f[i])
        print(f"  f[{i}]: calculated={f_scaled[i]:.12f}, expected={expected_f[i]:.12f}, diff={diff:.12f}")

if __name__ == "__main__":
    continue_comparison()