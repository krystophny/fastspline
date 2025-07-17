#!/usr/bin/env python3
"""
Systematic comparison of Fortran spl_five_reg with Python implementation
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))

def compare_implementations():
    # Test parameters
    n = 10
    x_min = 0.0
    x_max = 1.0
    h_step = (x_max - x_min) / (n - 1)
    
    # Create test data
    x = np.linspace(x_min, x_max, n)
    a = np.sin(2.0 * np.pi * x)  # Using 'a' to match Fortran variable name
    
    print("=== SYSTEMATIC COMPARISON ===")
    print(f"n = {n}, h = {h_step}")
    print(f"Input data a[0..{n-1}]:")
    for i in range(n):
        print(f"  a[{i}] = {a[i]:.12f}")
    
    # Constants
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    print(f"\nConstants:")
    print(f"  rhop = {rhop:.12f}")
    print(f"  rhom = {rhom:.12f}")
    
    # First matrix system (bbeg, dbeg, fbeg)
    print(f"\n=== FIRST MATRIX SYSTEM ===")
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
    
    print(f"Matrix coefficients:")
    print(f"  a11={a11}, a12={a12}, a13={a13}")
    print(f"  a21={a21}, a22={a22}, a23={a23}")
    print(f"  a31={a31}, a32={a32}, a33={a33}")
    print(f"  det = {det}")
    
    # Boundary values - beginning (Fortran: b1=a(4)-a(3), b2=a(5)-a(2), b3=a(6)-a(1))
    # In Python 0-based: b1=a[3]-a[2], b2=a[4]-a[1], b3=a[5]-a[0]
    b1 = a[3] - a[2]
    b2 = a[4] - a[1]
    b3 = a[5] - a[0]
    print(f"\nBoundary values (beginning):")
    print(f"  b1 = a[3] - a[2] = {a[3]} - {a[2]} = {b1}")
    print(f"  b2 = a[4] - a[1] = {a[4]} - {a[1]} = {b2}")
    print(f"  b3 = a[5] - a[0] = {a[5]} - {a[0]} = {b3}")
    
    bbeg = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    bbeg = bbeg/det
    dbeg = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    dbeg = dbeg/det
    fbeg = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    fbeg = fbeg/det
    
    print(f"  bbeg = {bbeg}")
    print(f"  dbeg = {dbeg}")
    print(f"  fbeg = {fbeg}")
    
    # Boundary values - end (Fortran: b1=a(n-2)-a(n-3), b2=a(n-1)-a(n-4), b3=a(n)-a(n-5))
    # In Python 0-based: b1=a[n-3]-a[n-4], b2=a[n-2]-a[n-5], b3=a[n-1]-a[n-6]
    b1 = a[n-3] - a[n-4]
    b2 = a[n-2] - a[n-5]
    b3 = a[n-1] - a[n-6]
    print(f"\nBoundary values (end):")
    print(f"  b1 = a[{n-3}] - a[{n-4}] = {a[n-3]} - {a[n-4]} = {b1}")
    print(f"  b2 = a[{n-2}] - a[{n-5}] = {a[n-2]} - {a[n-5]} = {b2}")
    print(f"  b3 = a[{n-1}] - a[{n-6}] = {a[n-1]} - {a[n-6]} = {b3}")
    
    bend = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    bend = bend/det
    dend = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    dend = dend/det
    fend = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    fend = fend/det
    
    print(f"  bend = {bend}")
    print(f"  dend = {dend}")
    print(f"  fend = {fend}")
    
    # Second matrix system (abeg, cbeg, ebeg, aend, cend, eend)
    print(f"\n=== SECOND MATRIX SYSTEM ===")
    a11 = 2.0
    a12 = 1.0/2.0
    a13 = 1.0/8.0
    a21 = 2.0
    a22 = 9.0/2.0
    a23 = 81.0/8.0
    a31 = 2.0
    a32 = 25.0/2.0
    a33 = 625.0/8.0
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    
    print(f"Matrix coefficients:")
    print(f"  a11={a11}, a12={a12}, a13={a13}")
    print(f"  a21={a21}, a22={a22}, a23={a23}")
    print(f"  a31={a31}, a32={a32}, a33={a33}")
    print(f"  det = {det}")
    
    # Boundary values - beginning (Fortran: b1=a(4)+a(3), b2=a(5)+a(2), b3=a(6)+a(1))
    # In Python 0-based: b1=a[3]+a[2], b2=a[4]+a[1], b3=a[5]+a[0]
    b1 = a[3] + a[2]
    b2 = a[4] + a[1]
    b3 = a[5] + a[0]
    print(f"\nBoundary values (beginning, sum):")
    print(f"  b1 = a[3] + a[2] = {a[3]} + {a[2]} = {b1}")
    print(f"  b2 = a[4] + a[1] = {a[4]} + {a[1]} = {b2}")
    print(f"  b3 = a[5] + a[0] = {a[5]} + {a[0]} = {b3}")
    
    abeg = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    abeg = abeg/det
    cbeg = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    cbeg = cbeg/det
    ebeg = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    ebeg = ebeg/det
    
    print(f"  abeg = {abeg}")
    print(f"  cbeg = {cbeg}")
    print(f"  ebeg = {ebeg}")
    
    # Boundary values - end (Fortran: b1=a(n-2)+a(n-3), b2=a(n-1)+a(n-4), b3=a(n)+a(n-5))
    # In Python 0-based: b1=a[n-3]+a[n-4], b2=a[n-2]+a[n-5], b3=a[n-1]+a[n-6]
    b1 = a[n-3] + a[n-4]
    b2 = a[n-2] + a[n-5]
    b3 = a[n-1] + a[n-6]
    print(f"\nBoundary values (end, sum):")
    print(f"  b1 = a[{n-3}] + a[{n-4}] = {a[n-3]} + {a[n-4]} = {b1}")
    print(f"  b2 = a[{n-2}] + a[{n-5}] = {a[n-2]} + {a[n-5]} = {b2}")
    print(f"  b3 = a[{n-1}] + a[{n-6}] = {a[n-1]} + {a[n-6]} = {b3}")
    
    aend = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    aend = aend/det
    cend = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    cend = cend/det
    eend = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    eend = eend/det
    
    print(f"  aend = {aend}")
    print(f"  cend = {cend}")
    print(f"  eend = {eend}")
    
    # First elimination loop for gamma
    print(f"\n=== FIRST ELIMINATION (GAMMA) ===")
    alp = np.zeros(n)
    bet = np.zeros(n)
    gam = np.zeros(n)
    
    # Fortran: alp(1)=0.0d0, bet(1)=ebeg*(2.d0+rhom)-5.d0*fbeg*(3.d0+1.5d0*rhom)
    # Python: alp[0]=0.0, bet[0]=ebeg*(2.0+rhom)-5.0*fbeg*(3.0+1.5*rhom)
    alp[0] = 0.0
    bet[0] = ebeg*(2.0+rhom) - 5.0*fbeg*(3.0+1.5*rhom)
    print(f"Initial values:")
    print(f"  alp[0] = {alp[0]}")
    print(f"  bet[0] = {bet[0]}")
    
    # Fortran: do i=1,n-4; ip1=i+1; alp(ip1)=-1.d0/(rhop+alp(i))
    # Python: for i in range(0, n-4): ip1=i+1; alp[ip1]=-1.0/(rhop+alp[i])
    print(f"Forward elimination loop i=0 to {n-5}:")
    for i in range(0, n-4):
        ip1 = i + 1
        alp[ip1] = -1.0/(rhop + alp[i])
        # Fortran: bet(ip1)=alp(ip1)*(bet(i)-5.d0*(a(i+4)-4.d0*a(i+3)+6.d0*a(i+2)-4.d0*a(ip1)+a(i)))
        # Python: bet[ip1]=alp[ip1]*(bet[i]-5.0*(a[i+4]-4.0*a[i+3]+6.0*a[i+2]-4.0*a[ip1]+a[i]))
        if i+4 < n:
            fifth_diff = a[i+4] - 4.0*a[i+3] + 6.0*a[i+2] - 4.0*a[ip1] + a[i]
        else:
            fifth_diff = 0.0
        bet[ip1] = alp[ip1] * (bet[i] - 5.0*fifth_diff)
        print(f"  i={i}: alp[{ip1}]={alp[ip1]:.12f}, bet[{ip1}]={bet[ip1]:.12f}, fifth_diff={fifth_diff:.12f}")
    
    # Fortran: gam(n-2)=eend*(2.d0+rhom)+5.d0*fend*(3.d0+1.5d0*rhom)
    # With n=10, 1-based: gam(8), 0-based: gam[7]
    # But expected shows gam[8] should have boundary value, so maybe gam[n-2]?
    gam[n-2] = eend*(2.0+rhom) + 5.0*fend*(3.0+1.5*rhom)
    print(f"Boundary condition:")
    print(f"  gam[{n-2}] = {gam[n-2]}")
    
    # Fortran: do i=n-3,1,-1; gam(i)=gam(i+1)*alp(i)+bet(i)
    # With n=10, 1-based: i=7,6,5,4,3,2,1
    # 0-based: i=6,5,4,3,2,1,0
    print(f"Backward substitution i={n-3} down to 0:")
    for i in range(n-3, -1, -1):
        gam[i] = gam[i+1] * alp[i] + bet[i]
        print(f"  i={i}: gam[{i}] = gam[{i+1}]*alp[{i}] + bet[{i}] = {gam[i+1]}*{alp[i]} + {bet[i]} = {gam[i]}")
    
    print(f"\nFinal gamma values:")
    for i in range(n):
        print(f"  gam[{i}] = {gam[i]:.12f}")
    
    # Compare with debug output from previous run
    expected_gam = [0.047851503644, 0.042669062258, 0.038295944390, 0.015108522555, 
                   -0.015112187635, -0.038207077382, -0.044731284144, 0.000000000000, 
                   -0.047851503644, 0.000000000000]
    
    print(f"\nComparison with expected gamma:")
    for i in range(n):
        print(f"  gam[{i}]: calculated={gam[i]:.12f}, expected={expected_gam[i]:.12f}, diff={abs(gam[i]-expected_gam[i]):.12f}")

if __name__ == "__main__":
    compare_implementations()