#!/usr/bin/env python3
"""
Complete systematic comparison of Fortran spl_five_reg with Python implementation
Line by line comparison and verification
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))

def complete_systematic_comparison():
    # Test parameters - exactly as in validation
    n = 10
    h = 0.1111111111111111  # Using h instead of h_step to match Fortran
    
    # Create test data - exactly as in validation
    x_min = 0.0
    x_max = 1.0
    x = np.linspace(x_min, x_max, n)
    a = np.sin(2.0 * np.pi * x)  # Using 'a' to match Fortran variable name
    
    print("=== COMPLETE SYSTEMATIC COMPARISON ===")
    print(f"n = {n}")
    print(f"h = {h}")
    print("Input data a:")
    for i in range(n):
        print(f"  a[{i}] = {a[i]:.15f}")
    
    # Constants - line 20-21 in Fortran
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    print(f"\nConstants:")
    print(f"  rhop = {rhop:.15f}")
    print(f"  rhom = {rhom:.15f}")
    
    # First matrix system - lines 23-32 in Fortran
    print(f"\n=== FIRST MATRIX SYSTEM (lines 23-32) ===")
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
    
    print(f"Matrix elements:")
    print(f"  a11 = {a11:.15f}")
    print(f"  a12 = {a12:.15f}")
    print(f"  a13 = {a13:.15f}")
    print(f"  a21 = {a21:.15f}")
    print(f"  a22 = {a22:.15f}")
    print(f"  a23 = {a23:.15f}")
    print(f"  a31 = {a31:.15f}")
    print(f"  a32 = {a32:.15f}")
    print(f"  a33 = {a33:.15f}")
    print(f"  det = {det:.15f}")
    
    # Boundary values for beginning - lines 33-35 in Fortran
    # Fortran: b1=a(4)-a(3), b2=a(5)-a(2), b3=a(6)-a(1)
    # Python: b1=a[3]-a[2], b2=a[4]-a[1], b3=a[5]-a[0]
    print(f"\n=== BEGINNING BOUNDARY VALUES (lines 33-35) ===")
    b1 = a[3] - a[2]  # a(4) - a(3) in Fortran
    b2 = a[4] - a[1]  # a(5) - a(2) in Fortran
    b3 = a[5] - a[0]  # a(6) - a(1) in Fortran
    print(f"  b1 = a[3] - a[2] = {a[3]:.15f} - {a[2]:.15f} = {b1:.15f}")
    print(f"  b2 = a[4] - a[1] = {a[4]:.15f} - {a[1]:.15f} = {b2:.15f}")
    print(f"  b3 = a[5] - a[0] = {a[5]:.15f} - {a[0]:.15f} = {b3:.15f}")
    
    # Calculate bbeg, dbeg, fbeg - lines 36-41 in Fortran
    print(f"\n=== CALCULATE BBEG, DBEG, FBEG (lines 36-41) ===")
    bbeg = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    bbeg = bbeg/det
    dbeg = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    dbeg = dbeg/det
    fbeg = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    fbeg = fbeg/det
    
    print(f"  bbeg = {bbeg:.15f}")
    print(f"  dbeg = {dbeg:.15f}")
    print(f"  fbeg = {fbeg:.15f}")
    
    # Boundary values for end - lines 42-44 in Fortran
    # Fortran: b1=a(n-2)-a(n-3), b2=a(n-1)-a(n-4), b3=a(n)-a(n-5)
    # Python: b1=a[n-3]-a[n-4], b2=a[n-2]-a[n-5], b3=a[n-1]-a[n-6]
    print(f"\n=== END BOUNDARY VALUES (lines 42-44) ===")
    b1 = a[n-3] - a[n-4]  # a(n-2) - a(n-3) in Fortran
    b2 = a[n-2] - a[n-5]  # a(n-1) - a(n-4) in Fortran
    b3 = a[n-1] - a[n-6]  # a(n) - a(n-5) in Fortran
    print(f"  b1 = a[{n-3}] - a[{n-4}] = {a[n-3]:.15f} - {a[n-4]:.15f} = {b1:.15f}")
    print(f"  b2 = a[{n-2}] - a[{n-5}] = {a[n-2]:.15f} - {a[n-5]:.15f} = {b2:.15f}")
    print(f"  b3 = a[{n-1}] - a[{n-6}] = {a[n-1]:.15f} - {a[n-6]:.15f} = {b3:.15f}")
    
    # Calculate bend, dend, fend - lines 45-50 in Fortran
    print(f"\n=== CALCULATE BEND, DEND, FEND (lines 45-50) ===")
    bend = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    bend = bend/det
    dend = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    dend = dend/det
    fend = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    fend = fend/det
    
    print(f"  bend = {bend:.15f}")
    print(f"  dend = {dend:.15f}")
    print(f"  fend = {fend:.15f}")
    
    # Second matrix system - lines 51-60 in Fortran
    print(f"\n=== SECOND MATRIX SYSTEM (lines 51-60) ===")
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
    
    print(f"Matrix elements:")
    print(f"  a11 = {a11:.15f}")
    print(f"  a12 = {a12:.15f}")
    print(f"  a13 = {a13:.15f}")
    print(f"  a21 = {a21:.15f}")
    print(f"  a22 = {a22:.15f}")
    print(f"  a23 = {a23:.15f}")
    print(f"  a31 = {a31:.15f}")
    print(f"  a32 = {a32:.15f}")
    print(f"  a33 = {a33:.15f}")
    print(f"  det = {det:.15f}")
    
    # Beginning boundary values for second system - lines 61-63 in Fortran
    # Fortran: b1=a(4)+a(3), b2=a(5)+a(2), b3=a(6)+a(1)
    # Python: b1=a[3]+a[2], b2=a[4]+a[1], b3=a[5]+a[0]
    print(f"\n=== BEGINNING BOUNDARY VALUES FOR SECOND SYSTEM (lines 61-63) ===")
    b1 = a[3] + a[2]  # a(4) + a(3) in Fortran
    b2 = a[4] + a[1]  # a(5) + a(2) in Fortran
    b3 = a[5] + a[0]  # a(6) + a(1) in Fortran
    print(f"  b1 = a[3] + a[2] = {a[3]:.15f} + {a[2]:.15f} = {b1:.15f}")
    print(f"  b2 = a[4] + a[1] = {a[4]:.15f} + {a[1]:.15f} = {b2:.15f}")
    print(f"  b3 = a[5] + a[0] = {a[5]:.15f} + {a[0]:.15f} = {b3:.15f}")
    
    # Calculate abeg, cbeg, ebeg - lines 64-69 in Fortran
    print(f"\n=== CALCULATE ABEG, CBEG, EBEG (lines 64-69) ===")
    abeg = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    abeg = abeg/det
    cbeg = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    cbeg = cbeg/det
    ebeg = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    ebeg = ebeg/det
    
    print(f"  abeg = {abeg:.15f}")
    print(f"  cbeg = {cbeg:.15f}")
    print(f"  ebeg = {ebeg:.15f}")
    
    # End boundary values for second system - lines 70-72 in Fortran
    # Fortran: b1=a(n-2)+a(n-3), b2=a(n-1)+a(n-4), b3=a(n)+a(n-5)
    # Python: b1=a[n-3]+a[n-4], b2=a[n-2]+a[n-5], b3=a[n-1]+a[n-6]
    print(f"\n=== END BOUNDARY VALUES FOR SECOND SYSTEM (lines 70-72) ===")
    b1 = a[n-3] + a[n-4]  # a(n-2) + a(n-3) in Fortran
    b2 = a[n-2] + a[n-5]  # a(n-1) + a(n-4) in Fortran
    b3 = a[n-1] + a[n-6]  # a(n) + a(n-5) in Fortran
    print(f"  b1 = a[{n-3}] + a[{n-4}] = {a[n-3]:.15f} + {a[n-4]:.15f} = {b1:.15f}")
    print(f"  b2 = a[{n-2}] + a[{n-5}] = {a[n-2]:.15f} + {a[n-5]:.15f} = {b2:.15f}")
    print(f"  b3 = a[{n-1}] + a[{n-6}] = {a[n-1]:.15f} + {a[n-6]:.15f} = {b3:.15f}")
    
    # Calculate aend, cend, eend - lines 73-78 in Fortran
    print(f"\n=== CALCULATE AEND, CEND, EEND (lines 73-78) ===")
    aend = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
    aend = aend/det
    cend = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
    cend = cend/det
    eend = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
    eend = eend/det
    
    print(f"  aend = {aend:.15f}")
    print(f"  cend = {cend:.15f}")
    print(f"  eend = {eend:.15f}")
    
    # Allocate arrays - line 80 in Fortran
    print(f"\n=== ALLOCATE ARRAYS (line 80) ===")
    alp = np.zeros(n, dtype=np.float64)
    bet = np.zeros(n, dtype=np.float64)
    gam = np.zeros(n, dtype=np.float64)
    
    # First elimination - lines 82-83 in Fortran
    print(f"\n=== FIRST ELIMINATION INITIALIZATION (lines 82-83) ===")
    # Fortran: alp(1)=0.0d0
    # Python: alp[0]=0.0
    alp[0] = 0.0
    # Fortran: bet(1)=ebeg*(2.d0+rhom)-5.d0*fbeg*(3.d0+1.5d0*rhom)
    # Python: bet[0]=ebeg*(2.0+rhom)-5.0*fbeg*(3.0+1.5*rhom)
    bet[0] = ebeg*(2.0+rhom) - 5.0*fbeg*(3.0+1.5*rhom)
    print(f"  alp[0] = {alp[0]:.15f}")
    print(f"  bet[0] = {bet[0]:.15f}")
    
    # First elimination loop - lines 85-90 in Fortran
    print(f"\n=== FIRST ELIMINATION LOOP (lines 85-90) ===")
    # Fortran: do i=1,n-4
    # Python: for i in range(0, n-4)
    for i in range(0, n-4):
        # Fortran: ip1=i+1
        # Python: ip1=i+1 (but we use i+1 directly)
        ip1 = i + 1
        
        # Fortran: alp(ip1)=-1.d0/(rhop+alp(i))
        # Python: alp[ip1]=-1.0/(rhop+alp[i])
        alp[ip1] = -1.0/(rhop + alp[i])
        
        # Fortran: bet(ip1)=alp(ip1)*(bet(i)-5.d0*(a(i+4)-4.d0*a(i+3)+6.d0*a(i+2)-4.d0*a(ip1)+a(i)))
        # Python: bet[ip1]=alp[ip1]*(bet[i]-5.0*(a[i+4]-4.0*a[i+3]+6.0*a[i+2]-4.0*a[ip1]+a[i]))
        if i+4 < n:
            # Note: In Fortran a(i+4) means a[i+3] in Python (0-based)
            # But wait, let me check this carefully...
            # Fortran loop: i=1,n-4, so i goes 1,2,3,4,5,6 (for n=10)
            # Python loop: i=0,1,2,3,4,5 (for n=10)
            # So Fortran a(i+4) with i=1 means a(5) = a[4] in Python
            # But Python a[i+4] with i=0 means a[4]
            # So this should be correct as written
            fifth_diff = a[i+4] - 4.0*a[i+3] + 6.0*a[i+2] - 4.0*a[ip1] + a[i]
        else:
            fifth_diff = 0.0
        
        bet[ip1] = alp[ip1] * (bet[i] - 5.0*fifth_diff)
        
        print(f"  i={i}: alp[{ip1}] = {alp[ip1]:.15f}")
        print(f"  i={i}: bet[{ip1}] = {bet[ip1]:.15f}")
        print(f"  i={i}: fifth_diff = {fifth_diff:.15f}")
    
    # Back substitution initialization - line 92 in Fortran
    print(f"\n=== BACK SUBSTITUTION INITIALIZATION (line 92) ===")
    # Fortran: gam(n-2)=eend*(2.d0+rhom)+5.d0*fend*(3.d0+1.5d0*rhom)
    # With n=10, Fortran 1-based: gam(8)
    # Python 0-based: gam[7]
    # BUT WAIT! Let me double-check this...
    # Actually, let me verify what the expected values show
    # From previous debug: gam[8] should be -0.047851503644
    # So maybe it should be gam[n-2] in Python = gam[8]
    
    # Let me test both and see which matches expected
    gam_test1 = np.zeros(n)
    gam_test2 = np.zeros(n)
    
    gam_test1[n-3] = eend*(2.0+rhom) + 5.0*fend*(3.0+1.5*rhom)  # gam[7]
    gam_test2[n-2] = eend*(2.0+rhom) + 5.0*fend*(3.0+1.5*rhom)  # gam[8]
    
    print(f"  Testing gam[{n-3}] = {gam_test1[n-3]:.15f}")
    print(f"  Testing gam[{n-2}] = {gam_test2[n-2]:.15f}")
    print(f"  Expected gam[8] = -0.047851503644")
    
    # Based on expected values, gam[8] should have the boundary value
    gam[n-2] = eend*(2.0+rhom) + 5.0*fend*(3.0+1.5*rhom)
    print(f"  gam[{n-2}] = {gam[n-2]:.15f}")
    
    # Back substitution loop - lines 93-95 in Fortran
    print(f"\n=== BACK SUBSTITUTION LOOP (lines 93-95) ===")
    # Fortran: do i=n-3,1,-1
    # With n=10: i=7,6,5,4,3,2,1
    # Python: for i in range(n-4, -1, -1)
    # With n=10: i=6,5,4,3,2,1,0
    for i in range(n-4, -1, -1):
        # Fortran: gam(i)=gam(i+1)*alp(i)+bet(i)
        # Python: gam[i]=gam[i+1]*alp[i]+bet[i]
        gam[i] = gam[i+1] * alp[i] + bet[i]
        print(f"  i={i}: gam[{i}] = gam[{i+1}]*alp[{i}] + bet[{i}] = {gam[i+1]:.15f}*{alp[i]:.15f} + {bet[i]:.15f} = {gam[i]:.15f}")
    
    print(f"\nFinal gamma values:")
    for i in range(n):
        print(f"  gam[{i}] = {gam[i]:.15f}")
    
    # Expected gamma values for verification
    expected_gam = [0.047851503644, 0.042669062258, 0.038295944390, 0.015108522555, 
                   -0.015112187635, -0.038207077382, -0.044731284144, 0.000000000000, 
                   -0.047851503644, 0.000000000000]
    
    print(f"\nVerification of gamma values:")
    for i in range(n):
        diff = abs(gam[i] - expected_gam[i])
        print(f"  gam[{i}]: calculated={gam[i]:.12f}, expected={expected_gam[i]:.12f}, diff={diff:.12f}")
    
    # Now continue with the second elimination...
    print(f"\n=== SECOND ELIMINATION INITIALIZATION (lines 97-98) ===")
    # Reset alp and bet for second elimination
    # Fortran: alp(1)=0.0d0
    # Python: alp[0]=0.0
    alp[0] = 0.0
    # Fortran: bet(1)=ebeg-2.5d0*5.d0*fbeg
    # Python: bet[0]=ebeg-2.5*5.0*fbeg
    bet[0] = ebeg - 2.5*5.0*fbeg
    print(f"  alp[0] = {alp[0]:.15f}")
    print(f"  bet[0] = {bet[0]:.15f}")
    
    # Second elimination loop - lines 100-104 in Fortran
    print(f"\n=== SECOND ELIMINATION LOOP (lines 100-104) ===")
    # Fortran: do i=1,n-2
    # Python: for i in range(0, n-2)
    for i in range(0, n-2):
        # Fortran: ip1=i+1
        # Python: ip1=i+1
        ip1 = i + 1
        
        # Fortran: alp(ip1)=-1.d0/(rhom+alp(i))
        # Python: alp[ip1]=-1.0/(rhom+alp[i])
        alp[ip1] = -1.0/(rhom + alp[i])
        
        # Fortran: bet(ip1)=alp(ip1)*(bet(i)-gam(i))
        # Python: bet[ip1]=alp[ip1]*(bet[i]-gam[i])
        bet[ip1] = alp[ip1] * (bet[i] - gam[i])
        
        print(f"  i={i}: alp[{ip1}] = {alp[ip1]:.15f}")
        print(f"  i={i}: bet[{ip1}] = {bet[ip1]:.15f}")
        print(f"  i={i}: gam[{i}] = {gam[i]:.15f}")
    
    # Initialize result arrays
    b = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    e = np.zeros(n, dtype=np.float64)
    f = np.zeros(n, dtype=np.float64)
    
    # Back substitution for e - lines 106-111 in Fortran
    print(f"\n=== BACK SUBSTITUTION FOR E (lines 106-111) ===")
    # Fortran: e(n)=eend+2.5d0*5.d0*fend
    # Python: e[n-1]=eend+2.5*5.0*fend
    e[n-1] = eend + 2.5*5.0*fend
    print(f"  e[{n-1}] = {e[n-1]:.15f}")
    
    # Fortran: e(n-1)=e(n)*alp(n-1)+bet(n-1)
    # Python: e[n-2]=e[n-1]*alp[n-2]+bet[n-2]
    e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
    print(f"  e[{n-2}] = {e[n-2]:.15f}")
    
    # Fortran: f(n-1)=(e(n)-e(n-1))/5.d0
    # Python: f[n-2]=(e[n-1]-e[n-2])/5.0
    f[n-2] = (e[n-1] - e[n-2])/5.0
    print(f"  f[{n-2}] = {f[n-2]:.15f}")
    
    # Fortran: e(n-2)=e(n-1)*alp(n-2)+bet(n-2)
    # Python: e[n-3]=e[n-2]*alp[n-3]+bet[n-3]
    e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
    print(f"  e[{n-3}] = {e[n-3]:.15f}")
    
    # Fortran: f(n-2)=(e(n-1)-e(n-2))/5.d0
    # Python: f[n-3]=(e[n-2]-e[n-3])/5.0
    f[n-3] = (e[n-2] - e[n-3])/5.0
    print(f"  f[{n-3}] = {f[n-3]:.15f}")
    
    # Fortran: d(n-2)=dend+1.5d0*4.d0*eend+1.5d0**2*10.d0*fend
    # Python: d[n-3]=dend+1.5*4.0*eend+1.5**2*10.0*fend
    d[n-3] = dend + 1.5*4.0*eend + 1.5**2*10.0*fend
    print(f"  d[{n-3}] = {d[n-3]:.15f}")

    # Main loop - lines 113-121 in Fortran
    print(f"\n=== MAIN LOOP (lines 113-121) ===")
    # Fortran: do i=n-3,1,-1
    # With n=10: i=7,6,5,4,3,2,1
    # Python: for i in range(n-4, -1, -1)
    # With n=10: i=6,5,4,3,2,1,0
    for i in range(n-4, -1, -1):
        # Fortran: e(i)=e(i+1)*alp(i)+bet(i)
        # Python: e[i]=e[i+1]*alp[i]+bet[i]
        e[i] = e[i+1]*alp[i] + bet[i]
        
        # Fortran: f(i)=(e(i+1)-e(i))/5.d0
        # Python: f[i]=(e[i+1]-e[i])/5.0
        f[i] = (e[i+1] - e[i])/5.0
        
        # Fortran: d(i)=(a(i+3)-3.d0*a(i+2)+3.d0*a(i+1)-a(i))/6.d0-(e(i+3)+27.d0*e(i+2)+93.d0*e(i+1)+59.d0*e(i))/30.d0
        # Python: d[i]=(a[i+3]-3.0*a[i+2]+3.0*a[i+1]-a[i])/6.0-(e[i+3]+27.0*e[i+2]+93.0*e[i+1]+59.0*e[i])/30.0
        if i+3 < n:
            fourth_diff = a[i+3] - 3.0*a[i+2] + 3.0*a[i+1] - a[i]
            e_term = e[i+3] + 27.0*e[i+2] + 93.0*e[i+1] + 59.0*e[i]
            d[i] = fourth_diff/6.0 - e_term/30.0
        else:
            d[i] = 0.0
        
        # Fortran: c(i)=0.5d0*(a(i+2)+a(i))-a(i+1)-0.5d0*d(i+1)-2.5d0*d(i)-0.1d0*(e(i+2)+18.d0*e(i+1)+31.d0*e(i))
        # Python: c[i]=0.5*(a[i+2]+a[i])-a[i+1]-0.5*d[i+1]-2.5*d[i]-0.1*(e[i+2]+18.0*e[i+1]+31.0*e[i])
        if i+2 < n:
            c_term = 0.5*(a[i+2] + a[i]) - a[i+1]
            if i+1 < n:
                c_term -= 0.5*d[i+1]
            c_term -= 2.5*d[i]
            e_contrib = e[i+2] + 18.0*e[i+1] + 31.0*e[i]
            c_term -= 0.1*e_contrib
            c[i] = c_term
        else:
            c[i] = 0.0
        
        # Fortran: b(i)=a(i+1)-a(i)-c(i)-d(i)-0.2d0*(4.d0*e(i)+e(i+1))
        # Python: b[i]=a[i+1]-a[i]-c[i]-d[i]-0.2*(4.0*e[i]+e[i+1])
        if i+1 < n:
            b_term = a[i+1] - a[i] - c[i] - d[i]
            e_contrib = 4.0*e[i] + e[i+1]
            b_term -= 0.2*e_contrib
            b[i] = b_term
        else:
            b[i] = 0.0
        
        print(f"  i={i}: e[{i}] = {e[i]:.15f}")
        print(f"  i={i}: f[{i}] = {f[i]:.15f}")
        print(f"  i={i}: d[{i}] = {d[i]:.15f}")
        print(f"  i={i}: c[{i}] = {c[i]:.15f}")
        print(f"  i={i}: b[{i}] = {b[i]:.15f}")
    
    # Final coefficient calculation - lines 123-129 in Fortran
    print(f"\n=== FINAL COEFFICIENT CALCULATION (lines 123-129) ===")
    # Fortran: do i=n-3,n
    # With n=10: i=7,8,9,10
    # Python: for i in range(n-4, n)
    # With n=10: i=6,7,8,9
    # But wait, Fortran goes to n=10, so in Python it should be range(n-4, n+1)
    # Actually, let me check the exact Fortran bounds...
    
    # Let me be more careful about this loop
    # Fortran: do i=n-3,n means i=7,8,9,10 for n=10
    # But arrays in Fortran are 1-based, so valid indices are 1,2,...,10
    # So i=10 is valid in Fortran
    # In Python 0-based: valid indices are 0,1,...,9
    # So the equivalent loop should be for i in range(n-4, n)
    # Which gives i=6,7,8,9 for n=10
    
    for i in range(n-4, n):
        if i >= 0:  # Safety check
            # Fortran: b(i)=b(i-1)+2.d0*c(i-1)+3.d0*d(i-1)+4.d0*e(i-1)+5.d0*f(i-1)
            # Python: b[i]=b[i-1]+2.0*c[i-1]+3.0*d[i-1]+4.0*e[i-1]+5.0*f[i-1]
            b[i] = b[i-1] + 2.0*c[i-1] + 3.0*d[i-1] + 4.0*e[i-1] + 5.0*f[i-1]
            
            # Fortran: c(i)=c(i-1)+3.d0*d(i-1)+6.d0*e(i-1)+10.d0*f(i-1)
            # Python: c[i]=c[i-1]+3.0*d[i-1]+6.0*e[i-1]+10.0*f[i-1]
            c[i] = c[i-1] + 3.0*d[i-1] + 6.0*e[i-1] + 10.0*f[i-1]
            
            # Fortran: d(i)=d(i-1)+4.d0*e(i-1)+10.d0*f(i-1)
            # Python: d[i]=d[i-1]+4.0*e[i-1]+10.0*f[i-1]
            d[i] = d[i-1] + 4.0*e[i-1] + 10.0*f[i-1]
            
            # Fortran: if(i.ne.n) f(i)=a(i+1)-a(i)-b(i)-c(i)-d(i)-e(i)
            # Python: if i != n-1: f[i]=a[i+1]-a[i]-b[i]-c[i]-d[i]-e[i]
            if i != n-1:
                f[i] = a[i+1] - a[i] - b[i] - c[i] - d[i] - e[i]
            
            print(f"  i={i}: b[{i}] = {b[i]:.15f}")
            print(f"  i={i}: c[{i}] = {c[i]:.15f}")
            print(f"  i={i}: d[{i}] = {d[i]:.15f}")
            if i != n-1:
                print(f"  i={i}: f[{i}] = {f[i]:.15f}")
    
    # Final f value - line 129 in Fortran
    print(f"\n=== FINAL F VALUE (line 129) ===")
    # Fortran: f(n)=f(n-1)
    # Python: f[n-1]=f[n-2]
    f[n-1] = f[n-2]
    print(f"  f[{n-1}] = {f[n-1]:.15f}")
    
    # Print final coefficients before scaling
    print(f"\n=== FINAL COEFFICIENTS BEFORE SCALING ===")
    print(f"b coefficients:")
    for i in range(n):
        print(f"  b[{i}] = {b[i]:.15f}")
    
    print(f"c coefficients:")
    for i in range(n):
        print(f"  c[{i}] = {c[i]:.15f}")
    
    print(f"d coefficients:")
    for i in range(n):
        print(f"  d[{i}] = {d[i]:.15f}")
    
    print(f"e coefficients:")
    for i in range(n):
        print(f"  e[{i}] = {e[i]:.15f}")
    
    print(f"f coefficients:")
    for i in range(n):
        print(f"  f[{i}] = {f[i]:.15f}")
    
    # Scaling - lines 131-140 in Fortran
    print(f"\n=== SCALING (lines 131-140) ===")
    # Fortran: fac=1.d0/h
    # Python: fac=1.0/h
    fac = 1.0/h
    print(f"  Initial fac = {fac:.15f}")
    
    # Fortran: b=b*fac
    # Python: b=b*fac
    b = b * fac
    print(f"  After scaling b, fac = {fac:.15f}")
    
    # Fortran: fac=fac/h
    # Python: fac=fac/h
    fac = fac/h
    # Fortran: c=c*fac
    # Python: c=c*fac
    c = c * fac
    print(f"  After scaling c, fac = {fac:.15f}")
    
    # Fortran: fac=fac/h
    # Python: fac=fac/h
    fac = fac/h
    # Fortran: d=d*fac
    # Python: d=d*fac
    d = d * fac
    print(f"  After scaling d, fac = {fac:.15f}")
    
    # Fortran: fac=fac/h
    # Python: fac=fac/h
    fac = fac/h
    # Fortran: e=e*fac
    # Python: e=e*fac
    e = e * fac
    print(f"  After scaling e, fac = {fac:.15f}")
    
    # Fortran: fac=fac/h
    # Python: fac=fac/h
    fac = fac/h
    # Fortran: f=f*fac
    # Python: f=f*fac
    f = f * fac
    print(f"  After scaling f, fac = {fac:.15f}")
    
    # Print final scaled coefficients
    print(f"\n=== FINAL SCALED COEFFICIENTS ===")
    print(f"f coefficients (scaled):")
    for i in range(n):
        print(f"  f[{i}] = {f[i]:.15f}")
    
    # Compare with expected
    expected_f = [-9.215304710619, -26.239525644771, 20.250410797674, -81.156412065059, 
                  -70.663660507931, -81.156412065050, 20.250410797658, -26.239525644656, 
                  -9.215304713004, -9.215304713004]
    
    print(f"\n=== COMPARISON WITH EXPECTED ===")
    for i in range(n):
        diff = abs(f[i] - expected_f[i])
        print(f"  f[{i}]: calculated={f[i]:.12f}, expected={expected_f[i]:.12f}, diff={diff:.12f}")
        if diff > 1e-10:
            print(f"    ERROR: Difference too large!")

if __name__ == "__main__":
    complete_systematic_comparison()