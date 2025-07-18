#!/usr/bin/env python3
"""
Trace gam calculation step by step
"""

n = 6

print("GAM CALCULATION ANALYSIS")
print("=" * 50)
print()

print("FORTRAN (1-based):")
print(f"gam({n-2}) = gam(4) is set before loop")
print(f"Loop: do i={n-3},1,-1  (i goes from 3 down to 1)")
print(f"  i=3: gam(3) = gam(4)*alp(3) + bet(3)")
print(f"  i=2: gam(2) = gam(3)*alp(2) + bet(2)")
print(f"  i=1: gam(1) = gam(2)*alp(1) + bet(1)")
print()

print("PYTHON (0-based) - CURRENT WRONG CODE:")
print(f"gam[{n-2}] = gam[4] is set before loop")
print(f"Loop: for i in range({n-3}, 0, -1):  (i goes from 3 down to 1)")
print(f"  i=3: gam[2] = gam[3]*alp[2] + bet[2]  # ERROR: gam[3] not set!")
print(f"  i=2: gam[1] = gam[2]*alp[1] + bet[1]")
print(f"  i=1: gam[0] = gam[1]*alp[0] + bet[0]")
print()

print("CORRECT PYTHON MAPPING:")
print("Need to calculate gam[3] first, then gam[2], then gam[1], then gam[0]")
print()

print("Option 1: Direct index mapping")
print(f"gam[{n-2}] = gam[4] is set before loop")
print(f"Loop: for i in range({n-3}, 0, -1):  (i goes from 3 down to 1)")
print(f"  i=3: gam[3-1] = gam[3]*alp[3-1] + bet[3-1]  # gam[2] = gam[3]*alp[2] + bet[2]")
print("  BUT this still uses gam[3] which isn't set!")
print()

print("Option 2: Calculate in correct order")
print(f"gam[{n-2}] = gam[4] is set")
print(f"Need to calculate indices 3, 2, 1, 0 in that order:")
print(f"  gam[3] = gam[4]*alp[3] + bet[3]")
print(f"  gam[2] = gam[3]*alp[2] + bet[2]")
print(f"  gam[1] = gam[2]*alp[1] + bet[1]")
print(f"  gam[0] = gam[1]*alp[0] + bet[0]")
print()

print("The key insight: Fortran loop variable i represents the INDEX BEING CALCULATED")
print("So when Fortran says 'gam(i) = ...', Python needs 'gam[i-1] = ...'")