#!/usr/bin/env python3
"""
Trace loop bounds in detail for n=6
"""

# For n=6:
n = 6

print("FORTRAN LOOP BOUNDS ANALYSIS (1-based indexing)")
print("=" * 50)
print(f"n = {n}")
print(f"n-2 = {n-2}")
print(f"n-3 = {n-3}")
print(f"n-4 = {n-4}")
print()

print("FIRST ELIMINATION:")
print("do i=1,n-4  means i goes from 1 to", n-4)
print("When i=1: sets alp(2) and bet(2)")
print("When i=2: sets alp(3) and bet(3)")
print("So alp and bet have values at indices: 1,2,3")
print()

print("BACK SUBSTITUTION:")
print("gam(n-2) is set, which is gam(4)")
print("do i=n-3,1,-1  means i goes from", n-3, "down to 1")
print("When i=3: calculates gam(3) = gam(4)*alp(3) + bet(3)")
print("When i=2: calculates gam(2) = gam(3)*alp(2) + bet(2)")
print("When i=1: calculates gam(1) = gam(2)*alp(1) + bet(1)")
print()

print("PYTHON EQUIVALENT (0-based indexing)")
print("=" * 50)
print("FIRST ELIMINATION:")
print("for i in range(1, n-3):  means i goes from 1 to", n-4)
print("When i=1: sets alp[1] and bet[1]")
print("When i=2: sets alp[2] and bet[2]")
print("So alp and bet have values at indices: 0,1,2")
print()

print("BACK SUBSTITUTION:")
print("gam[n-2] is set, which is gam[4]")
print("for i in range(n-3, 0, -1):  means i goes from", n-3, "down to 1")
print("When i=3: calculates gam[2] = gam[3]*alp[2] + bet[2]")
print("When i=2: calculates gam[1] = gam[2]*alp[1] + bet[1]")
print("When i=1: calculates gam[0] = gam[1]*alp[0] + bet[0]")
print()

print("ISSUE: When i=3, we need gam[3], but gam[3] was never set!")
print("gam[4] is set before the loop, but gam[3] is still 0")