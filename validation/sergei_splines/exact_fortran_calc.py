#!/usr/bin/env python3

import numpy as np

# Test data - exact same as Fortran
a = np.array([0.0, 0.642787609686539, 0.984807753012208, 0.866025403784439, 
              0.342020143325669, -0.342020143325669, -0.866025403784438, 
              -0.984807753012208, -0.642787609686540, -0.000000000000000])

n = len(a)

# Check if we're using the right indices
print("Fortran indices (1-based) -> Python indices (0-based):")
print(f"a(1) = {a[0]}")
print(f"a(2) = {a[1]}")
print(f"a(3) = {a[2]}")
print(f"a(4) = {a[3]}")
print(f"a(5) = {a[4]}")
print(f"a(6) = {a[5]}")

# Second matrix system calculation exactly as Fortran (lines 51-78)
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

print(f"\nSecond matrix system:")
print(f"a11={a11}, a12={a12}, a13={a13}")
print(f"a21={a21}, a22={a22}, a23={a23}")
print(f"a31={a31}, a32={a32}, a33={a33}")
print(f"det = {det}")

# Beginning boundary values (lines 61-63)
# Fortran: b1=a(4)+a(3)  -> Python: b1=a[3]+a[2]
# Fortran: b2=a(5)+a(2)  -> Python: b2=a[4]+a[1]
# Fortran: b3=a(6)+a(1)  -> Python: b3=a[5]+a[0]
b1 = a[3] + a[2]
b2 = a[4] + a[1]
b3 = a[5] + a[0]

print(f"\nBeginning boundary values:")
print(f"b1 = a[3] + a[2] = {a[3]} + {a[2]} = {b1}")
print(f"b2 = a[4] + a[1] = {a[4]} + {a[1]} = {b2}")
print(f"b3 = a[5] + a[0] = {a[5]} + {a[0]} = {b3}")

# Calculate abeg, cbeg, ebeg (lines 64-69)
abeg = b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32
abeg = abeg/det
cbeg = a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3
cbeg = cbeg/det
ebeg = a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32
ebeg = ebeg/det

print(f"\nCalculated values:")
print(f"abeg = {abeg}")
print(f"cbeg = {cbeg}")
print(f"ebeg = {ebeg}")

# End boundary values (lines 70-72)
# Fortran: b1=a(n-2)+a(n-3)  -> Python: b1=a[n-3]+a[n-4]
# Fortran: b2=a(n-1)+a(n-4)  -> Python: b2=a[n-2]+a[n-5]  
# Fortran: b3=a(n)+a(n-5)    -> Python: b3=a[n-1]+a[n-6]
b1_end = a[n-3] + a[n-4]
b2_end = a[n-2] + a[n-5]
b3_end = a[n-1] + a[n-6]

print(f"\nEnd boundary values:")
print(f"b1_end = a[{n-3}] + a[{n-4}] = {a[n-3]} + {a[n-4]} = {b1_end}")
print(f"b2_end = a[{n-2}] + a[{n-5}] = {a[n-2]} + {a[n-5]} = {b2_end}")
print(f"b3_end = a[{n-1}] + a[{n-6}] = {a[n-1]} + {a[n-6]} = {b3_end}")

# Calculate aend, cend, eend (lines 73-78)
aend = b1_end*a22*a33 + a12*a23*b3_end + a13*b2_end*a32 - a12*b2_end*a33 - a13*a22*b3_end - b1_end*a23*a32
aend = aend/det
cend = a11*b2_end*a33 + b1_end*a23*a31 + a13*a21*b3_end - b1_end*a21*a33 - a13*b2_end*a31 - a11*a23*b3_end
cend = cend/det
eend = a11*a22*b3_end + a12*b2_end*a31 + b1_end*a21*a32 - a12*a21*b3_end - b1_end*a22*a31 - a11*b2_end*a32
eend = eend/det

print(f"\nEnd calculated values:")
print(f"aend = {aend}")
print(f"cend = {cend}")
print(f"eend = {eend}")

print(f"\nFortran debug output shows:")
print(f"ebeg = 0.94693080664129126")
print(f"eend = -0.94693080657571382")

print(f"\nMy calculation shows:")
print(f"ebeg = {ebeg}")
print(f"eend = {eend}")

print(f"\nRatio:")
print(f"ebeg ratio = {0.94693080664129126 / ebeg}")
print(f"eend ratio = {-0.94693080657571382 / eend}")

# Maybe I need to check the first system as well? Let me go back to lines 20-50
print(f"\n=== FIRST SYSTEM ===")
# First matrix system (lines 23-32)
a11_1 = 1.0
a12_1 = 1.0/4.0
a13_1 = 1.0/16.0
a21_1 = 3.0
a22_1 = 27.0/4.0
a23_1 = 9.0*27.0/16.0
a31_1 = 5.0
a32_1 = 125.0/4.0
a33_1 = 5.0**5/16.0

det_1 = a11_1*a22_1*a33_1 + a12_1*a23_1*a31_1 + a13_1*a21_1*a32_1 - a12_1*a21_1*a33_1 - a13_1*a22_1*a31_1 - a11_1*a23_1*a32_1

print(f"First matrix det = {det_1}")

# Beginning boundary values (lines 33-35)
# Fortran: b1=a(4)-a(3)  -> Python: b1=a[3]-a[2]
# Fortran: b2=a(5)-a(2)  -> Python: b2=a[4]-a[1]
# Fortran: b3=a(6)-a(1)  -> Python: b3=a[5]-a[0]
b1_1 = a[3] - a[2]
b2_1 = a[4] - a[1]
b3_1 = a[5] - a[0]

print(f"First system boundary values:")
print(f"b1_1 = {b1_1}")
print(f"b2_1 = {b2_1}")
print(f"b3_1 = {b3_1}")

# Calculate bbeg, dbeg, fbeg (lines 36-41)
bbeg = b1_1*a22_1*a33_1 + a12_1*a23_1*b3_1 + a13_1*b2_1*a32_1 - a12_1*b2_1*a33_1 - a13_1*a22_1*b3_1 - b1_1*a23_1*a32_1
bbeg = bbeg/det_1
dbeg = a11_1*b2_1*a33_1 + b1_1*a23_1*a31_1 + a13_1*a21_1*b3_1 - b1_1*a21_1*a33_1 - a13_1*b2_1*a31_1 - a11_1*a23_1*b3_1
dbeg = dbeg/det_1
fbeg = a11_1*a22_1*b3_1 + a12_1*b2_1*a31_1 + b1_1*a21_1*a32_1 - a12_1*a21_1*b3_1 - b1_1*a22_1*a31_1 - a11_1*b2_1*a32_1
fbeg = fbeg/det_1

print(f"First system results:")
print(f"bbeg = {bbeg}")
print(f"dbeg = {dbeg}")
print(f"fbeg = {fbeg}")

print(f"\nFortran debug output shows:")
print(f"fbeg = -0.14546483754203274")
print(f"My calculation shows:")
print(f"fbeg = {fbeg}")

ratio_f = -0.14546483754203274 / fbeg
print(f"fbeg ratio = {ratio_f}")

# Maybe the issue is that I calculated the wrong matrix coefficients?
# Let me double-check the calculation as it's done in the code
print(f"\nDouble checking matrix coefficients:")
print(f"a23_1 = 9.0*27.0/16.0 = {9.0*27.0/16.0}")
print(f"a33_1 = 5.0**5/16.0 = {5.0**5/16.0}")

print(f"a23 = 81.0/8.0 = {81.0/8.0}")
print(f"a33 = 625.0/8.0 = {625.0/8.0}")