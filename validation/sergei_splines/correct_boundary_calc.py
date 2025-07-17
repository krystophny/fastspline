#!/usr/bin/env python3

import numpy as np

# Test data
a = np.array([0.0, 0.642787609686539, 0.984807753012208, 0.866025403784439, 
              0.342020143325669, -0.342020143325669, -0.866025403784438, 
              -0.984807753012208, -0.642787609686540, -0.000000000000000])

n = len(a)
h = 0.1111111111111111

# Constants
rhop = 13.0 + np.sqrt(105.0)
rhom = 13.0 - np.sqrt(105.0)

print(f"rhop = {rhop}")
print(f"rhom = {rhom}")

# Looking at the actual Fortran code more carefully
# Lines 83 and 98 show the Fortran code uses ebeg and fbeg differently

# The Fortran code calculates:
# 1. bbeg, dbeg, fbeg from first matrix system
# 2. abeg, cbeg, ebeg from second matrix system
# 3. Then uses ebeg (from second system) and fbeg (from first system) in the elimination

# Let me recalculate exactly as the Fortran code does it

# First matrix system (lines 23-32)
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

# Beginning boundary values
b1 = a[3] - a[2]
b2 = a[4] - a[1]  
b3 = a[5] - a[0]

bbeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
dbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
fbeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det

print(f"\nFirst system results:")
print(f"bbeg = {bbeg}")
print(f"dbeg = {dbeg}")
print(f"fbeg = {fbeg}")

# End boundary values
b1_end = a[7] - a[6]
b2_end = a[8] - a[5]
b3_end = a[9] - a[4]

bend = (b1_end*a22*a33 + a12*a23*b3_end + a13*b2_end*a32 - a12*b2_end*a33 - a13*a22*b3_end - b1_end*a23*a32) / det
dend = (a11*b2_end*a33 + b1_end*a23*a31 + a13*a21*b3_end - b1_end*a21*a33 - a13*b2_end*a31 - a11*a23*b3_end) / det
fend = (a11*a22*b3_end + a12*b2_end*a31 + b1_end*a21*a32 - a12*a21*b3_end - b1_end*a22*a31 - a11*b2_end*a32) / det

print(f"bend = {bend}")
print(f"dend = {dend}")
print(f"fend = {fend}")

# Second matrix system (lines 51-60)
a11_2 = 2.0
a12_2 = 1.0/2.0
a13_2 = 1.0/8.0
a21_2 = 2.0
a22_2 = 9.0/2.0
a23_2 = 81.0/8.0
a31_2 = 2.0
a32_2 = 25.0/2.0
a33_2 = 625.0/8.0

det_2 = a11_2*a22_2*a33_2 + a12_2*a23_2*a31_2 + a13_2*a21_2*a32_2 - a12_2*a21_2*a33_2 - a13_2*a22_2*a31_2 - a11_2*a23_2*a32_2

# Beginning boundary values for second system
b1_2 = a[3] + a[2]
b2_2 = a[4] + a[1]
b3_2 = a[5] + a[0]

abeg = (b1_2*a22_2*a33_2 + a12_2*a23_2*b3_2 + a13_2*b2_2*a32_2 - a12_2*b2_2*a33_2 - a13_2*a22_2*b3_2 - b1_2*a23_2*a32_2) / det_2
cbeg = (a11_2*b2_2*a33_2 + b1_2*a23_2*a31_2 + a13_2*a21_2*b3_2 - b1_2*a21_2*a33_2 - a13_2*b2_2*a31_2 - a11_2*a23_2*b3_2) / det_2
ebeg = (a11_2*a22_2*b3_2 + a12_2*b2_2*a31_2 + b1_2*a21_2*a32_2 - a12_2*a21_2*b3_2 - b1_2*a22_2*a31_2 - a11_2*b2_2*a32_2) / det_2

print(f"\nSecond system results:")
print(f"abeg = {abeg}")
print(f"cbeg = {cbeg}")
print(f"ebeg = {ebeg}")

# End boundary values for second system
b1_2_end = a[7] + a[6]
b2_2_end = a[8] + a[5]
b3_2_end = a[9] + a[4]

aend = (b1_2_end*a22_2*a33_2 + a12_2*a23_2*b3_2_end + a13_2*b2_2_end*a32_2 - a12_2*b2_2_end*a33_2 - a13_2*a22_2*b3_2_end - b1_2_end*a23_2*a32_2) / det_2
cend = (a11_2*b2_2_end*a33_2 + b1_2_end*a23_2*a31_2 + a13_2*a21_2*b3_2_end - b1_2_end*a21_2*a33_2 - a13_2*b2_2_end*a31_2 - a11_2*a23_2*b3_2_end) / det_2
eend = (a11_2*a22_2*b3_2_end + a12_2*b2_2_end*a31_2 + b1_2_end*a21_2*a32_2 - a12_2*a21_2*b3_2_end - b1_2_end*a22_2*a31_2 - a11_2*b2_2_end*a32_2) / det_2

print(f"aend = {aend}")
print(f"cend = {cend}")
print(f"eend = {eend}")

# Now let me understand what the Fortran code is doing at lines 83 and 98
print(f"\nFortran line 83: bet(1)=ebeg*(2.d0+rhom)-5.d0*fbeg*(3.d0+1.5d0*rhom)")
print(f"This uses ebeg from second system and fbeg from first system")

bet_1_gamma = ebeg*(2.0+rhom) - 5.0*fbeg*(3.0+1.5*rhom)
print(f"bet[1] for gamma = {bet_1_gamma}")

print(f"\nFortran line 98: bet(1)=ebeg-2.5d0*5.d0*fbeg")
print(f"This uses ebeg from second system and fbeg from first system")

bet_1_e = ebeg - 2.5*5.0*fbeg
print(f"bet[1] for e = {bet_1_e}")

# Similarly for the end values
print(f"\nEnd values:")
bet_n_gamma = eend*(2.0+rhom) + 5.0*fend*(3.0+1.5*rhom)
print(f"bet[n] for gamma = {bet_n_gamma}")

bet_n_e = eend + 2.5*5.0*fend
print(f"bet[n] for e = {bet_n_e}")

print(f"\nFortran debug shows:")
print(f"ebeg = 0.94693080664129126")
print(f"fbeg = -0.14546483754203274")
print(f"eend = -0.94693080657571382")
print(f"fend = -0.14546483746950811")

print(f"\nBut wait, the Fortran debug shows different values than my calculation!")
print(f"This suggests the Fortran code is doing something different...")

# Let me check if there's some other scaling or transformation
# Maybe the Fortran code is actually using the h scaling?

# Check if the Fortran values are scaled versions of mine
print(f"\nChecking if Fortran values are scaled versions:")
print(f"ebeg ratio: {0.94693080664129126 / ebeg}")
print(f"fbeg ratio: {-0.14546483754203274 / fbeg}")
print(f"eend ratio: {-0.94693080657571382 / eend}")
print(f"fend ratio: {-0.14546483746950811 / fend}")

# Maybe the Fortran code is using different input values?
# Let me check the values that would give the Fortran results
print(f"\nReverse engineering:")
print(f"To get ebeg = 0.946930807, we need ebeg_calc * factor = 0.946930807")
print(f"Factor = {0.94693080664129126 / ebeg}")

factor = 0.94693080664129126 / ebeg
print(f"If we scale ebeg by {factor}, we get {ebeg * factor}")
print(f"If we scale fbeg by {factor}, we get {fbeg * factor}")

# Hmm, this suggests the Fortran code might be using different matrix coefficients
# or there's some other transformation I'm missing