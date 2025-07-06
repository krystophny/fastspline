"""
Test the corrected f2py wrapper against our Numba implementation
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dierckx_numba_simple import (
    fpback_njit, fpgivs_njit, fprota_njit, fprati_njit
)
import dierckx_f2py

print("="*70)
print("TESTING CORRECTED F2PY WRAPPER")
print("="*70)

# Test 1: fpback (now returns correct size)
print("\n1. Testing fpback:")
n = 5
k = 3
nest = 10
np.random.seed(42)

a = np.zeros((nest, k), dtype=np.float64, order='F')
for i in range(n):
    a[i, 0] = 2.0 + 0.1 * i
    for j in range(1, min(k, n-i)):
        a[i, j] = 0.5 / (j + 1)
z = np.random.randn(n)

# f2py result
c_f2py = dierckx_f2py.fpback(a.copy(), z.copy(), n, k, nest)
print(f"   f2py result shape: {c_f2py.shape}")
print(f"   f2py result: {c_f2py}")

# Numba result
c_numba = np.zeros(n, dtype=np.float64)
fpback_njit(a.copy(), z.copy(), n, k, c_numba, nest)
print(f"   Numba result: {c_numba}")

# Compare
if c_f2py.shape[0] == n:
    error = np.max(np.abs(c_f2py - c_numba))
    print(f"   Max error: {error:.2e}")
    print(f"   ✅ fpback: {'PASS' if error < 1e-14 else 'FAIL'}")
else:
    print(f"   ❌ fpback: FAIL - wrong output size")

# Test 2: fprati (now works as function)
print("\n2. Testing fprati:")
p1, f1, p2, f2, p3, f3 = 1.0, 2.0, 2.0, 1.0, 3.0, -1.0

# f2py result
p_f2py = dierckx_f2py.fprati(p1, f1, p2, f2, p3, f3)
print(f"   f2py result: {p_f2py}")

# Numba result
p_numba, _, _, _, _ = fprati_njit(p1, f1, p2, f2, p3, f3)
print(f"   Numba result: {p_numba}")

# Compare
error = abs(p_f2py - p_numba)
print(f"   Error: {error:.2e}")
print(f"   ✅ fprati: {'PASS' if error < 1e-14 else 'FAIL'}")

# Test 3: fpgivs
print("\n3. Testing fpgivs:")
piv = 3.0
ww = 4.0

# f2py result
piv_f2py, ww_f2py, cos_f2py, sin_f2py = dierckx_f2py.fpgivs(piv, ww)
print(f"   f2py result: ww={ww_f2py}, cos={cos_f2py}, sin={sin_f2py}")

# Numba result
ww_numba, cos_numba, sin_numba = fpgivs_njit(piv, ww)
print(f"   Numba result: ww={ww_numba}, cos={cos_numba}, sin={sin_numba}")

# Compare
ww_error = abs(ww_f2py - ww_numba)
cos_error = abs(cos_f2py - cos_numba)
sin_error = abs(sin_f2py - sin_numba)
max_error = max(ww_error, cos_error, sin_error)
print(f"   Max error: {max_error:.2e}")
print(f"   ✅ fpgivs: {'PASS' if max_error < 1e-14 else 'FAIL'}")

# Test 4: fprota
print("\n4. Testing fprota:")
cos = 0.0
sin = 1.0
a = 2.0
b = -1.0

# f2py result
a_f2py, b_f2py = dierckx_f2py.fprota(cos, sin, a, b)
print(f"   f2py result: a={a_f2py}, b={b_f2py}")

# Numba result
a_numba, b_numba = fprota_njit(cos, sin, a, b)
print(f"   Numba result: a={a_numba}, b={b_numba}")

# Compare
a_error = abs(a_f2py - a_numba)
b_error = abs(b_f2py - b_numba)
max_error = max(a_error, b_error)
print(f"   Max error: {max_error:.2e}")
print(f"   ✅ fprota: {'PASS' if max_error < 1e-14 else 'FAIL'}")

print("\n" + "="*70)
print("SUMMARY: With corrected f2py interface, functions can be properly validated!")
print("="*70)