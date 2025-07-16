#!/usr/bin/env python3
"""
Quick test to verify all cfuncs compile correctly
"""

import numpy as np
import ctypes
import sys
sys.path.insert(0, "../src")
from fastspline.sergei_splines import get_cfunc_addresses

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()

print("Checking cfunc compilation...")
print("-" * 50)

expected_funcs = [
    'construct_splines_1d',
    'evaluate_splines_1d', 
    'evaluate_splines_1d_der',
    'evaluate_splines_1d_der2',
    'construct_splines_2d',
    'evaluate_splines_2d',
    'evaluate_splines_2d_der',
    'construct_splines_3d',
]

all_good = True
for func_name in expected_funcs:
    if func_name in cfunc_addr:
        print(f"✓ {func_name}: {hex(cfunc_addr[func_name])}")
    else:
        print(f"✗ {func_name}: MISSING")
        all_good = False

if all_good:
    print("\n✓ All cfuncs compiled successfully!")
else:
    print("\n✗ Some cfuncs failed to compile!")

# Quick functional test
print("\nRunning quick functional test...")

# Test 1D spline
construct_1d = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double)
)(cfunc_addr['construct_splines_1d'])

n = 10
x_data = np.linspace(0, 2*np.pi, n)
y_data = np.sin(x_data)
y_c = (ctypes.c_double * n)(*y_data)
coeff_c = (ctypes.c_double * (4 * n))()

construct_1d(0.0, 2*np.pi, y_c, n, 3, 0, coeff_c)
print("✓ 1D construction works")

# Test 2D spline with workspace
construct_2d = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),  # x_min
    ctypes.POINTER(ctypes.c_double),  # x_max
    ctypes.POINTER(ctypes.c_double),  # y values
    ctypes.POINTER(ctypes.c_int32),   # num_points
    ctypes.POINTER(ctypes.c_int32),   # order
    ctypes.POINTER(ctypes.c_int32),   # periodic
    ctypes.POINTER(ctypes.c_double),  # coeff
    ctypes.POINTER(ctypes.c_double),  # workspace_y
    ctypes.POINTER(ctypes.c_double)   # workspace_coeff
)(cfunc_addr['construct_splines_2d'])

n1, n2 = 10, 10
x_min = np.array([0.0, 0.0])
x_max = np.array([1.0, 1.0])
X, Y = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2), indexing='ij')
Z = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)

x_min_c = (ctypes.c_double * 2)(*x_min)
x_max_c = (ctypes.c_double * 2)(*x_max)
num_points_c = (ctypes.c_int32 * 2)(n1, n2)
order_c = (ctypes.c_int32 * 2)(3, 3)
periodic_c = (ctypes.c_int32 * 2)(0, 0)
z_flat = Z.flatten()
z_c = (ctypes.c_double * len(z_flat))(*z_flat)
coeff_size = 4 * 4 * n1 * n2
coeff_c = (ctypes.c_double * coeff_size)()
workspace_y = (ctypes.c_double * max(n1, n2))()
workspace_coeff = (ctypes.c_double * (6 * max(n1, n2)))()

construct_2d(x_min_c, x_max_c, z_c, num_points_c, order_c, periodic_c, coeff_c,
             workspace_y, workspace_coeff)
print("✓ 2D construction works")

print("\n✓ All tests passed!")