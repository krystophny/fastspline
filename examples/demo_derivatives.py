#!/usr/bin/env python3
"""
Test script for second derivative evaluation
"""

import numpy as np
import ctypes
import sys
sys.path.insert(0, "../src")
from fastspline.sergei_splines import get_cfunc_addresses

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()

# Set up ctypes signatures
construct_1d = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double)
)(cfunc_addr['construct_splines_1d'])

evaluate_1d_der2 = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
)(cfunc_addr['evaluate_splines_1d_der2'])

# Test with a cubic polynomial: f(x) = x^3 - 2x^2 + 3x + 1
# f'(x) = 3x^2 - 4x + 3
# f''(x) = 6x - 4

n_data = 20
x_min = -2.0
x_max = 2.0
order = 3

x_data = np.linspace(x_min, x_max, n_data)
y_data = x_data**3 - 2*x_data**2 + 3*x_data + 1

# Prepare ctypes arrays
y_c = (ctypes.c_double * n_data)(*y_data)
coeff_size = (order + 1) * n_data
coeff_c = (ctypes.c_double * coeff_size)()

# Construct spline
h_step = (x_max - x_min) / (n_data - 1)
construct_1d(x_min, x_max, y_c, n_data, order, 0, coeff_c)

# Test at several points
test_points = np.linspace(x_min + 0.1, x_max - 0.1, 10)

print("Testing second derivative evaluation:")
print(f"{'x':>10} {'f(x)':>12} {'f_prime':>12} {'f_double_prime':>12} {'True_f_double_prime':>20} {'Error':>12}")
print("-" * 80)

for x in test_points:
    y_out = (ctypes.c_double * 1)()
    dy_out = (ctypes.c_double * 1)()
    d2y_out = (ctypes.c_double * 1)()
    
    evaluate_1d_der2(order, n_data, 0, x_min, h_step, coeff_c, x, y_out, dy_out, d2y_out)
    
    # True values
    true_y = x**3 - 2*x**2 + 3*x + 1
    true_dy = 3*x**2 - 4*x + 3
    true_d2y = 6*x - 4
    
    error_d2y = abs(d2y_out[0] - true_d2y)
    
    print(f"{x:>10.3f} {y_out[0]:>12.6f} {dy_out[0]:>12.6f} {d2y_out[0]:>12.6f} {true_d2y:>12.6f} {error_d2y:>12.2e}")

print("\nTesting with sine function:")
x_data = np.linspace(0, 2*np.pi, n_data)
y_data = np.sin(x_data)

# Prepare ctypes arrays
x_min = 0.0
x_max = 2*np.pi
y_c = (ctypes.c_double * n_data)(*y_data)
h_step = (x_max - x_min) / (n_data - 1)
construct_1d(x_min, x_max, y_c, n_data, order, 0, coeff_c)

test_points = np.linspace(0.5, 2*np.pi - 0.5, 10)

print(f"\n{'x':>10} {'f(x)':>12} {'f_prime':>12} {'f_double_prime':>12} {'True_f_double_prime':>20} {'Error':>12}")
print("-" * 80)

for x in test_points:
    y_out = (ctypes.c_double * 1)()
    dy_out = (ctypes.c_double * 1)()
    d2y_out = (ctypes.c_double * 1)()
    
    evaluate_1d_der2(order, n_data, 0, x_min, h_step, coeff_c, x, y_out, dy_out, d2y_out)
    
    # True values
    true_y = np.sin(x)
    true_dy = np.cos(x)
    true_d2y = -np.sin(x)
    
    error_d2y = abs(d2y_out[0] - true_d2y)
    
    print(f"{x:>10.3f} {y_out[0]:>12.6f} {dy_out[0]:>12.6f} {d2y_out[0]:>12.6f} {true_d2y:>12.6f} {error_d2y:>12.2e}")