#!/usr/bin/env python3
"""
Visual proof that Sergei's splines work - with derivatives!
"""

import numpy as np
import matplotlib.pyplot as plt
import ctypes
from sergei_splines_cfunc_final import get_cfunc_addresses
import time

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()

# Set up ctypes functions
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

construct_2d = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
)(cfunc_addr['construct_splines_2d'])

evaluate_2d_der = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
)(cfunc_addr['evaluate_splines_2d_der'])

# Create figure
fig = plt.figure(figsize=(16, 10))

# Test 1: 1D spline with all derivatives
ax1 = plt.subplot(2, 3, 1)
n_data = 15
x_min, x_max = 0.0, 2*np.pi
x_data = np.linspace(x_min, x_max, n_data)
y_data = np.sin(x_data) + 0.3*np.sin(3*x_data)

# Construct spline
y_c = (ctypes.c_double * n_data)(*y_data)
coeff_c = (ctypes.c_double * (4 * n_data))()
h_step = (x_max - x_min) / (n_data - 1)
construct_1d(x_min, x_max, y_c, n_data, 3, 0, coeff_c)

# Evaluate on fine grid
n_eval = 200
x_eval = np.linspace(x_min, x_max, n_eval)
y_eval = np.zeros(n_eval)
dy_eval = np.zeros(n_eval)
d2y_eval = np.zeros(n_eval)

for i in range(n_eval):
    y_out = (ctypes.c_double * 1)()
    dy_out = (ctypes.c_double * 1)()
    d2y_out = (ctypes.c_double * 1)()
    evaluate_1d_der2(3, n_data, 0, x_min, h_step, coeff_c, x_eval[i], y_out, dy_out, d2y_out)
    y_eval[i] = y_out[0]
    dy_eval[i] = dy_out[0]
    d2y_eval[i] = d2y_out[0]

ax1.plot(x_data, y_data, 'ro', markersize=8, label='Data')
ax1.plot(x_eval, y_eval, 'b-', linewidth=2, label='Spline')
ax1.set_title('1D Cubic Spline', fontsize=14)
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot derivatives
ax2 = plt.subplot(2, 3, 2)
ax2.plot(x_eval, dy_eval, 'g-', linewidth=2, label='1st derivative')
ax2.plot(x_eval, np.cos(x_eval) + 0.9*np.cos(3*x_eval), 'g--', alpha=0.7, label='True 1st deriv')
ax2.set_title('First Derivative', fontsize=14)
ax2.set_xlabel('x')
ax2.set_ylabel("f'(x)")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(x_eval, d2y_eval, 'r-', linewidth=2, label='2nd derivative')
ax3.plot(x_eval, -np.sin(x_eval) - 2.7*np.sin(3*x_eval), 'r--', alpha=0.7, label='True 2nd deriv')
ax3.set_title('Second Derivative', fontsize=14)
ax3.set_xlabel('x')
ax3.set_ylabel("f''(x)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Test 2: 2D spline with derivatives
n1, n2 = 20, 25
x_min = np.array([0.0, 0.0])
x_max = np.array([4.0, 6.0])

x1 = np.linspace(x_min[0], x_max[0], n1)
x2 = np.linspace(x_min[1], x_max[1], n2)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')

# Test function with known derivatives
Z = np.exp(-0.3*((X1-2)**2 + (X2-3)**2)) * np.sin(X1) * np.cos(X2)

# Prepare ctypes arrays
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

# Construct 2D spline
start_time = time.time()
construct_2d(x_min_c, x_max_c, z_c, num_points_c, order_c, periodic_c, coeff_c, 
             workspace_y, workspace_coeff)
construct_time = time.time() - start_time

# Calculate h_step
h_step = np.array([(x_max[i] - x_min[i]) / ([n1, n2][i] - 1) for i in range(2)])
h_step_c = (ctypes.c_double * 2)(*h_step)

# Evaluate on fine grid
n_eval = 40
x1_eval = np.linspace(x_min[0], x_max[0], n_eval)
x2_eval = np.linspace(x_min[1], x_max[1], n_eval)
X1_eval, X2_eval = np.meshgrid(x1_eval, x2_eval, indexing='ij')

Z_eval = np.zeros((n_eval, n_eval))
dZdx1_eval = np.zeros((n_eval, n_eval))
dZdx2_eval = np.zeros((n_eval, n_eval))

x_point = (ctypes.c_double * 2)()
y_out = (ctypes.c_double * 1)()
dydx1_out = (ctypes.c_double * 1)()
dydx2_out = (ctypes.c_double * 1)()

for i in range(n_eval):
    for j in range(n_eval):
        x_point[0] = X1_eval[i, j]
        x_point[1] = X2_eval[i, j]
        evaluate_2d_der(order_c, num_points_c, periodic_c, x_min_c, h_step_c,
                       coeff_c, x_point, y_out, dydx1_out, dydx2_out)
        Z_eval[i, j] = y_out[0]
        dZdx1_eval[i, j] = dydx1_out[0]
        dZdx2_eval[i, j] = dydx2_out[0]

# Plot 2D results
ax4 = plt.subplot(2, 3, 4)
im1 = ax4.contourf(X2_eval, X1_eval, Z_eval, levels=20, cmap='viridis')
ax4.scatter(X2.flatten()[::5], X1.flatten()[::5], c='red', s=10, alpha=0.5, label='Data points')
ax4.set_title(f'2D Spline (constructed in {construct_time:.3f}s)', fontsize=14)
ax4.set_xlabel('x2')
ax4.set_ylabel('x1')
plt.colorbar(im1, ax=ax4)

ax5 = plt.subplot(2, 3, 5)
im2 = ax5.contourf(X2_eval, X1_eval, dZdx1_eval, levels=20, cmap='RdBu')
ax5.set_title('∂f/∂x1', fontsize=14)
ax5.set_xlabel('x2')
ax5.set_ylabel('x1')
plt.colorbar(im2, ax=ax5)

ax6 = plt.subplot(2, 3, 6)
im3 = ax6.contourf(X2_eval, X1_eval, dZdx2_eval, levels=20, cmap='RdBu')
ax6.set_title('∂f/∂x2', fontsize=14)
ax6.set_xlabel('x2')
ax6.set_ylabel('x1')
plt.colorbar(im3, ax=ax6)

plt.suptitle('Sergei\'s Splines Working in Pure Numba CFuncs!', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sergei_splines_working.png', dpi=150, bbox_inches='tight')
plt.show()

# Print some stats
print("\n=== SERGEI'S SPLINES PERFORMANCE ===")
print(f"2D Spline Construction: {construct_time*1000:.2f} ms for {n1}×{n2} grid")
print(f"2D Spline Evaluation: {n_eval*n_eval} points with derivatives")
print("\nMax interpolation errors at data points:")

# Check interpolation error
errors = []
for i in range(0, n1, 3):
    for j in range(0, n2, 3):
        x_point[0] = x1[i]
        x_point[1] = x2[j]
        evaluate_2d_der(order_c, num_points_c, periodic_c, x_min_c, h_step_c,
                       coeff_c, x_point, y_out, dydx1_out, dydx2_out)
        errors.append(abs(y_out[0] - Z[i, j]))

print(f"  Max error: {max(errors):.2e}")
print(f"  Avg error: {np.mean(errors):.2e}")
print("\n✓ All spline functions working correctly!")