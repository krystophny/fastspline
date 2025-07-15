#!/usr/bin/env python3
"""
Simple derivative comparison plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep, dfitpack
from fastspline.numba_implementation.parder import call_parder_safe
import warnings

warnings.filterwarnings('ignore')

# Create simple test function
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.exp(-(X**2 + Y**2))  # Gaussian

# Fit spline
tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
tx, ty, c = tck[0], tck[1], tck[2]

# Evaluation grid
x_eval = np.linspace(-1.5, 1.5, 30)
y_eval = np.linspace(-1.5, 1.5, 30)

# Test derivative (1,0) - ∂/∂x
print("Computing derivatives...")
z_scipy = np.zeros((len(x_eval), len(y_eval)))
z_cfunc = np.zeros((len(x_eval), len(y_eval)))

for i in range(len(x_eval)):
    for j in range(len(y_eval)):
        # Scipy
        z_s, ier_s = dfitpack.parder(tx, ty, c, 3, 3, 1, 0, 
                                     np.array([x_eval[i]]), 
                                     np.array([y_eval[j]]))
        if ier_s == 0:
            z_scipy[i, j] = z_s[0, 0]
        
        # Cfunc
        z_c, ier_c = call_parder_safe(tx, ty, c, 3, 3, 1, 0,
                                      np.array([x_eval[i]]), 
                                      np.array([y_eval[j]]))
        if ier_c == 0:
            z_cfunc[i, j] = z_c[0]

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Scipy result
im1 = ax1.contourf(x_eval, y_eval, z_scipy.T, levels=20, cmap='viridis')
ax1.set_title('Scipy ∂f/∂x')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1)

# Cfunc result
im2 = ax2.contourf(x_eval, y_eval, z_cfunc.T, levels=20, cmap='viridis')
ax2.set_title('FastSpline cfunc ∂f/∂x')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(im2, ax=ax2)

# Difference
diff = z_scipy - z_cfunc
max_diff = np.max(np.abs(diff))
im3 = ax3.contourf(x_eval, y_eval, diff.T, levels=20, cmap='RdBu_r')
ax3.set_title(f'Difference\nMax: {max_diff:.2e}')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
plt.colorbar(im3, ax=ax3)

plt.suptitle('Gaussian Function: First Derivative Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('simple_derivative_comparison.png', dpi=150)
print(f"Saved simple_derivative_comparison.png")
print(f"Maximum difference: {max_diff:.10f}")

# Create a multi-derivative plot
fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

derivatives = [(0,0), (1,0), (0,1), (2,0), (0,2), (1,1)]
titles = ['f(x,y)', '∂f/∂x', '∂f/∂y', '∂²f/∂x²', '∂²f/∂y²', '∂²f/∂x∂y']

for idx, ((nux, nuy), title) in enumerate(zip(derivatives, titles)):
    ax = axes.ravel()[idx]
    
    # Compute with cfunc
    z_deriv = np.zeros((len(x_eval), len(y_eval)))
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            z_val, ier = call_parder_safe(tx, ty, c, 3, 3, nux, nuy,
                                         np.array([x_eval[i]]), 
                                         np.array([y_eval[j]]))
            if ier == 0:
                z_deriv[i, j] = z_val[0]
    
    im = ax.contourf(x_eval, y_eval, z_deriv.T, levels=20, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax)

plt.suptitle('Gaussian Function: All Derivatives (FastSpline cfunc)', fontsize=14)
plt.tight_layout()
plt.savefig('all_derivatives_gaussian.png', dpi=150)
print(f"Saved all_derivatives_gaussian.png")

plt.show()