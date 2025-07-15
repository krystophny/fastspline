#!/usr/bin/env python3
"""
Side-by-side ASCII visualization of scipy vs cfunc derivatives.
"""
import numpy as np
from scipy.interpolate import bisplrep, dfitpack
from fastspline.numba_implementation.parder import call_parder_safe
import warnings

warnings.filterwarnings('ignore')

def create_side_by_side_ascii(data1, data2, title1, title2, width=25, height=15):
    """Create side-by-side ASCII heatmaps."""
    # Normalize data to 0-9 range
    vmin1, vmax1 = np.min(data1), np.max(data1)
    vmin2, vmax2 = np.min(data2), np.max(data2)
    
    # Use common scale for comparison
    vmin = min(vmin1, vmin2)
    vmax = max(vmax1, vmax2)
    
    if vmax - vmin > 1e-10:
        normalized1 = 9 * (data1 - vmin) / (vmax - vmin)
        normalized2 = 9 * (data2 - vmin) / (vmax - vmin)
    else:
        normalized1 = np.zeros_like(data1)
        normalized2 = np.zeros_like(data2)
    
    # Resample to target size
    h, w = data1.shape
    y_idx = np.linspace(0, h-1, height).astype(int)
    x_idx = np.linspace(0, w-1, width).astype(int)
    
    # Compute difference
    diff = data1 - data2
    max_diff = np.max(np.abs(diff))
    
    chars = " .:-=+*#%@"
    
    # Print header
    print("\n" + "="*80)
    print(f"{title1:^38} | {title2:^38}")
    print(f"Range: [{vmin:.3f}, {vmax:.3f}]" + " "*17 + f"| Range: [{vmin:.3f}, {vmax:.3f}]")
    print("-"*39 + "|" + "-"*40)
    
    # Print side by side
    for i in y_idx:
        line1 = ""
        line2 = ""
        for j in x_idx:
            val1 = int(normalized1[i, j])
            val2 = int(normalized2[i, j])
            line1 += chars[val1]
            line2 += chars[val2]
        print(f" {line1} | {line2}")
    
    print("-"*39 + "|" + "-"*40)
    print(f"Max absolute difference: {max_diff:.2e}")
    print("="*80)

# Create test function - Gaussian
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.exp(-(X**2 + Y**2))

# Fit spline
tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
tx, ty, c = tck[0], tck[1], tck[2]

# Evaluation grid
x_eval = np.linspace(-1.5, 1.5, 30)
y_eval = np.linspace(-1.5, 1.5, 30)

print("FastSpline vs Scipy Derivative Comparison")
print("========================================")
print("Function: f(x,y) = exp(-(x² + y²))")

# Test different derivatives
derivatives = [
    ((0, 0), "f(x,y)"),
    ((1, 0), "∂f/∂x"),
    ((0, 1), "∂f/∂y"),
    ((2, 0), "∂²f/∂x²"),
    ((0, 2), "∂²f/∂y²"),
    ((1, 1), "∂²f/∂x∂y"),
]

for (nux, nuy), name in derivatives:
    # Compute with scipy
    z_scipy = np.zeros((len(x_eval), len(y_eval)))
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            z_val, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy,
                                        np.array([x_eval[i]]), 
                                        np.array([y_eval[j]]))
            if ier == 0:
                z_scipy[i, j] = z_val[0, 0]
    
    # Compute with cfunc
    z_cfunc = np.zeros((len(x_eval), len(y_eval)))
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            z_val, ier = call_parder_safe(tx, ty, c, 3, 3, nux, nuy,
                                         np.array([x_eval[i]]), 
                                         np.array([y_eval[j]]))
            if ier == 0:
                z_cfunc[i, j] = z_val[0]
    
    create_side_by_side_ascii(z_scipy, z_cfunc, 
                             f"Scipy {name}", f"FastSpline {name}")

# Cross-section comparison
print("\nCross-section at y=0 for ∂f/∂x:")
print("="*60)
print("   x     |  Scipy   | FastSpline |  Difference")
print("-"*60)

mid_y = len(y_eval) // 2
for i in range(0, len(x_eval), 3):  # Every 3rd point
    xi = x_eval[i]
    
    # Scipy
    z_s, _ = dfitpack.parder(tx, ty, c, 3, 3, 1, 0,
                            np.array([xi]), np.array([0.0]))
    
    # Cfunc
    z_c, _ = call_parder_safe(tx, ty, c, 3, 3, 1, 0,
                             np.array([xi]), np.array([0.0]))
    
    diff = z_s[0,0] - z_c[0]
    print(f"{xi:7.3f} | {z_s[0,0]:8.5f} | {z_c[0]:10.5f} | {diff:11.2e}")

print("-"*60)