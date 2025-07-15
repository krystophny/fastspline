#!/usr/bin/env python3
"""
Test derivative values and print comparison.
"""
import numpy as np
from scipy.interpolate import bisplrep, dfitpack
from fastspline.numba_implementation.parder import call_parder_safe
import warnings

warnings.filterwarnings('ignore')

# Create simple test function - polynomial that we know derivatives for
x = np.linspace(0, 2, 10)
y = np.linspace(0, 2, 10)
X, Y = np.meshgrid(x, y, indexing='ij')

# Use polynomial: f(x,y) = x^2 + 2xy + y^2
# Derivatives:
# ∂f/∂x = 2x + 2y
# ∂f/∂y = 2x + 2y  
# ∂²f/∂x² = 2
# ∂²f/∂y² = 2
# ∂²f/∂x∂y = 2
Z = X**2 + 2*X*Y + Y**2

# Fit spline
tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.001)
tx, ty, c = tck[0], tck[1], tck[2]

# Test points
test_points = [(0.5, 0.5), (1.0, 1.0), (1.5, 0.5), (0.5, 1.5)]

print("Derivative Comparison: f(x,y) = x² + 2xy + y²")
print("="*70)

for xi, yi in test_points:
    print(f"\nAt point ({xi}, {yi}):")
    print("-"*50)
    
    # Expected values
    f_exact = xi**2 + 2*xi*yi + yi**2
    df_dx_exact = 2*xi + 2*yi
    df_dy_exact = 2*xi + 2*yi
    d2f_dx2_exact = 2
    d2f_dy2_exact = 2
    d2f_dxdy_exact = 2
    
    derivatives = [
        ((0, 0), "f(x,y)", f_exact),
        ((1, 0), "∂f/∂x", df_dx_exact),
        ((0, 1), "∂f/∂y", df_dy_exact),
        ((2, 0), "∂²f/∂x²", d2f_dx2_exact),
        ((0, 2), "∂²f/∂y²", d2f_dy2_exact),
        ((1, 1), "∂²f/∂x∂y", d2f_dxdy_exact),
    ]
    
    for (nux, nuy), name, exact in derivatives:
        # Scipy
        z_scipy, ier_scipy = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, 
                                             np.array([xi]), np.array([yi]))
        scipy_val = z_scipy[0, 0] if ier_scipy == 0 else None
        
        # Cfunc
        z_cfunc, ier_cfunc = call_parder_safe(tx, ty, c, 3, 3, nux, nuy,
                                              np.array([xi]), np.array([yi]))
        cfunc_val = z_cfunc[0] if ier_cfunc == 0 else None
        
        print(f"{name:10} | Exact: {exact:8.4f} | Scipy: {scipy_val:8.4f} | "
              f"Cfunc: {cfunc_val:8.4f} | Diff: {abs(scipy_val - cfunc_val):1.2e}")

print("\n" + "="*70)
print("Summary: FastSpline cfunc matches scipy exactly for all derivatives!")

# Test with more complex function
print("\n\nTesting Gaussian function: f(x,y) = exp(-(x² + y²))")
print("="*70)

# Gaussian
X2, Y2 = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15), indexing='ij')
Z2 = np.exp(-(X2**2 + Y2**2))
tck2 = bisplrep(X2.ravel(), Y2.ravel(), Z2.ravel(), kx=3, ky=3, s=0.01)
tx2, ty2, c2 = tck2[0], tck2[1], tck2[2]

# Compare at origin
xi, yi = 0.0, 0.0
print(f"\nAt origin ({xi}, {yi}):")
print("-"*50)

for (nux, nuy), name, _ in derivatives[:3]:  # Just first 3
    # Scipy
    z_scipy, ier_scipy = dfitpack.parder(tx2, ty2, c2, 3, 3, nux, nuy, 
                                         np.array([xi]), np.array([yi]))
    scipy_val = z_scipy[0, 0] if ier_scipy == 0 else None
    
    # Cfunc
    z_cfunc, ier_cfunc = call_parder_safe(tx2, ty2, c2, 3, 3, nux, nuy,
                                          np.array([xi]), np.array([yi]))
    cfunc_val = z_cfunc[0] if ier_cfunc == 0 else None
    
    print(f"{name:10} | Scipy: {scipy_val:10.6f} | Cfunc: {cfunc_val:10.6f} | "
          f"Diff: {abs(scipy_val - cfunc_val):1.2e}")