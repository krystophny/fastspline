#!/usr/bin/env python3
"""
Test to isolate the derivative mismatch issue
"""
import numpy as np
import warnings
from scipy.interpolate import bisplrep, bisplev, dfitpack

print('=== INVESTIGATING DERIVATIVE MISMATCH ===')

# Create test data with exact known derivatives
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y, indexing='ij')

# Test with polynomial function where derivatives are exact
# f(x,y) = x^2 + y^2
# df/dx = 2x, df/dy = 2y
# d²f/dx² = 2, d²f/dy² = 2, d²f/dxdy = 0
Z = X**2 + Y**2

print('Testing with polynomial f(x,y) = x² + y²')
print('Expected derivatives:')
print('  f(0.5, 0.5) = 0.5')
print('  ∂f/∂x(0.5, 0.5) = 1.0')
print('  ∂f/∂y(0.5, 0.5) = 1.0')
print('  ∂²f/∂x²(0.5, 0.5) = 2.0')
print('  ∂²f/∂y²(0.5, 0.5) = 2.0')
print('  ∂²f/∂x∂y(0.5, 0.5) = 0.0')

# Test with different smoothing values
smoothing_values = [0.0, 0.001, 0.01, 0.1]

for s in smoothing_values:
    print(f'\\n--- Testing with smoothing s = {s} ---')
    
    try:
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=s)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Test point
        xi = np.array([0.5])
        yi = np.array([0.5])
        
        # Test derivatives
        derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
        expected = [0.5, 1.0, 1.0, 2.0, 2.0, 0.0]
        
        print('dfitpack.parder results:')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            
            for i, (nux, nuy) in enumerate(derivatives):
                z_deriv, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                error = abs(z_deriv[0,0] - expected[i])
                print(f'  ∂^{nux+nuy}f/∂x^{nux}∂y^{nuy}: {z_deriv[0,0]:.6f} (expected: {expected[i]:.6f}, error: {error:.6f})')
                
                if ier != 0:
                    print(f'    WARNING: ier = {ier}')
        
        # Test consistency with bisplev for function value
        z_bisplev = bisplev(xi, yi, tck)
        z_parder, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 0, xi, yi)
        
        bisplev_parder_diff = abs(z_bisplev - z_parder[0,0])
        print(f'\\nConsistency check:')
        print(f'  bisplev: {z_bisplev:.6f}')
        print(f'  parder(0,0): {z_parder[0,0]:.6f}')
        print(f'  difference: {bisplev_parder_diff:.6f}')
        
        if bisplev_parder_diff > 1e-12:
            print('  ✗ INCONSISTENCY between bisplev and parder!')
        else:
            print('  ✓ bisplev and parder consistent')
        
    except Exception as e:
        print(f'  Error with s={s}: {e}')

print('\\n=== TESTING MULTIPLE POINTS ===')

# Test with s=0.01 at multiple points
s = 0.01
tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=s)
tx, ty, c = tck[0], tck[1], tck[2]

test_points = [(0.1, 0.1), (0.3, 0.7), (0.8, 0.2), (0.9, 0.9)]

for xi_val, yi_val in test_points:
    xi = np.array([xi_val])
    yi = np.array([yi_val])
    
    print(f'\\nAt point ({xi_val}, {yi_val}):')
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        
        # Test first derivatives
        z_dx, ier = dfitpack.parder(tx, ty, c, 3, 3, 1, 0, xi, yi)
        z_dy, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 1, xi, yi)
        
        expected_dx = 2 * xi_val
        expected_dy = 2 * yi_val
        
        error_dx = abs(z_dx[0,0] - expected_dx)
        error_dy = abs(z_dy[0,0] - expected_dy)
        
        print(f'  ∂f/∂x: {z_dx[0,0]:.6f} (expected: {expected_dx:.6f}, error: {error_dx:.6f})')
        print(f'  ∂f/∂y: {z_dy[0,0]:.6f} (expected: {expected_dy:.6f}, error: {error_dy:.6f})')
        
        if error_dx > 1e-3 or error_dy > 1e-3:
            print('  ✗ Large derivative errors detected!')
        else:
            print('  ✓ Derivatives within tolerance')

print('\\n=== SUMMARY ===')
print('If derivatives are consistently off, the issue is likely:')
print('1. Smoothing parameter affecting accuracy')
print('2. Spline degree or knot placement')
print('3. Implementation bug in cfunc vs dfitpack')
print('4. Coordinate system or indexing differences')