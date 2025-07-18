#!/usr/bin/env python3
"""
Implement proper periodic spline algorithms based on Fortran reference
"""

import numpy as np
from numba import njit

def solve_cyclic_tridiagonal(a, b, c, d):
    """
    Solve cyclic tridiagonal system Ax = d where:
    - b is the main diagonal
    - a is the lower diagonal (with a[0] connecting to last row)
    - c is the upper diagonal (with c[n-1] connecting to first row)
    - d is the right-hand side
    
    This is needed for periodic splines.
    """
    n = len(b)
    if n < 3:
        raise ValueError("System too small for cyclic tridiagonal")
    
    # Allocate working arrays
    gamma = np.zeros(n)
    alpha = np.zeros(n-1)
    beta = np.zeros(n-1)
    x = np.zeros(n)
    z = np.zeros(n)
    
    # Factor the matrix
    gamma[0] = -b[0]
    beta[0] = 2.0
    alpha[0] = c[0] / gamma[0]
    
    for i in range(1, n-1):
        gamma[i] = b[i] - a[i] * alpha[i-1]
        beta[i] = -a[i] * beta[i-1] / gamma[i]
        alpha[i] = c[i] / gamma[i]
    
    # Forward substitution
    z[0] = d[0] / gamma[0]
    for i in range(1, n-1):
        z[i] = (d[i] - a[i] * z[i-1]) / gamma[i]
    
    # Compute final gamma
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n-1):
        sum1 += alpha[i] * (beta[i] if i < n-2 else 1.0)
        sum2 += alpha[i] * z[i]
    
    gamma[n-1] = b[n-1] - a[n-1] * alpha[n-2] - c[n-1] * sum1
    
    # Final z value
    z[n-1] = (d[n-1] - a[n-1] * z[n-2] - c[n-1] * sum2) / gamma[n-1]
    
    # Back substitution
    x[n-1] = z[n-1]
    x[n-2] = z[n-2] - alpha[n-2] * x[n-1]
    
    for i in range(n-3, -1, -1):
        x[i] = z[i] - alpha[i] * x[i+1] - beta[i] * x[n-1]
    
    return x

@njit
def cubic_periodic_coefficients(y, n, h):
    """
    Compute cubic spline coefficients for periodic boundary conditions.
    Based on the cyclic tridiagonal system for periodic splines.
    """
    # Allocate coefficient arrays
    a = y.copy()  # a coefficients are the y values
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    
    # Set up the cyclic tridiagonal system for c coefficients
    # The system is: A * c = rhs
    # where A is cyclic tridiagonal with:
    # - main diagonal: 4
    # - upper/lower diagonals: 1
    # - corner elements: 1 (for periodicity)
    
    # Right-hand side
    rhs = np.zeros(n)
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        rhs[i] = 3.0 * (a[ip1] - a[im1]) / h
    
    # Solve for c coefficients using Thomas algorithm for cyclic systems
    # Simplified for the special case where all diagonals have same values
    diag = 4.0
    offdiag = 1.0
    
    # Forward elimination
    alpha = np.zeros(n)
    beta = np.zeros(n)
    
    alpha[0] = offdiag / diag
    beta[0] = rhs[0] / diag
    
    for i in range(1, n-1):
        denom = diag - offdiag * alpha[i-1]
        alpha[i] = offdiag / denom
        beta[i] = (rhs[i] - offdiag * beta[i-1]) / denom
    
    # Handle the cyclic part
    gamma = offdiag
    delta = offdiag
    
    u = np.zeros(n)
    v = np.zeros(n)
    u[0] = gamma
    u[n-1] = delta
    v[0] = 1.0
    v[n-1] = 1.0
    
    # Solve the modified system
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    
    # First solution
    y1[0] = beta[0]
    for i in range(1, n):
        if i < n-1:
            y1[i] = beta[i] - alpha[i-1] * y1[i-1]
        else:
            y1[i] = (rhs[i] - offdiag * y1[i-1] - offdiag * y1[0]) / (diag - offdiag * alpha[i-1] - offdiag * v[0])
    
    # Second solution
    y2[0] = v[0] / diag
    for i in range(1, n):
        if i < n-1:
            y2[i] = -alpha[i-1] * y2[i-1]
        else:
            y2[i] = (v[i] - offdiag * y2[i-1] - offdiag * y2[0]) / (diag - offdiag * alpha[i-1] - offdiag * v[0])
    
    # Combine solutions
    factor = (u[0] * y1[0] + u[n-1] * y1[n-1]) / (1.0 + u[0] * y2[0] + u[n-1] * y2[n-1])
    
    for i in range(n):
        c[i] = y1[i] - factor * y2[i]
    
    # Compute b and d coefficients
    for i in range(n):
        ip1 = (i + 1) % n
        b[i] = (a[ip1] - a[i]) / h - h * (c[ip1] + 2.0 * c[i]) / 3.0
        d[i] = (c[ip1] - c[i]) / (3.0 * h)
    
    return a, b, c, d

def test_periodic_cubic_fix():
    """Test the fixed periodic cubic spline"""
    print("Testing Fixed Periodic Cubic Spline")
    print("=" * 50)
    
    # Test function
    n = 16
    x = np.linspace(0, 2*np.pi, n, endpoint=False)
    y = np.sin(x)
    h = 2*np.pi / n
    
    print(f"Test function: sin(x) on [0, 2π)")
    print(f"Grid: n = {n} points")
    print(f"Step size: h = {h:.6f}")
    
    # Compute coefficients
    a, b, c, d = cubic_periodic_coefficients(y, n, h)
    
    # Test continuity at boundaries
    print("\nBoundary Continuity Tests:")
    print("-" * 40)
    
    # Function value at x=0 and x=2π should match
    f_0 = a[0]
    f_2pi_left = a[n-1] + b[n-1] + c[n-1] + d[n-1]
    print(f"f(0) = {f_0:.10f}")
    print(f"f(2π-) = {f_2pi_left:.10f}")
    print(f"Difference: {abs(f_0 - f_2pi_left):.2e}")
    
    # First derivative continuity
    df_0 = b[0]
    df_2pi_left = b[n-1] + 2*c[n-1] + 3*d[n-1]
    print(f"\nf'(0) = {df_0:.10f}")
    print(f"f'(2π-) = {df_2pi_left:.10f}")
    print(f"Difference: {abs(df_0 - df_2pi_left):.2e}")
    
    # Second derivative continuity
    d2f_0 = 2*c[0]
    d2f_2pi_left = 2*c[n-1] + 6*d[n-1]
    print(f"\nf''(0) = {d2f_0:.10f}")
    print(f"f''(2π-) = {d2f_2pi_left:.10f}")
    print(f"Difference: {abs(d2f_0 - d2f_2pi_left):.2e}")
    
    # Test accuracy at various points
    print("\n\nAccuracy Tests:")
    print("-" * 40)
    
    test_x = np.array([0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi - 0.001])
    
    for x_test in test_x:
        # Find interval
        i = int(x_test / h) % n
        t = (x_test - i*h) / h
        
        # Evaluate spline
        y_spline = a[i] + t*(b[i] + t*(c[i] + t*d[i]))
        y_exact = np.sin(x_test)
        error = abs(y_spline - y_exact)
        
        print(f"x = {x_test:.6f}: exact = {y_exact:.6f}, "
              f"spline = {y_spline:.6f}, error = {error:.2e}")
    
    print("\n" + "="*50)
    
    # Check if all continuity conditions are satisfied
    cont_f = abs(f_0 - f_2pi_left) < 1e-10
    cont_df = abs(df_0 - df_2pi_left) < 1e-10
    cont_d2f = abs(d2f_0 - d2f_2pi_left) < 1e-10
    
    if cont_f and cont_df and cont_d2f:
        print("✅ SUCCESS: Periodic cubic spline has perfect continuity!")
        return True
    else:
        print("❌ FAILURE: Continuity conditions not satisfied")
        return False

def generate_fixed_periodic_algorithm():
    """Generate the corrected algorithm for periodic splines"""
    print("\n\nCorrected Periodic Spline Algorithm")
    print("=" * 50)
    
    algorithm = '''
# Periodic Cubic Spline Algorithm (Corrected)

1. Input: y[0..n-1] (periodic data), h (step size)

2. Set up cyclic tridiagonal system for second derivatives (c):
   [4  1  0 ... 0  1] [c[0]  ]   [3(y[1] - y[n-1])/h]
   [1  4  1 ... 0  0] [c[1]  ]   [3(y[2] - y[0])/h  ]
   [0  1  4 ... 0  0] [c[2]  ] = [3(y[3] - y[1])/h  ]
   [............... ] [...    ]   [...                ]
   [0  0  0 ... 4  1] [c[n-2]]   [3(y[n-1] - y[n-3])/h]
   [1  0  0 ... 1  4] [c[n-1]]   [3(y[0] - y[n-2])/h]

3. Solve using cyclic Thomas algorithm

4. Compute other coefficients:
   a[i] = y[i]
   b[i] = (y[i+1] - y[i])/h - h*(c[i+1] + 2*c[i])/3
   d[i] = (c[i+1] - c[i])/(3*h)
   
   where indices wrap: i+1 = (i+1) % n

5. Evaluation at x in [0, L) where L = n*h:
   - Find interval: i = floor(x/h) % n
   - Local coordinate: t = (x - i*h)/h
   - Value: f(x) = a[i] + t*(b[i] + t*(c[i] + t*d[i]))
'''
    
    print(algorithm)
    
    return algorithm

if __name__ == "__main__":
    # Test the fixed algorithm
    success = test_periodic_cubic_fix()
    
    if success:
        # Generate the algorithm description
        algorithm = generate_fixed_periodic_algorithm()
        
        print("\nThis algorithm ensures C² continuity at all points,")
        print("including the periodic boundary.")
        
        # Additional test with different function
        print("\n\nAdditional Test: cos(x) + 0.5*sin(2x)")
        print("=" * 50)
        
        n = 20
        x = np.linspace(0, 2*np.pi, n, endpoint=False)
        y = np.cos(x) + 0.5*np.sin(2*x)
        h = 2*np.pi / n
        
        a, b, c, d = cubic_periodic_coefficients(y, n, h)
        
        # Test a few points
        test_x = [0.5, 1.5, 3.0, 5.0, 6.0]
        max_error = 0.0
        
        for xt in test_x:
            i = int(xt / h) % n
            t = (xt - i*h) / h
            y_spline = a[i] + t*(b[i] + t*(c[i] + t*d[i]))
            y_exact = np.cos(xt) + 0.5*np.sin(2*xt)
            error = abs(y_spline - y_exact)
            max_error = max(max_error, error)
            print(f"x = {xt:.3f}: error = {error:.2e}")
        
        print(f"\nMax error: {max_error:.2e}")