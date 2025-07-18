#!/usr/bin/env python3
"""
Test if periodic continuity fix works
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastspline.sergei_splines import (
    construct_splines_1d_cfunc, evaluate_splines_1d_cfunc
)

def test_periodic_continuity():
    """Test periodic spline continuity after fix"""
    print("Testing Periodic Continuity Fix")
    print("=" * 50)
    
    # Setup
    n = 16
    x_min, x_max = 0.0, 1.0
    
    # Generate periodic data
    x = np.linspace(x_min, x_max, n, endpoint=False)
    y = np.sin(2 * np.pi * x)
    
    print(f"Test function: f(x) = sin(2πx)")
    print(f"Grid: n = {n} points (periodic)")
    print(f"Domain: [{x_min}, {x_max})")
    print()
    
    # Test cubic splines
    order = 3
    coeff = np.zeros((order+1) * n)
    construct_splines_1d_cfunc(x_min, x_max, y, n, order, 1, coeff)
    
    # The h_step for evaluation should match construction
    h_step = (x_max - x_min) / n  # For periodic
    
    print("Continuity Tests:")
    print("-" * 30)
    
    # Test at boundaries with small epsilon
    eps = 1e-10
    
    # Test points near boundaries
    test_pairs = [
        (0.0 - eps, 0.0 + eps, "x=0 boundary"),
        (1.0 - eps, 0.0 + eps, "x=0/1 wrap"),
        (0.5 - eps, 0.5 + eps, "interior point"),
        (0.999, 0.001, "near boundary"),
        (-0.001, 0.999, "negative wrap")
    ]
    
    max_discontinuity = 0.0
    
    for x1, x2, desc in test_pairs:
        y1 = np.zeros(1)
        y2 = np.zeros(1)
        
        evaluate_splines_1d_cfunc(order, n, 1, x_min, h_step, coeff, x1, y1)
        evaluate_splines_1d_cfunc(order, n, 1, x_min, h_step, coeff, x2, y2)
        
        diff = abs(y1[0] - y2[0])
        max_discontinuity = max(max_discontinuity, diff)
        
        print(f"  {desc:20s}: |f({x1:7.4f}) - f({x2:7.4f})| = {diff:.2e}")
    
    print(f"\nMax discontinuity: {max_discontinuity:.2e}")
    
    # Test exact periodicity
    print("\nPeriodicity Tests:")
    print("-" * 30)
    
    test_values = [0.0, 0.25, 0.5, 0.75]
    max_period_error = 0.0
    
    for x_base in test_values:
        y0 = np.zeros(1)
        y1 = np.zeros(1)
        y2 = np.zeros(1)
        
        evaluate_splines_1d_cfunc(order, n, 1, x_min, h_step, coeff, x_base, y0)
        evaluate_splines_1d_cfunc(order, n, 1, x_min, h_step, coeff, x_base + 1.0, y1)
        evaluate_splines_1d_cfunc(order, n, 1, x_min, h_step, coeff, x_base - 1.0, y2)
        
        diff1 = abs(y0[0] - y1[0])
        diff2 = abs(y0[0] - y2[0])
        max_period_error = max(max_period_error, diff1, diff2)
        
        print(f"  f({x_base:.2f}) = {y0[0]:.6f}")
        print(f"    vs f({x_base + 1.0:.2f}) = {y1[0]:.6f}, diff = {diff1:.2e}")
        print(f"    vs f({x_base - 1.0:.2f}) = {y2[0]:.6f}, diff = {diff2:.2e}")
        print()
    
    print(f"Max periodicity error: {max_period_error:.2e}")
    
    # Test against exact function
    print("\nAccuracy Tests:")
    print("-" * 30)
    
    test_points = np.linspace(0, 1, 21)
    max_error = 0.0
    
    for x_test in test_points:
        y_spline = np.zeros(1)
        evaluate_splines_1d_cfunc(order, n, 1, x_min, h_step, coeff, x_test, y_spline)
        
        y_exact = np.sin(2 * np.pi * x_test)
        error = abs(y_spline[0] - y_exact)
        max_error = max(max_error, error)
        
        if x_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print(f"  x = {x_test:.2f}: exact = {y_exact:.6f}, "
                  f"spline = {y_spline[0]:.6f}, error = {error:.2e}")
    
    print(f"\nMax approximation error: {max_error:.2e}")
    
    # Final verdict
    print("\n" + "="*50)
    if max_discontinuity < 1e-10 and max_period_error < 1e-10:
        print("✅ SUCCESS: Periodic continuity is now working correctly!")
        print("   - Boundary continuity: < 1e-10")
        print("   - Periodicity: < 1e-10")
        print(f"   - Approximation accuracy: {max_error:.2e}")
    else:
        print("❌ FAILURE: Periodic continuity issues remain")
        print(f"   - Max discontinuity: {max_discontinuity:.2e}")
        print(f"   - Max periodicity error: {max_period_error:.2e}")

if __name__ == "__main__":
    test_periodic_continuity()