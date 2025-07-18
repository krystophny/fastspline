#!/usr/bin/env python3
"""
Test quintic periodic spline implementation
"""

import numpy as np
import sys
sys.path.insert(0, '../../src')

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_quintic_periodic():
    """Test quintic periodic spline on sin(2πx)"""
    print("QUINTIC PERIODIC SPLINE TEST")
    print("=" * 50)
    
    # Test parameters
    n = 16
    x_min = 0.0
    x_max = 1.0
    
    # Create test data: sin(2πx) on [0, 1)
    x = np.linspace(x_min, x_max, n, endpoint=False)
    y = np.sin(2 * np.pi * x)
    
    print(f"n = {n}")
    print(f"Domain: [{x_min}, {x_max})")
    print(f"Test function: sin(2πx)")
    print(f"x: {x}")
    print(f"y: {y}")
    print()
    
    # Construct quintic periodic spline
    coeff = np.zeros(6 * n, dtype=np.float64)  # 6 coefficients per point
    
    try:
        construct_splines_1d_cfunc(x_min, x_max, y, n, 5, True, coeff)
        print("✅ Quintic periodic spline constructed successfully!")
        
        # Test evaluation at a few points
        test_points = [0.1, 0.25, 0.5, 0.75, 0.9]
        h_step = (x_max - x_min) / n
        
        print(f"h_step = {h_step}")
        print()
        print("Evaluation test:")
        for x_test in test_points:
            y_exact = np.sin(2 * np.pi * x_test)
            y_out = np.array([0.0])
            evaluate_splines_1d_cfunc(5, n, True, x_min, h_step, coeff, x_test, y_out)
            y_spline = y_out[0]
            error = abs(y_spline - y_exact)
            print(f"  x={x_test:.2f}: spline={y_spline:.12f}, exact={y_exact:.12f}, error={error:.2e}")
        
        # Test periodicity
        print()
        print("Periodicity test:")
        x_test = 0.3
        y_exact = np.sin(2 * np.pi * x_test)
        
        # Evaluate at x_test and x_test + period
        y_out = np.array([0.0])
        evaluate_splines_1d_cfunc(5, n, True, x_min, h_step, coeff, x_test, y_out)
        y_base = y_out[0]
        
        evaluate_splines_1d_cfunc(5, n, True, x_min, h_step, coeff, x_test + 1.0, y_out)
        y_period = y_out[0]
        
        period_error = abs(y_period - y_base)
        print(f"  x={x_test:.2f}: spline={y_base:.12f}")
        print(f"  x={x_test + 1.0:.2f}: spline={y_period:.12f}")
        print(f"  Periodicity error: {period_error:.2e}")
        
        if period_error < 1e-14:
            print("✅ Perfect periodicity achieved!")
        else:
            print("❌ Periodicity issue")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quintic_periodic()