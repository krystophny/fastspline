#!/usr/bin/env python3
"""
Test quintic periodic spline with simple polynomial
"""

import numpy as np
import sys
sys.path.insert(0, '../../src')

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

def test_quintic_periodic_polynomial():
    """Test quintic periodic spline on x^4 polynomial"""
    print("QUINTIC PERIODIC SPLINE - POLYNOMIAL TEST")
    print("=" * 50)
    
    # Test parameters
    n = 8
    x_min = 0.0
    x_max = 1.0
    
    # Create test data: x^4 on [0, 1) - but make periodic
    x = np.linspace(x_min, x_max, n, endpoint=False)
    y = x**4
    
    print(f"n = {n}")
    print(f"Domain: [{x_min}, {x_max})")
    print(f"Test function: x^4")
    print(f"x: {x}")
    print(f"y: {y}")
    print()
    
    # Construct quintic periodic spline
    coeff = np.zeros(6 * n, dtype=np.float64)  # 6 coefficients per point
    
    try:
        construct_splines_1d_cfunc(x_min, x_max, y, n, 5, True, coeff)
        print("✅ Quintic periodic spline constructed successfully!")
        
        # Test evaluation at a few points
        test_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        h_step = (x_max - x_min) / n
        
        print(f"h_step = {h_step}")
        print()
        print("Evaluation test:")
        max_error = 0.0
        for x_test in test_points:
            y_exact = x_test**4
            y_out = np.array([0.0])
            evaluate_splines_1d_cfunc(5, n, True, x_min, h_step, coeff, x_test, y_out)
            y_spline = y_out[0]
            error = abs(y_spline - y_exact)
            max_error = max(max_error, error)
            print(f"  x={x_test:.2f}: spline={y_spline:.12f}, exact={y_exact:.12f}, error={error:.2e}")
        
        print(f"\nMax error: {max_error:.2e}")
        
        if max_error < 1e-14:
            print("✅ Perfect polynomial precision achieved!")
        else:
            print("❌ Polynomial precision issue")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quintic_periodic_polynomial()