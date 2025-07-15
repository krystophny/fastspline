#!/usr/bin/env python3
"""
Test the corrected parder implementation for function evaluation (0,0).
"""
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from fastspline.numba_implementation.parder import call_parder_safe

def test_simple_parder():
    """Test the corrected parder for function evaluation (0,0)"""
    print("=== Testing Corrected Parder for Function Evaluation ===")
    
    # Create simple constant function
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.ones_like(X)  # Constant = 1
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Scipy reference
    z_scipy = bisplev(xi, yi, tck)
    print(f"Scipy bisplev: {z_scipy:.6f}")
    
    # Our corrected parder
    try:
        c_arr = np.asarray(c, dtype=np.float64)
        z_parder, ier_parder = call_parder_safe(tx, ty, c_arr, 3, 3, 0, 0, xi, yi)
        print(f"Corrected parder: {z_parder[0]:.6f}, ier={ier_parder}")
        
        if abs(z_scipy - z_parder[0]) < 1e-10:
            print("✅ SUCCESS: Results match!")
            return True
        else:
            print(f"❌ MISMATCH: diff = {abs(z_scipy - z_parder[0]):.2e}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    test_simple_parder()