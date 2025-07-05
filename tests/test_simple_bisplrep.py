"""Simple test to debug bisplrep implementation."""

import numpy as np
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import bisplrep
from fastspline.wrappers import bisplev


def test_simple_grid():
    """Test on simple 2x2 grid."""
    # Create simple test data
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    z = np.array([0, 1, 2, 3])  # z = x + 2*y
    
    print("Input data:")
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    
    # Fit with linear spline
    tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx, ty, c, kx, ky = tck
    
    print(f"\nKnots:")
    print(f"tx: {tx}")
    print(f"ty: {ty}")
    print(f"coefficients: {c}")
    print(f"degrees: kx={kx}, ky={ky}")
    
    # Evaluate at grid points
    z_eval = bisplev(x, y, tck)
    print(f"\nEvaluated at grid points:")
    print(f"z_eval: {z_eval}")
    print(f"z_true: {z}")
    print(f"error: {np.abs(z_eval - z)}")
    
    # Test at intermediate point
    x_test = np.array([0.5])
    y_test = np.array([0.5])
    z_test = bisplev(x_test, y_test, tck)
    z_true = 0.5 + 2*0.5  # Should be 1.5
    
    print(f"\nAt (0.5, 0.5):")
    print(f"z_eval: {z_test}")
    print(f"z_true: {z_true}")
    print(f"error: {abs(z_test - z_true)}")
    
    # Compare with SciPy
    tck_scipy = interpolate.bisplrep(x, y, z, kx=1, ky=1, s=0)
    z_scipy = interpolate.bisplev(x_test, y_test, tck_scipy)
    print(f"\nSciPy result at (0.5, 0.5): {z_scipy}")


def test_regular_grid():
    """Test on regular grid."""
    # Create regular grid
    x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1])
    y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
    z = x + 2*y  # Linear function
    
    print("\n\nRegular grid test:")
    print(f"Points: {len(x)}")
    
    # Fit
    tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx, ty, c, kx, ky = tck
    
    print(f"Knots: nx={len(tx)}, ny={len(ty)}")
    print(f"Coefficients: {len(c)}")
    
    # Evaluate
    z_eval = bisplev(x, y, tck)
    error = np.abs(z_eval - z).max()
    print(f"Max error at grid points: {error:.2e}")
    
    # Compare with SciPy
    tck_scipy = interpolate.bisplrep(x, y, z, kx=1, ky=1, s=0)
    z_scipy = interpolate.bisplev(x, y, tck_scipy)
    error_scipy = np.abs(z_scipy - z).max()
    print(f"SciPy max error: {error_scipy:.2e}")


if __name__ == "__main__":
    test_simple_grid()
    test_regular_grid()