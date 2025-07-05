"""Simple accuracy test for DIERCKX bisplrep."""

import numpy as np
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline import bisplrep, bisplev


def test_exact_interpolation():
    """Test exact interpolation on a simple grid."""
    # Create simple test data
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    z = 1 + 2*xx + 3*yy + xx*yy  # Simple polynomial
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    # Fit with exact interpolation (s=0)
    print("Fitting with s=0 (exact interpolation)...")
    tck_fast = bisplrep(x_flat, y_flat, z_flat, s=0)
    tck_scipy = interpolate.bisplrep(x_flat, y_flat, z_flat, s=0)
    
    # Evaluate at test points
    x_test = np.linspace(0.1, 0.9, 10)
    y_test = np.linspace(0.1, 0.9, 10)
    
    z_fast = bisplev(x_test, y_test, tck_fast)
    z_scipy = interpolate.bisplev(x_test, y_test, tck_scipy)
    
    # True values
    xx_test, yy_test = np.meshgrid(x_test, y_test)
    z_true = 1 + 2*xx_test + 3*yy_test + xx_test*yy_test
    
    # Compute errors
    error_fast = np.abs(z_fast - z_true).max()
    error_scipy = np.abs(z_scipy - z_true).max()
    
    print(f"\nMaximum errors (s=0):")
    print(f"  FastSpline: {error_fast:.2e}")
    print(f"  SciPy:      {error_scipy:.2e}")
    print(f"  Difference: {np.abs(z_fast - z_scipy).max():.2e}")
    
    # Now test with smoothing
    print("\n\nFitting with s=0.01 (smoothing)...")
    tck_fast_smooth = bisplrep(x_flat, y_flat, z_flat, s=0.01)
    tck_scipy_smooth = interpolate.bisplrep(x_flat, y_flat, z_flat, s=0.01)
    
    # Print knot info
    print(f"\nKnot counts:")
    print(f"  FastSpline:  tx={len(tck_fast_smooth[0])}, ty={len(tck_fast_smooth[1])}")
    print(f"  SciPy:       tx={len(tck_scipy_smooth[0])}, ty={len(tck_scipy_smooth[1])}")
    
    z_fast_smooth = bisplev(x_test, y_test, tck_fast_smooth)
    z_scipy_smooth = interpolate.bisplev(x_test, y_test, tck_scipy_smooth)
    
    error_fast_smooth = np.abs(z_fast_smooth - z_true).max()
    error_scipy_smooth = np.abs(z_scipy_smooth - z_true).max()
    
    print(f"\nMaximum errors (s=0.01):")
    print(f"  FastSpline: {error_fast_smooth:.2e}")
    print(f"  SciPy:      {error_scipy_smooth:.2e}")
    print(f"  Difference: {np.abs(z_fast_smooth - z_scipy_smooth).max():.2e}")


if __name__ == "__main__":
    test_exact_interpolation()