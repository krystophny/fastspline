"""Test bisplev accuracy."""

import numpy as np
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline import bisplrep, bisplev
from fastspline.bisplev_fast import bisplev_fast


def test_simple_evaluation():
    """Test on simple polynomial."""
    # Simple 3x3 grid
    x = np.array([0, 0.5, 1])
    y = np.array([0, 0.5, 1])
    xx, yy = np.meshgrid(x, y)
    z = 1 + 2*xx + 3*yy  # Linear function
    
    # Fit
    tck = bisplrep(xx.ravel(), yy.ravel(), z.ravel(), kx=1, ky=1, s=0)
    tck_scipy = interpolate.bisplrep(xx.ravel(), yy.ravel(), z.ravel(), kx=1, ky=1, s=0)
    
    print("Knot vectors:")
    print(f"  tx: {tck[0]}")
    print(f"  ty: {tck[1]}")
    print(f"  Coefficients: {tck[2]}")
    
    # Test single point
    x_test = 0.25
    y_test = 0.75
    
    z_true = 1 + 2*x_test + 3*y_test
    z_scipy = interpolate.bisplev(x_test, y_test, tck_scipy)
    z_ours = bisplev(x_test, y_test, tck)
    z_fast = bisplev_fast(x_test, y_test, tck)
    
    print(f"\nSingle point evaluation at ({x_test}, {y_test}):")
    print(f"  True value:  {z_true}")
    print(f"  SciPy:       {z_scipy}")
    print(f"  Ours:        {z_ours}")
    print(f"  Fast:        {z_fast}")
    
    # Test grid
    x_grid = np.array([0.2, 0.8])
    y_grid = np.array([0.3, 0.7])
    
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    z_true_grid = 1 + 2*xx_grid + 3*yy_grid
    
    z_scipy_grid = interpolate.bisplev(x_grid, y_grid, tck_scipy)
    z_ours_grid = bisplev(x_grid, y_grid, tck)
    z_fast_grid = bisplev_fast(x_grid, y_grid, tck)
    
    print(f"\nGrid evaluation:")
    print(f"  True values:\n{z_true_grid}")
    print(f"  SciPy:\n{z_scipy_grid}")
    print(f"  Ours:\n{z_ours_grid}")
    print(f"  Fast:\n{z_fast_grid}")
    print(f"  Diff (Ours-SciPy): {np.abs(z_ours_grid - z_scipy_grid).max():.2e}")
    print(f"  Diff (Fast-SciPy): {np.abs(z_fast_grid - z_scipy_grid).max():.2e}")


if __name__ == "__main__":
    test_simple_evaluation()