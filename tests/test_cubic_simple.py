"""Test cubic spline fitting."""

import numpy as np
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import bisplrep
from fastspline.wrappers import bisplev


def test_cubic_on_grid():
    """Test cubic spline on regular grid."""
    # Create a 5x5 grid
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    
    # Simple quadratic surface
    z = xx**2 + yy**2
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    print(f"Fitting {len(x_flat)} points")
    
    # Fit with our implementation
    tck = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    tx, ty, c, kx, ky = tck
    
    print(f"\nOur result:")
    print(f"Knots x: {len(tx)} - {tx}")
    print(f"Knots y: {len(ty)} - {ty}")
    print(f"Coefficients: {len(c)}")
    
    # Evaluate at grid points
    z_eval = bisplev(x_flat, y_flat, tck)
    error = np.abs(z_eval - z_flat).max()
    print(f"Max error at grid points: {error:.2e}")
    
    # Compare with SciPy
    tck_scipy = interpolate.bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    print(f"\nSciPy result:")
    print(f"Knots x: {len(tck_scipy[0])}")
    print(f"Knots y: {len(tck_scipy[1])}")
    print(f"Coefficients: {len(tck_scipy[2])}")


if __name__ == "__main__":
    test_cubic_on_grid()