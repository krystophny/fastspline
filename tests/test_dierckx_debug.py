"""Debug DIERCKX implementation."""

import numpy as np
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline import bisplrep
from fastspline.wrappers import bisplev


def test_simple():
    """Test simple case."""
    # Very simple data
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    z = np.array([0, 1, 2, 3])  # z = x + 2*y
    
    print("Input data:")
    for i in range(len(x)):
        print(f"  ({x[i]}, {y[i]}) -> {z[i]}")
    
    # Fit with our implementation
    tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx, ty, c, kx, ky = tck
    
    print(f"\nOur result:")
    print(f"tx: {tx}")
    print(f"ty: {ty}")
    print(f"c: {c}")
    print(f"Degrees: kx={kx}, ky={ky}")
    
    # Fit with SciPy
    tck_scipy = interpolate.bisplrep(x, y, z, kx=1, ky=1, s=0)
    print(f"\nSciPy result:")
    print(f"tx: {tck_scipy[0]}")
    print(f"ty: {tck_scipy[1]}")
    print(f"c: {tck_scipy[2]}")
    
    # Test evaluation at a single point
    x_test = 0.5
    y_test = 0.5
    
    z_ours = bisplev(np.array([x_test]), np.array([y_test]), tck)
    z_scipy = interpolate.bisplev(x_test, y_test, tck_scipy)
    z_true = x_test + 2*y_test
    
    print(f"\nEvaluation at ({x_test}, {y_test}):")
    print(f"  Our result: {float(z_ours)}")
    print(f"  SciPy result: {float(z_scipy)}")
    print(f"  True value: {z_true}")
    
    # Test pointwise evaluation
    x_pts = np.array([0, 0.5, 1])
    y_pts = np.array([0, 0.5, 1])
    
    z_ours_pts = bisplev(x_pts, y_pts, tck)
    z_true_pts = x_pts + 2*y_pts
    
    print(f"\nPointwise evaluation:")
    for i in range(len(x_pts)):
        print(f"  ({x_pts[i]}, {y_pts[i]}): ours={z_ours_pts[i]:.3f}, true={z_true_pts[i]:.3f}")


def test_meshgrid():
    """Test meshgrid evaluation."""
    # Simple grid
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    z = np.array([0, 1, 2, 3])
    
    tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tck_scipy = interpolate.bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    # Meshgrid evaluation
    x_grid = np.array([0, 0.5, 1])
    y_grid = np.array([0, 0.5, 1])
    
    z_ours = bisplev(x_grid, y_grid, tck)
    z_scipy = interpolate.bisplev(x_grid, y_grid, tck_scipy)
    
    print("\nMeshgrid evaluation:")
    print(f"Our shape: {z_ours.shape}")
    print(f"SciPy shape: {z_scipy.shape}")
    print(f"\nOur result:\n{z_ours}")
    print(f"\nSciPy result:\n{z_scipy}")


if __name__ == "__main__":
    test_simple()
    test_meshgrid()