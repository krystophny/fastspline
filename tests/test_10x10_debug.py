"""Debug 10x10 grid fitting."""

import numpy as np
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import bisplrep
from fastspline.wrappers import bisplev


def test_10x10_grid():
    """Test on 10x10 grid."""
    # Create test data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    z = 1 + 2*xx + 3*yy + 4*xx*yy  # Bilinear
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    print(f"Data: {len(x_flat)} points")
    print(f"Function: z = 1 + 2x + 3y + 4xy (bilinear)")
    
    # Our implementation
    print(f"\nCalling bisplrep with s=0 (interpolation)")
    tck = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    tx, ty, c, kx_out, ky_out = tck
    
    print(f"\nOur result:")
    print(f"Knots x: {len(tx)} - {tx}")
    print(f"Knots y: {len(ty)} - {ty}")
    print(f"Coefficients: {len(c)}")
    print(f"Degrees: kx={kx_out}, ky={ky_out}")
    
    # Test at a few points
    test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    
    print("\nEvaluation at test points:")
    for x_test, y_test in test_points:
        z_true = 1 + 2*x_test + 3*y_test + 4*x_test*y_test
        z_eval = bisplev(np.array([x_test]), np.array([y_test]), tck)
        z_eval_scalar = float(z_eval)
        error = abs(z_eval_scalar - z_true)
        print(f"  ({x_test}, {y_test}): true={z_true:.3f}, eval={z_eval_scalar:.3f}, error={error:.2e}")
    
    # Evaluate on full grid
    z_eval_grid = bisplev(x, y, tck)
    z_true_grid = 1 + 2*xx + 3*yy + 4*xx*yy
    max_error = np.abs(z_eval_grid - z_true_grid).max()
    print(f"\nMax error on grid: {max_error:.2e}")
    
    # Show where the max error occurs
    idx = np.unravel_index(np.argmax(np.abs(z_eval_grid - z_true_grid)), z_eval_grid.shape)
    print(f"Max error at grid point ({xx[idx]:.2f}, {yy[idx]:.2f})")
    print(f"  True: {z_true_grid[idx]:.3f}, Eval: {z_eval_grid[idx]:.3f}")


if __name__ == "__main__":
    test_10x10_grid()