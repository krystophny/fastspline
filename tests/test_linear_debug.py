"""Debug linear spline test."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import bisplrep
from fastspline.wrappers import bisplev


def test_linear_debug():
    """Debug linear spline."""
    # Simple 3x3 grid
    x = np.array([0, 0.5, 1])
    y = np.array([0, 0.5, 1])
    xx, yy = np.meshgrid(x, y)
    z = xx + 2*yy  # Linear function
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    print("Data points:")
    for i in range(len(x_flat)):
        print(f"  ({x_flat[i]}, {y_flat[i]}) -> {z_flat[i]}")
    
    # Fit linear spline
    tck = bisplrep(x_flat, y_flat, z_flat, kx=1, ky=1, s=0)
    tx, ty, c, kx, ky = tck
    
    print(f"\nKnots:")
    print(f"tx: {tx}")
    print(f"ty: {ty}")
    print(f"Coefficients: {c}")
    
    # Evaluate at grid points
    z_eval_flat = bisplev(x_flat, y_flat, tck)
    print(f"\nEvaluation at grid points:")
    for i in range(len(x_flat)):
        print(f"  ({x_flat[i]}, {y_flat[i]}): true={z_flat[i]}, eval={z_eval_flat[i]:.3f}")
    
    # Test at intermediate point
    x_test = np.array([0.25])
    y_test = np.array([0.25])
    z_test = bisplev(x_test, y_test, tck)
    z_true = 0.25 + 2*0.25
    
    print(f"\nAt (0.25, 0.25):")
    print(f"  True: {z_true}")
    print(f"  Eval: {float(z_test)}")
    print(f"  Error: {abs(float(z_test) - z_true)}")


if __name__ == "__main__":
    test_linear_debug()