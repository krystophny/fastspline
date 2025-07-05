"""Debug knot span calculation."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import find_span


def test_span_calculation():
    """Test span calculation."""
    # Simple knot vector [0, 0, 1, 1] for linear spline
    tx = np.array([0., 0., 1., 1.])
    kx = 1
    n = len(tx)
    
    print("Knot vector:", tx)
    print("Degree:", kx)
    print("Number of knots:", n)
    print("Number of basis functions:", n - kx - 1)
    
    # Test at different points
    test_points = [0.0, 0.25, 0.5, 0.75, 0.99, 1.0]
    
    for x in test_points:
        # Find span
        span = find_span(n, kx, x, tx)
        print(f"\nx={x}: span={span}")
        
        # Check which knot interval
        for i in range(n-1):
            if tx[i] <= x < tx[i+1]:
                print(f"  In interval [{tx[i]}, {tx[i+1]}), i={i}")
                break
        else:
            if x == tx[-1]:
                print(f"  At end point, should use last valid span")


if __name__ == "__main__":
    test_span_calculation()