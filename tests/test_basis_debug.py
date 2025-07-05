"""Debug basis function evaluation."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import find_span, basis_funs


def test_basis_evaluation():
    """Test basis function evaluation."""
    # Simple knot vector [0, 0, 1, 1] for linear spline
    tx = np.array([0., 0., 1., 1.])
    kx = 1
    
    print("Knot vector:", tx)
    print("Degree:", kx)
    
    # Test at different points
    test_points = [0.0, 0.5, 1.0]
    
    for x in test_points:
        # Find span
        span = find_span(len(tx), kx, x, tx)
        print(f"\nAt x={x}:")
        print(f"  Span: {span}")
        
        # Evaluate basis functions
        N = basis_funs(span, x, kx, tx)
        print(f"  Basis functions: {N}")
        print(f"  Sum: {N.sum()}")
        
        # Check which basis functions are non-zero
        for i in range(len(N)):
            if N[i] != 0:
                print(f"    N[{span-kx+i}] = {N[i]}")


def test_2d_basis():
    """Test 2D tensor product basis."""
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    kx = ky = 1
    
    print("\n\n2D Basis Test:")
    print("tx:", tx)
    print("ty:", ty)
    
    # Test at corner points
    points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    for x, y in points:
        span_x = find_span(len(tx), kx, x, tx)
        span_y = find_span(len(ty), ky, y, ty)
        
        Nx = basis_funs(span_x, x, kx, tx)
        Ny = basis_funs(span_y, y, ky, ty)
        
        print(f"\nAt ({x}, {y}):")
        print(f"  Span x: {span_x}, Span y: {span_y}")
        print(f"  Nx: {Nx}")
        print(f"  Ny: {Ny}")
        
        # Show tensor product
        print("  Tensor product:")
        for i in range(kx+1):
            for j in range(ky+1):
                if Nx[i] * Ny[j] != 0:
                    idx_x = span_x - kx + i
                    idx_y = span_y - ky + j
                    print(f"    N[{idx_x},{idx_y}] = {Nx[i] * Ny[j]:.3f}")


if __name__ == "__main__":
    test_basis_evaluation()
    test_2d_basis()