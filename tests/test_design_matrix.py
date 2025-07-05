"""Debug design matrix construction."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_dierckx import build_design_matrix_dierckx, find_span, basis_funs


def test_design_matrix():
    """Test design matrix for simple case."""
    # Simple 2x2 grid
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([0., 1., 2., 3.])
    w = np.ones(4)
    
    # Knots
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    kx = ky = 1
    
    print("Data points:")
    for i in range(4):
        print(f"  Point {i}: ({x[i]}, {y[i]}) -> z={z[i]}")
    
    # Build design matrix
    A, b = build_design_matrix_dierckx(x, y, z, w, tx, ty, kx, ky)
    
    print(f"\nDesign matrix A:")
    print(A)
    print(f"\nRHS vector b:")
    print(b)
    
    # For linear splines on a 2x2 grid, the design matrix should be:
    # Point (0,0): affects c[0,0] only -> row should be [1, 0, 0, 0]
    # Point (1,0): affects c[1,0] only -> row should be [0, 0, 1, 0]
    # Point (0,1): affects c[0,1] only -> row should be [0, 1, 0, 0]
    # Point (1,1): affects c[1,1] only -> row should be [0, 0, 0, 1]
    
    print("\nExpected design matrix:")
    print("[[1, 0, 0, 0],")
    print(" [0, 0, 1, 0],")
    print(" [0, 1, 0, 0],")
    print(" [0, 0, 0, 1]]")
    
    # Check basis functions at each point
    print("\n\nBasis function evaluation:")
    for i in range(4):
        span_x = find_span(len(tx), kx, x[i], tx)
        span_y = find_span(len(ty), ky, y[i], ty)
        Nx = basis_funs(span_x, x[i], kx, tx)
        Ny = basis_funs(span_y, y[i], ky, ty)
        
        print(f"\nPoint ({x[i]}, {y[i]}):")
        print(f"  span_x={span_x}, span_y={span_y}")
        print(f"  Nx={Nx}")
        print(f"  Ny={Ny}")
        
        # Which coefficients are affected?
        for ix in range(kx+1):
            for iy in range(ky+1):
                i_coef = span_x - kx + ix
                j_coef = span_y - ky + iy
                if 0 <= i_coef < 2 and 0 <= j_coef < 2:
                    coef_idx = i_coef * 2 + j_coef
                    val = Nx[ix] * Ny[iy]
                    if val != 0:
                        print(f"  Affects c[{i_coef},{j_coef}] (idx={coef_idx}) with weight {val}")


if __name__ == "__main__":
    test_design_matrix()