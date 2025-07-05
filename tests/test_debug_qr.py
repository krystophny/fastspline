"""Debug QR implementation."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import build_design_matrix, qr_decomposition_givens, back_substitution


def test_qr_simple():
    """Test QR decomposition on simple problem."""
    # Simple 2x2 grid
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([0., 1., 2., 3.])  # z = x + 2*y
    w = np.ones(4)
    
    # Knots for linear spline
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    kx = ky = 1
    
    # Build design matrix
    A, b = build_design_matrix(x, y, z, w, tx, ty, kx, ky)
    
    print("Design matrix A:")
    print(A)
    print("\nRHS vector b:")
    print(b)
    
    # Make a copy for QR
    A_qr = A.copy()
    b_qr = b.copy()
    
    # Apply QR decomposition
    b_qr = qr_decomposition_givens(A_qr, b_qr)
    
    print("\nAfter QR - R matrix:")
    print(A_qr)
    print("\nModified b:")
    print(b_qr)
    
    # Back substitution
    n = A.shape[1]
    c = back_substitution(A_qr, b_qr, n)
    
    print("\nCoefficients:")
    print(c)
    
    # Check solution
    residual = A @ c - b
    print("\nResidual (Ac - b):")
    print(residual)
    print(f"Residual norm: {np.linalg.norm(residual):.2e}")
    
    # Compare with numpy
    c_numpy, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("\nNumPy solution:")
    print(c_numpy)
    print(f"NumPy residual: {res}")


if __name__ == "__main__":
    test_qr_simple()