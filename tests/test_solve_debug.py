"""Debug QR solve."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_dierckx import build_design_matrix_dierckx, qr_givens_dierckx, solve_qr_system


def test_qr_solve():
    """Test QR solve for simple case."""
    # Simple system
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([0., 1., 2., 3.])
    w = np.ones(4)
    
    tx = np.array([0., 0., 1., 1.])
    ty = np.array([0., 0., 1., 1.])
    kx = ky = 1
    
    # Build design matrix
    A, b = build_design_matrix_dierckx(x, y, z, w, tx, ty, kx, ky)
    
    print("Original system:")
    print("A =")
    print(A)
    print("\nb =", b)
    
    # Make copies for QR
    A_qr = A.copy()
    b_qr = b.copy()
    
    # Apply QR decomposition
    A_qr, b_qr = qr_givens_dierckx(A_qr, b_qr)
    
    print("\n\nAfter QR decomposition:")
    print("R =")
    print(A_qr)
    print("\nQ'b =", b_qr)
    
    # Solve
    c = solve_qr_system(A_qr, b_qr, 4)
    
    print("\n\nSolution:")
    print("c =", c)
    
    # Check solution
    residual = A @ c - b
    print("\nResidual (Ac - b):", residual)
    print("Residual norm:", np.linalg.norm(residual))
    
    # Compare with numpy
    c_numpy = np.linalg.lstsq(A, b, rcond=None)[0]
    print("\nNumPy solution:", c_numpy)
    
    # The solution should be [0, 2, 1, 3]
    print("\nExpected solution: [0, 2, 1, 3]")


if __name__ == "__main__":
    test_qr_solve()