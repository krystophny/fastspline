"""Test basis functions directly."""

import numpy as np
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_qr import basis_funs


def test_basis_direct():
    """Test basis function evaluation directly."""
    # For knot vector [0, 0, 1, 1] and degree 1
    U = np.array([0., 0., 1., 1.])
    p = 1
    
    print("Testing basis_funs directly")
    print("Knot vector:", U)
    print("Degree:", p)
    
    # Test at span 1, x=0.5
    i = 1  # span index
    u = 0.5
    
    print(f"\nAt span {i}, u={u}:")
    print(f"Relevant knots: {U[i-p:i+p+1]}")
    
    N = basis_funs(i, u, p, U)
    print(f"Basis functions: {N}")
    
    # Manual calculation for linear case
    # N[0] should be (U[i+1] - u) / (U[i+1] - U[i])
    # N[1] should be (u - U[i]) / (U[i+1] - U[i])
    
    if U[i+1] != U[i]:
        N0_manual = (U[i+1] - u) / (U[i+1] - U[i])
        N1_manual = (u - U[i]) / (U[i+1] - U[i])
        print(f"Manual calculation: N[0]={N0_manual}, N[1]={N1_manual}")


if __name__ == "__main__":
    test_basis_direct()