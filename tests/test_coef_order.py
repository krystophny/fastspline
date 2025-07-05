"""Test coefficient ordering."""

import numpy as np


def test_coefficient_mapping():
    """Understand DIERCKX coefficient mapping."""
    # For a 2x2 grid with kx=ky=1
    # We have (nx-kx-1) * (ny-ky-1) = 2 * 2 = 4 coefficients
    
    # Grid points:
    # (0,0) -> z=0
    # (1,0) -> z=1  
    # (0,1) -> z=2
    # (1,1) -> z=3
    
    # For linear splines, the coefficients should equal the z values
    # at the grid points
    
    # DIERCKX ordering: c[i,j] -> c[(ny-ky-1)*i + j]
    # With nx=4, ny=4, kx=1, ky=1:
    # nk1x = 4-1-1 = 2
    # nk1y = 4-1-1 = 2
    
    print("DIERCKX coefficient mapping:")
    print("c[i,j] -> c_flat[(ny-ky-1)*i + j]")
    print("\nFor 2x2 grid:")
    
    nk1y = 2
    for i in range(2):
        for j in range(2):
            idx = nk1y * i + j
            print(f"c[{i},{j}] -> c_flat[{idx}]")
    
    # So the mapping should be:
    # c[0,0] -> c_flat[0] = z(0,0) = 0
    # c[0,1] -> c_flat[1] = z(0,1) = 2
    # c[1,0] -> c_flat[2] = z(1,0) = 1
    # c[1,1] -> c_flat[3] = z(1,1) = 3
    
    print("\nExpected coefficients: [0, 2, 1, 3]")
    
    # The issue is that our implementation gives [0, 1, 0, 3]
    # This suggests the basis function evaluation or assembly is wrong


if __name__ == "__main__":
    test_coefficient_mapping()