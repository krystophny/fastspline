import numpy as np
import dierckx_f2py
from dierckx_wrapper import bisplrep_dierckx
from scipy import interpolate

# Test basic functionality of f2py wrapper
def test_f2py_wrapper():
    # Create test data
    np.random.seed(42)
    nx, ny = 11, 11
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    z = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    
    # Flatten for surfit
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    # Test using our wrapper
    print("Testing DIERCKX wrapper:")
    try:
        tck_dierckx = bisplrep_dierckx(x_flat, y_flat, z_flat, kx=3, ky=3, s=0.0)
        tx_d, ty_d, c_d, kx_d, ky_d = tck_dierckx
        print(f"  Success! nx = {len(tx_d)}, ny = {len(ty_d)}")
        print(f"  Coefficient array shape: {c_d.shape}")
    except Exception as e:
        print(f"  Error: {e}")
        return False
    
    # Compare with scipy
    print("\nComparing with scipy.interpolate.bisplrep:")
    tck = interpolate.bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0.0)
    tx_scipy, ty_scipy, c_scipy = tck[:3]
    
    print(f"  scipy nx = {len(tx_scipy)}, ny = {len(ty_scipy)}")
    print(f"  scipy coefficient array shape: {c_scipy.shape}")
    
    # Compare knot arrays
    print(f"\nKnot comparison:")
    print(f"  tx match: {np.allclose(tx_d, tx_scipy)}")
    print(f"  ty match: {np.allclose(ty_d, ty_scipy)}")
    
    return True

if __name__ == "__main__":
    success = test_f2py_wrapper()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")