#!/usr/bin/env python3
"""
Simple test for exact floating-point equality between scipy and direct Fortran.
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import _dfitpack

def test_exact_equality():
    print("Testing exact floating-point equality...")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    x = np.random.uniform(-5, 5, n)
    y = np.random.uniform(-5, 5, n)
    z = np.sin(np.sqrt(x**2 + y**2))
    
    # Fit spline
    tck = bisplrep(x, y, z, s=n)
    tx, ty, c, kx, ky = tck
    
    # Test different evaluation scenarios
    test_cases = [
        ("10x10 grid", np.linspace(-4, 4, 10), np.linspace(-4, 4, 10)),
        ("50x50 grid", np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)),
        ("100x100 grid", np.linspace(-4, 4, 100), np.linspace(-4, 4, 100)),
        ("Random points", np.random.uniform(-4, 4, 30), np.random.uniform(-4, 4, 30)),
    ]
    
    all_exact = True
    
    for name, xi, yi in test_cases:
        print(f"\nTest: {name}")
        print("-" * 40)
        
        # Ensure proper format
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        yi = np.ascontiguousarray(yi, dtype=np.float64)
        
        # Method 1: scipy
        z_scipy = bisplev(xi, yi, tck)
        
        # Method 2: direct fortran
        z_fortran, ier = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)
        
        # Check equality
        if np.array_equal(z_scipy, z_fortran):
            print("✓ EXACT equality (bit-for-bit identical)")
        else:
            all_exact = False
            # Check if close
            max_diff = np.max(np.abs(z_scipy - z_fortran))
            rel_diff = np.max(np.abs(z_scipy - z_fortran) / np.abs(z_scipy))
            
            print(f"✗ Not exactly equal")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Max relative difference: {rel_diff:.2e}")
            
            if np.allclose(z_scipy, z_fortran, rtol=0, atol=0):
                print("  But: Arrays are equal with zero tolerance(?)")
            elif np.allclose(z_scipy, z_fortran, rtol=1e-15, atol=1e-15):
                print("  But: Arrays are equal within 1e-15")
            elif np.allclose(z_scipy, z_fortran):
                print("  But: Arrays are equal within numpy defaults")
    
    # Test with derivatives
    print(f"\nTest: Derivatives (dx=1, dy=0)")
    print("-" * 40)
    
    xi = np.linspace(-4, 4, 20)
    yi = np.linspace(-4, 4, 20)
    
    # scipy derivatives
    z_scipy_dx = bisplev(xi, yi, tck, dx=1, dy=0)
    
    # Direct fortran derivatives
    from scipy.interpolate import dfitpack
    z_fortran_dx, ier = dfitpack.parder(tx, ty, c, kx, ky, 1, 0, xi, yi)
    
    if np.array_equal(z_scipy_dx, z_fortran_dx):
        print("✓ EXACT equality for derivatives")
    else:
        all_exact = False
        max_diff = np.max(np.abs(z_scipy_dx - z_fortran_dx))
        print(f"✗ Not exactly equal for derivatives")
        print(f"  Max difference: {max_diff:.2e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    if all_exact:
        print("✓ ALL tests show EXACT floating-point equality!")
    else:
        print("✗ Some differences found (but typically within machine precision)")
    
    # Additional bit-level check on a few values
    print("\nBit-level comparison of first few values:")
    print("-" * 40)
    xi = np.array([0.0, 1.0, -1.0])
    yi = np.array([0.0, 1.0, -1.0])
    
    z_scipy = bisplev(xi, yi, tck)
    z_fortran, _ = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)
    
    for i in range(3):
        for j in range(3):
            vs = z_scipy[i, j] if z_scipy.ndim > 1 else z_scipy
            vf = z_fortran[i, j]
            
            # Get binary representation
            vs_bytes = vs.tobytes() if hasattr(vs, 'tobytes') else float(vs).hex()
            vf_bytes = vf.tobytes() if hasattr(vf, 'tobytes') else float(vf).hex()
            
            if vs == vf:
                print(f"({i},{j}): {vs:.15g} - EXACT match")
            else:
                print(f"({i},{j}): scipy={vs:.15g}, fortran={vf:.15g}, diff={vs-vf:.2e}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    test_exact_equality()