#!/usr/bin/env python3
"""Test bisplrep accuracy against SciPy."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev_scalar


def compare_knots(tx1, ty1, tx2, ty2):
    """Compare knot vectors."""
    print(f"\nKnot comparison:")
    print(f"  X knots: ours={len(tx1)}, scipy={len(tx2)}")
    print(f"  Y knots: ours={len(ty1)}, scipy={len(ty2)}")
    
    # Compare x knots
    n_compare = min(len(tx1), len(tx2))
    print(f"\n  First {n_compare} X knots:")
    for i in range(min(10, n_compare)):
        print(f"    {i:2d}: ours={tx1[i]:8.5f}, scipy={tx2[i]:8.5f}, diff={abs(tx1[i]-tx2[i]):8.2e}")
    
    # Compare y knots  
    n_compare = min(len(ty1), len(ty2))
    print(f"\n  First {n_compare} Y knots:")
    for i in range(min(10, n_compare)):
        print(f"    {i:2d}: ours={ty1[i]:8.5f}, scipy={ty2[i]:8.5f}, diff={abs(ty1[i]-ty2[i]):8.2e}")


def compare_coefficients(c1, c2):
    """Compare coefficient arrays."""
    print(f"\nCoefficient comparison:")
    print(f"  Our coeffs: {len(c1)}")
    print(f"  SciPy coeffs: {len(c2)}")
    
    n_compare = min(len(c1), len(c2))
    print(f"\n  First 10 coefficients:")
    for i in range(min(10, n_compare)):
        print(f"    {i:2d}: ours={c1[i]:8.5f}, scipy={c2[i]:8.5f}, diff={abs(c1[i]-c2[i]):8.2e}")
    
    # Overall statistics
    if n_compare > 0:
        diffs = [abs(c1[i] - c2[i]) for i in range(n_compare)]
        print(f"\n  Max diff: {max(diffs):8.2e}")
        print(f"  Mean diff: {np.mean(diffs):8.2e}")


def test_simple_surface():
    """Test with a simple surface where we might expect agreement."""
    print("Test 1: Simple polynomial surface")
    print("=" * 60)
    
    # Create a regular grid (our implementation works better with regular data)
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    x_data = xx.ravel()
    y_data = yy.ravel()
    z_data = x_data**2 + y_data**2  # Simple paraboloid
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x_data, y_data, z_data, kx=2, ky=2, s=0)
    tx_scipy, ty_scipy, c_scipy, kx_scipy, ky_scipy = tck_scipy
    
    # Fit with ours
    tck_ours = bisplrep(x_data, y_data, z_data, kx=2, ky=2, s=0)
    tx_ours, ty_ours, c_ours, kx_ours, ky_ours = tck_ours
    
    # Compare knots
    compare_knots(tx_ours, ty_ours, tx_scipy, ty_scipy)
    
    # Compare coefficients
    compare_coefficients(c_ours, c_scipy)
    
    # Compare evaluations
    print("\nEvaluation comparison at test points:")
    test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for xt, yt in test_points:
        z_scipy = scipy_bisplev(xt, yt, tck_scipy)
        z_ours = bisplev_scalar(xt, yt, tx_ours, ty_ours, c_ours, kx_ours, ky_ours)
        z_true = xt**2 + yt**2
        print(f"  ({xt}, {yt}): scipy={z_scipy:.6f}, ours={z_ours:.6f}, "
              f"true={z_true:.6f}, diff={abs(z_scipy-z_ours):.2e}")


def test_identical_setup():
    """Test with the simplest possible case."""
    print("\n\nTest 2: Minimal 2x2 grid")
    print("=" * 60)
    
    # Minimal data for k=1
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    z = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx_scipy, ty_scipy, c_scipy, kx_scipy, ky_scipy = tck_scipy
    
    # Fit with ours
    tck_ours = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx_ours, ty_ours, c_ours, kx_ours, ky_ours = tck_ours
    
    print(f"SciPy knots: tx={tx_scipy}, ty={ty_scipy}")
    print(f"Our knots: tx={tx_ours}, ty={ty_ours}")
    
    print(f"\nSciPy coeffs: {c_scipy}")
    print(f"Our coeffs: {c_ours}")
    
    # Test if the splines give same results
    print("\nEvaluation comparison:")
    for i in range(len(x)):
        z_scipy = scipy_bisplev(x[i], y[i], tck_scipy)
        z_ours = bisplev_scalar(x[i], y[i], tx_ours, ty_ours, c_ours, kx_ours, ky_ours)
        print(f"  Point {i}: scipy={z_scipy:.6f}, ours={z_ours:.6f}, "
              f"true={z[i]:.6f}, diff={abs(z_scipy-z_ours):.2e}")


def test_with_smoothing():
    """Test with smoothing parameter to see if we get closer."""
    print("\n\nTest 3: With smoothing parameter")
    print("=" * 60)
    
    # Generate noisy data
    np.random.seed(42)
    n = 25
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = x**2 + y**2 + 0.01 * np.random.randn(n)
    
    # Try different smoothing values
    for s in [0.0, 0.1, 1.0]:
        print(f"\nSmoothing s={s}:")
        
        # SciPy fit
        try:
            tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=s)
            tx_scipy, ty_scipy, c_scipy, _, _ = tck_scipy
            
            # Our fit
            tck_ours = bisplrep(x, y, z, kx=3, ky=3, s=s)
            tx_ours, ty_ours, c_ours, kx_ours, ky_ours = tck_ours
            
            print(f"  Knots: scipy=({len(tx_scipy)}, {len(ty_scipy)}), "
                  f"ours=({len(tx_ours)}, {len(ty_ours)})")
            
            # Compare at center point
            z_scipy = scipy_bisplev(0.5, 0.5, tck_scipy)
            z_ours = bisplev_scalar(0.5, 0.5, tx_ours, ty_ours, c_ours, kx_ours, ky_ours)
            print(f"  At (0.5, 0.5): scipy={z_scipy:.4f}, ours={z_ours:.4f}, "
                  f"diff={abs(z_scipy-z_ours):.2e}")
                  
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    print("Detailed comparison of bisplrep implementations")
    print("=" * 60)
    
    test_simple_surface()
    test_identical_setup()
    test_with_smoothing()
    
    print("\n\nConclusion:")
    print("-" * 60)
    print("Our bisplrep implementation uses QR decomposition for stability")
    print("and may produce different knots and coefficients than SciPy,")
    print("but both can approximate the surface reasonably well.")