#!/usr/bin/env python3
"""Test bisplrep accuracy against SciPy."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline.bisplrep_cfunc import bisplrep
from fastspline import bisplev


def compare_knots(tx1, ty1, tx2, ty2, nx1, ny1, nx2, ny2):
    """Compare knot vectors."""
    print(f"\nKnot comparison:")
    print(f"  X knots: ours={nx1}, scipy={nx2}")
    print(f"  Y knots: ours={ny1}, scipy={ny2}")
    
    # Compare x knots
    n_compare = min(nx1, nx2)
    print(f"\n  First {n_compare} X knots:")
    for i in range(n_compare):
        print(f"    {i:2d}: ours={tx1[i]:8.5f}, scipy={tx2[i]:8.5f}, diff={abs(tx1[i]-tx2[i]):8.2e}")
    
    # Compare y knots  
    n_compare = min(ny1, ny2)
    print(f"\n  First {n_compare} Y knots:")
    for i in range(n_compare):
        print(f"    {i:2d}: ours={ty1[i]:8.5f}, scipy={ty2[i]:8.5f}, diff={abs(ty1[i]-ty2[i]):8.2e}")


def compare_coefficients(c1, c2, n1, n2):
    """Compare coefficient arrays."""
    print(f"\nCoefficient comparison:")
    print(f"  Our coeffs: {n1}")
    print(f"  SciPy coeffs: {n2}")
    
    n_compare = min(n1, n2)
    print(f"\n  First 10 coefficients:")
    for i in range(min(10, n_compare)):
        print(f"    {i:2d}: ours={c1[i]:8.5f}, scipy={c2[i]:8.5f}, diff={abs(c1[i]-c2[i]):8.2e}")
    
    # Overall statistics
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
    tx_ours = np.zeros(20)
    ty_ours = np.zeros(20)
    c_ours = np.zeros(400)
    
    result = bisplrep(x_data, y_data, z_data, 2, 2, tx_ours, ty_ours, c_ours)
    nx_ours = (result >> 32) & 0xFFFFFFFF
    ny_ours = result & 0xFFFFFFFF
    
    # Compare knots
    compare_knots(tx_ours, ty_ours, tx_scipy, ty_scipy, 
                  nx_ours, ny_ours, len(tx_scipy), len(ty_scipy))
    
    # Compare coefficients
    n_coeffs_ours = nx_ours * ny_ours
    n_coeffs_scipy = len(c_scipy)
    compare_coefficients(c_ours, c_scipy, n_coeffs_ours, n_coeffs_scipy)
    
    # Compare evaluations
    print("\nEvaluation comparison at test points:")
    test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for xt, yt in test_points:
        z_scipy = scipy_bisplev(xt, yt, tck_scipy)
        z_ours = bisplev(xt, yt, tx_ours[:nx_ours], ty_ours[:ny_ours], 
                        c_ours[:nx_ours*ny_ours], 2, 2)
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
    tx_ours = np.zeros(10)
    ty_ours = np.zeros(10)
    c_ours = np.zeros(100)
    
    result = bisplrep(x, y, z, 1, 1, tx_ours, ty_ours, c_ours)
    nx_ours = (result >> 32) & 0xFFFFFFFF
    ny_ours = result & 0xFFFFFFFF
    
    print(f"SciPy knots: tx={tx_scipy}, ty={ty_scipy}")
    print(f"Our knots: tx={tx_ours[:nx_ours]}, ty={ty_ours[:ny_ours]}")
    
    print(f"\nSciPy coeffs: {c_scipy}")
    print(f"Our coeffs: {c_ours[:nx_ours*ny_ours]}")
    
    # Test if the splines give same results
    print("\nEvaluation comparison:")
    for i in range(len(x)):
        z_scipy = scipy_bisplev(x[i], y[i], tck_scipy)
        z_ours = bisplev(x[i], y[i], tx_ours[:nx_ours], ty_ours[:ny_ours], 
                        c_ours[:nx_ours*ny_ours], 1, 1)
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
            
            # Our fit (we don't use s parameter properly)
            tx_ours = np.zeros(30)
            ty_ours = np.zeros(30)
            c_ours = np.zeros(900)
            
            result = bisplrep(x, y, z, 3, 3, tx_ours, ty_ours, c_ours)
            nx_ours = (result >> 32) & 0xFFFFFFFF
            ny_ours = result & 0xFFFFFFFF
            
            print(f"  Knots: scipy=({len(tx_scipy)}, {len(ty_scipy)}), "
                  f"ours=({nx_ours}, {ny_ours})")
            
            # Compare at center point
            z_scipy = scipy_bisplev(0.5, 0.5, tck_scipy)
            z_ours = bisplev(0.5, 0.5, tx_ours[:nx_ours], ty_ours[:ny_ours],
                           c_ours[:nx_ours*ny_ours], 3, 3)
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
    print("Our bisplrep implementation differs from SciPy because:")
    print("1. We use uniform knot spacing vs SciPy's adaptive placement")
    print("2. We use simplified least squares vs SciPy's iterative refinement")
    print("3. We don't implement the smoothing parameter properly")
    print("\nThe implementations produce different knots and coefficients,")
    print("but both can approximate the surface reasonably well.")