#!/usr/bin/env python3
"""Test the full bisplrep implementation against SciPy."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline.bisplrep_full import bisplrep_full, bisplrep_full_py
from fastspline import bisplev


def test_simple_case():
    """Test with simple data to check if algorithm works."""
    print("Test 1: Simple 3x3 grid")
    print("=" * 60)
    
    # Simple grid
    x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], dtype=np.float64)
    y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1], dtype=np.float64) 
    z = x**2 + y**2
    w = np.ones_like(x)
    
    # Fit with our full implementation
    tx = np.zeros(20, dtype=np.float64)
    ty = np.zeros(20, dtype=np.float64)
    c = np.zeros(400, dtype=np.float64)
    
    result = bisplrep_full(x, y, z, w, 2, 2, 0.0, tx, ty, c)
    nx = (result >> 32) & 0xFFFFFFFF
    ny = result & 0xFFFFFFFF
    
    print(f"Our implementation: nx={nx}, ny={ny}")
    print(f"X knots: {tx[:nx]}")
    print(f"Y knots: {ty[:ny]}")
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x, y, z, w, kx=2, ky=2, s=0.0)
    print(f"\nSciPy: nx={len(tck_scipy[0])}, ny={len(tck_scipy[1])}")
    print(f"X knots: {tck_scipy[0]}")
    print(f"Y knots: {tck_scipy[1]}")
    
    # Test evaluation
    print("\nEvaluation test:")
    test_pts = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for xt, yt in test_pts:
        z_ours = bisplev(xt, yt, tx[:nx], ty[:ny], c[:nx*ny], 2, 2)
        z_scipy = scipy_bisplev(xt, yt, tck_scipy)
        z_true = xt**2 + yt**2
        print(f"  ({xt}, {yt}): ours={z_ours:.6f}, scipy={z_scipy:.6f}, "
              f"true={z_true:.6f}, diff={abs(z_ours-z_scipy):.2e}")


def test_knot_placement():
    """Test adaptive knot placement."""
    print("\n\nTest 2: Adaptive knot placement")
    print("=" * 60)
    
    # Generate data with varying density
    np.random.seed(42)
    n = 50
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    # Function with more variation in one region
    z = np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1) + 0.1*x + 0.1*y
    w = np.ones_like(x)
    
    # Test with different smoothing values
    for s in [0.0, 0.1, 1.0]:
        print(f"\nSmoothing s={s}:")
        
        # Our implementation
        tx = np.zeros(30, dtype=np.float64)
        ty = np.zeros(30, dtype=np.float64)
        c = np.zeros(900, dtype=np.float64)
        
        result = bisplrep_full(x, y, z, w, 3, 3, s, tx, ty, c)
        nx_ours = (result >> 32) & 0xFFFFFFFF
        ny_ours = result & 0xFFFFFFFF
        
        # SciPy
        try:
            tck_scipy = scipy_bisplrep(x, y, z, w, kx=3, ky=3, s=s)
            nx_scipy = len(tck_scipy[0])
            ny_scipy = len(tck_scipy[1])
            
            print(f"  Knots: ours=({nx_ours}, {ny_ours}), scipy=({nx_scipy}, {ny_scipy})")
            
            # Compare knot positions (first few interior knots)
            print("  X knot comparison (interior):")
            n_compare = min(nx_ours-8, nx_scipy-8, 3)
            if n_compare > 0:
                for i in range(n_compare):
                    idx = i + 4  # Skip boundary knots
                    print(f"    {i}: ours={tx[idx]:.4f}, scipy={tck_scipy[0][idx]:.4f}, "
                          f"diff={abs(tx[idx]-tck_scipy[0][idx]):.2e}")
            
        except Exception as e:
            print(f"  SciPy failed: {e}")


def test_accuracy_match():
    """Test if we can match SciPy's accuracy."""
    print("\n\nTest 3: Accuracy comparison")
    print("=" * 60)
    
    # Use Python wrapper for easier testing
    np.random.seed(42)
    n = 25
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    z = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
    
    # Fit with both
    tck_ours = bisplrep_full_py(x, y, z, kx=3, ky=3, s=0.01)
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.01)
    
    print(f"Our knots: x={len(tck_ours[0])}, y={len(tck_ours[1])}")
    print(f"SciPy knots: x={len(tck_scipy[0])}, y={len(tck_scipy[1])}")
    
    # Evaluate on test grid
    n_test = 10
    x_test = np.linspace(0.1, 0.9, n_test)
    y_test = np.linspace(0.1, 0.9, n_test)
    
    max_diff = 0
    sum_sq_diff = 0
    count = 0
    
    for i in range(n_test):
        for j in range(n_test):
            z_ours = bisplev(x_test[i], y_test[j], 
                           tck_ours[0], tck_ours[1], tck_ours[2], 3, 3)
            z_scipy = scipy_bisplev(x_test[i], y_test[j], tck_scipy)
            z_true = np.sin(2*np.pi*x_test[i]) * np.cos(2*np.pi*y_test[j])
            
            diff = abs(z_ours - z_scipy)
            max_diff = max(max_diff, diff)
            sum_sq_diff += diff**2
            count += 1
            
            if i == n_test//2 and j == n_test//2:
                print(f"\nAt center point (0.5, 0.5):")
                print(f"  Ours:  {z_ours:.6f}")
                print(f"  SciPy: {z_scipy:.6f}")  
                print(f"  True:  {z_true:.6f}")
                print(f"  Diff:  {diff:.2e}")
    
    rms_diff = np.sqrt(sum_sq_diff / count)
    print(f"\nOverall comparison:")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  RMS difference:  {rms_diff:.2e}")


if __name__ == "__main__":
    print("Testing full FITPACK-style bisplrep implementation")
    print("=" * 60)
    
    test_simple_case()
    test_knot_placement()
    test_accuracy_match()
    
    print("\n\nConclusion:")
    print("-" * 60)
    print("The full implementation includes:")
    print("- Iterative knot placement based on residuals")
    print("- Proper handling of smoothing factor")
    print("- More sophisticated linear algebra")
    print("\nHowever, exact agreement with SciPy requires:")
    print("- Identical knot placement strategy") 
    print("- Same numerical tolerances")
    print("- Exact same QR/Givens rotation implementation")
    print("- Same handling of edge cases")
    print("\nThis is why SciPy uses the original Fortran code!")