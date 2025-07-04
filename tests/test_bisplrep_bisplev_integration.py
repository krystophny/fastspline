"""Integration tests for bisplrep and bisplev working together."""

import numpy as np
import pytest
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastspline import bisplrep, bisplev, bisplev_scalar


def test_bisplrep_bisplev_integration():
    """Test that our bisplrep output works with our bisplev."""
    np.random.seed(42)
    
    # Generate test data
    n = 50
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.exp(-(x**2 + y**2))
    
    # Fit with our bisplrep (returns tck tuple)
    tck = bisplrep(x, y, z, kx=3, ky=3)
    tx, ty, c, kx, ky = tck
    
    print(f"Our bisplrep produced nx={len(tx)}, ny={len(ty)} knots")
    
    # Test evaluation at various points
    test_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.5, -0.5),
        (-0.5, -0.5)
    ]
    
    for xt, yt in test_points:
        tx, ty, c, kx, ky = tck
        z_eval = bisplev_scalar(xt, yt, tx, ty, c, kx, ky)
        z_true = np.exp(-(xt**2 + yt**2))
        error = abs(z_eval - z_true)
        print(f"Point ({xt:5.1f}, {yt:5.1f}): eval={z_eval:.4f}, true={z_true:.4f}, error={error:.2e}")
        assert np.isfinite(z_eval), f"Non-finite result at ({xt}, {yt})"
        assert error < 0.1, f"Large error at ({xt}, {yt}): {error}"


def test_compare_with_scipy_fit():
    """Compare our bisplrep+bisplev with SciPy's bisplrep+bisplev."""
    np.random.seed(42)
    
    # Generate smooth test surface
    n = 100
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = np.sin(np.pi * x) * np.cos(np.pi * y)
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.1)
    
    # Fit with our implementation
    tck_ours = bisplrep(x, y, z, kx=3, ky=3, s=0.1)
    
    print(f"\nSciPy: nx={len(tck_scipy[0])}, ny={len(tck_scipy[1])}")
    print(f"Ours:  nx={len(tck_ours[0])}, ny={len(tck_ours[1])}")
    
    # Compare evaluations on a grid
    n_test = 10
    x_test = np.linspace(-0.8, 0.8, n_test)
    y_test = np.linspace(-0.8, 0.8, n_test)
    
    max_scipy_ours_diff = 0
    max_true_scipy_diff = 0
    max_true_ours_diff = 0
    
    for i in range(n_test):
        for j in range(n_test):
            xt, yt = x_test[i], y_test[j]
            
            # True value
            z_true = np.sin(np.pi * xt) * np.cos(np.pi * yt)
            
            # SciPy evaluation
            z_scipy = scipy_bisplev(xt, yt, tck_scipy)
            
            # Our evaluation
            tx, ty, c, kx, ky = tck_ours
            z_ours = bisplev_scalar(xt, yt, tx, ty, c, kx, ky)
            
            # Track differences
            max_scipy_ours_diff = max(max_scipy_ours_diff, abs(z_scipy - z_ours))
            max_true_scipy_diff = max(max_true_scipy_diff, abs(z_true - z_scipy))
            max_true_ours_diff = max(max_true_ours_diff, abs(z_true - z_ours))
    
    print(f"\nMax differences:")
    print(f"  SciPy vs Ours:  {max_scipy_ours_diff:.2e}")
    print(f"  True vs SciPy:  {max_true_scipy_diff:.2e}")
    print(f"  True vs Ours:   {max_true_ours_diff:.2e}")
    
    # Both should give reasonable approximations
    assert max_true_scipy_diff < 0.2, "SciPy approximation error too large"
    assert max_true_ours_diff < 0.5, "Our approximation error too large"


def test_scipy_knots_with_our_bisplev():
    """Verify our bisplev works perfectly with SciPy's knots."""
    np.random.seed(42)
    
    # Generate data
    n = 100
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    z = x**3 + y**3 + x*y
    
    # Fit with SciPy only
    tck = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.1)
    tx, ty, c, kx, ky = tck
    
    # Test evaluation agreement
    n_test = 50
    x_test = np.random.uniform(-0.9, 0.9, n_test)
    y_test = np.random.uniform(-0.9, 0.9, n_test)
    
    max_diff = 0
    for i in range(n_test):
        z_scipy = scipy_bisplev(x_test[i], y_test[i], tck)
        z_ours = bisplev_scalar(x_test[i], y_test[i], tx, ty, c, kx, ky)
        diff = abs(z_scipy - z_ours)
        max_diff = max(max_diff, diff)
    
    print(f"\nUsing SciPy knots with our bisplev:")
    print(f"  Maximum difference: {max_diff:.2e}")
    
    assert max_diff < 1e-14, f"Our bisplev doesn't match SciPy bisplev: {max_diff}"
    print("  ✓ Perfect agreement!")


if __name__ == "__main__":
    print("Integration tests for bisplrep and bisplev")
    print("=" * 50)
    
    test_bisplrep_bisplev_integration()
    test_compare_with_scipy_fit()
    test_scipy_knots_with_our_bisplev()
    
    print("\nAll integration tests passed!")