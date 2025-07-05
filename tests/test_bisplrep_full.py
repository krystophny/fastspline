"""Test bisplrep_full implementation against SciPy."""

import numpy as np
import pytest
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline.bisplrep_full import bisplrep


def test_simple_surface():
    """Test on a simple polynomial surface."""
    # Create test data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2  # Simple quadratic surface
    
    # Flatten for bisplrep
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    # Fit with our implementation
    tck_ours = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    
    # Fit with SciPy
    tck_scipy = interpolate.bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
    
    # Check that we get reasonable knots
    assert len(tck_ours[0]) >= 2*(3+1)  # At least boundary knots
    assert len(tck_ours[1]) >= 2*(3+1)
    
    # Evaluate on test points
    x_test = np.linspace(0.1, 0.9, 5)
    y_test = np.linspace(0.1, 0.9, 5)
    
    # For now just check that evaluation doesn't crash
    # Will add proper evaluation once bisplev is integrated
    print(f"Our knots x: {len(tck_ours[0])}, y: {len(tck_ours[1])}")
    print(f"SciPy knots x: {len(tck_scipy[0])}, y: {len(tck_scipy[1])}")
    print(f"Our coefficients: {len(tck_ours[2])}")
    print(f"SciPy coefficients: {len(tck_scipy[2])}")


def test_data_ordering():
    """Test that data ordering works correctly."""
    # Create unordered data
    np.random.seed(42)
    n = 20
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = x + y
    
    # Fit should work regardless of order
    tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    assert len(tck[0]) >= 4  # At least 2*(1+1) knots
    assert len(tck[1]) >= 4
    

def test_weighted_fitting():
    """Test weighted least squares fitting."""
    # Create test data with outlier
    x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1])
    y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
    z = np.array([0, 0.5, 1, 0.5, 1, 1.5, 1, 1.5, 10])  # Last point is outlier
    w = np.ones_like(x)
    w[-1] = 0.01  # Down-weight outlier
    
    # Fit with weights
    tck_weighted = bisplrep(x, y, z, w=w, kx=1, ky=1, s=0)
    
    # Fit without weights  
    tck_unweighted = bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    # Just check that both produce valid results for now
    assert len(tck_weighted[2]) > 0
    assert len(tck_unweighted[2]) > 0


def test_linear_interpolation():
    """Test linear (k=1) interpolation."""
    # Simple 2x2 grid
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    z = np.array([0, 1, 2, 3])
    
    tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    # Should have minimal knots for linear
    assert len(tck[0]) == 4  # [0, 0, 1, 1]
    assert len(tck[1]) == 4  # [0, 0, 1, 1]


def test_smoothing():
    """Test smoothing parameter."""
    # Noisy data
    np.random.seed(42)
    x = np.random.rand(50)
    y = np.random.rand(50)
    z = np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + 0.1*np.random.randn(50)
    
    # Interpolating spline
    tck_interp = bisplrep(x, y, z, s=0)
    
    # Smoothing spline
    tck_smooth = bisplrep(x, y, z, s=1.0)
    
    # Smoothing spline should have fewer knots
    assert len(tck_smooth[0]) <= len(tck_interp[0])
    assert len(tck_smooth[1]) <= len(tck_interp[1])


if __name__ == "__main__":
    test_simple_surface()
    print("✓ Simple surface test passed")
    
    test_data_ordering()
    print("✓ Data ordering test passed")
    
    test_weighted_fitting()
    print("✓ Weighted fitting test passed")
    
    test_linear_interpolation()
    print("✓ Linear interpolation test passed")
    
    test_smoothing()
    print("✓ Smoothing test passed")