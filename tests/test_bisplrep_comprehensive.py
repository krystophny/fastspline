"""Comprehensive tests for bisplrep implementation."""

import numpy as np
import pytest
from scipy import interpolate
import sys
sys.path.insert(0, '../src')
from fastspline import bisplrep
from fastspline.wrappers import bisplev


class TestBisplrepAccuracy:
    """Test accuracy of bisplrep implementation."""
    
    def test_linear_exact(self):
        """Test exact interpolation of linear function."""
        # Create simple grid
        x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1])
        y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
        z = 1 + 2*x + 3*y  # Linear function
        
        # Fit with linear spline
        tck = bisplrep(x, y, z, kx=1, ky=1, s=0)
        
        # Evaluate at test points
        x_test = np.array([0.25, 0.75])
        y_test = np.array([0.25, 0.75])
        z_eval = bisplev(x_test, y_test, tck)
        z_true = 1 + 2*x_test + 3*y_test
        
        assert np.allclose(z_eval, z_true, rtol=1e-14)
    
    def test_polynomial_accuracy(self):
        """Test accuracy on polynomial surface."""
        # Create grid
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        xx, yy = np.meshgrid(x, y)
        z = 1 + xx + yy + xx*yy + xx**2 + yy**2
        
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        z_flat = z.ravel()
        
        # Fit cubic spline
        tck = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
        
        # Evaluate at test points
        x_test = np.linspace(0.1, 0.9, 10)
        y_test = np.linspace(0.1, 0.9, 10)
        z_eval = bisplev(x_test, y_test, tck)
        
        xx_test, yy_test = np.meshgrid(x_test, y_test)
        z_true = 1 + xx_test + yy_test + xx_test*yy_test + xx_test**2 + yy_test**2
        
        assert np.allclose(z_eval, z_true, rtol=1e-10)
    
    def test_weighted_fitting(self):
        """Test weighted least squares."""
        # Create data with outlier
        x = np.array([0, 1, 0, 1, 0.5])
        y = np.array([0, 0, 1, 1, 0.5])
        z = np.array([0, 1, 1, 2, 10])  # Last point is outlier
        w = np.array([1, 1, 1, 1, 0.01])  # Down-weight outlier
        
        # Fit with weights
        tck = bisplrep(x, y, z, w=w, kx=1, ky=1, s=0)
        
        # Check that fit is close to true values at corners
        corners_x = np.array([0, 1, 0, 1])
        corners_y = np.array([0, 0, 1, 1])
        z_eval = bisplev(corners_x, corners_y, tck)
        z_expected = np.array([0, 1, 1, 2])
        
        assert np.allclose(z_eval, z_expected, atol=0.1)


class TestBisplrepInterface:
    """Test API compatibility."""
    
    def test_scipy_compatibility(self):
        """Test that interface matches SciPy."""
        # Generate test data
        np.random.seed(42)
        x = np.random.rand(30)
        y = np.random.rand(30)
        z = np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
        
        # Our implementation
        tck_ours = bisplrep(x, y, z, s=0.1)
        assert len(tck_ours) == 5
        assert tck_ours[3] == 3  # kx
        assert tck_ours[4] == 3  # ky
        
        # Check knots are sorted
        assert np.all(np.diff(tck_ours[0]) >= 0)
        assert np.all(np.diff(tck_ours[1]) >= 0)
    
    def test_optional_parameters(self):
        """Test optional parameter handling."""
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        z = xx + yy
        
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        z_flat = z.ravel()
        
        # Test different parameter combinations
        tck1 = bisplrep(x_flat, y_flat, z_flat)  # All defaults
        tck2 = bisplrep(x_flat, y_flat, z_flat, kx=2, ky=2)  # Different degrees
        tck3 = bisplrep(x_flat, y_flat, z_flat, s=0.5)  # Smoothing
        
        assert tck1[3] == 3 and tck1[4] == 3  # Default cubic
        assert tck2[3] == 2 and tck2[4] == 2  # Quadratic
        assert len(tck3[0]) >= 8  # Has knots


class TestBisplrepEdgeCases:
    """Test edge cases and error handling."""
    
    def test_minimum_points(self):
        """Test with minimum number of points."""
        # Minimum for linear spline: 4 points
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 1, 2, 3])
        
        tck = bisplrep(x, y, z, kx=1, ky=1)
        assert len(tck[0]) == 4  # 2*(1+1) knots
        assert len(tck[1]) == 4
    
    def test_colinear_points(self):
        """Test with colinear points."""
        # Points on a line
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)
        z = x**2
        
        # Should still work with appropriate knots
        tck = bisplrep(x, y, z, kx=2, ky=1, s=0)
        assert len(tck[0]) >= 5  # At least 2*(2+1)-1 knots for quadratic
    
    def test_large_smoothing(self):
        """Test with very large smoothing parameter."""
        # Random data
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        
        # Large smoothing should give simple surface
        tck = bisplrep(x, y, z, s=1e6)
        
        # Should have minimal knots
        assert len(tck[0]) == 8  # Just boundary knots
        assert len(tck[1]) == 8


class TestBisplrepPerformance:
    """Performance-related tests."""
    
    def test_vectorization(self):
        """Test that evaluation is properly vectorized."""
        # Fit surface
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(2*np.pi*xx) * np.cos(2*np.pi*yy)
        
        tck = bisplrep(xx.ravel(), yy.ravel(), z.ravel(), s=0)
        
        # Test different evaluation modes
        x_eval = np.linspace(0, 1, 50)
        y_eval = np.linspace(0, 1, 50)
        
        # Meshgrid evaluation
        z_mesh = bisplev(x_eval, y_eval, tck)
        assert z_mesh.shape == (50, 50)
        
        # Pointwise evaluation (same length)
        z_point = bisplev(x_eval, y_eval, tck)
        assert z_point.shape == (50,)
        
        # Scalar evaluation
        z_scalar = bisplev(np.array([0.5]), np.array([0.5]), tck)
        assert z_scalar.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])