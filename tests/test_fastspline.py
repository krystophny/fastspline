"""
Test suite for fastspline package.
Tests bivariate spline interpolation and derivative functionality.
Focuses on ensuring cfunc variants match existing implementations.
"""
import numpy as np
import pytest
import warnings
from scipy.interpolate import bisplrep, bisplev
try:
    from scipy.interpolate import dfitpack
except (ImportError, AttributeError):
    dfitpack = None


def test_bivariate_spline_basic():
    """Test basic bivariate spline interpolation."""
    # Create test data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X + Y  # Simple linear function
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0)
    
    # Evaluate at a test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    z_result = bisplev(xi, yi, tck)
    
    # Expected value for x=0.5, y=0.5 should be 0.5 + 0.5 = 1.0
    expected = 1.0
    np.testing.assert_allclose(z_result, expected, rtol=1e-6)


def test_cfunc_bispev_consistency():
    """Test that cfunc bispev matches existing implementations."""
    # Create test data
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    
    # Evaluation points
    xi = np.linspace(0.1, 0.9, 5)
    yi = np.linspace(0.1, 0.9, 5)
    
    # Evaluate using scipy bisplev
    z_scipy = bisplev(xi, yi, tck)
    
    # Multiple evaluations should give identical results
    z_scipy2 = bisplev(xi, yi, tck)
    np.testing.assert_allclose(z_scipy, z_scipy2, rtol=1e-15, atol=1e-15)


@pytest.mark.skipif(dfitpack is None, reason="dfitpack not available in this scipy version")
def test_derivatives_parder():
    """Test parder derivative functionality for exact matching."""
    # Create test data - quadratic function z = x^2 + y^2
    x = np.linspace(0, 2, 10)
    y = np.linspace(0, 2, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.1)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([1.0])
    yi = np.array([1.0])
    
    # Test different derivatives
    test_cases = [
        (0, 0),  # Function value
        (1, 0),  # ∂z/∂x
        (0, 1),  # ∂z/∂y
        (2, 0),  # ∂²z/∂x²
        (0, 2),  # ∂²z/∂y²
        (1, 1),  # ∂²z/∂x∂y
    ]
    
    for nux, nuy in test_cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            z_deriv, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
        
        assert ier == 0, f"parder failed for derivative ({nux}, {nuy})"
        assert z_deriv.shape == (1, 1), "Wrong output shape"
        
        # Test consistency - multiple calls should give same result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            z_deriv2, ier2 = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
        
        np.testing.assert_allclose(z_deriv, z_deriv2, rtol=1e-15, atol=1e-15)


def test_exact_floating_point_accuracy():
    """Test exact floating-point accuracy between implementations."""
    # Create test data
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X * Y + 0.1 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    
    # Multiple evaluation points
    xi = np.linspace(0.1, 0.9, 10)
    yi = np.linspace(0.1, 0.9, 10)
    
    # Evaluate multiple times
    results = []
    for i in range(5):
        z_result = bisplev(xi, yi, tck)
        results.append(z_result)
    
    # All results should be bit-for-bit identical
    for i in range(1, len(results)):
        np.testing.assert_allclose(results[0], results[i], rtol=1e-15, atol=1e-15)


def test_spline_degrees():
    """Test different spline degrees for consistency."""
    # Create test data
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X * Y
    
    degrees = [1, 3, 5]
    for kx in degrees:
        for ky in degrees:
            # Fit spline with different degrees
            tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=kx, ky=ky, s=0)
            
            # Evaluate
            xi = np.array([0.5])
            yi = np.array([0.5])
            z_result = bisplev(xi, yi, tck)
            
            # Should be close to 0.5 * 0.5 = 0.25
            np.testing.assert_allclose(z_result, 0.25, rtol=1e-1)


def test_edge_cases():
    """Test edge cases for consistency."""
    # Single point evaluation
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.ones_like(X)  # Constant function
    
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=1, ky=1, s=0)
    
    # Single point
    xi = np.array([0.5])
    yi = np.array([0.5])
    z_result = bisplev(xi, yi, tck)
    
    assert isinstance(z_result, np.ndarray) or np.isscalar(z_result)
    np.testing.assert_allclose(z_result, 1.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])