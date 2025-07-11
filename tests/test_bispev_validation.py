"""
Validation tests comparing the C wrapper output with scipy.interpolate.
"""
import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from bispev_ctypes import bispev as bispev_c
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate._fitpack_impl import bisplev as bisplev_impl


def test_simple_constant_spline():
    """Test evaluation of a constant spline."""
    # Create a simple constant spline
    kx = ky = 3
    tx = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    c = np.ones((len(tx)-kx-1) * (len(ty)-ky-1))
    
    x = np.linspace(0.1, 0.9, 5)
    y = np.linspace(0.1, 0.9, 5)
    
    # Our wrapper
    z_c = bispev_c(tx, ty, c, kx, ky, x, y)
    
    # Scipy's bisplev - need to create tck tuple
    tck = [tx, ty, c, kx, ky]
    z_scipy = bisplev(x, y, tck)
    
    np.testing.assert_allclose(z_c, z_scipy, rtol=1e-14, atol=1e-14)
    print("✓ Simple constant spline test passed")


def test_polynomial_surface():
    """Test with a polynomial surface."""
    # Generate data for a polynomial z = x^2 + y^2
    x_data = np.linspace(0, 2, 20)
    y_data = np.linspace(0, 2, 20)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = x_grid**2 + y_grid**2
    
    # Fit with bisplrep with some smoothing to avoid warning
    tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01)
    tx, ty, c, kx, ky = tck
    
    # Evaluation points
    x_eval = np.linspace(0.5, 1.5, 7)
    y_eval = np.linspace(0.5, 1.5, 7)
    
    # Our wrapper
    z_c = bispev_c(tx, ty, c, kx, ky, x_eval, y_eval)
    
    # Scipy's bisplev
    z_scipy = bisplev(x_eval, y_eval, tck)
    
    np.testing.assert_allclose(z_c, z_scipy, rtol=1e-14, atol=1e-14)
    print("✓ Polynomial surface test passed")


def test_random_data():
    """Test with random scattered data."""
    np.random.seed(42)
    
    # Generate random data on a grid to ensure successful fitting
    x_data = np.linspace(0, 1, 10)
    y_data = np.linspace(0, 1, 10)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = np.sin(2*np.pi*x_grid) * np.cos(2*np.pi*y_grid)
    
    # Fit with smoothing
    tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0.01)
    tx, ty, c, kx, ky = tck
    
    # Evaluation points
    x_eval = np.linspace(0.1, 0.9, 10)
    y_eval = np.linspace(0.1, 0.9, 10)
    
    # Our wrapper
    z_c = bispev_c(tx, ty, c, kx, ky, x_eval, y_eval)
    
    # Scipy's bisplev
    z_scipy = bisplev(x_eval, y_eval, tck)
    
    np.testing.assert_allclose(z_c, z_scipy, rtol=1e-14, atol=1e-14)
    print("✓ Random data test passed")


def test_edge_cases():
    """Test edge cases like single evaluation point."""
    # Create a simple spline with correct coefficient array size
    kx = ky = 3
    tx = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    ty = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    # Need (nx-kx-1)*(ny-ky-1) = (8-3-1)*(8-3-1) = 4*4 = 16 coefficients
    c = np.arange(16, dtype=float) + 1.0
    
    # Single evaluation point
    x = np.array([0.5])
    y = np.array([0.5])
    
    # Our wrapper
    z_c = bispev_c(tx, ty, c, kx, ky, x, y)
    
    # Scipy's bisplev
    tck = [tx, ty, c, kx, ky]
    z_scipy = bisplev(x, y, tck)
    
    np.testing.assert_allclose(z_c, z_scipy, rtol=1e-14, atol=1e-14)
    print("✓ Edge cases test passed")


def test_different_degrees():
    """Test with different spline degrees."""
    # Generate test data
    x_data = np.linspace(0, 1, 8)
    y_data = np.linspace(0, 1, 8)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = np.sin(2*np.pi*x_grid) * np.cos(2*np.pi*y_grid)
    
    for kx in [1, 3, 5]:
        for ky in [1, 3, 5]:
            # Fit with bisplrep
            tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), 
                          kx=kx, ky=ky, s=0)
            tx, ty, c, kx_out, ky_out = tck
            
            # Evaluation points
            x_eval = np.linspace(0.1, 0.9, 5)
            y_eval = np.linspace(0.1, 0.9, 5)
            
            # Our wrapper
            z_c = bispev_c(tx, ty, c, kx_out, ky_out, x_eval, y_eval)
            
            # Scipy's bisplev
            z_scipy = bisplev(x_eval, y_eval, tck)
            
            np.testing.assert_allclose(z_c, z_scipy, rtol=1e-13, atol=1e-13)
    
    print("✓ Different degrees test passed")


def test_large_grid():
    """Test with a larger evaluation grid."""
    # Create test data
    x_data = np.linspace(0, 1, 15)
    y_data = np.linspace(0, 1, 15)
    x_grid, y_grid = np.meshgrid(x_data, y_data)
    z_data = x_grid * y_grid
    
    # Fit
    tck = bisplrep(x_grid.ravel(), y_grid.ravel(), z_data.ravel(), s=0)
    tx, ty, c, kx, ky = tck
    
    # Large evaluation grid
    x_eval = np.linspace(0.05, 0.95, 50)
    y_eval = np.linspace(0.05, 0.95, 50)
    
    # Our wrapper
    z_c = bispev_c(tx, ty, c, kx, ky, x_eval, y_eval)
    
    # Scipy's bisplev
    z_scipy = bisplev(x_eval, y_eval, tck)
    
    np.testing.assert_allclose(z_c, z_scipy, rtol=1e-14, atol=1e-14)
    print("✓ Large grid test passed")


if __name__ == "__main__":
    print("Running validation tests...")
    test_simple_constant_spline()
    test_polynomial_surface()
    test_random_data()
    test_edge_cases()
    test_different_degrees()
    test_large_grid()
    print("\nAll validation tests passed! ✓")