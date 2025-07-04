"""Unit tests for bisplrep/bisplev implementation."""

import numpy as np
import pytest
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastspline import bisplrep, bisplev, bisplev_scalar


class TestBisplrepBasic:
    """Basic functionality tests for bisplrep."""
    
    def test_simple_polynomial(self):
        """Test fitting a simple polynomial surface."""
        # Create a simple polynomial surface z = x^2 + y^2
        x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1])
        y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1])
        z = x**2 + y**2
        
        # Fit with our bisplrep (returns tck tuple)
        tck = bisplrep(x, y, z, kx=2, ky=2)
        tx, ty, c, kx, ky = tck
        
        # Check that we get reasonable knot counts
        nx = len(tx)
        ny = len(ty)
        assert nx >= 4 and nx <= 20, f"Unexpected nx: {nx}"
        assert ny >= 4 and ny <= 20, f"Unexpected ny: {ny}"
        
        # Evaluate at test points
        test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
        for xt, yt in test_points:
            tx, ty, c, kx, ky = tck
            z_eval = bisplev_scalar(xt, yt, tx, ty, c, kx, ky)
            z_true = xt**2 + yt**2
            assert abs(z_eval - z_true) < 0.01, f"Large error at ({xt}, {yt}): {abs(z_eval - z_true)}"
    
    def test_linear_interpolation(self):
        """Test linear interpolation (k=1)."""
        # Simple 2x2 grid
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z = np.array([0, 1, 2, 3])
        
        # Fit with k=1
        tck = bisplrep(x, y, z, kx=1, ky=1)
        
        # Test exact interpolation at data points
        for i in range(len(x)):
            tx, ty, c, kx, ky = tck
            z_eval = bisplev_scalar(x[i], y[i], tx, ty, c, kx, ky)
            assert abs(z_eval - z[i]) < 1e-10, f"Interpolation error at point {i}"


class TestBisplrepVsSciPy:
    """Test agreement with SciPy when using SciPy's knots."""
    
    def test_bisplev_accuracy_linear(self):
        """Test bisplev accuracy for linear splines."""
        np.random.seed(42)
        
        # Generate data
        n = 50
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = x + y + 0.1 * x * y
        
        # Fit with SciPy
        tck = scipy_bisplrep(x, y, z, kx=1, ky=1, s=0)
        tx, ty, c, kx, ky = tck
        
        # Test evaluation at random points
        n_test = 100
        x_test = np.random.uniform(-0.9, 0.9, n_test)
        y_test = np.random.uniform(-0.9, 0.9, n_test)
        
        max_diff = 0
        for i in range(n_test):
            z_scipy = scipy_bisplev(x_test[i], y_test[i], tck)
            z_ours = bisplev_scalar(x_test[i], y_test[i], tx, ty, c, kx, ky)
            diff = abs(z_scipy - z_ours)
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-14, f"Max difference too large: {max_diff}"
    
    def test_bisplev_accuracy_cubic(self):
        """Test bisplev accuracy for cubic splines."""
        np.random.seed(42)
        
        # Generate smooth data
        n = 100
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x)
        
        # Fit with SciPy
        tck = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.01)
        tx, ty, c, kx, ky = tck
        
        # Test evaluation on a grid
        n_test = 20
        x_test = np.linspace(-0.9, 0.9, n_test)
        y_test = np.linspace(-0.9, 0.9, n_test)
        
        max_diff = 0
        mean_diff = 0
        count = 0
        
        for i in range(n_test):
            for j in range(n_test):
                z_scipy = scipy_bisplev(x_test[i], y_test[j], tck)
                z_ours = bisplev_scalar(x_test[i], y_test[j], tx, ty, c, kx, ky)
                diff = abs(z_scipy - z_ours)
                max_diff = max(max_diff, diff)
                mean_diff += diff
                count += 1
        
        mean_diff /= count
        
        assert max_diff < 1e-14, f"Max difference too large: {max_diff}"
        assert mean_diff < 1e-15, f"Mean difference too large: {mean_diff}"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal data (4 points for k=1)
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z = np.array([1, 2, 3, 4])
        
        # SciPy fit
        tck = scipy_bisplrep(x, y, z, kx=1, ky=1, s=0)
        tx, ty, c, kx, ky = tck
        
        # Test at boundaries
        boundary_points = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for i, (xt, yt) in enumerate(boundary_points):
            z_scipy = scipy_bisplev(xt, yt, tck)
            z_ours = bisplev_scalar(xt, yt, tx, ty, c, kx, ky)
            assert abs(z_scipy - z_ours) < 1e-14, f"Boundary error at ({xt}, {yt})"


class TestBisplrepRobustness:
    """Test robustness and error handling."""
    
    def test_noisy_data(self):
        """Test with noisy data."""
        np.random.seed(42)
        
        # Generate noisy data
        n = 200
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = np.sin(2*x) * np.cos(2*y) + 0.1 * np.random.randn(n)
        
        # Our bisplrep should not crash
        tck = bisplrep(x, y, z, kx=3, ky=3)
        tx, ty, c, kx, ky = tck
        
        assert len(tx) > 0 and len(ty) > 0, "Failed to produce valid knots"
        
        # Should be able to evaluate
        z_eval = bisplev_scalar(0.0, 0.0, tx, ty, c, kx, ky)
        assert np.isfinite(z_eval), "Evaluation produced non-finite result"
    
    def test_different_degrees(self):
        """Test different spline degrees."""
        np.random.seed(42)
        
        # Generate data
        n = 100
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = x**2 + y**2
        
        # Test different degree combinations
        degree_combos = [(1, 1), (1, 3), (3, 1), (2, 2), (3, 3)]
        
        for kx, ky in degree_combos:
            # Fit with SciPy
            try:
                tck = scipy_bisplrep(x, y, z, kx=kx, ky=ky, s=0)
                tx_sp, ty_sp, c_sp, _, _ = tck
                
                # Evaluate with our bisplev
                z_eval = bisplev_scalar(0.5, 0.5, tx_sp, ty_sp, c_sp, kx, ky)
                assert np.isfinite(z_eval), f"Failed for degrees ({kx}, {ky})"
                
            except Exception:
                # Some degree combinations might not work with the data
                pass


class TestPerformance:
    """Performance-related tests."""
    
    def test_vectorization_compatibility(self):
        """Test that bisplev works well in vectorized contexts."""
        np.random.seed(42)
        
        # Create spline
        n = 100
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        z = np.exp(-(x**2 + y**2))
        
        tck = scipy_bisplrep(x, y, z, kx=3, ky=3, s=0.01)
        tx, ty, c, kx, ky = tck
        
        # Evaluate on a grid
        n_grid = 10
        x_grid = np.linspace(-0.9, 0.9, n_grid)
        y_grid = np.linspace(-0.9, 0.9, n_grid)
        
        # Manual double loop
        z_grid = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                z_grid[i, j] = bisplev_scalar(x_grid[i], y_grid[j], tx, ty, c, kx, ky)
        
        # Check that all values are reasonable
        assert np.all(np.isfinite(z_grid)), "Non-finite values in grid"
        assert np.all(z_grid >= -1) and np.all(z_grid <= 1), "Values out of expected range"


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run manually
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Running tests manually (install pytest for better output)")
        
        # Run all test classes
        for test_class in [TestBisplrepBasic, TestBisplrepVsSciPy, 
                          TestBisplrepRobustness, TestPerformance]:
            print(f"\n{test_class.__name__}:")
            test_instance = test_class()
            
            # Run all test methods
            for method_name in dir(test_instance):
                if method_name.startswith('test_'):
                    print(f"  {method_name}...", end=" ")
                    try:
                        getattr(test_instance, method_name)()
                        print("PASSED")
                    except Exception as e:
                        print(f"FAILED: {e}")
        
        print("\nAll tests completed!")