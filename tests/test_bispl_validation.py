"""
Comprehensive validation tests for bispl implementation against scipy.interpolate.bisplev.

This test suite validates that fastspline's bisplev_cfunc matches scipy's bisplev
to floating point accuracy.
"""

import numpy as np
import pytest
from scipy import interpolate
from fastspline import Spline2D, bisplev_scalar


class TestBisplValidation:
    """Validation tests for bispl implementation accuracy."""
    
    def test_exact_polynomial_reproduction(self):
        """Test that bisplev reproduces polynomials exactly up to degree kx,ky."""
        # Test polynomials of various degrees
        degrees = [(1, 1), (2, 2), (3, 3)]
        
        for kx, ky in degrees:
            # Create polynomial test function
            x = np.linspace(0, 1, 10)
            y = np.linspace(0, 1, 10)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            # Test polynomial: sum of x^i * y^j for i <= kx, j <= ky
            Z = np.zeros_like(X)
            for i in range(kx + 1):
                for j in range(ky + 1):
                    Z += (i + j + 1) * X**i * Y**j
            
            # Create scipy spline
            tck = interpolate.bisplrep(X.ravel(), Y.ravel(), Z.ravel(), 
                                     kx=kx, ky=ky, s=0)
            
            # Test at random points
            np.random.seed(42)
            n_test = 100
            x_test = np.random.uniform(0.1, 0.9, n_test)
            y_test = np.random.uniform(0.1, 0.9, n_test)
            
            for xi, yi in zip(x_test, y_test):
                # Scipy reference value
                scipy_val = interpolate.bisplev(xi, yi, tck)
                
                # Our implementation
                our_val = bisplev_cfunc(xi, yi, tck[0], tck[1], tck[2], 
                                      kx, ky, len(tck[0]), len(tck[1]))
                
                # Should match to machine precision for polynomials
                assert np.abs(scipy_val - our_val) < 1e-13, \
                    f"Polynomial reproduction failed at ({xi}, {yi}): " \
                    f"scipy={scipy_val}, ours={our_val}, diff={abs(scipy_val - our_val)}"
    
    def test_smooth_function_approximation(self):
        """Test approximation of smooth functions with various smoothing parameters."""
        # Test function: Gaussian-like
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.exp(-(X**2 + Y**2))
        
        # Test different smoothing values
        smoothing_vals = [0, 0.01, 0.1]
        
        for s in smoothing_vals:
            # Create scipy spline
            tck = interpolate.bisplrep(X.ravel(), Y.ravel(), Z.ravel(), 
                                     kx=3, ky=3, s=s)
            
            # Test on a fine grid
            x_test = np.linspace(-1.5, 1.5, 30)
            y_test = np.linspace(-1.5, 1.5, 30)
            
            max_diff = 0
            rel_errors = []
            
            for xi in x_test:
                for yi in y_test:
                    scipy_val = interpolate.bisplev(xi, yi, tck)
                    our_val = bisplev_cfunc(xi, yi, tck[0], tck[1], tck[2], 
                                          3, 3, len(tck[0]), len(tck[1]))
                    
                    diff = abs(scipy_val - our_val)
                    max_diff = max(max_diff, diff)
                    
                    if abs(scipy_val) > 1e-10:
                        rel_errors.append(diff / abs(scipy_val))
            
            # Check absolute and relative errors
            assert max_diff < 1e-12, \
                f"Max absolute error too large for s={s}: {max_diff}"
            
            if rel_errors:
                max_rel_error = max(rel_errors)
                assert max_rel_error < 1e-12, \
                    f"Max relative error too large for s={s}: {max_rel_error}"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimal spline (2x2 grid)
        x = np.array([0, 1])
        y = np.array([0, 1])
        z = np.array([[1, 2], [3, 4]]).ravel()
        
        tck = interpolate.bisplrep(np.array([0, 0, 1, 1]), 
                                 np.array([0, 1, 0, 1]), 
                                 z, kx=1, ky=1, s=0)
        
        # Test at corners
        corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
        expected = [1, 2, 3, 4]
        
        for (xi, yi), expected_val in zip(corners, expected):
            scipy_val = interpolate.bisplev(xi, yi, tck)
            our_val = bisplev_cfunc(xi, yi, tck[0], tck[1], tck[2], 
                                  1, 1, len(tck[0]), len(tck[1]))
            
            assert np.abs(scipy_val - expected_val) < 1e-12
            assert np.abs(our_val - scipy_val) < 1e-12
    
    def test_derivative_consistency(self):
        """Test that derivatives are consistent with scipy."""
        # Smooth test function
        x = np.linspace(0, 2*np.pi, 15)
        y = np.linspace(0, 2*np.pi, 15)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.sin(X) * np.cos(Y)
        
        tck = interpolate.bisplrep(X.ravel(), Y.ravel(), Z.ravel(), 
                                 kx=3, ky=3, s=0)
        
        # Test derivatives at random points
        np.random.seed(123)
        n_test = 50
        x_test = np.random.uniform(0.5, 5.5, n_test)
        y_test = np.random.uniform(0.5, 5.5, n_test)
        
        for xi, yi in zip(x_test, y_test):
            # Function value
            scipy_val = interpolate.bisplev(xi, yi, tck)
            our_val = bisplev_cfunc(xi, yi, tck[0], tck[1], tck[2], 
                                  3, 3, len(tck[0]), len(tck[1]))
            
            assert np.abs(scipy_val - our_val) < 1e-12
            
            # First derivatives (when implemented)
            # scipy_dx = interpolate.bisplev(xi, yi, tck, dx=1, dy=0)
            # scipy_dy = interpolate.bisplev(xi, yi, tck, dx=0, dy=1)
            # Compare with our derivatives...
    
    def test_periodic_splines(self):
        """Test periodic boundary conditions."""
        # Periodic function
        n = 20
        x = np.linspace(0, 2*np.pi, n, endpoint=False)
        y = np.linspace(0, 2*np.pi, n, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.sin(X) + np.cos(Y)
        
        # Note: scipy's bisplrep doesn't directly support periodic splines
        # but we should test our implementation's handling of periodic data
        
        # Create spline with our implementation
        spline = Spline2D(x, y, Z.ravel(), kx=3, ky=3, periodic=(True, True))
        
        # Test periodicity: f(0, y) should equal f(2π, y)
        y_test = np.linspace(0, 2*np.pi, 10)
        for yi in y_test:
            val_0 = spline(0, yi, grid=False)
            val_2pi = spline(2*np.pi, yi, grid=False)
            assert np.abs(val_0 - val_2pi) < 1e-12
    
    def test_numerical_stability(self):
        """Test numerical stability with ill-conditioned data."""
        # Test with data at different scales
        scales = [1e-6, 1.0, 1e6]
        
        for scale in scales:
            x = np.linspace(0, 1, 10) * scale
            y = np.linspace(0, 1, 10) * scale
            X, Y = np.meshgrid(x, y, indexing='ij')
            Z = (X/scale)**2 + (Y/scale)**2
            
            tck = interpolate.bisplrep(X.ravel(), Y.ravel(), Z.ravel(), 
                                     kx=3, ky=3, s=0)
            
            # Test at center point
            x_mid = 0.5 * scale
            y_mid = 0.5 * scale
            
            scipy_val = interpolate.bisplev(x_mid, y_mid, tck)
            our_val = bisplev_cfunc(x_mid, y_mid, tck[0], tck[1], tck[2], 
                                  3, 3, len(tck[0]), len(tck[1]))
            
            expected = 0.5  # (0.5)^2 + (0.5)^2
            
            # Check both implementations are close to expected
            assert np.abs(scipy_val - expected) < 1e-10
            assert np.abs(our_val - scipy_val) < 1e-12


def generate_validation_report():
    """Generate a detailed validation report comparing our implementation with scipy."""
    print("BISPL Validation Report")
    print("=" * 50)
    
    # Run all tests and collect results
    test_results = {
        'polynomial_reproduction': [],
        'smooth_approximation': [],
        'edge_cases': [],
        'derivatives': [],
        'periodic': [],
        'stability': []
    }
    
    # TODO: Run tests and collect detailed metrics
    # This would include:
    # - Maximum absolute errors
    # - RMS errors
    # - Relative errors
    # - Performance comparisons
    # - Memory usage
    
    return test_results


if __name__ == "__main__":
    # Run validation suite
    pytest.main([__file__, "-v"])
    
    # Generate report
    # report = generate_validation_report()