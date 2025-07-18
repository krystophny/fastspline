#!/usr/bin/env python3
"""
Comprehensive pytest test suite for Sergei splines
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class TestCubicSplinesFortranPrecision:
    """Test cubic splines achieve Fortran-level precision"""
    
    def test_cubic_nonperiodic_reference_case(self):
        """Test cubic non-periodic splines match reference precision"""
        # Reference test case that gives 6.70e-17 error
        n = 10
        x = np.linspace(0, 1, n)
        y = np.sin(2*np.pi*x)
        h = 1.0 / (n - 1)
        x_test = 0.5
        
        # Construct and evaluate
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
        
        y_out = np.zeros(1)
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_spline - y_exact)
        
        # Should achieve double precision
        assert error < 1e-15, f"Cubic precision {error:.2e} not at double precision level"
    
    def test_cubic_periodic_boundary_precision(self):
        """Test cubic periodic splines have perfect periodicity"""
        n = 16
        x = np.linspace(0, 2*np.pi, n, endpoint=False)
        y = np.sin(x)
        h = 2*np.pi / n
        
        # Construct periodic spline
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
        
        # Test periodicity at multiple points
        test_pairs = [(0.5, 0.5 + 2*np.pi), (1.0, 1.0 + 2*np.pi), (2.5, 2.5 + 2*np.pi)]
        
        y_out = np.zeros(1)
        for x1, x2 in test_pairs:
            evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x1, y_out)
            y1 = y_out[0]
            evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, x2, y_out)
            y2 = y_out[0]
            error = abs(y1 - y2)
            
            assert error < 1e-14, f"Periodicity error {error:.2e} not at machine precision"
    
    def test_cubic_evaluation_accuracy(self):
        """Test cubic splines achieve reasonable evaluation accuracy"""
        n = 15
        x = np.linspace(0, 1, n)
        y = np.exp(-x) * np.cos(3*x)
        h = 1.0 / (n - 1)
        
        # Construct spline
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
        
        # Test at multiple points
        test_points = np.linspace(0.1, 0.9, 10)
        max_error = 0.0
        
        y_out = np.zeros(1)
        for x_test in test_points:
            evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x_test, y_out)
            y_spline = y_out[0]
            y_exact = np.exp(-x_test) * np.cos(3*x_test)
            error = abs(y_spline - y_exact)
            max_error = max(max_error, error)
        
        # Should achieve reasonable spline accuracy
        assert max_error < 1e-2, f"Cubic evaluation error {max_error:.2e} too large"

class TestQuinticSplinesFortranPrecision:
    """Test quintic splines work correctly"""
    
    def test_quintic_nonperiodic_reference_case(self):
        """Test quintic non-periodic splines match reference behavior"""
        # Reference test case that gives ~1.17e-04 error
        n = 20
        x = np.linspace(0, 1, n)
        y = np.sin(2*np.pi*x)
        h = 1.0 / (n - 1)
        x_test = 0.5
        
        # Construct and evaluate
        coeff = np.zeros(6*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
        
        y_out = np.zeros(1)
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        y_spline = y_out[0]
        y_exact = np.sin(2*np.pi*x_test)
        error = abs(y_spline - y_exact)
        
        # Should achieve reference precision level
        assert error < 1e-3, f"Quintic precision {error:.2e} worse than reference"
    
    def test_quintic_construction_stability(self):
        """Test quintic splines construct without errors"""
        test_cases = [
            (10, 0, 1),    # Small domain
            (15, 0, 1),    # Medium domain  
            (20, 0, 1),    # Reference size
            (25, 0, 2),    # Larger domain
        ]
        
        for n, x_min, x_max in test_cases:
            x = np.linspace(x_min, x_max, n)
            y = np.sin(x) + 0.1*x**2
            
            # Should construct without raising exceptions
            coeff = np.zeros(6*n, dtype=np.float64)
            construct_splines_1d_cfunc(x_min, x_max, y, n, 5, False, coeff)
            
            # Basic sanity check - coefficients should be finite
            assert np.all(np.isfinite(coeff)), f"Non-finite coefficients for n={n}"

@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestSciPyComparison:
    """Test Sergei splines are competitive with SciPy"""
    
    def test_cubic_vs_scipy_smooth_function(self):
        """Test cubic splines match SciPy performance on smooth functions"""
        # Smooth test function
        def f(x):
            return np.exp(-x) * np.cos(4*x) + 0.1*x
        
        n = 20
        x_data = np.linspace(0, 2, n)
        y_data = f(x_data)
        h = 2.0 / (n - 1)
        
        # Sergei spline
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2.0, y_data, n, 3, False, coeff)
        
        # SciPy spline
        scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
        
        # Compare at test points
        x_test = np.linspace(0.1, 1.9, 20)
        sergei_errors = []
        scipy_errors = []
        
        y_out = np.zeros(1)
        for x in x_test:
            # Sergei
            evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x, y_out)
            sergei_error = abs(y_out[0] - f(x))
            sergei_errors.append(sergei_error)
            
            # SciPy
            scipy_error = abs(scipy_spline(x) - f(x))
            scipy_errors.append(scipy_error)
        
        sergei_max = np.max(sergei_errors)
        scipy_max = np.max(scipy_errors)
        
        # Sergei should be competitive (within factor of 3)
        ratio = sergei_max / scipy_max if scipy_max > 0 else 1.0
        assert ratio < 3.0, f"Sergei cubic {ratio:.1f}x worse than SciPy"
    
    def test_cubic_vs_scipy_identical_results(self):
        """Test cubic splines give identical results to SciPy on simple cases"""
        # Simple polynomial data
        x_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        y_data = np.array([1.0, 1.25, 2.0, 3.25, 5.0])  # Quadratic
        n = len(x_data)
        h = 0.5
        
        # Sergei spline
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2.0, y_data, n, 3, False, coeff)
        
        # SciPy spline
        scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
        
        # Should give very similar results
        x_test = [0.25, 0.75, 1.25, 1.75]
        
        y_out = np.zeros(1)
        for x in x_test:
            evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, x, y_out)
            sergei_val = y_out[0]
            scipy_val = scipy_spline(x)
            
            diff = abs(sergei_val - scipy_val)
            assert diff < 1e-12, f"Large difference {diff:.2e} at x={x}"

class TestSplineStability:
    """Test spline stability and edge cases"""
    
    def test_quartic_enabled(self):
        """Test that quartic splines work correctly"""
        n = 10
        x = np.linspace(0, 1, n)
        y = np.sin(x)
        coeff = np.zeros(5*n, dtype=np.float64)
        
        # Should not raise an error
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 4, False, coeff)
        
        # Should evaluate without error
        h = 1.0 / (n - 1)
        y_out = np.zeros(1)
        evaluate_splines_1d_cfunc(4, n, False, 0.0, h, coeff, 0.5, y_out)
        
        # Should give reasonable result
        expected = np.sin(0.5)
        error = abs(y_out[0] - expected)
        assert error < 0.1, f"Quartic evaluation error {error:.2e} too large"
    
    def test_quartic_periodic_enabled(self):
        """Test that quartic periodic splines work correctly"""
        n = 8
        x_min = 0.0
        x_max = 1.0
        h = (x_max - x_min) / n
        x = np.array([x_min + i * h for i in range(n)])
        y = np.sin(2*np.pi*x)  # Periodic function
        
        coeff = np.zeros(5*n, dtype=np.float64)
        
        # Should not raise an error
        construct_splines_1d_cfunc(x_min, x_max, y, n, 4, True, coeff)
        
        # Should evaluate at grid points with machine precision
        y_out = np.zeros(1)
        for i in range(n):
            evaluate_splines_1d_cfunc(4, n, True, x_min, h, coeff, x[i], y_out)
            error = abs(y_out[0] - y[i])
            assert error < 1e-12, f"Grid point {i} error {error:.2e} too large"
        
        # Should have perfect periodicity
        evaluate_splines_1d_cfunc(4, n, True, x_min, h, coeff, x_max, y_out)
        error = abs(y_out[0] - y[0])
        assert error < 1e-12, f"Periodicity error {error:.2e} too large"
    
    def test_minimum_points_cubic(self):
        """Test cubic splines work with minimum number of points"""
        n = 4  # Minimum for cubic
        x = np.linspace(0, 1, n)
        y = x**2  # Simple quadratic
        h = 1.0 / (n - 1)
        
        # Should work without errors
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, False, coeff)
        
        # Should evaluate reasonably
        y_out = np.zeros(1)
        evaluate_splines_1d_cfunc(3, n, False, 0.0, h, coeff, 0.5, y_out)
        
        # Should be close to exact value for quadratic
        expected = 0.25
        error = abs(y_out[0] - expected)
        assert error < 0.1, f"Large error {error:.2e} with minimum points"
    
    def test_cfunc_addresses_available(self):
        """Test that all cfunc addresses are accessible"""
        from fastspline.sergei_splines import get_cfunc_addresses
        
        addresses = get_cfunc_addresses()
        
        expected_funcs = [
            'construct_splines_1d',
            'evaluate_splines_1d', 
            'evaluate_splines_1d_der',
            'evaluate_splines_1d_der2',
            'construct_splines_2d',
            'evaluate_splines_2d',
            'evaluate_splines_2d_der',
            'construct_splines_3d',
            'evaluate_splines_3d',
            'evaluate_splines_3d_der',
        ]
        
        for func_name in expected_funcs:
            assert func_name in addresses, f"Missing cfunc address for {func_name}"
            assert isinstance(addresses[func_name], int), f"Invalid address for {func_name}"
            assert addresses[func_name] > 0, f"Zero address for {func_name}"

class TestPeriodicSplines:
    """Test periodic spline functionality"""
    
    def test_periodic_cubic_perfect_periodicity(self):
        """Test periodic cubic splines achieve perfect periodicity"""
        n = 16
        x = np.linspace(0, 2*np.pi, n, endpoint=False)
        y = np.cos(x) + 0.3*np.sin(3*x)
        h = 2*np.pi / n
        
        # Construct periodic spline
        coeff = np.zeros(4*n, dtype=np.float64)
        construct_splines_1d_cfunc(0.0, 2*np.pi, y, n, 3, True, coeff)
        
        # Test periodicity at many points
        test_offsets = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
        
        y_out = np.zeros(1)
        for offset in test_offsets:
            # Evaluate at x and x + 2π  
            evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, offset, y_out)
            y1 = y_out[0]
            
            evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff, offset + 2*np.pi, y_out)
            y2 = y_out[0]
            
            periodicity_error = abs(y1 - y2)
            assert periodicity_error < 1e-14, f"Periodicity error {periodicity_error:.2e} at offset {offset}"
    
    def test_periodic_vs_nonperiodic_boundary_behavior(self):
        """Test that periodic splines handle boundaries differently than non-periodic"""
        n = 8
        x = np.linspace(0, 1, n, endpoint=False)
        y = np.sin(2*np.pi*x)
        h = 1.0 / n
        
        # Construct both types
        coeff_periodic = np.zeros(4*n, dtype=np.float64)
        coeff_nonperiodic = np.zeros(4*n, dtype=np.float64)
        
        construct_splines_1d_cfunc(0.0, 1.0, y, n, 3, True, coeff_periodic)
        
        # Non-periodic needs endpoint included
        x_nonper = np.linspace(0, 1, n)
        y_nonper = np.sin(2*np.pi*x_nonper)
        construct_splines_1d_cfunc(0.0, 1.0, y_nonper, n, 3, False, coeff_nonperiodic)
        
        # Evaluate near boundaries
        test_points = [0.05, 0.95]
        
        y_out = np.zeros(1)
        for x_test in test_points:
            # Both should evaluate without error but may give different results
            evaluate_splines_1d_cfunc(3, n, True, 0.0, h, coeff_periodic, x_test, y_out)
            y_periodic = y_out[0]
            
            h_nonper = 1.0 / (n - 1)
            evaluate_splines_1d_cfunc(3, n, False, 0.0, h_nonper, coeff_nonperiodic, x_test, y_out)
            y_nonperiodic = y_out[0]
            
            # Both should be finite
            assert np.isfinite(y_periodic), f"Non-finite periodic result at x={x_test}"
            assert np.isfinite(y_nonperiodic), f"Non-finite non-periodic result at x={x_test}"

# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])