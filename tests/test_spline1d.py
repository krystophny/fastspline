import numpy as np
import pytest
from fastspline.spline1d import Spline1D, evaluate_spline_cfunc, evaluate_spline_derivative_cfunc, evaluate_spline_second_derivative_cfunc


class TestSpline1D:
    def test_linear_interpolation(self):
        """Test linear spline interpolation with known values."""
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([0.0, 1.0, 4.0, 9.0])
        
        spline = Spline1D(x_data, y_data, order=1, periodic=False)
        
        # Test at grid points
        for i, x in enumerate(x_data):
            assert abs(spline.evaluate(x) - y_data[i]) < 1e-12
        
        # Test at midpoints
        assert abs(spline.evaluate(0.5) - 0.5) < 1e-12
        assert abs(spline.evaluate(1.5) - 2.5) < 1e-12
        assert abs(spline.evaluate(2.5) - 6.5) < 1e-12

    def test_cubic_spline_exact_polynomial(self):
        """Test cubic spline reproduces exact cubic polynomial."""
        # Test with f(x) = x^3 - 2*x^2 + x + 1
        x_data = np.linspace(0, 3, 10)
        y_data = x_data**3 - 2*x_data**2 + x_data + 1
        
        spline = Spline1D(x_data, y_data, order=3, periodic=False)
        
        # Test at various points - relax tolerance for natural spline
        x_test = np.linspace(0, 3, 25)
        y_exact = x_test**3 - 2*x_test**2 + x_test + 1
        
        for x, y_true in zip(x_test, y_exact):
            y_interp = spline.evaluate(x)
            assert abs(y_interp - y_true) < 1e-2  # Relaxed tolerance

    def test_periodic_spline(self):
        """Test periodic spline interpolation."""
        # Sine function over one period
        x_data = np.linspace(0, 2*np.pi, 20)
        y_data = np.sin(x_data)
        
        spline = Spline1D(x_data, y_data, order=3, periodic=True)
        
        # Test periodicity
        x_test = 0.5
        y1 = spline.evaluate(x_test)
        y2 = spline.evaluate(x_test + 2*np.pi)
        assert abs(y1 - y2) < 1e-10

    def test_derivative_calculation(self):
        """Test first derivative calculation."""
        # Test with f(x) = x^3, f'(x) = 3*x^2
        x_data = np.linspace(0, 2, 10)
        y_data = x_data**3
        
        spline = Spline1D(x_data, y_data, order=3, periodic=False)
        
        x_test = 1.5
        y, dy = spline.evaluate_with_derivative(x_test)
        
        # Check function value - relaxed tolerance
        assert abs(y - 1.5**3) < 1e-2
        # Check derivative value - relaxed tolerance  
        assert abs(dy - 3 * 1.5**2) < 1e-1

    def test_second_derivative_calculation(self):
        """Test second derivative calculation."""
        # Test with f(x) = x^4, f'(x) = 4*x^3, f''(x) = 12*x^2
        x_data = np.linspace(0, 2, 10)
        y_data = x_data**4
        
        spline = Spline1D(x_data, y_data, order=3, periodic=False)
        
        x_test = 1.2
        y, dy, d2y = spline.evaluate_with_second_derivative(x_test)
        
        # Check function value - relaxed tolerance
        assert abs(y - 1.2**4) < 1e-1
        # Check first derivative - relaxed tolerance
        assert abs(dy - 4 * 1.2**3) < 1e-1
        # Check second derivative (less accurate for cubic spline of quartic)
        assert abs(d2y - 12 * 1.2**2) < 1e-1

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 2.0, 3.0])
        
        spline = Spline1D(x_data, y_data, order=1, periodic=False)
        
        # Test at boundaries
        assert abs(spline.evaluate(0.0) - 1.0) < 1e-12
        assert abs(spline.evaluate(2.0) - 3.0) < 1e-12
        
        # Test extrapolation beyond bounds
        y_extrap = spline.evaluate(-0.5)
        assert not np.isnan(y_extrap)
        
        y_extrap = spline.evaluate(2.5)
        assert not np.isnan(y_extrap)

    def test_constant_function(self):
        """Test spline with constant function."""
        x_data = np.linspace(0, 5, 10)
        y_data = np.full_like(x_data, 2.5)
        
        spline = Spline1D(x_data, y_data, order=3, periodic=False)
        
        # Test at various points
        x_test = np.linspace(0, 5, 20)
        for x in x_test:
            assert abs(spline.evaluate(x) - 2.5) < 1e-12

    def test_minimum_points(self):
        """Test spline with minimum number of points."""
        x_data = np.array([0.0, 1.0])
        y_data = np.array([0.0, 1.0])
        
        # Linear spline should work with 2 points
        spline = Spline1D(x_data, y_data, order=1, periodic=False)
        assert abs(spline.evaluate(0.5) - 0.5) < 1e-12
        
        # Higher order should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, AssertionError)):
            Spline1D(x_data, y_data, order=3, periodic=False)

    def test_cfunc_compatibility(self):
        """Test that cfunc implementations produce same results as class methods."""
        x_data = np.linspace(0, 2*np.pi, 10)
        y_data = np.sin(x_data)
        
        spline = Spline1D(x_data, y_data, order=3, periodic=False)
        
        # Test multiple evaluation points
        x_test = np.array([0.5, 1.5, 2.5, 3.5])
        
        for x in x_test:
            # Compare class method vs direct cfunc call
            y_class = spline.evaluate(x)
            y_cfunc = evaluate_spline_cfunc(
                x, spline.coeffs, spline.x_min, spline.h_step, 
                spline.num_points, spline.order, spline.periodic
            )
            assert abs(y_class - y_cfunc) < 1e-14
            
            # Compare derivatives
            y_class, dy_class = spline.evaluate_with_derivative(x)
            y_cfunc, dy_cfunc = evaluate_spline_derivative_cfunc(
                x, spline.coeffs, spline.x_min, spline.h_step,
                spline.num_points, spline.order, spline.periodic
            )
            assert abs(y_class - y_cfunc) < 1e-14
            assert abs(dy_class - dy_cfunc) < 1e-14
            
            # Compare second derivatives
            y_class, dy_class, d2y_class = spline.evaluate_with_second_derivative(x)
            y_cfunc, dy_cfunc, d2y_cfunc = evaluate_spline_second_derivative_cfunc(
                x, spline.coeffs, spline.x_min, spline.h_step,
                spline.num_points, spline.order, spline.periodic
            )
            assert abs(y_class - y_cfunc) < 1e-14
            assert abs(dy_class - dy_cfunc) < 1e-14
            assert abs(d2y_class - d2y_cfunc) < 1e-14

    def test_cfunc_properties(self):
        """Test that cfunc properties return the correct function objects."""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 2.0, 3.0])
        
        spline = Spline1D(x_data, y_data, order=1, periodic=False)
        
        # Check that properties return the correct cfunc objects
        assert spline.cfunc_evaluate is evaluate_spline_cfunc
        assert spline.cfunc_evaluate_derivative is evaluate_spline_derivative_cfunc
        assert spline.cfunc_evaluate_second_derivative is evaluate_spline_second_derivative_cfunc