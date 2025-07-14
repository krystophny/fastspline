"""
Comprehensive test suite for parder derivative implementation.

This test suite validates that the fastspline parder implementation matches
scipy's dfitpack.parder exactly for all derivative orders using analytical
test functions with known derivatives.

TDD Approach: These tests are designed to FAIL initially, then guide the
implementation fixes until all derivatives work correctly.
"""
import numpy as np
import pytest
import warnings
from scipy.interpolate import bisplrep

try:
    from scipy.interpolate import dfitpack
    if not hasattr(dfitpack, 'parder'):
        dfitpack = None
except (ImportError, AttributeError):
    dfitpack = None

try:
    from fastspline.numba_implementation.parder import call_parder_safe
    PARDER_AVAILABLE = True
except ImportError:
    # Try adding current directory for direct test execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from fastspline.numba_implementation.parder import call_parder_safe
        PARDER_AVAILABLE = True
    except ImportError:
        PARDER_AVAILABLE = False


@pytest.mark.skipif(dfitpack is None, reason="dfitpack not available")
@pytest.mark.skipif(not PARDER_AVAILABLE, reason="fastspline parder not available")
class TestParderComprehensive:
    """Comprehensive tests for parder derivative implementation."""
    
    def test_parder_linear_derivatives(self):
        """Test derivatives of linear function f(x,y) = 2x + 3y."""
        # Analytical derivatives:
        # f(x,y) = 2x + 3y
        # df/dx = 2, df/dy = 3
        # d²f/dx² = 0, d²f/dy² = 0, d²f/dxdy = 0
        
        # Create test data
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = 2*X + 3*Y
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Test points
        test_points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        
        # Expected values (independent of x,y for linear function)
        expected = {
            (0, 0): lambda x, y: 2*x + 3*y,
            (1, 0): lambda x, y: 2.0,
            (0, 1): lambda x, y: 3.0,
            (2, 0): lambda x, y: 0.0,
            (0, 2): lambda x, y: 0.0,
            (1, 1): lambda x, y: 0.0,
        }
        
        for xi_val, yi_val in test_points:
            xi = np.array([xi_val])
            yi = np.array([yi_val])
            
            for (nux, nuy), expected_func in expected.items():
                expected_val = expected_func(xi_val, yi_val)
                
                # Get scipy result
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    z_scipy, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                
                # Get fastspline result
                c_arr = np.asarray(c, dtype=np.float64)
                z_fs, ier_fs = call_parder_safe(tx, ty, c_arr, 3, 3, nux, nuy, xi, yi)
                
                # Both should succeed
                assert ier == 0, f"scipy parder failed for derivative ({nux}, {nuy})"
                assert ier_fs == 0, f"fastspline parder failed for derivative ({nux}, {nuy})"
                
                # Results should match scipy
                assert abs(z_scipy[0,0] - z_fs[0]) < 1e-10, \
                    f"Derivative ({nux}, {nuy}) mismatch at ({xi_val}, {yi_val}): " \
                    f"scipy={z_scipy[0,0]:.10f}, fastspline={z_fs[0]:.10f}, " \
                    f"expected≈{expected_val:.10f}"
    
    def test_parder_quadratic_derivatives(self):
        """Test derivatives of quadratic function f(x,y) = x² + y²."""
        # Analytical derivatives:
        # f(x,y) = x² + y²
        # df/dx = 2x, df/dy = 2y
        # d²f/dx² = 2, d²f/dy² = 2, d²f/dxdy = 0
        
        # Create test data
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X**2 + Y**2
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Test points
        test_points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        
        # Expected values (functions of x,y)
        expected = {
            (0, 0): lambda x, y: x**2 + y**2,
            (1, 0): lambda x, y: 2*x,
            (0, 1): lambda x, y: 2*y,
            (2, 0): lambda x, y: 2.0,
            (0, 2): lambda x, y: 2.0,
            (1, 1): lambda x, y: 0.0,
        }
        
        for xi_val, yi_val in test_points:
            xi = np.array([xi_val])
            yi = np.array([yi_val])
            
            for (nux, nuy), expected_func in expected.items():
                expected_val = expected_func(xi_val, yi_val)
                
                # Get scipy result
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    z_scipy, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                
                # Get fastspline result
                c_arr = np.asarray(c, dtype=np.float64)
                z_fs, ier_fs = call_parder_safe(tx, ty, c_arr, 3, 3, nux, nuy, xi, yi)
                
                # Both should succeed
                assert ier == 0, f"scipy parder failed for derivative ({nux}, {nuy})"
                assert ier_fs == 0, f"fastspline parder failed for derivative ({nux}, {nuy})"
                
                # Results should match scipy (allowing for spline approximation error)
                tolerance = 1e-6 if nux + nuy > 0 else 1e-10
                assert abs(z_scipy[0,0] - z_fs[0]) < tolerance, \
                    f"Derivative ({nux}, {nuy}) mismatch at ({xi_val}, {yi_val}): " \
                    f"scipy={z_scipy[0,0]:.10f}, fastspline={z_fs[0]:.10f}, " \
                    f"expected≈{expected_val:.10f}"
    
    def test_parder_product_derivatives(self):
        """Test derivatives of product function f(x,y) = xy."""
        # Analytical derivatives:
        # f(x,y) = xy
        # df/dx = y, df/dy = x
        # d²f/dx² = 0, d²f/dy² = 0, d²f/dxdy = 1
        
        # Create test data
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X * Y
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Test points
        test_points = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)]
        
        # Expected values (functions of x,y)
        expected = {
            (0, 0): lambda x, y: x * y,
            (1, 0): lambda x, y: y,
            (0, 1): lambda x, y: x,
            (2, 0): lambda x, y: 0.0,
            (0, 2): lambda x, y: 0.0,
            (1, 1): lambda x, y: 1.0,
        }
        
        for xi_val, yi_val in test_points:
            xi = np.array([xi_val])
            yi = np.array([yi_val])
            
            for (nux, nuy), expected_func in expected.items():
                expected_val = expected_func(xi_val, yi_val)
                
                # Get scipy result
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    z_scipy, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                
                # Get fastspline result
                c_arr = np.asarray(c, dtype=np.float64)
                z_fs, ier_fs = call_parder_safe(tx, ty, c_arr, 3, 3, nux, nuy, xi, yi)
                
                # Both should succeed
                assert ier == 0, f"scipy parder failed for derivative ({nux}, {nuy})"
                assert ier_fs == 0, f"fastspline parder failed for derivative ({nux}, {nuy})"
                
                # Results should match scipy (allowing for spline approximation error)
                tolerance = 1e-6 if nux + nuy > 0 else 1e-10
                assert abs(z_scipy[0,0] - z_fs[0]) < tolerance, \
                    f"Derivative ({nux}, {nuy}) mismatch at ({xi_val}, {yi_val}): " \
                    f"scipy={z_scipy[0,0]:.10f}, fastspline={z_fs[0]:.10f}, " \
                    f"expected≈{expected_val:.10f}"
    
    def test_parder_cubic_derivatives(self):
        """Test derivatives of cubic function f(x,y) = x³ + y³."""
        # Analytical derivatives:
        # f(x,y) = x³ + y³
        # df/dx = 3x², df/dy = 3y²
        # d²f/dx² = 6x, d²f/dy² = 6y, d²f/dxdy = 0
        
        # Create test data
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X**3 + Y**3
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Test points
        test_points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        
        # Expected values (functions of x,y)
        expected = {
            (0, 0): lambda x, y: x**3 + y**3,
            (1, 0): lambda x, y: 3*x**2,
            (0, 1): lambda x, y: 3*y**2,
            (2, 0): lambda x, y: 6*x,
            (0, 2): lambda x, y: 6*y,
            (1, 1): lambda x, y: 0.0,
        }
        
        for xi_val, yi_val in test_points:
            xi = np.array([xi_val])
            yi = np.array([yi_val])
            
            for (nux, nuy), expected_func in expected.items():
                expected_val = expected_func(xi_val, yi_val)
                
                # Get scipy result
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    z_scipy, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                
                # Get fastspline result
                c_arr = np.asarray(c, dtype=np.float64)
                z_fs, ier_fs = call_parder_safe(tx, ty, c_arr, 3, 3, nux, nuy, xi, yi)
                
                # Both should succeed
                assert ier == 0, f"scipy parder failed for derivative ({nux}, {nuy})"
                assert ier_fs == 0, f"fastspline parder failed for derivative ({nux}, {nuy})"
                
                # Results should match scipy (allowing for spline approximation error)
                tolerance = 1e-5 if nux + nuy > 0 else 1e-10
                assert abs(z_scipy[0,0] - z_fs[0]) < tolerance, \
                    f"Derivative ({nux}, {nuy}) mismatch at ({xi_val}, {yi_val}): " \
                    f"scipy={z_scipy[0,0]:.10f}, fastspline={z_fs[0]:.10f}, " \
                    f"expected≈{expected_val:.10f}"
    
    def test_parder_multiple_points(self):
        """Test derivative computation with multiple evaluation points."""
        # Use linear function for exact results
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 6)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = 2*X + 3*Y
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Multiple test points
        xi = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        yi = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Test first derivatives
        for nux, nuy in [(1, 0), (0, 1)]:
            # Get scipy result
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                z_scipy, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
            
            # Get fastspline result
            c_arr = np.asarray(c, dtype=np.float64)
            z_fs, ier_fs = call_parder_safe(tx, ty, c_arr, 3, 3, nux, nuy, xi, yi)
            
            # Both should succeed
            assert ier == 0, f"scipy parder failed for derivative ({nux}, {nuy})"
            assert ier_fs == 0, f"fastspline parder failed for derivative ({nux}, {nuy})"
            
            # Results should match scipy for all points
            for i in range(len(xi)):
                assert abs(z_scipy[i,0] - z_fs[i]) < 1e-10, \
                    f"Derivative ({nux}, {nuy}) mismatch at point {i}: " \
                    f"scipy={z_scipy[i,0]:.10f}, fastspline={z_fs[i]:.10f}"
    
    def test_parder_different_degrees(self):
        """Test derivative computation with different spline degrees."""
        # Use quadratic function
        x = np.linspace(0, 1, 8)
        y = np.linspace(0, 1, 8)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X**2 + Y**2
        
        # Test different spline degrees
        degree_pairs = [(2, 2), (3, 3), (4, 4), (3, 4), (4, 3)]
        
        for kx, ky in degree_pairs:
            # Fit spline with specific degrees
            tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=kx, ky=ky, s=0.01)
            tx, ty, c = tck[0], tck[1], tck[2]
            
            # Test point
            xi = np.array([0.5])
            yi = np.array([0.5])
            
            # Test derivatives up to available order
            max_nux = min(kx, 2)
            max_nuy = min(ky, 2)
            
            for nux in range(max_nux + 1):
                for nuy in range(max_nuy + 1):
                    # Get scipy result
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', DeprecationWarning)
                        z_scipy, ier = dfitpack.parder(tx, ty, c, kx, ky, nux, nuy, xi, yi)
                    
                    # Get fastspline result
                    c_arr = np.asarray(c, dtype=np.float64)
                    z_fs, ier_fs = call_parder_safe(tx, ty, c_arr, kx, ky, nux, nuy, xi, yi)
                    
                    # Both should succeed
                    assert ier == 0, f"scipy parder failed for derivative ({nux}, {nuy}), degrees ({kx}, {ky})"
                    assert ier_fs == 0, f"fastspline parder failed for derivative ({nux}, {nuy}), degrees ({kx}, {ky})"
                    
                    # Results should match scipy
                    tolerance = 1e-6 if nux + nuy > 0 else 1e-10
                    assert abs(z_scipy[0,0] - z_fs[0]) < tolerance, \
                        f"Derivative ({nux}, {nuy}) mismatch for degrees ({kx}, {ky}): " \
                        f"scipy={z_scipy[0,0]:.10f}, fastspline={z_fs[0]:.10f}"


if __name__ == "__main__":
    # Run tests directly to see failures
    if not PARDER_AVAILABLE:
        print("❌ FastSpline parder not available - cannot run tests")
        exit(1)
    
    if dfitpack is None:
        print("❌ scipy dfitpack not available - cannot run tests")
        exit(1)
    
    test = TestParderComprehensive()
    print("Running comprehensive parder tests...")
    
    try:
        test.test_parder_linear_derivatives()
        print("✅ Linear derivatives test passed")
    except Exception as e:
        print(f"❌ Linear derivatives test failed: {e}")
    
    try:
        test.test_parder_quadratic_derivatives()
        print("✅ Quadratic derivatives test passed")
    except Exception as e:
        print(f"❌ Quadratic derivatives test failed: {e}")
    
    try:
        test.test_parder_product_derivatives()
        print("✅ Product derivatives test passed")
    except Exception as e:
        print(f"❌ Product derivatives test failed: {e}")
    
    try:
        test.test_parder_cubic_derivatives()
        print("✅ Cubic derivatives test passed")
    except Exception as e:
        print(f"❌ Cubic derivatives test failed: {e}")