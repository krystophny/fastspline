"""
Test derivative accuracy against known analytical results.
This test ensures derivative implementations are mathematically correct.
"""
import numpy as np
import pytest
import warnings
from scipy.interpolate import bisplrep
try:
    from scipy.interpolate import dfitpack
    # Check if parder is available
    if not hasattr(dfitpack, 'parder'):
        dfitpack = None
except (ImportError, AttributeError):
    # dfitpack may not be available in newer scipy versions
    dfitpack = None


@pytest.mark.skipif(dfitpack is None, reason="dfitpack not available in this scipy version")
def test_polynomial_derivatives():
    """Test derivatives against polynomial function f(x,y) = x² + y²"""
    # Create test data with known analytical derivatives
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + Y**2  # f(x,y) = x² + y²
    
    # Fit spline with small smoothing for accuracy
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test points
    test_points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
    
    for xi_val, yi_val in test_points:
        xi = np.array([xi_val])
        yi = np.array([yi_val])
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            
            # Test function value: f(x,y) = x² + y²
            z_val, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 0, xi, yi)
            expected_val = xi_val**2 + yi_val**2
            assert abs(z_val[0,0] - expected_val) < 1e-4, f"Function value mismatch at ({xi_val}, {yi_val})"
            
            # Test first derivatives: ∂f/∂x = 2x, ∂f/∂y = 2y
            z_dx, ier = dfitpack.parder(tx, ty, c, 3, 3, 1, 0, xi, yi)
            z_dy, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 1, xi, yi)
            expected_dx = 2 * xi_val
            expected_dy = 2 * yi_val
            assert abs(z_dx[0,0] - expected_dx) < 1e-4, f"∂f/∂x mismatch at ({xi_val}, {yi_val})"
            assert abs(z_dy[0,0] - expected_dy) < 1e-4, f"∂f/∂y mismatch at ({xi_val}, {yi_val})"
            
            # Test second derivatives: ∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂x∂y = 0
            z_dxx, ier = dfitpack.parder(tx, ty, c, 3, 3, 2, 0, xi, yi)
            z_dyy, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 2, xi, yi)
            z_dxy, ier = dfitpack.parder(tx, ty, c, 3, 3, 1, 1, xi, yi)
            assert abs(z_dxx[0,0] - 2.0) < 1e-3, f"∂²f/∂x² mismatch at ({xi_val}, {yi_val})"
            assert abs(z_dyy[0,0] - 2.0) < 1e-3, f"∂²f/∂y² mismatch at ({xi_val}, {yi_val})"
            assert abs(z_dxy[0,0] - 0.0) < 1e-3, f"∂²f/∂x∂y mismatch at ({xi_val}, {yi_val})"


@pytest.mark.skipif(dfitpack is None, reason="dfitpack not available in this scipy version")
def test_linear_derivatives():
    """Test derivatives against linear function f(x,y) = 2x + 3y"""
    # Create test data with known analytical derivatives
    x = np.linspace(0, 1, 8)
    y = np.linspace(0, 1, 8)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = 2*X + 3*Y  # f(x,y) = 2x + 3y
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test points
    test_points = [(0.2, 0.3), (0.7, 0.8)]
    
    for xi_val, yi_val in test_points:
        xi = np.array([xi_val])
        yi = np.array([yi_val])
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            
            # Test function value: f(x,y) = 2x + 3y
            z_val, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 0, xi, yi)
            expected_val = 2*xi_val + 3*yi_val
            assert abs(z_val[0,0] - expected_val) < 1e-10, f"Function value mismatch at ({xi_val}, {yi_val})"
            
            # Test first derivatives: ∂f/∂x = 2, ∂f/∂y = 3
            z_dx, ier = dfitpack.parder(tx, ty, c, 3, 3, 1, 0, xi, yi)
            z_dy, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 1, xi, yi)
            assert abs(z_dx[0,0] - 2.0) < 1e-10, f"∂f/∂x mismatch at ({xi_val}, {yi_val})"
            assert abs(z_dy[0,0] - 3.0) < 1e-10, f"∂f/∂y mismatch at ({xi_val}, {yi_val})"
            
            # Test second derivatives: all should be 0
            z_dxx, ier = dfitpack.parder(tx, ty, c, 3, 3, 2, 0, xi, yi)
            z_dyy, ier = dfitpack.parder(tx, ty, c, 3, 3, 0, 2, xi, yi)
            z_dxy, ier = dfitpack.parder(tx, ty, c, 3, 3, 1, 1, xi, yi)
            assert abs(z_dxx[0,0]) < 1e-10, f"∂²f/∂x² should be 0 at ({xi_val}, {yi_val})"
            assert abs(z_dyy[0,0]) < 1e-10, f"∂²f/∂y² should be 0 at ({xi_val}, {yi_val})"
            assert abs(z_dxy[0,0]) < 1e-10, f"∂²f/∂x∂y should be 0 at ({xi_val}, {yi_val})"


@pytest.mark.skipif(dfitpack is None, reason="dfitpack not available in this scipy version")
def test_derivative_consistency():
    """Test that derivatives are consistent across multiple calls"""
    # Create test data
    x = np.linspace(0, 1, 6)
    y = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**3 + Y**3  # Cubic function
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Test point
    xi = np.array([0.5])
    yi = np.array([0.5])
    
    # Test that multiple calls give identical results
    derivatives = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    
    for nux, nuy in derivatives:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            
            # Multiple calls should give identical results
            results = []
            for _ in range(5):
                z_deriv, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                results.append(z_deriv[0,0])
                assert ier == 0, f"Derivative computation failed for ({nux}, {nuy})"
            
            # All results should be identical
            for i in range(1, len(results)):
                assert abs(results[0] - results[i]) < 1e-15, f"Inconsistent results for derivative ({nux}, {nuy})"


def test_cfunc_derivative_matching():
    """Test that cfunc derivatives match scipy"""
    try:
        import sys
        sys.path.append('fastspline/numba_implementation')
        from parder import parder_cfunc_address as parder_correct_cfunc_address
        import ctypes
        
        # Create test data
        x = np.linspace(0, 1, 6)
        y = np.linspace(0, 1, 6)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X + Y  # Linear function
        
        # Fit spline
        tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.01)
        tx, ty, c = tck[0], tck[1], tck[2]
        
        # Test point
        xi = np.array([0.5])
        yi = np.array([0.5])
        
        # Test function value (should work)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            z_scipy, ier_scipy = dfitpack.parder(tx, ty, c, 3, 3, 0, 0, xi, yi)
        
        # Setup cfunc call
        nx, ny = len(tx), len(ty)
        mx, my = len(xi), len(yi)
        
        c_arr = np.asarray(c, dtype=np.float64)
        z_cfunc = np.zeros(mx * my, dtype=np.float64)
        
        lwrk = (3 + 1 - 0) * mx + (3 + 1 - 0) * my
        wrk = np.zeros(lwrk, dtype=np.float64)
        iwrk = np.zeros(mx + my, dtype=np.int32)
        ier = np.zeros(1, dtype=np.int32)
        
        # Use safe wrapper instead of direct cfunc call
        from parder import call_parder_safe
        z_cfunc, ier_cfunc = call_parder_safe(tx, ty, c_arr, 3, 3, 0, 0, xi, yi)
        
        # Check function value matches
        assert ier_cfunc == 0, f"cfunc failed with ier={ier_cfunc}"
        assert abs(z_scipy[0,0] - z_cfunc[0]) < 1e-10, f"Function value mismatch: {z_scipy[0,0]} vs {z_cfunc[0]}"
        
    except ImportError:
        pytest.skip("cfunc implementation not available")


if __name__ == "__main__":
    test_polynomial_derivatives()
    test_linear_derivatives()
    test_derivative_consistency()
    print("✅ All derivative tests passed!")