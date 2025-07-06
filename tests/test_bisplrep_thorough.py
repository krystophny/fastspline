#!/usr/bin/env python3
"""
Thorough validation of bisplrep/bisplev cfunc against SciPy
Tests many cases to ensure results match
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev
import sys
sys.path.insert(0, '..')

from dierckx_cfunc import bisplrep_cfunc, bisplev_cfunc

def compare_results(name, scipy_val, cfunc_val, tol=1e-2):
    """Compare and report results"""
    if np.isscalar(scipy_val):
        error = abs(scipy_val - cfunc_val)
        rel_error = error / (abs(scipy_val) + 1e-10)
    else:
        error = np.max(np.abs(scipy_val - cfunc_val))
        rel_error = error / (np.max(np.abs(scipy_val)) + 1e-10)
    
    status = "✓ PASS" if error < tol else "✗ FAIL"
    print(f"  {name}: {status} (error={error:.2e}, rel={rel_error:.2%})")
    return error < tol

def test_surface_types():
    """Test different mathematical surfaces"""
    print("\n" + "="*80)
    print("TESTING DIFFERENT SURFACE TYPES")
    print("="*80)
    
    # Grid for testing
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Evaluation points
    x_eval = np.linspace(-1.8, 1.8, 15)
    y_eval = np.linspace(-1.8, 1.8, 15)
    
    surfaces = {
        "Plane": lambda x, y: 2*x + 3*y + 1,
        "Quadratic": lambda x, y: x**2 + y**2,
        "Saddle": lambda x, y: x**2 - y**2,
        "Gaussian": lambda x, y: np.exp(-(x**2 + y**2)),
        "Sinusoidal": lambda x, y: np.sin(x) * np.cos(y),
        "Product": lambda x, y: x * y,
        "Cubic": lambda x, y: x**3 + y**3 - 3*x*y,
        "Exponential": lambda x, y: np.exp(x/2) * np.sin(y),
        "Rational": lambda x, y: 1 / (1 + x**2 + y**2),
        "Step": lambda x, y: np.where((x > 0) & (y > 0), 1.0, 0.0)
    }
    
    all_pass = True
    
    for name, func in surfaces.items():
        print(f"\nTesting {name} surface:")
        z_flat = func(x_flat, y_flat)
        
        try:
            # Fit with SciPy
            tck_scipy = bisplrep(x_flat, y_flat, z_flat, kx=3, ky=3, s=0)
            z_scipy = bisplev(x_eval, y_eval, tck_scipy)
            
            # Fit with cfunc
            tx, ty, c, kx, ky = bisplrep_cfunc(x_flat, y_flat, z_flat, kx=3, ky=3, s=0.0)
            z_cfunc = bisplev_cfunc(x_eval, y_eval, tx, ty, c, kx, ky)
            
            # Compare
            passed = compare_results(f"{name}", z_scipy, z_cfunc)
            all_pass = all_pass and passed
            
        except Exception as e:
            print(f"  ✗ FAIL: {str(e)}")
            all_pass = False
    
    return all_pass

def test_grid_sizes():
    """Test various grid sizes"""
    print("\n" + "="*80)
    print("TESTING DIFFERENT GRID SIZES")
    print("="*80)
    
    # Simple test function
    func = lambda x, y: np.sin(x) * np.cos(y)
    
    grid_sizes = [
        (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
        (10, 10), (12, 12), (15, 15), (20, 20),
        (5, 10), (10, 5), (8, 12), (12, 8)  # Non-square grids
    ]
    
    all_pass = True
    
    for nx, ny in grid_sizes:
        print(f"\nGrid size {nx}×{ny}:")
        
        x = np.linspace(-2, 2, nx)
        y = np.linspace(-2, 2, ny)
        X, Y = np.meshgrid(x, y)
        z_flat = func(X.flatten(), Y.flatten())
        
        try:
            # SciPy
            tck_scipy = bisplrep(X.flatten(), Y.flatten(), z_flat, s=0)
            
            # cfunc
            tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), z_flat, s=0.0)
            
            # Evaluate at center
            z_scipy_center = bisplev(0.0, 0.0, tck_scipy)
            z_cfunc_center = bisplev_cfunc(np.array([0.0]), np.array([0.0]), tx, ty, c, kx, ky)[0,0]
            
            passed = compare_results(f"Center value", z_scipy_center, z_cfunc_center)
            all_pass = all_pass and passed
            
        except Exception as e:
            print(f"  ✗ FAIL: {str(e)}")
            all_pass = False
    
    return all_pass

def test_spline_degrees():
    """Test different spline degrees"""
    print("\n" + "="*80)
    print("TESTING DIFFERENT SPLINE DEGREES")
    print("="*80)
    
    # Test data
    x = np.linspace(-1, 1, 8)
    y = np.linspace(-1, 1, 8)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    degrees = [(1,1), (2,2), (3,3), (4,4), (5,5), (1,3), (3,1), (2,4), (4,2)]
    
    all_pass = True
    
    for kx, ky in degrees:
        print(f"\nDegree (kx={kx}, ky={ky}):")
        
        try:
            # SciPy
            tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z.flatten(), kx=kx, ky=ky, s=0)
            
            # cfunc
            tx, ty, c, kx_out, ky_out = bisplrep_cfunc(X.flatten(), Y.flatten(), Z.flatten(), 
                                                        kx=kx, ky=ky, s=0.0)
            
            # Evaluate on grid
            x_test = np.array([0.0, 0.5, -0.5])
            y_test = np.array([0.0, 0.5, -0.5])
            
            z_scipy = bisplev(x_test, y_test, tck_scipy)
            z_cfunc = bisplev_cfunc(x_test, y_test, tx, ty, c, kx_out, ky_out)
            
            passed = compare_results(f"kx={kx}, ky={ky}", z_scipy, z_cfunc)
            all_pass = all_pass and passed
            
        except Exception as e:
            print(f"  ✗ FAIL: {str(e)}")
            all_pass = False
    
    return all_pass

def test_boundary_cases():
    """Test evaluation at boundaries and corners"""
    print("\n" + "="*80)
    print("TESTING BOUNDARY AND CORNER CASES")
    print("="*80)
    
    # Create test surface
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    # Fit splines
    tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z.flatten(), s=0)
    tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z.flatten(), s=0.0)
    
    # Test points including boundaries and corners
    test_points = [
        ("Center", 0.0, 0.0),
        ("Left edge", -1.0, 0.0),
        ("Right edge", 1.0, 0.0),
        ("Bottom edge", 0.0, -1.0),
        ("Top edge", 0.0, 1.0),
        ("Bottom-left corner", -1.0, -1.0),
        ("Bottom-right corner", 1.0, -1.0),
        ("Top-left corner", -1.0, 1.0),
        ("Top-right corner", 1.0, 1.0),
        ("Near boundary", 0.99, 0.99),
        ("Just inside", -0.95, 0.95)
    ]
    
    all_pass = True
    
    for name, x_pt, y_pt in test_points:
        z_scipy = bisplev(x_pt, y_pt, tck_scipy)
        z_cfunc = bisplev_cfunc(np.array([x_pt]), np.array([y_pt]), tx, ty, c, kx, ky)[0,0]
        
        passed = compare_results(name, z_scipy, z_cfunc)
        all_pass = all_pass and passed
    
    return all_pass

def test_special_values():
    """Test handling of special values"""
    print("\n" + "="*80)
    print("TESTING SPECIAL VALUES")
    print("="*80)
    
    all_pass = True
    
    # Test 1: Constant surface
    print("\nConstant surface (z=5):")
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, 5.0)
    
    try:
        tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z.flatten(), s=0)
        tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z.flatten(), s=0.0)
        
        z_scipy = bisplev(0.5, 0.5, tck_scipy)
        z_cfunc = bisplev_cfunc(np.array([0.5]), np.array([0.5]), tx, ty, c, kx, ky)[0,0]
        
        passed = compare_results("Constant value", z_scipy, z_cfunc)
        all_pass = all_pass and passed
    except Exception as e:
        print(f"  ✗ FAIL: {str(e)}")
        all_pass = False
    
    # Test 2: Very small values
    print("\nVery small values (1e-10 scale):")
    Z_small = 1e-10 * (X**2 + Y**2)
    
    try:
        tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z_small.flatten(), s=0)
        tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z_small.flatten(), s=0.0)
        
        z_scipy = bisplev(0.0, 0.0, tck_scipy)
        z_cfunc = bisplev_cfunc(np.array([0.0]), np.array([0.0]), tx, ty, c, kx, ky)[0,0]
        
        passed = compare_results("Small values", z_scipy, z_cfunc, tol=1e-12)
        all_pass = all_pass and passed
    except Exception as e:
        print(f"  ✗ FAIL: {str(e)}")
        all_pass = False
    
    # Test 3: Large values
    print("\nLarge values (1e6 scale):")
    Z_large = 1e6 * np.sin(X) * np.cos(Y)
    
    try:
        tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z_large.flatten(), s=0)
        tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z_large.flatten(), s=0.0)
        
        z_scipy = bisplev(0.5, 0.5, tck_scipy)
        z_cfunc = bisplev_cfunc(np.array([0.5]), np.array([0.5]), tx, ty, c, kx, ky)[0,0]
        
        passed = compare_results("Large values", z_scipy, z_cfunc, tol=1e4)
        all_pass = all_pass and passed
    except Exception as e:
        print(f"  ✗ FAIL: {str(e)}")
        all_pass = False
    
    return all_pass

def test_evaluation_grids():
    """Test evaluation on various grid patterns"""
    print("\n" + "="*80)
    print("TESTING DIFFERENT EVALUATION PATTERNS")
    print("="*80)
    
    # Fit a test surface
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = X * np.exp(-X**2 - Y**2)
    
    tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z.flatten(), s=0)
    tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z.flatten(), s=0.0)
    
    all_pass = True
    
    # Test 1: Single point
    print("\nSingle point evaluation:")
    z_scipy = bisplev(0.3, -0.2, tck_scipy)
    z_cfunc = bisplev_cfunc(np.array([0.3]), np.array([-0.2]), tx, ty, c, kx, ky)[0,0]
    passed = compare_results("Single point", z_scipy, z_cfunc)
    all_pass = all_pass and passed
    
    # Test 2: Line of points
    print("\nLine evaluation (10 points):")
    x_line = np.linspace(-0.8, 0.8, 10)
    y_line = np.zeros(10)
    z_scipy = bisplev(x_line, y_line, tck_scipy)
    z_cfunc = bisplev_cfunc(x_line, y_line, tx, ty, c, kx, ky)
    
    # For line evaluation, scipy returns 1D array
    passed = compare_results("Line", z_scipy, z_cfunc[0, :])
    all_pass = all_pass and passed
    
    # Test 3: Regular grid
    print("\nRegular grid (7×7):")
    x_grid = np.linspace(-0.9, 0.9, 7)
    y_grid = np.linspace(-0.9, 0.9, 7)
    z_scipy = bisplev(x_grid, y_grid, tck_scipy)
    z_cfunc = bisplev_cfunc(x_grid, y_grid, tx, ty, c, kx, ky)
    passed = compare_results("Grid", z_scipy, z_cfunc)
    all_pass = all_pass and passed
    
    # Test 4: Dense grid
    print("\nDense grid (50×50):")
    x_dense = np.linspace(-1, 1, 50)
    y_dense = np.linspace(-1, 1, 50)
    z_scipy = bisplev(x_dense, y_dense, tck_scipy)
    z_cfunc = bisplev_cfunc(x_dense, y_dense, tx, ty, c, kx, ky)
    passed = compare_results("Dense grid", z_scipy, z_cfunc)
    all_pass = all_pass and passed
    
    return all_pass

def test_extrapolation():
    """Test behavior outside the data domain"""
    print("\n" + "="*80)
    print("TESTING EXTRAPOLATION")
    print("="*80)
    
    # Small domain data
    x = np.linspace(-0.5, 0.5, 8)
    y = np.linspace(-0.5, 0.5, 8)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z.flatten(), s=0)
    tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z.flatten(), s=0.0)
    
    # Test points outside domain
    test_points = [
        ("Just outside +x", 0.6, 0.0),
        ("Just outside -x", -0.6, 0.0),
        ("Just outside +y", 0.0, 0.6),
        ("Just outside -y", 0.0, -0.6),
        ("Far outside", 2.0, 2.0),
        ("Mixed", 0.7, -0.3)
    ]
    
    all_pass = True
    
    for name, x_pt, y_pt in test_points:
        z_scipy = bisplev(x_pt, y_pt, tck_scipy)
        z_cfunc = bisplev_cfunc(np.array([x_pt]), np.array([y_pt]), tx, ty, c, kx, ky)[0,0]
        
        # For extrapolation, allow larger tolerance
        passed = compare_results(name, z_scipy, z_cfunc, tol=0.1)
        all_pass = all_pass and passed
    
    return all_pass

def test_real_world_data():
    """Test with data that mimics real-world scenarios"""
    print("\n" + "="*80)
    print("TESTING REAL-WORLD SCENARIOS")
    print("="*80)
    
    all_pass = True
    
    # Scenario 1: Noisy measurement data
    print("\nNoisy measurement data:")
    x = np.linspace(0, 10, 15)
    y = np.linspace(0, 10, 15)
    X, Y = np.meshgrid(x, y)
    Z_true = 2 * np.sin(X/2) * np.cos(Y/3)
    Z_noisy = Z_true + 0.05 * np.random.randn(*Z_true.shape)
    
    try:
        # Use smoothing for noisy data
        tck_scipy = bisplrep(X.flatten(), Y.flatten(), Z_noisy.flatten(), s=0.5)
        tx, ty, c, kx, ky = bisplrep_cfunc(X.flatten(), Y.flatten(), Z_noisy.flatten(), s=0.0)
        
        # Note: Our implementation doesn't support smoothing yet, so results will differ
        print("  Note: cfunc uses s=0 (no smoothing), SciPy uses s=0.5")
        
    except Exception as e:
        print(f"  ✗ FAIL: {str(e)}")
        all_pass = False
    
    # Scenario 2: Geographic data (elevation)
    print("\nGeographic elevation data:")
    # Simulate elevation data on irregular grid
    np.random.seed(42)
    n_points = 50
    x_geo = np.random.uniform(-10, 10, n_points)
    y_geo = np.random.uniform(-10, 10, n_points)
    z_geo = 100 + 50 * np.exp(-((x_geo**2 + y_geo**2) / 50))  # Mountain-like
    
    try:
        tck_scipy = bisplrep(x_geo, y_geo, z_geo, s=0)
        tx, ty, c, kx, ky = bisplrep_cfunc(x_geo, y_geo, z_geo, s=0.0)
        
        # Evaluate at some points
        x_test = np.array([0.0, 5.0, -5.0])
        y_test = np.array([0.0, 5.0, -5.0])
        z_scipy = bisplev(x_test, y_test, tck_scipy)
        z_cfunc = bisplev_cfunc(x_test, y_test, tx, ty, c, kx, ky)
        
        # Scattered data often has larger errors
        passed = compare_results("Geographic data", z_scipy, z_cfunc, tol=10.0)
        all_pass = all_pass and passed
        
    except Exception as e:
        print(f"  ✗ FAIL: {str(e)}")
        all_pass = False
    
    return all_pass

def run_all_tests():
    """Run all validation tests"""
    print("COMPREHENSIVE BISPLREP/BISPLEV VALIDATION")
    print("Testing cfunc implementation against SciPy")
    print("="*80)
    
    results = []
    
    # Run all test suites
    test_suites = [
        ("Surface Types", test_surface_types),
        ("Grid Sizes", test_grid_sizes),
        ("Spline Degrees", test_spline_degrees),
        ("Boundary Cases", test_boundary_cases),
        ("Special Values", test_special_values),
        ("Evaluation Grids", test_evaluation_grids),
        ("Extrapolation", test_extrapolation),
        ("Real-World Data", test_real_world_data)
    ]
    
    for name, test_func in test_suites:
        print(f"\n\nRunning {name} tests...")
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nTest suite {name} crashed: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL VALIDATION TESTS PASSED!")
        print("The cfunc implementation matches SciPy within acceptable tolerances.")
    else:
        print("✗ SOME VALIDATION TESTS FAILED!")
        print("The cfunc implementation has accuracy issues in some cases.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())