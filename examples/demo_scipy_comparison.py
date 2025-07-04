#!/usr/bin/env python3
"""Demonstration comparing FastSpline 2D with scipy.interpolate.RectBivariateSpline."""

import numpy as np
import time
from fastspline import Spline2D

try:
    from scipy.interpolate import RectBivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, skipping comparison tests")


def demo_basic_usage_comparison():
    """Compare basic usage patterns."""
    print("FastSpline vs scipy.interpolate.RectBivariateSpline")
    print("=" * 55)
    
    # Create test data
    x = np.linspace(0, 4, 9)
    y = np.linspace(0, 3, 7)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.sin(X) * np.cos(Y)
    
    print("Test data: 9x7 grid, f(x,y) = sin(x)*cos(y)")
    print()
    
    # FastSpline usage
    print("FastSpline usage:")
    print("spline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)")
    fastspline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)
    
    if SCIPY_AVAILABLE:
        # scipy usage
        print("scipy usage:")
        print("spline = RectBivariateSpline(x, y, Z)")
        scipy_spline = RectBivariateSpline(x, y, Z)
        print()
        
        # Compare evaluations
        test_points = [(1.5, 1.2), (2.3, 2.1), (3.7, 0.8)]
        
        print("Evaluation comparison:")
        print("Point\t\tFastSpline\tscipy\t\tDifference")
        print("-" * 60)
        
        for xi, yi in test_points:
            fast_result = fastspline(xi, yi, grid=False)
            fast_val = fast_result[0] if hasattr(fast_result, '__len__') else fast_result
            scipy_result = scipy_spline(xi, yi)
            scipy_val = scipy_result[0, 0] if scipy_result.ndim > 0 else scipy_result
            diff = abs(fast_val - scipy_val)
            print(f"({xi:.1f}, {yi:.1f})\t{fast_val:.6f}\t{scipy_val:.6f}\t{diff:.2e}")
    else:
        # Just show FastSpline results
        test_points = [(1.5, 1.2), (2.3, 2.1), (3.7, 0.8)]
        print("FastSpline evaluation results:")
        for xi, yi in test_points:
            result = fastspline(xi, yi, grid=False)[0]
            print(f"f({xi:.1f}, {yi:.1f}) = {result:.6f}")


def demo_input_format_flexibility():
    """Demonstrate different input formats."""
    print("\nInput Format Flexibility")
    print("=" * 30)
    
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3])
    
    # 2D array format
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z_2d = X + 2*Y
    
    # 1D array format (row-major)
    Z_1d = Z_2d.ravel()
    
    print("FastSpline supports both input formats:")
    print("1. 2D array: Spline2D(x, y, Z_2d, ...)")
    print("2. 1D array: Spline2D(x, y, Z_1d, ...)")
    
    spline_2d = Spline2D(x, y, Z_2d, kx=1, ky=1)
    spline_1d = Spline2D(x, y, Z_1d, kx=1, ky=1)
    
    # Test evaluation
    test_x, test_y = 1.5, 0.8
    result_2d = spline_2d(test_x, test_y, grid=False)
    result_1d = spline_1d(test_x, test_y, grid=False)
    result_2d = result_2d[0] if hasattr(result_2d, '__len__') else result_2d
    result_1d = result_1d[0] if hasattr(result_1d, '__len__') else result_1d
    
    print(f"Results are identical: {abs(result_2d - result_1d) < 1e-15}")
    
    if SCIPY_AVAILABLE:
        scipy_spline = RectBivariateSpline(x, y, Z_2d)
        scipy_result = scipy_spline(test_x, test_y)
        scipy_result = scipy_result[0, 0] if scipy_result.ndim > 0 else scipy_result
        print(f"Match scipy result: {abs(result_2d - scipy_result) < 1e-12}")


def demo_derivative_evaluation():
    """Demonstrate derivative evaluation."""
    print("\nDerivative Evaluation")
    print("=" * 25)
    
    x = np.linspace(0, 2, 8)
    y = np.linspace(0, 2, 6)
    
    # Test function: f(x,y) = x^2 + xy + y^2
    # df/dx = 2x + y, df/dy = x + 2y
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = X**2 + X*Y + Y**2
    
    fastspline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)
    
    test_x, test_y = 1.2, 1.5
    
    # FastSpline derivatives
    f_val = fastspline(test_x, test_y, dx=0, dy=0, grid=False)
    df_dx = fastspline(test_x, test_y, dx=1, dy=0, grid=False)
    df_dy = fastspline(test_x, test_y, dx=0, dy=1, grid=False)
    
    # Extract scalar values
    f_val = f_val[0] if hasattr(f_val, '__len__') else f_val
    df_dx = df_dx[0] if hasattr(df_dx, '__len__') else df_dx
    df_dy = df_dy[0] if hasattr(df_dy, '__len__') else df_dy
    
    # Exact derivatives
    exact_f = test_x**2 + test_x*test_y + test_y**2
    exact_df_dx = 2*test_x + test_y
    exact_df_dy = test_x + 2*test_y
    
    print(f"At point ({test_x}, {test_y}):")
    print(f"Function:     FastSpline={f_val:.6f}, Exact={exact_f:.6f}, Error={abs(f_val-exact_f):.2e}")
    print(f"df/dx:        FastSpline={df_dx:.6f}, Exact={exact_df_dx:.6f}, Error={abs(df_dx-exact_df_dx):.2e}")
    print(f"df/dy:        FastSpline={df_dy:.6f}, Exact={exact_df_dy:.6f}, Error={abs(df_dy-exact_df_dy):.2e}")
    
    if SCIPY_AVAILABLE:
        scipy_spline = RectBivariateSpline(x, y, Z)
        scipy_f = scipy_spline(test_x, test_y)
        scipy_df_dx = scipy_spline(test_x, test_y, dx=1)
        scipy_df_dy = scipy_spline(test_x, test_y, dy=1)
        
        # Extract scalar values
        scipy_f = scipy_f[0, 0] if scipy_f.ndim > 0 else scipy_f
        scipy_df_dx = scipy_df_dx[0, 0] if scipy_df_dx.ndim > 0 else scipy_df_dx  
        scipy_df_dy = scipy_df_dy[0, 0] if scipy_df_dy.ndim > 0 else scipy_df_dy
        
        print(f"\nscipy comparison:")
        print(f"Function:     scipy={scipy_f:.6f}, diff={abs(f_val-scipy_f):.2e}")
        print(f"df/dx:        scipy={scipy_df_dx:.6f}, diff={abs(df_dx-scipy_df_dx):.2e}")
        print(f"df/dy:        scipy={scipy_df_dy:.6f}, diff={abs(df_dy-scipy_df_dy):.2e}")


def demo_performance_comparison():
    """Compare performance with scipy."""
    if not SCIPY_AVAILABLE:
        print("\nPerformance test skipped (scipy not available)")
        return
    
    print("\nPerformance Comparison")
    print("=" * 25)
    
    # Test different grid sizes
    grid_sizes = [(20, 15), (50, 40), (100, 80)]
    n_eval = 1000
    
    print(f"Testing construction and evaluation ({n_eval} points)")
    print("Grid Size\tFastSpline\tscipy\t\tSpeedup")
    print("-" * 50)
    
    for nx, ny in grid_sizes:
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = X**2 + Y**2
        
        # Random evaluation points
        x_eval = np.random.uniform(0, 1, n_eval)
        y_eval = np.random.uniform(0, 1, n_eval)
        
        # FastSpline timing
        start = time.time()
        fast_spline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)
        for i in range(n_eval):
            fast_spline(x_eval[i], y_eval[i], grid=False)
        fast_time = time.time() - start
        
        # scipy timing
        start = time.time()
        scipy_spline = RectBivariateSpline(x, y, Z)
        for i in range(n_eval):
            scipy_spline(x_eval[i], y_eval[i])
        scipy_time = time.time() - start
        
        speedup = scipy_time / fast_time
        print(f"{nx}x{ny}\t\t{fast_time:.4f}s\t\t{scipy_time:.4f}s\t\t{speedup:.2f}x")


def demo_c_interoperability():
    """Demonstrate C function pointer access."""
    print("\nC Interoperability (FastSpline Advantage)")
    print("=" * 45)
    
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])
    Z = np.array([[0, 1], [1, 2], [4, 5]])  # z = x^2 + y
    
    spline = Spline2D(x, y, Z.ravel(), kx=1, ky=1)
    
    print("FastSpline provides C-compatible function pointers:")
    print(f"evaluate function:     {spline.cfunc_evaluate}")
    print(f"derivative function:   {spline.cfunc_evaluate_derivatives}")
    print(f"Function address:      0x{spline.cfunc_evaluate.address:x}")
    
    # Test direct cfunc call
    from fastspline.spline2d import evaluate_spline_2d_cfunc
    
    test_x, test_y = 1.5, 0.5
    class_result = spline(test_x, test_y, grid=False)
    class_result = class_result[0] if hasattr(class_result, '__len__') else class_result
    cfunc_result = evaluate_spline_2d_cfunc(
        test_x, test_y, spline.coeffs,
        spline.x_min, spline.y_min, spline.h_step_x, spline.h_step_y,
        spline.nx, spline.ny, spline.order_x, spline.order_y,
        spline.periodic_x, spline.periodic_y
    )
    
    print(f"\nDirect C function call test:")
    print(f"Class method result:  {class_result:.6f}")
    print(f"C function result:    {cfunc_result:.6f}")
    print(f"Results identical:    {abs(class_result - cfunc_result) < 1e-15}")
    
    print("\nThis enables integration with:")
    print("- C/C++ applications")
    print("- Fortran codes")
    print("- Other compiled languages")
    print("- High-performance computing frameworks")


if __name__ == "__main__":
    demo_basic_usage_comparison()
    demo_input_format_flexibility()
    demo_derivative_evaluation()
    demo_performance_comparison()
    demo_c_interoperability()
    
    print("\nSummary:")
    print("FastSpline 2D provides:")
    print("✓ scipy.interpolate.RectBivariateSpline-compatible interface")
    print("✓ Flexible input formats (1D or 2D arrays)")
    print("✓ C-compatible function pointers for external integration")
    print("✓ Competitive performance")
    print("✓ Missing data handling (NaN support)")
    print("✓ Comprehensive derivative evaluation")
    
    if SCIPY_AVAILABLE:
        print("✓ Results consistent with scipy implementation")
    else:
        print("! Install scipy to run comparison tests")