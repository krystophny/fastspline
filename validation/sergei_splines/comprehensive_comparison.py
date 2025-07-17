#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splrep, splev
import sys
import os

# Add the parent directory to the path so we can import fastspline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc
    from fastspline.bispev_numba import bispev_cfunc_address
    from fastspline.parder import call_parder_safe, parder_cfunc_address
    fastspline_available = True
except ImportError as e:
    print(f"Warning: Could not import fastspline modules: {e}")
    fastspline_available = False

def test_function(x):
    """Test function: sin(2*pi*x)"""
    return np.sin(2.0 * np.pi * x)

def create_comparison_plots():
    """Create comprehensive comparison plots for all available implementations"""
    
    # Test data points
    n_data = 10
    x_data = np.linspace(0, 1, n_data)
    y_data = test_function(x_data)
    
    # Evaluation points (higher resolution)
    n_eval = 201
    x_eval = np.linspace(0, 1, n_eval)
    y_exact = test_function(x_eval)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Spline Comparison: SciPy vs FastSpline (Sergei)', fontsize=16)
    
    # Test different spline orders
    orders_to_test = [3, 4, 5]
    
    for order_idx, order in enumerate(orders_to_test):
        ax_interp = axes[0, order_idx]
        ax_error = axes[1, order_idx]
        
        # SciPy implementation
        if order == 3:
            # Use CubicSpline for natural boundary conditions
            scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
            y_scipy = scipy_spline(x_eval)
            scipy_available = True
        else:
            # Use scipy.interpolate.splrep/splev for higher orders
            try:
                tck = splrep(x_data, y_data, k=order, s=0)
                y_scipy = splev(x_eval, tck)
                scipy_available = True
            except Exception as e:
                print(f"SciPy order {order} failed: {e}")
                scipy_available = False
                y_scipy = np.zeros_like(x_eval)
        
        # Sergei (FastSpline) implementation
        if fastspline_available:
            try:
                # Construct Sergei spline
                coeff_sergei = np.zeros((order+1) * n_data)
                construct_splines_1d_cfunc(0.0, 1.0, y_data, n_data, order, False, coeff_sergei)
                
                # Evaluate Sergei spline with correct signature
                y_sergei = np.zeros(n_eval)
                for i, x in enumerate(x_eval):
                    y_val = np.zeros(1)
                    # Correct function signature: order, num_points, periodic, x_min, h_step, coeff, x, y_out
                    h_step = 1.0 / (n_data - 1)
                    evaluate_splines_1d_cfunc(order, n_data, False, 0.0, h_step, coeff_sergei, x, y_val)
                    y_sergei[i] = y_val[0]
                sergei_available = True
            except Exception as e:
                print(f"Sergei order {order} failed: {e}")
                sergei_available = False
                y_sergei = np.zeros_like(x_eval)
        else:
            sergei_available = False
            y_sergei = np.zeros_like(x_eval)
        
        # Plot interpolations
        ax_interp.plot(x_eval, y_exact, 'k-', linewidth=3, label='Exact sin(2πx)', alpha=0.8)
        ax_interp.plot(x_data, y_data, 'ko', markersize=8, label='Data points', zorder=5)
        
        if scipy_available:
            ax_interp.plot(x_eval, y_scipy, 'r--', linewidth=2, label='SciPy', alpha=0.9)
        if sergei_available:
            ax_interp.plot(x_eval, y_sergei, 'b-.', linewidth=2, label='FastSpline (Sergei)', alpha=0.9)
        
        ax_interp.set_title(f'Order {order} Spline Interpolation')
        ax_interp.set_xlabel('x')
        ax_interp.set_ylabel('y')
        ax_interp.legend()
        ax_interp.grid(True, alpha=0.3)
        
        # Plot errors
        if scipy_available:
            error_scipy = np.abs(y_scipy - y_exact)
            ax_error.semilogy(x_eval, error_scipy, 'r--', linewidth=2, label='SciPy Error', alpha=0.9)
        if sergei_available:
            error_sergei = np.abs(y_sergei - y_exact)
            ax_error.semilogy(x_eval, error_sergei, 'b-.', linewidth=2, label='FastSpline Error', alpha=0.9)
        
        ax_error.set_title(f'Order {order} Absolute Error')
        ax_error.set_xlabel('x')
        ax_error.set_ylabel('|Error|')
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        ax_error.set_ylim(1e-16, 1e-2)
        
        # Print numerical comparison
        print(f"\nOrder {order} Results:")
        print("-" * 40)
        if scipy_available:
            rms_scipy = np.sqrt(np.mean((y_scipy - y_exact)**2))
            max_error_scipy = np.max(np.abs(y_scipy - y_exact))
            print(f"  SciPy:      RMS={rms_scipy:.2e}, Max={max_error_scipy:.2e}")
        if sergei_available:
            rms_sergei = np.sqrt(np.mean((y_sergei - y_exact)**2))
            max_error_sergei = np.max(np.abs(y_sergei - y_exact))
            print(f"  FastSpline: RMS={rms_sergei:.2e}, Max={max_error_sergei:.2e}")
            
        # Check if implementations agree
        if scipy_available and sergei_available:
            diff_rms = np.sqrt(np.mean((y_scipy - y_sergei)**2))
            diff_max = np.max(np.abs(y_scipy - y_sergei))
            print(f"  Difference: RMS={diff_rms:.2e}, Max={diff_max:.2e}")
            
            # Check if they are essentially identical
            if diff_rms < 1e-10:
                print("  ✓ Implementations are essentially identical")
            elif diff_rms < 1e-6:
                print("  ≈ Implementations are very close")
            else:
                print("  ✗ Implementations differ significantly")
    
    plt.tight_layout()
    plt.savefig('comprehensive_spline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show available FastSpline modules
    print("\n" + "="*80)
    print("AVAILABLE FASTSPLINE MODULES")
    print("="*80)
    
    if fastspline_available:
        print("✓ FastSpline is available with the following modules:")
        print("  - sergei_splines: Sergei's equidistant spline implementation")
        print("  - bispev_numba: Bivariate spline evaluation")
        print("  - parder: Partial derivative calculation")
        print("  - fpbisp_numba: Bivariate spline fitting")
        print("  - fpbspl_numba: B-spline basis functions")
        
        # Show Sergei coefficients for order 5
        print("\n" + "="*80)
        print("SERGEI SPLINE COEFFICIENTS (Order 5, n=10)")
        print("="*80)
        
        try:
            order = 5
            coeff_sergei = np.zeros((order+1) * n_data)
            construct_splines_1d_cfunc(0.0, 1.0, y_data, n_data, order, False, coeff_sergei)
            
            # Reshape for easier viewing
            coeff_matrix = coeff_sergei.reshape(order+1, n_data)
            coeff_names = ['a (function)', 'b (1st deriv)', 'c (2nd deriv)', 'd (3rd deriv)', 'e (4th deriv)', 'f (5th deriv)']
            
            for i in range(order+1):
                print(f"\n{coeff_names[i]}:")
                print("  Point:", end="")
                for j in range(n_data):
                    print(f"{j:>10}", end="")
                print()
                print("  Value:", end="")
                for j in range(n_data):
                    print(f"{coeff_matrix[i,j]:>10.3f}", end="")
                print()
        except Exception as e:
            print(f"Error displaying coefficients: {e}")
    else:
        print("✗ FastSpline is not available")

    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    import time
    
    n_trials = 100
    
    for order in orders_to_test:
        print(f"\nOrder {order} Performance (n={n_data}, {n_trials} trials):")
        print("-" * 50)
        
        # SciPy timing
        if order == 3:
            start_time = time.time()
            for _ in range(n_trials):
                scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
                y_scipy_perf = scipy_spline(x_eval)
            scipy_time = time.time() - start_time
            print(f"  SciPy:      {scipy_time:.4f}s ({scipy_time/n_trials*1000:.2f}ms per trial)")
        else:
            start_time = time.time()
            for _ in range(n_trials):
                tck = splrep(x_data, y_data, k=order, s=0)
                y_scipy_perf = splev(x_eval, tck)
            scipy_time = time.time() - start_time
            print(f"  SciPy:      {scipy_time:.4f}s ({scipy_time/n_trials*1000:.2f}ms per trial)")
        
        # FastSpline timing
        if fastspline_available:
            try:
                start_time = time.time()
                for _ in range(n_trials):
                    coeff_sergei_perf = np.zeros((order+1) * n_data)
                    construct_splines_1d_cfunc(0.0, 1.0, y_data, n_data, order, False, coeff_sergei_perf)
                    
                    y_sergei_perf = np.zeros(n_eval)
                    h_step = 1.0 / (n_data - 1)
                    for i, x in enumerate(x_eval):
                        y_val = np.zeros(1)
                        evaluate_splines_1d_cfunc(order, n_data, False, 0.0, h_step, coeff_sergei_perf, x, y_val)
                        y_sergei_perf[i] = y_val[0]
                fastspline_time = time.time() - start_time
                
                print(f"  FastSpline: {fastspline_time:.4f}s ({fastspline_time/n_trials*1000:.2f}ms per trial)")
                
                # Speed comparison
                if scipy_time > 0:
                    speedup = scipy_time / fastspline_time
                    print(f"  Speedup:    {speedup:.2f}x {'(FastSpline faster)' if speedup > 1 else '(SciPy faster)'}")
                
            except Exception as e:
                print(f"  FastSpline: Error - {e}")

if __name__ == "__main__":
    create_comparison_plots()