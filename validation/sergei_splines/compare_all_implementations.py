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
    from fastspline.dierckx_splines import construct_splines_1d_cfunc as dierckx_construct
    from fastspline.dierckx_splines import evaluate_splines_1d_cfunc as dierckx_evaluate
    fastspline_available = True
except ImportError as e:
    print(f"Warning: Could not import fastspline modules: {e}")
    fastspline_available = False

def test_function(x):
    """Test function: sin(2*pi*x)"""
    return np.sin(2.0 * np.pi * x)

def test_function_derivative(x):
    """Derivative of test function"""
    return 2.0 * np.pi * np.cos(2.0 * np.pi * x)

def create_comparison_plots():
    """Create side-by-side comparison plots for all three implementations"""
    
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
    fig.suptitle('Spline Implementation Comparison: SciPy vs Dierckx vs Sergei (Fortran)', fontsize=16)
    
    # Test different spline orders
    orders_to_test = [3, 4, 5]
    
    for order_idx, order in enumerate(orders_to_test):
        ax_interp = axes[0, order_idx]
        ax_error = axes[1, order_idx]
        
        # SciPy implementation (only cubic available)
        if order == 3:
            scipy_spline = CubicSpline(x_data, y_data, bc_type='natural')
            y_scipy = scipy_spline(x_eval)
            scipy_available = True
        else:
            # Use scipy.interpolate.splrep/splev for higher orders
            try:
                tck = splrep(x_data, y_data, k=order, s=0)
                y_scipy = splev(x_eval, tck)
                scipy_available = True
            except:
                scipy_available = False
                y_scipy = np.zeros_like(x_eval)
        
        # Dierckx implementation
        if fastspline_available:
            try:
                # Construct Dierckx spline
                coeff_dierckx = np.zeros((order+1) * n_data)
                dierckx_construct(0.0, 1.0, y_data, n_data, order, False, coeff_dierckx)
                
                # Evaluate Dierckx spline
                y_dierckx = np.zeros(n_eval)
                for i, x in enumerate(x_eval):
                    y_val = np.zeros(1)
                    dierckx_evaluate(x, 0.0, 1.0, coeff_dierckx, n_data, order, y_val)
                    y_dierckx[i] = y_val[0]
                dierckx_available = True
            except Exception as e:
                print(f"Dierckx order {order} failed: {e}")
                dierckx_available = False
                y_dierckx = np.zeros_like(x_eval)
        else:
            dierckx_available = False
            y_dierckx = np.zeros_like(x_eval)
        
        # Sergei (Fortran) implementation
        if fastspline_available:
            try:
                # Construct Sergei spline
                coeff_sergei = np.zeros((order+1) * n_data)
                construct_splines_1d_cfunc(0.0, 1.0, y_data, n_data, order, False, coeff_sergei)
                
                # Evaluate Sergei spline
                y_sergei = np.zeros(n_eval)
                for i, x in enumerate(x_eval):
                    y_val = np.zeros(1)
                    evaluate_splines_1d_cfunc(x, 0.0, 1.0, coeff_sergei, n_data, order, y_val)
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
        ax_interp.plot(x_eval, y_exact, 'k-', linewidth=2, label='Exact', alpha=0.7)
        ax_interp.plot(x_data, y_data, 'ko', markersize=8, label='Data points')
        
        if scipy_available:
            ax_interp.plot(x_eval, y_scipy, 'r--', linewidth=2, label='SciPy', alpha=0.8)
        if dierckx_available:
            ax_interp.plot(x_eval, y_dierckx, 'g:', linewidth=2, label='Dierckx', alpha=0.8)
        if sergei_available:
            ax_interp.plot(x_eval, y_sergei, 'b-.', linewidth=2, label='Sergei', alpha=0.8)
        
        ax_interp.set_title(f'Order {order} Spline Interpolation')
        ax_interp.set_xlabel('x')
        ax_interp.set_ylabel('y')
        ax_interp.legend()
        ax_interp.grid(True, alpha=0.3)
        
        # Plot errors
        if scipy_available:
            error_scipy = np.abs(y_scipy - y_exact)
            ax_error.semilogy(x_eval, error_scipy, 'r--', linewidth=2, label='SciPy Error', alpha=0.8)
        if dierckx_available:
            error_dierckx = np.abs(y_dierckx - y_exact)
            ax_error.semilogy(x_eval, error_dierckx, 'g:', linewidth=2, label='Dierckx Error', alpha=0.8)
        if sergei_available:
            error_sergei = np.abs(y_sergei - y_exact)
            ax_error.semilogy(x_eval, error_sergei, 'b-.', linewidth=2, label='Sergei Error', alpha=0.8)
        
        ax_error.set_title(f'Order {order} Absolute Error')
        ax_error.set_xlabel('x')
        ax_error.set_ylabel('|Error|')
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        
        # Print numerical comparison
        print(f"\nOrder {order} RMS Errors:")
        if scipy_available:
            rms_scipy = np.sqrt(np.mean((y_scipy - y_exact)**2))
            print(f"  SciPy:  {rms_scipy:.2e}")
        if dierckx_available:
            rms_dierckx = np.sqrt(np.mean((y_dierckx - y_exact)**2))
            print(f"  Dierckx: {rms_dierckx:.2e}")
        if sergei_available:
            rms_sergei = np.sqrt(np.mean((y_sergei - y_exact)**2))
            print(f"  Sergei:  {rms_sergei:.2e}")
    
    plt.tight_layout()
    plt.savefig('spline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed coefficient comparison table
    print("\n" + "="*80)
    print("COEFFICIENT COMPARISON")
    print("="*80)
    
    for order in orders_to_test:
        print(f"\nOrder {order} Coefficients (first 5 points):")
        print("-" * 50)
        
        if fastspline_available:
            try:
                # Get Dierckx coefficients
                coeff_dierckx = np.zeros((order+1) * n_data)
                dierckx_construct(0.0, 1.0, y_data, n_data, order, False, coeff_dierckx)
                
                # Get Sergei coefficients
                coeff_sergei = np.zeros((order+1) * n_data)
                construct_splines_1d_cfunc(0.0, 1.0, y_data, n_data, order, False, coeff_sergei)
                
                print(f"{'Coeff':<8} {'Dierckx':<15} {'Sergei':<15} {'Diff':<15}")
                print("-" * 58)
                
                for coeff_order in range(order+1):
                    for i in range(min(5, n_data)):
                        idx = coeff_order * n_data + i
                        dierckx_val = coeff_dierckx[idx]
                        sergei_val = coeff_sergei[idx]
                        diff = abs(dierckx_val - sergei_val)
                        print(f"c{coeff_order}[{i}]  {dierckx_val:13.6f}  {sergei_val:13.6f}  {diff:13.6e}")
                
            except Exception as e:
                print(f"Error comparing coefficients for order {order}: {e}")

if __name__ == "__main__":
    create_comparison_plots()