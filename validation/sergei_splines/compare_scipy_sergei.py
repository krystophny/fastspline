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
    fastspline_available = True
except ImportError as e:
    print(f"Warning: Could not import fastspline modules: {e}")
    fastspline_available = False

def test_function(x):
    """Test function: sin(2*pi*x)"""
    return np.sin(2.0 * np.pi * x)

def create_comparison_plots():
    """Create side-by-side comparison plots for SciPy vs Sergei implementations"""
    
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
    fig.suptitle('Spline Implementation Comparison: SciPy vs Sergei (Fortran-based)', fontsize=16)
    
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
        
        # Sergei (Fortran-based) implementation
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
        ax_interp.plot(x_eval, y_exact, 'k-', linewidth=3, label='Exact', alpha=0.8)
        ax_interp.plot(x_data, y_data, 'ko', markersize=8, label='Data points', zorder=5)
        
        if scipy_available:
            ax_interp.plot(x_eval, y_scipy, 'r--', linewidth=2, label='SciPy', alpha=0.9)
        if sergei_available:
            ax_interp.plot(x_eval, y_sergei, 'b-.', linewidth=2, label='Sergei/Fortran', alpha=0.9)
        
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
            ax_error.semilogy(x_eval, error_sergei, 'b-.', linewidth=2, label='Sergei Error', alpha=0.9)
        
        ax_error.set_title(f'Order {order} Absolute Error')
        ax_error.set_xlabel('x')
        ax_error.set_ylabel('|Error|')
        ax_error.legend()
        ax_error.grid(True, alpha=0.3)
        ax_error.set_ylim(1e-16, 1e-2)
        
        # Print numerical comparison
        print(f"\nOrder {order} RMS Errors:")
        if scipy_available:
            rms_scipy = np.sqrt(np.mean((y_scipy - y_exact)**2))
            max_error_scipy = np.max(np.abs(y_scipy - y_exact))
            print(f"  SciPy:  RMS={rms_scipy:.2e}, Max={max_error_scipy:.2e}")
        if sergei_available:
            rms_sergei = np.sqrt(np.mean((y_sergei - y_exact)**2))
            max_error_sergei = np.max(np.abs(y_sergei - y_exact))
            print(f"  Sergei: RMS={rms_sergei:.2e}, Max={max_error_sergei:.2e}")
            
        # Check if implementations agree
        if scipy_available and sergei_available:
            diff_rms = np.sqrt(np.mean((y_scipy - y_sergei)**2))
            diff_max = np.max(np.abs(y_scipy - y_sergei))
            print(f"  Implementation difference: RMS={diff_rms:.2e}, Max={diff_max:.2e}")
    
    plt.tight_layout()
    plt.savefig('scipy_sergei_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed coefficient comparison for order 5
    print("\n" + "="*80)
    print("COEFFICIENT COMPARISON (Order 5)")
    print("="*80)
    
    if fastspline_available:
        try:
            order = 5
            # Get Sergei coefficients
            coeff_sergei = np.zeros((order+1) * n_data)
            construct_splines_1d_cfunc(0.0, 1.0, y_data, n_data, order, False, coeff_sergei)
            
            print(f"Sergei spline coefficients (n={n_data}, order={order}):")
            print(f"{'Coeff':<8} {'i':<3} {'Value':<15}")
            print("-" * 26)
            
            coeff_names = ['a', 'b', 'c', 'd', 'e', 'f']
            for coeff_order in range(order+1):
                for i in range(n_data):
                    idx = coeff_order * n_data + i
                    value = coeff_sergei[idx]
                    print(f"{coeff_names[coeff_order]}[{i}]   {i:<3} {value:13.6f}")
                print()
            
        except Exception as e:
            print(f"Error getting Sergei coefficients: {e}")

    # Create a convergence analysis
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    n_values = [6, 8, 10, 12, 15, 20]
    
    for order in orders_to_test:
        print(f"\nOrder {order} convergence:")
        print(f"{'n':<4} {'SciPy RMS':<12} {'Sergei RMS':<12} {'Ratio':<10}")
        print("-" * 40)
        
        for n in n_values:
            x_test = np.linspace(0, 1, n)
            y_test = test_function(x_test)
            x_eval_conv = np.linspace(0, 1, 101)
            y_exact_conv = test_function(x_eval_conv)
            
            # SciPy
            try:
                if order == 3:
                    scipy_spline = CubicSpline(x_test, y_test, bc_type='natural')
                    y_scipy_conv = scipy_spline(x_eval_conv)
                else:
                    tck = splrep(x_test, y_test, k=order, s=0)
                    y_scipy_conv = splev(x_eval_conv, tck)
                rms_scipy_conv = np.sqrt(np.mean((y_scipy_conv - y_exact_conv)**2))
                scipy_conv_available = True
            except:
                scipy_conv_available = False
                rms_scipy_conv = np.nan
            
            # Sergei
            try:
                if fastspline_available:
                    coeff_sergei_conv = np.zeros((order+1) * n)
                    construct_splines_1d_cfunc(0.0, 1.0, y_test, n, order, False, coeff_sergei_conv)
                    
                    y_sergei_conv = np.zeros(len(x_eval_conv))
                    for i, x in enumerate(x_eval_conv):
                        y_val = np.zeros(1)
                        evaluate_splines_1d_cfunc(x, 0.0, 1.0, coeff_sergei_conv, n, order, y_val)
                        y_sergei_conv[i] = y_val[0]
                    
                    rms_sergei_conv = np.sqrt(np.mean((y_sergei_conv - y_exact_conv)**2))
                    sergei_conv_available = True
                else:
                    sergei_conv_available = False
                    rms_sergei_conv = np.nan
            except:
                sergei_conv_available = False
                rms_sergei_conv = np.nan
            
            # Calculate ratio
            if scipy_conv_available and sergei_conv_available and rms_scipy_conv > 0:
                ratio = rms_sergei_conv / rms_scipy_conv
            else:
                ratio = np.nan
            
            print(f"{n:<4} {rms_scipy_conv:<12.2e} {rms_sergei_conv:<12.2e} {ratio:<10.3f}")

if __name__ == "__main__":
    create_comparison_plots()