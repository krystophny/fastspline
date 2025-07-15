#!/usr/bin/env python3
"""
Create comparison plots of scipy vs cfunc derivative computation for 2D splines.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep, dfitpack
from fastspline.numba_implementation.parder import call_parder_safe
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def create_test_function(func_type='gaussian'):
    """Create test data for spline fitting."""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if func_type == 'gaussian':
        # Gaussian function
        Z = np.exp(-(X**2 + Y**2))
    elif func_type == 'polynomial':
        # Polynomial function
        Z = X**3 + 2*X**2*Y + Y**3 - 3*X*Y
    elif func_type == 'sinusoidal':
        # Sinusoidal function
        Z = np.sin(2*X) * np.cos(2*Y)
    elif func_type == 'peaks':
        # Matlab peaks function
        Z = (3*(1-X)**2 * np.exp(-(X**2) - (Y+1)**2) - 
             10*(X/5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) - 
             1/3*np.exp(-(X+1)**2 - Y**2))
    
    return X, Y, Z


def plot_derivative_comparison(func_type='gaussian', derivative_order=(1, 0)):
    """Plot comparison of scipy vs cfunc derivatives."""
    # Create test data
    X, Y, Z = create_test_function(func_type)
    
    # Fit spline
    tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.1)
    tx, ty, c = tck[0], tck[1], tck[2]
    
    # Create evaluation grid
    x_eval = np.linspace(-2, 2, 50)
    y_eval = np.linspace(-2, 2, 50)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    
    # Compute derivatives with scipy
    nux, nuy = derivative_order
    z_scipy = np.zeros_like(X_eval)
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            xi = np.array([x_eval[i]])
            yi = np.array([y_eval[j]])
            z_val, ier = dfitpack.parder(tx, ty, c, 3, 3, nux, nuy, xi, yi)
            if ier == 0:
                z_scipy[i, j] = z_val[0, 0]
    
    # Compute derivatives with cfunc
    z_cfunc = np.zeros_like(X_eval)
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            xi = np.array([x_eval[i]])
            yi = np.array([y_eval[j]])
            z_val, ier = call_parder_safe(tx, ty, c, 3, 3, nux, nuy, xi, yi)
            if ier == 0:
                z_cfunc[i, j] = z_val[0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original function
    im1 = axes[0, 0].contourf(X, Y, Z.T, levels=20, cmap='viridis')
    axes[0, 0].set_title(f'Original {func_type} function')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Scipy derivative
    im2 = axes[0, 1].contourf(X_eval, Y_eval, z_scipy.T, levels=20, cmap='viridis')
    axes[0, 1].set_title(f'Scipy derivative ({nux}, {nuy})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Cfunc derivative
    im3 = axes[0, 2].contourf(X_eval, Y_eval, z_cfunc.T, levels=20, cmap='viridis')
    axes[0, 2].set_title(f'Cfunc derivative ({nux}, {nuy})')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Difference plot
    diff = z_scipy - z_cfunc
    max_diff = np.max(np.abs(diff))
    im4 = axes[1, 0].contourf(X_eval, Y_eval, diff.T, levels=20, cmap='RdBu_r',
                               vmin=-max_diff, vmax=max_diff)
    axes[1, 0].set_title(f'Difference (scipy - cfunc)\nMax diff: {max_diff:.2e}')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Cross-section plots
    mid_idx = len(x_eval) // 2
    
    # X cross-section
    axes[1, 1].plot(x_eval, z_scipy[:, mid_idx], 'b-', label='Scipy', linewidth=2)
    axes[1, 1].plot(x_eval, z_cfunc[:, mid_idx], 'r--', label='Cfunc', linewidth=2)
    axes[1, 1].set_title(f'X cross-section at Y={y_eval[mid_idx]:.1f}')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel(f'Derivative ({nux}, {nuy})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Y cross-section
    axes[1, 2].plot(y_eval, z_scipy[mid_idx, :], 'b-', label='Scipy', linewidth=2)
    axes[1, 2].plot(y_eval, z_cfunc[mid_idx, :], 'r--', label='Cfunc', linewidth=2)
    axes[1, 2].set_title(f'Y cross-section at X={x_eval[mid_idx]:.1f}')
    axes[1, 2].set_xlabel('Y')
    axes[1, 2].set_ylabel(f'Derivative ({nux}, {nuy})')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Derivative Comparison: {func_type.capitalize()} Function\n'
                 f'Derivative order: ∂^{nux+nuy}/∂x^{nux}∂y^{nuy}', fontsize=16)
    plt.tight_layout()
    
    return fig


def create_all_derivative_plots():
    """Create plots for all derivative orders and function types."""
    # Test different functions
    function_types = ['gaussian', 'polynomial', 'sinusoidal', 'peaks']
    
    # Test different derivative orders
    derivative_orders = [
        (0, 0),  # Function value
        (1, 0),  # ∂/∂x
        (0, 1),  # ∂/∂y
        (2, 0),  # ∂²/∂x²
        (0, 2),  # ∂²/∂y²
        (1, 1),  # ∂²/∂x∂y
    ]
    
    # Create figure for each function type showing all derivatives
    for func_type in function_types:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (nux, nuy) in enumerate(derivative_orders):
            # Create test data
            X, Y, Z = create_test_function(func_type)
            
            # Fit spline
            tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=3, ky=3, s=0.1)
            tx, ty, c = tck[0], tck[1], tck[2]
            
            # Create evaluation grid
            x_eval = np.linspace(-2, 2, 40)
            y_eval = np.linspace(-2, 2, 40)
            X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
            
            # Compute derivatives with cfunc
            z_cfunc = np.zeros_like(X_eval)
            for i in range(len(x_eval)):
                for j in range(len(y_eval)):
                    xi = np.array([x_eval[i]])
                    yi = np.array([y_eval[j]])
                    z_val, ier = call_parder_safe(tx, ty, c, 3, 3, nux, nuy, xi, yi)
                    if ier == 0:
                        z_cfunc[i, j] = z_val[0]
            
            # Plot
            im = axes[idx].contourf(X_eval, Y_eval, z_cfunc.T, levels=20, cmap='viridis')
            
            # Title with derivative notation
            if nux == 0 and nuy == 0:
                title = 'f(x,y)'
            elif nux == 1 and nuy == 0:
                title = '∂f/∂x'
            elif nux == 0 and nuy == 1:
                title = '∂f/∂y'
            elif nux == 2 and nuy == 0:
                title = '∂²f/∂x²'
            elif nux == 0 and nuy == 2:
                title = '∂²f/∂y²'
            elif nux == 1 and nuy == 1:
                title = '∂²f/∂x∂y'
            
            axes[idx].set_title(title, fontsize=12)
            axes[idx].set_xlabel('X')
            axes[idx].set_ylabel('Y')
            plt.colorbar(im, ax=axes[idx])
        
        plt.suptitle(f'All Derivatives: {func_type.capitalize()} Function\n'
                     'Computed with FastSpline cfunc', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'derivatives_{func_type}.png', dpi=150, bbox_inches='tight')
        print(f"Saved derivatives_{func_type}.png")
    
    # Create detailed comparison for key derivatives
    for func_type in ['gaussian', 'peaks']:
        for deriv in [(1, 0), (0, 1), (1, 1)]:
            fig = plot_derivative_comparison(func_type, deriv)
            plt.savefig(f'comparison_{func_type}_d{deriv[0]}{deriv[1]}.png', 
                       dpi=150, bbox_inches='tight')
            print(f"Saved comparison_{func_type}_d{deriv[0]}{deriv[1]}.png")
            plt.close(fig)


if __name__ == "__main__":
    print("Creating derivative comparison plots...")
    create_all_derivative_plots()
    print("\nAll plots saved!")
    
    # Show one example interactively
    print("\nShowing example: Gaussian function, first derivative")
    fig = plot_derivative_comparison('gaussian', (1, 0))
    plt.show()