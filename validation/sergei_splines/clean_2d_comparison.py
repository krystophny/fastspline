#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import sys
import os

# Add the parent directory to the path so we can import fastspline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastspline.sergei_splines import construct_splines_2d_cfunc, evaluate_splines_2d_cfunc

def test_function(x, y):
    """Clean test function: sin(π*x) * cos(π*y)"""
    return np.sin(np.pi * x) * np.cos(np.pi * y)

def create_clean_2d_comparison():
    """Create focused 2D comparison with single test function"""
    
    # Data grid
    nx, ny = 8, 8
    x_data = np.linspace(0, 1, nx)
    y_data = np.linspace(0, 1, ny)
    X_data, Y_data = np.meshgrid(x_data, y_data)
    Z_data = test_function(X_data, Y_data)
    
    # Evaluation grid
    nx_eval, ny_eval = 41, 41
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    Z_exact = test_function(X_eval, Y_eval)
    
    # Create clean figure with 3 columns, 3 rows
    fig = plt.figure(figsize=(18, 12))
    
    # Test cubic splines only (most stable)
    order = 3
    
    print(f"2D Spline Comparison - Order {order}")
    print(f"Function: f(x,y) = sin(π*x) * cos(π*y)")
    print(f"Data grid: {nx}×{ny}, Evaluation: {nx_eval}×{ny_eval}")
    print("="*60)
    
    # SciPy implementation
    try:
        scipy_spline = RectBivariateSpline(x_data, y_data, Z_data.T, kx=order, ky=order, s=0)
        Z_scipy = scipy_spline(x_eval, y_eval).T
        scipy_success = True
        print("✓ SciPy RectBivariateSpline - Success")
    except Exception as e:
        print(f"✗ SciPy failed: {e}")
        scipy_success = False
        Z_scipy = np.zeros_like(Z_exact)
    
    # FastSpline implementation
    try:
        # Setup
        coeff_2d = np.zeros((order+1)**2 * nx * ny)
        x_min = np.array([0.0, 0.0])
        x_max = np.array([1.0, 1.0])
        orders_2d = np.array([order, order])
        periodic_2d = np.array([False, False])
        
        # Construct
        z_flat = Z_data.flatten()
        workspace_y = np.zeros(nx * ny)
        workspace_coeff = np.zeros((order+1) * nx * ny)
        
        construct_splines_2d_cfunc(x_min, x_max, z_flat, 
                                  np.array([nx, ny]), orders_2d, periodic_2d, 
                                  coeff_2d, workspace_y, workspace_coeff)
        
        # Evaluate
        Z_fastspline = np.zeros_like(Z_exact)
        h_step = np.array([1.0/(nx-1), 1.0/(ny-1)])
        
        for i in range(nx_eval):
            for j in range(ny_eval):
                x_eval_point = np.array([x_eval[i], y_eval[j]])
                z_val = np.zeros(1)
                evaluate_splines_2d_cfunc(orders_2d, np.array([nx, ny]), periodic_2d, 
                                         x_min, h_step, coeff_2d, x_eval_point, z_val)
                Z_fastspline[j, i] = z_val[0]
        
        fastspline_success = True
        print("✓ FastSpline Sergei 2D - Success")
        
    except Exception as e:
        print(f"✗ FastSpline failed: {e}")
        fastspline_success = False
        Z_fastspline = np.zeros_like(Z_exact)
    
    # Row 1: 3D Surface plots
    # Exact function
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X_eval, Y_eval, Z_exact, cmap='viridis', alpha=0.8)
    ax1.scatter(X_data.flatten(), Y_data.flatten(), Z_data.flatten(), 
               color='red', s=30, alpha=0.8)
    ax1.set_title('Exact Function\nf(x,y) = sin(πx)cos(πy)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # SciPy result
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    if scipy_success:
        surf2 = ax2.plot_surface(X_eval, Y_eval, Z_scipy, cmap='plasma', alpha=0.8)
        ax2.set_title('SciPy RectBivariateSpline')
    else:
        ax2.text(0.5, 0.5, 0.5, 'SciPy\nFailed', transform=ax2.transAxes, 
                ha='center', va='center', fontsize=12)
        ax2.set_title('SciPy (Failed)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    # FastSpline result
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    if fastspline_success:
        surf3 = ax3.plot_surface(X_eval, Y_eval, Z_fastspline, cmap='coolwarm', alpha=0.8)
        ax3.set_title('FastSpline Sergei 2D')
    else:
        ax3.text(0.5, 0.5, 0.5, 'FastSpline\nFailed', transform=ax3.transAxes, 
                ha='center', va='center', fontsize=12)
        ax3.set_title('FastSpline (Failed)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    
    # Row 2: Error contours
    ax4 = fig.add_subplot(3, 3, 4)
    if scipy_success:
        error_scipy = np.abs(Z_scipy - Z_exact)
        contour4 = ax4.contourf(X_eval, Y_eval, error_scipy, levels=20, cmap='Reds')
        ax4.set_title(f'SciPy Error\nRMS: {np.sqrt(np.mean(error_scipy**2)):.2e}')
        plt.colorbar(contour4, ax=ax4, shrink=0.8)
    else:
        ax4.text(0.5, 0.5, 'SciPy\nFailed', transform=ax4.transAxes, 
                ha='center', va='center', fontsize=12)
        ax4.set_title('SciPy Error')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    
    ax5 = fig.add_subplot(3, 3, 5)
    if fastspline_success:
        error_fastspline = np.abs(Z_fastspline - Z_exact)
        contour5 = ax5.contourf(X_eval, Y_eval, error_fastspline, levels=20, cmap='Blues')
        ax5.set_title(f'FastSpline Error\nRMS: {np.sqrt(np.mean(error_fastspline**2)):.2e}')
        plt.colorbar(contour5, ax=ax5, shrink=0.8)
    else:
        ax5.text(0.5, 0.5, 'FastSpline\nFailed', transform=ax5.transAxes, 
                ha='center', va='center', fontsize=12)
        ax5.set_title('FastSpline Error')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    
    # Difference between methods
    ax6 = fig.add_subplot(3, 3, 6)
    if scipy_success and fastspline_success:
        diff = np.abs(Z_scipy - Z_fastspline)
        contour6 = ax6.contourf(X_eval, Y_eval, diff, levels=20, cmap='Greys')
        ax6.set_title(f'Method Difference\nRMS: {np.sqrt(np.mean(diff**2)):.2e}')
        plt.colorbar(contour6, ax=ax6, shrink=0.8)
    else:
        ax6.text(0.5, 0.5, 'Cannot\nCompare', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Method Difference')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    
    # Row 3: Cross-sections
    y_cross = 0.5
    idx_cross = np.argmin(np.abs(y_eval - y_cross))
    
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(x_eval, Z_exact[idx_cross, :], 'k-', linewidth=3, label='Exact')
    if scipy_success:
        ax7.plot(x_eval, Z_scipy[idx_cross, :], 'r--', linewidth=2, label='SciPy')
    if fastspline_success:
        ax7.plot(x_eval, Z_fastspline[idx_cross, :], 'b-.', linewidth=2, label='FastSpline')
    ax7.set_title(f'Cross-section at y = {y_cross}')
    ax7.set_xlabel('x')
    ax7.set_ylabel('z')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Error comparison
    ax8 = fig.add_subplot(3, 3, 8)
    if scipy_success:
        error_scipy = np.abs(Z_scipy - Z_exact)
        ax8.semilogy(x_eval, error_scipy[idx_cross, :], 'r--', linewidth=2, label='SciPy Error')
    if fastspline_success:
        error_fastspline = np.abs(Z_fastspline - Z_exact)
        ax8.semilogy(x_eval, error_fastspline[idx_cross, :], 'b-.', linewidth=2, label='FastSpline Error')
    ax8.set_title(f'Error at y = {y_cross}')
    ax8.set_xlabel('x')
    ax8.set_ylabel('|Error|')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Summary statistics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"2D Spline Comparison Summary\\n"
    summary_text += f"{'='*35}\\n"
    summary_text += f"Function: sin(πx)cos(πy)\\n"
    summary_text += f"Order: {order} (cubic)\\n"
    summary_text += f"Grid: {nx}×{ny} → {nx_eval}×{ny_eval}\\n\\n"
    
    if scipy_success:
        rms_scipy = np.sqrt(np.mean((Z_scipy - Z_exact)**2))
        max_scipy = np.max(np.abs(Z_scipy - Z_exact))
        summary_text += f"SciPy Results:\\n"
        summary_text += f"  RMS Error: {rms_scipy:.2e}\\n"
        summary_text += f"  Max Error: {max_scipy:.2e}\\n\\n"
    
    if fastspline_success:
        rms_fastspline = np.sqrt(np.mean((Z_fastspline - Z_exact)**2))
        max_fastspline = np.max(np.abs(Z_fastspline - Z_exact))
        summary_text += f"FastSpline Results:\\n"
        summary_text += f"  RMS Error: {rms_fastspline:.2e}\\n"
        summary_text += f"  Max Error: {max_fastspline:.2e}\\n\\n"
    
    if scipy_success and fastspline_success:
        diff_rms = np.sqrt(np.mean((Z_scipy - Z_fastspline)**2))
        diff_max = np.max(np.abs(Z_scipy - Z_fastspline))
        summary_text += f"Method Difference:\\n"
        summary_text += f"  RMS: {diff_rms:.2e}\\n"
        summary_text += f"  Max: {diff_max:.2e}\\n\\n"
        
        if diff_rms < 1e-10:
            summary_text += "Status: ✓ Identical"
        elif diff_rms < 1e-6:
            summary_text += "Status: ≈ Very close"
        else:
            summary_text += "Status: ⚠ Different"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('clean_2d_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print console results
    print()
    print("Results:")
    print("-" * 40)
    if scipy_success:
        rms_scipy = np.sqrt(np.mean((Z_scipy - Z_exact)**2))
        max_scipy = np.max(np.abs(Z_scipy - Z_exact))
        print(f"SciPy:      RMS={rms_scipy:.2e}, Max={max_scipy:.2e}")
    
    if fastspline_success:
        rms_fastspline = np.sqrt(np.mean((Z_fastspline - Z_exact)**2))
        max_fastspline = np.max(np.abs(Z_fastspline - Z_exact))
        print(f"FastSpline: RMS={rms_fastspline:.2e}, Max={max_fastspline:.2e}")
    
    if scipy_success and fastspline_success:
        diff_rms = np.sqrt(np.mean((Z_scipy - Z_fastspline)**2))
        diff_max = np.max(np.abs(Z_scipy - Z_fastspline))
        print(f"Difference: RMS={diff_rms:.2e}, Max={diff_max:.2e}")
        
        if diff_rms < 1e-10:
            print("✓ Methods produce essentially identical results")
        elif diff_rms < 1e-6:
            print("≈ Methods produce very similar results")
        else:
            print("⚠ Methods produce different results")

if __name__ == "__main__":
    create_clean_2d_comparison()