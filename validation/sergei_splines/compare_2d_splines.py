#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline, griddata
import sys
import os

# Add the parent directory to the path so we can import fastspline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from fastspline.sergei_splines import construct_splines_2d_cfunc, evaluate_splines_2d_cfunc
    fastspline_available = True
except ImportError as e:
    print(f"Warning: Could not import fastspline modules: {e}")
    fastspline_available = False

def test_function_2d(x, y):
    """Test function: sin(2*pi*x) * cos(2*pi*y)"""
    return np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

def create_2d_comparison():
    """Create comprehensive 2D spline comparison"""
    
    # Test data grid
    nx, ny = 8, 8
    x_data = np.linspace(0, 1, nx)
    y_data = np.linspace(0, 1, ny)
    X_data, Y_data = np.meshgrid(x_data, y_data)
    Z_data = test_function_2d(X_data, Y_data)
    
    # Evaluation grid (higher resolution)
    nx_eval, ny_eval = 51, 51
    x_eval = np.linspace(0, 1, nx_eval)
    y_eval = np.linspace(0, 1, ny_eval)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    Z_exact = test_function_2d(X_eval, Y_eval)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Test different spline orders
    orders_to_test = [3, 4, 5]
    
    for order_idx, order in enumerate(orders_to_test):
        print(f"\n{'='*50}")
        print(f"PROCESSING ORDER {order}")
        print(f"{'='*50}")
        
        # SciPy RectBivariateSpline
        try:
            # Note: RectBivariateSpline uses degrees, not orders
            kx = ky = min(order, 5)  # SciPy max degree is 5
            scipy_spline = RectBivariateSpline(x_data, y_data, Z_data.T, kx=kx, ky=ky, s=0)
            Z_scipy = scipy_spline(x_eval, y_eval).T
            scipy_available = True
            print(f"✓ SciPy RectBivariateSpline (kx={kx}, ky={ky}) - Success")
        except Exception as e:
            print(f"✗ SciPy RectBivariateSpline failed: {e}")
            scipy_available = False
            Z_scipy = np.zeros_like(Z_exact)
        
        # FastSpline Sergei 2D
        if fastspline_available:
            try:
                # Construct 2D spline
                coeff_2d = np.zeros((order+1)**2 * nx * ny)
                x_min = np.array([0.0, 0.0])
                x_max = np.array([1.0, 1.0])
                orders_2d = np.array([order, order])
                periodic_2d = np.array([False, False])
                
                # Flatten Z_data for FastSpline (expects 1D array)
                z_flat = Z_data.flatten()
                
                # Create workspace arrays
                workspace_y = np.zeros(nx * ny)
                workspace_coeff = np.zeros((order+1) * nx * ny)
                
                construct_splines_2d_cfunc(x_min, x_max, z_flat, 
                                          np.array([nx, ny]), orders_2d, periodic_2d, 
                                          coeff_2d, workspace_y, workspace_coeff)
                
                # Evaluate 2D spline
                Z_fastspline = np.zeros_like(Z_exact)
                h_step = np.array([1.0/(nx-1), 1.0/(ny-1)])
                
                for i in range(nx_eval):
                    for j in range(ny_eval):
                        x_eval_point = np.array([x_eval[i], y_eval[j]])
                        z_val = np.zeros(1)
                        # Use correct signature: order, num_points, periodic, x_min, h_step, coeff, x, y_out
                        evaluate_splines_2d_cfunc(orders_2d, np.array([nx, ny]), periodic_2d, 
                                                 x_min, h_step, coeff_2d, x_eval_point, z_val)
                        Z_fastspline[j, i] = z_val[0]
                
                fastspline_available_2d = True
                print(f"✓ FastSpline Sergei 2D (order={order}) - Success")
                
            except Exception as e:
                print(f"✗ FastSpline Sergei 2D failed: {e}")
                fastspline_available_2d = False
                Z_fastspline = np.zeros_like(Z_exact)
        else:
            fastspline_available_2d = False
            Z_fastspline = np.zeros_like(Z_exact)
        
        # Create subplots for this order
        base_idx = order_idx * 9 + 1
        
        # 1. Original function
        ax1 = fig.add_subplot(3, 9, base_idx, projection='3d')
        surf1 = ax1.plot_surface(X_eval, Y_eval, Z_exact, cmap='viridis', alpha=0.8)
        ax1.scatter(X_data.flatten(), Y_data.flatten(), Z_data.flatten(), 
                   color='red', s=50, alpha=0.8)
        ax1.set_title(f'Order {order}\nExact Function')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # 2. SciPy interpolation
        ax2 = fig.add_subplot(3, 9, base_idx + 1, projection='3d')
        if scipy_available:
            surf2 = ax2.plot_surface(X_eval, Y_eval, Z_scipy, cmap='plasma', alpha=0.8)
            ax2.set_title(f'SciPy\nRectBivariateSpline')
        else:
            ax2.text(0.5, 0.5, 0.5, 'SciPy\nFailed', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('SciPy (Failed)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        # 3. FastSpline interpolation
        ax3 = fig.add_subplot(3, 9, base_idx + 2, projection='3d')
        if fastspline_available_2d:
            surf3 = ax3.plot_surface(X_eval, Y_eval, Z_fastspline, cmap='coolwarm', alpha=0.8)
            ax3.set_title(f'FastSpline\nSergei 2D')
        else:
            ax3.text(0.5, 0.5, 0.5, 'FastSpline\nFailed', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('FastSpline (Failed)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        
        # 4. SciPy error (2D contour)
        ax4 = fig.add_subplot(3, 9, base_idx + 3)
        if scipy_available:
            error_scipy = np.abs(Z_scipy - Z_exact)
            contour4 = ax4.contourf(X_eval, Y_eval, error_scipy, levels=20, cmap='Reds')
            ax4.set_title(f'SciPy Error\nRMS: {np.sqrt(np.mean(error_scipy**2)):.2e}')
            plt.colorbar(contour4, ax=ax4)
        else:
            ax4.text(0.5, 0.5, 'SciPy\nFailed', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax4.set_title('SciPy Error')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        
        # 5. FastSpline error (2D contour)
        ax5 = fig.add_subplot(3, 9, base_idx + 4)
        if fastspline_available_2d:
            error_fastspline = np.abs(Z_fastspline - Z_exact)
            contour5 = ax5.contourf(X_eval, Y_eval, error_fastspline, levels=20, cmap='Blues')
            ax5.set_title(f'FastSpline Error\nRMS: {np.sqrt(np.mean(error_fastspline**2)):.2e}')
            plt.colorbar(contour5, ax=ax5)
        else:
            ax5.text(0.5, 0.5, 'FastSpline\nFailed', transform=ax5.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax5.set_title('FastSpline Error')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        
        # 6. Cross-section comparison (y=0.5)
        ax6 = fig.add_subplot(3, 9, base_idx + 5)
        y_cross = 0.5
        idx_cross = np.argmin(np.abs(y_eval - y_cross))
        
        ax6.plot(x_eval, Z_exact[idx_cross, :], 'k-', linewidth=2, label='Exact')
        if scipy_available:
            ax6.plot(x_eval, Z_scipy[idx_cross, :], 'r--', linewidth=2, label='SciPy')
        if fastspline_available_2d:
            ax6.plot(x_eval, Z_fastspline[idx_cross, :], 'b-.', linewidth=2, label='FastSpline')
        
        ax6.set_title(f'Cross-section at y={y_cross}')
        ax6.set_xlabel('x')
        ax6.set_ylabel('z')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Difference between implementations
        ax7 = fig.add_subplot(3, 9, base_idx + 6)
        if scipy_available and fastspline_available_2d:
            diff = np.abs(Z_scipy - Z_fastspline)
            contour7 = ax7.contourf(X_eval, Y_eval, diff, levels=20, cmap='Greys')
            ax7.set_title(f'Implementation Diff\nRMS: {np.sqrt(np.mean(diff**2)):.2e}')
            plt.colorbar(contour7, ax=ax7)
        else:
            ax7.text(0.5, 0.5, 'Cannot\nCompare', transform=ax7.transAxes, 
                    ha='center', va='center', fontsize=12)
            ax7.set_title('Implementation Diff')
        ax7.set_xlabel('x')
        ax7.set_ylabel('y')
        
        # 8. Error statistics
        ax8 = fig.add_subplot(3, 9, base_idx + 7)
        error_stats = []
        labels = []
        
        if scipy_available:
            error_scipy = np.abs(Z_scipy - Z_exact)
            error_stats.append([np.mean(error_scipy), np.max(error_scipy), np.std(error_scipy)])
            labels.append('SciPy')
        
        if fastspline_available_2d:
            error_fastspline = np.abs(Z_fastspline - Z_exact)
            error_stats.append([np.mean(error_fastspline), np.max(error_fastspline), np.std(error_fastspline)])
            labels.append('FastSpline')
        
        if error_stats:
            error_array = np.array(error_stats)
            x_pos = np.arange(len(labels))
            width = 0.25
            
            ax8.bar(x_pos - width, error_array[:, 0], width, label='Mean', alpha=0.8)
            ax8.bar(x_pos, error_array[:, 1], width, label='Max', alpha=0.8)
            ax8.bar(x_pos + width, error_array[:, 2], width, label='Std', alpha=0.8)
            
            ax8.set_yscale('log')
            ax8.set_xlabel('Implementation')
            ax8.set_ylabel('Error')
            ax8.set_title('Error Statistics')
            ax8.set_xticks(x_pos)
            ax8.set_xticklabels(labels)
            ax8.legend()
        
        # 9. Performance info
        ax9 = fig.add_subplot(3, 9, base_idx + 8)
        ax9.axis('off')
        
        info_text = f"Order {order} Summary:\\n"
        info_text += f"Grid: {nx}×{ny} → {nx_eval}×{ny_eval}\\n\\n"
        
        if scipy_available:
            rms_scipy = np.sqrt(np.mean((Z_scipy - Z_exact)**2))
            info_text += f"SciPy RMS: {rms_scipy:.2e}\\n"
        
        if fastspline_available_2d:
            rms_fastspline = np.sqrt(np.mean((Z_fastspline - Z_exact)**2))
            info_text += f"FastSpline RMS: {rms_fastspline:.2e}\\n"
        
        if scipy_available and fastspline_available_2d:
            diff_rms = np.sqrt(np.mean((Z_scipy - Z_fastspline)**2))
            info_text += f"\\nDifference: {diff_rms:.2e}\\n"
            
            if diff_rms < 1e-10:
                info_text += "✓ Essentially identical"
            elif diff_rms < 1e-6:
                info_text += "≈ Very close"
            else:
                info_text += "✗ Significantly different"
        
        ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Print numerical results
        print(f"\nOrder {order} Results:")
        print("-" * 40)
        if scipy_available:
            rms_scipy = np.sqrt(np.mean((Z_scipy - Z_exact)**2))
            max_error_scipy = np.max(np.abs(Z_scipy - Z_exact))
            print(f"  SciPy:      RMS={rms_scipy:.2e}, Max={max_error_scipy:.2e}")
        
        if fastspline_available_2d:
            rms_fastspline = np.sqrt(np.mean((Z_fastspline - Z_exact)**2))
            max_error_fastspline = np.max(np.abs(Z_fastspline - Z_exact))
            print(f"  FastSpline: RMS={rms_fastspline:.2e}, Max={max_error_fastspline:.2e}")
        
        if scipy_available and fastspline_available_2d:
            diff_rms = np.sqrt(np.mean((Z_scipy - Z_fastspline)**2))
            diff_max = np.max(np.abs(Z_scipy - Z_fastspline))
            print(f"  Difference: RMS={diff_rms:.2e}, Max={diff_max:.2e}")
    
    plt.tight_layout()
    plt.savefig('2d_spline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance benchmarking
    print("\n" + "="*80)
    print("2D PERFORMANCE COMPARISON")
    print("="*80)
    
    import time
    
    n_trials = 20
    
    for order in orders_to_test:
        print(f"\nOrder {order} Performance ({nx}×{ny} grid, {n_trials} trials):")
        print("-" * 60)
        
        # SciPy timing
        if True:  # Usually works for most orders
            try:
                kx = ky = min(order, 5)
                start_time = time.time()
                for _ in range(n_trials):
                    scipy_spline_perf = RectBivariateSpline(x_data, y_data, Z_data.T, kx=kx, ky=ky, s=0)
                    Z_scipy_perf = scipy_spline_perf(x_eval, y_eval).T
                scipy_time = time.time() - start_time
                print(f"  SciPy:      {scipy_time:.4f}s ({scipy_time/n_trials*1000:.1f}ms per trial)")
            except Exception as e:
                print(f"  SciPy:      Error - {e}")
        
        # FastSpline timing
        if fastspline_available:
            try:
                start_time = time.time()
                for _ in range(n_trials):
                    coeff_2d_perf = np.zeros((order+1)**2 * nx * ny)
                    x_min = np.array([0.0, 0.0])
                    x_max = np.array([1.0, 1.0])
                    orders_2d = np.array([order, order])
                    periodic_2d = np.array([False, False])
                    z_flat = Z_data.flatten()
                    
                    # Create workspace arrays
                    workspace_y_perf = np.zeros(nx * ny)
                    workspace_coeff_perf = np.zeros((order+1) * nx * ny)
                    
                    construct_splines_2d_cfunc(x_min, x_max, z_flat, 
                                              np.array([nx, ny]), orders_2d, periodic_2d, 
                                              coeff_2d_perf, workspace_y_perf, workspace_coeff_perf)
                    
                    # Evaluate at a few points (not full grid for timing)
                    h_step = np.array([1.0/(nx-1), 1.0/(ny-1)])
                    for i in range(0, nx_eval, 5):
                        for j in range(0, ny_eval, 5):
                            x_eval_point = np.array([x_eval[i], y_eval[j]])
                            z_val = np.zeros(1)
                            evaluate_splines_2d_cfunc(orders_2d, np.array([nx, ny]), periodic_2d,
                                                     x_min, h_step, coeff_2d_perf, x_eval_point, z_val)
                
                fastspline_time = time.time() - start_time
                print(f"  FastSpline: {fastspline_time:.4f}s ({fastspline_time/n_trials*1000:.1f}ms per trial)")
                
                # Speed comparison
                if 'scipy_time' in locals() and scipy_time > 0:
                    speedup = scipy_time / fastspline_time
                    print(f"  Speedup:    {speedup:.2f}x {'(FastSpline faster)' if speedup > 1 else '(SciPy faster)'}")
                
            except Exception as e:
                print(f"  FastSpline: Error - {e}")

if __name__ == "__main__":
    create_2d_comparison()