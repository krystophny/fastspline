#!/usr/bin/env python3
"""Visual test of 2D spline interpolation using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fastspline import Spline2D


def test_2d_surface_interpolation():
    """Visual test of 2D surface interpolation."""
    # Create coarse grid
    x = np.linspace(0, 2*np.pi, 8)
    y = np.linspace(0, np.pi, 6)
    
    # Create test surface: f(x,y) = sin(x)*cos(y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.sin(X) * np.cos(Y)
    
    # Create spline
    spline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)
    
    # Create fine grid for evaluation
    x_fine = np.linspace(0, 2*np.pi, 50)
    y_fine = np.linspace(0, np.pi, 30)
    
    # Evaluate spline on fine grid
    Z_fine = spline(x_fine, y_fine, grid=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data points in 3D
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X.ravel(), Y.ravel(), Z.ravel(), c='red', s=50, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Original Data Points')
    
    # Plot 2: Interpolated surface
    ax2 = fig.add_subplot(132, projection='3d')
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    ax2.plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis', alpha=0.8)
    ax2.scatter(X.ravel(), Y.ravel(), Z.ravel(), c='red', s=30)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('Cubic Spline Interpolation')
    
    # Plot 3: Contour plot
    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(X_fine, Y_fine, Z_fine, levels=20, cmap='viridis')
    ax3.scatter(X.ravel(), Y.ravel(), c='red', s=30, alpha=0.6)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('2D Spline Contours')
    plt.colorbar(contour, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('2d_surface_interpolation.png', dpi=150)
    plt.close()
    print("2D surface interpolation plot saved as '2d_surface_interpolation.png'")


def test_comparison_with_exact_function():
    """Compare spline interpolation with exact function."""
    # Create test function: f(x,y) = exp(-((x-1)^2 + (y-1)^2)/2)
    x = np.linspace(0, 2, 9)
    y = np.linspace(0, 2, 7)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z_exact = np.exp(-((X-1)**2 + (Y-1)**2)/2)
    
    # Create splines with different orders
    spline_linear = Spline2D(x, y, Z_exact.ravel(), kx=1, ky=1)
    spline_cubic = Spline2D(x, y, Z_exact.ravel(), kx=3, ky=3)
    
    # Evaluate on fine grid
    x_fine = np.linspace(0, 2, 40)
    y_fine = np.linspace(0, 2, 40)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    Z_exact_fine = np.exp(-((X_fine-1)**2 + (Y_fine-1)**2)/2)
    
    Z_linear = spline_linear(x_fine, y_fine, grid=True)
    Z_cubic = spline_cubic(x_fine, y_fine, grid=True)
    
    # Create figure with comparisons
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: Functions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot_surface(X_fine, Y_fine, Z_exact_fine, cmap='viridis', alpha=0.8)
    ax1.set_title('Exact Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot_surface(X_fine, Y_fine, Z_linear, cmap='viridis', alpha=0.8)
    ax2.scatter(X.ravel(), Y.ravel(), Z_exact.ravel(), c='red', s=20)
    ax2.set_title('Linear Spline')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot_surface(X_fine, Y_fine, Z_cubic, cmap='viridis', alpha=0.8)
    ax3.scatter(X.ravel(), Y.ravel(), Z_exact.ravel(), c='red', s=20)
    ax3.set_title('Cubic Spline')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    
    # Row 2: Errors
    error_linear = np.abs(Z_linear - Z_exact_fine)
    error_cubic = np.abs(Z_cubic - Z_exact_fine)
    
    ax4 = fig.add_subplot(2, 3, 4)
    im1 = ax4.imshow(error_linear.T, origin='lower', cmap='hot', 
                     extent=[0, 2, 0, 2], aspect='auto')
    ax4.set_title(f'Linear Error (max={error_linear.max():.4f})')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im1, ax=ax4)
    
    ax5 = fig.add_subplot(2, 3, 5)
    im2 = ax5.imshow(error_cubic.T, origin='lower', cmap='hot', 
                     extent=[0, 2, 0, 2], aspect='auto')
    ax5.set_title(f'Cubic Error (max={error_cubic.max():.4f})')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(im2, ax=ax5)
    
    # Error comparison plot
    ax6 = fig.add_subplot(2, 3, 6)
    x_slice = len(x_fine) // 2
    ax6.plot(y_fine, Z_exact_fine[x_slice, :], 'k-', label='Exact', linewidth=2)
    ax6.plot(y_fine, Z_linear[x_slice, :], 'b--', label='Linear', linewidth=2)
    ax6.plot(y_fine, Z_cubic[x_slice, :], 'r:', label='Cubic', linewidth=2)
    ax6.set_xlabel('y')
    ax6.set_ylabel('z')
    ax6.set_title(f'Cross-section at x={x_fine[x_slice]:.2f}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_spline_comparison.png', dpi=150)
    plt.close()
    print("Spline comparison plot saved as '2d_spline_comparison.png'")


def test_derivatives_visualization():
    """Visualize 2D spline derivatives."""
    # Create smooth test function
    x = np.linspace(0, 2*np.pi, 10)
    y = np.linspace(0, 2*np.pi, 8)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.sin(X) * np.sin(Y)
    
    spline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)
    
    # Evaluate function and derivatives on fine grid
    x_fine = np.linspace(0, 2*np.pi, 50)
    y_fine = np.linspace(0, 2*np.pi, 40)
    
    Z_fine = spline(x_fine, y_fine, grid=True)
    dZ_dx = spline(x_fine, y_fine, dx=1, dy=0, grid=True)
    dZ_dy = spline(x_fine, y_fine, dx=0, dy=1, grid=True)
    
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    
    # Create figure with derivatives
    fig = plt.figure(figsize=(15, 5))
    
    # Function
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('Function: sin(x)sin(y)')
    
    # X-derivative
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_fine, Y_fine, dZ_dx, cmap='coolwarm', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('∂f/∂x')
    ax2.set_title('∂f/∂x = cos(x)sin(y)')
    
    # Y-derivative
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X_fine, Y_fine, dZ_dy, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('∂f/∂y')
    ax3.set_title('∂f/∂y = sin(x)cos(y)')
    
    plt.tight_layout()
    plt.savefig('2d_derivatives.png', dpi=150)
    plt.close()
    print("Derivatives plot saved as '2d_derivatives.png'")
    
    # Also create a 2D view with contours
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Function contours
    c1 = axes[0].contourf(X_fine, Y_fine, Z_fine, levels=20, cmap='viridis')
    axes[0].set_title('f(x,y) = sin(x)sin(y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(c1, ax=axes[0])
    
    # X-derivative contours
    c2 = axes[1].contourf(X_fine, Y_fine, dZ_dx, levels=20, cmap='coolwarm')
    axes[1].set_title('∂f/∂x')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(c2, ax=axes[1])
    
    # Y-derivative contours
    c3 = axes[2].contourf(X_fine, Y_fine, dZ_dy, levels=20, cmap='coolwarm')
    axes[2].set_title('∂f/∂y')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(c3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('2d_derivatives_contours.png', dpi=150)
    plt.close()
    print("Derivatives contour plot saved as '2d_derivatives_contours.png'")


def test_missing_data_visualization():
    """Visualize handling of missing data."""
    # Create grid with holes
    x = np.linspace(0, 3, 12)
    y = np.linspace(0, 2, 10)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.sin(X) * np.exp(-Y/2)
    
    # Introduce missing data (NaN values)
    Z_with_holes = Z.copy()
    # Create a hole in the middle
    Z_with_holes[4:7, 3:6] = np.nan
    # Random missing points
    np.random.seed(42)
    mask = np.random.random(Z.shape) < 0.1
    Z_with_holes[mask] = np.nan
    
    # Create spline (it will handle NaN values)
    spline = Spline2D(x, y, Z_with_holes.ravel(), kx=3, ky=3)
    
    # Evaluate on fine grid
    x_fine = np.linspace(0, 3, 60)
    y_fine = np.linspace(0, 2, 40)
    Z_interp = spline(x_fine, y_fine, grid=True)
    
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    Z_exact_fine = np.sin(X_fine) * np.exp(-Y_fine/2)
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Original data with holes
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(Z_with_holes.T, origin='lower', cmap='viridis', 
                     extent=[0, 3, 0, 2], aspect='auto')
    ax1.set_title('Original Data with Missing Values')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    # Interpolated result
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(Z_interp.T, origin='lower', cmap='viridis', 
                     extent=[0, 3, 0, 2], aspect='auto')
    ax2.set_title('Spline Interpolation Result')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)
    
    # Error compared to exact
    ax3 = fig.add_subplot(133)
    error = np.abs(Z_interp - Z_exact_fine)
    im3 = ax3.imshow(error.T, origin='lower', cmap='hot', 
                     extent=[0, 3, 0, 2], aspect='auto')
    ax3.set_title(f'Interpolation Error (max={error.max():.4f})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('2d_missing_data.png', dpi=150)
    plt.close()
    print("Missing data handling plot saved as '2d_missing_data.png'")


def test_performance_visualization():
    """Visualize performance characteristics."""
    import time
    
    # Test different grid sizes
    grid_sizes = [(10, 8), (20, 15), (30, 25), (50, 40), (70, 55)]
    construction_times = []
    evaluation_times = []
    total_points = []
    
    for nx, ny in grid_sizes:
        # Create test data
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
        
        # Time spline construction
        start_time = time.time()
        spline = Spline2D(x, y, Z.ravel(), kx=3, ky=3)
        construction_times.append(time.time() - start_time)
        
        # Time evaluation
        x_eval = np.linspace(0, 1, 50)
        y_eval = np.linspace(0, 1, 50)
        
        start_time = time.time()
        Z_interp = spline(x_eval, y_eval, grid=True)
        evaluation_times.append(time.time() - start_time)
        
        total_points.append(nx * ny)
    
    # Create performance plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Construction time
    ax1.loglog(total_points, construction_times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Total Grid Points')
    ax1.set_ylabel('Construction Time (s)')
    ax1.set_title('Spline Construction Performance')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (n, t) in enumerate(zip(total_points, construction_times)):
        ax1.annotate(f'{grid_sizes[i][0]}×{grid_sizes[i][1]}', 
                    (n, t), xytext=(5, 5), textcoords='offset points')
    
    # Evaluation time
    ax2.plot(total_points, np.array(evaluation_times)*1000, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Total Grid Points')
    ax2.set_ylabel('Evaluation Time (ms)')
    ax2.set_title('Evaluation on 50×50 Grid')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_performance.png', dpi=150)
    plt.close()
    print("Performance plot saved as '2d_performance.png'")
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("Grid Size\tConstruction(ms)\tEvaluation(ms)")
    print("-" * 45)
    for (nx, ny), ct, et in zip(grid_sizes, construction_times, evaluation_times):
        print(f"{nx}×{ny}\t\t{ct*1000:.2f}\t\t{et*1000:.2f}")


if __name__ == "__main__":
    print("Running 2D spline visual tests with matplotlib...")
    
    test_2d_surface_interpolation()
    test_comparison_with_exact_function()
    test_derivatives_visualization()
    test_missing_data_visualization()
    test_performance_visualization()
    
    print("\nAll 2D visual tests completed!")
    print("Generated plots:")
    print("- 2d_surface_interpolation.png: 3D surface and contour visualization")
    print("- 2d_spline_comparison.png: Linear vs cubic spline comparison")
    print("- 2d_derivatives.png: 3D derivative visualization")
    print("- 2d_derivatives_contours.png: 2D contour plots of derivatives")
    print("- 2d_missing_data.png: Handling of missing data (NaN values)")
    print("- 2d_performance.png: Performance characteristics")