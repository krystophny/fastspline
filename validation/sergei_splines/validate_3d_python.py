#!/usr/bin/env python3
"""
Validate 3D Sergei splines against Fortran reference values
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastspline.sergei_splines import (
    construct_splines_3d_cfunc, evaluate_splines_3d_cfunc,
    evaluate_splines_3d_der_cfunc
)

def test_function(x, y, z):
    """Test function: sin(πx) * cos(πy) * exp(-z/2)"""
    return np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-z/2.0)

def validate_3d_splines():
    """Main validation function for 3D splines"""
    
    # Grid parameters
    n1, n2, n3 = 8, 8, 8
    x_min = np.array([0.0, 0.0, 0.0])
    x_max = np.array([1.0, 1.0, 2.0])
    
    # Create grid with correct indexing
    x1 = np.linspace(x_min[0], x_max[0], n1)
    x2 = np.linspace(x_min[1], x_max[1], n2)
    x3 = np.linspace(x_min[2], x_max[2], n3)
    
    # IMPORTANT: Use indexing='ij' for consistency with Fortran
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    
    # Generate data
    Z_data = test_function(X1, X2, X3)
    
    # Debug: Check data shape and values
    print(f"Debug: Z_data shape = {Z_data.shape}")
    print(f"Debug: First few Z_data values (before flatten):")
    print(f"  Z_data[0,0,0] = {Z_data[0,0,0]:.10f}")
    print(f"  Z_data[1,0,0] = {Z_data[1,0,0]:.10f}")
    print(f"  Z_data[0,1,0] = {Z_data[0,1,0]:.10f}")
    print(f"  Z_data[0,0,1] = {Z_data[0,0,1]:.10f}")
    
    print("3D Spline Validation - Python FastSpline")
    print("=" * 50)
    print(f"Test function: f(x,y,z) = sin(πx) * cos(πy) * exp(-z/2)")
    print(f"Grid size: {n1} x {n2} x {n3}")
    print(f"Domain: [0,1] x [0,1] x [0,2]")
    print()
    
    # Flatten data for spline construction
    z_flat = Z_data.flatten()
    
    # Print first few data points to compare with Fortran
    print("First 10 data points (flattened):")
    for i in range(min(10, len(z_flat))):
        print(f"  z_flat[{i}] = {z_flat[i]:.10f}")
    print()
    
    # Expected values from Fortran
    fortran_values = {
        (0.5, 0.5, 1.0): 0.0000000000,
        (0.25, 0.75, 0.5): -0.3894003915,
        (0.8, 0.3, 1.5): 0.1631986302
    }
    
    # Test points
    test_points = np.array([
        [0.5, 0.5, 1.0],
        [0.25, 0.75, 0.5],
        [0.8, 0.3, 1.5],
        [0.1, 0.9, 0.2],
        [0.6, 0.4, 1.8]
    ])
    
    # Test different orders
    for order in [3, 4, 5]:
        print(f"\nOrder: {order}")
        print("-" * 20)
        
        # Prepare arrays
        orders = np.array([order, order, order])
        num_points = np.array([n1, n2, n3])
        periodic = np.array([0, 0, 0], dtype=np.int32)  # Non-periodic
        
        # Calculate coefficient array size
        coeff_size = (order+1)**3 * n1 * n2 * n3
        coeff = np.zeros(coeff_size)
        
        # Workspace arrays
        # For 3D construction, we need larger workspaces
        max_n = max(n1, n2, n3)
        max_n2 = max(n1*n2, n2*n3, n1*n3)
        
        workspace_1d_size = max_n
        workspace_1d = np.zeros(workspace_1d_size)
        workspace_1d_coeff_size = (order+1) * max_n
        workspace_1d_coeff = np.zeros(workspace_1d_coeff_size)
        # Need to accommodate the full 2D intermediate result
        workspace_2d_coeff_size = (order+1)**2 * n1 * n2 * n3
        workspace_2d_coeff = np.zeros(workspace_2d_coeff_size)
        
        # Construct splines
        construct_splines_3d_cfunc(x_min, x_max, z_flat, num_points, orders, periodic,
                                 coeff, workspace_1d, workspace_1d_coeff, workspace_2d_coeff)
        
        # Print first few coefficients for comparison
        if order == 3:
            print("\nFirst 10 coefficients (for Fortran comparison):")
            for i in range(min(10, coeff_size)):
                print(f"  coeff[{i}] = {coeff[i]:.10f}")
        
        # Evaluate at test points
        h_step = np.array([(x_max[0]-x_min[0])/(n1-1), 
                          (x_max[1]-x_min[1])/(n2-1),
                          (x_max[2]-x_min[2])/(n3-1)])
        
        max_error = 0.0
        rms_error = 0.0
        
        print("\nTest point evaluations:")
        for i, point in enumerate(test_points):
            # Exact value
            exact_val = test_function(point[0], point[1], point[2])
            
            # Spline evaluation
            y_out = np.zeros(1)
            evaluate_splines_3d_cfunc(orders, num_points, periodic, x_min, h_step,
                                    coeff, point, y_out)
            spline_val = y_out[0]
            
            # Error
            error = abs(spline_val - exact_val)
            max_error = max(max_error, error)
            rms_error += error**2
            
            # Check against Fortran if available
            fortran_key = tuple(point[:3])
            if fortran_key in fortran_values:
                fortran_val = fortran_values[fortran_key]
                fortran_error = abs(exact_val - fortran_val)
                print(f"  Point ({point[0]:.4f},{point[1]:.4f},{point[2]:.4f}): "
                      f"exact={exact_val:.10f}, spline={spline_val:.10f}, "
                      f"error={error:.4e}")
                print(f"    Fortran: exact={fortran_val:.10f}, error={fortran_error:.4e}")
            else:
                print(f"  Point ({point[0]:.4f},{point[1]:.4f},{point[2]:.4f}): "
                      f"exact={exact_val:.10f}, spline={spline_val:.10f}, "
                      f"error={error:.4e}")
        
        rms_error = np.sqrt(rms_error / len(test_points))
        print(f"\nMax error: {max_error:.4e}, RMS error: {rms_error:.4e}")
    
    # Test derivatives for order 3
    print("\n" + "="*50)
    print("Testing 3D Derivatives (Order 3)")
    print("="*50)
    
    order = 3
    orders = np.array([order, order, order])
    periodic = np.array([0, 0, 0], dtype=np.int32)
    
    # Reconstruct spline
    coeff_size = (order+1)**3 * n1 * n2 * n3
    coeff = np.zeros(coeff_size)
    construct_splines_3d_cfunc(x_min, x_max, z_flat, num_points, orders, periodic,
                             coeff, workspace_1d, workspace_1d_coeff, workspace_2d_coeff)
    
    # Test derivatives at a point
    test_point = np.array([0.5, 0.5, 1.0])
    y_out = np.zeros(1)
    dydx1_out = np.zeros(1)
    dydx2_out = np.zeros(1)
    dydx3_out = np.zeros(1)
    
    evaluate_splines_3d_der_cfunc(orders, num_points, periodic, x_min, h_step,
                                coeff, test_point, y_out, dydx1_out, dydx2_out, dydx3_out)
    
    print(f"\nDerivatives at point ({test_point[0]}, {test_point[1]}, {test_point[2]}):")
    print(f"  f = {y_out[0]:.10f}")
    print(f"  ∂f/∂x = {dydx1_out[0]:.10f}")
    print(f"  ∂f/∂y = {dydx2_out[0]:.10f}")
    print(f"  ∂f/∂z = {dydx3_out[0]:.10f}")
    
    # Analytical derivatives at this point
    x, y, z = test_point
    exact_f = test_function(x, y, z)
    exact_dfdx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y) * np.exp(-z/2.0)
    exact_dfdy = -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-z/2.0)
    exact_dfdz = -0.5 * np.sin(np.pi * x) * np.cos(np.pi * y) * np.exp(-z/2.0)
    
    print(f"\nExact derivatives:")
    print(f"  f = {exact_f:.10f}")
    print(f"  ∂f/∂x = {exact_dfdx:.10f}")
    print(f"  ∂f/∂y = {exact_dfdy:.10f}")
    print(f"  ∂f/∂z = {exact_dfdz:.10f}")
    
    print(f"\nDerivative errors:")
    print(f"  |Δf| = {abs(y_out[0] - exact_f):.4e}")
    print(f"  |Δ(∂f/∂x)| = {abs(dydx1_out[0] - exact_dfdx):.4e}")
    print(f"  |Δ(∂f/∂y)| = {abs(dydx2_out[0] - exact_dfdy):.4e}")
    print(f"  |Δ(∂f/∂z)| = {abs(dydx3_out[0] - exact_dfdz):.4e}")

def create_3d_visualization():
    """Create visualization of 3D spline interpolation"""
    
    print("\n" + "="*50)
    print("Creating 3D Visualization")
    print("="*50)
    
    # Use smaller grid for visualization
    n1, n2, n3 = 6, 6, 6
    x_min = np.array([0.0, 0.0, 0.0])
    x_max = np.array([1.0, 1.0, 2.0])
    
    # Create grid
    x1 = np.linspace(x_min[0], x_max[0], n1)
    x2 = np.linspace(x_min[1], x_max[1], n2)
    x3 = np.linspace(x_min[2], x_max[2], n3)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    Z_data = test_function(X1, X2, X3)
    
    # Construct spline
    order = 3
    orders = np.array([order, order, order])
    num_points = np.array([n1, n2, n3])
    periodic = np.array([0, 0, 0], dtype=np.int32)
    
    coeff_size = (order+1)**3 * n1 * n2 * n3
    coeff = np.zeros(coeff_size)
    
    workspace_1d_size = max(n1, n2, n3)
    workspace_1d = np.zeros(workspace_1d_size)
    workspace_1d_coeff_size = (order+1) * max(n1, n2, n3)
    workspace_1d_coeff = np.zeros(workspace_1d_coeff_size)
    workspace_2d_coeff_size = (order+1)**2 * n1 * n2 * n3
    workspace_2d_coeff = np.zeros(workspace_2d_coeff_size)
    
    z_flat = Z_data.flatten()
    construct_splines_3d_cfunc(x_min, x_max, z_flat, num_points, orders, periodic,
                             coeff, workspace_1d, workspace_1d_coeff, workspace_2d_coeff)
    
    # Create evaluation grid
    n_eval = 20
    x1_eval = np.linspace(x_min[0], x_max[0], n_eval)
    x2_eval = np.linspace(x_min[1], x_max[1], n_eval)
    
    # Create slices at different z values
    z_slices = [0.5, 1.0, 1.5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    h_step = np.array([(x_max[0]-x_min[0])/(n1-1), 
                      (x_max[1]-x_min[1])/(n2-1),
                      (x_max[2]-x_min[2])/(n3-1)])
    
    for idx, z_val in enumerate(z_slices):
        # Exact values
        X1_eval, X2_eval = np.meshgrid(x1_eval, x2_eval, indexing='ij')
        Z_exact = test_function(X1_eval, X2_eval, z_val)
        
        # Spline values
        Z_spline = np.zeros_like(Z_exact)
        for i in range(n_eval):
            for j in range(n_eval):
                point = np.array([x1_eval[i], x2_eval[j], z_val])
                y_out = np.zeros(1)
                evaluate_splines_3d_cfunc(orders, num_points, periodic, x_min, h_step,
                                        coeff, point, y_out)
                Z_spline[i, j] = y_out[0]
        
        # Plot exact
        ax = axes[idx]
        im = ax.contourf(X1_eval, X2_eval, Z_exact, levels=20, cmap='viridis')
        ax.set_title(f'Exact at z={z_val}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        
        # Plot spline
        ax = axes[idx+3]
        im = ax.contourf(X1_eval, X2_eval, Z_spline, levels=20, cmap='viridis')
        ax.set_title(f'Spline at z={z_val}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('3D Spline Interpolation: Slices at Different z-values', fontsize=16)
    plt.tight_layout()
    plt.savefig('3d_spline_validation.png', dpi=150, bbox_inches='tight')
    print("Saved: 3d_spline_validation.png")
    
    # Create error plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Evaluate error over entire domain
    n_test = 15
    x_test = np.linspace(x_min[0], x_max[0], n_test)
    y_test = np.linspace(x_min[1], x_max[1], n_test)
    z_test = np.linspace(x_min[2], x_max[2], n_test)
    
    errors = []
    for x in x_test:
        for y in y_test:
            for z in z_test:
                exact = test_function(x, y, z)
                point = np.array([x, y, z])
                y_out = np.zeros(1)
                evaluate_splines_3d_cfunc(orders, num_points, periodic, x_min, h_step,
                                        coeff, point, y_out)
                error = abs(y_out[0] - exact)
                errors.append(error)
    
    errors = np.array(errors)
    ax.hist(np.log10(errors + 1e-16), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('log10(|Error|)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'3D Spline Error Distribution (n={n_test}³ test points)')
    
    print(f"\nError statistics over {n_test}³ = {n_test**3} points:")
    print(f"  Max error: {np.max(errors):.4e}")
    print(f"  Mean error: {np.mean(errors):.4e}")
    print(f"  RMS error: {np.sqrt(np.mean(errors**2)):.4e}")
    
    plt.tight_layout()
    plt.savefig('3d_spline_error_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: 3d_spline_error_distribution.png")

if __name__ == "__main__":
    validate_3d_splines()
    # create_3d_visualization()  # Skip for now due to quintic errors
    # plt.show()