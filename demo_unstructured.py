#!/usr/bin/env python3
"""
Demonstration of unstructured data support in fastspline.

This script shows how the Spline2D class can now handle scattered data points
similar to scipy.interpolate.bisplrep.
"""

import numpy as np
import matplotlib.pyplot as plt
from fastspline.spline2d import Spline2D

def demo_unstructured_data():
    """Demonstrate unstructured data interpolation."""
    print("=== Unstructured Data Interpolation Demo ===")
    
    # Generate scattered data points
    np.random.seed(42)
    n_points = 30
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    z = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 0.1 * np.random.randn(n_points)
    
    print(f"Input data: {n_points} scattered points")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"z range: [{z.min():.3f}, {z.max():.3f}]")
    
    # Create spline from unstructured data
    print("\nCreating spline from unstructured data...")
    spline = Spline2D(x, y, z, kx=3, ky=3)
    print(f"Spline created successfully (unstructured: {spline.is_unstructured})")
    
    # Create evaluation grid
    x_eval = np.linspace(0, 1, 21)
    y_eval = np.linspace(0, 1, 21)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    
    # Evaluate spline on grid
    print("Evaluating spline on regular grid...")
    Z_eval = spline(x_eval, y_eval, grid=True)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot scattered data
    scatter = ax1.scatter(x, y, c=z, cmap='viridis', s=50, edgecolors='black')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Scattered Input Data')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1)
    
    # Plot interpolated surface
    contour = ax2.contourf(X_eval, Y_eval, Z_eval, levels=20, cmap='viridis')
    ax2.scatter(x, y, c='white', s=20, edgecolors='black', alpha=0.7)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Interpolated Surface')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('unstructured_spline_demo.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'unstructured_spline_demo.png'")
    
    # Test evaluation at specific points
    print("\nTesting evaluation at specific points:")
    test_points = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for xi, yi in test_points:
        result = spline(xi, yi, grid=False)
        analytical = np.sin(2 * np.pi * xi) * np.cos(2 * np.pi * yi)
        print(f"  ({xi}, {yi}): spline={float(result):.4f}, analytical={analytical:.4f}")

def compare_structured_vs_unstructured():
    """Compare structured vs unstructured data handling."""
    print("\n=== Structured vs Unstructured Comparison ===")
    
    # Create structured grid
    x_grid = np.linspace(0, 1, 6)
    y_grid = np.linspace(0, 1, 6)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y)
    
    # Create structured spline
    spline_structured = Spline2D(x_grid, y_grid, Z.ravel(), kx=3, ky=3)
    print(f"Structured spline created (unstructured: {spline_structured.is_unstructured})")
    
    # Convert to unstructured format
    x_unstructured = X.ravel()
    y_unstructured = Y.ravel()
    z_unstructured = Z.ravel()
    
    # Create unstructured spline
    spline_unstructured = Spline2D(x_unstructured, y_unstructured, z_unstructured, kx=3, ky=3)
    print(f"Unstructured spline created (unstructured: {spline_unstructured.is_unstructured})")
    
    # Compare results at test points
    test_points = [(0.2, 0.3), (0.7, 0.8)]
    print("\nComparison at test points:")
    for xi, yi in test_points:
        result_structured = spline_structured(xi, yi, grid=False)
        result_unstructured = spline_unstructured(xi, yi, grid=False)
        analytical = np.sin(np.pi * xi) * np.cos(np.pi * yi)
        
        print(f"  ({xi}, {yi}):")
        print(f"    Structured:   {float(result_structured):.6f}")
        print(f"    Unstructured: {float(result_unstructured):.6f}")
        print(f"    Analytical:   {analytical:.6f}")
        print(f"    Difference:   {abs(float(result_structured) - float(result_unstructured)):.6f}")

if __name__ == "__main__":
    demo_unstructured_data()
    compare_structured_vs_unstructured()
    print("\nDemo completed successfully!")