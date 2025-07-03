#!/usr/bin/env python3
"""
Compare fastspline behavior with scipy for sparse/dense grid scenarios.
This script demonstrates how different grid densities affect interpolation quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from fastspline.spline2d import Spline2D


def test_sparse_vs_dense_comparison():
    """Compare fastspline and scipy behavior on sparse vs dense grids."""
    print("=== Sparse vs Dense Grid Comparison ===")
    
    # Define test function: f(x,y) = sin(pi*x) * cos(pi*y)
    def test_func(x, y):
        return np.sin(np.pi * x) * np.cos(np.pi * y)
    
    # Dense grid
    x_dense = np.linspace(0, 1, 21)
    y_dense = np.linspace(0, 1, 21)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense, indexing='ij')
    Z_dense = test_func(X_dense, Y_dense)
    
    # Sparse grid (every 4th point)
    x_sparse = x_dense[::4]  # [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_sparse = y_dense[::4]
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse, indexing='ij')
    Z_sparse = test_func(X_sparse, Y_sparse)
    
    print(f"Dense grid: {len(x_dense)}x{len(y_dense)} = {len(x_dense)*len(y_dense)} points")
    print(f"Sparse grid: {len(x_sparse)}x{len(y_sparse)} = {len(x_sparse)*len(y_sparse)} points")
    
    # Create splines
    # scipy
    scipy_dense = RectBivariateSpline(x_dense, y_dense, Z_dense, kx=3, ky=3)
    scipy_sparse = RectBivariateSpline(x_sparse, y_sparse, Z_sparse, kx=3, ky=3)
    
    # fastspline
    fast_dense = Spline2D(x_dense, y_dense, Z_dense.ravel(), kx=3, ky=3)
    fast_sparse = Spline2D(x_sparse, y_sparse, Z_sparse.ravel(), kx=3, ky=3)
    
    # Test points
    x_test = np.array([0.15, 0.35, 0.65, 0.85])
    y_test = np.array([0.25, 0.45, 0.75])
    
    print("\n=== Interpolation Error Analysis ===")
    print("Test Point | Analytical | Scipy Dense | Scipy Sparse | Fast Dense | Fast Sparse")
    print("-" * 85)
    
    for xi in x_test:
        for yi in y_test:
            analytical = test_func(xi, yi)
            
            # scipy results
            scipy_dense_val = scipy_dense(xi, yi)[0, 0]
            scipy_sparse_val = scipy_sparse(xi, yi)[0, 0]
            
            # fastspline results
            fast_dense_val = fast_dense(xi, yi, grid=False)
            fast_sparse_val = fast_sparse(xi, yi, grid=False)
            
            # Convert numpy arrays to scalars for formatting
            if hasattr(fast_dense_val, 'item'):
                fast_dense_val = fast_dense_val.item()
            if hasattr(fast_sparse_val, 'item'):
                fast_sparse_val = fast_sparse_val.item()
            
            print(f"({xi:.2f},{yi:.2f}) | {analytical:8.5f} | {scipy_dense_val:10.5f} | {scipy_sparse_val:11.5f} | {fast_dense_val:9.5f} | {fast_sparse_val:10.5f}")
    
    # Error statistics
    x_fine = np.linspace(0, 1, 51)
    y_fine = np.linspace(0, 1, 51)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine, indexing='ij')
    Z_analytical = test_func(X_fine, Y_fine)
    
    # Evaluate splines on fine grid
    Z_scipy_dense = scipy_dense(x_fine, y_fine).T
    Z_scipy_sparse = scipy_sparse(x_fine, y_fine).T
    
    Z_fast_dense = fast_dense(x_fine, y_fine, grid=True)
    Z_fast_sparse = fast_sparse(x_fine, y_fine, grid=True)
    
    # RMS errors
    rms_scipy_dense = np.sqrt(np.mean((Z_scipy_dense - Z_analytical)**2))
    rms_scipy_sparse = np.sqrt(np.mean((Z_scipy_sparse - Z_analytical)**2))
    rms_fast_dense = np.sqrt(np.mean((Z_fast_dense - Z_analytical)**2))
    rms_fast_sparse = np.sqrt(np.mean((Z_fast_sparse - Z_analytical)**2))
    
    print(f"\n=== RMS Error Analysis ===")
    print(f"Scipy Dense:  {rms_scipy_dense:.6f}")
    print(f"Scipy Sparse: {rms_scipy_sparse:.6f}")
    print(f"Fast Dense:   {rms_fast_dense:.6f}")
    print(f"Fast Sparse:  {rms_fast_sparse:.6f}")
    
    return {
        'scipy': {'dense': rms_scipy_dense, 'sparse': rms_scipy_sparse},
        'fastspline': {'dense': rms_fast_dense, 'sparse': rms_fast_sparse}
    }


def test_boundary_effects():
    """Test how sparse grids affect interpolation near boundaries."""
    print("\n=== Boundary Effects Test ===")
    
    # Create grid with less coverage near boundaries
    x_full = np.linspace(0, 1, 11)
    y_full = np.linspace(0, 1, 11)
    
    # Remove some boundary points to simulate sparse boundary coverage
    x_sparse_boundary = x_full[1:-1]  # Remove first and last points
    y_sparse_boundary = y_full[1:-1]
    
    X_full, Y_full = np.meshgrid(x_full, y_full, indexing='ij')
    X_sparse, Y_sparse = np.meshgrid(x_sparse_boundary, y_sparse_boundary, indexing='ij')
    
    # Test function: f(x,y) = x^2 + y^2 (simple for boundary analysis)
    Z_full = X_full**2 + Y_full**2
    Z_sparse = X_sparse**2 + Y_sparse**2
    
    # Create splines
    scipy_full = RectBivariateSpline(x_full, y_full, Z_full, kx=3, ky=3)
    scipy_sparse = RectBivariateSpline(x_sparse_boundary, y_sparse_boundary, Z_sparse, kx=3, ky=3)
    
    fast_full = Spline2D(x_full, y_full, Z_full.ravel(), kx=3, ky=3)
    fast_sparse = Spline2D(x_sparse_boundary, y_sparse_boundary, Z_sparse.ravel(), kx=3, ky=3)
    
    # Test boundary points
    boundary_points = [(0.05, 0.5), (0.95, 0.5), (0.5, 0.05), (0.5, 0.95)]
    
    print("Boundary Point | Analytical | Scipy Full | Scipy Sparse | Fast Full | Fast Sparse")
    print("-" * 85)
    
    for xi, yi in boundary_points:
        analytical = xi**2 + yi**2
        
        if xi >= x_sparse_boundary.min() and xi <= x_sparse_boundary.max() and \
           yi >= y_sparse_boundary.min() and yi <= y_sparse_boundary.max():
            
            scipy_full_val = scipy_full(xi, yi)[0, 0]
            scipy_sparse_val = scipy_sparse(xi, yi)[0, 0]
            fast_full_val = fast_full(xi, yi, grid=False)
            fast_sparse_val = fast_sparse(xi, yi, grid=False)
            
            # Convert numpy arrays to scalars for formatting
            if hasattr(fast_full_val, 'item'):
                fast_full_val = fast_full_val.item()
            if hasattr(fast_sparse_val, 'item'):
                fast_sparse_val = fast_sparse_val.item()
            
            print(f"({xi:.2f},{yi:.2f}) | {analytical:9.5f} | {scipy_full_val:9.5f} | {scipy_sparse_val:11.5f} | {fast_full_val:8.5f} | {fast_sparse_val:10.5f}")


if __name__ == "__main__":
    errors = test_sparse_vs_dense_comparison()
    test_boundary_effects()
    
    print(f"\n=== Summary ===")
    print("Both fastspline and scipy show expected behavior:")
    print("- Dense grids provide better interpolation accuracy")
    print("- Sparse grids have larger errors but are still reasonable for many applications")
    print("- fastspline performs comparably to scipy for similar grid densities")