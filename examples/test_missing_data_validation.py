#!/usr/bin/env python3
"""Test missing data handling with holes in the data and validate against scipy."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev, bisplev_scalar

def create_data_with_holes():
    """Create test data with large holes (missing regions)."""
    # Create a regular grid
    x_grid = np.linspace(-2, 2, 50)
    y_grid = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create a test function
    Z = np.exp(-(X**2 + Y**2)) * np.cos(np.pi * X) * np.sin(np.pi * Y)
    
    # Create holes by removing data in specific regions
    # Hole 1: Circle in the center
    center_hole = (X**2 + Y**2) < 0.5**2
    
    # Hole 2: Rectangle in upper right
    rect_hole = (X > 0.5) & (X < 1.5) & (Y > 0.5) & (Y < 1.5)
    
    # Hole 3: Triangle-like region in lower left
    triangle_hole = (X < -0.5) & (Y < -0.5) & (Y > X - 0.5)
    
    # Combine holes
    mask = center_hole | rect_hole | triangle_hole
    
    # Convert to scattered data format, removing holes
    x_data = X[~mask].flatten()
    y_data = Y[~mask].flatten()
    z_data = Z[~mask].flatten()
    
    print(f"Original grid: {X.size} points")
    print(f"After removing holes: {len(x_data)} points ({len(x_data)/X.size*100:.1f}%)")
    print(f"Removed {np.sum(mask)} points in holes")
    
    return x_data, y_data, z_data, X, Y, Z, mask

def test_missing_data_accuracy():
    """Test that FastSpline matches SciPy with missing data."""
    print("Testing missing data accuracy...")
    
    # Create data with holes
    x_data, y_data, z_data, X_orig, Y_orig, Z_orig, mask = create_data_with_holes()
    
    # Create splines with both libraries
    print("Fitting SciPy spline...")
    tck_scipy = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)  # Small smoothing to avoid warnings
    
    print("Fitting FastSpline spline...")
    tck_fast = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    
    # Test evaluation on a regular grid (including hole regions)
    x_test = np.linspace(-1.8, 1.8, 30)
    y_test = np.linspace(-1.8, 1.8, 30)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    x_eval = X_test.flatten()
    y_eval = Y_test.flatten()
    
    print(f"Evaluating at {len(x_eval)} test points...")
    
    # Evaluate with both methods
    scipy_results = np.array([scipy_bisplev(x, y, tck_scipy) for x, y in zip(x_eval, y_eval)])
    
    # For FastSpline, extract components and pre-allocate result
    tx, ty, c, kx, ky = tck_fast
    fast_results = np.zeros(len(x_eval))
    bisplev(x_eval, y_eval, tx, ty, c, kx, ky, fast_results)
    
    # Compare results
    diff = np.abs(scipy_results - fast_results)
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"\nAccuracy comparison:")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  RMS difference: {rms_diff:.6e}")
    print(f"  Mean scipy value: {np.mean(np.abs(scipy_results)):.6e}")
    print(f"  Relative error: {max_diff/np.mean(np.abs(scipy_results))*100:.3f}%")
    
    # Test passes if relative error is small
    relative_error = max_diff / np.mean(np.abs(scipy_results))
    if relative_error < 0.01:  # Less than 1% relative error
        print("✓ Missing data accuracy test PASSED")
        return True
    else:
        print("✗ Missing data accuracy test FAILED")
        return False

def test_cfunc_with_missing_data():
    """Test vectorized cfunc with missing data."""
    print("\nTesting vectorized cfunc with missing data...")
    
    # Create data with holes
    x_data, y_data, z_data, _, _, _, _ = create_data_with_holes()
    
    # Fit with scipy (for ground truth)
    tck = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    tx, ty, c, kx, ky = tck
    
    # Test points
    x_test = np.array([-1.0, 0.0, 1.0, -0.5, 0.5])
    y_test = np.array([-1.0, 0.0, 1.0, 0.5, -0.5])
    
    # Method 1: SciPy individual evaluation
    scipy_results = np.array([scipy_bisplev(x, y, tck) for x, y in zip(x_test, y_test)])
    
    # Method 2: FastSpline bisplev
    fast_results = np.zeros(len(x_test))
    bisplev(x_test, y_test, tx, ty, c, kx, ky, fast_results)
    
    # Compare methods
    diff_fast = np.abs(scipy_results - fast_results)
    
    print(f"FastSpline vs SciPy - Max diff: {np.max(diff_fast):.2e}")
    
    if np.max(diff_fast) < 1e-12:
        print("✓ Cfunc missing data test PASSED")
        return True
    else:
        print("✗ Cfunc missing data test FAILED")
        return False

def visualize_missing_data():
    """Create visualization of missing data handling."""
    print("\nCreating visualization...")
    
    # Create data with holes
    x_data, y_data, z_data, X_orig, Y_orig, Z_orig, mask = create_data_with_holes()
    
    # Fit splines
    tck_scipy = scipy_bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    tck_fast = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.01)
    
    # Evaluate on original grid
    scipy_interp = np.zeros_like(X_orig)
    fast_interp = np.zeros_like(X_orig)
    
    # Extract FastSpline components once
    tx, ty, c, kx, ky = tck_fast
    
    for i in range(X_orig.shape[0]):
        for j in range(X_orig.shape[1]):
            x, y = X_orig[i, j], Y_orig[i, j]
            scipy_interp[i, j] = scipy_bisplev(x, y, tck_scipy)
            fast_interp[i, j] = bisplev_scalar(x, y, tx, ty, c, kx, ky)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original data with holes
    Z_with_holes = Z_orig.copy()
    Z_with_holes[mask] = np.nan
    im1 = axes[0, 0].imshow(Z_with_holes, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
    axes[0, 0].set_title('Original Data with Holes')
    axes[0, 0].scatter(x_data, y_data, c='red', s=1, alpha=0.5)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # SciPy interpolation
    im2 = axes[0, 1].imshow(scipy_interp, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
    axes[0, 1].set_title('SciPy Interpolation')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # FastSpline interpolation
    im3 = axes[0, 2].imshow(fast_interp, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
    axes[0, 2].set_title('FastSpline Interpolation')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Difference
    diff = np.abs(scipy_interp - fast_interp)
    im4 = axes[1, 0].imshow(diff, extent=[-2, 2, -2, 2], origin='lower', cmap='hot')
    axes[1, 0].set_title(f'Absolute Difference (max: {np.max(diff):.2e})')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Relative error
    rel_error = diff / (np.abs(scipy_interp) + 1e-10)
    im5 = axes[1, 1].imshow(rel_error, extent=[-2, 2, -2, 2], origin='lower', cmap='hot')
    axes[1, 1].set_title(f'Relative Error (max: {np.max(rel_error):.2e})')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Data distribution
    axes[1, 2].scatter(x_data, y_data, c=z_data, s=1, cmap='viridis')
    axes[1, 2].set_title(f'Data Points ({len(x_data)} points)')
    axes[1, 2].set_xlim(-2, 2)
    axes[1, 2].set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('missing_data_validation.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to missing_data_validation.png")

if __name__ == "__main__":
    print("Missing Data Validation Test")
    print("=" * 60)
    
    accuracy_ok = test_missing_data_accuracy()
    if accuracy_ok:
        cfunc_ok = test_cfunc_with_missing_data()
        if cfunc_ok:
            visualize_missing_data()
            print("\n✓ All missing data tests PASSED")
        else:
            print("\n✗ Cfunc test failed")
    else:
        print("\n✗ Accuracy test failed")