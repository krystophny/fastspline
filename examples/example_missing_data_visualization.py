#!/usr/bin/env python3
"""
Example script demonstrating 2D spline interpolation with missing data holes.
Generates PNG visualization showing original data, missing data mask, and interpolated result.
Compares FastSpline (handles NaN in regular grid) with scipy.interpolate methods
(RectBivariateSpline with complete data and griddata with scattered incomplete data).
"""

import numpy as np
import matplotlib.pyplot as plt
from fastspline.spline2d import Spline2D
from scipy.interpolate import RectBivariateSpline, griddata


def create_test_data():
    """Create test data with a circular hole and scattered missing points."""
    # Create fine resolution grid
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create interesting test function: combination of sine waves
    Z = np.sin(3*np.pi*X) * np.cos(2*np.pi*Y) + 0.5*np.sin(5*np.pi*X*Y)
    
    # Create a circular hole in the center
    center_x, center_y = 0.5, 0.5
    hole_radius = 0.2
    
    # Calculate distance from center for each grid point
    distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Create hole by setting points within radius to NaN
    Z_with_hole = Z.copy()
    hole_mask = distances <= hole_radius
    Z_with_hole[hole_mask] = np.nan
    
    # Add some scattered missing points outside the hole
    Z_with_hole[2, 18] = np.nan   # Top right area
    Z_with_hole[18, 2] = np.nan   # Bottom left area
    Z_with_hole[15, 15] = np.nan  # Right side
    Z_with_hole[5, 16] = np.nan   # Additional scattered points
    Z_with_hole[16, 5] = np.nan
    
    return x, y, X, Y, Z, Z_with_hole, hole_mask


def create_scattered_data(X, Y, Z, hole_mask):
    """Create scattered x,y,z points by excluding the hole region."""
    # Find valid (non-hole) points
    valid_mask = ~hole_mask
    
    # Also exclude the scattered missing points we added
    # Get the original data without NaN handling
    valid_mask = valid_mask & np.isfinite(Z)
    
    # Extract valid points as scattered data
    x_scattered = X[valid_mask].ravel()
    y_scattered = Y[valid_mask].ravel() 
    z_scattered = Z[valid_mask].ravel()
    
    return x_scattered, y_scattered, z_scattered


def main():
    """Generate visualization of 2D spline interpolation with missing data."""
    print("Creating test data with missing data hole...")
    x, y, X, Y, Z_original, Z_with_hole, hole_mask = create_test_data()
    
    print("Creating scattered data points (excluding hole region)...")
    x_scattered, y_scattered, z_scattered = create_scattered_data(X, Y, Z_with_hole, hole_mask)
    
    print("Fitting FastSpline to data with missing values...")
    # Create FastSpline interpolation (handles NaN in regular grid)
    spline_fast = Spline2D(x, y, Z_with_hole.ravel(), kx=3, ky=3)
    
    print("Fitting scipy RectBivariateSpline to complete data for comparison...")
    # Create scipy spline with complete data (for comparison)
    spline_scipy_complete = RectBivariateSpline(x, y, Z_original, kx=3, ky=3)
    
    print("Evaluating splines on fine grid...")
    # Create fine evaluation grid
    x_eval = np.linspace(0, 1, 101)
    y_eval = np.linspace(0, 1, 101)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='ij')
    
    # Evaluate FastSpline on fine grid
    Z_interp_fast = np.zeros_like(X_eval)
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            result = spline_fast(x_eval[i], y_eval[j], grid=False)
            Z_interp_fast[i, j] = float(result)
    
    # Evaluate scipy spline (complete data) on fine grid
    Z_interp_scipy_complete = spline_scipy_complete(x_eval, y_eval)
    
    # Evaluate scipy griddata (incomplete/scattered data) on fine grid
    print("Using scipy griddata on scattered incomplete data...")
    points_scattered = np.column_stack((x_scattered, y_scattered))
    points_eval = np.column_stack((X_eval.ravel(), Y_eval.ravel()))
    
    # Try different interpolation methods
    Z_interp_scipy_scattered_cubic = griddata(points_scattered, z_scattered, points_eval, method='cubic')
    Z_interp_scipy_scattered_cubic = Z_interp_scipy_scattered_cubic.reshape(X_eval.shape)
    
    Z_interp_scipy_scattered_linear = griddata(points_scattered, z_scattered, points_eval, method='linear')
    Z_interp_scipy_scattered_linear = Z_interp_scipy_scattered_linear.reshape(X_eval.shape)
    
    print("Creating visualization...")
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle('2D Spline Interpolation: FastSpline vs SciPy with Missing Data', fontsize=16)
    
    # Row 1: Original data and missing data
    # Plot 1: Original complete data
    im1 = axes[0, 0].imshow(Z_original.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='viridis', aspect='equal')
    axes[0, 0].set_title('Original Complete Data')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Data with missing values (hole)
    Z_display = Z_with_hole.copy()
    im2 = axes[0, 1].imshow(Z_display.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='viridis', aspect='equal')
    axes[0, 1].set_title('Data with Missing Values')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Scattered data points
    axes[0, 2].scatter(x_scattered, y_scattered, c=z_scattered, cmap='viridis', s=5)
    axes[0, 2].set_title(f'Scattered Data Points\n({len(x_scattered)} points)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    
    # Row 2: Interpolation methods using complete data
    # Plot 4: SciPy spline (complete data)
    im4 = axes[1, 0].imshow(Z_interp_scipy_complete.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='viridis', aspect='equal')
    axes[1, 0].set_title('SciPy RectBivariateSpline\n(Complete Data)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: FastSpline interpolation result
    im5 = axes[1, 1].imshow(Z_interp_fast.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='viridis', aspect='equal')
    axes[1, 1].set_title('FastSpline\n(Regular Grid with NaN)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1])
    circle1 = plt.Circle((0.5, 0.5), 0.2, fill=False, color='red', linewidth=2)
    axes[1, 1].add_patch(circle1)
    
    # Plot 6: SciPy griddata cubic
    im6 = axes[1, 2].imshow(Z_interp_scipy_scattered_cubic.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='viridis', aspect='equal')
    axes[1, 2].set_title('SciPy griddata Cubic\n(Scattered Data)')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2])
    circle2 = plt.Circle((0.5, 0.5), 0.2, fill=False, color='red', linewidth=2)
    axes[1, 2].add_patch(circle2)
    
    # Row 3: More comparisons and differences
    # Plot 7: SciPy griddata linear
    im7 = axes[2, 0].imshow(Z_interp_scipy_scattered_linear.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='viridis', aspect='equal')
    axes[2, 0].set_title('SciPy griddata Linear\n(Scattered Data)')
    axes[2, 0].set_xlabel('x')
    axes[2, 0].set_ylabel('y')
    plt.colorbar(im7, ax=axes[2, 0])
    circle3 = plt.Circle((0.5, 0.5), 0.2, fill=False, color='red', linewidth=2)
    axes[2, 0].add_patch(circle3)
    
    # Plot 8: Difference FastSpline vs SciPy griddata cubic
    diff_fast_cubic = Z_interp_fast - Z_interp_scipy_scattered_cubic
    # Handle NaN values in the difference
    diff_fast_cubic = np.where(np.isnan(diff_fast_cubic), 0, diff_fast_cubic)
    im8 = axes[2, 1].imshow(diff_fast_cubic.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', aspect='equal')
    axes[2, 1].set_title('Difference\n(FastSpline - griddata cubic)')
    axes[2, 1].set_xlabel('x')
    axes[2, 1].set_ylabel('y')
    plt.colorbar(im8, ax=axes[2, 1])
    circle4 = plt.Circle((0.5, 0.5), 0.2, fill=False, color='black', linewidth=2)
    axes[2, 1].add_patch(circle4)
    
    # Plot 9: Difference complete vs incomplete methods
    diff_complete_incomplete = Z_interp_scipy_complete - Z_interp_scipy_scattered_cubic
    diff_complete_incomplete = np.where(np.isnan(diff_complete_incomplete), 0, diff_complete_incomplete)
    im9 = axes[2, 2].imshow(diff_complete_incomplete.T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='RdBu_r', aspect='equal')
    axes[2, 2].set_title('Difference\n(Complete - Incomplete)')
    axes[2, 2].set_xlabel('x')
    axes[2, 2].set_ylabel('y')
    plt.colorbar(im9, ax=axes[2, 2])
    circle5 = plt.Circle((0.5, 0.5), 0.2, fill=False, color='black', linewidth=2)
    axes[2, 2].add_patch(circle5)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'spline2d_missing_data_demo.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Display some statistics
    print(f"\nStatistics:")
    print(f"Original grid size: {Z_original.shape}")
    missing_count = np.sum(np.isnan(Z_with_hole))
    print(f"Missing data points: {missing_count}")
    print(f"Missing data percentage: {100 * missing_count / Z_with_hole.size:.1f}%")
    print(f"Scattered data points: {len(x_scattered)}")
    print(f"Interpolation grid size: {Z_interp_fast.shape}")
    
    # Test interpolation quality at a few points
    print(f"\nInterpolation quality check:")
    test_points = [(0.1, 0.1), (0.9, 0.9), (0.3, 0.7)]
    print("Point        | Original | FastSpline | SciPy Complete | Griddata Cubic | Griddata Linear")
    print("-------------|----------|------------|----------------|----------------|----------------")
    for tx, ty in test_points:
        original = np.sin(3*np.pi*tx) * np.cos(2*np.pi*ty) + 0.5*np.sin(5*np.pi*tx*ty)
        interp_fast = float(spline_fast(tx, ty, grid=False))
        interp_scipy_complete = float(spline_scipy_complete(tx, ty))
        
        # Interpolate scattered data at test point
        interp_griddata_cubic = float(griddata(points_scattered, z_scattered, [(tx, ty)], method='cubic')[0])
        interp_griddata_linear = float(griddata(points_scattered, z_scattered, [(tx, ty)], method='linear')[0])
        
        print(f"({tx:3.1f}, {ty:3.1f}) | {original:8.4f} | {interp_fast:10.4f} | {interp_scipy_complete:14.4f} | {interp_griddata_cubic:14.4f} | {interp_griddata_linear:15.4f}")
    
    # Calculate overall differences
    print(f"\nOverall comparison (ignoring NaN regions):")
    
    # FastSpline vs griddata cubic
    valid_mask = np.isfinite(diff_fast_cubic)
    if np.any(valid_mask):
        print(f"FastSpline vs griddata cubic:")
        print(f"  Max difference: {np.max(np.abs(diff_fast_cubic[valid_mask])):.4f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(diff_fast_cubic[valid_mask])):.4f}")
        print(f"  RMS difference: {np.sqrt(np.mean(diff_fast_cubic[valid_mask]**2)):.4f}")
    
    # Complete vs incomplete
    valid_mask2 = np.isfinite(diff_complete_incomplete)
    if np.any(valid_mask2):
        print(f"Complete vs incomplete data:")
        print(f"  Max difference: {np.max(np.abs(diff_complete_incomplete[valid_mask2])):.4f}")
        print(f"  Mean absolute difference: {np.mean(np.abs(diff_complete_incomplete[valid_mask2])):.4f}")
        print(f"  RMS difference: {np.sqrt(np.mean(diff_complete_incomplete[valid_mask2]**2)):.4f}")


if __name__ == "__main__":
    main()