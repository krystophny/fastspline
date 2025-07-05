"""Example usage of FastSpline bisplrep/bisplev."""

import numpy as np
import matplotlib.pyplot as plt
from fastspline import bisplrep, bisplev


def example_surface_fitting():
    """Example of fitting a surface with bisplrep."""
    # Generate sample data - peaks function
    x = np.linspace(-3, 3, 40)
    y = np.linspace(-3, 3, 40)
    xx, yy = np.meshgrid(x, y)
    
    # Peaks function (similar to MATLAB peaks)
    z = 3*(1-xx)**2 * np.exp(-(xx**2) - (yy+1)**2) \
        - 10*(xx/5 - xx**3 - yy**5) * np.exp(-xx**2 - yy**2) \
        - 1/3 * np.exp(-(xx+1)**2 - yy**2)
    
    # Add some noise
    np.random.seed(42)
    z_noisy = z + 0.1 * np.random.randn(*z.shape)
    
    # Flatten for bisplrep
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z_noisy.ravel()
    
    # Fit spline surface with smoothing
    print("Fitting B-spline surface...")
    tck = bisplrep(x_flat, y_flat, z_flat, s=10.0)
    tx, ty, c, kx, ky = tck
    print(f"Knots: {len(tx)}x{len(ty)}")
    print(f"Degrees: kx={kx}, ky={ky}")
    
    # Evaluate on finer grid
    x_fine = np.linspace(-3, 3, 100)
    y_fine = np.linspace(-3, 3, 100)
    z_smooth = bisplev(x_fine, y_fine, tck)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    im1 = ax1.contourf(xx, yy, z, levels=20, cmap='viridis')
    ax1.set_title('Original Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Noisy data
    im2 = ax2.contourf(xx, yy, z_noisy, levels=20, cmap='viridis')
    ax2.set_title('Noisy Data')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    # Smoothed spline
    xx_fine, yy_fine = np.meshgrid(x_fine, y_fine)
    im3 = ax3.contourf(xx_fine, yy_fine, z_smooth, levels=20, cmap='viridis')
    ax3.set_title('B-spline Fit')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()


def example_interpolation():
    """Example of exact interpolation."""
    # Sparse data points
    np.random.seed(42)
    n_points = 50
    x = np.random.uniform(-2, 2, n_points)
    y = np.random.uniform(-2, 2, n_points)
    z = np.sin(x) * np.cos(y) + 0.1 * x * y
    
    # Fit interpolating spline (s=0)
    print("\nFitting interpolating spline...")
    tck = bisplrep(x, y, z, s=0)
    
    # Evaluate on grid
    x_grid = np.linspace(-2, 2, 100)
    y_grid = np.linspace(-2, 2, 100)
    z_grid = bisplev(x_grid, y_grid, tck)
    
    # Check interpolation at data points
    z_interp = np.zeros(n_points)
    for i in range(n_points):
        z_interp[i] = float(bisplev(np.array([x[i]]), np.array([y[i]]), tck))
    
    max_error = np.abs(z_interp - z).max()
    print(f"Maximum interpolation error: {max_error:.2e}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Surface
    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    im = ax.contourf(xx_grid, yy_grid, z_grid, levels=20, cmap='viridis', alpha=0.8)
    
    # Data points
    scatter = ax.scatter(x, y, c=z, s=50, cmap='viridis', edgecolors='black', linewidth=1)
    
    ax.set_title('Interpolating B-spline Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Z')
    
    plt.tight_layout()
    plt.show()


def example_weighted_fitting():
    """Example with weighted least squares."""
    # Generate data with outliers
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    xx, yy = np.meshgrid(x, y)
    z = xx**2 + yy**2  # Paraboloid
    
    # Add outliers
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    z_flat = z.ravel()
    
    # Corrupt some points
    outlier_idx = np.random.choice(len(z_flat), 20, replace=False)
    z_flat[outlier_idx] += np.random.uniform(-2, 2, 20)
    
    # Create weights (down-weight outliers)
    w = np.ones_like(z_flat)
    w[outlier_idx] = 0.1
    
    # Fit with and without weights
    print("\nComparing weighted vs unweighted fitting...")
    tck_weighted = bisplrep(x_flat, y_flat, z_flat, w=w, s=0.1)
    tck_unweighted = bisplrep(x_flat, y_flat, z_flat, s=0.1)
    
    # Evaluate
    x_eval = np.linspace(-1, 1, 50)
    y_eval = np.linspace(-1, 1, 50)
    z_weighted = bisplev(x_eval, y_eval, tck_weighted)
    z_unweighted = bisplev(x_eval, y_eval, tck_unweighted)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    xx_eval, yy_eval = np.meshgrid(x_eval, y_eval)
    
    # Unweighted
    im1 = ax1.contourf(xx_eval, yy_eval, z_unweighted, levels=20, cmap='viridis')
    ax1.scatter(x_flat[outlier_idx], y_flat[outlier_idx], c='red', s=50, marker='x')
    ax1.set_title('Unweighted Fit')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1)
    
    # Weighted
    im2 = ax2.contourf(xx_eval, yy_eval, z_weighted, levels=20, cmap='viridis')
    ax2.scatter(x_flat[outlier_idx], y_flat[outlier_idx], c='red', s=50, marker='x')
    ax2.set_title('Weighted Fit (outliers down-weighted)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_surface_fitting()
    example_interpolation()
    example_weighted_fitting()