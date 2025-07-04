#!/usr/bin/env python3
"""Test QR-based bisplrep implementation."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline import bisplrep, bisplev_scalar as bisplev


def test_qr_implementation():
    """Test QR-based bisplrep implementation."""
    print("Testing QR-based bisplrep implementation")
    print("=" * 70)
    
    # Test data - exact polynomial
    x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], dtype=np.float64)
    y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1], dtype=np.float64)
    z = x**2 + y**2
    w = np.ones_like(x)
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x, y, z, w, kx=2, ky=2, s=0.0)
    print(f"SciPy coefficients: {tck_scipy[2][:9]}")
    
    # Fit with our QR version
    tck_qr = bisplrep(x, y, z, w, kx=2, ky=2, s=0.0)
    print(f"\nFastspline (QR) coefficients: {tck_qr[2][:9]}")
    
    # Compare coefficient differences
    print(f"\nCoefficient differences from SciPy:")
    print(f"{'Index':<6} {'Difference':<12} {'Relative Error':<15}")
    print("-" * 35)
    
    for i in range(9):
        diff = abs(tck_qr[2][i] - tck_scipy[2][i])
        rel_err = diff / (abs(tck_scipy[2][i]) + 1e-10)
        print(f"{i:<6} {diff:<12.2e} {rel_err:<15.2e}")
    
    # Test residuals
    print(f"\nResidual sum of squares:")
    
    # Compute residuals for each method
    residuals_scipy = []
    residuals_qr = []
    
    for i in range(len(x)):
        z_scipy = scipy_bisplev(x[i], y[i], tck_scipy)
        z_qr = bisplev(x[i], y[i], tck_qr[0], tck_qr[1], tck_qr[2], 2, 2)
        
        residuals_scipy.append(z[i] - z_scipy)
        residuals_qr.append(z[i] - z_qr)
    
    rss_scipy = np.sum(np.array(residuals_scipy)**2)
    rss_qr = np.sum(np.array(residuals_qr)**2)
    
    print(f"  SciPy:           {rss_scipy:.2e}")
    print(f"  FastSpline (QR): {rss_qr:.2e}")
    
    # Test on ill-conditioned problem
    print("\n\nTesting on ill-conditioned problem:")
    print("=" * 70)
    
    # Create clustered data points
    n_cluster = 10
    x_cluster = []
    y_cluster = []
    z_cluster = []
    
    # Add clusters at corners
    for cx, cy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        for _ in range(n_cluster):
            x_cluster.append(cx + np.random.normal(0, 0.01))
            y_cluster.append(cy + np.random.normal(0, 0.01))
    
    x_cluster = np.array(x_cluster)
    y_cluster = np.array(y_cluster)
    z_cluster = np.sin(np.pi * x_cluster) * np.cos(np.pi * y_cluster)
    
    # Fit with both methods
    try:
        tck_scipy_ill = scipy_bisplrep(x_cluster, y_cluster, z_cluster, kx=3, ky=3, s=0.0)
        print("SciPy: SUCCESS")
    except Exception as e:
        print(f"SciPy: FAILED - {e}")
        tck_scipy_ill = None
    
    try:
        tck_qr_ill = bisplrep(x_cluster, y_cluster, z_cluster, kx=3, ky=3, s=0.0)
        print("FastSpline (QR): SUCCESS")
    except Exception as e:
        print(f"FastSpline (QR): FAILED - {e}")
        tck_qr_ill = None
    
    # If both succeeded, compare accuracy
    if tck_scipy_ill is not None and tck_qr_ill is not None:
        # Test on a grid
        x_test = np.linspace(0, 1, 20)
        y_test = np.linspace(0, 1, 20)
        
        max_diff = 0
        for xi in x_test:
            for yi in y_test:
                z_true = np.sin(np.pi * xi) * np.cos(np.pi * yi)
                z_scipy = scipy_bisplev(xi, yi, tck_scipy_ill)
                z_qr = bisplev(xi, yi, tck_qr_ill[0], tck_qr_ill[1], tck_qr_ill[2], 3, 3)
                
                diff = abs(z_qr - z_true)
                max_diff = max(max_diff, diff)
        
        print(f"\nMax error on test grid: {max_diff:.2e}")


if __name__ == "__main__":
    test_qr_implementation()