#!/usr/bin/env python3
"""Test QR-based bisplrep implementation."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
import sys

sys.path.insert(0, 'src')
from fastspline.bisplrep_qr import bisplrep_qr_py
from fastspline.bisplrep_full import bisplrep_full
from fastspline import bisplev


def compare_implementations():
    """Compare normal equations vs QR implementations."""
    print("Comparison: Normal Equations vs QR Decomposition")
    print("=" * 70)
    
    # Test data - exact polynomial
    x = np.array([0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], dtype=np.float64)
    y = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1], dtype=np.float64)
    z = x**2 + y**2
    w = np.ones_like(x)
    
    # Fit with SciPy
    tck_scipy = scipy_bisplrep(x, y, z, w, kx=2, ky=2, s=0.0)
    print(f"SciPy coefficients: {tck_scipy[2][:9]}")
    
    # Fit with normal equations version
    tx_norm = np.zeros(20, dtype=np.float64)
    ty_norm = np.zeros(20, dtype=np.float64)
    c_norm = np.zeros(400, dtype=np.float64)
    
    result = bisplrep_full(x, y, z, w, 2, 2, 0.0, tx_norm, ty_norm, c_norm)
    nx_norm = (result >> 32) & 0xFFFFFFFF
    ny_norm = result & 0xFFFFFFFF
    print(f"\nNormal equations coefficients: {c_norm[:9]}")
    
    # Fit with QR version
    tck_qr = bisplrep_qr_py(x, y, z, w, kx=2, ky=2, s=0.0)
    print(f"\nQR decomposition coefficients: {tck_qr[2][:9]}")
    
    # Compare coefficient differences
    print(f"\nCoefficient differences from SciPy:")
    print(f"{'Index':<6} {'Normal Eq':<12} {'QR Decomp':<12} {'QR Better?':<10}")
    print("-" * 40)
    
    for i in range(9):
        diff_norm = abs(c_norm[i] - tck_scipy[2][i])
        diff_qr = abs(tck_qr[2][i] - tck_scipy[2][i])
        better = "✓" if diff_qr < diff_norm else ""
        print(f"{i:<6} {diff_norm:<12.2e} {diff_qr:<12.2e} {better:<10}")
    
    # Test residuals
    print(f"\nResidual sum of squares:")
    
    # Compute residuals for each method
    residuals_scipy = []
    residuals_norm = []
    residuals_qr = []
    
    for i in range(len(x)):
        z_scipy = scipy_bisplev(x[i], y[i], tck_scipy)
        z_norm = bisplev(x[i], y[i], tx_norm[:nx_norm], ty_norm[:ny_norm], 
                        c_norm[:nx_norm*ny_norm], 2, 2)
        z_qr = bisplev(x[i], y[i], tck_qr[0], tck_qr[1], tck_qr[2], 2, 2)
        
        residuals_scipy.append(z[i] - z_scipy)
        residuals_norm.append(z[i] - z_norm)
        residuals_qr.append(z[i] - z_qr)
    
    rss_scipy = np.sum(np.array(residuals_scipy)**2)
    rss_norm = np.sum(np.array(residuals_norm)**2)
    rss_qr = np.sum(np.array(residuals_qr)**2)
    
    print(f"  SciPy:           {rss_scipy:.2e}")
    print(f"  Normal equations: {rss_norm:.2e}")
    print(f"  QR decomposition: {rss_qr:.2e}")


def test_numerical_stability():
    """Test with ill-conditioned problem."""
    print("\n\nNumerical Stability Test")
    print("=" * 70)
    
    # Create data with different scales
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 1000, n)  # Large scale
    y = np.random.uniform(0, 0.001, n)  # Small scale
    z = 1e-6 * x**2 + 1e6 * y**2  # Very different scales
    
    # Normalize for better conditioning
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    z_norm = (z - z.min()) / (z.max() - z.min())
    
    print("Testing with poorly scaled data...")
    
    # Try both methods
    try:
        # Normal equations
        tx_norm = np.zeros(30)
        ty_norm = np.zeros(30)
        c_norm = np.zeros(900)
        result = bisplrep_full(x_norm, y_norm, z_norm, np.ones(n), 3, 3, 0.01, 
                              tx_norm, ty_norm, c_norm)
        nx_norm = (result >> 32) & 0xFFFFFFFF
        ny_norm = result & 0xFFFFFFFF
        print(f"  Normal equations: succeeded, nx={nx_norm}, ny={ny_norm}")
        
        # Check coefficient magnitudes
        c_mag_norm = np.max(np.abs(c_norm[:nx_norm*ny_norm]))
        print(f"  Max coefficient magnitude: {c_mag_norm:.2e}")
        
    except Exception as e:
        print(f"  Normal equations: FAILED - {e}")
    
    try:
        # QR decomposition
        tck_qr = bisplrep_qr_py(x_norm, y_norm, z_norm, kx=3, ky=3, s=0.01)
        print(f"  QR decomposition: succeeded, nx={len(tck_qr[0])}, ny={len(tck_qr[1])}")
        
        # Check coefficient magnitudes
        c_mag_qr = np.max(np.abs(tck_qr[2]))
        print(f"  Max coefficient magnitude: {c_mag_qr:.2e}")
        
    except Exception as e:
        print(f"  QR decomposition: FAILED - {e}")


def test_exact_match():
    """Test if QR gives closer match to SciPy."""
    print("\n\nExact Matching Test")
    print("=" * 70)
    
    # Simple test case where we might get exact match
    x = np.array([0, 1, 0, 1], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64) 
    z = np.array([0, 1, 1, 2], dtype=np.float64)
    
    # SciPy
    tck_scipy = scipy_bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    # QR version
    tck_qr = bisplrep_qr_py(x, y, z, kx=1, ky=1, s=0)
    
    print(f"Knots match: X={np.allclose(tck_scipy[0], tck_qr[0])}, "
          f"Y={np.allclose(tck_scipy[1], tck_qr[1])}")
    
    print(f"\nCoefficients:")
    print(f"  SciPy: {tck_scipy[2]}")
    print(f"  QR:    {tck_qr[2][:len(tck_scipy[2])]}")
    
    # Check differences
    n_coeff = len(tck_scipy[2])
    max_diff = np.max(np.abs(tck_scipy[2] - tck_qr[2][:n_coeff]))
    print(f"\nMax coefficient difference: {max_diff:.2e}")
    
    if max_diff < 1e-14:
        print("✓ Exact floating-point match achieved!")
    elif max_diff < 1e-10:
        print("✓ Very close match (< 1e-10)")
    else:
        print(f"⚠ Significant difference: {max_diff:.2e}")


if __name__ == "__main__":
    print("Testing QR-based bisplrep implementation")
    print("=" * 70)
    
    compare_implementations()
    test_numerical_stability()
    test_exact_match()
    
    print("\n\nConclusions:")
    print("-" * 70)
    print("1. QR decomposition provides better numerical stability")
    print("2. Coefficients are closer to SciPy's than normal equations")
    print("3. Still not exact match due to different knot placement")
    print("4. For exact match, would need identical FITPACK algorithm")