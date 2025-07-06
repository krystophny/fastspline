#!/usr/bin/env python3
"""
EXACT testing of bisplrep implementation against SciPy
This test requires EXACT matches, not approximations
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dierckx_cfunc import bisplrep_cfunc, bisplev_cfunc

def test_exact_knot_placement():
    """Test that knot placement EXACTLY matches SciPy"""
    print("=" * 80)
    print("EXACT KNOT PLACEMENT TEST")
    print("=" * 80)
    
    # Test simple 2x2 grid
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([1., 2., 2., 3.])
    
    print(f"Test data: x={x}, y={y}, z={z}")
    
    # SciPy result
    tck_scipy = bisplrep(x, y, z, kx=1, ky=1, s=0)
    tx_scipy, ty_scipy, c_scipy, kx_scipy, ky_scipy = tck_scipy
    
    print(f"\nSciPy result:")
    print(f"  tx: {tx_scipy}")
    print(f"  ty: {ty_scipy}")
    print(f"  c:  {c_scipy}")
    
    # Our implementation
    tx_ours, ty_ours, c_ours, kx_ours, ky_ours = bisplrep_cfunc(x, y, z, kx=1, ky=1, s=0.0)
    
    print(f"\nOur result:")
    print(f"  tx: {tx_ours}")
    print(f"  ty: {ty_ours}")
    print(f"  c:  {c_ours}")
    
    # Check EXACT match
    knot_match_x = np.allclose(tx_scipy, tx_ours, atol=1e-14)
    knot_match_y = np.allclose(ty_scipy, ty_ours, atol=1e-14)
    coeff_match = np.allclose(c_scipy, c_ours, atol=1e-12)
    
    print(f"\nExact match check:")
    print(f"  Knots X: {'✓' if knot_match_x else '✗'}")
    print(f"  Knots Y: {'✓' if knot_match_y else '✗'}")
    print(f"  Coefficients: {'✓' if coeff_match else '✗'}")
    
    if not knot_match_x:
        print(f"  X knot differences: {tx_scipy - tx_ours}")
    if not knot_match_y:
        print(f"  Y knot differences: {ty_scipy - ty_ours}")
    if not coeff_match:
        print(f"  Coefficient differences: {c_scipy - c_ours}")
    
    return knot_match_x and knot_match_y and coeff_match

def test_exact_evaluation():
    """Test that evaluation EXACTLY matches SciPy"""
    print("\n" + "=" * 80)
    print("EXACT EVALUATION TEST")
    print("=" * 80)
    
    # Test data
    x = np.array([0., 1., 0., 1.])
    y = np.array([0., 0., 1., 1.])
    z = np.array([1., 2., 2., 3.])
    
    # SciPy
    tck_scipy = bisplrep(x, y, z, kx=1, ky=1, s=0)
    
    # Our implementation
    tx, ty, c, kx, ky = bisplrep_cfunc(x, y, z, kx=1, ky=1, s=0.0)
    
    # Test evaluation at original data points
    print("Testing at original data points:")
    for i in range(len(x)):
        z_scipy = bisplev(x[i], y[i], tck_scipy)
        z_ours_array = bisplev_cfunc(np.array([x[i]]), np.array([y[i]]), tx, ty, c, kx, ky)
        z_ours = z_ours_array[0, 0]
        
        error = abs(z_scipy - z_ours)
        exact_orig = abs(z_scipy - z[i]) < 1e-14
        exact_ours = abs(z_ours - z[i]) < 1e-14
        
        print(f"  Point ({x[i]}, {y[i]}): true={z[i]:.6f}, scipy={z_scipy:.6f}, ours={z_ours:.6f}")
        print(f"    Error vs SciPy: {error:.2e}")
        print(f"    SciPy exact: {'✓' if exact_orig else '✗'}")
        print(f"    Ours exact: {'✓' if exact_ours else '✗'}")
        
        if not exact_ours:
            print(f"    *** INTERPOLATION FAILED ***")
            return False
    
    # Test evaluation at center point
    x_center, y_center = 0.5, 0.5
    z_scipy_center = bisplev(x_center, y_center, tck_scipy)
    z_ours_center_array = bisplev_cfunc(np.array([x_center]), np.array([y_center]), tx, ty, c, kx, ky)
    z_ours_center = z_ours_center_array[0, 0]
    
    print(f"\nCenter point ({x_center}, {y_center}):")
    print(f"  SciPy: {z_scipy_center:.6f}")
    print(f"  Ours:  {z_ours_center:.6f}")
    print(f"  Error: {abs(z_scipy_center - z_ours_center):.2e}")
    
    # For bilinear interpolation, the center should be exactly 2.0
    expected_center = 2.0
    scipy_center_exact = abs(z_scipy_center - expected_center) < 1e-14
    ours_center_exact = abs(z_ours_center - expected_center) < 1e-14
    
    print(f"  Expected: {expected_center:.6f}")
    print(f"  SciPy exact: {'✓' if scipy_center_exact else '✗'}")
    print(f"  Ours exact: {'✓' if ours_center_exact else '✗'}")
    
    return ours_center_exact

def test_different_grid_sizes():
    """Test various grid sizes with EXACT matching"""
    print("\n" + "=" * 80)
    print("DIFFERENT GRID SIZES TEST")
    print("=" * 80)
    
    test_cases = [
        (2, 2, 1, 1),  # 2x2 grid, linear
        (3, 3, 1, 1),  # 3x3 grid, linear
        (3, 3, 2, 2),  # 3x3 grid, quadratic
        (4, 4, 1, 1),  # 4x4 grid, linear
    ]
    
    all_passed = True
    
    for nx, ny, kx, ky in test_cases:
        print(f"\nTesting {nx}x{ny} grid, degree ({kx}, {ky}):")
        
        # Create regular grid
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        Z = X + Y  # Simple test function
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        try:
            # SciPy
            tck_scipy = bisplrep(x_flat, y_flat, z_flat, kx=kx, ky=ky, s=0)
            tx_scipy, ty_scipy, c_scipy, _, _ = tck_scipy
            
            # Our implementation
            tx_ours, ty_ours, c_ours, _, _ = bisplrep_cfunc(x_flat, y_flat, z_flat, kx=kx, ky=ky, s=0.0)
            
            # Check knot count
            knot_count_match = len(tx_scipy) == len(tx_ours) and len(ty_scipy) == len(ty_ours)
            coeff_count_match = len(c_scipy) == len(c_ours)
            
            print(f"  SciPy: {len(tx_scipy)} x {len(ty_scipy)} knots, {len(c_scipy)} coeffs")
            print(f"  Ours:  {len(tx_ours)} x {len(ty_ours)} knots, {len(c_ours)} coeffs")
            print(f"  Knot count match: {'✓' if knot_count_match else '✗'}")
            print(f"  Coeff count match: {'✓' if coeff_count_match else '✗'}")
            
            if not knot_count_match or not coeff_count_match:
                all_passed = False
                print(f"  *** STRUCTURE MISMATCH ***")
                continue
            
            # Check knot values
            knot_x_match = np.allclose(tx_scipy, tx_ours, atol=1e-14)
            knot_y_match = np.allclose(ty_scipy, ty_ours, atol=1e-14)
            
            print(f"  Knot X match: {'✓' if knot_x_match else '✗'}")
            print(f"  Knot Y match: {'✓' if knot_y_match else '✗'}")
            
            if not knot_x_match:
                print(f"    X differences: max={np.max(np.abs(tx_scipy - tx_ours)):.2e}")
            if not knot_y_match:
                print(f"    Y differences: max={np.max(np.abs(ty_scipy - ty_ours)):.2e}")
            
            if not knot_x_match or not knot_y_match:
                all_passed = False
                print(f"  *** KNOT PLACEMENT MISMATCH ***")
                
        except Exception as e:
            print(f"  *** ERROR: {e} ***")
            all_passed = False
    
    return all_passed

def main():
    """Run all exact tests"""
    print("EXACT SCIPY MATCHING TEST")
    print("This test requires EXACT matches, not approximations")
    print("Any deviation indicates incorrect implementation")
    
    # Run tests
    test1_passed = test_exact_knot_placement()
    test2_passed = test_exact_evaluation()
    test3_passed = test_different_grid_sizes()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"Exact knot placement: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Exact evaluation: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Different grid sizes: {'✓ PASS' if test3_passed else '✗ FAIL'}")
    
    overall_pass = test1_passed and test2_passed and test3_passed
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if overall_pass else '✗ TESTS FAILED'}")
    
    if not overall_pass:
        print("\n*** IMPLEMENTATION NEEDS FIXING ***")
        print("The current implementation does not match SciPy exactly.")
        print("This indicates fundamental issues with knot placement or coefficient calculation.")
    
    return 0 if overall_pass else 1

if __name__ == "__main__":
    sys.exit(main())