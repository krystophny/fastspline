#!/usr/bin/env python3
"""
Verify floating-point accuracy between scipy bisplev and direct Fortran calls.
Tests for exact bit-for-bit equality and analyzes any differences.
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import _dfitpack
import struct

def float_to_hex(f):
    """Convert float to hex representation for bit-level comparison."""
    return hex(struct.unpack('>Q', struct.pack('>d', f))[0])

def analyze_differences(arr1, arr2, name1="Array 1", name2="Array 2"):
    """Detailed analysis of differences between two arrays."""
    print(f"\nComparing {name1} vs {name2}:")
    print("-" * 50)
    
    # Check if arrays are identical
    if np.array_equal(arr1, arr2):
        print("✓ Arrays are EXACTLY identical (bit-for-bit)")
        return True
    
    # Check shape
    if arr1.shape != arr2.shape:
        print(f"✗ Shape mismatch: {arr1.shape} vs {arr2.shape}")
        return False
    
    # Flatten for analysis
    a1_flat = arr1.flatten()
    a2_flat = arr2.flatten()
    
    # Count exact matches
    exact_matches = np.sum(a1_flat == a2_flat)
    total_elements = len(a1_flat)
    print(f"Exact matches: {exact_matches}/{total_elements} ({exact_matches/total_elements*100:.2f}%)")
    
    # Find differences
    diff_mask = a1_flat != a2_flat
    if np.any(diff_mask):
        diff_indices = np.where(diff_mask)[0]
        num_diffs = len(diff_indices)
        print(f"Number of differences: {num_diffs}")
        
        # Analyze differences
        diffs = np.abs(a1_flat[diff_mask] - a2_flat[diff_mask])
        rel_diffs = diffs / np.abs(a1_flat[diff_mask])
        
        print(f"\nDifference statistics:")
        print(f"  Max absolute difference: {np.max(diffs):.2e}")
        print(f"  Min absolute difference: {np.min(diffs):.2e}")
        print(f"  Mean absolute difference: {np.mean(diffs):.2e}")
        print(f"  Max relative difference: {np.max(rel_diffs):.2e}")
        print(f"  Mean relative difference: {np.mean(rel_diffs):.2e}")
        
        # Show first few differences at bit level
        print(f"\nFirst 5 differences (showing actual values and hex):")
        for i in range(min(5, num_diffs)):
            idx = diff_indices[i]
            v1, v2 = a1_flat[idx], a2_flat[idx]
            print(f"  Index {idx}:")
            print(f"    {name1}: {v1:.17e} ({float_to_hex(v1)})")
            print(f"    {name2}: {v2:.17e} ({float_to_hex(v2)})")
            print(f"    Diff: {v1-v2:.2e} (relative: {abs(v1-v2)/abs(v1):.2e})")
    
    # Check if differences are within machine epsilon
    if np.allclose(arr1, arr2, rtol=0, atol=0):
        print("\n✓ Arrays are identical within zero tolerance")
    elif np.allclose(arr1, arr2, rtol=np.finfo(float).eps, atol=0):
        print("\n✓ Arrays are identical within machine epsilon (relative)")
    elif np.allclose(arr1, arr2, rtol=0, atol=np.finfo(float).eps):
        print("\n✓ Arrays are identical within machine epsilon (absolute)")
    elif np.allclose(arr1, arr2):
        print("\n✓ Arrays are close within numpy default tolerance")
    else:
        print("\n✗ Arrays differ beyond numpy default tolerance")
    
    return False

def test_accuracy_comparison():
    """Test floating-point accuracy across different methods."""
    
    # Generate test data
    np.random.seed(42)
    n_points = 500
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.sin(np.sqrt(x**2 + y**2)) + 0.1 * np.random.randn(n_points)
    
    # Fit spline
    tck = bisplrep(x, y, z, s=n_points)
    tx, ty, c, kx, ky = tck
    
    # Test on different grid types
    test_cases = [
        ("Regular grid 10x10", 
         np.linspace(x.min(), x.max(), 10),
         np.linspace(y.min(), y.max(), 10)),
        ("Regular grid 50x50",
         np.linspace(x.min(), x.max(), 50),
         np.linspace(y.min(), y.max(), 50)),
        ("Single point",
         np.array([0.0]),
         np.array([0.0])),
        ("Random points",
         np.random.uniform(-4, 4, 20),
         np.random.uniform(-4, 4, 20)),
        ("Edge case - boundaries",
         np.array([x.min(), 0.0, x.max()]),
         np.array([y.min(), 0.0, y.max()])),
    ]
    
    print("="*60)
    print("FLOATING-POINT ACCURACY VERIFICATION")
    print("="*60)
    
    all_exact = True
    
    for test_name, xi, yi in test_cases:
        print(f"\n{'='*60}")
        print(f"Test case: {test_name}")
        print(f"Evaluation points: {len(xi)} x {len(yi)}")
        
        # Ensure proper array format
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        yi = np.ascontiguousarray(yi, dtype=np.float64)
        
        # Method 1: scipy.bisplev
        z_scipy = bisplev(xi, yi, tck)
        
        # Method 2: Direct _dfitpack.bispev
        z_fortran, ier = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)
        
        # Method 3: Direct with pre-conversion
        xi_copy = xi.copy()
        yi_copy = yi.copy()
        z_fortran2, ier2 = _dfitpack.bispev(tx, ty, c, kx, ky, xi_copy, yi_copy)
        
        # Compare results
        # Handle shape differences for single points
        if z_scipy.shape != z_fortran.shape:
            print(f"\nShape difference: scipy {z_scipy.shape} vs fortran {z_fortran.shape}")
            if z_scipy.ndim == 0 and z_fortran.shape == (1, 1):
                print("Converting scalar to array for comparison...")
                z_scipy_compare = np.array([[z_scipy]])
            else:
                z_scipy_compare = z_scipy
        else:
            z_scipy_compare = z_scipy
            
        exact1 = analyze_differences(z_scipy_compare, z_fortran, "scipy.bisplev", "_dfitpack.bispev")
        exact2 = analyze_differences(z_fortran, z_fortran2, "_dfitpack (1st call)", "_dfitpack (2nd call)")
        
        if not exact1:
            all_exact = False
            
        # Additional checks
        print(f"\nArray properties:")
        print(f"  scipy result dtype: {z_scipy.dtype}")
        print(f"  fortran result dtype: {z_fortran.dtype}")
        print(f"  Arrays share memory: {np.shares_memory(z_scipy, z_fortran)}")
    
    # Test derivative evaluation
    print(f"\n{'='*60}")
    print("Testing derivative evaluation (dx=1, dy=0):")
    
    xi = np.linspace(-4, 4, 20)
    yi = np.linspace(-4, 4, 20)
    
    # scipy with derivatives
    z_scipy_dx = bisplev(xi, yi, tck, dx=1, dy=0)
    
    # Direct fortran call for derivatives
    from scipy.interpolate import dfitpack
    z_fortran_dx, ier = dfitpack.parder(tx, ty, c, kx, ky, 1, 0, xi, yi)
    
    exact_deriv = analyze_differences(z_scipy_dx, z_fortran_dx, 
                                     "scipy.bisplev(dx=1)", "dfitpack.parder")
    if not exact_deriv:
        all_exact = False
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    if all_exact:
        print("✓ ALL comparisons show EXACT bit-for-bit equality!")
        print("  scipy.bisplev and direct Fortran calls produce identical results")
    else:
        print("✗ Some differences detected")
        print("  However, differences are typically within machine precision")
    
    # Test specific edge cases
    print("\n" + "="*60)
    print("EDGE CASE TESTING:")
    print("="*60)
    
    # Test with special values
    special_x = np.array([0.0, 1e-15, 1e15, -1e-15])
    special_y = np.array([0.0, 1e-15, 1e15, -1e-15])
    
    print("\nTesting special values (near zero, large magnitudes)...")
    z_scipy_special = bisplev(special_x, special_y, tck)
    z_fortran_special, _ = _dfitpack.bispev(tx, ty, c, kx, ky, special_x, special_y)
    
    analyze_differences(z_scipy_special, z_fortran_special,
                       "scipy (special)", "fortran (special)")

if __name__ == "__main__":
    test_accuracy_comparison()