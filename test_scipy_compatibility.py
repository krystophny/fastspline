#!/usr/bin/env python3
"""Test SciPy compatibility for bisplev behavior."""

import numpy as np
from scipy.interpolate import bisplrep as scipy_bisplrep, bisplev as scipy_bisplev
from fastspline import bisplrep, bisplev

def test_bisplev_shape_compatibility():
    """Test that bisplev returns same shapes as SciPy for different input combinations."""
    print("Testing bisplev shape compatibility...")
    
    # Create test data
    np.random.seed(42)
    x_data = np.random.uniform(-1, 1, 100)
    y_data = np.random.uniform(-1, 1, 100)
    z_data = np.exp(-(x_data**2 + y_data**2)) * np.cos(np.pi * x_data)
    
    # Fit spline with more relaxed smoothing to avoid warnings
    tck = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0.1)
    
    test_cases = [
        # Case 1: Both 1D arrays (should create meshgrid)
        ("1D × 1D", np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5])),
        
        # Case 2: Scalar × 1D array
        ("scalar × 1D", 0.1, np.array([0.4, 0.5, 0.6])),
        
        # Case 3: 1D array × scalar
        ("1D × scalar", np.array([0.1, 0.2, 0.3]), 0.4),
        
        # Case 4: Both scalars
        ("scalar × scalar", 0.1, 0.4),
        
        # Case 5: Larger 1D arrays (like user's case)
        ("Large 1D × 1D", np.linspace(-0.8, 0.8, 50), np.linspace(-0.8, 0.8, 40)),
    ]
    
    all_passed = True
    
    for case_name, x_test, y_test in test_cases:
        print(f"\nTesting {case_name}:")
        
        # SciPy result
        result_scipy = scipy_bisplev(x_test, y_test, tck)
        
        # FastSpline result
        result_fast = bisplev(x_test, y_test, tck)
        
        # Check shapes
        if np.isscalar(result_scipy) and np.isscalar(result_fast):
            shape_match = True
            print(f"  SciPy: scalar, FastSpline: scalar")
        elif hasattr(result_scipy, 'shape') and hasattr(result_fast, 'shape'):
            shape_match = result_scipy.shape == result_fast.shape
            print(f"  SciPy shape: {result_scipy.shape}")
            print(f"  FastSpline shape: {result_fast.shape}")
        else:
            shape_match = False
            print(f"  SciPy type: {type(result_scipy)}")
            print(f"  FastSpline type: {type(result_fast)}")
        
        print(f"  Shape match: {shape_match}")
        
        # Check accuracy
        diff = np.abs(result_scipy - result_fast)
        max_diff = np.max(diff)
        print(f"  Max difference: {max_diff:.2e}")
        
        # Check types for scalar cases
        if np.isscalar(result_scipy):
            print(f"  SciPy type: {type(result_scipy)}")
            print(f"  FastSpline type: {type(result_fast)}")
        
        # Test passes if shapes match and difference is small
        case_passed = shape_match and max_diff < 1e-12
        print(f"  {'✓ PASSED' if case_passed else '✗ FAILED'}")
        
        if not case_passed:
            all_passed = False
    
    return all_passed

def test_meshgrid_behavior():
    """Test that 1D × 1D arrays create proper meshgrid behavior."""
    print("\nTesting meshgrid behavior in detail...")
    
    # Create simple test data - scattered points
    x_data = np.array([0.0, 1.0, 2.0, 0.5, 1.5, 2.5])
    y_data = np.array([0.0, 1.0, 0.5, 1.5, 0.25, 0.75])
    z_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Fit spline
    tck = bisplrep(x_data, y_data, z_data, kx=1, ky=1, s=0)
    
    # Test evaluation arrays
    x_eval = np.array([0.5, 1.5])  # 2 points
    y_eval = np.array([0.25, 0.75])  # 2 points
    
    # SciPy: Should create 2×2 meshgrid
    result_scipy = scipy_bisplev(x_eval, y_eval, tck)
    result_fast = bisplev(x_eval, y_eval, tck)
    
    print(f"Input x_eval shape: {x_eval.shape}")
    print(f"Input y_eval shape: {y_eval.shape}")
    print(f"SciPy result shape: {result_scipy.shape}")
    print(f"FastSpline result shape: {result_fast.shape}")
    print(f"Expected shape: (2, 2)")
    
    # Check that we get the same values at corresponding meshgrid points
    for i in range(len(x_eval)):
        for j in range(len(y_eval)):
            x_pt, y_pt = x_eval[i], y_eval[j]
            
            # Single point evaluation
            val_scipy = scipy_bisplev(x_pt, y_pt, tck)
            val_fast = bisplev(x_pt, y_pt, tck)
            
            # Meshgrid evaluation
            val_scipy_mesh = result_scipy[i, j]
            val_fast_mesh = result_fast[i, j]
            
            print(f"Point ({x_pt}, {y_pt}): scipy={val_scipy:.6f}, fast={val_fast:.6f}")
            print(f"  From meshgrid: scipy={val_scipy_mesh:.6f}, fast={val_fast_mesh:.6f}")
            
            # Check consistency
            if abs(val_scipy - val_scipy_mesh) > 1e-12 or abs(val_fast - val_fast_mesh) > 1e-12:
                print("  ✗ INCONSISTENCY in meshgrid behavior!")
                return False
    
    print("✓ Meshgrid behavior is consistent")
    return True

if __name__ == "__main__":
    print("SciPy Compatibility Test")
    print("=" * 50)
    
    shape_ok = test_bisplev_shape_compatibility()
    meshgrid_ok = test_meshgrid_behavior()
    
    if shape_ok and meshgrid_ok:
        print("\n✓ All SciPy compatibility tests PASSED")
    else:
        print("\n✗ Some SciPy compatibility tests FAILED")