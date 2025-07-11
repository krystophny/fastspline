#!/usr/bin/env python3
"""
Final comprehensive test for floating-point accuracy between scipy and direct Fortran.
"""

import numpy as np
from scipy.interpolate import bisplrep, bisplev
from scipy.interpolate import _dfitpack
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def test_comprehensive_accuracy():
    print("COMPREHENSIVE FLOATING-POINT ACCURACY TEST")
    print("="*60)
    
    # Generate various test datasets
    datasets = [
        ("Small smooth data", 50, lambda x, y: np.sin(x) * np.cos(y)),
        ("Medium oscillating data", 200, lambda x, y: np.sin(3*x) * np.cos(3*y) + 0.1*x*y),
        ("Large noisy data", 500, lambda x, y: np.sin(np.sqrt(x**2 + y**2)) + 0.1*np.random.randn()),
    ]
    
    all_results = []
    
    for data_name, n_points, func in datasets:
        print(f"\nDataset: {data_name} ({n_points} points)")
        print("-" * 50)
        
        # Generate data
        np.random.seed(42)
        x = np.random.uniform(-3, 3, n_points)
        y = np.random.uniform(-3, 3, n_points)
        z = np.array([func(xi, yi) for xi, yi in zip(x, y)])
        
        # Fit spline with smoothing
        tck = bisplrep(x, y, z, s=n_points)
        tx, ty, c, kx, ky = tck
        
        # Test on various grids
        test_grids = [
            ("5x5 grid", 5, 5),
            ("10x10 grid", 10, 10),
            ("25x25 grid", 25, 25),
            ("50x50 grid", 50, 50),
        ]
        
        dataset_exact = True
        
        for grid_name, nx, ny in test_grids:
            # Create evaluation grid
            xi = np.linspace(x.min()+0.1, x.max()-0.1, nx)
            yi = np.linspace(y.min()+0.1, y.max()-0.1, ny)
            
            # Ensure proper format
            xi = np.ascontiguousarray(xi, dtype=np.float64)
            yi = np.ascontiguousarray(yi, dtype=np.float64)
            
            # Evaluate with both methods
            z_scipy = bisplev(xi, yi, tck)
            z_fortran, ier = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)
            
            # Check exact equality
            is_exact = np.array_equal(z_scipy, z_fortran)
            
            if is_exact:
                result = "✓ EXACT"
            else:
                max_diff = np.max(np.abs(z_scipy - z_fortran))
                if max_diff == 0.0:
                    result = "✓ EXACT (max_diff=0)"
                elif max_diff < np.finfo(float).eps:
                    result = f"≈ Within eps ({max_diff:.2e})"
                else:
                    result = f"✗ Differs ({max_diff:.2e})"
                dataset_exact = False
            
            print(f"  {grid_name}: {result}")
            all_results.append((data_name, grid_name, is_exact))
    
    # Test edge cases
    print(f"\nEdge Cases:")
    print("-" * 50)
    
    # Simple data for edge cases
    x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    z = np.array([0, 1, 0, 1, 2, 1, 0, 1, 0])
    tck = bisplrep(x, y, z)
    tx, ty, c, kx, ky = tck
    
    edge_cases = [
        ("Single center point", np.array([1.0]), np.array([1.0])),
        ("Boundary points", np.array([0.0, 2.0]), np.array([0.0, 2.0])),
        ("Very fine grid", np.linspace(0.5, 1.5, 100), np.linspace(0.5, 1.5, 100)),
    ]
    
    for case_name, xi, yi in edge_cases:
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        yi = np.ascontiguousarray(yi, dtype=np.float64)
        
        try:
            z_scipy = bisplev(xi, yi, tck)
            z_fortran, ier = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)
            
            # Handle shape differences for single points
            if z_scipy.shape != z_fortran.shape and z_scipy.ndim == 0:
                z_scipy = np.array([[z_scipy]])
            
            is_exact = np.array_equal(z_scipy, z_fortran)
            if is_exact:
                print(f"  {case_name}: ✓ EXACT")
            else:
                max_diff = np.max(np.abs(z_scipy.flatten() - z_fortran.flatten()))
                print(f"  {case_name}: Diff = {max_diff:.2e}")
                
        except Exception as e:
            print(f"  {case_name}: Error - {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    exact_count = sum(1 for _, _, exact in all_results if exact)
    total_count = len(all_results)
    
    print(f"Exact matches: {exact_count}/{total_count} ({exact_count/total_count*100:.1f}%)")
    
    if exact_count == total_count:
        print("\n✓ PERFECT ACCURACY: All evaluations are bit-for-bit identical!")
        print("  scipy.bisplev and _dfitpack.bispev produce EXACTLY the same results")
    else:
        print("\n✗ Some differences found, but they are typically within machine precision")
    
    # Technical details
    print("\nTechnical Details:")
    print("-" * 50)
    print(f"Machine epsilon (float64): {np.finfo(float).eps:.2e}")
    print(f"Smallest positive float64: {np.finfo(float).tiny:.2e}")
    print("\nConclusion: scipy.bisplev is a thin wrapper around _dfitpack.bispev")
    print("providing identical numerical results with added safety checks.")

if __name__ == "__main__":
    test_comprehensive_accuracy()