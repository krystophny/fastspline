#!/usr/bin/env python3
"""
Compare results between Fortran and Python implementations of Sergei splines.
This script reads output files from both implementations and reports differences.
"""

import numpy as np
import sys
import os


def read_data_file(filename):
    """Read a data file, skipping comment lines."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip().startswith('#'):
                data.append([float(x) for x in line.split()])
    return np.array(data)


def compare_files(file1, file2, description, tol=1e-10):
    """Compare two data files and report differences."""
    print(f"\nComparing {description}")
    print("=" * 60)
    
    if not os.path.exists(file1):
        print(f"ERROR: {file1} does not exist!")
        return False
    
    if not os.path.exists(file2):
        print(f"ERROR: {file2} does not exist!")
        return False
    
    try:
        data1 = read_data_file(file1)
        data2 = read_data_file(file2)
    except Exception as e:
        print(f"ERROR reading files: {e}")
        return False
    
    if data1.shape != data2.shape:
        print(f"ERROR: Shape mismatch! {file1}: {data1.shape}, {file2}: {data2.shape}")
        return False
    
    # Compute differences
    abs_diff = np.abs(data1 - data2)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    # Find location of maximum difference
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    
    print(f"Data shape: {data1.shape}")
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Max diff at index {max_idx}: {data1[max_idx]:.12f} vs {data2[max_idx]:.12f}")
    
    if max_diff > tol:
        print(f"WARNING: Maximum difference exceeds tolerance ({tol:.2e})")
        
        # Show first few differences
        print("\nFirst 5 largest differences:")
        flat_diff = abs_diff.flatten()
        sorted_indices = np.argsort(flat_diff)[::-1][:5]
        
        for idx in sorted_indices:
            unraveled = np.unravel_index(idx, abs_diff.shape)
            print(f"  Index {unraveled}: {data1[unraveled]:.12f} vs {data2[unraveled]:.12f} "
                  f"(diff: {abs_diff[unraveled]:.2e})")
        
        return False
    else:
        print("PASS: All values within tolerance")
        return True


def compare_coefficients():
    """Compare spline coefficients between implementations."""
    print("\nComparing spline coefficients")
    print("=" * 60)
    
    # Read Fortran coefficients
    fortran_coeffs = []
    with open('data/spline_coeffs_1d.txt', 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) == 3:
                    i, j, coeff = int(parts[0]), int(parts[1]), float(parts[2])
                    fortran_coeffs.append((i, j, coeff))
    
    # Read Python coefficients
    python_coeffs = []
    with open('data/spline_coeffs_1d_python.txt', 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.split()
                if len(parts) == 3:
                    i, j, coeff = int(parts[0]), int(parts[1]), float(parts[2])
                    python_coeffs.append((i, j, coeff))
    
    if len(fortran_coeffs) != len(python_coeffs):
        print(f"ERROR: Different number of coefficients! Fortran: {len(fortran_coeffs)}, "
              f"Python: {len(python_coeffs)}")
        return False
    
    max_diff = 0.0
    max_diff_info = None
    
    for (i1, j1, c1), (i2, j2, c2) in zip(fortran_coeffs, python_coeffs):
        if i1 != i2 or j1 != j2:
            print(f"ERROR: Index mismatch! ({i1},{j1}) vs ({i2},{j2})")
            return False
        
        diff = abs(c1 - c2)
        if diff > max_diff:
            max_diff = diff
            max_diff_info = (i1, j1, c1, c2)
    
    print(f"Number of coefficients: {len(fortran_coeffs)}")
    print(f"Maximum coefficient difference: {max_diff:.2e}")
    if max_diff_info:
        i, j, c1, c2 = max_diff_info
        print(f"Max diff at ({i},{j}): {c1:.12f} vs {c2:.12f}")
    
    return max_diff < 1e-10


def analyze_memory_alignment():
    """Analyze potential memory alignment issues."""
    print("\nAnalyzing memory alignment patterns")
    print("=" * 60)
    
    # Check if differences follow a pattern
    if os.path.exists('data/evaluation_results.txt') and os.path.exists('data/evaluation_results_python.txt'):
        fortran_data = read_data_file('data/evaluation_results.txt')
        python_data = read_data_file('data/evaluation_results_python.txt')
        
        if fortran_data.shape == python_data.shape and len(fortran_data) > 0:
            # Check evaluation values (column 1)
            fortran_vals = fortran_data[:, 1]
            python_vals = python_data[:, 1]
            
            diffs = fortran_vals - python_vals
            
            # Look for patterns
            print(f"Evaluation differences statistics:")
            print(f"  Mean: {np.mean(diffs):.2e}")
            print(f"  Std:  {np.std(diffs):.2e}")
            print(f"  Min:  {np.min(diffs):.2e}")
            print(f"  Max:  {np.max(diffs):.2e}")
            
            # Check for periodic patterns
            if len(diffs) > 10:
                print(f"\nFirst 10 differences:")
                for i in range(min(10, len(diffs))):
                    print(f"  {i}: {diffs[i]:.2e}")


def main():
    """Main comparison function."""
    print("Sergei Splines Validation Comparison")
    print("====================================")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("ERROR: data directory not found. Run Fortran and Python programs first.")
        sys.exit(1)
    
    all_pass = True
    
    # Compare 1D results
    all_pass &= compare_files(
        'data/evaluation_results.txt',
        'data/evaluation_results_python.txt',
        '1D spline evaluation results',
        tol=1e-10
    )
    
    # Compare coefficients
    all_pass &= compare_coefficients()
    
    # Compare 2D results
    all_pass &= compare_files(
        'data/evaluation_results_2d.txt',
        'data/evaluation_results_2d_python.txt',
        '2D spline evaluation results',
        tol=1e-10
    )
    
    # Analyze memory alignment issues
    analyze_memory_alignment()
    
    # Summary
    print("\n" + "=" * 60)
    if all_pass:
        print("OVERALL: All tests passed!")
    else:
        print("OVERALL: Some tests failed - check for memory alignment issues")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())