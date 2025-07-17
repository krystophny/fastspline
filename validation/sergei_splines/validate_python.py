#!/usr/bin/env python3
"""
Python validation program for Sergei splines using cfunc implementation.
This program replicates the Fortran validation to compare results.
"""

import numpy as np
import sys
import os

# Add the fastspline source to the path
sys.path.insert(0, os.path.abspath('../../src'))

try:
    from fastspline.sergei_splines import (
        construct_splines_1d, construct_splines_2d, construct_splines_3d,
        eval_spline_1d, eval_spline_2d, eval_spline_3d,
        eval_spline_1d_derivative, eval_spline_2d_derivative
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct import of sergei_splines module...")
    # Try importing the module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("sergei_splines", 
                                                   "../../src/fastspline/sergei_splines.py")
    sergei_splines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sergei_splines)
    
    construct_splines_1d = sergei_splines.construct_splines_1d
    construct_splines_2d = sergei_splines.construct_splines_2d
    construct_splines_3d = sergei_splines.construct_splines_3d
    eval_spline_1d = sergei_splines.eval_spline_1d
    eval_spline_2d = sergei_splines.eval_spline_2d
    eval_spline_3d = sergei_splines.eval_spline_3d
    eval_spline_1d_derivative = sergei_splines.eval_spline_1d_derivative
    eval_spline_2d_derivative = sergei_splines.eval_spline_2d_derivative


def test_1d_spline():
    """Test 1D spline construction and evaluation."""
    # Test parameters (matching Fortran)
    n = 10
    x_min = 0.0
    x_max = 1.0
    order = 5
    periodic = False
    
    # Create test data
    x = np.linspace(x_min, x_max, n)
    y = np.sin(2.0 * np.pi * x)
    
    # Write input data
    os.makedirs('data', exist_ok=True)
    with open('data/input_data_python.txt', 'w') as f:
        f.write('# Input data for Python validation\n')
        f.write(f'# n = {n}\n')
        f.write(f'# x_min = {x_min:12.6f}\n')
        f.write(f'# x_max = {x_max:12.6f}\n')
        f.write(f'# order = {order}\n')
        f.write(f'# periodic = {periodic}\n')
        f.write('# x, y values:\n')
        for i in range(n):
            f.write(f'{x[i]:20.12f}{y[i]:20.12f}\n')
    
    # Construct spline
    print("Constructing 1D spline...")
    x_min_out, h_step, coeffs = construct_splines_1d(x_min, x_max, y, order, periodic)
    
    # Write spline coefficients
    with open('data/spline_coeffs_1d_python.txt', 'w') as f:
        f.write('# 1D Spline coefficients (Python)\n')
        f.write(f'# order = {order}\n')
        f.write(f'# num_points = {n}\n')
        f.write(f'# periodic = {periodic}\n')
        f.write(f'# x_min = {x_min_out:20.12f}\n')
        f.write(f'# h_step = {h_step:20.12f}\n')
        f.write('# Coefficients shape: {}\n'.format(coeffs.shape))
        f.write('# Coefficients (i, j, coeff[i,j]):\n')
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                f.write(f'{i+1:5d}{j+1:5d}{coeffs[i,j]:20.12f}\n')
    
    # Evaluate spline at test points
    x_eval = np.linspace(x_min, x_max, 21)
    y_exact = np.sin(2.0 * np.pi * x_eval)
    
    with open('data/evaluation_results_python.txt', 'w') as f:
        f.write('# Spline evaluation results (Python)\n')
        f.write('# x, y_spline, y_exact, error\n')
        
        for i in range(len(x_eval)):
            y_spline = eval_spline_1d(x_eval[i], x_min_out, h_step, coeffs, order, periodic)
            error = abs(y_spline - y_exact[i])
            f.write(f'{x_eval[i]:20.12f}{y_spline:20.12f}{y_exact[i]:20.12f}{error:20.12f}\n')
    
    # Test derivatives
    with open('data/derivatives_1d_python.txt', 'w') as f:
        f.write('# 1D Spline derivatives (Python)\n')
        f.write('# x, dy/dx, d2y/dx2\n')
        
        for i in range(len(x_eval)):
            dy = eval_spline_1d_derivative(x_eval[i], x_min_out, h_step, coeffs, order, periodic, 1)
            d2y = eval_spline_1d_derivative(x_eval[i], x_min_out, h_step, coeffs, order, periodic, 2)
            f.write(f'{x_eval[i]:20.12f}{dy:20.12f}{d2y:20.12f}\n')
    
    print("1D spline test complete.")


def test_2d_spline():
    """Test 2D spline construction and evaluation."""
    # 2D test parameters (matching Fortran)
    nx = 8
    ny = 8
    x_min = np.array([0.0, 0.0])
    x_max = np.array([1.0, 1.0])
    order = np.array([5, 5], dtype=np.int32)
    periodic = np.array([False, False])
    
    # Create 2D test data
    x1 = np.linspace(x_min[0], x_max[0], nx)
    x2 = np.linspace(x_min[1], x_max[1], ny)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    z = np.sin(2.0 * np.pi * X1) * np.cos(2.0 * np.pi * X2)
    
    # Write 2D input data
    with open('data/input_data_2d_python.txt', 'w') as f:
        f.write('# 2D Input data for Python validation\n')
        f.write(f'# nx, ny = {nx:5d}{ny:5d}\n')
        f.write(f'# x_min = {x_min[0]:12.6f}{x_min[1]:12.6f}\n')
        f.write(f'# x_max = {x_max[0]:12.6f}{x_max[1]:12.6f}\n')
        f.write(f'# order = {order[0]:5d}{order[1]:5d}\n')
        f.write(f'# periodic = {periodic[0]:5}{periodic[1]:5}\n')
        f.write('# z values (row by row):\n')
        for i in range(nx):
            f.write(' '.join(f'{z[i,j]:12.6f}' for j in range(ny)) + '\n')
    
    # Construct 2D spline
    print("Constructing 2D spline...")
    x_min_out, h_step, coeffs = construct_splines_2d(x_min, x_max, z, order, periodic)
    
    # Write some coefficient info
    with open('data/spline_info_2d_python.txt', 'w') as f:
        f.write('# 2D Spline info (Python)\n')
        f.write(f'# x_min_out = {x_min_out}\n')
        f.write(f'# h_step = {h_step}\n')
        f.write(f'# coeffs shape = {coeffs.shape}\n')
        f.write(f'# coeffs dtype = {coeffs.dtype}\n')
        f.write(f'# coeffs min/max = {coeffs.min():.12f} / {coeffs.max():.12f}\n')
    
    # Evaluate 2D spline at test points
    x1_eval = np.linspace(x_min[0], x_max[0], 11)
    x2_eval = np.linspace(x_min[1], x_max[1], 11)
    
    with open('data/evaluation_results_2d_python.txt', 'w') as f:
        f.write('# 2D Spline evaluation results (Python)\n')
        f.write('# x1, x2, z_spline, z_exact, error\n')
        
        for i in range(11):
            for j in range(11):
                x_eval = np.array([x1_eval[i], x2_eval[j]])
                z_spline = eval_spline_2d(x_eval, x_min_out, h_step, coeffs, order, periodic)
                z_exact = np.sin(2.0 * np.pi * x_eval[0]) * np.cos(2.0 * np.pi * x_eval[1])
                error = abs(z_spline - z_exact)
                f.write(f'{x_eval[0]:20.12f}{x_eval[1]:20.12f}{z_spline:20.12f}{z_exact:20.12f}{error:20.12f}\n')
    
    print("2D spline test complete.")


def print_memory_info():
    """Print memory alignment information for debugging."""
    print("\nMemory alignment information:")
    
    # Test arrays
    test_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    print(f"Float64 array alignment: {test_array.ctypes.data % 8} (should be 0 for 8-byte alignment)")
    
    # Test different array creation methods
    test_c = np.ascontiguousarray(test_array)
    test_f = np.asfortranarray(test_array)
    print(f"C-contiguous alignment: {test_c.ctypes.data % 8}")
    print(f"F-contiguous alignment: {test_f.ctypes.data % 8}")
    
    # Test coefficient arrays
    coeffs_1d = np.zeros((10, 6), dtype=np.float64)
    print(f"1D coeffs alignment: {coeffs_1d.ctypes.data % 8}")
    
    coeffs_2d = np.zeros((8, 8, 6, 6), dtype=np.float64)
    print(f"2D coeffs alignment: {coeffs_2d.ctypes.data % 8}")


def main():
    """Main validation function."""
    print("Python validation for Sergei splines")
    print("====================================")
    
    # Print memory info for debugging
    print_memory_info()
    
    # Run tests
    test_1d_spline()
    test_2d_spline()
    
    print("\nPython validation complete. Check data/ directory for output files.")
    print("Compare *_python.txt files with corresponding Fortran output files.")


if __name__ == "__main__":
    main()