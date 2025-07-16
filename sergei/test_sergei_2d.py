#!/usr/bin/env python3
"""
Test script for Sergei's 2D spline cfunc implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import ctypes
from sergei_splines_cfunc_2d import get_cfunc_addresses

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()

# Set up ctypes function signatures
def setup_ctypes_signatures():
    """Setup ctypes function signatures for 2D cfuncs"""
    
    # construct_splines_2d_cfunc
    construct_2d = ctypes.CFUNCTYPE(
        None,  # return type
        ctypes.POINTER(ctypes.c_double),  # x_min array (2)
        ctypes.POINTER(ctypes.c_double),  # x_max array (2)
        ctypes.POINTER(ctypes.c_double),  # y values (flattened)
        ctypes.POINTER(ctypes.c_int32),  # num_points array (2)
        ctypes.POINTER(ctypes.c_int32),  # order array (2)
        ctypes.POINTER(ctypes.c_int32),  # periodic array (2)
        ctypes.POINTER(ctypes.c_double)  # output coeff array
    )(cfunc_addr['construct_splines_2d'])
    
    # evaluate_splines_2d_cfunc
    evaluate_2d = ctypes.CFUNCTYPE(
        None,  # return type
        ctypes.POINTER(ctypes.c_int32),  # order array (2)
        ctypes.POINTER(ctypes.c_int32),  # num_points array (2)
        ctypes.POINTER(ctypes.c_int32),  # periodic array (2)
        ctypes.POINTER(ctypes.c_double),  # x_min array (2)
        ctypes.POINTER(ctypes.c_double),  # h_step array (2)
        ctypes.POINTER(ctypes.c_double),  # coeff array
        ctypes.POINTER(ctypes.c_double),  # x array (2)
        ctypes.POINTER(ctypes.c_double)  # output y
    )(cfunc_addr['evaluate_splines_2d'])
    
    return construct_2d, evaluate_2d


def test_2d_spline(order=(3, 3), periodic=(False, False)):
    """Test 2D spline with visual validation"""
    
    construct_2d, evaluate_2d = setup_ctypes_signatures()
    
    # Create test data
    n1, n2 = 15, 20
    x_min = np.array([0.0, 0.0])
    x_max = np.array([4.0, 6.0])
    
    # Create grid
    x1 = np.linspace(x_min[0], x_max[0], n1)
    x2 = np.linspace(x_min[1], x_max[1], n2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    
    # Test function: 2D Gaussian with some oscillation
    Z = np.exp(-0.5 * ((X1 - 2.0)**2 + (X2 - 3.0)**2)) * np.sin(X1) * np.cos(X2)
    
    # Prepare ctypes arrays
    x_min_c = (ctypes.c_double * 2)(*x_min)
    x_max_c = (ctypes.c_double * 2)(*x_max)
    num_points_c = (ctypes.c_int32 * 2)(n1, n2)
    order_c = (ctypes.c_int32 * 2)(*order)
    periodic_c = (ctypes.c_int32 * 2)(1 if periodic[0] else 0, 1 if periodic[1] else 0)
    
    # Flatten Z array (row-major order)
    z_flat = Z.flatten()
    z_c = (ctypes.c_double * len(z_flat))(*z_flat)
    
    # Allocate coefficient array
    coeff_size = (order[0] + 1) * (order[1] + 1) * n1 * n2
    coeff_c = (ctypes.c_double * coeff_size)()
    
    # Construct spline
    print(f"Constructing 2D spline with order={order}, periodic={periodic}")
    construct_2d(x_min_c, x_max_c, z_c, num_points_c, order_c, periodic_c, coeff_c)
    
    # Calculate h_step for evaluation
    h_step = np.array([(x_max[i] - x_min[i]) / (np.array([n1, n2])[i] - 1) for i in range(2)])
    h_step_c = (ctypes.c_double * 2)(*h_step)
    
    # Create evaluation grid
    n_eval = 50
    x1_eval = np.linspace(x_min[0], x_max[0], n_eval)
    x2_eval = np.linspace(x_min[1], x_max[1], n_eval)
    X1_eval, X2_eval = np.meshgrid(x1_eval, x2_eval, indexing='ij')
    
    # Evaluate spline
    Z_eval = np.zeros((n_eval, n_eval))
    x_point = (ctypes.c_double * 2)()
    y_out = (ctypes.c_double * 1)()
    
    print("Evaluating spline on fine grid...")
    for i in range(n_eval):
        for j in range(n_eval):
            x_point[0] = X1_eval[i, j]
            x_point[1] = X2_eval[i, j]
            evaluate_2d(order_c, num_points_c, periodic_c, x_min_c, h_step_c, 
                       coeff_c, x_point, y_out)
            Z_eval[i, j] = y_out[0]
    
    # Compute error on original grid points
    errors = []
    for i in range(n1):
        for j in range(n2):
            x_point[0] = x1[i]
            x_point[1] = x2[j]
            evaluate_2d(order_c, num_points_c, periodic_c, x_min_c, h_step_c,
                       coeff_c, x_point, y_out)
            error = abs(y_out[0] - Z[i, j])
            errors.append(error)
    
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    print(f"Interpolation error - Max: {max_error:.2e}, Avg: {avg_error:.2e}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    im1 = axes[0, 0].contourf(X2, X1, Z, levels=20, cmap='viridis')
    axes[0, 0].scatter(X2.flatten(), X1.flatten(), c='red', s=10, alpha=0.5)
    axes[0, 0].set_title('Original Data')
    axes[0, 0].set_xlabel('x2')
    axes[0, 0].set_ylabel('x1')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Interpolated data
    im2 = axes[0, 1].contourf(X2_eval, X1_eval, Z_eval, levels=20, cmap='viridis')
    axes[0, 1].set_title('Spline Interpolation')
    axes[0, 1].set_xlabel('x2')
    axes[0, 1].set_ylabel('x1')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Cross-section at x1 = 2.0
    idx1 = n_eval // 2
    axes[1, 0].plot(x2_eval, Z_eval[idx1, :], 'b-', label='Spline')
    # Plot original data points near this cross-section
    idx1_orig = np.argmin(np.abs(x1 - 2.0))
    axes[1, 0].plot(x2, Z[idx1_orig, :], 'ro', label='Data points')
    axes[1, 0].set_title('Cross-section at x1 = 2.0')
    axes[1, 0].set_xlabel('x2')
    axes[1, 0].set_ylabel('z')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Cross-section at x2 = 3.0
    idx2 = n_eval // 2
    axes[1, 1].plot(x1_eval, Z_eval[:, idx2], 'b-', label='Spline')
    # Plot original data points near this cross-section
    idx2_orig = np.argmin(np.abs(x2 - 3.0))
    axes[1, 1].plot(x1, Z[:, idx2_orig], 'ro', label='Data points')
    axes[1, 1].set_title('Cross-section at x2 = 3.0')
    axes[1, 1].set_xlabel('x1')
    axes[1, 1].set_ylabel('z')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'sergei_2d_spline_order_{order[0]}_{order[1]}_periodic_{periodic[0]}_{periodic[1]}.png')
    plt.show()
    
    return max_error, avg_error


if __name__ == "__main__":
    print("Testing Sergei's 2D spline implementation\n")
    
    # Test different configurations
    test_cases = [
        ((3, 3), (False, False)),  # Cubic, non-periodic
        ((3, 3), (True, False)),   # Cubic, periodic in x1
        ((3, 3), (False, True)),   # Cubic, periodic in x2
        ((3, 3), (True, True)),    # Cubic, fully periodic
    ]
    
    for order, periodic in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing order={order}, periodic={periodic}")
        print('='*60)
        
        try:
            max_err, avg_err = test_2d_spline(order, periodic)
            print(f"✓ Test passed! Max error: {max_err:.2e}, Avg error: {avg_err:.2e}")
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()