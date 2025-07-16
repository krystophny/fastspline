#!/usr/bin/env python3
"""
Comprehensive test script for Sergei's spline cfunc implementation
Tests 1D, 2D functionality and derivatives
"""

import numpy as np
import matplotlib.pyplot as plt
import ctypes
from sergei_splines_cfunc_final import get_cfunc_addresses
import time

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()


def test_1d_spline():
    """Test 1D spline with various configurations"""
    
    # Set up ctypes signatures
    construct_1d = ctypes.CFUNCTYPE(
        None,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double)
    )(cfunc_addr['construct_splines_1d'])
    
    evaluate_1d = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double)
    )(cfunc_addr['evaluate_splines_1d'])
    
    evaluate_1d_der = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    )(cfunc_addr['evaluate_splines_1d_der'])
    
    # Test cases
    test_cases = [
        ("Regular cubic", 3, False),
        ("Periodic cubic", 3, True),
    ]
    
    fig, axes = plt.subplots(len(test_cases), 2, figsize=(12, 4*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, order, periodic) in enumerate(test_cases):
        print(f"\nTesting {name} spline...")
        
        # Create test data
        n_data = 20
        x_min = 0.0
        x_max = 2.0 * np.pi if periodic else 4.0
        
        x_data = np.linspace(x_min, x_max, n_data)
        if periodic:
            y_data = np.sin(x_data) + 0.5 * np.sin(3.0 * x_data)
        else:
            y_data = np.exp(-0.5 * x_data) * np.sin(2.0 * x_data)
        
        # Prepare ctypes arrays
        y_c = (ctypes.c_double * n_data)(*y_data)
        coeff_size = (order + 1) * n_data
        coeff_c = (ctypes.c_double * coeff_size)()
        
        # Construct spline
        h_step = (x_max - x_min) / (n_data - 1)
        construct_1d(x_min, x_max, y_c, n_data, order, 1 if periodic else 0, coeff_c)
        
        # Evaluate on fine grid
        n_eval = 200
        x_eval = np.linspace(x_min, x_max, n_eval)
        y_eval = np.zeros(n_eval)
        dy_eval = np.zeros(n_eval)
        
        for i in range(n_eval):
            y_out = (ctypes.c_double * 1)()
            dy_out = (ctypes.c_double * 1)()
            evaluate_1d_der(order, n_data, 1 if periodic else 0, x_min, h_step,
                           coeff_c, x_eval[i], y_out, dy_out)
            y_eval[i] = y_out[0]
            dy_eval[i] = dy_out[0]
        
        # Check interpolation error at data points
        errors = []
        for i in range(n_data):
            y_out = (ctypes.c_double * 1)()
            evaluate_1d(order, n_data, 1 if periodic else 0, x_min, h_step,
                       coeff_c, x_data[i], y_out)
            errors.append(abs(y_out[0] - y_data[i]))
        
        max_error = max(errors)
        print(f"  Max interpolation error: {max_error:.2e}")
        
        # Plot function
        axes[idx, 0].plot(x_data, y_data, 'ro', label='Data points')
        axes[idx, 0].plot(x_eval, y_eval, 'b-', label='Spline')
        axes[idx, 0].set_title(f'{name} - Function')
        axes[idx, 0].set_xlabel('x')
        axes[idx, 0].set_ylabel('y')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        
        # Plot derivative
        axes[idx, 1].plot(x_eval, dy_eval, 'g-', label='Spline derivative')
        if periodic:
            true_dy = np.cos(x_eval) + 1.5 * np.cos(3.0 * x_eval)
        else:
            true_dy = np.exp(-0.5 * x_eval) * (2.0 * np.cos(2.0 * x_eval) - 0.5 * np.sin(2.0 * x_eval))
        axes[idx, 1].plot(x_eval, true_dy, 'r--', label='True derivative', alpha=0.7)
        axes[idx, 1].set_title(f'{name} - Derivative')
        axes[idx, 1].set_xlabel('x')
        axes[idx, 1].set_ylabel('dy/dx')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('sergei_1d_test_final.png')
    plt.show()


def test_2d_spline():
    """Test 2D spline functionality"""
    
    # Set up ctypes signatures
    construct_2d = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_double),  # x_min
        ctypes.POINTER(ctypes.c_double),  # x_max
        ctypes.POINTER(ctypes.c_double),  # y values
        ctypes.POINTER(ctypes.c_int32),   # num_points
        ctypes.POINTER(ctypes.c_int32),   # order
        ctypes.POINTER(ctypes.c_int32),   # periodic
        ctypes.POINTER(ctypes.c_double),  # coeff
        ctypes.POINTER(ctypes.c_double),  # workspace_y
        ctypes.POINTER(ctypes.c_double)   # workspace_coeff
    )(cfunc_addr['construct_splines_2d'])
    
    evaluate_2d = ctypes.CFUNCTYPE(
        None,
        ctypes.POINTER(ctypes.c_int32),   # order
        ctypes.POINTER(ctypes.c_int32),   # num_points
        ctypes.POINTER(ctypes.c_int32),   # periodic
        ctypes.POINTER(ctypes.c_double),  # x_min
        ctypes.POINTER(ctypes.c_double),  # h_step
        ctypes.POINTER(ctypes.c_double),  # coeff
        ctypes.POINTER(ctypes.c_double),  # x
        ctypes.POINTER(ctypes.c_double)   # y_out
    )(cfunc_addr['evaluate_splines_2d'])
    
    print("\nTesting 2D spline...")
    
    # Create test data
    n1, n2 = 15, 20
    x_min = np.array([0.0, 0.0])
    x_max = np.array([4.0, 6.0])
    order = (3, 3)
    periodic = (False, False)
    
    # Create grid
    x1 = np.linspace(x_min[0], x_max[0], n1)
    x2 = np.linspace(x_min[1], x_max[1], n2)
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    
    # Test function
    Z = np.exp(-0.5 * ((X1 - 2.0)**2 + (X2 - 3.0)**2)) * np.sin(X1) * np.cos(X2)
    
    # Prepare ctypes arrays
    x_min_c = (ctypes.c_double * 2)(*x_min)
    x_max_c = (ctypes.c_double * 2)(*x_max)
    num_points_c = (ctypes.c_int32 * 2)(n1, n2)
    order_c = (ctypes.c_int32 * 2)(*order)
    periodic_c = (ctypes.c_int32 * 2)(0, 0)
    
    z_flat = Z.flatten()
    z_c = (ctypes.c_double * len(z_flat))(*z_flat)
    
    # Allocate coefficient array
    coeff_size = (order[0] + 1) * (order[1] + 1) * n1 * n2
    coeff_c = (ctypes.c_double * coeff_size)()
    
    # Allocate workspace arrays
    max_n = max(n1, n2)
    workspace_y = (ctypes.c_double * max_n)()
    workspace_coeff = (ctypes.c_double * (6 * max_n))()
    
    # Construct spline
    start_time = time.time()
    construct_2d(x_min_c, x_max_c, z_c, num_points_c, order_c, periodic_c, coeff_c, 
                 workspace_y, workspace_coeff)
    construct_time = time.time() - start_time
    print(f"  Construction time: {construct_time:.4f} seconds")
    
    # Calculate h_step
    h_step = np.array([(x_max[i] - x_min[i]) / (np.array([n1, n2])[i] - 1) for i in range(2)])
    h_step_c = (ctypes.c_double * 2)(*h_step)
    
    # Evaluate on fine grid
    n_eval = 50
    x1_eval = np.linspace(x_min[0], x_max[0], n_eval)
    x2_eval = np.linspace(x_min[1], x_max[1], n_eval)
    X1_eval, X2_eval = np.meshgrid(x1_eval, x2_eval, indexing='ij')
    
    Z_eval = np.zeros((n_eval, n_eval))
    x_point = (ctypes.c_double * 2)()
    y_out = (ctypes.c_double * 1)()
    
    start_time = time.time()
    for i in range(n_eval):
        for j in range(n_eval):
            x_point[0] = X1_eval[i, j]
            x_point[1] = X2_eval[i, j]
            evaluate_2d(order_c, num_points_c, periodic_c, x_min_c, h_step_c,
                       coeff_c, x_point, y_out)
            Z_eval[i, j] = y_out[0]
    eval_time = time.time() - start_time
    print(f"  Evaluation time ({n_eval}x{n_eval} grid): {eval_time:.4f} seconds")
    
    # Check interpolation error
    errors = []
    for i in range(n1):
        for j in range(n2):
            x_point[0] = x1[i]
            x_point[1] = x2[j]
            evaluate_2d(order_c, num_points_c, periodic_c, x_min_c, h_step_c,
                       coeff_c, x_point, y_out)
            errors.append(abs(y_out[0] - Z[i, j]))
    
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    print(f"  Interpolation error - Max: {max_error:.2e}, Avg: {avg_error:.2e}")
    
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
    
    # Difference
    # Interpolate original to eval grid for comparison
    Z_orig_interp = np.zeros((n_eval, n_eval))
    for i in range(n_eval):
        for j in range(n_eval):
            # Find nearest point in original grid
            i1 = np.argmin(np.abs(x1 - x1_eval[i]))
            i2 = np.argmin(np.abs(x2 - x2_eval[j]))
            Z_orig_interp[i, j] = Z[i1, i2]
    
    diff = np.abs(Z_eval - Z_orig_interp)
    im3 = axes[1, 0].contourf(X2_eval, X1_eval, diff, levels=20, cmap='hot')
    axes[1, 0].set_title('Absolute Difference')
    axes[1, 0].set_xlabel('x2')
    axes[1, 0].set_ylabel('x1')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Cross-section
    idx1 = n_eval // 2
    axes[1, 1].plot(x2_eval, Z_eval[idx1, :], 'b-', label='Spline', linewidth=2)
    idx1_orig = np.argmin(np.abs(x1 - x1_eval[idx1]))
    axes[1, 1].plot(x2, Z[idx1_orig, :], 'ro', label='Data points')
    axes[1, 1].set_title(f'Cross-section at x1 = {x1_eval[idx1]:.2f}')
    axes[1, 1].set_xlabel('x2')
    axes[1, 1].set_ylabel('z')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('sergei_2d_test_final.png')
    plt.show()


def benchmark_performance():
    """Benchmark performance of spline operations"""
    
    print("\nPerformance Benchmark")
    print("=" * 50)
    
    # 1D benchmark
    construct_1d = ctypes.CFUNCTYPE(
        None,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double)
    )(cfunc_addr['construct_splines_1d'])
    
    evaluate_1d = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double)
    )(cfunc_addr['evaluate_splines_1d'])
    
    # Test different sizes
    sizes = [10, 50, 100, 500]
    
    print("\n1D Spline Performance:")
    print(f"{'Size':>6} {'Construct (ms)':>15} {'Eval/point (μs)':>15}")
    print("-" * 40)
    
    for n in sizes:
        x_data = np.linspace(0, 10, n)
        y_data = np.sin(x_data)
        
        y_c = (ctypes.c_double * n)(*y_data)
        coeff_c = (ctypes.c_double * (4 * n))()
        
        # Time construction
        start = time.time()
        for _ in range(100):
            construct_1d(0.0, 10.0, y_c, n, 3, 0, coeff_c)
        construct_time = (time.time() - start) / 100 * 1000  # ms
        
        # Time evaluation
        h_step = 10.0 / (n - 1)
        y_out = (ctypes.c_double * 1)()
        
        start = time.time()
        n_eval = 10000
        for _ in range(n_eval):
            evaluate_1d(3, n, 0, 0.0, h_step, coeff_c, 5.0, y_out)
        eval_time = (time.time() - start) / n_eval * 1e6  # μs
        
        print(f"{n:>6} {construct_time:>15.3f} {eval_time:>15.3f}")


if __name__ == "__main__":
    print("Testing Sergei's Spline Implementation (Final Version)")
    print("=" * 60)
    
    # Test 1D functionality
    test_1d_spline()
    
    # Test 2D functionality
    test_2d_spline()
    
    # Benchmark performance
    benchmark_performance()
    
    print("\n✓ All tests completed!")