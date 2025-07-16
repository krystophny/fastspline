#!/usr/bin/env python3
"""
Test script for Sergei's spline cfunc implementation
Visual validation with plots
"""

import numpy as np
import matplotlib.pyplot as plt
import ctypes
from sergei_splines_cfunc import get_cfunc_addresses

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()

# Set up ctypes function signatures
def setup_ctypes_signatures():
    """Setup ctypes function signatures for all cfuncs"""
    
    # construct_splines_1d_cfunc
    construct_1d = ctypes.CFUNCTYPE(
        None,  # return type
        ctypes.c_double,  # x_min
        ctypes.c_double,  # x_max
        ctypes.POINTER(ctypes.c_double),  # y values
        ctypes.c_int32,  # num_points
        ctypes.c_int32,  # order
        ctypes.c_int32,  # periodic
        ctypes.POINTER(ctypes.c_double)  # coeff array
    )(cfunc_addr['construct_splines_1d'])
    
    # evaluate_splines_1d_cfunc
    evaluate_1d = ctypes.CFUNCTYPE(
        None,  # return type
        ctypes.c_int32,  # order
        ctypes.c_int32,  # num_points
        ctypes.c_int32,  # periodic
        ctypes.c_double,  # x_min
        ctypes.c_double,  # h_step
        ctypes.POINTER(ctypes.c_double),  # coeff array
        ctypes.c_double,  # x
        ctypes.POINTER(ctypes.c_double)  # output y
    )(cfunc_addr['evaluate_splines_1d'])
    
    # construct_splines_2d_cfunc
    construct_2d = ctypes.CFUNCTYPE(
        None,  # return type
        ctypes.POINTER(ctypes.c_double),  # x_min array (2)
        ctypes.POINTER(ctypes.c_double),  # x_max array (2)
        ctypes.POINTER(ctypes.c_double),  # y values
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
    
    return construct_1d, evaluate_1d, construct_2d, evaluate_2d


def test_1d_spline(order=3, periodic=False):
    """Test 1D spline with visual validation"""
    
    construct_1d, evaluate_1d, _, _ = setup_ctypes_signatures()
    
    # Create test data
    n_data = 20
    x_min = 0.0
    x_max = 2.0 * np.pi if periodic else 4.0
    
    # Test function
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
    construct_1d(
        ctypes.c_double(x_min),
        ctypes.c_double(x_max),
        y_c,
        ctypes.c_int32(n_data),
        ctypes.c_int32(order),
        ctypes.c_int32(1 if periodic else 0),
        coeff_c
    )
    
    # Evaluate spline on fine grid
    n_eval = 200
    x_eval = np.linspace(x_min, x_max, n_eval)
    y_eval = np.zeros(n_eval)
    
    h_step = (x_max - x_min) / (n_data - 1)
    
    for i, x in enumerate(x_eval):
        y_out = ctypes.c_double()
        evaluate_1d(
            ctypes.c_int32(order),
            ctypes.c_int32(n_data),
            ctypes.c_int32(1 if periodic else 0),
            ctypes.c_double(x_min),
            ctypes.c_double(h_step),
            coeff_c,
            ctypes.c_double(x),
            ctypes.byref(y_out)
        )
        y_eval[i] = y_out.value
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ro', markersize=8, label='Data points')
    plt.plot(x_eval, y_eval, 'b-', linewidth=2, label=f'Spline (order {order})')
    
    # Plot exact function for comparison
    if periodic:
        y_exact = np.sin(x_eval) + 0.5 * np.sin(3.0 * x_eval)
    else:
        y_exact = np.exp(-0.5 * x_eval) * np.sin(2.0 * x_eval)
    plt.plot(x_eval, y_exact, 'g--', alpha=0.7, label='Exact function')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'1D Spline Test (Order {order}, {"Periodic" if periodic else "Regular"})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Compute error
    error = np.abs(y_eval - y_exact)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    print(f"1D Spline (Order {order}, {'Periodic' if periodic else 'Regular'}):")
    print(f"  Max error: {max_error:.2e}")
    print(f"  RMS error: {rms_error:.2e}")
    
    return max_error, rms_error


def test_2d_spline(order=(3, 3), periodic=(False, False)):
    """Test 2D spline with visual validation"""
    
    _, _, construct_2d, evaluate_2d = setup_ctypes_signatures()
    
    # Create test data
    n_data = [15, 12]
    x_min = np.array([0.0, 0.0])
    x_max = np.array([2.0 * np.pi if periodic[0] else 3.0, 
                      2.0 * np.pi if periodic[1] else 2.0])
    
    # Test function
    x1_data = np.linspace(x_min[0], x_max[0], n_data[0])
    x2_data = np.linspace(x_min[1], x_max[1], n_data[1])
    X1, X2 = np.meshgrid(x1_data, x2_data, indexing='ij')
    
    if periodic[0] and periodic[1]:
        Z_data = np.sin(X1) * np.cos(X2) + 0.3 * np.sin(2.0 * X1) * np.sin(3.0 * X2)
    else:
        Z_data = np.exp(-0.2 * (X1 + X2)) * np.sin(X1) * np.cos(X2)
    
    # Prepare ctypes arrays
    z_flat = Z_data.flatten()
    z_c = (ctypes.c_double * len(z_flat))(*z_flat)
    
    x_min_c = (ctypes.c_double * 2)(*x_min)
    x_max_c = (ctypes.c_double * 2)(*x_max)
    num_points_c = (ctypes.c_int32 * 2)(*n_data)
    order_c = (ctypes.c_int32 * 2)(*order)
    periodic_c = (ctypes.c_int32 * 2)(*(1 if p else 0 for p in periodic))
    
    coeff_size = (order[0] + 1) * (order[1] + 1) * n_data[0] * n_data[1]
    coeff_c = (ctypes.c_double * coeff_size)()
    
    # Construct spline
    construct_2d(
        x_min_c,
        x_max_c,
        z_c,
        num_points_c,
        order_c,
        periodic_c,
        coeff_c
    )
    
    # Evaluate spline on fine grid
    n_eval = [50, 40]
    x1_eval = np.linspace(x_min[0], x_max[0], n_eval[0])
    x2_eval = np.linspace(x_min[1], x_max[1], n_eval[1])
    X1_eval, X2_eval = np.meshgrid(x1_eval, x2_eval, indexing='ij')
    Z_eval = np.zeros_like(X1_eval)
    
    h_step = np.array([(x_max[0] - x_min[0]) / (n_data[0] - 1),
                       (x_max[1] - x_min[1]) / (n_data[1] - 1)])
    h_step_c = (ctypes.c_double * 2)(*h_step)
    
    for i in range(n_eval[0]):
        for j in range(n_eval[1]):
            x_point = np.array([X1_eval[i, j], X2_eval[i, j]])
            x_c = (ctypes.c_double * 2)(*x_point)
            y_out = ctypes.c_double()
            
            evaluate_2d(
                order_c,
                num_points_c,
                periodic_c,
                x_min_c,
                h_step_c,
                coeff_c,
                x_c,
                ctypes.byref(y_out)
            )
            Z_eval[i, j] = y_out.value
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original data
    im1 = axes[0].contourf(X1, X2, Z_data, levels=20, cmap='viridis')
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot spline interpolation
    im2 = axes[1].contourf(X1_eval, X2_eval, Z_eval, levels=20, cmap='viridis')
    axes[1].set_title(f'Spline Interpolation (Order {order})')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot exact function for comparison
    if periodic[0] and periodic[1]:
        Z_exact = np.sin(X1_eval) * np.cos(X2_eval) + 0.3 * np.sin(2.0 * X1_eval) * np.sin(3.0 * X2_eval)
    else:
        Z_exact = np.exp(-0.2 * (X1_eval + X2_eval)) * np.sin(X1_eval) * np.cos(X2_eval)
    
    error = np.abs(Z_eval - Z_exact)
    im3 = axes[2].contourf(X1_eval, X2_eval, error, levels=20, cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # Compute error statistics
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    print(f"2D Spline (Order {order}, Periodic {periodic}):")
    print(f"  Max error: {max_error:.2e}")
    print(f"  RMS error: {rms_error:.2e}")
    
    return max_error, rms_error


def main():
    """Main test function"""
    
    print("Testing Sergei's Spline CFuncs")
    print("=" * 40)
    
    # Test 1D splines
    print("\n1D Spline Tests:")
    print("-" * 20)
    
    # Test different orders and boundary conditions
    test_cases_1d = [
        (3, False),  # Cubic regular
        (3, True),   # Cubic periodic
        (4, False),  # Quartic regular
        (4, True),   # Quartic periodic
        (5, False),  # Quintic regular
        (5, True),   # Quintic periodic
    ]
    
    for order, periodic in test_cases_1d:
        try:
            max_err, rms_err = test_1d_spline(order, periodic)
            plt.show()
        except Exception as e:
            print(f"Error in 1D test (order={order}, periodic={periodic}): {e}")
    
    # Test 2D splines
    print("\n2D Spline Tests:")
    print("-" * 20)
    
    # Test different orders and boundary conditions
    test_cases_2d = [
        ((3, 3), (False, False)),  # Cubic regular
        ((3, 3), (True, True)),    # Cubic periodic
        ((4, 4), (False, False)),  # Quartic regular
        ((5, 5), (False, False)),  # Quintic regular
    ]
    
    for order, periodic in test_cases_2d:
        try:
            max_err, rms_err = test_2d_spline(order, periodic)
            plt.show()
        except Exception as e:
            print(f"Error in 2D test (order={order}, periodic={periodic}): {e}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()