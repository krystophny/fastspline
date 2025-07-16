#!/usr/bin/env python3
"""
Test 1D-only Sergei spline implementation
"""

import numpy as np
import ctypes
from sergei_splines_cfunc_temp import get_cfunc_addresses

# Get cfunc addresses
cfunc_addr = get_cfunc_addresses()

# Set up ctypes function signatures
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

def test_1d_cubic():
    """Test 1D cubic spline"""
    print("Testing 1D cubic spline...")
    
    # Create test data
    n_data = 10
    x_min = 0.0
    x_max = 2.0
    x_data = np.linspace(x_min, x_max, n_data)
    y_data = np.sin(x_data)
    
    # Prepare ctypes arrays
    y_c = (ctypes.c_double * n_data)(*y_data)
    coeff_size = 4 * n_data  # cubic spline
    coeff_c = (ctypes.c_double * coeff_size)()
    
    # Construct spline
    construct_1d(
        ctypes.c_double(x_min),
        ctypes.c_double(x_max),
        y_c,
        ctypes.c_int32(n_data),
        ctypes.c_int32(3),
        ctypes.c_int32(0),  # not periodic
        coeff_c
    )
    
    # Evaluate spline at test points
    h_step = (x_max - x_min) / (n_data - 1)
    test_points = [0.5, 1.0, 1.5]
    
    for x in test_points:
        y_out = ctypes.c_double()
        evaluate_1d(
            ctypes.c_int32(3),
            ctypes.c_int32(n_data),
            ctypes.c_int32(0),  # not periodic
            ctypes.c_double(x_min),
            ctypes.c_double(h_step),
            coeff_c,
            ctypes.c_double(x),
            ctypes.byref(y_out)
        )
        y_spline = y_out.value
        y_exact = np.sin(x)
        error = abs(y_spline - y_exact)
        print(f"  x = {x:.1f}: spline = {y_spline:.6f}, exact = {y_exact:.6f}, error = {error:.6f}")
    
    print("  1D cubic spline test completed")

def test_1d_periodic():
    """Test 1D periodic spline"""
    print("\nTesting 1D periodic cubic spline...")
    
    # Create test data
    n_data = 8
    x_min = 0.0
    x_max = 2.0 * np.pi
    x_data = np.linspace(x_min, x_max, n_data + 1)[:-1]  # exclude endpoint for periodic
    y_data = np.sin(x_data)
    
    # Prepare ctypes arrays
    y_c = (ctypes.c_double * n_data)(*y_data)
    coeff_size = 4 * n_data  # cubic spline
    coeff_c = (ctypes.c_double * coeff_size)()
    
    # Construct spline
    construct_1d(
        ctypes.c_double(x_min),
        ctypes.c_double(x_max),
        y_c,
        ctypes.c_int32(n_data),
        ctypes.c_int32(3),
        ctypes.c_int32(1),  # periodic
        coeff_c
    )
    
    # Evaluate spline at test points
    h_step = (x_max - x_min) / n_data
    test_points = [0.5, 1.0, 1.5, 2.0, 5.0, 7.0]  # Include some outside original range
    
    for x in test_points:
        y_out = ctypes.c_double()
        evaluate_1d(
            ctypes.c_int32(3),
            ctypes.c_int32(n_data),
            ctypes.c_int32(1),  # periodic
            ctypes.c_double(x_min),
            ctypes.c_double(h_step),
            coeff_c,
            ctypes.c_double(x),
            ctypes.byref(y_out)
        )
        y_spline = y_out.value
        y_exact = np.sin(x)
        error = abs(y_spline - y_exact)
        print(f"  x = {x:.1f}: spline = {y_spline:.6f}, exact = {y_exact:.6f}, error = {error:.6f}")
    
    print("  1D periodic spline test completed")

def main():
    """Main test function"""
    print("Testing Sergei's 1D Spline CFuncs")
    print("=" * 40)
    
    try:
        test_1d_cubic()
        test_1d_periodic()
        print("\n✓ All 1D tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()