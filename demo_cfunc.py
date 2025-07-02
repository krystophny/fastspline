#!/usr/bin/env python3
"""Demonstration of C-compatible function interface."""

import numpy as np
from fastspline.spline1d import Spline1D, evaluate_spline_cfunc


def demo_basic_usage():
    """Basic usage demonstration."""
    print("FastSpline C Function Demo")
    print("=" * 30)
    
    # Create sample data
    x_data = np.linspace(0, 2*np.pi, 10)
    y_data = np.sin(x_data)
    
    # Create spline (same as before)
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    # Evaluate using class method (recommended for Python usage)
    x_test = 1.5
    result_class = spline.evaluate(x_test)
    
    # Evaluate using direct cfunc call (for C interoperability)
    result_cfunc = evaluate_spline_cfunc(
        x_test, spline.coeffs, spline.x_min, spline.h_step,
        spline.num_points, spline.order, spline.periodic
    )
    
    print(f"Test point: x = {x_test}")
    print(f"Class method result:  {result_class:.6f}")
    print(f"CFfunc result:        {result_cfunc:.6f}")
    print(f"Exact sin(1.5):       {np.sin(x_test):.6f}")
    print(f"Results identical:    {abs(result_class - result_cfunc) < 1e-15}")
    

def demo_c_function_pointers():
    """Demonstrate C function pointer access."""
    print("\nC Function Pointer Access")
    print("=" * 30)
    
    x_data = np.array([0.0, 1.0, 2.0, 3.0])
    y_data = x_data**2  # f(x) = x^2
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    # Get C function pointers
    eval_func = spline.cfunc_evaluate
    eval_deriv_func = spline.cfunc_evaluate_derivative
    eval_second_deriv_func = spline.cfunc_evaluate_second_derivative
    
    print("Available C function pointers:")
    print(f"  evaluate:           {eval_func}")
    print(f"  evaluate_derivative: {eval_deriv_func}")
    print(f"  evaluate_2nd_deriv:  {eval_second_deriv_func}")
    print()
    
    # Function addresses (for calling from C/C++/Fortran)
    print("Function addresses for external calling:")
    print(f"  evaluate:           0x{eval_func.address:x}")
    print(f"  evaluate_derivative: 0x{eval_deriv_func.address:x}")
    print(f"  evaluate_2nd_deriv:  0x{eval_second_deriv_func.address:x}")
    

def demo_data_structure():
    """Show the data structure layout for C interoperability."""
    print("\nData Structure Layout")
    print("=" * 25)
    
    x_data = np.linspace(0, 3, 8)
    y_data = x_data**3 - 2*x_data**2 + x_data + 1
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    print("Spline parameters (pass these to C functions):")
    print(f"  coeffs shape:    {spline.coeffs.shape}")
    print(f"  coeffs dtype:    {spline.coeffs.dtype}")
    print(f"  x_min:           {spline.x_min}")
    print(f"  h_step:          {spline.h_step}")
    print(f"  num_points:      {spline.num_points}")
    print(f"  order:           {spline.order}")
    print(f"  periodic:        {spline.periodic}")
    print()
    
    print("Coefficient matrix (first few columns):")
    print(f"  Shape: ({spline.coeffs.shape[0]} x {spline.coeffs.shape[1]})")
    print("  Rows: [constant, linear, quadratic, cubic] coefficients")
    print(f"  Data:\n{spline.coeffs[:, :4]}")


def demo_external_integration():
    """Example of how to integrate with external C/C++/Fortran code."""
    print("\nExternal Integration Example")
    print("=" * 35)
    
    spline = Spline1D(np.array([0., 1., 2.]), np.array([0., 1., 4.]), order=1)
    
    print("To call from C/C++:")
    print("```c")
    print("// Function signature:")
    print("double evaluate_spline_cfunc(")
    print("    double x,")
    print("    double* coeffs,      // 2D array, shape (order+1, num_points)")
    print("    double x_min,")
    print("    double h_step,")
    print("    int64_t num_points,")
    print("    int64_t order,")
    print("    bool periodic")
    print(");")
    print()
    print("// Function pointer address:")
    print(f"void* func_ptr = (void*)0x{spline.cfunc_evaluate.address:x};")
    print("```")
    print()
    
    print("To call from Fortran:")
    print("```fortran")
    print("interface")
    print("  real(c_double) function evaluate_spline_cfunc(x, coeffs, x_min, &")
    print("                                                h_step, num_points, order, periodic) &")
    print("                                                bind(c)")
    print("    import :: c_double, c_int64_t, c_bool")
    print("    real(c_double), value :: x, x_min, h_step")
    print("    real(c_double) :: coeffs(*)")
    print("    integer(c_int64_t), value :: num_points, order")
    print("    logical(c_bool), value :: periodic")
    print("  end function")
    print("end interface")
    print("```")


if __name__ == "__main__":
    demo_basic_usage()
    demo_c_function_pointers()
    demo_data_structure()
    demo_external_integration()
    
    print("\nKey Benefits of CFuncs:")
    print("- C-compatible function pointers")
    print("- No Python GIL dependency")
    print("- Direct callable from C/C++/Fortran")
    print("- Same performance as njit")
    print("- Clean Python API maintained through class wrapper")