#!/usr/bin/env python3
"""Performance benchmark comparing cfunc vs direct evaluation."""

import numpy as np
import time
from fastspline.spline1d import Spline1D, evaluate_spline_cfunc


def benchmark_evaluation_methods():
    """Compare performance of class methods vs direct cfunc calls."""
    print("Performance Benchmark: Class Methods vs Direct CFuncs")
    print("=" * 60)
    
    # Create test spline
    x_data = np.linspace(0, 2*np.pi, 100)
    y_data = np.sin(x_data) + 0.1 * np.sin(5*x_data)
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    # Create evaluation points
    n_eval = 10000
    x_eval = np.linspace(0, 2*np.pi, n_eval)
    
    print(f"Spline: {len(x_data)} data points, cubic order")
    print(f"Evaluations: {n_eval} points")
    print()
    
    # Benchmark class method
    start_time = time.time()
    for x in x_eval:
        y = spline.evaluate(x)
    class_time = time.time() - start_time
    
    # Benchmark direct cfunc call
    start_time = time.time()
    for x in x_eval:
        y = evaluate_spline_cfunc(
            x, spline.coeffs, spline.x_min, spline.h_step,
            spline.num_points, spline.order, spline.periodic
        )
    cfunc_time = time.time() - start_time
    
    print(f"Class method time:    {class_time:.6f} seconds")
    print(f"Direct cfunc time:    {cfunc_time:.6f} seconds")
    print(f"Speedup factor:       {class_time/cfunc_time:.2f}x")
    print(f"Time per evaluation:")
    print(f"  Class method:       {class_time/n_eval*1e6:.2f} μs")
    print(f"  Direct cfunc:       {cfunc_time/n_eval*1e6:.2f} μs")


def benchmark_c_function_pointer():
    """Demonstrate accessing C function pointer for external integration."""
    print("\nC Function Pointer Access")
    print("=" * 30)
    
    x_data = np.array([0.0, 1.0, 2.0, 3.0])
    y_data = np.array([0.0, 1.0, 4.0, 9.0])
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    # Get C function pointer
    cfunc_ptr = spline.cfunc_evaluate
    
    print(f"C function object: {cfunc_ptr}")
    print(f"Function address:  {cfunc_ptr.address}")
    print(f"Function type:     {type(cfunc_ptr)}")
    
    # Test evaluation using function pointer
    x_test = 1.5
    result = cfunc_ptr(
        x_test, spline.coeffs, spline.x_min, spline.h_step,
        spline.num_points, spline.order, spline.periodic
    )
    print(f"Evaluation at x={x_test}: {result}")


def benchmark_vectorized_evaluation():
    """Compare vectorized vs loop-based evaluation."""
    print("\nVectorized vs Loop Evaluation")
    print("=" * 35)
    
    x_data = np.linspace(0, 4*np.pi, 50)
    y_data = np.sin(x_data) * np.exp(-x_data/10)
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    n_eval = 5000
    x_eval = np.linspace(0, 4*np.pi, n_eval)
    
    # Loop-based evaluation
    start_time = time.time()
    y_loop = np.zeros(n_eval)
    for i, x in enumerate(x_eval):
        y_loop[i] = spline.evaluate(x)
    loop_time = time.time() - start_time
    
    # Vectorized evaluation using list comprehension
    start_time = time.time()
    y_vectorized = np.array([spline.evaluate(x) for x in x_eval])
    vectorized_time = time.time() - start_time
    
    print(f"Loop-based time:      {loop_time:.6f} seconds")
    print(f"Vectorized time:      {vectorized_time:.6f} seconds")
    print(f"Results identical:    {np.allclose(y_loop, y_vectorized)}")
    
    # Check accuracy
    y_exact = np.sin(x_eval) * np.exp(-x_eval/10)
    mse_spline = np.mean((y_vectorized - y_exact)**2)
    print(f"Mean squared error:   {mse_spline:.2e}")


if __name__ == "__main__":
    benchmark_evaluation_methods()
    benchmark_c_function_pointer()
    benchmark_vectorized_evaluation()
    
    print("\nBenchmark Complete!")
    print("The cfunc implementation provides C-compatible function pointers")
    print("that can be called directly from other compiled code while")
    print("maintaining the same performance characteristics.")