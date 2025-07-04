#!/usr/bin/env python3
"""Visual test of 1D spline interpolation using fortplotlib."""

import numpy as np
import fortplotlib.fortplot as plt
from fastspline import Spline1D


def test_linear_spline_visual():
    """Visual test of linear spline interpolation."""
    # Create test data
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([0.0, 1.0, 0.5, 2.0, 1.5])
    
    # Create spline
    spline = Spline1D(x_data, y_data, order=1, periodic=False)
    
    # Generate fine grid for plotting
    x_fine = np.linspace(0, 4, 100)
    y_fine = np.array([spline.evaluate(x) for x in x_fine])
    
    # Create plot
    plt.figure()
    
    # Plot original data points
    plt.plot(x_data, y_data, 'o', label='Data points')
    
    # Plot spline interpolation
    plt.plot(x_fine, y_fine, '-', label='Linear spline')
    
    plt.xlabel('x')
    plt.ylabel('y')  
    plt.title('Linear Spline Interpolation')
    plt.legend()
    
    plt.savefig('linear_spline_test.png')
    print("Linear spline plot saved as 'linear_spline_test.png'")


def test_cubic_spline_visual():
    """Visual test of cubic spline interpolation."""
    # Create test data with a smooth function
    x_data = np.linspace(0, 2*np.pi, 10)
    y_data = np.sin(x_data) + 0.1 * np.sin(5*x_data)  # Sine with high frequency component
    
    # Create splines
    linear_spline = Spline1D(x_data, y_data, order=1, periodic=False)
    cubic_spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    # Generate fine grid for plotting
    x_fine = np.linspace(0, 2*np.pi, 200)
    y_linear = np.array([linear_spline.evaluate(x) for x in x_fine])
    y_cubic = np.array([cubic_spline.evaluate(x) for x in x_fine])
    y_exact = np.sin(x_fine) + 0.1 * np.sin(5*x_fine)
    
    # Create plot
    plt.figure()
    
    # Plot original data points
    plt.plot(x_data, y_data, 'o', label='Data points')
    
    # Plot exact function
    plt.plot(x_fine, y_exact, ':', label='Exact function')
    
    # Plot spline interpolations
    plt.plot(x_fine, y_linear, '--', label='Linear spline')
    plt.plot(x_fine, y_cubic, '-', label='Cubic spline')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear vs Cubic Spline Interpolation')
    plt.legend()
    
    plt.savefig('cubic_spline_comparison.png')
    print("Cubic spline comparison plot saved as 'cubic_spline_comparison.png'")


def test_periodic_spline_visual():
    """Visual test of periodic spline interpolation."""
    # Create periodic test data
    x_data = np.linspace(0, 2*np.pi, 8)
    y_data = np.sin(x_data)
    
    # Create splines
    regular_spline = Spline1D(x_data, y_data, order=3, periodic=False)
    periodic_spline = Spline1D(x_data, y_data, order=3, periodic=True)
    
    # Generate fine grid for plotting including extension beyond period
    x_fine = np.linspace(-np.pi, 3*np.pi, 300)
    y_regular = np.array([regular_spline.evaluate(x) for x in x_fine])
    y_periodic = np.array([periodic_spline.evaluate(x) for x in x_fine])
    y_exact = np.sin(x_fine)
    
    # Create plot
    plt.figure()
    
    # Plot original data points
    plt.plot(x_data, y_data, 'o', label='Data points')
    
    # Plot exact function
    plt.plot(x_fine, y_exact, ':', label='sin(x)')
    
    # Plot spline interpolations
    plt.plot(x_fine, y_regular, '--', label='Regular cubic spline')
    plt.plot(x_fine, y_periodic, '-', label='Periodic cubic spline')
    
    # Mark the period boundaries (vertical lines not supported in fortplotlib)
    # plt.axvline(x=0, color='k', linestyle=':')
    # plt.axvline(x=2*np.pi, color='k', linestyle=':')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regular vs Periodic Cubic Spline')
    plt.legend()
    
    plt.savefig('periodic_spline_test.png')
    print("Periodic spline plot saved as 'periodic_spline_test.png'")


def test_derivative_visual():
    """Visual test of spline derivatives."""
    # Create test data with known derivative
    x_data = np.linspace(0, 3, 12)
    y_data = x_data**3 - 2*x_data**2 + x_data + 1
    
    # Create spline
    spline = Spline1D(x_data, y_data, order=3, periodic=False)
    
    # Generate fine grid for plotting
    x_fine = np.linspace(0, 3, 100)
    y_fine = np.array([spline.evaluate(x) for x in x_fine])
    
    # Calculate derivatives
    derivatives = np.array([spline.evaluate_with_derivative(x) for x in x_fine])
    dy_fine = derivatives[:, 1]
    
    # Exact derivatives
    y_exact = x_fine**3 - 2*x_fine**2 + x_fine + 1
    dy_exact = 3*x_fine**2 - 4*x_fine + 1
    
    # Function plot
    plt.figure()
    plt.plot(x_data, y_data, 'o', label='Data points')
    plt.plot(x_fine, y_exact, ':', label='Exact f(x)')
    plt.plot(x_fine, y_fine, '-', label='Spline f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Comparison')
    plt.legend()
    plt.savefig('function_test.png')
    print("Function test plot saved as 'function_test.png'")
    
    # Derivative plot
    plt.figure()
    plt.plot(x_fine, dy_exact, ':', label="Exact f'(x)")
    plt.plot(x_fine, dy_fine, '-', label="Spline f'(x)")
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Derivative Comparison')
    plt.legend()
    plt.savefig('derivative_test.png')
    print("Derivative test plot saved as 'derivative_test.png'")


def test_performance_visual():
    """Visual test showing performance with different numbers of points."""
    import time
    
    # Test different numbers of points
    n_points = [10, 50, 100, 500, 1000]
    times_construct = []
    times_evaluate = []
    
    for n in n_points:
        x_data = np.linspace(0, 2*np.pi, n)
        y_data = np.sin(x_data)
        
        # Time construction
        start = time.time()
        spline = Spline1D(x_data, y_data, order=3, periodic=False)
        times_construct.append(time.time() - start)
        
        # Time evaluation
        x_test = np.linspace(0, 2*np.pi, 1000)
        start = time.time()
        for x in x_test:
            spline.evaluate(x)
        times_evaluate.append(time.time() - start)
    
    # Create performance plot (linear scale since loglog not available)
    plt.figure()
    
    plt.plot(n_points, times_construct, 'o-', label='Construction time')
    plt.plot(n_points, times_evaluate, 's-', label='Evaluation time (1000 points)')
    
    plt.xlabel('Number of data points')
    plt.ylabel('Time (seconds)')
    plt.title('Spline Performance Test')
    plt.legend()
    
    plt.savefig('performance_test.png')
    print("Performance test plot saved as 'performance_test.png'")
    
    # Print timing results
    print("\nPerformance Results:")
    print("Points\tConstruct(s)\tEvaluate(s)")
    for i, n in enumerate(n_points):
        print(f"{n}\t{times_construct[i]:.6f}\t{times_evaluate[i]:.6f}")


if __name__ == "__main__":
    print("Running visual tests for 1D spline interpolation...")
    
    test_linear_spline_visual()
    test_cubic_spline_visual()
    test_periodic_spline_visual()
    test_derivative_visual()
    test_performance_visual()
    
    print("\nAll visual tests completed. Check the generated PNG files.")