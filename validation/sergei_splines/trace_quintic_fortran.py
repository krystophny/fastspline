#!/usr/bin/env python3
"""
Trace quintic implementation and compare with Fortran output
"""

import numpy as np
import subprocess
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

def write_fortran_debug_program():
    """Create a Fortran program that outputs intermediate values"""
    fortran_code = """
program debug_quintic
    use spl_three_to_five_sub
    implicit none
    
    integer, parameter :: dp = kind(1.0d0)
    integer :: n, i
    real(dp) :: h
    real(dp), allocatable :: a(:), b(:), c(:), d(:), e(:), f(:)
    
    ! Test case: x^4 on 10 points
    n = 10
    h = 1.0d0 / (n - 1)
    
    allocate(a(n), b(n), c(n), d(n), e(n), f(n))
    
    ! Initialize with x^4 values
    do i = 1, n
        a(i) = ((i-1) * h)**4
    end do
    
    write(*,*) 'Fortran quintic debug for x^4:'
    write(*,*) 'n =', n
    write(*,*) 'h =', h
    write(*,*) 'Input values:'
    do i = 1, n
        write(*,'(A,I2,A,F20.15)') 'a(', i, ') = ', a(i)
    end do
    
    ! Call quintic spline
    call spl_five_reg(n, h, a, b, c, d, e, f)
    
    write(*,*) ''
    write(*,*) 'Output coefficients:'
    do i = 1, n
        write(*,'(A,I2,A,6F15.10)') 'i=', i, ': ', a(i), b(i), c(i), d(i), e(i), f(i)
    end do
    
    ! Evaluate at x=0.5
    write(*,*) ''
    write(*,*) 'Evaluation at x=0.5:'
    i = 5  ! interval containing x=0.5
    write(*,*) 'Using interval i=', i
    write(*,*) 'Coefficients for interval:'
    write(*,'(A,6F15.10)') 'a,b,c,d,e,f = ', a(i), b(i), c(i), d(i), e(i), f(i)
    
    deallocate(a, b, c, d, e, f)
    
end program debug_quintic
"""
    
    with open('debug_quintic_fortran.f90', 'w') as f:
        f.write(fortran_code)
    
    # Compile
    try:
        subprocess.run(['gfortran', '-o', 'debug_quintic_fortran', 
                       'debug_quintic_fortran.f90', 'src/spl_three_to_five.f90'],
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e.stderr.decode()}")
        return False

def run_fortran_debug():
    """Run Fortran debug program"""
    if not os.path.exists('debug_quintic_fortran'):
        if not write_fortran_debug_program():
            return None
    
    try:
        result = subprocess.run(['./debug_quintic_fortran'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Fortran execution failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running Fortran: {e}")
        return None

def trace_python_quintic():
    """Trace Python implementation with same test case"""
    print("PYTHON QUINTIC TRACE")
    print("=" * 50)
    
    # Same test case as Fortran
    n = 10
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4
    
    print(f"Test case: x^4 on {n} points")
    print(f"h = {h:.15e}")
    print("Input values:")
    for i in range(n):
        print(f"y[{i}] = {y[i]:.15e}")
    
    # Run construction
    from fastspline.sergei_splines import construct_splines_1d_cfunc
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    print("\nOutput coefficients:")
    for i in range(n):
        print(f"i={i}: a={coeff[i]:.10f} b={coeff[n+i]:.10f} c={coeff[2*n+i]:.10f} " +
              f"d={coeff[3*n+i]:.10f} e={coeff[4*n+i]:.10f} f={coeff[5*n+i]:.10f}")
    
    # Evaluate at x=0.5
    from fastspline.sergei_splines import evaluate_splines_1d_cfunc
    y_out = np.zeros(1)
    evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, 0.5, y_out)
    
    print(f"\nEvaluation at x=0.5:")
    print(f"Result: {y_out[0]:.15f}")
    print(f"Exact:  {0.5**4:.15f}")
    print(f"Error:  {abs(y_out[0] - 0.5**4):.2e}")
    
    return coeff

def compare_coefficients():
    """Compare Python and Fortran coefficients"""
    print("\n" + "=" * 50)
    print("FORTRAN vs PYTHON COMPARISON")
    print("=" * 50)
    
    fortran_output = run_fortran_debug()
    if fortran_output:
        print("\nFortran output:")
        print(fortran_output)
    
    python_coeff = trace_python_quintic()
    
    # Manual comparison if we can parse Fortran output
    if fortran_output:
        print("\nDetailed coefficient comparison needed...")
        print("Check if boundary handling matches exactly")

def test_simpler_cases():
    """Test on simpler polynomials that should work perfectly"""
    print("\n" + "=" * 50)
    print("TESTING SIMPLER POLYNOMIALS")
    print("=" * 50)
    
    from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc
    
    # Test x^3 (should be exact for quintic)
    print("\nTest 1: x^3 polynomial")
    n = 8
    x = np.linspace(0, 1, n)
    y = x**3
    h = 1.0 / (n - 1)
    
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    max_error = 0.0
    y_out = np.zeros(1)
    for x_test in [0.25, 0.5, 0.75]:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        error = abs(y_out[0] - x_test**3)
        max_error = max(max_error, error)
        print(f"  x={x_test}: error = {error:.2e}")
    
    print(f"  Max error for x^3: {max_error:.2e}")
    
    # Test x^1 (should be exact for any spline)
    print("\nTest 2: x^1 polynomial (linear)")
    y = x
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    max_error = 0.0
    for x_test in [0.25, 0.5, 0.75]:
        evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, x_test, y_out)
        error = abs(y_out[0] - x_test)
        max_error = max(max_error, error)
        print(f"  x={x_test}: error = {error:.2e}")
    
    print(f"  Max error for x^1: {max_error:.2e}")

if __name__ == "__main__":
    compare_coefficients()
    test_simpler_cases()