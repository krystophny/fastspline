#!/usr/bin/env python3
"""
Simple test to understand quintic issue
"""

import numpy as np
import subprocess

def test_fortran_quintic_directly():
    """Test Fortran quintic with minimal example"""
    fortran_code = """
program test_quintic_simple
    use spl_three_to_five_sub
    implicit none
    
    integer, parameter :: dp = kind(1.0d0)
    integer :: n, i
    real(dp) :: h, x_test, y_test
    real(dp), allocatable :: a(:), b(:), c(:), d(:), e(:), f(:)
    
    ! Minimal test
    n = 10
    h = 1.0d0 / (n - 1)
    
    allocate(a(n), b(n), c(n), d(n), e(n), f(n))
    
    ! x^4 values
    do i = 1, n
        a(i) = ((i-1) * h)**4
    end do
    
    write(*,*) 'Testing Fortran quintic on x^4, n=10'
    
    ! Call quintic
    call spl_five_reg(n, h, a, b, c, d, e, f)
    
    ! Evaluate at x=0.5
    x_test = 0.5d0
    i = int(x_test / h) + 1  ! Find interval
    if (i >= n) i = n - 1
    
    ! Local coordinate
    x_test = x_test - (i-1)*h
    
    ! Evaluate polynomial
    y_test = a(i) + x_test*(b(i) + x_test*(c(i) + x_test*(d(i) + x_test*(e(i) + x_test*f(i)))))
    
    write(*,*) 'x=0.5: y=', y_test, ' exact=', 0.5d0**4, ' error=', abs(y_test - 0.5d0**4)
    
    ! Check if e coefficients are constant
    write(*,*) 'e coefficients (should be ~1.0):'
    do i = 1, n
        write(*,'(A,I2,A,F20.15)') '  e(', i, ') = ', e(i)
    end do
    
    deallocate(a, b, c, d, e, f)
    
end program test_quintic_simple
"""
    
    with open('test_quintic_simple.f90', 'w') as f:
        f.write(fortran_code)
    
    try:
        subprocess.run(['gfortran', '-o', 'test_quintic_simple',
                       'test_quintic_simple.f90', 'src/spl_three_to_five.f90'],
                      check=True, capture_output=True)
        
        result = subprocess.run(['./test_quintic_simple'],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("FORTRAN QUINTIC TEST")
            print("=" * 50)
            print(result.stdout)
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Failed: {e}")
        return False

def test_python_quintic_directly():
    """Test Python quintic implementation directly"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from fastspline.sergei_splines import construct_splines_1d_cfunc, evaluate_splines_1d_cfunc
    
    print("\nPYTHON QUINTIC TEST")
    print("=" * 50)
    
    n = 10
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4
    
    print(f"Testing Python quintic on x^4, n=10")
    
    # Construct
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    # Evaluate at x=0.5
    y_out = np.zeros(1)
    evaluate_splines_1d_cfunc(5, n, False, 0.0, h, coeff, 0.5, y_out)
    
    print(f"x=0.5: y={y_out[0]:.15f} exact={0.5**4:.15f} error={abs(y_out[0] - 0.5**4):.2e}")
    
    # Check e coefficients
    print("e coefficients (should be ~1.0):")
    e_coeff = coeff[4*n:5*n]
    for i in range(n):
        print(f"  e[{i}] = {e_coeff[i]:.15f}")
    
    return abs(y_out[0] - 0.5**4)

if __name__ == "__main__":
    fortran_ok = test_fortran_quintic_directly()
    python_error = test_python_quintic_directly()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if fortran_ok and python_error > 1e-14:
        print("Fortran works perfectly, Python has errors")
        print("This confirms the Python implementation needs fixing")
    else:
        print("Need to investigate further")