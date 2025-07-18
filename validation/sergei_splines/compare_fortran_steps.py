#!/usr/bin/env python3
"""
Compare Python implementation with expected Fortran behavior
"""

import numpy as np
import subprocess

def write_fortran_tracer():
    """Write a Fortran program that traces intermediate values"""
    fortran_code = """
program trace_quintic
    use spl_three_to_five_sub
    implicit none
    
    integer, parameter :: dp = kind(1.0d0)
    integer :: n, i
    real(dp) :: h
    real(dp), allocatable :: a(:), b(:), c(:), d(:), e(:), f(:)
    
    ! Small test case
    n = 6
    h = 1.0d0 / (n - 1)
    
    allocate(a(n), b(n), c(n), d(n), e(n), f(n))
    
    ! x^4 values
    do i = 1, n
        a(i) = ((i-1) * h)**4
    end do
    
    write(*,*) 'Fortran trace for n=6, x^4:'
    write(*,*) 'Input a:', (a(i), i=1,n)
    
    call spl_five_reg(n, h, a, b, c, d, e, f)
    
    write(*,*) 'Output e:', (e(i), i=1,n)
    write(*,*) 'e should be constant for x^4'
    write(*,*) 'e range:', minval(e), 'to', maxval(e)
    
    deallocate(a, b, c, d, e, f)
    
end program trace_quintic
"""
    
    with open('trace_fortran_quintic.f90', 'w') as f:
        f.write(fortran_code)
    
    try:
        # Compile
        subprocess.run(['gfortran', '-o', 'trace_fortran_quintic',
                       'trace_fortran_quintic.f90', 'src/spl_three_to_five.f90'],
                      check=True, capture_output=True)
        
        # Run
        result = subprocess.run(['./trace_fortran_quintic'],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
            
    except Exception as e:
        return f"Failed to run Fortran: {str(e)}"

def trace_python_implementation():
    """Trace Python implementation for same test"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    from fastspline.sergei_splines import construct_splines_1d_cfunc
    
    print("Python trace for n=6, x^4:")
    n = 6
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4
    
    print(f"Input y: {y}")
    
    coeff = np.zeros(6*n, dtype=np.float64)
    construct_splines_1d_cfunc(0.0, 1.0, y, n, 5, False, coeff)
    
    e_coeff = coeff[4*n:5*n]
    print(f"Output e: {e_coeff}")
    print(f"e should be constant for x^4")
    print(f"e range: {np.min(e_coeff):.6f} to {np.max(e_coeff):.6f}")
    
    return e_coeff

def main():
    """Compare Fortran and Python"""
    print("FORTRAN vs PYTHON COMPARISON FOR QUINTIC")
    print("=" * 50)
    
    # Run Fortran
    fortran_output = write_fortran_tracer()
    print("Fortran output:")
    print(fortran_output)
    
    print("\n" + "-" * 50 + "\n")
    
    # Run Python
    python_e = trace_python_implementation()
    
    print("\n" + "=" * 50)
    print("ANALYSIS:")
    
    # Check if Fortran shows constant e
    if "1.000000" in fortran_output and "1.000000" in fortran_output:
        print("✓ Fortran produces constant e coefficients (as expected)")
        print("✗ Python produces varying e coefficients (BUG)")
        print("\nThis confirms the Python implementation has an error")
        print("in the calculation of e coefficients.")
    else:
        print("? Both implementations may have issues")

if __name__ == "__main__":
    main()