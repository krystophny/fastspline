#!/usr/bin/env python3
"""
Trace Fortran loop indices to understand the mapping
"""

import subprocess

def write_fortran_loop_tracer():
    """Write Fortran program to trace loop indices"""
    fortran_code = """
program trace_loops
    implicit none
    integer :: i, n
    
    n = 6
    write(*,*) 'Fortran loop traces for n=', n
    write(*,*) ''
    
    write(*,*) 'Loop: do i=1,n-4'
    write(*,*) 'n-4 =', n-4
    do i = 1, n-4
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Loop: do i=n-3,1,-1'
    write(*,*) 'n-3 =', n-3
    do i = n-3, 1, -1
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Loop: do i=1,n-2'
    write(*,*) 'n-2 =', n-2
    do i = 1, n-2
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Loop: do i=n-3,n'
    write(*,*) 'n-3 =', n-3, ', n =', n
    do i = n-3, n
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Array access in loops:'
    write(*,*) 'When i=1, a(i) accesses element 1'
    write(*,*) 'When i=n, a(i) accesses element n'
    
end program trace_loops
"""
    
    with open('trace_loops.f90', 'w') as f:
        f.write(fortran_code)
    
    try:
        subprocess.run(['gfortran', '-o', 'trace_loops', 'trace_loops.f90'],
                      check=True, capture_output=True)
        result = subprocess.run(['./trace_loops'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"

def show_python_equivalent():
    """Show Python equivalent loops"""
    print("PYTHON EQUIVALENT LOOPS")
    print("=" * 50)
    
    n = 6
    print(f"For n = {n}:")
    print()
    
    print("Fortran: do i=1,n-4")
    print(f"Python: for i in range(0, {n-4}):")
    for i in range(0, n-4):
        print(f"  i = {i} (accesses array[{i}])")
    
    print()
    print("Fortran: do i=n-3,1,-1")
    print(f"Python: for i in range({n-4}, -1, -1):")
    for i in range(n-4, -1, -1):
        print(f"  i = {i} (accesses array[{i}])")
    
    print()
    print("Fortran: do i=1,n-2")
    print(f"Python: for i in range(0, {n-2}):")
    for i in range(0, n-2):
        print(f"  i = {i} (accesses array[{i}])")
    
    print()
    print("Fortran: do i=n-3,n")
    print(f"Python: for i in range({n-4}, {n}):")
    for i in range(n-4, n):
        print(f"  i = {i} (accesses array[{i}])")

def main():
    print("FORTRAN LOOP INDEX TRACING")
    print("=" * 50)
    
    fortran_output = write_fortran_loop_tracer()
    print(fortran_output)
    
    print("\n")
    show_python_equivalent()
    
    print("\n" + "=" * 50)
    print("KEY INSIGHT:")
    print("Fortran 1-based index i corresponds to Python 0-based index i-1")
    print("So when Fortran accesses a(i), Python accesses a[i-1]")
    print("\nBUT: Loop variables themselves don't need adjustment!")
    print("The issue is when using loop variable to index arrays.")

if __name__ == "__main__":
    main()