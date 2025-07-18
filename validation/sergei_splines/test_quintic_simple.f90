
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
