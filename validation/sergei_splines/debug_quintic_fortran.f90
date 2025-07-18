
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
