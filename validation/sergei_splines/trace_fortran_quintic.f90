
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
