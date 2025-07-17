program test_orders
    use interpolate
    use spl_three_to_five_sub
    implicit none
    integer :: i, j, n, order
    real(dp) :: x_min, x_max, h, x_eval, y_eval
    real(dp), allocatable :: x(:), y(:), y_test(:)
    type(SplineData1D) :: spl_1d
    logical :: periodic
    
    ! Test parameters
    n = 10
    x_min = 0.0_dp
    x_max = 1.0_dp
    periodic = .false.
    
    ! Allocate arrays
    allocate(x(n), y(n), y_test(n))
    
    ! Create test data
    h = (x_max - x_min) / real(n - 1, dp)
    do i = 1, n
        x(i) = x_min + real(i - 1, dp) * h
        y(i) = sin(2.0_dp * 3.14159265358979_dp * x(i))
    end do
    
    ! Test order 3 (cubic)
    order = 3
    call construct_splines_1d(x_min, x_max, y, order, periodic, spl_1d)
    
    ! Write order 3 coefficients
    open(unit=11, file='order3_coeffs.txt', status='replace')
    write(11, '(A)') '# Order 3 (cubic) coefficients'
    write(11, '(A,I5)') '# order = ', spl_1d%order
    write(11, '(A,I5)') '# num_points = ', spl_1d%num_points
    write(11, '(A,L5)') '# periodic = ', spl_1d%periodic
    write(11, '(A,F20.12)') '# x_min = ', spl_1d%x_min
    write(11, '(A,F20.12)') '# h_step = ', spl_1d%h_step
    write(11, '(A)') '# Coefficients (i, j, coeff(i,j)):'
    do i = 0, order
        do j = 1, spl_1d%num_points
            write(11, '(2I5,F20.12)') i, j, spl_1d%coeff(i,j)
        end do
    end do
    close(11)
    
    ! Test order 4 (quartic)
    order = 4
    call construct_splines_1d(x_min, x_max, y, order, periodic, spl_1d)
    
    ! Write order 4 coefficients
    open(unit=12, file='order4_coeffs.txt', status='replace')
    write(12, '(A)') '# Order 4 (quartic) coefficients'
    write(12, '(A,I5)') '# order = ', spl_1d%order
    write(12, '(A,I5)') '# num_points = ', spl_1d%num_points
    write(12, '(A,L5)') '# periodic = ', spl_1d%periodic
    write(12, '(A,F20.12)') '# x_min = ', spl_1d%x_min
    write(12, '(A,F20.12)') '# h_step = ', spl_1d%h_step
    write(12, '(A)') '# Coefficients (i, j, coeff(i,j)):'
    do i = 0, order
        do j = 1, spl_1d%num_points
            write(12, '(2I5,F20.12)') i, j, spl_1d%coeff(i,j)
        end do
    end do
    close(12)
    
    write(*,*) 'Fortran coefficient generation complete. Check order3_coeffs.txt and order4_coeffs.txt'
    
    deallocate(x, y, y_test)
    
end program test_orders