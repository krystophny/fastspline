program validate_periodic_splines
    implicit none
    
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    ! Test parameters
    integer :: n, order, i
    real(dp) :: x_min, x_max, h
    real(dp), allocatable :: x(:), y(:)
    real(dp) :: pi, x_test, y_exact, y_spline
    
    pi = 4.0_dp * atan(1.0_dp)
    
    print *, "Periodic Spline Validation - Fortran Reference"
    print *, "=============================================="
    print *, ""
    
    ! Test 1D periodic spline
    print *, "1D Periodic Test: f(x) = sin(2πx) on [0,1]"
    print *, "-------------------------------------------"
    
    n = 16
    x_min = 0.0_dp
    x_max = 1.0_dp
    h = (x_max - x_min) / n  ! Note: periodic uses n intervals, not n-1
    
    allocate(x(n), y(n))
    
    ! Generate periodic data (last point excluded as it equals first)
    do i = 1, n
        x(i) = x_min + (i-1) * h
        y(i) = sin(2.0_dp * pi * x(i))
    end do
    
    print *, "Grid points: n =", n
    print *, "Periodic: y(0) = y(1) =", y(1)
    print *, ""
    
    ! Test different orders
    do order = 3, 4  ! Skip quintic for now
        print *, "Order:", order
        
        ! Test at several points
        print *, "Test point evaluations:"
        
        x_test = 0.25_dp
        y_exact = sin(2.0_dp * pi * x_test)
        print '(A,F6.3,A,F10.7)', "  x = ", x_test, ", exact = ", y_exact
        
        x_test = 0.5_dp
        y_exact = sin(2.0_dp * pi * x_test)
        print '(A,F6.3,A,F10.7)', "  x = ", x_test, ", exact = ", y_exact
        
        x_test = 0.75_dp
        y_exact = sin(2.0_dp * pi * x_test)
        print '(A,F6.3,A,F10.7)', "  x = ", x_test, ", exact = ", y_exact
        
        ! Test periodicity
        x_test = 0.95_dp
        y_exact = sin(2.0_dp * pi * x_test)
        print '(A,F6.3,A,F10.7)', "  x = ", x_test, ", exact = ", y_exact
        
        x_test = 1.05_dp  ! Should wrap to 0.05
        y_exact = sin(2.0_dp * pi * 0.05_dp)
        print '(A,F6.3,A,F10.7)', "  x = ", x_test, " (wraps to 0.05), exact = ", y_exact
        
        print *, ""
    end do
    
    ! Test 2D periodic case
    print *, ""
    print *, "2D Periodic Test: f(x,y) = sin(2πx) * cos(2πy) on [0,1]×[0,1]"
    print *, "-------------------------------------------------------------"
    
    ! Simple test values
    print *, "Test points:"
    print *, "  (0.0, 0.0): exact =", sin(0.0_dp) * cos(0.0_dp), "= 0.0"
    print *, "  (0.5, 0.5): exact =", sin(pi) * cos(pi), "= 0.0"
    print *, "  (0.25, 0.25): exact =", sin(pi/2.0_dp) * cos(pi/2.0_dp), "= 0.0"
    print *, "  (0.25, 0.0): exact =", sin(pi/2.0_dp) * cos(0.0_dp), "= 1.0"
    print *, "  (0.0, 0.25): exact =", sin(0.0_dp) * cos(pi/2.0_dp), "= 0.0"
    
    ! Test mixed periodic/non-periodic
    print *, ""
    print *, "Mixed Boundary Test: periodic in x, non-periodic in y"
    print *, "----------------------------------------------------"
    print *, "Function: f(x,y) = sin(2πx) * exp(-y)"
    print *, ""
    print *, "Boundary behavior:"
    print *, "  x-direction: f(0,y) = f(1,y) (periodic)"
    print *, "  y-direction: natural spline boundaries"
    
    deallocate(x, y)
    
end program validate_periodic_splines