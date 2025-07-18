program validate_3d_simple
    implicit none
    
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    ! Grid parameters
    integer, parameter :: n1 = 8, n2 = 8, n3 = 8
    real(dp) :: x_min(3), x_max(3)
    real(dp), allocatable :: y_data(:)
    real(dp) :: x_val, y_val, z_val
    real(dp) :: h1, h2, h3
    real(dp) :: pi
    integer :: i, j, k, idx, order
    
    ! Test points
    real(dp) :: test_x, test_y, test_z
    real(dp) :: exact_val, spline_val
    
    pi = 4.0_dp * atan(1.0_dp)
    
    ! Domain
    x_min = [0.0_dp, 0.0_dp, 0.0_dp]
    x_max = [1.0_dp, 1.0_dp, 2.0_dp]
    
    ! Grid steps
    h1 = (x_max(1) - x_min(1)) / (n1 - 1)
    h2 = (x_max(2) - x_min(2)) / (n2 - 1)
    h3 = (x_max(3) - x_min(3)) / (n3 - 1)
    
    ! Allocate flattened data array
    allocate(y_data(n1 * n2 * n3))
    
    ! Fill data: f(x,y,z) = sin(πx) * cos(πy) * exp(-z/2)
    ! Using same ordering as Python (i varies fastest)
    idx = 0
    do k = 1, n3
        z_val = x_min(3) + (k-1) * h3
        do j = 1, n2
            y_val = x_min(2) + (j-1) * h2
            do i = 1, n1
                idx = idx + 1
                x_val = x_min(1) + (i-1) * h1
                y_data(idx) = sin(pi * x_val) * cos(pi * y_val) * exp(-z_val/2.0_dp)
            end do
        end do
    end do
    
    print *, "3D Spline Test Data - Fortran"
    print *, "============================="
    print *, "Test function: f(x,y,z) = sin(πx) * cos(πy) * exp(-z/2)"
    print *, "Grid size:", n1, "x", n2, "x", n3
    print *, "Domain: [0,1] x [0,1] x [0,2]"
    print *, ""
    
    ! Print first few data points for verification
    print *, "First 10 data points (flattened):"
    do i = 1, min(10, n1*n2*n3)
        print '(A,I4,A,F16.10)', "  y_data(", i, ") = ", y_data(i)
    end do
    print *, ""
    
    ! Test specific points
    print *, "Test point values:"
    
    test_x = 0.5_dp
    test_y = 0.5_dp
    test_z = 1.0_dp
    exact_val = sin(pi * test_x) * cos(pi * test_y) * exp(-test_z/2.0_dp)
    print '(A,3F8.4,A,F16.10)', "  Point (", test_x, test_y, test_z, "): ", exact_val
    
    test_x = 0.25_dp
    test_y = 0.75_dp
    test_z = 0.5_dp
    exact_val = sin(pi * test_x) * cos(pi * test_y) * exp(-test_z/2.0_dp)
    print '(A,3F8.4,A,F16.10)', "  Point (", test_x, test_y, test_z, "): ", exact_val
    
    test_x = 0.8_dp
    test_y = 0.3_dp
    test_z = 1.5_dp
    exact_val = sin(pi * test_x) * cos(pi * test_y) * exp(-test_z/2.0_dp)
    print '(A,3F8.4,A,F16.10)', "  Point (", test_x, test_y, test_z, "): ", exact_val
    
    deallocate(y_data)
    
end program validate_3d_simple