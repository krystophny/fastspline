program validate_3d_splines
    implicit none
    
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    ! Grid parameters
    integer, parameter :: n1 = 8, n2 = 8, n3 = 8
    real(dp) :: x_min(3), x_max(3)
    real(dp), allocatable :: y_data(:,:,:)
    real(dp) :: x1(n1), x2(n2), x3(n3)
    real(dp) :: h1, h2, h3
    real(dp) :: pi
    integer :: i, j, k, order
    
    ! Spline arrays
    real(dp), allocatable :: coeff(:)
    integer :: ndim, coeff_size
    integer :: num_points(3), orders(3)
    logical :: periodic(3)
    
    ! Test points
    real(dp) :: test_points(3,5)
    real(dp) :: x_eval(3), y_eval, y_exact
    real(dp) :: error, max_error, rms_error
    integer :: n_test
    
    ! External functions
    real(dp) :: evaluate_splines_3d
    external :: init_splines_3d, evaluate_splines_3d, destroy_splines_3d
    
    pi = 4.0_dp * atan(1.0_dp)
    
    ! Domain
    x_min = [0.0_dp, 0.0_dp, 0.0_dp]
    x_max = [1.0_dp, 1.0_dp, 2.0_dp]
    
    ! Grid parameters
    num_points = [n1, n2, n3]
    periodic = [.false., .false., .false.]
    
    ! Create grid
    h1 = (x_max(1) - x_min(1)) / (n1 - 1)
    h2 = (x_max(2) - x_min(2)) / (n2 - 1)
    h3 = (x_max(3) - x_min(3)) / (n3 - 1)
    
    do i = 1, n1
        x1(i) = x_min(1) + (i-1) * h1
    end do
    do j = 1, n2
        x2(j) = x_min(2) + (j-1) * h2
    end do
    do k = 1, n3
        x3(k) = x_min(3) + (k-1) * h3
    end do
    
    ! Allocate and fill data: f(x,y,z) = sin(πx) * cos(πy) * exp(-z/2)
    allocate(y_data(n1, n2, n3))
    do k = 1, n3
        do j = 1, n2
            do i = 1, n1
                y_data(i,j,k) = sin(pi * x1(i)) * cos(pi * x2(j)) * exp(-x3(k)/2.0_dp)
            end do
        end do
    end do
    
    ! Define test points
    test_points(:,1) = [0.5_dp, 0.5_dp, 1.0_dp]
    test_points(:,2) = [0.25_dp, 0.75_dp, 0.5_dp]
    test_points(:,3) = [0.8_dp, 0.3_dp, 1.5_dp]
    test_points(:,4) = [0.1_dp, 0.9_dp, 0.2_dp]
    test_points(:,5) = [0.6_dp, 0.4_dp, 1.8_dp]
    
    print *, "3D Spline Validation - Fortran Reference"
    print *, "========================================="
    print *, "Test function: f(x,y,z) = sin(πx) * cos(πy) * exp(-z/2)"
    print *, "Grid size:", n1, "x", n2, "x", n3
    print *, "Domain: [0,1] x [0,1] x [0,2]"
    print *, ""
    
    ! Test different orders
    do order = 3, 5
        print *, "Order:", order
        print *, "--------------"
        
        orders = [order, order, order]
        
        ! Calculate coefficient size
        coeff_size = (order+1)**3 * n1 * n2 * n3
        allocate(coeff(coeff_size))
        
        ! Initialize splines
        call init_splines_3d(x_min, x_max, reshape(y_data, [n1*n2*n3]), &
                           num_points, orders, periodic, coeff)
        
        max_error = 0.0_dp
        rms_error = 0.0_dp
        
        ! Evaluate at test points
        do n_test = 1, 5
            x_eval = test_points(:, n_test)
            
            ! Exact value
            y_exact = sin(pi * x_eval(1)) * cos(pi * x_eval(2)) * exp(-x_eval(3)/2.0_dp)
            
            ! Interpolated value
            y_eval = evaluate_splines_3d(orders, num_points, periodic, &
                                       x_min, x_max, coeff, x_eval)
            
            ! Error
            error = abs(y_eval - y_exact)
            max_error = max(max_error, error)
            rms_error = rms_error + error**2
            
            print '(A,3F8.4,A,F12.8,A,F12.8,A,E12.4)', &
                "  Point (", x_eval(1), x_eval(2), x_eval(3), &
                "): exact=", y_exact, ", spline=", y_eval, ", error=", error
        end do
        
        rms_error = sqrt(rms_error / 5.0_dp)
        print '(A,E12.4,A,E12.4)', "  Max error:", max_error, ", RMS error:", rms_error
        print *, ""
        
        ! Print some coefficients for comparison
        if (order == 3) then
            print *, "First 10 coefficients (for Python comparison):"
            do i = 1, min(10, coeff_size)
                print '(A,I4,A,F16.10)', "  coeff(", i, ") = ", coeff(i)
            end do
            print *, ""
        end if
        
        deallocate(coeff)
    end do
    
    deallocate(y_data)
    
contains

    ! Include the interpolation routines
    include 'src/interpolate.f90'
    
end program validate_3d_splines