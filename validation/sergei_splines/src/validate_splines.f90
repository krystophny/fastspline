program validate_splines
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
    order = 5
    periodic = .false.
    
    ! Allocate arrays
    allocate(x(n), y(n), y_test(n))
    
    ! Create test data
    h = (x_max - x_min) / real(n - 1, dp)
    do i = 1, n
        x(i) = x_min + real(i - 1, dp) * h
        y(i) = sin(2.0_dp * 3.14159265358979_dp * x(i))
    end do
    
    ! Write input data
    open(unit=10, file='data/input_data.txt', status='replace')
    write(10, '(A)') '# Input data for validation'
    write(10, '(A,I5)') '# n = ', n
    write(10, '(A,F12.6)') '# x_min = ', x_min
    write(10, '(A,F12.6)') '# x_max = ', x_max
    write(10, '(A,I5)') '# order = ', order
    write(10, '(A,L5)') '# periodic = ', periodic
    write(10, '(A)') '# x, y values:'
    do i = 1, n
        write(10, '(2F20.12)') x(i), y(i)
    end do
    close(10)
    
    ! Construct 1D spline
    call construct_splines_1d(x_min, x_max, y, order, periodic, spl_1d)
    
    ! Write spline coefficients
    open(unit=11, file='data/spline_coeffs_1d.txt', status='replace')
    write(11, '(A)') '# 1D Spline coefficients'
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
    
    ! Evaluate spline at test points
    open(unit=12, file='data/evaluation_results.txt', status='replace')
    write(12, '(A)') '# Spline evaluation results'
    write(12, '(A)') '# x, y_spline, y_exact, error'
    
    do i = 1, 21
        x_eval = x_min + real(i - 1, dp) * (x_max - x_min) / 20.0_dp
        call evaluate_splines_1d(spl_1d, x_eval, y_eval)
        write(12, '(4F20.12)') x_eval, y_eval, &
            sin(2.0_dp * 3.14159265358979_dp * x_eval), &
            abs(y_eval - sin(2.0_dp * 3.14159265358979_dp * x_eval))
    end do
    close(12)
    
    ! Test 2D spline
    call test_2d_spline()
    
    write(*,*) 'Fortran validation complete. Check data/ directory for output files.'
    
contains

    subroutine test_2d_spline()
        integer :: nx, ny, ix, iy
        real(dp) :: x_min_2d(2), x_max_2d(2), x_eval_2d(2), z_eval
        real(dp), allocatable :: z(:,:)
        type(SplineData2D) :: spl_2d
        integer :: order_2d(2)
        logical :: periodic_2d(2)
        
        ! 2D test parameters
        nx = 8
        ny = 8
        x_min_2d = [0.0_dp, 0.0_dp]
        x_max_2d = [1.0_dp, 1.0_dp]
        order_2d = [5, 5]
        periodic_2d = [.false., .false.]
        
        allocate(z(nx, ny))
        
        ! Create 2D test data
        do ix = 1, nx
            do iy = 1, ny
                x_eval_2d(1) = x_min_2d(1) + real(ix - 1, dp) * (x_max_2d(1) - x_min_2d(1)) / real(nx - 1, dp)
                x_eval_2d(2) = x_min_2d(2) + real(iy - 1, dp) * (x_max_2d(2) - x_min_2d(2)) / real(ny - 1, dp)
                z(ix, iy) = sin(2.0_dp * 3.14159265358979_dp * x_eval_2d(1)) * &
                           cos(2.0_dp * 3.14159265358979_dp * x_eval_2d(2))
            end do
        end do
        
        ! Write 2D input data
        open(unit=13, file='data/input_data_2d.txt', status='replace')
        write(13, '(A)') '# 2D Input data for validation'
        write(13, '(A,2I5)') '# nx, ny = ', nx, ny
        write(13, '(A,2F12.6)') '# x_min = ', x_min_2d
        write(13, '(A,2F12.6)') '# x_max = ', x_max_2d
        write(13, '(A,2I5)') '# order = ', order_2d
        write(13, '(A,2L5)') '# periodic = ', periodic_2d
        write(13, '(A)') '# z values (row by row):'
        do ix = 1, nx
            write(13, '(8F12.6)') (z(ix, iy), iy = 1, ny)
        end do
        close(13)
        
        ! Construct 2D spline
        call construct_splines_2d(x_min_2d, x_max_2d, z, order_2d, periodic_2d, spl_2d)
        
        ! Evaluate 2D spline at test points
        open(unit=14, file='data/evaluation_results_2d.txt', status='replace')
        write(14, '(A)') '# 2D Spline evaluation results'
        write(14, '(A)') '# x1, x2, z_spline, z_exact, error'
        
        do ix = 1, 11
            do iy = 1, 11
                x_eval_2d(1) = x_min_2d(1) + real(ix - 1, dp) * (x_max_2d(1) - x_min_2d(1)) / 10.0_dp
                x_eval_2d(2) = x_min_2d(2) + real(iy - 1, dp) * (x_max_2d(2) - x_min_2d(2)) / 10.0_dp
                call evaluate_splines_2d(spl_2d, x_eval_2d, z_eval)
                write(14, '(5F20.12)') x_eval_2d(1), x_eval_2d(2), z_eval, &
                    sin(2.0_dp * 3.14159265358979_dp * x_eval_2d(1)) * &
                    cos(2.0_dp * 3.14159265358979_dp * x_eval_2d(2)), &
                    abs(z_eval - sin(2.0_dp * 3.14159265358979_dp * x_eval_2d(1)) * &
                                cos(2.0_dp * 3.14159265358979_dp * x_eval_2d(2)))
            end do
        end do
        close(14)
        
        deallocate(z)
    end subroutine test_2d_spline
    
end program validate_splines