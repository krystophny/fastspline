program debug_fortran_problem_point
    use interpolate
    implicit none

    integer, parameter :: nx = 8, ny = 8
    real(dp), parameter :: x_min(2) = [0.0_dp, 0.0_dp]
    real(dp), parameter :: x_max(2) = [1.0_dp, 1.0_dp]
    integer, parameter :: order(2) = [3, 3]
    logical, parameter :: periodic(2) = [.false., .false.]
    
    real(dp) :: x_data(nx), y_data(ny)
    real(dp) :: Z_data(nx, ny)
    real(dp) :: x_eval(2), z_eval
    type(SplineData2D) :: spl
    
    integer :: i, j
    real(dp) :: pi
    
    pi = 4.0_dp * atan(1.0_dp)
    
    ! Create data grid
    do i = 1, nx
        x_data(i) = real(i-1, dp) / real(nx-1, dp)
    end do
    do j = 1, ny
        y_data(j) = real(j-1, dp) / real(ny-1, dp)
    end do
    
    ! Test function: sin(π*x) * cos(π*y)
    do i = 1, nx
        do j = 1, ny
            Z_data(i, j) = sin(pi * x_data(i)) * cos(pi * y_data(j))
        end do
    end do
    
    write(*,*) 'Fortran Debug for Problem Point (0.8, 0.3)'
    write(*,*) '==========================================='
    write(*,*) 'Grid size:', nx, 'x', ny
    write(*,*) 'Function: sin(π*x) * cos(π*y)'
    write(*,*) 'Order:', order
    write(*,*) ''
    
    ! Construct 2D spline
    call construct_splines_2d(x_min, x_max, Z_data, order, periodic, spl)
    write(*,*) 'Spline construction completed'
    write(*,*) ''
    
    ! Test the specific problem point
    x_eval = [0.8_dp, 0.3_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    
    write(*,'(A,F8.4,A,F8.4,A,F10.6)') &
        'Problem point (', x_eval(1), ',', x_eval(2), '): got ', z_eval
    write(*,'(A,F10.6)') &
        'Expected: ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    write(*,'(A,E10.2)') &
        'Error: ', abs(z_eval - sin(pi * x_eval(1)) * cos(pi * x_eval(2)))
    write(*,*) ''
    
    ! Test a few more points around the problem area
    write(*,*) 'Testing nearby points:'
    
    x_eval = [0.75_dp, 0.25_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    x_eval = [0.85_dp, 0.35_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    x_eval = [0.8_dp, 0.2_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    x_eval = [0.8_dp, 0.4_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    write(*,*) ''
    write(*,*) 'Fortran 2D spline evaluation completed'
    
    ! Clean up
    call destroy_splines_2d(spl)
    
end program debug_fortran_problem_point