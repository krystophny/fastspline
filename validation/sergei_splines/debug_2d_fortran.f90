program debug_2d_fortran
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
    
    write(*,*) '2D Fortran Debug Program'
    write(*,*) '========================'
    write(*,*) 'Grid size:', nx, 'x', ny
    write(*,*) 'Function: sin(π*x) * cos(π*y)'
    write(*,*) 'Order:', order
    write(*,*) 'Domain: [0,1] x [0,1]'
    write(*,*) ''
    
    ! Print input data
    write(*,*) 'Input data (first few points):'
    do i = 1, min(3, nx)
        do j = 1, min(3, ny)
            write(*,'(A,I0,A,I0,A,F8.4,A,F8.4,A,F10.6)') &
                'Z(', i, ',', j, ') at (', x_data(i), ',', y_data(j), ') = ', Z_data(i, j)
        end do
    end do
    write(*,*) ''
    
    ! Construct 2D spline
    call construct_splines_2d(x_min, x_max, Z_data, order, periodic, spl)
    write(*,*) 'Spline construction completed'
    
    ! Print some coefficient information
    write(*,*) 'Coefficient array shape:', shape(spl%coeff)
    write(*,*) 'h_step:', spl%h_step
    write(*,*) ''
    
    ! Test evaluation at data points
    write(*,*) 'Testing evaluation at data points:'
    do i = 1, min(3, nx)
        do j = 1, min(3, ny)
            x_eval = [x_data(i), y_data(j)]
            call evaluate_splines_2d(spl, x_eval, z_eval)
            write(*,'(A,I0,A,I0,A,F8.4,A,F8.4,A,F10.6,A,F10.6,A,E10.2)') &
                'Point (', i, ',', j, ') at (', x_eval(1), ',', x_eval(2), &
                '): got ', z_eval, ', expected ', Z_data(i, j), &
                ', error ', abs(z_eval - Z_data(i, j))
        end do
    end do
    write(*,*) ''
    
    ! Test evaluation at intermediate points
    write(*,*) 'Testing evaluation at intermediate points:'
    x_eval = [0.25_dp, 0.25_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    x_eval = [0.5_dp, 0.5_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    x_eval = [0.75_dp, 0.75_dp]
    call evaluate_splines_2d(spl, x_eval, z_eval)
    write(*,'(A,F8.4,A,F8.4,A,F10.6,A,F10.6)') &
        'Point (', x_eval(1), ',', x_eval(2), '): got ', z_eval, &
        ', expected ', sin(pi * x_eval(1)) * cos(pi * x_eval(2))
    
    write(*,*) ''
    write(*,*) 'Fortran 2D spline evaluation completed'
    
    ! Clean up
    call destroy_splines_2d(spl)
    
end program debug_2d_fortran