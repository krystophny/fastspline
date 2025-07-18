program test_periodic_direct
  ! Direct test of Fortran periodic implementation
  implicit none
  
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 16
  real(dp), parameter :: pi = 4.0_dp * atan(1.0_dp)
  
  real(dp) :: h, x_min, x_max
  real(dp), dimension(n) :: x, y, bi, ci, di
  integer :: i
  real(dp) :: x_test, y_spline, y_exact, error
  
  ! Test setup
  x_min = 0.0_dp
  x_max = 1.0_dp
  h = (x_max - x_min) / n  ! Periodic: h = L/n
  
  write(*,*) 'Fortran Periodic Cubic Spline Test'
  write(*,*) '=================================='
  write(*,'(a,i0)') 'n = ', n
  write(*,'(a,f12.8)') 'h = ', h
  
  ! Create test data: sin(2πx)
  do i = 1, n
    x(i) = x_min + (i-1) * h
    y(i) = sin(2.0_dp * pi * x(i))
  end do
  
  ! Print input data
  write(*,*) 'Input data:'
  do i = 1, min(5, n)
    write(*,'(a,i2,a,f8.4,a,f8.4)') 'x[', i, '] = ', x(i), ', y[', y(i)
  end do
  if (n > 5) write(*,*) '...'
  
  ! Call Fortran periodic spline - add explicit interface
  call splper(n, h, y, bi, ci, di)
  
  ! Print coefficients
  write(*,*) 'Coefficients:'
  do i = 1, min(5, n)
    write(*,'(a,i2,a,3f12.6)') 'i=', i, ': bi=', bi(i), ', ci=', ci(i), ', di=', di(i)
  end do
  if (n > 5) write(*,*) '...'
  
  ! Test continuity
  write(*,*) 'Boundary continuity:'
  write(*,'(a,f15.12)') 'f(0) = ', y(1)
  write(*,'(a,f15.12)') 'f(1-) = ', y(n) + bi(n) + ci(n) + di(n)
  write(*,'(a,e12.2)') 'Difference = ', abs(y(1) - (y(n) + bi(n) + ci(n) + di(n)))
  
  write(*,'(a,f15.12)') 'f''(0) = ', bi(1)
  write(*,'(a,f15.12)') 'f''(1-) = ', bi(n) + 2*ci(n) + 3*di(n)
  write(*,'(a,e12.2)') 'Difference = ', abs(bi(1) - (bi(n) + 2*ci(n) + 3*di(n)))
  
  write(*,'(a,f15.12)') 'f''''(0) = ', 2*ci(1)
  write(*,'(a,f15.12)') 'f''''(1-) = ', 2*ci(n) + 6*di(n)
  write(*,'(a,e12.2)') 'Difference = ', abs(2*ci(1) - (2*ci(n) + 6*di(n)))
  
  ! Test evaluation accuracy
  write(*,*) 'Accuracy test:'
  x_test = 0.1_dp
  call evaluate_cubic(x_test, x_min, h, n, y, bi, ci, di, y_spline)
  y_exact = sin(2.0_dp * pi * x_test)
  error = abs(y_spline - y_exact)
  write(*,'(a,f6.3,a,f12.8,a,f12.8,a,e10.2)') &
    'x=', x_test, ': spline=', y_spline, ', exact=', y_exact, ', error=', error
  
contains
  
  subroutine evaluate_cubic(x_eval, x_min, h_step, num_points, a, b, c, d, result)
    implicit none
    real(dp), intent(in) :: x_eval, x_min, h_step
    integer, intent(in) :: num_points
    real(dp), dimension(num_points), intent(in) :: a, b, c, d
    real(dp), intent(out) :: result
    
    integer :: i
    real(dp) :: t
    
    ! Find interval
    i = int((x_eval - x_min) / h_step) + 1
    if (i < 1) i = 1
    if (i > num_points) i = num_points
    
    ! Local coordinate
    t = (x_eval - x_min - (i-1)*h_step) / h_step
    
    ! Evaluate cubic polynomial
    result = a(i) + t*(b(i) + t*(c(i) + t*d(i)))
  end subroutine evaluate_cubic

end program test_periodic_direct