program debug_quartic_fortran
  implicit none
  
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n = 10
  real(dp), parameter :: pi = 4.0d0 * atan(1.0d0)
  real(dp), parameter :: h = 1.0d0 / (n - 1)
  
  real(dp), dimension(n) :: x, y, a, b, c, d, e
  integer :: i
  real(dp) :: x_test, y_spline, y_exact, x_local, error
  integer :: interval
  
  ! Generate test data
  do i = 1, n
    x(i) = real(i-1, dp) / real(n-1, dp)
    y(i) = sin(2.0d0 * pi * x(i))
  enddo
  
  ! Initialize coefficients
  a = y
  
  print *, 'FORTRAN QUARTIC DEBUG'
  print *, '===================='
  print *, 'Input data:'
  do i = 1, n
    write(*, '(a, i2, a, f8.6, a, f8.6)') '  x[', i, '] = ', x(i), ', y[', y(i)
  enddo
  
  ! Call quartic spline construction
  call spl_four_reg(n, h, a, b, c, d, e)
  
  print *, ''
  print *, 'Coefficients from Fortran:'
  print *, '   i        a            b            c            d            e'
  print *, '------------------------------------------------------------------'
  do i = 1, n
    write(*, '(i4, 5f13.6)') i, a(i), b(i), c(i), d(i), e(i)
  enddo
  
  ! Test evaluation at x=0.5
  x_test = 0.5d0
  y_exact = sin(2.0d0 * pi * x_test)
  
  ! Find interval (0-based like Python)
  interval = int(x_test / h) + 1  ! Convert to 1-based
  if (interval > n-1) interval = n-1
  x_local = x_test - real(interval-1, dp) * h
  
  ! Evaluate spline
  y_spline = a(interval) + x_local * (b(interval) + x_local * (c(interval) + &
             x_local * (d(interval) + x_local * e(interval))))
  
  error = abs(y_spline - y_exact)
  
  print *, ''
  print *, 'Evaluation at x=0.5:'
  write(*, '(a, f15.12)') '  Spline result: ', y_spline
  write(*, '(a, f15.12)') '  Exact result:  ', y_exact
  write(*, '(a, es10.2)') '  Error:         ', error
  write(*, '(a, i0)') '  Interval used: ', interval
  write(*, '(a, f15.12)') '  x_local:       ', x_local

contains

subroutine spl_four_reg(n,h,a,b,c,d,e)
  implicit none

  integer, parameter :: dp = kind(1.0d0)

  integer :: n,i,ip1
  real(dp) :: h,fac,fpl31,fpl40,fmn31,fmn40
  real(dp), dimension(n) :: a,b,c,d,e
  real(dp), dimension(:), allocatable :: alp,bet,gam

  allocate(alp(n),bet(n),gam(n))

  fpl31=.5d0*(a(2)+a(4))-a(3)
  fpl40=.5d0*(a(1)+a(5))-a(3)
  fmn31=.5d0*(a(4)-a(2))
  fmn40=.5d0*(a(5)-a(1))
  d(3)=(fmn40-2.d0*fmn31)/6.d0
  e(3)=(fpl40-4.d0*fpl31)/12.d0
  d(2)=d(3)-4.d0*e(3)
  d(1)=d(3)-8.d0*e(3)

  alp(1)=0.0d0
  bet(1)=d(1)+d(2)

  do i=1,n-3
    ip1=i+1
    alp(ip1)=-1.d0/(10.d0+alp(i))
    bet(ip1)=alp(ip1)*(bet(i)-4.d0*(a(i+3)-3.d0*(a(i+2)-a(ip1))-a(i)))
  enddo

  fpl31=.5d0*(a(n-3)+a(n-1))-a(n-2)
  fpl40=.5d0*(a(n-4)+a(n))-a(n-2)
  fmn31=.5d0*(a(n-1)-a(n-3))
  fmn40=.5d0*(a(n)-a(n-4))
  d(n-2)=(fmn40-2.d0*fmn31)/6.d0
  e(n-2)=(fpl40-4.d0*fpl31)/12.d0
  d(n-1)=d(n-2)+4.d0*e(n-2)
  d(n)=d(n-2)+8.d0*e(n-2)

  gam(n-1)=d(n)+d(n-1)

  do i=n-2,1,-1
    gam(i)=gam(i+1)*alp(i)+bet(i)
    d(i)=gam(i)-d(i+1)
    e(i)=(d(i+1)-d(i))/4.d0
    c(i)=0.5d0*(a(i+2)+a(i))-a(i+1)-0.125d0*(d(i+2)+12.d0*d(i+1)+11.d0*d(i))
    b(i)=a(i+1)-a(i)-c(i)-(3.d0*d(i)+d(i+1))/4.d0
  enddo

  b(n-1)=b(n-2)+2.d0*c(n-2)+3.d0*d(n-2)+4.d0*e(n-2)
  c(n-1)=c(n-2)+3.d0*d(n-2)+6.d0*e(n-2)

end subroutine spl_four_reg

end program debug_quartic_fortran