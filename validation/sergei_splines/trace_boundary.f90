
program trace_boundary
    use spl_three_to_five_sub
    implicit none
    
    integer, parameter :: dp = kind(1.0d0)
    real(dp) :: a11,a12,a13,a21,a22,a23,a31,a32,a33,b1,b2,b3,det
    real(dp) :: abeg,bbeg,cbeg,dbeg,ebeg,fbeg
    real(dp) :: aend,bend,cend,dend,eend,fend
    real(dp) :: h
    real(dp), dimension(6) :: a
    integer :: i
    
    ! x^4 values
    h = 1.0d0 / 5.0d0
    do i = 1, 6
        a(i) = ((i-1) * h)**4
    end do
    
    write(*,*) 'Fortran boundary values for x^4, n=6:'
    write(*,*) 'Input a:', (a(i), i=1,6)
    
    ! First system
    a11=1.d0
    a12=1.d0/4.d0
    a13=1.d0/16.d0
    a21=3.d0
    a22=27.d0/4.d0
    a23=9.d0*27.d0/16.d0
    a31=5.d0
    a32=125.d0/4.d0
    a33=5.d0**5/16.d0
    det=a11*a22*a33+a12*a23*a31+a13*a21*a32-a12*a21*a33-a13*a22*a31-a11*a23*a32
    
    b1=a(4)-a(3)
    b2=a(5)-a(2)
    b3=a(6)-a(1)
    bbeg=b1*a22*a33+a12*a23*b3+a13*b2*a32-a12*b2*a33-a13*a22*b3-b1*a23*a32
    bbeg=bbeg/det
    dbeg=a11*b2*a33+b1*a23*a31+a13*a21*b3-b1*a21*a33-a13*b2*a31-a11*a23*b3
    dbeg=dbeg/det
    fbeg=a11*a22*b3+a12*b2*a31+b1*a21*a32-a12*a21*b3-b1*a22*a31-a11*b2*a32
    fbeg=fbeg/det
    
    write(*,*) 'fbeg =', fbeg
    
    ! Second system  
    a11=2.d0
    a12=1.d0/2.d0
    a13=1.d0/8.d0
    a21=2.d0
    a22=9.d0/2.d0
    a23=81.d0/8.d0
    a31=2.d0
    a32=25.d0/2.d0
    a33=625.d0/8.d0
    det=a11*a22*a33+a12*a23*a31+a13*a21*a32-a12*a21*a33-a13*a22*a31-a11*a23*a32
    
    b1=a(4)+a(3)
    b2=a(5)+a(2)
    b3=a(6)+a(1)
    abeg=b1*a22*a33+a12*a23*b3+a13*b2*a32-a12*b2*a33-a13*a22*b3-b1*a23*a32
    abeg=abeg/det
    cbeg=a11*b2*a33+b1*a23*a31+a13*a21*b3-b1*a21*a33-a13*b2*a31-a11*a23*b3
    cbeg=cbeg/det
    ebeg=a11*a22*b3+a12*b2*a31+b1*a21*a32-a12*a21*b3-b1*a22*a31-a11*b2*a32
    ebeg=ebeg/det
    
    write(*,*) 'ebeg =', ebeg
    
end program trace_boundary
