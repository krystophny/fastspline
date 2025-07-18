
program debug_quintic_detailed
    use spl_three_to_five_sub
    implicit none
    
    integer, parameter :: dp = kind(1.0d0)
    integer :: n, i, ip1
    real(dp) :: h, rhop, rhom, fac
    real(dp) :: a11,a12,a13,a21,a22,a23,a31,a32,a33,b1,b2,b3,det
    real(dp) :: abeg,bbeg,cbeg,dbeg,ebeg,fbeg
    real(dp) :: aend,bend,cend,dend,eend,fend
    real(dp), allocatable :: a(:), b(:), c(:), d(:), e(:), f(:)
    real(dp), allocatable :: alp(:), bet(:), gam(:)
    
    ! Small test case
    n = 6
    h = 1.0d0 / (n - 1)
    
    allocate(a(n), b(n), c(n), d(n), e(n), f(n))
    allocate(alp(n), bet(n), gam(n))
    
    ! x^4 values
    do i = 1, n
        a(i) = ((i-1) * h)**4
    end do
    
    write(*,*) '=== FORTRAN QUINTIC DEBUG n=6, x^4 ==='
    write(*,*) 'Input a:', (a(i), i=1,n)
    write(*,*) ''
    
    ! Start of spl_five_reg
    rhop=13.d0+sqrt(105.d0)
    rhom=13.d0-sqrt(105.d0)
    write(*,*) 'rhop=', rhop
    write(*,*) 'rhom=', rhom
    write(*,*) ''
    
    ! First boundary system
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
    write(*,*) 'First system det=', det
    
    ! Beginning boundary
    b1=a(4)-a(3)
    b2=a(5)-a(2)
    b3=a(6)-a(1)
    write(*,*) 'Beginning diffs: b1=', b1, ' b2=', b2, ' b3=', b3
    
    bbeg=b1*a22*a33+a12*a23*b3+a13*b2*a32-a12*b2*a33-a13*a22*b3-b1*a23*a32
    bbeg=bbeg/det
    dbeg=a11*b2*a33+b1*a23*a31+a13*a21*b3-b1*a21*a33-a13*b2*a31-a11*a23*b3
    dbeg=dbeg/det
    fbeg=a11*a22*b3+a12*b2*a31+b1*a21*a32-a12*a21*b3-b1*a22*a31-a11*b2*a32
    fbeg=fbeg/det
    write(*,*) 'bbeg=', bbeg, ' dbeg=', dbeg, ' fbeg=', fbeg
    
    ! End boundary
    b1=a(n-2)-a(n-3)
    b2=a(n-1)-a(n-4)
    b3=a(n)-a(n-5)
    write(*,*) 'End diffs: b1=', b1, ' b2=', b2, ' b3=', b3
    
    bend=b1*a22*a33+a12*a23*b3+a13*b2*a32-a12*b2*a33-a13*a22*b3-b1*a23*a32
    bend=bend/det
    dend=a11*b2*a33+b1*a23*a31+a13*a21*b3-b1*a21*a33-a13*b2*a31-a11*a23*b3
    dend=dend/det
    fend=a11*a22*b3+a12*b2*a31+b1*a21*a32-a12*a21*b3-b1*a22*a31-a11*b2*a32
    fend=fend/det
    write(*,*) 'bend=', bend, ' dend=', dend, ' fend=', fend
    
    ! Second boundary system
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
    write(*,*) ''
    write(*,*) 'Second system det=', det
    
    ! Beginning boundary (second)
    b1=a(4)+a(3)
    b2=a(5)+a(2)
    b3=a(6)+a(1)
    write(*,*) 'Beginning sums: b1=', b1, ' b2=', b2, ' b3=', b3
    
    abeg=b1*a22*a33+a12*a23*b3+a13*b2*a32-a12*b2*a33-a13*a22*b3-b1*a23*a32
    abeg=abeg/det
    cbeg=a11*b2*a33+b1*a23*a31+a13*a21*b3-b1*a21*a33-a13*b2*a31-a11*a23*b3
    cbeg=cbeg/det
    ebeg=a11*a22*b3+a12*b2*a31+b1*a21*a32-a12*a21*b3-b1*a22*a31-a11*b2*a32
    ebeg=ebeg/det
    write(*,*) 'abeg=', abeg, ' cbeg=', cbeg, ' ebeg=', ebeg
    
    ! End boundary (second)
    b1=a(n-2)+a(n-3)
    b2=a(n-1)+a(n-4)
    b3=a(n)+a(n-5)
    write(*,*) 'End sums: b1=', b1, ' b2=', b2, ' b3=', b3
    
    aend=b1*a22*a33+a12*a23*b3+a13*b2*a32-a12*b2*a33-a13*a22*b3-b1*a23*a32
    aend=aend/det
    cend=a11*b2*a33+b1*a23*a31+a13*a21*b3-b1*a21*a33-a13*b2*a31-a11*a23*b3
    cend=cend/det
    eend=a11*a22*b3+a12*b2*a31+b1*a21*a32-a12*a21*b3-b1*a22*a31-a11*b2*a32
    eend=eend/det
    write(*,*) 'aend=', aend, ' cend=', cend, ' eend=', eend
    
    ! First elimination
    write(*,*) ''
    write(*,*) '=== FIRST ELIMINATION ==='
    alp(1)=0.0d0
    bet(1)=ebeg*(2.d0+rhom)-5.d0*fbeg*(3.d0+1.5d0*rhom)
    write(*,*) 'alp(1)=', alp(1), ' bet(1)=', bet(1)
    
    write(*,*) 'Loop i=1,n-4 (i=1 to', n-4, '):'
    do i=1,n-4
        ip1=i+1
        alp(ip1)=-1.d0/(rhop+alp(i))
        bet(ip1)=alp(ip1)*(bet(i)- &
                 5.d0*(a(i+4)-4.d0*a(i+3)+6.d0*a(i+2)-4.d0*a(ip1)+a(i)))
        write(*,*) 'i=', i, ': alp(', ip1, ')=', alp(ip1), ' bet(', ip1, ')=', bet(ip1)
        write(*,*) '  5*(a(i+4)-4*a(i+3)+6*a(i+2)-4*a(ip1)+a(i))='
        write(*,*) '  5*(a(', i+4, ')-4*a(', i+3, ')+6*a(', i+2, ')-4*a(', ip1, ')+a(', i, '))='
        write(*,*) '  5*(', a(i+4), '-4*', a(i+3), '+6*', a(i+2), '-4*', a(ip1), '+', a(i), ')='
        write(*,*) '  ', 5.d0*(a(i+4)-4.d0*a(i+3)+6.d0*a(i+2)-4.d0*a(ip1)+a(i))
    enddo
    
    ! Back substitution
    write(*,*) ''
    write(*,*) '=== BACK SUBSTITUTION ==='
    gam(n-2)=eend*(2.d0+rhom)+5.d0*fend*(3.d0+1.5d0*rhom)
    write(*,*) 'gam(', n-2, ')=', gam(n-2)
    
    write(*,*) 'Loop i=n-3,1,-1 (i=', n-3, ' to 1):'
    do i=n-3,1,-1
        gam(i)=gam(i+1)*alp(i)+bet(i)
        write(*,*) 'i=', i, ': gam(', i, ')=gam(', i+1, ')*alp(', i, ')+bet(', i, ')'
        write(*,*) '       =', gam(i+1), '*', alp(i), '+', bet(i), '=', gam(i)
    enddo
    
    ! Second elimination
    write(*,*) ''
    write(*,*) '=== SECOND ELIMINATION ==='
    alp(1)=0.0d0
    bet(1)=ebeg-2.5d0*5.d0*fbeg
    write(*,*) 'alp(1)=', alp(1), ' bet(1)=', bet(1)
    
    write(*,*) 'Loop i=1,n-2 (i=1 to', n-2, '):'
    do i=1,n-2
        ip1=i+1
        alp(ip1)=-1.d0/(rhom+alp(i))
        bet(ip1)=alp(ip1)*(bet(i)-gam(i))
        write(*,*) 'i=', i, ': alp(', ip1, ')=', alp(ip1), ' bet(', ip1, ')=', bet(ip1)
        write(*,*) '  bet(i)-gam(i)=', bet(i), '-', gam(i), '=', bet(i)-gam(i)
    enddo
    
    ! Final e values
    write(*,*) ''
    write(*,*) '=== FINAL E VALUES ==='
    e(n)=eend+2.5d0*5.d0*fend
    write(*,*) 'e(', n, ')=', e(n)
    e(n-1)=e(n)*alp(n-1)+bet(n-1)
    write(*,*) 'e(', n-1, ')=e(', n, ')*alp(', n-1, ')+bet(', n-1, ')=', e(n-1)
    f(n-1)=(e(n)-e(n-1))/5.d0
    write(*,*) 'f(', n-1, ')=(e(', n, ')-e(', n-1, '))/5=', f(n-1)
    e(n-2)=e(n-1)*alp(n-2)+bet(n-2)
    write(*,*) 'e(', n-2, ')=e(', n-1, ')*alp(', n-2, ')+bet(', n-2, ')=', e(n-2)
    f(n-2)=(e(n-1)-e(n-2))/5.d0
    write(*,*) 'f(', n-2, ')=(e(', n-1, ')-e(', n-2, '))/5=', f(n-2)
    d(n-2)=dend+1.5d0*4.d0*eend+1.5d0**2*10.d0*fend
    write(*,*) 'd(', n-2, ')=dend+6*eend+22.5*fend=', d(n-2)
    
    write(*,*) ''
    write(*,*) 'Main loop i=n-3,1,-1 (i=', n-3, ' to 1):'
    do i=n-3,1,-1
        e(i)=e(i+1)*alp(i)+bet(i)
        f(i)=(e(i+1)-e(i))/5.d0
        write(*,*) 'i=', i, ': e(', i, ')=', e(i), ' f(', i, ')=', f(i)
    enddo
    
    write(*,*) ''
    write(*,*) '=== FINAL E ARRAY ==='
    write(*,*) 'e:', (e(i), i=1,n)
    
    deallocate(a, b, c, d, e, f, alp, bet, gam)
    
end program debug_quintic_detailed
