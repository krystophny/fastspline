
program trace_loops
    implicit none
    integer :: i, n
    
    n = 6
    write(*,*) 'Fortran loop traces for n=', n
    write(*,*) ''
    
    write(*,*) 'Loop: do i=1,n-4'
    write(*,*) 'n-4 =', n-4
    do i = 1, n-4
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Loop: do i=n-3,1,-1'
    write(*,*) 'n-3 =', n-3
    do i = n-3, 1, -1
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Loop: do i=1,n-2'
    write(*,*) 'n-2 =', n-2
    do i = 1, n-2
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Loop: do i=n-3,n'
    write(*,*) 'n-3 =', n-3, ', n =', n
    do i = n-3, n
        write(*,*) '  i =', i
    end do
    
    write(*,*) ''
    write(*,*) 'Array access in loops:'
    write(*,*) 'When i=1, a(i) accesses element 1'
    write(*,*) 'When i=n, a(i) accesses element n'
    
end program trace_loops
