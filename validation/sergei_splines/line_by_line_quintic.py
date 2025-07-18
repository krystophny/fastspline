#!/usr/bin/env python3
"""
Line-by-line comparison of quintic implementation with Fortran
"""

import numpy as np
import subprocess

def write_fortran_debug_program():
    """Create Fortran program that prints every intermediate value"""
    fortran_code = """
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
"""
    
    with open('debug_quintic_detailed.f90', 'w') as f:
        f.write(fortran_code)
    
    try:
        subprocess.run(['gfortran', '-o', 'debug_quintic_detailed',
                       'debug_quintic_detailed.f90', 'src/spl_three_to_five.f90'],
                      check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Compilation failed: {e}")
        return False

def run_fortran_debug():
    """Run the Fortran debug program"""
    try:
        result = subprocess.run(['./debug_quintic_detailed'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Execution failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Failed to run: {e}")
        return None

def trace_python_detailed():
    """Trace Python implementation with same detail"""
    print("\n=== PYTHON QUINTIC DEBUG n=6, x^4 ===")
    
    n = 6
    h = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = x**4
    coeff = np.copy(y).astype(np.float64)
    
    print(f"Input y: {y}")
    print()
    
    # Constants
    rhop = 13.0 + np.sqrt(105.0)
    rhom = 13.0 - np.sqrt(105.0)
    print(f"rhop= {rhop}")
    print(f"rhom= {rhom}")
    print()
    
    # Working arrays
    alp = np.zeros(n, dtype=np.float64)
    bet = np.zeros(n, dtype=np.float64)
    gam = np.zeros(n, dtype=np.float64)
    
    # First boundary system
    a11 = 1.0
    a12 = 1.0/4.0
    a13 = 1.0/16.0
    a21 = 3.0
    a22 = 27.0/4.0
    a23 = 9.0*27.0/16.0
    a31 = 5.0
    a32 = 125.0/4.0
    a33 = 5.0**5/16.0
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    print(f"First system det= {det}")
    
    # Beginning boundary
    b1 = coeff[3] - coeff[2]
    b2 = coeff[4] - coeff[1]
    b3 = coeff[5] - coeff[0]
    print(f"Beginning diffs: b1= {b1}  b2= {b2}  b3= {b3}")
    
    bbeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    dbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    fbeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    print(f"bbeg= {bbeg}  dbeg= {dbeg}  fbeg= {fbeg}")
    
    # End boundary
    b1 = coeff[n-3] - coeff[n-4]
    b2 = coeff[n-2] - coeff[n-5]
    b3 = coeff[n-1] - coeff[n-6]
    print(f"End diffs: b1= {b1}  b2= {b2}  b3= {b3}")
    
    bend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    dend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    fend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    print(f"bend= {bend}  dend= {dend}  fend= {fend}")
    
    # Second system
    a11 = 2.0
    a12 = 1.0/2.0
    a13 = 1.0/8.0
    a21 = 2.0
    a22 = 9.0/2.0
    a23 = 81.0/8.0
    a31 = 2.0
    a32 = 25.0/2.0
    a33 = 625.0/8.0
    det = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a12*a21*a33 - a13*a22*a31 - a11*a23*a32
    print()
    print(f"Second system det= {det}")
    
    # Beginning boundary (second)
    b1 = coeff[3] + coeff[2]
    b2 = coeff[4] + coeff[1]
    b3 = coeff[5] + coeff[0]
    print(f"Beginning sums: b1= {b1}  b2= {b2}  b3= {b3}")
    
    abeg = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    cbeg = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    ebeg = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    print(f"abeg= {abeg}  cbeg= {cbeg}  ebeg= {ebeg}")
    
    # End boundary (second)
    b1 = coeff[n-3] + coeff[n-4]
    b2 = coeff[n-2] + coeff[n-5]
    b3 = coeff[n-1] + coeff[n-6]
    print(f"End sums: b1= {b1}  b2= {b2}  b3= {b3}")
    
    aend = (b1*a22*a33 + a12*a23*b3 + a13*b2*a32 - a12*b2*a33 - a13*a22*b3 - b1*a23*a32) / det
    cend = (a11*b2*a33 + b1*a23*a31 + a13*a21*b3 - b1*a21*a33 - a13*b2*a31 - a11*a23*b3) / det
    eend = (a11*a22*b3 + a12*b2*a31 + b1*a21*a32 - a12*a21*b3 - b1*a22*a31 - a11*b2*a32) / det
    print(f"aend= {aend}  cend= {cend}  eend= {eend}")
    
    # First elimination
    print()
    print("=== FIRST ELIMINATION ===")
    alp[0] = 0.0
    bet[0] = ebeg*(2.0 + rhom) - 5.0*fbeg*(3.0 + 1.5*rhom)
    print(f"alp[0]= {alp[0]}  bet[0]= {bet[0]}")
    
    # Check current implementation
    print(f"Loop i=1,n-3 (Python: range(1, {n-3})):")
    for i in range(1, n-3):
        ip1 = i + 1
        alp[ip1] = -1.0 / (rhop + alp[i])
        diff = 5.0*(coeff[i+3] - 4.0*coeff[i+2] + 6.0*coeff[i+1] - 4.0*coeff[i] + coeff[i-1])
        bet[ip1] = alp[ip1] * (bet[i] - diff)
        print(f"i={i}: alp[{ip1}]= {alp[ip1]}  bet[{ip1}]= {bet[ip1]}")
        print(f"  5*(coeff[{i+3}]-4*coeff[{i+2}]+6*coeff[{i+1}]-4*coeff[{i}]+coeff[{i-1}])=")
        print(f"  5*({coeff[i+3]}-4*{coeff[i+2]}+6*{coeff[i+1]}-4*{coeff[i]}+{coeff[i-1]})=")
        print(f"  {diff}")
    
    # Back substitution
    print()
    print("=== BACK SUBSTITUTION ===")
    gam[n-2] = eend*(2.0 + rhom) + 5.0*fend*(3.0 + 1.5*rhom)
    print(f"gam[{n-2}]= {gam[n-2]}")
    
    print(f"Loop i=n-3,0,-1 (Python: range({n-3}, 0, -1)):")
    print(f"DEBUG: Before loop, gam array = {gam}")
    for i in range(n-3, 0, -1):
        print(f"DEBUG: About to calculate gam[{i}] using gam[{i+1}]={gam[i+1]}")
        gam[i] = gam[i+1]*alp[i] + bet[i]
        print(f"i={i}: gam[{i}]=gam[{i+1}]*alp[{i}]+bet[{i}]")
        print(f"       ={gam[i+1]}*{alp[i]}+{bet[i]}={gam[i]}")
    
    # Second elimination
    print()
    print("=== SECOND ELIMINATION ===")
    alp[0] = 0.0
    bet[0] = ebeg - 2.5*5.0*fbeg
    print(f"alp[0]= {alp[0]}  bet[0]= {bet[0]}")
    
    print(f"Loop i=1,n-1 (Python: range(1, {n-1})):")
    for i in range(1, n-1):
        ip1 = i + 1
        alp[ip1] = -1.0 / (rhom + alp[i])
        bet[ip1] = alp[ip1] * (bet[i] - gam[i])
        print(f"i={i}: alp[{ip1}]= {alp[ip1]}  bet[{ip1}]= {bet[ip1]}")
        print(f"  bet[{i}]-gam[{i}]= {bet[i]}-{gam[i]}= {bet[i]-gam[i]}")
    
    # Final e values
    print()
    print("=== FINAL E VALUES ===")
    e = np.zeros(n, dtype=np.float64)
    f = np.zeros(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    
    e[n-1] = eend + 2.5*5.0*fend
    print(f"e[{n-1}]= {e[n-1]}")
    e[n-2] = e[n-1]*alp[n-2] + bet[n-2]
    print(f"e[{n-2}]=e[{n-1}]*alp[{n-2}]+bet[{n-2}]= {e[n-2]}")
    f[n-2] = (e[n-1] - e[n-2]) / 5.0
    print(f"f[{n-2}]=(e[{n-1}]-e[{n-2}])/5= {f[n-2]}")
    e[n-3] = e[n-2]*alp[n-3] + bet[n-3]
    print(f"e[{n-3}]=e[{n-2}]*alp[{n-3}]+bet[{n-3}]= {e[n-3]}")
    f[n-3] = (e[n-2] - e[n-3]) / 5.0
    print(f"f[{n-3}]=(e[{n-2}]-e[{n-3}])/5= {f[n-3]}")
    d[n-2] = dend + 1.5*4.0*eend + 1.5**2*10.0*fend
    print(f"d[{n-2}]=dend+6*eend+22.5*fend= {d[n-2]}")
    
    print()
    print(f"Main loop i=n-3,0,-1 (Python: range({n-3}, 0, -1)):")
    for i in range(n-3, 0, -1):
        e[i] = e[i+1]*alp[i] + bet[i]
        f[i] = (e[i+1] - e[i]) / 5.0
        print(f"i={i}: e[{i}]= {e[i]}  f[{i}]= {f[i]}")
    
    print()
    print("=== FINAL E ARRAY ===")
    print(f"e: {e}")
    
    return e

def main():
    """Run line-by-line comparison"""
    print("LINE-BY-LINE QUINTIC COMPARISON")
    print("=" * 60)
    
    if write_fortran_debug_program():
        fortran_output = run_fortran_debug()
        if fortran_output:
            print("FORTRAN OUTPUT:")
            print(fortran_output)
    
    python_e = trace_python_detailed()
    
    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS:")
    print("Compare the intermediate values between Fortran and Python")
    print("Look for any discrepancies in:")
    print("1. Loop bounds")
    print("2. Array indices")
    print("3. Intermediate calculations")

if __name__ == "__main__":
    main()