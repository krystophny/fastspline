#!/usr/bin/env python
"""
Rebuild the DIERCKX f2py wrapper with correct interface
"""

import os
import subprocess
import sys

# Create temporary versions of the FORTRAN files with f2py directives
fortran_files = {
    'fpgivs.f': """      subroutine fpgivs(piv,ww,cos,sin)
cf2py intent(in,out) :: piv
cf2py intent(in,out) :: ww  
cf2py intent(out) :: cos
cf2py intent(out) :: sin
c  subroutine fpgivs calculates the parameters of a givens
c  transformation .
c  ..
c  ..scalar arguments..
      real piv,ww,cos,sin
c  ..local scalars..
      real dd,one,store
c  ..function references..
      real abs,sqrt
c  ..
      one = 0.1e+01
      store = abs(piv)
      if(store.ge.ww) dd = store*sqrt(one+(ww/piv)**2)
      if(store.lt.ww) dd = ww*sqrt(one+(piv/ww)**2)
      cos = ww/dd
      sin = piv/dd
      ww = dd
      return
      end
""",
    'fprati.f': """      real function fprati(p1,f1,p2,f2,p3,f3)
cf2py real :: fprati
c  given three points (p1,f1),(p2,f2) and (p3,f3), function fprati
c  gives the value of p such that the rational interpolating function
c  of the form r(p) = (u*p+v)/(p+w) equals zero at p.
c  ..
c  ..scalar arguments..
      real p1,f1,p2,f2,p3,f3
c  ..local scalars..
      real h1,h2,h3,p
c  ..
      if(p3.gt.0.) go to 10
c  value of p in case p3 = infinity.
      p = (p1*(f1-f3)*f2-p2*(f2-f3)*f1)/((f1-f2)*f3)
      go to 20
c  value of p in case p3 ^= infinity.
  10  h1 = f1*(f2-f3)
      h2 = f2*(f3-f1)
      h3 = f3*(f1-f2)
      p = -(p1*p2*h3+p2*p3*h1+p3*p1*h2)/(p1*h1+p2*h2+p3*h3)
c  adjust the value of p1,f1,p3 and f3 such that f1 > 0 and f3 < 0.
  20  if(f2.lt.0.) go to 30
      p1 = p2
      f1 = f2
      go to 40
  30  p3 = p2
      f3 = f2
  40  fprati = p
      return
      end
""",
    'fpback.f': """      subroutine fpback(a,z,n,k,c,nest)
cf2py intent(in) :: a
cf2py intent(in) :: z
cf2py intent(in) :: n
cf2py intent(in) :: k
cf2py intent(out) :: c
cf2py intent(in) :: nest
cf2py depend(n) :: c
c  subroutine fpback calculates the solution of the system of
c  equations a*c = z with a a n x n upper triangular matrix
c  of bandwidth k.
c  ..
c  ..scalar arguments..
      integer n,k,nest
c  ..array arguments..
      real a(nest,k),z(n),c(n)
c  ..local scalars..
      real store
      integer i,i1,j,k1,l,m
c  ..
      k1 = k-1
      c(n) = z(n)/a(n,1)
      i = n-1
      if(i.eq.0) go to 30
      do 20 j=2,n
        store = z(i)
        i1 = k1
        if(j.le.k1) i1 = j-1
        m = i
        do 10 l=1,i1
          m = m+1
          store = store-c(m)*a(i,l+1)
  10    continue
        c(i) = store/a(i,1)
        i = i-1
  20  continue
  30  return
      end
""",
    'fprota.f': """      subroutine fprota(cos,sin,a,b)
cf2py intent(in) :: cos
cf2py intent(in) :: sin
cf2py intent(in,out) :: a
cf2py intent(in,out) :: b
c subroutine fprota performs one rotation in the jacobi-iteration
c for the least-squares matrix a, by applying a givens transformation.
c  ..
c  ..scalar arguments..
      real cos,sin,a,b
c  ..local scalars..
      real stor1,stor2
c  ..
      stor1 = a
      stor2 = b
      b = cos*stor2+sin*stor1
      a = cos*stor1-sin*stor2
      return
      end
""",
    'fpbspl.f': """      subroutine fpbspl(t,n,k,x,l,h)
cf2py intent(in) :: t
cf2py intent(in) :: n  
cf2py intent(in) :: k
cf2py intent(in) :: x
cf2py intent(in) :: l
cf2py intent(out) :: h
cf2py depend(k) :: h
c  subroutine fpbspl evaluates the (k+1) non-zero b-splines of
c  degree k at t(l) <= x < t(l+1) using the stable recurrence
c  relation of de boor and cox.
c  ..
c  ..scalar arguments..
      real x
      integer n,k,l
c  ..array arguments..
      real t(n),h(6)
c  ..local scalars..
      real f,one
      integer i,j,li,lj
c  ..local arrays..
      real hh(5)
c  ..
      one = 0.1e+01
      h(1) = one
      do 20 j=1,k
        do 10 i=1,j
          hh(i) = h(i)
  10    continue
        h(1) = 0.
        do 20 i=1,j
          li = l+i
          lj = li-j
          f = hh(i)/(t(li)-t(lj))
          h(i) = h(i)+f*(t(li)-x)
          h(i+1) = f*(x-t(lj))
  20  continue
      return
      end
"""
}

# Create temp directory
os.makedirs('temp_fortran', exist_ok=True)

# Write modified FORTRAN files
for filename, content in fortran_files.items():
    with open(f'temp_fortran/{filename}', 'w') as f:
        f.write(content)

# Copy remaining FORTRAN files
import shutil
fortran_dir = 'thirdparty/dierckx'
for f in os.listdir(fortran_dir):
    if f.endswith('.f') and f not in fortran_files:
        shutil.copy(os.path.join(fortran_dir, f), f'temp_fortran/{f}')

# Build f2py wrapper
print("Building DIERCKX f2py wrapper with corrected interfaces...")

# Get all .f files
import glob
fortran_files_list = sorted(glob.glob('temp_fortran/*.f'))
print(f"Found {len(fortran_files_list)} FORTRAN files")

cmd = [
    sys.executable, '-m', 'numpy.f2py',
    '-c'
] + fortran_files_list + [
    '-m', 'dierckx_f2py_fixed',
    '--f77flags=-O3',
    '--f90flags=-O3'
]

try:
    subprocess.run(cmd, check=True)
    print("Build successful!")
    
    # Test the wrapper
    print("\nTesting the corrected wrapper...")
    import dierckx_f2py_fixed
    
    # Test fpgivs
    result = dierckx_f2py_fixed.fpgivs(3.0, 4.0)
    print(f"fpgivs(3.0, 4.0) = {result}")
    print(f"Expected: (3.0, 5.0, 0.8, 0.6)")
    
    # Test fprati
    p = dierckx_f2py_fixed.fprati(1.0, 2.0, 2.0, 1.0, 3.0, -1.0)
    print(f"\nfprati(1,2,2,1,3,-1) = {p}")
    print(f"Expected: 2.6")
    
except subprocess.CalledProcessError as e:
    print(f"Build failed: {e}")
    
finally:
    # Cleanup
    shutil.rmtree('temp_fortran', ignore_errors=True)