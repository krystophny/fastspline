#!/bin/bash
# Build DIERCKX f2py wrapper with correct interfaces

set -e

echo "Building DIERCKX f2py wrapper..."

# Create temporary directory for modified sources
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy all FORTRAN files
cp thirdparty/dierckx/*.f $TEMP_DIR/

# Add f2py directives to specific files
cat > $TEMP_DIR/fpgivs.f << 'EOF'
      subroutine fpgivs(piv,ww,cos,sin)
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
EOF

cat > $TEMP_DIR/fprati.f << 'EOF'
      real function fprati(p1,f1,p2,f2,p3,f3)
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
EOF

# Add directives for fpback
sed -i '1a\
cf2py intent(in) :: a\
cf2py intent(in) :: z\
cf2py intent(in) :: n\
cf2py intent(in) :: k\
cf2py intent(out) :: c\
cf2py intent(in) :: nest\
cf2py depend(n) :: c' $TEMP_DIR/fpback.f

# Add directives for fprota
sed -i '1a\
cf2py intent(in) :: cos\
cf2py intent(in) :: sin\
cf2py intent(in,out) :: a\
cf2py intent(in,out) :: b' $TEMP_DIR/fprota.f

# Add directives for fpbspl
sed -i '1a\
cf2py intent(in) :: t\
cf2py intent(in) :: n\
cf2py intent(in) :: k\
cf2py intent(in) :: x\
cf2py intent(in) :: l\
cf2py intent(out) :: h\
cf2py depend(k) :: h' $TEMP_DIR/fpbspl.f

# Build the module
python -m numpy.f2py -c $TEMP_DIR/*.f -m dierckx_f2py --f77flags="-O3" --f90flags="-O3"

echo "Build complete! Module saved as dierckx_f2py.*.so"