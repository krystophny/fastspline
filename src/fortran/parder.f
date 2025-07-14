      recursive subroutine parder(tx,nx,ty,ny,c,kx,ky,nux,nuy,x,mx,y,my,
     * z,wrk,lwrk,iwrk,kwrk,ier)
      implicit none
c  given the b-spline representation of a bivariate spline s(x,y) of
c  degree kx in x and ky in y, parder calculates the partial derivative
c  of order (nux,nuy) of s(x,y) at the points (x(i),y(j))
c  i=1,2,...,mx; j=1,2,...,my.
c
c calling sequence:
c     call parder(tx,nx,ty,ny,c,kx,ky,nux,nuy,x,mx,y,my,z,wrk,lwrk,
c    * iwrk,kwrk,ier)
c
c  input parameters:
c    tx    : real array, length nx, which contains the position of the
c            knots in the x-direction.
c    nx    : integer, giving the total number of knots in the x-direction
c    ty    : real array, length ny, which contains the position of the
c            knots in the y-direction.
c    ny    : integer, giving the total number of knots in the y-direction
c    c     : real array, length (nx-kx-1)*(ny-ky-1), which contains the
c            b-spline coefficients.
c    kx,ky : integer values, giving the degrees of the spline.
c    nux   : integer, specifying the order of the partial derivative
c            with respect to x.
c    nuy   : integer, specifying the order of the partial derivative
c            with respect to y.
c    x     : real array of dimension (mx).
c    mx    : integer, giving the number of points in the x-direction.
c    y     : real array of dimension (my).
c    my    : integer, giving the number of points in the y-direction.
c    wrk   : real array of dimension lwrk. used as workspace.
c    lwrk  : integer, giving the dimension of the array wrk.
c    iwrk  : integer array of dimension kwrk. used as workspace.
c    kwrk  : integer, giving the dimension of the array iwrk.
c
c  output parameters:
c    z     : real array of dimension (mx*my).
c            z(my*(i-1)+j) gives the value of the partial derivative
c            at the point (x(i),y(j)), i=1,2,...,mx; j=1,2,...,my.
c    ier   : integer error flag
c      ier=0 : normal return
c      ier=10: invalid input data (see restrictions)
c
c  restrictions:
c    0 <= nux < kx, 0 <= nuy < ky
c    mx >= 1, my >= 1
c    lwrk >= mx*(kx+1-nux)+my*(ky+1-nuy)
c    kwrk >= mx+my
c    tx(kx+1) <= x(i) <= tx(nx-kx), i=1,2,...,mx
c    ty(ky+1) <= y(j) <= ty(ny-ky), j=1,2,...,my
c
c  other subroutines required: fpbisp,fpbspl
c
c  author: paul dierckx
c          dept. computer science, k.u.leuven
c          celestijnenlaan 200a, b-3001 heverlee, belgium.
c          e-mail : Paul.Dierckx@cs.kuleuven.ac.be
c
c  latest update : march 1987
c
c  ..scalar arguments..
      integer nx,ny,kx,ky,nux,nuy,mx,my,lwrk,kwrk,ier
c  ..array arguments..
      integer iwrk(kwrk)
      real*8 tx(nx),ty(ny),c((nx-kx-1)*(ny-ky-1)),x(mx),y(my),
     * z(mx*my),wrk(lwrk)
c  ..local scalars..
      integer i,iwx,iwy,j,kkx,kky,kx1,ky1,lx,ly,lwest,l1,l2,m,m0,m1,
     * nc,nkx1,nky1,nxx,nyy
      real*8 ak,fac
c  ..
c  before starting computations a data check is made. if the input data
c  are invalid control is immediately repassed to the calling program.
      ier = 10
      kx1 = kx+1
      ky1 = ky+1
      nkx1 = nx-kx1
      nky1 = ny-ky1
      nc = nkx1*nky1
      if(nux.lt.0 .or. nux.ge.kx) go to 400
      if(nuy.lt.0 .or. nuy.ge.ky) go to 400
      lwest = (kx1-nux)*mx+(ky1-nuy)*my
      if(lwrk.lt.lwest) go to 400
      if(kwrk.lt.(mx+my)) go to 400
      if(mx-1) 400,30,10
  10  do 20 i=2,mx
        if(x(i).lt.x(i-1)) go to 400
  20  continue
  30  if(my-1) 400,60,40
  40  do 50 j=2,my
        if(y(j).lt.y(j-1)) go to 400
  50  continue
  60  ier = 0
c  if nux=0 the partial derivative is not computed with respect to x.
      if(nux.eq.0) go to 70
c  if nux>0 we compute the partial derivative with respect to x.
      kkx = kx-nux
      nxx = nx-nux
      do 65 i=1,mx
        if(x(i).lt.tx(kx1) .or. x(i).gt.tx(nkx1)) go to 400
  65  continue
c  if nuy=0 the partial derivative is not computed with respect to y.
  70  if(nuy.eq.0) go to 80
c  if nuy>0 we compute the partial derivative with respect to y.
      kky = ky-nuy
      nyy = ny-nuy
      do 75 j=1,my
        if(y(j).lt.ty(ky1) .or. y(j).gt.ty(nky1)) go to 400
  75  continue
c  the partial derivative is computed.
  80  m = 0
      do 300 i=1,mx
        l = kx1
        l1 = l+1
        if(nux.eq.0) go to 100
        ak = x(i)
        nkx1 = nx-nux
        kx1 = kx+1
        tb = tx(nux+1)
        te = tx(nkx1)
        if(ak.lt.tb) ak = tb
        if(ak.gt.te) ak = te
c  search for knot interval t(l) <= ak < t(l+1)
        l = nux
        l1 = l+1
  85    if(ak.lt.tx(l1) .or. l.eq.nkx1) go to 90
        l = l1
        l1 = l+1
        go to 85
  90    if(ak.eq.tx(l1)) l = l1
        iwx = (i-1)*(kx1-nux)+1
        call fpbspl(tx,nx,kx,ak,nux,l,wrk(iwx))
c  if nuy=0 the partial derivative is not computed with respect to y.
 100    if(nuy.eq.0) go to 130
c  if nuy>0 we compute the partial derivative with respect to y.
        do 120 j=1,my
          l = ky1
          l1 = l+1
          ak = y(j)
          nky1 = ny-nuy
          ky1 = ky+1
          tb = ty(nuy+1)
          te = ty(nky1)
          if(ak.lt.tb) ak = tb
          if(ak.gt.te) ak = te
c  search for knot interval t(l) <= ak < t(l+1)
          l = nuy
          l1 = l+1
 105      if(ak.lt.ty(l1) .or. l.eq.nky1) go to 110
          l = l1
          l1 = l+1
          go to 105
 110      if(ak.eq.ty(l1)) l = l1
          iwy = (j-1)*(ky1-nuy)+1
          call fpbspl(ty,ny,ky,ak,nuy,l,wrk(iwy))
c  compute the partial derivative.
          iwrk(i) = l-nuy
          iwrk(mx+j) = l-nuy
          m = m+1
          z(m) = 0.
          l2 = l-nuy
          do 115 lx=1,kx1-nux
            l1 = l2
            do 115 ly=1,ky1-nuy
              l1 = l1+1
              z(m) = z(m)+c(l1)*wrk(iwx+lx-1)*wrk(iwy+ly-1)
 115      l2 = l2+nky1
 120    continue
        go to 300
c  if nuy=0 the partial derivative is only computed with respect to x.
 130    do 200 j=1,my
          l = ky1
          l1 = l+1
          ak = y(j)
          if(ak.lt.ty(ky1) .or. ak.gt.ty(nky1)) go to 400
c  search for knot interval t(l) <= ak < t(l+1)
          l = ky
          l1 = l+1
 140      if(ak.lt.ty(l1) .or. l.eq.nky1) go to 150
          l = l1
          l1 = l+1
          go to 140
 150      if(ak.eq.ty(l1)) l = l1
          iwy = (j-1)*ky1+1
          call fpbspl(ty,ny,ky,ak,0,l,wrk(iwy))
c  compute the partial derivative.
          iwrk(mx+j) = l-ky
          m = m+1
          z(m) = 0.
          l2 = l-ky
          do 160 lx=1,kx1-nux
            l1 = l2
            do 160 ly=1,ky1
              l1 = l1+1
              z(m) = z(m)+c(l1)*wrk(iwx+lx-1)*wrk(iwy+ly-1)
 160      l2 = l2+nky1
 200    continue
 300  continue
 400  return
      end