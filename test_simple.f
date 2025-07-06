      subroutine test_givs(piv,ww,cos,sin)
cf2py intent(in,out) :: piv
cf2py intent(in,out) :: ww
cf2py intent(out) :: cos
cf2py intent(out) :: sin
      real piv,ww,cos,sin
      real dd,one,store
      real abs,sqrt
      one = 1.0
      store = abs(piv)
      if(store.ge.ww) dd = store*sqrt(one+(ww/piv)**2)
      if(store.lt.ww) dd = ww*sqrt(one+(piv/ww)**2)
      cos = ww/dd
      sin = piv/dd
      ww = dd
      return
      end