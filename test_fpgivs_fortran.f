      program test_fpgivs
      real piv, ww, cos, sin
      
      piv = 3.0
      ww = 4.0
      
      print *, 'Before fpgivs: piv=', piv, ' ww=', ww
      
      call fpgivs(piv, ww, cos, sin)
      
      print *, 'After fpgivs: piv=', piv, ' ww=', ww
      print *, 'cos=', cos, ' sin=', sin
      print *, 'cos^2 + sin^2 =', cos**2 + sin**2
      
      end program test_fpgivs