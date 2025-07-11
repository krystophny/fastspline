# Fortran to Python Array Indexing Translation Notes

## Call Hierarchy
1. `bispev` - Top level, handles input validation and workspace allocation
2. `fpbisp` - Evaluates tensor product B-splines, calls fpbspl for each direction  
3. `fpbspl` - Evaluates non-zero B-splines at a single point using de Boor's algorithm

## Key Indexing Patterns

### fpbspl.f
- Arrays: `t(n)`, `h(20)`, `hh(19)`
- Loop indices: `j=1,k` and `i=1,j`
- Array access: `t(li)` where `li = l+i`
- Python translation: 
  - `t[li-1]` where `li = l+i-1` (adjusting both index and computation)
  - Or better: use 0-based throughout

### fpbisp.f
- Arrays:
  - `tx(nx)`, `ty(ny)` - knot vectors
  - `c((nx-kx-1)*(ny-ky-1))` - coefficients (1D array)
  - `wx(mx,kx+1)`, `wy(my,ky+1)` - workspace for B-spline values
  - `lx(mx)`, `ly(my)` - indices for coefficient lookup
- Key patterns:
  - `l = kx1` starts at kx+1 (Fortran 1-based)
  - `lx(i) = l-kx1` stores 0-based offset
  - Coefficient indexing: `c(l2)` where `l2 = l1 + 1` in inner loop
  
### bispev.f  
- Workspace calculation: `iw = mx*(kx+1)+1`
- Calls fpbisp with: `wrk(1)`, `wrk(iw)` - splitting workspace array

## Translation Strategy

1. Keep all loop indices 0-based in Python
2. Adjust array declarations:
   - Fortran: `real*8 h(6)` → Python: `h = np.zeros(6)`
3. Translate array access:
   - Fortran: `array(i)` → Python: `array[i-1]`
   - But rewrite loops to be naturally 0-based
4. Handle workspace splitting carefully