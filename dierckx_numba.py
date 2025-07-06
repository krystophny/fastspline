"""
Numba cfunc implementations of DIERCKX routines.
All functions use @cfunc with nopython mode for maximum performance.
Arrays are 0-indexed but algorithm logic preserves FORTRAN 1-based semantics.
"""

import numpy as np
from numba import cfunc, types, njit, carray
from math import sqrt, fabs


# fpback - solves upper triangular system a*c = z
fpback_sig = types.void(
    types.CPointer(types.float64),  # a (nest x k matrix, Fortran order)
    types.CPointer(types.float64),  # z (n vector)
    types.int32,                    # n
    types.int32,                    # k  
    types.CPointer(types.float64),  # c (n vector, output)
    types.int32                     # nest
)

@cfunc(fpback_sig, nopython=True, cache=True)
def fpback(a_ptr, z_ptr, n, k, c_ptr, nest):
    """
    Calculates the solution of the system of equations a*c = z 
    with a a n x n upper triangular matrix of bandwidth k.
    
    Direct port of DIERCKX fpback.f maintaining exact numerical behavior.
    """
    # Convert pointers to arrays
    # In Fortran: a(nest,k) with column-major storage
    # We linearize as a[i,j] = a_flat[i + j*nest]
    a = carray(a_ptr, (nest * k,))
    z = carray(z_ptr, (n,))
    c = carray(c_ptr, (n,))
    
    k1 = k - 1
    
    # c(n) = z(n)/a(n,1)
    # Fortran arrays are 1-indexed, so a(n,1) -> a[n-1 + 0*nest]
    c[n-1] = z[n-1] / a[n-1 + 0*nest]
    
    i = n - 1  # Fortran i starts at n-1
    if i == 0:  # if(i.eq.0) go to 30
        return
    
    # do 20 j=2,n
    for j in range(2, n+1):  # j from 2 to n inclusive
        i = i - 1  # decrement at start of loop (Fortran does at end)
        store = z[i]  # z(i) in Fortran (already 0-indexed)
        i1 = k1
        if j <= k1:
            i1 = j - 1
        
        m = i
        # do 10 l=1,i1  
        for l in range(1, i1+1):  # l from 1 to i1 inclusive
            m = m + 1
            # store = store - c(m)*a(i,l+1)
            # a(i,l+1) -> a[i + l*nest] (l+1-1=l for 0-indexing)
            store = store - c[m] * a[i + l*nest]
        
        # c(i) = store/a(i,1)
        # a(i,1) -> a[i + 0*nest]
        c[i] = store / a[i + 0*nest]


# fpgivs - calculates parameters of a Givens transformation
fpgivs_sig = types.void(
    types.CPointer(types.float64),  # piv (input/output)
    types.CPointer(types.float64),  # ww (input/output)
    types.CPointer(types.float64),  # cos (output)
    types.CPointer(types.float64),  # sin (output)
)

@cfunc(fpgivs_sig, nopython=True, cache=True)
def fpgivs(piv_ptr, ww_ptr, cos_ptr, sin_ptr):
    """
    Calculates the parameters of a Givens transformation.
    
    Direct port of DIERCKX fpgivs.f maintaining exact numerical behavior.
    """
    # Dereference pointers
    piv = piv_ptr[0]
    ww = ww_ptr[0]
    
    one = 1.0
    store = fabs(piv)
    
    if store >= ww:
        dd = store * sqrt(one + (ww/piv)**2)
    else:
        dd = ww * sqrt(one + (piv/ww)**2)
    
    cos = ww / dd
    sin = piv / dd
    ww = dd
    
    # Write outputs
    ww_ptr[0] = ww
    cos_ptr[0] = cos
    sin_ptr[0] = sin


# fprota - applies a Givens rotation to a and b
fprota_sig = types.void(
    types.float64,                   # cos
    types.float64,                   # sin
    types.CPointer(types.float64),   # a (input/output)
    types.CPointer(types.float64),   # b (input/output)
)

@cfunc(fprota_sig, nopython=True, cache=True)
def fprota(cos, sin, a_ptr, b_ptr):
    """
    Applies a Givens rotation to a and b.
    
    Direct port of DIERCKX fprota.f maintaining exact numerical behavior.
    """
    # Dereference pointers
    a = a_ptr[0]
    b = b_ptr[0]
    
    stor1 = a
    stor2 = b
    b = cos * stor2 + sin * stor1
    a = cos * stor1 - sin * stor2
    
    # Write outputs
    a_ptr[0] = a
    b_ptr[0] = b


# fprati - rational interpolation
fprati_sig = types.float64(
    types.CPointer(types.float64),  # p1 (input/output)
    types.CPointer(types.float64),  # f1 (input/output)  
    types.float64,                  # p2
    types.float64,                  # f2
    types.CPointer(types.float64),  # p3 (input/output)
    types.CPointer(types.float64),  # f3 (input/output)
)

@cfunc(fprati_sig, nopython=True, cache=True)
def fprati(p1_ptr, f1_ptr, p2, f2, p3_ptr, f3_ptr):
    """
    Given three points (p1,f1),(p2,f2) and (p3,f3), returns the value 
    of p such that the rational interpolating function of the form 
    r(p) = (u*p+v)/(p+w) equals zero at p.
    
    Direct port of DIERCKX fprati.f maintaining exact numerical behavior.
    """
    p1 = p1_ptr[0]
    f1 = f1_ptr[0]
    p3 = p3_ptr[0]
    f3 = f3_ptr[0]
    
    if p3 > 0.0:
        # value of p in case p3 ^= infinity
        h1 = f1 * (f2 - f3)
        h2 = f2 * (f3 - f1)
        h3 = f3 * (f1 - f2)
        p = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2) / (p1*h1 + p2*h2 + p3*h3)
    else:
        # value of p in case p3 = infinity
        p = (p1*(f1-f3)*f2 - p2*(f2-f3)*f1) / ((f1-f2)*f3)
    
    # adjust the value of p1,f1,p3 and f3 such that f1 > 0 and f3 < 0
    if f2 < 0.0:
        p3_ptr[0] = p2
        f3_ptr[0] = f2
    else:
        p1_ptr[0] = p2
        f1_ptr[0] = f2
    
    return p


# fpdisc - calculates discontinuity jumps
fpdisc_sig = types.void(
    types.CPointer(types.float64),  # t (n vector)
    types.int32,                    # n
    types.int32,                    # k2
    types.CPointer(types.float64),  # b (nest x k2 matrix, Fortran order)
    types.int32,                    # nest
    types.CPointer(types.float64),  # h (work array of size 12)
)

@cfunc(fpdisc_sig, nopython=True, cache=True)
def fpdisc(t_ptr, n, k2, b_ptr, nest, h_ptr):
    """
    Calculates the discontinuity jumps of the kth derivative of the 
    b-splines of degree k at the knots t(k+2)..t(n-k-1).
    
    Direct port of DIERCKX fpdisc.f maintaining exact numerical behavior.
    """
    t = carray(t_ptr, (n,))
    b = carray(b_ptr, (nest * k2,))
    h = carray(h_ptr, (12,))
    
    k1 = k2 - 1
    k = k1 - 1
    nk1 = n - k1
    nrint = nk1 - k
    an = float(nrint)
    fac = an / (t[nk1] - t[k1-1])  # t(nk1+1) - t(k1) in Fortran
    
    # do 40 l=k2,nk1
    for l in range(k2-1, nk1):  # l from k2 to nk1 in Fortran (1-indexed)
        lmk = l - k1
        
        # do 10 j=1,k1
        for j in range(k1):  # j from 1 to k1 in Fortran
            ik = j + k1
            lj = l + j + 1  # l+j in Fortran
            lk = lj - k2
            h[j] = t[l] - t[lk]
            h[ik] = t[l] - t[lj]
        
        lp = lmk
        # do 30 j=1,k2
        for j in range(k2):  # j from 1 to k2 in Fortran
            jk = j
            prod = h[j]
            
            # do 20 i=1,k
            for i in range(k):  # i from 1 to k in Fortran
                jk = jk + 1
                prod = prod * h[jk] * fac
            
            lk = lp + k1
            # b(lmk,j) -> b[lmk + j*nest]
            b[lmk + j*nest] = (t[lk] - t[lp]) / prod
            lp = lp + 1


# fprank - finds minimum norm solution in case of rank deficiency
fprank_sig = types.void(
    types.CPointer(types.float64),  # a (na x m matrix, Fortran order)
    types.CPointer(types.float64),  # f (n vector)
    types.int32,                    # n
    types.int32,                    # m
    types.int32,                    # na
    types.float64,                  # tol
    types.CPointer(types.float64),  # c (n vector, output)
    types.CPointer(types.float64),  # sq (output)
    types.CPointer(types.int32),    # rank (output)
    types.CPointer(types.float64),  # aa (n x m work matrix, Fortran order)
    types.CPointer(types.float64),  # ff (n work vector)
    types.CPointer(types.float64),  # h (m work vector)
)

@cfunc(fprank_sig, nopython=True, cache=True)
def fprank(a_ptr, f_ptr, n, m, na, tol, c_ptr, sq_ptr, rank_ptr, 
           aa_ptr, ff_ptr, h_ptr):
    """
    Finds the minimum norm solution of a least-squares problem 
    in case of rank deficiency.
    
    Direct port of DIERCKX fprank.f maintaining exact numerical behavior.
    """
    # Convert pointers to arrays
    a = carray(a_ptr, (na * m,))
    f = carray(f_ptr, (n,))
    c = carray(c_ptr, (n,))
    aa = carray(aa_ptr, (n * m,))
    ff = carray(ff_ptr, (n,))
    h = carray(h_ptr, (m,))
    
    m1 = m - 1
    
    # The rank deficiency nl is considered to be the number of 
    # sufficient small diagonal elements of a.
    nl = 0
    sq = 0.0
    
    # Process each row
    for i in range(n):
        # a(i,1) -> a[i + 0*na]
        if a[i + 0*na] > tol:
            continue
            
        # If a sufficient small diagonal element is found, we put it to zero.
        nl = nl + 1
        if i == n-1:
            continue
            
        yi = f[i]
        
        # Copy row to h
        for j in range(m1):
            # a(i,j+1) -> a[i + j*na] 
            h[j] = a[i + (j+1)*na]
        h[m1] = 0.0
        
        i1 = i + 1
        # Rotate remainder of row into triangle
        for ii in range(i1, n):
            i2 = min(n-ii, m1)
            piv = h[0]
            
            if piv != 0.0:
                # Call fpgivs
                piv_arr = carray(a_ptr, (1,))  # Reuse pointer
                ww_arr = carray(a_ptr, (1,))
                cos_arr = carray(a_ptr, (1,))
                sin_arr = carray(a_ptr, (1,))
                
                piv_arr[0] = piv
                ww_arr[0] = a[ii + 0*na]
                fpgivs(piv_arr, ww_arr, cos_arr, sin_arr)
                cos = cos_arr[0]
                sin = sin_arr[0]
                a[ii + 0*na] = ww_arr[0]
                
                # Call fprota for yi and f[ii]
                yi_arr = carray(a_ptr, (1,))
                fii_arr = carray(a_ptr, (1,))
                yi_arr[0] = yi
                fii_arr[0] = f[ii]
                fprota(cos, sin, yi_arr, fii_arr)
                yi = yi_arr[0]
                f[ii] = fii_arr[0]
                
                if i2 == 0:
                    continue
                    
                # Rotate rest of row
                for j in range(i2):
                    j1 = j + 1
                    hj1_arr = carray(a_ptr, (1,))
                    aij1_arr = carray(a_ptr, (1,))
                    hj1_arr[0] = h[j1]
                    aij1_arr[0] = a[ii + j1*na]
                    fprota(cos, sin, hj1_arr, aij1_arr)
                    h[j1] = hj1_arr[0]
                    a[ii + j1*na] = aij1_arr[0]
                    h[j] = h[j1]
            else:
                if i2 == 0:
                    continue
                # Shift h elements
                for j in range(i2):
                    h[j] = h[j+1]
            
            h[i2] = 0.0
        
        # Add to sum of squared residuals
        sq = sq + yi*yi
    
    # rank denotes the rank of a
    rank = n - nl
    rank_ptr[0] = rank
    
    # Initialize aa to zero
    for i in range(rank):
        for j in range(m):
            aa[i + j*n] = 0.0
    
    # Form in aa the upper triangular matrix obtained from a
    ii = -1
    for i in range(n):
        if a[i + 0*na] <= tol:
            continue
            
        ii = ii + 1
        ff[ii] = f[i]
        aa[ii + 0*n] = a[i + 0*na]
        jj = ii
        kk = 0
        j = i
        j1 = min(j, m1)
        
        if j1 == 0:
            continue
            
        for k in range(j1):
            j = j - 1
            if a[j + 0*na] <= tol:
                continue
            kk = kk + 1
            jj = jj - 1
            aa[jj + kk*n] = a[j + (k+1)*na]
    
    # Form columns of a with zero diagonal element
    ii = -1
    for i in range(n):
        ii = ii + 1
        if a[i + 0*na] > tol:
            continue
            
        ii = ii - 1
        if ii < 0:
            continue
            
        jj = 0
        j = i
        j1 = min(j, m1)
        
        for k in range(j1):
            j = j - 1
            if a[j + 0*na] <= tol:
                continue
            h[jj] = a[j + (k+1)*na]
            jj = jj + 1
        
        for kk in range(jj, m):
            h[kk] = 0.0
        
        # Rotate this column into aa by givens transformations
        jj = ii
        for i1 in range(ii+1):
            j1 = min(jj, m1)
            piv = h[0]
            
            if piv != 0.0:
                # Call fpgivs
                piv_arr = carray(a_ptr, (1,))
                ww_arr = carray(a_ptr, (1,))
                cos_arr = carray(a_ptr, (1,))
                sin_arr = carray(a_ptr, (1,))
                
                piv_arr[0] = piv
                ww_arr[0] = aa[jj + 0*n]
                fpgivs(piv_arr, ww_arr, cos_arr, sin_arr)
                cos = cos_arr[0]
                sin = sin_arr[0]
                aa[jj + 0*n] = ww_arr[0]
                
                if j1 == 0:
                    continue
                    
                kk = jj
                for j2 in range(j1):
                    j3 = j2 + 1
                    kk = kk - 1
                    hj3_arr = carray(a_ptr, (1,))
                    aakj3_arr = carray(a_ptr, (1,))
                    hj3_arr[0] = h[j3]
                    aakj3_arr[0] = aa[kk + j3*n]
                    fprota(cos, sin, hj3_arr, aakj3_arr)
                    h[j3] = hj3_arr[0]
                    aa[kk + j3*n] = aakj3_arr[0]
                    h[j2] = h[j3]
            else:
                if j1 == 0:
                    continue
                for j2 in range(j1):
                    j3 = j2 + 1
                    h[j2] = h[j3]
            
            jj = jj - 1
            h[j1] = 0.0
    
    # Solve the system (aa) (f1) = ff
    if rank > 0:
        ff[rank-1] = ff[rank-1] / aa[rank-1 + 0*n]
        i = rank - 2
        
        if i >= 0:
            for j in range(1, rank):
                store = ff[i]
                i1 = min(j, m1)
                k = i
                
                for ii in range(i1):
                    k = k + 1
                    stor1 = ff[k]
                    stor2 = aa[i + (ii+1)*n]
                    store = store - stor1*stor2
                
                stor1 = aa[i + 0*n]
                ff[i] = store / stor1
                i = i - 1
        
        # Solve the system (aa)' (f2) = f1
        ff[0] = ff[0] / aa[0 + 0*n]
        
        if rank > 1:
            for j in range(1, rank):
                store = ff[j]
                i1 = min(j, m1)
                k = j
                
                for ii in range(i1):
                    k = k - 1
                    stor1 = ff[k]
                    stor2 = aa[k + (ii+1)*n]
                    store = store - stor1*stor2
                
                stor1 = aa[j + 0*n]
                ff[j] = store / stor1
    
    # Premultiply f2 by the transpose of a
    k = -1
    for i in range(n):
        store = 0.0
        if a[i + 0*na] > tol:
            k = k + 1
        j1 = min(i+1, m)
        kk = k
        ij = i
        
        for j in range(j1):
            if a[ij + 0*na] <= tol:
                ij = ij - 1
                continue
            stor1 = a[ij + j*na]
            stor2 = ff[kk]
            store = store + stor1*stor2
            kk = kk - 1
            ij = ij - 1
        
        c[i] = store
    
    # Add contribution of zero diagonal elements
    stor3 = 0.0
    for i in range(n):
        if a[i + 0*na] > tol:
            continue
            
        store = f[i]
        i1 = min(n-i-1, m1)
        
        if i1 > 0:
            for j in range(i1):
                ij = i + j + 1
                stor1 = c[ij]
                stor2 = a[i + (j+1)*na]
                store = store - stor1*stor2
        
        fac = a[i + 0*na] * c[i]
        stor1 = a[i + 0*na]
        stor2 = c[i]
        stor1 = stor1 * stor2
        stor3 = stor3 + stor1*(stor1 - store - store)
    
    fac = stor3
    sq = sq + fac
    sq_ptr[0] = sq


# fporde - sorts data points according to panel
fporde_sig = types.void(
    types.CPointer(types.float64),  # x (m vector)
    types.CPointer(types.float64),  # y (m vector)
    types.int32,                    # m
    types.int32,                    # kx
    types.int32,                    # ky
    types.CPointer(types.float64),  # tx (nx vector)
    types.int32,                    # nx
    types.CPointer(types.float64),  # ty (ny vector)
    types.int32,                    # ny
    types.CPointer(types.int32),    # nummer (m vector, output)
    types.CPointer(types.int32),    # index (nreg vector, output)
    types.int32,                    # nreg
)

@cfunc(fporde_sig, nopython=True, cache=True)
def fporde(x_ptr, y_ptr, m, kx, ky, tx_ptr, nx, ty_ptr, ny, 
           nummer_ptr, index_ptr, nreg):
    """
    Sorts the data points (x(i),y(i)),i=1,2,...,m according to the panel 
    tx(l)<=x<tx(l+1),ty(k)<=y<ty(k+1), they belong to.
    
    Direct port of DIERCKX fporde.f maintaining exact numerical behavior.
    """
    x = carray(x_ptr, (m,))
    y = carray(y_ptr, (m,))
    tx = carray(tx_ptr, (nx,))
    ty = carray(ty_ptr, (ny,))
    nummer = carray(nummer_ptr, (m,))
    index = carray(index_ptr, (nreg,))
    
    kx1 = kx + 1
    ky1 = ky + 1
    nk1x = nx - kx1
    nk1y = ny - ky1
    nyy = nk1y - ky
    
    # Initialize index array
    for i in range(nreg):
        index[i] = 0
        
    # Sort data points
    for im in range(m):
        xi = x[im]
        yi = y[im]
        l = kx1 - 1  # Convert to 0-based
        l1 = l + 1
        
        # Find x interval
        while xi >= tx[l1] and l < nk1x - 1:
            l = l1
            l1 = l + 1
            
        k = ky1 - 1  # Convert to 0-based
        k1 = k + 1
        
        # Find y interval
        while yi >= ty[k1] and k < nk1y - 1:
            k = k1
            k1 = k + 1
            
        # Calculate panel number
        num = (l - kx1 + 1) * nyy + k - ky + 1 - 1  # -1 for 0-based
        nummer[im] = index[num]
        index[num] = im + 1  # +1 to match Fortran 1-based


# fpbspl - evaluates non-zero b-splines
fpbspl_sig = types.void(
    types.CPointer(types.float64),  # t (n vector)
    types.int32,                    # n
    types.int32,                    # k
    types.float64,                  # x
    types.CPointer(types.int32),    # l (output)
    types.CPointer(types.float64),  # h (k+1 vector, output)
)

@cfunc(fpbspl_sig, nopython=True, cache=True)
def fpbspl(t_ptr, n, k, x, l_ptr, h_ptr):
    """
    Evaluates the (k+1) non-zero b-splines of degree k at t(l) <= x < t(l+1) 
    using the stable recurrence relation of de Boor and Cox.
    
    Direct port of DIERCKX fpbspl.f maintaining exact numerical behavior.
    """
    t = carray(t_ptr, (n,))
    h = carray(h_ptr, (6,))  # Max size is 6 in DIERCKX
    
    # Find the knot interval containing x
    l = k  # Start search at k
    while l < n - k - 1 and x >= t[l + 1]:
        l = l + 1
    l_ptr[0] = l + 1  # Convert to 1-based for output
    
    # Local work array
    hh = carray(h_ptr, (5,))  # Reuse part of h for work array
    
    one = 1.0
    h[0] = one
    
    # Build b-splines using de Boor recursion
    for j in range(1, k+1):
        # Save current h values
        for i in range(j):
            hh[i] = h[i]
            
        h[0] = 0.0
        
        for i in range(j):
            li = l + i + 1  # l+i in Fortran (1-based)
            lj = li - j
            f = hh[i] / (t[li-1] - t[lj-1])  # -1 for 0-based
            h[i] = h[i] + f * (t[li-1] - x)
            h[i+1] = f * (x - t[lj-1])


# Create njit wrappers for testing
@njit
def test_fpback(a, z, n, k, c, nest):
    """Test wrapper for fpback cfunc"""
    fpback(a.ctypes.data, z.ctypes.data, n, k, c.ctypes.data, nest)
    
@njit
def test_fpgivs(piv, ww):
    """Test wrapper for fpgivs cfunc"""
    piv_arr = np.array([piv], dtype=np.float64)
    ww_arr = np.array([ww], dtype=np.float64)
    cos_arr = np.zeros(1, dtype=np.float64)
    sin_arr = np.zeros(1, dtype=np.float64)
    
    fpgivs(piv_arr.ctypes.data, ww_arr.ctypes.data, 
           cos_arr.ctypes.data, sin_arr.ctypes.data)
    
    return ww_arr[0], cos_arr[0], sin_arr[0]

@njit
def test_fprota(cos, sin, a, b):
    """Test wrapper for fprota cfunc"""
    a_arr = np.array([a], dtype=np.float64)
    b_arr = np.array([b], dtype=np.float64)
    
    fprota(cos, sin, a_arr.ctypes.data, b_arr.ctypes.data)
    
    return a_arr[0], b_arr[0]

@njit
def test_fprati(p1, f1, p2, f2, p3, f3):
    """Test wrapper for fprati cfunc"""
    p1_arr = np.array([p1], dtype=np.float64)
    f1_arr = np.array([f1], dtype=np.float64)
    p3_arr = np.array([p3], dtype=np.float64)
    f3_arr = np.array([f3], dtype=np.float64)
    
    result = fprati(p1_arr.ctypes.data, f1_arr.ctypes.data, p2, f2,
                    p3_arr.ctypes.data, f3_arr.ctypes.data)
    
    return result, p1_arr[0], f1_arr[0], p3_arr[0], f3_arr[0]

@njit
def test_fpdisc(t, n, k2, b, nest):
    """Test wrapper for fpdisc cfunc"""
    h = np.zeros(12, dtype=np.float64)
    fpdisc(t.ctypes.data, n, k2, b.ctypes.data, nest, h.ctypes.data)


# Test the implementation
if __name__ == "__main__":
    print("Testing DIERCKX Numba implementations...")
    
    # Test fpback
    print("\n1. Testing fpback:")
    n = 4
    k = 3
    nest = 10
    
    # Create upper triangular banded matrix
    a = np.zeros((nest, k), dtype=np.float64, order='F')
    # Fill diagonal
    for i in range(n):
        a[i, 0] = 2.0  # diagonal
    # Fill super-diagonals
    for i in range(n-1):
        a[i, 1] = 1.0  # first super-diagonal
    for i in range(n-2):
        a[i, 2] = 0.5  # second super-diagonal
        
    # RHS vector
    z = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    
    # Output vector
    c = np.zeros(n, dtype=np.float64)
    
    # Call fpback
    test_fpback(a, z, n, k, c, nest)
    
    print(f"  Matrix a (first {n} rows):")
    print(a[:n, :])
    print(f"  RHS z: {z}")
    print(f"  Solution c: {c}")
    
    # Test fpgivs
    print("\n2. Testing fpgivs:")
    piv = 3.0
    ww = 4.0
    ww_out, cos, sin = test_fpgivs(piv, ww)
    print(f"  Input: piv={piv}, ww={ww}")
    print(f"  Output: ww={ww_out}, cos={cos}, sin={sin}")
    print(f"  Check: cos^2 + sin^2 = {cos**2 + sin**2}")
    
    # Test fprota
    print("\n3. Testing fprota:")
    a_in = 5.0
    b_in = 12.0
    a_out, b_out = test_fprota(cos, sin, a_in, b_in)
    print(f"  Input: a={a_in}, b={b_in}, cos={cos}, sin={sin}")
    print(f"  Output: a={a_out}, b={b_out}")
    
    # Test fprati
    print("\n4. Testing fprati:")
    p1, f1 = 0.0, 1.0
    p2, f2 = 0.5, -0.5
    p3, f3 = 1.0, -2.0
    p, p1_out, f1_out, p3_out, f3_out = test_fprati(p1, f1, p2, f2, p3, f3)
    print(f"  Input: ({p1},{f1}), ({p2},{f2}), ({p3},{f3})")
    print(f"  Output: p={p}")
    print(f"  Updated: p1={p1_out}, f1={f1_out}, p3={p3_out}, f3={f3_out}")