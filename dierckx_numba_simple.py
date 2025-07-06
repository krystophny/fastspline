"""
Simplified DIERCKX Numba implementations using regular njit functions.
All functions maintain exact numerical behavior of FORTRAN originals.
"""

import numpy as np
from numba import njit
from math import sqrt, fabs


@njit(cache=True)
def fpback_njit(a, z, n, k, c, nest):
    """
    Calculates the solution of the system of equations a*c = z 
    with a a n x n upper triangular matrix of bandwidth k.
    
    Direct port of DIERCKX fpback.f maintaining exact numerical behavior.
    a should be (nest, k) Fortran-order array
    """
    k1 = k - 1
    
    # c(n) = z(n)/a(n,1)
    c[n-1] = z[n-1] / a[n-1, 0]
    
    i = n - 1  # Fortran i starts at n-1
    if i == 0:  # if(i.eq.0) go to 30
        return
    
    # do 20 j=2,n
    for j in range(2, n+1):  # j from 2 to n inclusive
        i = i - 1  # decrement at start of loop
        store = z[i]
        i1 = k1
        if j <= k1:
            i1 = j - 1
        
        m = i
        # do 10 l=1,i1  
        for l in range(1, i1+1):  # l from 1 to i1 inclusive
            m = m + 1
            # store = store - c(m)*a(i,l+1)
            store = store - c[m] * a[i, l]
        
        # c(i) = store/a(i,1)
        c[i] = store / a[i, 0]


@njit(cache=True)
def fpgivs_njit(piv, ww):
    """
    Calculates the parameters of a Givens transformation.
    Returns: ww_new, cos, sin
    """
    one = 1.0
    store = fabs(piv)
    
    if store >= ww:
        dd = store * sqrt(one + (ww/piv)**2)
    else:
        dd = ww * sqrt(one + (piv/ww)**2)
    
    cos = ww / dd
    sin = piv / dd
    ww = dd
    
    return ww, cos, sin


@njit(cache=True)
def fprota_njit(cos, sin, a, b):
    """
    Applies a Givens rotation to a and b.
    Returns: a_new, b_new
    """
    stor1 = a
    stor2 = b
    b = cos * stor2 + sin * stor1
    a = cos * stor1 - sin * stor2
    
    return a, b


@njit(cache=True)
def fprati_njit(p1, f1, p2, f2, p3, f3):
    """
    Rational interpolation function.
    Returns: p, p1_new, f1_new, p3_new, f3_new
    """
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
        p3 = p2
        f3 = f2
    else:
        p1 = p2
        f1 = f2
    
    return p, p1, f1, p3, f3


@njit(cache=True)
def fpdisc_njit(t, n, k2, b, nest):
    """
    Calculates the discontinuity jumps of the kth derivative of the 
    b-splines of degree k at the knots t(k+2)..t(n-k-1).
    b should be (nest, k2) Fortran-order array
    """
    # Local array h - max size 12 as in Fortran
    h = np.zeros(12, dtype=np.float64)
    
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
            # b(lmk,j) -> b[lmk, j]
            b[lmk, j] = (t[lk] - t[lp]) / prod
            lp = lp + 1


@njit(cache=True)
def fprank_njit(a, f, n, m, na, tol, c, aa, ff, h):
    """
    Finds the minimum norm solution of a least-squares problem 
    in case of rank deficiency.
    Returns: sq, rank
    a should be (na, m) Fortran-order array
    aa should be (n, m) Fortran-order array
    """
    m1 = m - 1
    
    # The rank deficiency nl
    nl = 0
    sq = 0.0
    
    # Process each row
    for i in range(n):
        if a[i, 0] > tol:
            continue
            
        # Small diagonal element found
        nl = nl + 1
        if i == n-1:
            continue
            
        yi = f[i]
        
        # Copy row to h
        for j in range(m1):
            h[j] = a[i, j+1]
        h[m1] = 0.0
        
        i1 = i + 1
        # Rotate remainder of row into triangle
        for ii in range(i1, n):
            i2 = min(n-ii, m1)
            piv = h[0]
            
            if piv != 0.0:
                # Call fpgivs
                ww, cos, sin = fpgivs_njit(piv, a[ii, 0])
                a[ii, 0] = ww
                
                # Apply rotation to yi and f[ii]
                yi, f[ii] = fprota_njit(cos, sin, yi, f[ii])
                
                if i2 == 0:
                    continue
                    
                # Rotate rest of row
                for j in range(i2):
                    j1 = j + 1
                    h[j1], a[ii, j1] = fprota_njit(cos, sin, h[j1], a[ii, j1])
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
    
    # Initialize aa to zero
    for i in range(rank):
        for j in range(m):
            aa[i, j] = 0.0
    
    # Form in aa the upper triangular matrix obtained from a
    ii = -1
    for i in range(n):
        if a[i, 0] <= tol:
            continue
            
        ii = ii + 1
        ff[ii] = f[i]
        aa[ii, 0] = a[i, 0]
        jj = ii
        kk = 0
        j = i
        j1 = min(j, m1)
        
        if j1 == 0:
            continue
            
        for k in range(j1):
            j = j - 1
            if a[j, 0] <= tol:
                continue
            kk = kk + 1
            jj = jj - 1
            aa[jj, kk] = a[j, k+1]
    
    # Form columns of a with zero diagonal element
    ii = -1
    for i in range(n):
        ii = ii + 1
        if a[i, 0] > tol:
            continue
            
        ii = ii - 1
        if ii < 0:
            continue
            
        jj = 0
        j = i
        j1 = min(j, m1)
        
        for k in range(j1):
            j = j - 1
            if a[j, 0] <= tol:
                continue
            h[jj] = a[j, k+1]
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
                ww, cos, sin = fpgivs_njit(piv, aa[jj, 0])
                aa[jj, 0] = ww
                
                if j1 == 0:
                    continue
                    
                kk = jj
                for j2 in range(j1):
                    j3 = j2 + 1
                    kk = kk - 1
                    h[j3], aa[kk, j3] = fprota_njit(cos, sin, h[j3], aa[kk, j3])
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
        ff[rank-1] = ff[rank-1] / aa[rank-1, 0]
        i = rank - 2
        
        if i >= 0:
            for j in range(1, rank):
                store = ff[i]
                i1 = min(j, m1)
                k = i
                
                for ii in range(i1):
                    k = k + 1
                    stor1 = ff[k]
                    stor2 = aa[i, ii+1]
                    store = store - stor1*stor2
                
                stor1 = aa[i, 0]
                ff[i] = store / stor1
                i = i - 1
        
        # Solve the system (aa)' (f2) = f1
        ff[0] = ff[0] / aa[0, 0]
        
        if rank > 1:
            for j in range(1, rank):
                store = ff[j]
                i1 = min(j, m1)
                k = j
                
                for ii in range(i1):
                    k = k - 1
                    stor1 = ff[k]
                    stor2 = aa[k, ii+1]
                    store = store - stor1*stor2
                
                stor1 = aa[j, 0]
                ff[j] = store / stor1
    
    # Premultiply f2 by the transpose of a
    k = -1
    for i in range(n):
        store = 0.0
        if a[i, 0] > tol:
            k = k + 1
        j1 = min(i+1, m)
        kk = k
        ij = i
        
        for j in range(j1):
            if a[ij, 0] <= tol:
                ij = ij - 1
                continue
            stor1 = a[ij, j]
            stor2 = ff[kk]
            store = store + stor1*stor2
            kk = kk - 1
            ij = ij - 1
        
        c[i] = store
    
    # Add contribution of zero diagonal elements
    stor3 = 0.0
    for i in range(n):
        if a[i, 0] > tol:
            continue
            
        store = f[i]
        i1 = min(n-i-1, m1)
        
        if i1 > 0:
            for j in range(i1):
                ij = i + j + 1
                stor1 = c[ij]
                stor2 = a[i, j+1]
                store = store - stor1*stor2
        
        fac = a[i, 0] * c[i]
        stor1 = a[i, 0]
        stor2 = c[i]
        stor1 = stor1 * stor2
        stor3 = stor3 + stor1*(stor1 - store - store)
    
    fac = stor3
    sq = sq + fac
    
    return sq, rank


@njit(cache=True)
def fporde_njit(x, y, m, kx, ky, tx, nx, ty, ny, nummer, index, nreg):
    """
    Sorts the data points (x(i),y(i)),i=1,2,...,m according to the panel 
    tx(l)<=x<tx(l+1),ty(k)<=y<ty(k+1), they belong to.
    """
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


@njit(cache=True)
def fpbspl_njit(t, n, k, x):
    """
    Evaluates the (k+1) non-zero b-splines of degree k at t(l) <= x < t(l+1) 
    using the stable recurrence relation of de Boor and Cox.
    Returns: l, h
    """
    # Find the knot interval containing x
    l = k  # Start search at k
    while l < n - k - 1 and x >= t[l + 1]:
        l = l + 1
    
    # Output arrays
    h = np.zeros(6, dtype=np.float64)  # Max size is 6 in DIERCKX
    hh = np.zeros(5, dtype=np.float64)  # Work array
    
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
    
    return l + 1, h[:k+1]  # Convert l to 1-based for output