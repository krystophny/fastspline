"""
Ultra-optimized DIERCKX cfunc implementations
Maximum performance: SIMD, native arch, static memory, loop unrolling
"""

import numpy as np
from numba import cfunc, types, njit, literally, prange
import numba as nb
from numba.core.typing import signature

# ============================================================================
# ULTRA-OPTIMIZED CFUNC IMPLEMENTATIONS
# ============================================================================

# fpback - Backward substitution with cfunc
fpback_sig = types.void(
    types.CPointer(types.float64),  # a
    types.CPointer(types.float64),  # z  
    types.int64,                    # n
    types.int64,                    # k
    types.CPointer(types.float64),  # c
    types.int64,                    # nest
)

@cfunc(fpback_sig, 
       fastmath=True, 
       nopython=True,
       cache=True)
def fpback_cfunc(a, z, n, k, c, nest):
    """Ultra-optimized backward substitution with cfunc"""
    k1 = k - 1
    
    # Direct assignment for last element
    c[n-1] = z[n-1] / a[(n-1) * k + 0]
    
    if n == 1:
        return
    
    # Optimized backward substitution with manual loop unrolling
    for j in range(2, n + 1):
        i = n - j
        store = z[i]
        i1 = min(k1, j - 1)
        
        # Manual loop unrolling for common bandwidth cases
        if i1 == 1:
            store -= c[i + 1] * a[i * k + 1]
        elif i1 == 2:
            store -= c[i + 1] * a[i * k + 1] + c[i + 2] * a[i * k + 2]
        elif i1 == 3:
            store -= (c[i + 1] * a[i * k + 1] + 
                     c[i + 2] * a[i * k + 2] + 
                     c[i + 3] * a[i * k + 3])
        elif i1 == 4:
            store -= (c[i + 1] * a[i * k + 1] + c[i + 2] * a[i * k + 2] + 
                     c[i + 3] * a[i * k + 3] + c[i + 4] * a[i * k + 4])
        else:
            # General case for larger bandwidth
            m = i
            for l in range(i1):
                m += 1
                store -= c[m] * a[i * k + l + 1]
        
        c[i] = store / a[i * k + 0]


# fpgivs - Givens rotation with cfunc
fpgivs_sig = types.Tuple([types.float64, types.float64, types.float64])(
    types.float64,  # piv
    types.float64,  # ww
)

@cfunc(fpgivs_sig,
       fastmath=True,
       nopython=True, 
       cache=True)
def fpgivs_cfunc(piv, ww):
    """Ultra-optimized Givens rotation with cfunc"""
    EPS = 1e-300
    
    abs_piv = abs(piv)
    abs_ww = abs(ww)
    
    if abs_piv < EPS and abs_ww < EPS:
        return (ww, 1.0, 0.0)
    
    if abs_piv >= abs_ww:
        if abs_piv > EPS:
            ratio = ww / piv
            factor = abs_piv * (1.0 + ratio * ratio) ** 0.5
        else:
            factor = abs_ww
    else:
        ratio = piv / ww
        factor = abs_ww * (1.0 + ratio * ratio) ** 0.5
    
    if factor > EPS:
        cos = ww / factor
        sin = piv / factor
    else:
        cos = 1.0
        sin = 0.0
    
    return (factor, cos, sin)


# fprota - Apply rotation with cfunc
fprota_sig = types.Tuple([types.float64, types.float64])(
    types.float64,  # cos
    types.float64,  # sin
    types.float64,  # a
    types.float64,  # b
)

@cfunc(fprota_sig,
       fastmath=True,
       nopython=True,
       cache=True)
def fprota_cfunc(cos, sin, a, b):
    """Ultra-optimized rotation application with cfunc"""
    return (cos * a - sin * b, cos * b + sin * a)


# fprati - Rational interpolation with cfunc  
fprati_sig = types.Tuple([types.float64, types.float64, types.float64, 
                          types.float64, types.float64])(
    types.float64,  # p1
    types.float64,  # f1
    types.float64,  # p2
    types.float64,  # f2
    types.float64,  # p3
    types.float64,  # f3
)

@cfunc(fprati_sig,
       fastmath=True,
       nopython=True,
       cache=True)
def fprati_cfunc(p1, f1, p2, f2, p3, f3):
    """Ultra-optimized rational interpolation with cfunc"""
    EPS = 1e-15
    
    if p3 > 0.0:
        # Case: p3 != infinity
        h1 = f1 * (f2 - f3)
        h2 = f2 * (f3 - f1)
        h3 = f3 * (f1 - f2)
        denom = p1*h1 + p2*h2 + p3*h3
        
        if abs(denom) > EPS:
            p = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2) / denom
        else:
            p = 0.5 * (p1 + p2)
    else:
        # Case: p3 = infinity
        denom = (f1 - f2) * f3
        if abs(denom) > EPS:
            p = (p1*(f1-f3)*f2 - p2*(f2-f3)*f1) / denom
        else:
            p = 0.5 * (p1 + p2)
    
    # Optimized parameter update
    if f2 < 0.0:
        return (p, p1, f1, p2, f2)
    else:
        return (p, p2, f2, p3, f3)


# fpbspl - B-spline evaluation with cfunc
fpbspl_sig = types.void(
    types.CPointer(types.float64),  # t
    types.int64,                    # n
    types.int64,                    # k
    types.float64,                  # x
    types.int64,                    # l
    types.CPointer(types.float64),  # h (output)
)

@cfunc(fpbspl_sig,
       fastmath=True,
       nopython=True,
       cache=True)
def fpbspl_cfunc(t, n, k, x, l, h):
    """Ultra-optimized B-spline basis evaluation with cfunc - DIERCKX algorithm"""
    EPS = 1e-15
    
    # Initialize static array
    for i in range(k + 1):
        h[i] = 0.0
    h[0] = 1.0
    
    # Temporary storage - static allocation
    hh = np.zeros(5, dtype=np.float64)  # Max degree 5
    
    # DIERCKX algorithm - stable recurrence relation
    for j in range(1, k + 1):
        # Save current h values
        for i in range(j):
            hh[i] = h[i]
        
        # Reset h[0]
        h[0] = 0.0
        
        # Recurrence relation
        for i in range(j):
            li = l + i + 1
            lj = li - j
            
            if lj >= 0 and li < n and abs(t[li] - t[lj]) > EPS:
                f = hh[i] / (t[li] - t[lj])
                h[i] = h[i] + f * (t[li] - x)
                h[i + 1] = f * (x - t[lj])
            else:
                h[i + 1] = 0.0


# ============================================================================
# WRAPPER FUNCTIONS FOR PYTHON INTERFACE
# ============================================================================

from numba.core import types as nb_types
from numba.typed import Dict
import ctypes

# Get native function pointers for cfuncs
fpback_ptr = fpback_cfunc.address
fpgivs_ptr = fpgivs_cfunc.address  
fprota_ptr = fprota_cfunc.address
fprati_ptr = fprati_cfunc.address
fpbspl_ptr = fpbspl_cfunc.address

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def fpback_ultra(a, z, n, k, c, nest):
    """Ultra-optimized fpback with maximum performance"""
    k1 = k - 1
    
    # Direct assignment for last element
    c[n-1] = z[n-1] / a[n-1, 0]
    
    if n == 1:
        return
    
    # Optimized backward substitution with manual loop unrolling
    for j in range(2, n + 1):
        i = n - j
        store = z[i]
        i1 = min(k1, j - 1)
        
        # Manual loop unrolling for common bandwidth cases
        if i1 == 1:
            store -= c[i + 1] * a[i, 1]
        elif i1 == 2:
            store -= c[i + 1] * a[i, 1] + c[i + 2] * a[i, 2]
        elif i1 == 3:
            store -= (c[i + 1] * a[i, 1] + 
                     c[i + 2] * a[i, 2] + 
                     c[i + 3] * a[i, 3])
        elif i1 == 4:
            store -= (c[i + 1] * a[i, 1] + c[i + 2] * a[i, 2] + 
                     c[i + 3] * a[i, 3] + c[i + 4] * a[i, 4])
        else:
            # General case for larger bandwidth
            m = i
            for l in range(i1):
                m += 1
                store -= c[m] * a[i, l + 1]
        
        c[i] = store / a[i, 0]

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def fpgivs_ultra(piv, ww):
    """Ultra-optimized fpgivs with maximum performance"""
    EPS = 1e-300
    
    abs_piv = abs(piv)
    abs_ww = abs(ww)
    
    if abs_piv < EPS and abs_ww < EPS:
        return (ww, 1.0, 0.0)
    
    if abs_piv >= abs_ww:
        if abs_piv > EPS:
            ratio = ww / piv
            factor = abs_piv * (1.0 + ratio * ratio) ** 0.5
        else:
            factor = abs_ww
    else:
        ratio = piv / ww
        factor = abs_ww * (1.0 + ratio * ratio) ** 0.5
    
    if factor > EPS:
        cos = ww / factor
        sin = piv / factor
    else:
        cos = 1.0
        sin = 0.0
    
    return (factor, cos, sin)

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def fprota_ultra(cos, sin, a, b):
    """Ultra-optimized fprota with maximum performance"""
    return (cos * a - sin * b, cos * b + sin * a)

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)  
def fprati_ultra(p1, f1, p2, f2, p3, f3):
    """Ultra-optimized fprati with maximum performance"""
    EPS = 1e-15
    
    if p3 > 0.0:
        # Case: p3 != infinity
        h1 = f1 * (f2 - f3)
        h2 = f2 * (f3 - f1)
        h3 = f3 * (f1 - f2)
        denom = p1*h1 + p2*h2 + p3*h3
        
        if abs(denom) > EPS:
            p = -(p1*p2*h3 + p2*p3*h1 + p3*p1*h2) / denom
        else:
            p = 0.5 * (p1 + p2)
    else:
        # Case: p3 = infinity
        denom = (f1 - f2) * f3
        if abs(denom) > EPS:
            p = (p1*(f1-f3)*f2 - p2*(f2-f3)*f1) / denom
        else:
            p = 0.5 * (p1 + p2)
    
    # Optimized parameter update
    if f2 < 0.0:
        return (p, p1, f1, p2, f2)
    else:
        return (p, p2, f2, p3, f3)

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def fpbspl_ultra(t, n, k, x, l):
    """Ultra-optimized fpbspl - EXACT DIERCKX FORTRAN ALGORITHM"""
    # Initialize result array
    h = np.zeros(k + 1, dtype=np.float64)
    
    # h[1] = 1.0 in FORTRAN becomes h[0] = 1.0 in Python
    h[0] = 1.0
    
    # Temporary storage
    hh = np.zeros(k, dtype=np.float64)
    
    # Main loop - exact translation from FORTRAN
    for j in range(1, k + 1):
        # Save h values - FORTRAN loop: do 10 i=1,j
        for i in range(j):
            hh[i] = h[i]
        
        # h[1] = 0 in FORTRAN
        h[0] = 0.0
        
        # FORTRAN loop: do 20 i=1,j
        for i in range(1, j + 1):
            # FORTRAN: li = l+i, lj = li-j
            # Note: FORTRAN arrays are 1-based, Python 0-based
            # So FORTRAN t(li) becomes Python t[li-1]
            li = l + i
            lj = li - j
            
            # Convert to 0-based for array access
            li_idx = li - 1
            lj_idx = lj - 1
            
            if lj_idx >= 0 and li_idx < n:
                denom = t[li_idx] - t[lj_idx]
                if abs(denom) > 1e-10:
                    # FORTRAN: f = hh(i)/(t(li)-t(lj))
                    f = hh[i-1] / denom
                    # FORTRAN: h(i) = h(i)+f*(t(li)-x)
                    h[i-1] = h[i-1] + f * (t[li_idx] - x)
                    # FORTRAN: h(i+1) = f*(x-t(lj))
                    h[i] = f * (x - t[lj_idx])
                else:
                    h[i] = 0.0
            else:
                h[i] = 0.0
    
    return h

# ============================================================================
# WARMUP FUNCTION
# ============================================================================

def warmup_ultra_functions():
    """Warmup all ultra-optimized cfunc implementations"""
    print("Warming up ultra-optimized DIERCKX cfuncs...")
    
    # Test data
    n, k, nest = 10, 3, 15
    
    # fpback warmup
    a = np.zeros((nest, k), dtype=np.float64, order='F')
    for i in range(n):
        a[i, 0] = 2.0 + 0.1 * i
        for j in range(1, min(k, n-i)):
            a[i, j] = 0.1 / (j + 1)
    z = np.ones(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    fpback_ultra(a, z, n, k, c, nest)
    
    # Other function warmups
    fpgivs_ultra(3.0, 4.0)
    fprota_ultra(0.8, 0.6, 1.0, 2.0)
    fprati_ultra(1.0, 2.0, 2.0, 1.0, 3.0, -1.0)
    
    t = np.array([0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1.], dtype=np.float64)
    fpbspl_ultra(t, 11, 3, 0.5, 5)
    
    print("✓ All ultra-optimized cfuncs warmed up!")


# ============================================================================
# HIGH-LEVEL BIVARIATE SPLINE INTERFACES USING CFUNC BUILDING BLOCKS
# ============================================================================

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True)
def find_unique_sorted(arr, tol=1e-10):
    """Find unique values in a sorted array"""
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=arr.dtype)
    
    unique = np.empty(n, dtype=arr.dtype)
    unique[0] = arr[0]
    count = 1
    
    for i in range(1, n):
        if arr[i] - unique[count-1] > tol:
            unique[count] = arr[i]
            count += 1
    
    return unique[:count]

@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, parallel=True)
def bisplrep_cfunc(x, y, z, kx=3, ky=3, s=0.0):
    """
    Simplified bivariate spline representation using cfunc building blocks.
    
    Parameters
    ----------
    x, y, z : ndarray
        Data points (x[i], y[i], z[i])
    kx, ky : int
        Degrees of the spline (1 <= kx, ky <= 5)
    s : float
        Smoothing factor (simplified, s=0 for interpolation)
    
    Returns
    -------
    tx, ty : ndarray
        Knot vectors
    c : ndarray
        B-spline coefficients
    kx, ky : int
        Spline degrees (returned for compatibility)
    """
    m = len(x)
    
    # Get unique x and y values to determine actual grid structure
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    x_unique = find_unique_sorted(x_sorted)
    y_unique = find_unique_sorted(y_sorted)
    
    nx_unique = len(x_unique)
    ny_unique = len(y_unique)
    
    # For interpolation (s=0), number of knots depends on unique values
    # We need exactly as many basis functions as data points
    # For a regular grid: m = nx_unique * ny_unique
    # Number of basis functions = (n_knots - kx - 1) * (n_knots - ky - 1)
    
    if s == 0.0:
        # For interpolation on regular grid
        if m == nx_unique * ny_unique:
            # Regular grid - we need nx_unique + kx + 1 knots
            nx = nx_unique + kx + 1
            ny = ny_unique + ky + 1
        else:
            # Scattered data - need fewer basis functions than data points
            # Use approximation based on sqrt(m)
            n_basis = int(np.sqrt(m))
            nx = n_basis + kx + 1
            ny = n_basis + ky + 1
            
            # Ensure minimum knots
            nx = max(2*(kx+1), nx)
            ny = max(2*(ky+1), ny)
    else:
        # For smoothing
        n_basis = int(np.sqrt(m))
        nx = n_basis + kx + 1
        ny = n_basis + ky + 1
        nx = max(2*(kx+1), nx)
        ny = max(2*(ky+1), ny)
    
    # Create knot vectors
    xmin, xmax = x_unique[0], x_unique[-1]
    ymin, ymax = y_unique[0], y_unique[-1]
    
    # Create full knot vectors with multiplicity at ends
    tx = np.zeros(nx)
    ty = np.zeros(ny)
    
    # Set end knots with multiplicity kx+1 and ky+1
    for i in range(kx + 1):
        tx[i] = xmin
        tx[nx - kx - 1 + i] = xmax
    for i in range(ky + 1):
        ty[i] = ymin
        ty[ny - ky - 1 + i] = ymax
        
    # Set interior knots
    n_interior_x = nx - 2*(kx + 1)
    n_interior_y = ny - 2*(ky + 1)
    
    if n_interior_x > 0:
        # For regular grid interpolation, place knots at interior data points
        if m == nx_unique * ny_unique and s == 0.0 and n_interior_x == nx_unique - 2:
            # Place at interior unique values
            for i in range(n_interior_x):
                tx[kx + 1 + i] = x_unique[i + 1]
        else:
            # Uniform spacing
            dx = (xmax - xmin) / (n_interior_x + 1)
            for i in range(n_interior_x):
                tx[kx + 1 + i] = xmin + (i + 1) * dx
            
    if n_interior_y > 0:
        # For regular grid interpolation, place knots at interior data points
        if m == nx_unique * ny_unique and s == 0.0 and n_interior_y == ny_unique - 2:
            # Place at interior unique values
            for i in range(n_interior_y):
                ty[ky + 1 + i] = y_unique[i + 1]
        else:
            # Uniform spacing
            dy = (ymax - ymin) / (n_interior_y + 1)
            for i in range(n_interior_y):
                ty[ky + 1 + i] = ymin + (i + 1) * dy
    
    # Number of B-spline coefficients
    ncx = nx - kx - 1
    ncy = ny - ky - 1
    
    # Build collocation matrix
    A = np.zeros((m, ncx * ncy))
    
    # Parallelize over data points
    for i in prange(m):
        xi, yi = x[i], y[i]
        
        # Find knot intervals
        lx = kx
        while lx < nx - kx - 1 and xi > tx[lx+1]:  # Use > not >= to handle boundary
            lx += 1
        # Special case: if at right boundary, use last valid interval
        if xi == tx[-1] and lx == nx - kx - 1:
            lx = nx - kx - 2
            
        ly = ky  
        while ly < ny - ky - 1 and yi > ty[ly+1]:  # Use > not >= to handle boundary
            ly += 1
        # Special case: if at right boundary, use last valid interval
        if yi == ty[-1] and ly == ny - ky - 1:
            ly = ny - ky - 2
        
        # Evaluate B-splines at this point
        hx = fpbspl_ultra(tx, nx, kx, xi, lx)
        hy = fpbspl_ultra(ty, ny, ky, yi, ly)
        
        # Tensor product B-splines
        for jx in range(kx + 1):
            ix = lx - kx + jx
            if 0 <= ix < ncx:
                for jy in range(ky + 1):
                    iy = ly - ky + jy
                    if 0 <= iy < ncy:
                        col_idx = ix * ncy + iy
                        A[i, col_idx] = hx[jx] * hy[jy]
    
    # For small problems, use least squares directly
    # For interpolation (s=0), we need exact fit at data points
    if m == ncx * ncy:
        # Square system - direct solve
        c = np.linalg.solve(A, z)
    else:
        # Overdetermined - use least squares
        # Solve min ||Ac - z||^2 using QR decomposition would be better,
        # but for simplicity use normal equations
        AtA = A.T @ A
        Atz = A.T @ z
        
        # Add small regularization for numerical stability
        reg = 1e-10
        for i in range(ncx * ncy):
            AtA[i, i] += reg
        
        c = np.linalg.solve(AtA, Atz)
    
    return tx, ty, c, kx, ky


@njit(fastmath=True, cache=True, boundscheck=False, nogil=True, parallel=True)
def bisplev_cfunc(x, y, tx, ty, c, kx, ky):
    """
    Evaluate bivariate B-spline using cfunc building blocks.
    
    Parameters
    ----------
    x, y : ndarray
        Evaluation points (must be arrays, not scalars)
    tx, ty : ndarray
        Knot vectors
    c : ndarray
        B-spline coefficients
    kx, ky : int
        Degrees of the spline
        
    Returns
    -------
    z : ndarray
        Evaluated spline values
    """
    nx = len(x)
    ny = len(y)
    ntx = len(tx)
    nty = len(ty)
    ncx = ntx - kx - 1
    ncy = nty - ky - 1
    
    # Output array
    z = np.zeros((nx, ny))
    
    # Parallelize over x values
    for i in prange(nx):
        xi = x[i]
        
        # Find x interval
        lx = kx
        while lx < ntx - kx - 1 and xi > tx[lx+1]:
            lx += 1
        # Special case: if at right boundary, use last valid interval
        if xi == tx[-1] and lx == ntx - kx - 1:
            lx = ntx - kx - 2
            
        # Evaluate x B-splines
        hx = fpbspl_ultra(tx, ntx, kx, xi, lx)
        
        for j in range(ny):
            yj = y[j]
            
            # Find y interval  
            ly = ky
            while ly < nty - ky - 1 and yj > ty[ly+1]:
                ly += 1
            # Special case: if at right boundary, use last valid interval
            if yj == ty[-1] and ly == nty - ky - 1:
                ly = nty - ky - 2
                
            # Evaluate y B-splines
            hy = fpbspl_ultra(ty, nty, ky, yj, ly)
            
            # Sum tensor product B-splines
            val = 0.0
            for jx in range(kx + 1):
                cx_idx = lx - kx + jx
                if 0 <= cx_idx < ncx:
                    for jy in range(ky + 1):
                        cy_idx = ly - ky + jy
                        if 0 <= cy_idx < ncy:
                            c_idx = cx_idx * ncy + cy_idx
                            val += hx[jx] * hy[jy] * c[c_idx]
            
            z[i, j] = val
    
    return z

if __name__ == "__main__":
    warmup_ultra_functions()