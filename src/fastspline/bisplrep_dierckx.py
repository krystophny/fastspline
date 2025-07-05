"""
DIERCKX-compatible bisplrep implementation with correct coefficient ordering.

This implementation closely follows the original FITPACK algorithms.
"""

import numpy as np
from numba import njit, cfunc, types


@njit(cache=True, fastmath=True)
def find_span(n, p, u, U):
    """Find the knot span index for parameter u."""
    # Number of basis functions = n - p - 1
    num_basis = n - p - 1
    
    # Special case: u is at or beyond the end
    if u >= U[num_basis]:
        return num_basis - 1
    
    # Special case: u is at or before the start
    if u <= U[p]:
        return p
    
    # Binary search
    low = p
    high = num_basis
    mid = (low + high) // 2
    
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid


@njit(cache=True, fastmath=True)
def basis_funs(i, u, p, U):
    """Compute the non-vanishing basis functions."""
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if abs(denom) > 1e-15:
                temp = N[r] / denom
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            else:
                N[r] = saved
                saved = 0.0
        
        N[j] = saved
    
    return N


@njit(cache=True, fastmath=True)
def fpgivs(piv, ww):
    """
    Compute Givens rotation coefficients.
    Following DIERCKX fpgivs.f exactly.
    """
    if abs(piv) < 1e-15:
        cos = 1.0
        sin = 0.0
        ww = abs(ww)
    elif abs(ww) < 1e-15:
        cos = 0.0
        sin = 1.0 if piv > 0 else -1.0
        ww = abs(piv)
    else:
        # DIERCKX formula
        if abs(piv) >= abs(ww):
            r = ww / piv
            dd = abs(piv) * np.sqrt(1.0 + r*r)
            cos = 1.0 / np.sqrt(1.0 + r*r)
            sin = cos * r
            if piv < 0:
                cos = -cos
                sin = -sin
                dd = -dd
        else:
            r = piv / ww
            dd = abs(ww) * np.sqrt(1.0 + r*r)
            sin = 1.0 / np.sqrt(1.0 + r*r)
            cos = sin * r
            if ww < 0:
                sin = -sin
                cos = -cos
                dd = -dd
        ww = dd
    
    return cos, sin, ww


@njit(cache=True, fastmath=True)
def fprota(cos, sin, a, b):
    """Apply Givens rotation to two scalars."""
    stor1 = a
    stor2 = b
    a = cos * stor1 + sin * stor2
    b = -sin * stor1 + cos * stor2
    return a, b


@njit(cache=True, fastmath=True)
def fpback(a, z, n, k, c):
    """
    Back substitution after QR decomposition.
    Following DIERCKX fpback.f
    """
    # Initialize coefficients
    for i in range(k):
        c[i] = 0.0
    
    # Back substitution
    for i in range(k-1, -1, -1):
        store = z[i]
        for j in range(i+1, k):
            store -= a[i, j] * c[j]
        if abs(a[i, i]) > 1e-14:
            c[i] = store / a[i, i]
        else:
            c[i] = 0.0


@njit(cache=True, fastmath=True)
def build_design_matrix_dierckx(x, y, z, w, tx, ty, kx, ky):
    """
    Build design matrix with DIERCKX coefficient ordering.
    
    DIERCKX uses row-major ordering: c[i,j] -> c[(ny-ky-1)*(i-1)+j]
    where i = 1,...,nx-kx-1 and j = 1,...,ny-ky-1
    """
    m = len(x)
    nx = len(tx)
    ny = len(ty)
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    n_cols = nk1x * nk1y
    
    A = np.zeros((m, n_cols))
    b = np.zeros(m)
    
    for idx in range(m):
        # Find knot spans
        span_x = find_span(nx, kx, x[idx], tx)
        span_y = find_span(ny, ky, y[idx], ty)
        
        # Compute basis functions
        Nx = basis_funs(span_x, x[idx], kx, tx)
        Ny = basis_funs(span_y, y[idx], ky, ty)
        
        # Weight
        wi = np.sqrt(w[idx])
        
        # Fill matrix row with DIERCKX ordering
        for i in range(kx + 1):
            ix = span_x - kx + i
            if 0 <= ix < nk1x:
                for j in range(ky + 1):
                    iy = span_y - ky + j
                    if 0 <= iy < nk1y:
                        # DIERCKX ordering: row-major
                        col = ix * nk1y + iy
                        A[idx, col] = wi * Nx[i] * Ny[j]
        
        b[idx] = wi * z[idx]
    
    return A, b


@njit(cache=True, fastmath=True)
def qr_givens_dierckx(A, b):
    """
    QR decomposition using Givens rotations.
    Following DIERCKX approach.
    """
    m, n = A.shape
    
    # For each column
    for j in range(min(m, n)):
        # Find pivot (non-zero element) in column j
        pivot_row = -1
        for i in range(j, m):
            if abs(A[i, j]) > 1e-15:
                pivot_row = i
                break
        
        if pivot_row == -1:
            continue
            
        # If pivot is not on diagonal, swap rows
        if pivot_row != j:
            for k in range(n):
                A[j, k], A[pivot_row, k] = A[pivot_row, k], A[j, k]
            b[j], b[pivot_row] = b[pivot_row], b[j]
        
        # Eliminate elements below diagonal
        for i in range(j+1, m):
            if abs(A[i, j]) > 1e-15:
                # Compute Givens rotation
                cos, sin, piv = fpgivs(A[j, j], A[i, j])
                A[j, j] = piv
                A[i, j] = 0.0
                
                # Apply rotation to rest of row
                for k in range(j+1, n):
                    A[j, k], A[i, k] = fprota(cos, sin, A[j, k], A[i, k])
                
                # Apply rotation to RHS
                b[j], b[i] = fprota(cos, sin, b[j], b[i])
    
    return A, b


@njit(cache=True, fastmath=True)
def solve_qr_system(A, b, n):
    """Solve the QR system."""
    c = np.zeros(n)
    
    # Back substitution
    for i in range(min(A.shape[0], n)-1, -1, -1):
        if abs(A[i, i]) > 1e-14:
            c[i] = b[i]
            for j in range(i+1, n):
                c[i] -= A[i, j] * c[j]
            c[i] /= A[i, i]
        else:
            c[i] = 0.0
    
    return c


@njit(cache=True, fastmath=True)
def compute_residuals(x, y, z, w, tx, ty, c, kx, ky):
    """Compute weighted residuals."""
    m = len(x)
    nx = len(tx)
    ny = len(ty)
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    
    residuals = np.zeros(m)
    fp = 0.0
    
    for idx in range(m):
        # Find knot spans
        span_x = find_span(nx, kx, x[idx], tx)
        span_y = find_span(ny, ky, y[idx], ty)
        
        # Compute basis functions
        Nx = basis_funs(span_x, x[idx], kx, tx)
        Ny = basis_funs(span_y, y[idx], ky, ty)
        
        # Evaluate spline
        val = 0.0
        for i in range(kx + 1):
            ix = span_x - kx + i
            if 0 <= ix < nk1x:
                for j in range(ky + 1):
                    iy = span_y - ky + j
                    if 0 <= iy < nk1y:
                        # DIERCKX ordering
                        coef_idx = ix * nk1y + iy
                        val += c[coef_idx] * Nx[i] * Ny[j]
        
        residuals[idx] = z[idx] - val
        fp += w[idx] * residuals[idx]**2
    
    return residuals, fp


@njit(cache=True, fastmath=True)
def add_knot_dierckx(t, n, k, t_new):
    """Add a new knot to the knot vector."""
    # Find insertion point
    insert_idx = n
    
    for i in range(k, n-k):
        if t[i] <= t_new <= t[i+1]:
            insert_idx = i + 1
            break
    
    if insert_idx < n:
        # Shift knots to make room
        for j in range(n-1, insert_idx-1, -1):
            t[j+1] = t[j]
        t[insert_idx] = t_new
        return n + 1
    
    return n


@njit(cache=True, fastmath=True)
def find_knot_insertion_dierckx(x, y, residuals, w, tx, ty, nx, ny, kx, ky):
    """
    Find optimal position for knot insertion based on residuals.
    Following DIERCKX strategy.
    """
    m = len(x)
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    
    # Compute sum of squared residuals in each knot interval
    fpintx = np.zeros(nk1x + 1)
    coordx = np.zeros(nk1x + 1)
    fpinty = np.zeros(nk1y + 1)
    coordy = np.zeros(nk1y + 1)
    
    for idx in range(m):
        # Find knot intervals
        ix = kx
        while ix < nx-kx-1 and x[idx] >= tx[ix+1]:
            ix += 1
        
        iy = ky
        while iy < ny-ky-1 and y[idx] >= ty[iy+1]:
            iy += 1
        
        # Accumulate residuals
        res2 = w[idx] * residuals[idx]**2
        fpintx[ix-kx] += res2
        coordx[ix-kx] += res2 * x[idx]
        fpinty[iy-ky] += res2
        coordy[iy-ky] += res2 * y[idx]
    
    # Find interval with maximum error
    max_fpx = 0.0
    max_ix = 0
    for i in range(nk1x - 1):
        if fpintx[i] > max_fpx and tx[i+kx+1] - tx[i+kx] > 1e-6:
            max_fpx = fpintx[i]
            max_ix = i
    
    max_fpy = 0.0
    max_iy = 0
    for i in range(nk1y - 1):
        if fpinty[i] > max_fpy and ty[i+ky+1] - ty[i+ky] > 1e-6:
            max_fpy = fpinty[i]
            max_iy = i
    
    # Compute new knot positions
    add_x = False
    add_y = False
    new_x = 0.0
    new_y = 0.0
    
    if max_fpx > 0 and nx < len(tx) - 1:
        # Weighted average position
        new_x = coordx[max_ix] / fpintx[max_ix]
        # Ensure it's in the interval
        new_x = max(tx[max_ix+kx], min(new_x, tx[max_ix+kx+1]))
        add_x = True
    
    if max_fpy > 0 and ny < len(ty) - 1:
        new_y = coordy[max_iy] / fpinty[max_iy]
        new_y = max(ty[max_iy+ky], min(new_y, ty[max_iy+ky+1]))
        add_y = True
    
    return add_x, new_x, add_y, new_y


@njit(cache=True, fastmath=True)
def bisplrep_dierckx_core(x, y, z, w, kx, ky, s, nxest, nyest, tx_out, ty_out, c_out):
    """
    DIERCKX-compatible bisplrep core algorithm.
    """
    m = len(x)
    
    # Data bounds
    xb, xe = x.min(), x.max()
    yb, ye = y.min(), y.max()
    
    # Initialize knots
    nx = 2 * (kx + 1)
    ny = 2 * (ky + 1)
    
    # Boundary knots with multiplicity k+1
    for i in range(kx + 1):
        tx_out[i] = xb
        tx_out[nx - 1 - i] = xe
        ty_out[i] = yb
        ty_out[ny - 1 - i] = ye
    
    # For interpolation, add initial interior knots
    if s == 0:
        # Add uniform interior knots
        nx_target = int(np.sqrt(m) + kx + 1)
        ny_target = nx_target
        
        if nx < nx_target:
            n_add = min(nx_target - nx, nxest - nx - 1)
            for i in range(n_add):
                new_knot = xb + (i + 1) * (xe - xb) / (n_add + 1)
                nx = add_knot_dierckx(tx_out, nx, kx, new_knot)
        
        if ny < ny_target:
            n_add = min(ny_target - ny, nyest - ny - 1)
            for i in range(n_add):
                new_knot = yb + (i + 1) * (ye - yb) / (n_add + 1)
                ny = add_knot_dierckx(ty_out, ny, ky, new_knot)
    
    # Main iteration loop
    fp = np.inf
    maxit = 20
    
    for iteration in range(maxit):
        # Build design matrix with DIERCKX ordering
        A, b = build_design_matrix_dierckx(x, y, z, w, tx_out[:nx], ty_out[:ny], kx, ky)
        
        # QR decomposition
        A_qr, b_qr = qr_givens_dierckx(A.copy(), b.copy())
        
        # Solve system
        n_cols = (nx - kx - 1) * (ny - ky - 1)
        c = solve_qr_system(A_qr, b_qr, n_cols)
        
        # Copy coefficients
        for i in range(min(n_cols, len(c_out))):
            c_out[i] = c[i]
        
        # Compute residuals
        residuals, fp_new = compute_residuals(x, y, z, w, tx_out[:nx], ty_out[:ny], c, kx, ky)
        
        # Check convergence
        if fp_new <= s:
            fp = fp_new
            break
        
        # Check if we can add more knots
        if nx >= nxest - 1 and ny >= nyest - 1:
            fp = fp_new
            break
        
        # Find where to add knots based on residuals
        add_x, new_x, add_y, new_y = find_knot_insertion_dierckx(
            x, y, residuals, w, tx_out[:nx], ty_out[:ny], nx, ny, kx, ky)
        
        # Add knots
        if add_x and nx < nxest - 1:
            nx = add_knot_dierckx(tx_out, nx, kx, new_x)
        if add_y and ny < nyest - 1:
            ny = add_knot_dierckx(ty_out, ny, ky, new_y)
        
        # Check if no knots were added
        if not (add_x or add_y):
            fp = fp_new
            break
        
        fp = fp_new
    
    return nx, ny, fp


def bisplrep_dierckx(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
                     kx=3, ky=3, s=0, nxest=None, nyest=None, task=0, eps=1e-16):
    """
    Find a bivariate B-spline representation of a surface.
    
    DIERCKX-compatible implementation with correct coefficient ordering.
    """
    # Convert inputs
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    
    m = len(x)
    
    if w is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
    
    # Data bounds
    if xb is None:
        xb = x.min()
    if xe is None:
        xe = x.max()
    if yb is None:
        yb = y.min()
    if ye is None:
        ye = y.max()
    
    # Estimate knot array sizes
    if nxest is None:
        if s == 0:
            nxest = max(2*(kx+1), int(np.sqrt(m) + kx + 1) + 4)
        else:
            nxest = max(2*(kx+1), min(int(kx + 1.5*np.sqrt(m)), m//2 + kx + 1))
    if nyest is None:
        if s == 0:
            nyest = max(2*(ky+1), int(np.sqrt(m) + ky + 1) + 4)
        else:
            nyest = max(2*(ky+1), min(int(ky + 1.5*np.sqrt(m)), m//2 + ky + 1))
    
    # Allocate arrays
    tx_arr = np.zeros(nxest)
    ty_arr = np.zeros(nyest)
    c_arr = np.zeros(nxest * nyest)
    
    # Fit spline
    nx, ny, fp = bisplrep_dierckx_core(x, y, z, w, kx, ky, s, nxest, nyest,
                                       tx_arr, ty_arr, c_arr)
    
    # Extract results
    tx_out = tx_arr[:nx].copy()
    ty_out = ty_arr[:ny].copy()
    c_out = c_arr[:(nx-kx-1)*(ny-ky-1)].copy()
    
    return (tx_out, ty_out, c_out, kx, ky)