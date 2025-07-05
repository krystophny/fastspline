"""
Optimized DIERCKX-compatible bisplrep with proper QR decomposition.
"""

import numpy as np
from numba import njit, cfunc, types


@njit(cache=True, fastmath=True)
def find_span(n, p, u, U):
    """Find the knot span index for parameter u."""
    # n is the number of knots
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
def givens_rotation(a, b):
    """Compute Givens rotation coefficients."""
    if b == 0:
        c = 1.0
        s = 0.0
    elif abs(b) > abs(a):
        t = a / b
        s = 1.0 / np.sqrt(1.0 + t*t)
        c = s * t
    else:
        t = b / a
        c = 1.0 / np.sqrt(1.0 + t*t)
        s = c * t
    
    return c, s


@njit(cache=True, fastmath=True)
def qr_decomposition_givens(A, b):
    """
    QR decomposition using Givens rotations.
    Modifies A in-place to contain R, returns modified b.
    """
    m, n = A.shape
    
    for j in range(n):
        for i in range(m-1, j, -1):
            if abs(A[i, j]) > 1e-15:
                # Compute Givens rotation
                c, s = givens_rotation(A[i-1, j], A[i, j])
                
                # Apply rotation to rows i-1 and i
                for k in range(j, n):
                    temp = c * A[i-1, k] + s * A[i, k]
                    A[i, k] = -s * A[i-1, k] + c * A[i, k]
                    A[i-1, k] = temp
                
                # Apply rotation to b
                temp = c * b[i-1] + s * b[i]
                b[i] = -s * b[i-1] + c * b[i]
                b[i-1] = temp
    
    return b


@njit(cache=True, fastmath=True)
def back_substitution(R, b, n):
    """Solve Rx = b by back substitution."""
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        if abs(R[i, i]) > 1e-14:
            x[i] = b[i]
            for j in range(i+1, n):
                x[i] -= R[i, j] * x[j]
            x[i] /= R[i, i]
        else:
            x[i] = 0.0
    
    return x


@njit(cache=True, fastmath=True)
def build_design_matrix(x, y, z, w, tx, ty, kx, ky):
    """Build the design matrix for B-spline fitting."""
    m = len(x)
    nx = len(tx) - kx - 1
    ny = len(ty) - ky - 1
    n_cols = nx * ny
    
    A = np.zeros((m, n_cols))
    b = np.zeros(m)
    
    for i in range(m):
        # Find knot spans
        span_x = find_span(len(tx), kx, x[i], tx)
        span_y = find_span(len(ty), ky, y[i], ty)
        
        # Compute basis functions
        Nx = basis_funs(span_x, x[i], kx, tx)
        Ny = basis_funs(span_y, y[i], ky, ty)
        
        # Fill matrix row (weighted)
        wi = np.sqrt(w[i])
        for p in range(kx + 1):
            for q in range(ky + 1):
                col_x = span_x - kx + p
                col_y = span_y - ky + q
                if 0 <= col_x < nx and 0 <= col_y < ny:
                    col = col_x * ny + col_y
                    A[i, col] = wi * Nx[p] * Ny[q]
        
        b[i] = wi * z[i]
    
    return A, b


@njit(cache=True, fastmath=True)
def fit_spline_qr(x, y, z, w, tx, ty, kx, ky):
    """Fit B-spline surface using QR decomposition."""
    # Build design matrix
    A, b = build_design_matrix(x, y, z, w, tx, ty, kx, ky)
    
    m, n = A.shape
    
    # QR decomposition
    b_qr = qr_decomposition_givens(A, b.copy())
    
    # Back substitution
    c = back_substitution(A, b_qr, n)
    
    # Compute residual
    fp = 0.0
    for i in range(m):
        res = z[i]
        span_x = find_span(len(tx), kx, x[i], tx)
        span_y = find_span(len(ty), ky, y[i], ty)
        
        Nx = basis_funs(span_x, x[i], kx, tx)
        Ny = basis_funs(span_y, y[i], ky, ty)
        
        nx = len(tx) - kx - 1
        ny = len(ty) - ky - 1
        
        for p in range(kx + 1):
            for q in range(ky + 1):
                col_x = span_x - kx + p
                col_y = span_y - ky + q
                if 0 <= col_x < nx and 0 <= col_y < ny:
                    col = col_x * ny + col_y
                    res -= c[col] * Nx[p] * Ny[q]
        
        fp += w[i] * res * res
    
    return c, fp


@njit(cache=True, fastmath=True)
def add_knot(t, n, k, t_new):
    """Add a new knot to the knot vector."""
    # Find insertion point
    insert_idx = n  # Default to end
    
    for i in range(k, n-k):
        if t[i] <= t_new <= t[i+1]:
            # Found the interval
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
def bisplrep_qr_fit(x, y, z, w, kx, ky, s, nxest, nyest, tx_out, ty_out, c_out):
    """Main bisplrep algorithm with QR decomposition."""
    m = len(x)
    
    # Data bounds
    xb, xe = x.min(), x.max()
    yb, ye = y.min(), y.max()
    
    # Initialize knots
    nx = 2 * (kx + 1)
    ny = 2 * (ky + 1)
    
    # Boundary knots
    for i in range(kx + 1):
        tx_out[i] = xb
        tx_out[nx - 1 - i] = xe
        ty_out[i] = yb
        ty_out[ny - 1 - i] = ye
    
    # For interpolation (s=0), we need enough knots to have at least m coefficients
    # Number of coefficients = (nx - kx - 1) * (ny - ky - 1)
    if s == 0:
        # Calculate minimum knots needed
        nx_min = int(np.sqrt(m) + kx + 1)
        ny_min = nx_min
        
        # Add interior knots uniformly
        if nx < nx_min and nx < nxest - 1:
            n_add = min(nx_min - nx, nxest - nx - 1)
            for i in range(n_add):
                new_knot = xb + (i + 1) * (xe - xb) / (n_add + 1)
                nx = add_knot(tx_out, nx, kx, new_knot)
        
        if ny < ny_min and ny < nyest - 1:
            n_add = min(ny_min - ny, nyest - ny - 1)
            for i in range(n_add):
                new_knot = yb + (i + 1) * (ye - yb) / (n_add + 1)
                ny = add_knot(ty_out, ny, ky, new_knot)
    else:
        # For smoothing splines, start with fewer knots
        # Add some initial interior knots based on data distribution
        if nx < nxest - 2:
            x_unique = np.unique(x)
            if len(x_unique) > 4:
                n_add = min(2, (nxest - nx) // 2)
                for i in range(n_add):
                    idx = (i + 1) * len(x_unique) // (n_add + 1)
                    new_knot = x_unique[idx]
                    if xb < new_knot < xe:
                        nx = add_knot(tx_out, nx, kx, new_knot)
        
        if ny < nyest - 2:
            y_unique = np.unique(y)
            if len(y_unique) > 4:
                n_add = min(2, (nyest - ny) // 2)
                for i in range(n_add):
                    idx = (i + 1) * len(y_unique) // (n_add + 1)
                    new_knot = y_unique[idx]
                    if yb < new_knot < ye:
                        ny = add_knot(ty_out, ny, ky, new_knot)
    
    # Main fitting loop
    fp = np.inf
    maxit = 20
    
    for it in range(maxit):
        # Fit with current knots
        c, fp_new = fit_spline_qr(x, y, z, w, tx_out[:nx], ty_out[:ny], kx, ky)
        
        # Copy coefficients
        n_coef = (nx - kx - 1) * (ny - ky - 1)
        for i in range(min(n_coef, len(c_out))):
            c_out[i] = c[i]
        
        # Check convergence
        if fp_new <= s:
            fp = fp_new
            break
        
        # Try to add knots
        if nx < nxest - 1:
            # Add knot in x direction at midpoint of largest interval
            max_gap = 0.0
            max_idx = kx
            for i in range(kx, nx-kx-1):
                gap = tx_out[i+1] - tx_out[i]
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            
            if max_gap > 1e-6:
                new_knot = 0.5 * (tx_out[max_idx] + tx_out[max_idx+1])
                nx = add_knot(tx_out, nx, kx, new_knot)
        
        if ny < nyest - 1:
            # Add knot in y direction
            max_gap = 0.0
            max_idx = ky
            for i in range(ky, ny-ky-1):
                gap = ty_out[i+1] - ty_out[i]
                if gap > max_gap:
                    max_gap = gap
                    max_idx = i
            
            if max_gap > 1e-6:
                new_knot = 0.5 * (ty_out[max_idx] + ty_out[max_idx+1])
                ny = add_knot(ty_out, ny, ky, new_knot)
        
        # Check if we can't add more knots
        if nx >= nxest - 1 and ny >= nyest - 1:
            fp = fp_new
            break
        
        fp = fp_new
    
    return nx, ny, fp


def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
             kx=3, ky=3, task=0, s=0, eps=1e-16, tx=None, ty=None,
             nxest=None, nyest=None, wrk=None, lwrk1=None, lwrk2=None):
    """
    Find a bivariate B-spline representation of a surface.
    
    Optimized implementation using QR decomposition.
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
    
    # Estimate knot array sizes
    if nxest is None:
        # For interpolation, ensure we have enough knots
        if s == 0:
            nxest = max(2*(kx+1), int(np.sqrt(m) + kx + 1) + 2)
        else:
            nxest = max(2*(kx+1), min(int(kx + np.sqrt(m)), m//2 + kx + 1))
    if nyest is None:
        if s == 0:
            nyest = max(2*(ky+1), int(np.sqrt(m) + ky + 1) + 2)
        else:
            nyest = max(2*(ky+1), min(int(ky + np.sqrt(m)), m//2 + ky + 1))
    
    # Allocate arrays
    tx_arr = np.zeros(nxest)
    ty_arr = np.zeros(nyest)
    c_arr = np.zeros(nxest * nyest)
    
    # Fit spline
    nx, ny, fp = bisplrep_qr_fit(x, y, z, w, kx, ky, s, nxest, nyest,
                                 tx_arr, ty_arr, c_arr)
    
    # Extract results
    tx_out = tx_arr[:nx].copy()
    ty_out = ty_arr[:ny].copy()
    c_out = c_arr[:(nx-kx-1)*(ny-ky-1)].copy()
    
    return (tx_out, ty_out, c_out, kx, ky)