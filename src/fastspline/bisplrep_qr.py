"""
FITPACK-compatible bisplrep using QR decomposition via LAPACK.

This version uses QR decomposition for better numerical stability,
matching the approach used in the original FITPACK implementation.
"""

import numpy as np
from numba import njit, cfunc, types
import scipy.linalg


@njit(fastmath=True)
def find_knot_span(knots, degree, u):
    """Find the knot span index for parameter u."""
    n = len(knots) - degree - 1
    
    # Special cases
    if u >= knots[n]:
        return n - 1
    if u <= knots[degree]:
        return degree
    
    # Binary search
    low = degree
    high = n
    mid = (low + high) // 2
    
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    
    return mid


@njit(fastmath=True)
def basis_functions(knots, degree, span, u):
    """Compute non-zero B-spline basis functions."""
    N = np.zeros(degree + 1)
    left = np.zeros(degree + 1)
    right = np.zeros(degree + 1)
    
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N


@njit(fastmath=True)
def build_collocation_matrix(x_data, y_data, z_data, w_data, tx, ty, kx, ky, nx, ny):
    """Build weighted collocation matrix for B-spline fitting."""
    m = len(x_data)
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    n_cols = nk1x * nk1y
    
    # For QR, we need the full matrix (not sparse)
    A = np.zeros((m, n_cols))
    b = np.zeros(m)
    
    for i in range(m):
        # Find knot spans
        span_x = find_knot_span(tx[:nx], kx, x_data[i])
        span_y = find_knot_span(ty[:ny], ky, y_data[i])
        
        # Evaluate basis functions
        Nx = basis_functions(tx[:nx], kx, span_x, x_data[i])
        Ny = basis_functions(ty[:ny], ky, span_y, y_data[i])
        
        # Weight
        wi = np.sqrt(w_data[i])
        
        # Fill matrix row
        for p in range(kx + 1):
            for q in range(ky + 1):
                col_x = span_x - kx + p
                col_y = span_y - ky + q
                if 0 <= col_x < nk1x and 0 <= col_y < nk1y:
                    col = col_x * nk1y + col_y
                    A[i, col] = wi * Nx[p] * Ny[q]
        
        b[i] = wi * z_data[i]
    
    return A, b


@njit(fastmath=True)
def solve_qr_lstsq(A, b):
    """Solve least squares using QR decomposition."""
    m, n = A.shape
    
    # Use numpy's QR which is available in numba
    Q, R = np.linalg.qr(A)
    
    # Solve R * x = Q^T * b
    Qb = Q.T @ b
    
    # Back substitution
    c = np.zeros(n)
    for i in range(min(m, n)-1, -1, -1):
        c[i] = Qb[i]
        for j in range(i+1, n):
            c[i] -= R[i, j] * c[j]
        if abs(R[i, i]) > 1e-12:
            c[i] /= R[i, i]
        else:
            c[i] = 0.0
    
    # Compute residual
    residual = b - A @ c
    fp = np.sum(residual**2)
    
    return c, residual, fp


@njit(fastmath=True)
def add_knot_at_max_residual(x_data, y_data, residuals, tx, ty, nx, ny, kx, ky):
    """Add knots at location of maximum residual."""
    m = len(x_data)
    
    # Find maximum absolute residual
    max_res = 0.0
    max_idx = 0
    for i in range(m):
        if abs(residuals[i]) > max_res:
            max_res = abs(residuals[i])
            max_idx = i
    
    x_max = x_data[max_idx]
    y_max = y_data[max_idx]
    
    # Try to add x knot
    added_x = False
    for i in range(kx + 1, nx - kx - 1):
        if tx[i] <= x_max < tx[i + 1]:
            # Check if there's room
            if tx[i + 1] - tx[i] > 1e-6:
                new_knot = 0.5 * (tx[i] + tx[i + 1])
                # Shift knots
                for j in range(nx - 1, i, -1):
                    tx[j + 1] = tx[j]
                tx[i + 1] = new_knot
                added_x = True
                nx += 1
            break
    
    # Try to add y knot
    added_y = False
    for i in range(ky + 1, ny - ky - 1):
        if ty[i] <= y_max < ty[i + 1]:
            # Check if there's room
            if ty[i + 1] - ty[i] > 1e-6:
                new_knot = 0.5 * (ty[i] + ty[i + 1])
                # Shift knots
                for j in range(ny - 1, i, -1):
                    ty[j + 1] = ty[j]
                ty[i + 1] = new_knot
                added_y = True
                ny += 1
            break
    
    return nx, ny, added_x or added_y


@njit(fastmath=True)
def bisplrep_qr_core(x_data, y_data, z_data, w_data, kx, ky, s, 
                     nxest, nyest, maxit, tx, ty, c):
    """Core bisplrep algorithm using QR decomposition."""
    m = len(x_data)
    
    # Data bounds
    xb = np.min(x_data)
    xe = np.max(x_data)
    yb = np.min(y_data)
    ye = np.max(y_data)
    
    # Initialize with minimal knots
    nx = 2 * (kx + 1)
    ny = 2 * (ky + 1)
    
    # Set boundary knots
    for i in range(kx + 1):
        tx[i] = xb
        tx[nx - 1 - i] = xe
        ty[i] = yb
        ty[ny - 1 - i] = ye
    
    # Main iteration loop
    fp = np.inf
    for iteration in range(maxit):
        # Build collocation matrix
        A, b = build_collocation_matrix(x_data, y_data, z_data, w_data, 
                                       tx, ty, kx, ky, nx, ny)
        
        # Solve using QR decomposition
        n_cols = (nx - kx - 1) * (ny - ky - 1)
        c_work, residuals, fp_new = solve_qr_lstsq(A, b)
        
        # Copy coefficients
        for i in range(min(n_cols, len(c))):
            c[i] = c_work[i]
        
        # Check convergence
        if fp_new <= s:
            fp = fp_new
            break
        
        # Check if we should stop
        if iteration == maxit - 1 or nx >= nxest - 1 or ny >= nyest - 1:
            fp = fp_new
            break
        
        # Add knots based on residuals
        nx_new, ny_new, added = add_knot_at_max_residual(
            x_data, y_data, residuals, tx, ty, nx, ny, kx, ky)
        
        if not added:
            # Can't add more knots
            fp = fp_new
            break
        
        nx, ny = nx_new, ny_new
        fp = fp_new
    
    return nx, ny, fp


@cfunc(types.int64(types.float64[:], types.float64[:], types.float64[:],
                   types.float64[:], types.int64, types.int64, types.float64,
                   types.float64[:], types.float64[:], types.float64[:]),
       nopython=True, fastmath=True, boundscheck=False)
def bisplrep_qr(x_data, y_data, z_data, w_data, kx, ky, s,
                tx_out, ty_out, c_out):
    """
    FITPACK-compatible B-spline surface fitting using QR decomposition.
    
    Uses Householder QR decomposition for numerical stability.
    
    Parameters:
    -----------
    x_data, y_data, z_data : Data points
    w_data : Weights
    kx, ky : Spline degrees
    s : Smoothing factor
    tx_out, ty_out : Pre-allocated knot arrays
    c_out : Pre-allocated coefficient array
    
    Returns:
    --------
    Packed integer: (nx << 32) | ny
    """
    nxest = len(tx_out)
    nyest = len(ty_out)
    maxit = 20
    
    nx, ny, fp = bisplrep_qr_core(x_data, y_data, z_data, w_data,
                                  kx, ky, s, nxest, nyest, maxit,
                                  tx_out, ty_out, c_out)
    
    return (nx << 32) | ny


# Python wrapper for testing
def bisplrep_qr_py(x, y, z, w=None, kx=3, ky=3, s=0):
    """Python wrapper for QR-based bisplrep."""
    if w is None:
        w = np.ones_like(x)
    
    # Allocate arrays
    nxest = min(int(kx + np.sqrt(2*len(x))), 50)
    nyest = min(int(ky + np.sqrt(2*len(x))), 50)
    
    tx = np.zeros(nxest)
    ty = np.zeros(nyest)
    c = np.zeros(nxest * nyest)
    
    # Call implementation
    nx, ny, fp = bisplrep_qr_core(x, y, z, w, kx, ky, s, 
                                  nxest, nyest, 20, tx, ty, c)
    
    return (tx[:nx], ty[:ny], c[:nx*ny], kx, ky)