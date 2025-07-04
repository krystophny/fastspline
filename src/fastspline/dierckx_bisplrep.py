"""
DIERCKX SURFIT implementation for bisplrep.

This implements the algorithm from FITPACK by P. Dierckx.
"""

import numpy as np
from numba import njit, types
from .bisplrep import find_knot_span, basis_functions


@njit(fastmath=True, cache=True)
def givens_rotation(a, b):
    """Compute Givens rotation parameters."""
    if abs(b) < 1e-15:
        c = 1.0
        s = 0.0
        r = a
    else:
        if abs(a) < abs(b):
            t = a / b
            s = 1.0 / np.sqrt(1.0 + t*t)
            c = s * t
            r = b / s
        else:
            t = b / a
            c = 1.0 / np.sqrt(1.0 + t*t)
            s = c * t
            r = a / c
    return c, s, r


@njit(fastmath=True, cache=True)
def solve_least_squares(A, b):
    """Solve least squares using normal equations for overdetermined system."""
    m, n = A.shape
    
    # Form normal equations: A^T A x = A^T b
    AtA = np.zeros((n, n))
    Atb = np.zeros(n)
    
    # Compute A^T A and A^T b
    for i in range(n):
        for j in range(i, n):
            sum_val = 0.0
            for k in range(m):
                sum_val += A[k, i] * A[k, j]
            AtA[i, j] = sum_val
            AtA[j, i] = sum_val
        
        sum_val = 0.0
        for k in range(m):
            sum_val += A[k, i] * b[k]
        Atb[i] = sum_val
    
    # Add small regularization for numerical stability
    for i in range(n):
        AtA[i, i] += 1e-10
    
    # Cholesky decomposition
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            sum_val = AtA[i, j]
            for k in range(j):
                sum_val -= L[i, k] * L[j, k]
            
            if i == j:
                if sum_val <= 0:
                    sum_val = 1e-10
                L[i, j] = np.sqrt(sum_val)
            else:
                L[i, j] = sum_val / L[j, j] if abs(L[j, j]) > 1e-15 else 0.0
    
    # Forward substitution: L y = A^T b
    y = np.zeros(n)
    for i in range(n):
        sum_val = Atb[i]
        for j in range(i):
            sum_val -= L[i, j] * y[j]
        y[i] = sum_val / L[i, i] if abs(L[i, i]) > 1e-15 else 0.0
    
    # Back substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_val = y[i]
        for j in range(i + 1, n):
            sum_val -= L[j, i] * x[j]
        x[i] = sum_val / L[i, i] if abs(L[i, i]) > 1e-15 else 0.0
    
    return x


@njit(fastmath=True, cache=True)
def build_observation_matrix(x, y, z, w, tx, ty, nx, ny, kx, ky):
    """Build observation matrix for spline fitting."""
    m = len(x)
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    ncof = nk1x * nk1y
    
    # Determine bandwidth
    bandwidth = (kx + 1) * (ky + 1)
    
    # Sparse matrix representation (dense for now)
    A = np.zeros((m, ncof))
    b = np.zeros(m)
    
    for i in range(m):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        wi = np.sqrt(w[i])
        
        # Find knot spans
        ix = find_knot_span(tx[:nx], kx, xi)
        iy = find_knot_span(ty[:ny], ky, yi)
        
        # Compute basis functions
        bx = basis_functions(tx[:nx], kx, ix, xi)
        by = basis_functions(ty[:ny], ky, iy, yi)
        
        # Fill matrix row
        for p in range(kx + 1):
            for q in range(ky + 1):
                col_x = ix - kx + p
                col_y = iy - ky + q
                if 0 <= col_x < nk1x and 0 <= col_y < nk1y:
                    col = col_x * nk1y + col_y
                    A[i, col] = wi * bx[p] * by[q]
        
        b[i] = wi * zi
    
    return A, b, bandwidth


@njit(fastmath=True, cache=True)
def compute_fp(x, y, z, w, tx, ty, c, nx, ny, kx, ky):
    """Compute sum of squared residuals."""
    m = len(x)
    nk1y = ny - ky - 1
    fp = 0.0
    
    for i in range(m):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        wi = w[i]
        
        # Find knot spans
        ix = find_knot_span(tx[:nx], kx, xi)
        iy = find_knot_span(ty[:ny], ky, yi)
        
        # Compute basis functions
        bx = basis_functions(tx[:nx], kx, ix, xi)
        by = basis_functions(ty[:ny], ky, iy, yi)
        
        # Evaluate spline
        val = 0.0
        for p in range(kx + 1):
            for q in range(ky + 1):
                col_x = ix - kx + p
                col_y = iy - ky + q
                if 0 <= col_x < nx - kx - 1 and 0 <= col_y < ny - ky - 1:
                    idx = col_x * nk1y + col_y
                    if idx < len(c):
                        val += c[idx] * bx[p] * by[q]
        
        # Accumulate weighted squared residual
        residual = zi - val
        fp += wi * residual * residual
    
    return fp


@njit(fastmath=True, cache=True)
def find_knot_insertion_point(x, y, z, w, tx, ty, c, nx, ny, kx, ky):
    """Find optimal position to insert new knot based on residuals."""
    m = len(x)
    nk1y = ny - ky - 1
    
    # Compute residuals for each data point
    residuals = np.zeros(m)
    for i in range(m):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        
        # Evaluate spline at this point
        ix = find_knot_span(tx[:nx], kx, xi)
        iy = find_knot_span(ty[:ny], ky, yi)
        
        bx = basis_functions(tx[:nx], kx, ix, xi)
        by = basis_functions(ty[:ny], ky, iy, yi)
        
        val = 0.0
        for p in range(kx + 1):
            for q in range(ky + 1):
                col_x = ix - kx + p
                col_y = iy - ky + q
                if 0 <= col_x < nx - kx - 1 and 0 <= col_y < ny - ky - 1:
                    idx = col_x * nk1y + col_y
                    if idx < len(c):
                        val += c[idx] * bx[p] * by[q]
        
        residuals[i] = w[i] * (zi - val) ** 2
    
    # Find interval with maximum total residual
    # X direction
    max_res_x = 0.0
    best_x_interval = -1
    for i in range(kx, nx - kx - 1):
        if tx[i+1] - tx[i] < 1e-6:
            continue
        
        interval_res = 0.0
        count = 0
        for j in range(m):
            if tx[i] <= x[j] < tx[i+1]:
                interval_res += residuals[j]
                count += 1
        
        if count > 0 and interval_res > max_res_x:
            max_res_x = interval_res
            best_x_interval = i
    
    # Y direction
    max_res_y = 0.0
    best_y_interval = -1
    for i in range(ky, ny - ky - 1):
        if ty[i+1] - ty[i] < 1e-6:
            continue
        
        interval_res = 0.0
        count = 0
        for j in range(m):
            if ty[i] <= y[j] < ty[i+1]:
                interval_res += residuals[j]
                count += 1
        
        if count > 0 and interval_res > max_res_y:
            max_res_y = interval_res
            best_y_interval = i
    
    return best_x_interval, best_y_interval, max_res_x, max_res_y


@njit(fastmath=True, cache=True)
def dierckx_surfit(x, y, z, w, xb, xe, yb, ye, kx, ky, s, nxest, nyest, eps):
    """DIERCKX SURFIT algorithm implementation."""
    m = len(x)
    kx1 = kx + 1
    ky1 = ky + 1
    
    # Initialize knot vectors
    tx = np.zeros(nxest)
    ty = np.zeros(nyest)
    
    # Set boundary knots
    for i in range(kx1):
        tx[i] = xb
        tx[nxest - 1 - i] = xe
    for i in range(ky1):
        ty[i] = yb
        ty[nyest - 1 - i] = ye
    
    # Initial interior knots (start with uniform distribution)
    n_init_x = 2
    n_init_y = 2
    dx = (xe - xb) / (n_init_x + 1)
    dy = (ye - yb) / (n_init_y + 1)
    
    for i in range(n_init_x):
        tx[kx1 + i] = xb + (i + 1) * dx
    for i in range(n_init_y):
        ty[ky1 + i] = yb + (i + 1) * dy
    
    nx = 2 * kx1 + n_init_x
    ny = 2 * ky1 + n_init_y
    
    # Main iteration
    maxit = 20
    fp_old = np.inf
    
    for iteration in range(maxit):
        # Build observation matrix
        A, b, bandwidth = build_observation_matrix(x, y, z, w, tx, ty, nx, ny, kx, ky)
        
        # Solve least squares problem
        c = solve_least_squares(A, b)
        
        # Compute fp
        fp = compute_fp(x, y, z, w, tx, ty, c, nx, ny, kx, ky)
        
        # Check convergence
        if fp <= s:
            break
        
        # Check if we can add more knots
        if nx >= nxest - kx1 or ny >= nyest - ky1:
            break
        
        # Find where to add knots
        best_x, best_y, res_x, res_y = find_knot_insertion_point(
            x, y, z, w, tx, ty, c, nx, ny, kx, ky)
        
        # Add knot in direction with larger residual
        if res_x >= res_y and best_x >= 0 and nx < nxest - kx1:
            # Add knot in x
            new_knot = 0.5 * (tx[best_x] + tx[best_x + 1])
            # Shift knots
            for i in range(nx, best_x, -1):
                tx[i + 1] = tx[i]
            tx[best_x + 1] = new_knot
            nx += 1
        elif best_y >= 0 and ny < nyest - ky1:
            # Add knot in y
            new_knot = 0.5 * (ty[best_y] + ty[best_y + 1])
            # Shift knots
            for i in range(ny, best_y, -1):
                ty[i + 1] = ty[i]
            ty[best_y + 1] = new_knot
            ny += 1
        else:
            # No good place to add knots
            break
        
        # Check improvement
        if fp_old - fp < eps * fp_old:
            break
        
        fp_old = fp
    
    # Return results
    nk1x = nx - kx - 1
    nk1y = ny - ky - 1
    ncof = nk1x * nk1y
    
    return tx[:nx].copy(), ty[:ny].copy(), c[:ncof].copy(), fp


def bisplrep_dierckx(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
                     kx=3, ky=3, s=0.0, nxest=None, nyest=None, eps=1e-16,
                     task=0):
    """
    DIERCKX-compatible bisplrep implementation.
    
    Parameters match scipy.interpolate.bisplrep.
    """
    # Convert to arrays
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64).ravel()
    
    m = len(x)
    
    # Set weights
    if w is None:
        w = np.ones(m, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
    
    # Set boundaries
    if xb is None:
        xb = x.min()
    if xe is None:
        xe = x.max()
    if yb is None:
        yb = y.min()
    if ye is None:
        ye = y.max()
    
    # Estimate knot numbers
    if nxest is None:
        nxest = max(int(kx + np.sqrt(m/2)), 2*kx + 3)
    if nyest is None:
        nyest = max(int(ky + np.sqrt(m/2)), 2*ky + 3)
    
    # Ensure adequate size
    nxest = max(nxest, 2*(kx+1) + 4)
    nyest = max(nyest, 2*(ky+1) + 4)
    
    # Call SURFIT
    tx, ty, c, fp = dierckx_surfit(x, y, z, w, xb, xe, yb, ye, 
                                   kx, ky, s, nxest, nyest, eps)
    
    return (tx, ty, c, kx, ky)