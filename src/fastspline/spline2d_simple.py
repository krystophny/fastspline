"""Simple, fast bisplev implementation."""

import numpy as np
from numba import cfunc, types


@cfunc(types.float64(types.float64, types.float64, types.float64[:], types.float64[:],
                     types.float64[:], types.int64, types.int64), 
       nopython=True, fastmath=True, boundscheck=False, cache=True)
def bisplev_scalar_simple(x, y, tx, ty, c, kx, ky):
    """
    Simple B-spline evaluation - prioritizes correctness over extreme optimization.
    """
    nx = len(tx)
    ny = len(ty)
    mx = nx - kx - 1
    my = ny - ky - 1
    
    # Find knot spans using simple search
    # X direction
    span_x = kx
    for i in range(kx, nx - kx):
        if x < tx[i + 1]:
            span_x = i
            break
    
    # Y direction  
    span_y = ky
    for i in range(ky, ny - ky):
        if y < ty[i + 1]:
            span_y = i
            break
    
    # Compute basis functions using de Boor's algorithm
    # X basis
    Nx = np.zeros(kx + 1)
    Nx[0] = 1.0
    for j in range(1, kx + 1):
        for k in range(j):
            if k == j - 1:
                alpha = (x - tx[span_x - j + 1 + k]) / (tx[span_x + 1] - tx[span_x - j + 1 + k])
                Nx[k] = (1.0 - alpha) * Nx[k]
                Nx[k + 1] = alpha * Nx[k]
            else:
                alpha1 = (x - tx[span_x - j + 1 + k]) / (tx[span_x + 1 + k - j + 1] - tx[span_x - j + 1 + k])
                alpha2 = (tx[span_x + 1 + k] - x) / (tx[span_x + 1 + k] - tx[span_x + k - j + 1])
                if k == 0:
                    Nx[k] = alpha2 * Nx[k]
                else:
                    Nx[k] = alpha1 * Nx[k - 1] + alpha2 * Nx[k]
                    
    # Y basis  
    Ny = np.zeros(ky + 1)
    Ny[0] = 1.0
    for j in range(1, ky + 1):
        for k in range(j):
            if k == j - 1:
                alpha = (y - ty[span_y - j + 1 + k]) / (ty[span_y + 1] - ty[span_y - j + 1 + k])
                Ny[k] = (1.0 - alpha) * Ny[k]
                Ny[k + 1] = alpha * Ny[k]
            else:
                alpha1 = (y - ty[span_y - j + 1 + k]) / (ty[span_y + 1 + k - j + 1] - ty[span_y - j + 1 + k])
                alpha2 = (ty[span_y + 1 + k] - y) / (ty[span_y + 1 + k] - ty[span_y + k - j + 1])
                if k == 0:
                    Ny[k] = alpha2 * Ny[k]
                else:
                    Ny[k] = alpha1 * Ny[k - 1] + alpha2 * Ny[k]
    
    # Compute tensor product
    result = 0.0
    for i in range(kx + 1):
        for j in range(ky + 1):
            coeff_idx = (span_x - kx + i) * my + (span_y - ky + j)
            if 0 <= coeff_idx < len(c):
                result += Nx[i] * Ny[j] * c[coeff_idx]
    
    return result