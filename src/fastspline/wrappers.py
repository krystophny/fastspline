"""High-level wrappers for bisplrep/bisplev with SciPy-compatible interface."""

import numpy as np
from .spline2d import bisplev as bisplev_cfunc
from .bisplrep_full import bisplrep


def bisplev(x, y, tck, dx=0, dy=0):
    """
    Evaluate a bivariate B-spline and its derivatives.
    
    SciPy-compatible interface for bisplev.
    
    Parameters
    ----------
    x, y : array_like
        1-D arrays of coordinates.
    tck : tuple
        A tuple (tx, ty, c, kx, ky) containing the knot locations, coefficients,
        and degrees, as returned by bisplrep.
    dx, dy : int, optional
        The partial derivatives to be evaluated.
        
    Returns
    -------
    z : ndarray
        The interpolated values.
    """
    if dx != 0 or dy != 0:
        raise NotImplementedError("Derivatives not yet implemented")
    
    tx, ty, c, kx, ky = tck
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    x_is_scalar = x.ndim == 0
    y_is_scalar = y.ndim == 0
    
    # Convert to arrays
    if x_is_scalar:
        x = np.array([float(x)])
    if y_is_scalar:
        y = np.array([float(y)])
        
    x = x.ravel()
    y = y.ravel()
    
    # Allocate result
    if len(x) == len(y):
        result = np.zeros(len(x), dtype=np.float64)
    else:
        result = np.zeros((len(y), len(x)), dtype=np.float64)
    
    # Call low-level function
    bisplev_cfunc(x, y, tx, ty, c, kx, ky, result.T if len(x) != len(y) else result.reshape(-1, 1))
    
    # Handle scalar output
    if x_is_scalar and y_is_scalar:
        return float(result.flat[0])
    
    return result