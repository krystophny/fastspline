"""FastSpline - Fast spline interpolation library with numba acceleration."""

from .spline1d import Spline1D
from .spline2d import (
    Spline2D, bisplrep, bisplev_python as bisplev,
    bisplev as bisplev_cfunc,
    bisplev_grid_cfunc, bisplev_points_cfunc
)

__version__ = "0.1.0"
__all__ = ["Spline1D", "Spline2D", "bisplrep", "bisplev", 
           "bisplev_cfunc", "bisplev_grid_cfunc", "bisplev_points_cfunc"]