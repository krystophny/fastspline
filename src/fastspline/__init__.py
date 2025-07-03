"""FastSpline - Fast spline interpolation library with numba acceleration."""

from .spline1d import Spline1D
from .spline2d import Spline2D, bisplrep, bisplev

__version__ = "0.1.0"
__all__ = ["Spline1D", "Spline2D", "bisplrep", "bisplev"]