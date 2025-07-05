"""
FastSpline - Ultra-fast B-spline interpolation for Python.

A high-performance alternative to scipy.interpolate for B-spline operations,
optimized with Numba for 5-10x speedups on common workloads.
"""

from .spline1d import Spline1D
from .spline2d import Spline2D
from .bisplrep_dierckx import bisplrep_dierckx as bisplrep
from .bisplev_fast import bisplev_fast as bisplev, bisplev_scalar_fast as bisplev_scalar

__version__ = "0.1.0"
__all__ = ["Spline1D", "Spline2D", "bisplrep", "bisplev", "bisplev_scalar"]