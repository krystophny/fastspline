"""
FastSpline - High-performance spline interpolation library

Provides optimized implementations of univariate and bivariate spline
interpolation algorithms with exact scipy compatibility.
"""

__version__ = "0.2.0"

# Import main APIs
from .bispev_numba import bispev_cfunc_address
from .parder import call_parder_safe, parder_cfunc_address
from .sergei_splines import get_cfunc_addresses as get_sergei_cfunc_addresses

__all__ = [
    'bispev_cfunc_address',
    'call_parder_safe', 
    'parder_cfunc_address',
    'get_sergei_cfunc_addresses'
]