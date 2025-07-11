"""
FastSpline: High-performance bivariate spline interpolation implementations.

This package provides multiple implementations of bivariate spline evaluation:

1. ctypes_wrapper: Direct interface to DIERCKX Fortran routines via ctypes
2. numba_implementation: Pure Python/Numba cfunc implementation

All implementations provide bit-exact compatibility with scipy.interpolate.bisplev
while exploring different performance optimization strategies.
"""

__version__ = "0.1.0"
__author__ = "FastSpline Contributors"

# Import main interfaces
try:
    from .ctypes_wrapper import bispev as bispev_ctypes
    __all__ = ['bispev_ctypes']
except ImportError:
    __all__ = []

# Optional Numba imports
try:
    from .numba_implementation import (
        bispev_cfunc_address,
        fpbisp_cfunc_address, 
        fpbspl_cfunc_address
    )
    __all__.extend([
        'bispev_cfunc_address',
        'fpbisp_cfunc_address',
        'fpbspl_cfunc_address'
    ])
except ImportError:
    pass

# Expose subpackages
from . import ctypes_wrapper
try:
    from . import numba_implementation
    __all__.extend(['ctypes_wrapper', 'numba_implementation'])
except ImportError:
    __all__.append('ctypes_wrapper')