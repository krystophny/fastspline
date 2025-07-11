# FastSpline

High-performance bivariate spline interpolation implementations exploring various optimization strategies for the DIERCKX Fortran library.

## Overview

This repository contains multiple implementations of the DIERCKX `bispev` routine for bivariate spline evaluation:

1. **Fortran/C Wrapper** - Direct C wrapper around the original DIERCKX Fortran code
2. **Pure Numba Implementation** - Complete rewrite in Python/Numba as cfuncs with `nopython=True`

Both implementations provide bit-exact compatibility with scipy's `bisplev` function while exploring different performance optimization strategies.

## Repository Structure

```
fastspline/
├── src/                         # Source code
│   ├── fortran/                 # DIERCKX Fortran sources
│   └── c/                       # C wrapper implementation
├── include/                     # C header files
├── lib/                         # Compiled libraries
├── python/                      # Python implementations
│   ├── ctypes_wrapper/          # Python ctypes interface
│   └── numba_implementation/    # Pure Numba cfunc implementation
├── benchmarks/                  # Performance comparisons
│   ├── scripts/                 # Benchmark scripts
│   └── results/                 # Performance plots
├── tests/                       # Test suites
└── examples/                    # Usage examples
```

## Building

To build the C wrapper around the Fortran code:

```bash
make
```

This creates `lib/libbispev.so` which can be used via ctypes or called from C.

## Usage

### Python Ctypes Wrapper

```python
from python.ctypes_wrapper import bispev

# Evaluate bivariate spline
z = bispev(tx, ty, c, kx, ky, x, y)
```

### Numba cfunc

```python
import ctypes
from python.numba_implementation import bispev_cfunc_address

# Create ctypes wrapper for the cfunc
bispev = ctypes.CFUNCTYPE(...)(bispev_cfunc_address)
```

## Performance

Benchmark results for 100x100 evaluation grid:
- scipy.interpolate.bisplev: 0.083 ms (baseline)
- Fortran wrapper (ctypes): 0.097 ms (+17% overhead)
- Numba cfunc (ctypes): 0.108 ms (+30% overhead)

The overhead is primarily from the ctypes interface. When called directly from compiled code, both implementations have negligible overhead.

## Implementation Details

### Fortran/C Wrapper
- Minimal C wrapper around original DIERCKX Fortran routines
- Handles Fortran calling conventions and array layouts
- Provides exact compatibility with scipy

### Numba Implementation
- Complete rewrite of fpbspl, fpbisp, and bispev in pure Python/Numba
- All functions are cfuncs with nopython=True and fastmath=True
- Algorithms are inlined to avoid function pointer limitations
- Careful translation of Fortran 1-based to Python 0-based indexing

## Testing

Both implementations are validated to provide bit-exact results (rtol=1e-14) compared to scipy.interpolate.bisplev.

See the `tests/` directory for comprehensive validation tests.