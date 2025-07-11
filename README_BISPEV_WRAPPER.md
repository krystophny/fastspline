# DIERCKX bispev C Wrapper

This directory contains a C wrapper around the DIERCKX `bispev` Fortran routine for bivariate spline evaluation, making it accessible as a C function that can be called via ctypes or integrated with Numba.

## Overview

The wrapper provides:
- Direct C interface to the Fortran `bispev` routine
- Python ctypes interface for easy access from Python
- Validation tests ensuring bit-exact compatibility with scipy
- Performance benchmarks comparing with scipy.interpolate.bisplev
- Example of Numba integration for JIT-compiled code

## Building

```bash
make
```

This will compile the Fortran sources and C wrapper into a shared library `libbispev.so`.

## Files

- `src/fortran/` - DIERCKX Fortran sources (bispev.f, fpbisp.f, fpbspl.f)
- `src/c/bispev_wrapper.c` - C wrapper implementation
- `include/bispev_wrapper.h` - C header file
- `bispev_ctypes.py` - Python ctypes interface
- `tests/test_bispev_validation.py` - Validation tests against scipy
- `benchmark_bispev.py` - Performance benchmarks
- `bispev_numba_simple.py` - Example usage from Python/Numba

## Usage

### From Python (ctypes)

```python
from bispev_ctypes import bispev

# Evaluate bivariate spline
z = bispev(tx, ty, c, kx, ky, x, y)
```

### From C

```c
#include "bispev_wrapper.h"

int result = bispev_c(tx, nx, ty, ny, c, kx, ky, 
                      x, mx, y, my, z, wrk, lwrk, 
                      iwrk, kwrk, &ier);
```

## Performance

The ctypes wrapper shows overhead compared to scipy for small grids but becomes competitive for larger evaluations:

- 10x10 grid: ~5x slower than scipy (ctypes overhead dominates)
- 100x100 grid: ~20% slower than scipy (reasonable overhead)

For high-performance applications, direct C integration or Numba cfunc would eliminate the ctypes overhead.

## Validation

All tests pass with bit-exact accuracy (rtol=1e-14) compared to scipy.interpolate.bisplev, confirming the wrapper correctly interfaces with the DIERCKX routines.