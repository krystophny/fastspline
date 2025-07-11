# FastSpline

High-performance bivariate spline interpolation implementations exploring various optimization strategies for the DIERCKX Fortran library.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FastSpline is a Python package that provides multiple high-performance implementations of bivariate spline evaluation:

1. **Fortran/C Wrapper** - Direct C wrapper around the original DIERCKX Fortran code
2. **Pure Numba Implementation** - Complete rewrite in Python/Numba as cfuncs with `nopython=True`

Both implementations provide bit-exact compatibility with scipy's `bisplev` function while exploring different performance optimization strategies.

## Installation

### From Source
```bash
git clone https://github.com/krystophny/fastspline.git
cd fastspline
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/krystophny/fastspline.git
cd fastspline
pip install -e ".[dev,numba]"
```

## Package Structure

```
fastspline/
├── fastspline/                  # Python package
│   ├── ctypes_wrapper/          # Python ctypes interface
│   └── numba_implementation/    # Pure Numba cfunc implementation
├── src/                         # Source code
│   ├── fortran/                 # DIERCKX Fortran sources
│   └── c/                       # C wrapper implementation
├── benchmarks/                  # Performance comparisons
├── tests/                       # Test suites
├── thirdparty/licenses/         # Third-party licenses
└── docs/                        # Documentation
```

## Quick Start

### Using the ctypes wrapper
```python
import fastspline

# Use the ctypes interface to DIERCKX Fortran routines
z = fastspline.bispev_ctypes(tx, ty, c, kx, ky, x, y)
```

### Using Numba cfuncs (requires numba)
```python
import ctypes
from fastspline.numba_implementation import bispev_cfunc_address

# Create ctypes wrapper for the cfunc
bispev_numba = ctypes.CFUNCTYPE(...)(bispev_cfunc_address)
```

## Performance

Benchmark results for 100x100 evaluation grid:
- scipy.interpolate.bisplev: 0.083 ms (baseline)
- FastSpline Fortran wrapper: 0.097 ms (+17% overhead)
- FastSpline Numba cfunc: 0.108 ms (+30% overhead)

The overhead is primarily from the ctypes interface. When called directly from compiled code, both implementations have negligible overhead.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: This project incorporates code from third-party sources:

- **DIERCKX FITPACK routines** (in `src/fortran/`) - Used under the same license terms as SciPy (BSD 3-Clause)
- **SciPy components** - Licensed under BSD 3-Clause License

See `thirdparty/licenses/` for complete license information for third-party components.

**All code not directly copied or ported from third-party libraries is licensed under the MIT License.**

## Implementation Details

### Fortran/C Wrapper
- Minimal C wrapper around original DIERCKX Fortran routines
- Handles Fortran calling conventions and array layouts
- Provides exact compatibility with scipy

### Numba Implementation
- Complete rewrite of fpbspl, fpbisp, and bispev in pure Python/Numba
- All functions are cfuncs with `nopython=True` and `fastmath=True`
- Algorithms are inlined to avoid function pointer limitations
- Careful translation of Fortran 1-based to Python 0-based indexing

## Testing

Both implementations are validated to provide bit-exact results (rtol=1e-14) compared to scipy.interpolate.bisplev.

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.