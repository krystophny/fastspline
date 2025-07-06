# FastSpline - High-Performance DIERCKX Implementation

Ultra-optimized Numba cfunc implementation of DIERCKX spline library.

## Directory Structure

```
fastspline/
├── dierckx_cfunc.py          # Main cfunc implementation
├── build_dierckx_f2py.sh     # Build script for DIERCKX reference
├── tests/
│   └── test_dierckx_cfunc.py # Validation tests against DIERCKX
├── examples/
│   ├── usage_example.py      # How to use the functions
│   └── benchmark_performance.py # Performance benchmarks
└── thirdparty/dierckx/       # Original FORTRAN source
```

## Quick Start

```bash
# Build DIERCKX f2py reference
./build_dierckx_f2py.sh

# Run validation tests
python tests/test_dierckx_cfunc.py

# Run performance benchmark
python examples/benchmark_performance.py

# See usage examples
python examples/usage_example.py
```

## Key Features

- **Ultra-optimized cfunc implementations** with maximum Numba performance flags
- **Validated against DIERCKX** f2py reference to floating point accuracy
- **Performance advantage** for larger problems (up to 1.15× speedup)
- **Clean API** matching original DIERCKX interface

## Implemented Functions

- `fpback` - Backward substitution for banded matrices
- `fpgivs` - Givens rotation computation
- `fprota` - Apply rotation to two values
- `fprati` - Rational interpolation
- `fpbspl` - B-spline basis evaluation

## Performance

- **fpback**: Up to 1.15× speedup for n≥200
- **Scaling**: Performance advantage increases with problem size
- **Validation**: All functions match DIERCKX to floating point precision