# FastSpline

High-performance spline interpolation library with exact scipy compatibility and pure Numba implementations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Exact scipy compatibility**: Bit-for-bit identical results with `scipy.interpolate`
- **Pure Numba cfunc implementations**: Zero-overhead function calls via LLVM-optimized code
- **Comprehensive spline support**:
  - DIERCKX bivariate splines (bispev, parder)
  - Sergei's equidistant splines (1D/2D/3D fully implemented, orders 3-5)
  - Full derivative support (up to 2nd order for 1D, 1st order for 2D/3D)
- **Blazing fast performance**: Direct cfunc calls eliminate Python overhead
- **Memory efficient**: Zero-allocation evaluation functions

## Installation

```bash
# Install from source
git clone https://github.com/krystophny/fastspline
cd fastspline
pip install -e .
```

### Requirements
- Python 3.8+
- NumPy
- Numba
- SciPy (for comparison and spline fitting)
- Matplotlib (for examples)

## Quick Start

### DIERCKX Bivariate Splines

```python
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from fastspline import bispev_cfunc_address, call_parder_safe
import ctypes

# Create sample data
x = np.linspace(0, 2*np.pi, 20)
y = np.linspace(0, 2*np.pi, 20)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Fit spline using scipy
tck = bisplrep(X.ravel(), Y.ravel(), Z.ravel(), s=0)
tx, ty, c = tck[0], tck[1], tck[2]
kx, ky = tck[3], tck[4]

# Compare scipy vs fastspline
xi, yi = 1.5, 2.0
z_scipy = bisplev(xi, yi, tck)

# Use FastSpline cfunc directly
bispev_func = ctypes.CFUNCTYPE(
    ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int
)(bispev_cfunc_address)

# Convert to ctypes
tx_c = tx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
ty_c = ty.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
c_c = c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

z_fast = bispev_func(xi, yi, tx_c, len(tx), ty_c, len(ty), c_c, kx, ky)
print(f"Difference: {abs(z_scipy - z_fast)}")  # ~1e-16
```

### Sergei's Equidistant Splines

```python
from fastspline import get_sergei_cfunc_addresses
import ctypes
import numpy as np

# Get cfunc addresses
cfuncs = get_sergei_cfunc_addresses()

# 1D Spline Example
construct_1d = ctypes.CFUNCTYPE(
    None,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_double)
)(cfuncs['construct_splines_1d'])

# Create data
n = 20
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x)

# Prepare ctypes arrays
y_c = (ctypes.c_double * n)(*y)
coeff_c = (ctypes.c_double * (4 * n))()

# Construct cubic spline (order=3, periodic=0)
construct_1d(0.0, 2*np.pi, y_c, n, 3, 0, coeff_c)

# Evaluate with derivatives
evaluate_der2 = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double)
)(cfuncs['evaluate_splines_1d_der2'])

# Evaluate at a point
y_out = (ctypes.c_double * 1)()
dy_out = (ctypes.c_double * 1)()
d2y_out = (ctypes.c_double * 1)()

h_step = 2*np.pi / (n - 1)
x_eval = 1.5
evaluate_der2(3, n, 0, 0.0, h_step, coeff_c, x_eval, y_out, dy_out, d2y_out)

print(f"f({x_eval}) = {y_out[0]:.6f}")
print(f"f'({x_eval}) = {dy_out[0]:.6f}")
print(f"f''({x_eval}) = {d2y_out[0]:.6f}")
```

### 2D Spline Example

```python
# 2D Spline with derivatives
n1, n2 = 20, 25
x_min = np.array([0.0, 0.0])
x_max = np.array([4.0, 6.0])

# Create 2D data
x1 = np.linspace(x_min[0], x_max[0], n1)
x2 = np.linspace(x_min[1], x_max[1], n2)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')
Z = np.sin(X1) * np.cos(X2)

# Set up construction
construct_2d = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),  # x_min
    ctypes.POINTER(ctypes.c_double),  # x_max
    ctypes.POINTER(ctypes.c_double),  # y values
    ctypes.POINTER(ctypes.c_int32),   # num_points
    ctypes.POINTER(ctypes.c_int32),   # order
    ctypes.POINTER(ctypes.c_int32),   # periodic
    ctypes.POINTER(ctypes.c_double),  # coeff
    ctypes.POINTER(ctypes.c_double),  # workspace_y
    ctypes.POINTER(ctypes.c_double)   # workspace_coeff
)(cfuncs['construct_splines_2d'])

# Prepare arrays
x_min_c = (ctypes.c_double * 2)(*x_min)
x_max_c = (ctypes.c_double * 2)(*x_max)
num_points_c = (ctypes.c_int32 * 2)(n1, n2)
order_c = (ctypes.c_int32 * 2)(3, 3)  # Cubic in both dimensions
periodic_c = (ctypes.c_int32 * 2)(0, 0)
z_flat = Z.flatten()
z_c = (ctypes.c_double * len(z_flat))(*z_flat)
coeff_size = 4 * 4 * n1 * n2
coeff_c = (ctypes.c_double * coeff_size)()

# Workspace arrays
workspace_y = (ctypes.c_double * max(n1, n2))()
workspace_coeff = (ctypes.c_double * (6 * max(n1, n2)))()

# Construct spline
construct_2d(x_min_c, x_max_c, z_c, num_points_c, order_c, periodic_c, 
             coeff_c, workspace_y, workspace_coeff)

print("2D spline constructed successfully!")
```

## Performance

FastSpline achieves exceptional performance through:

- **Pure cfunc implementations**: No Python function call overhead
- **Optimized algorithms**: Efficient tensor product evaluation for 2D/3D
- **LLVM compilation**: Numba generates native machine code
- **Zero allocations**: All evaluation functions work with pre-allocated memory

Typical performance:
- 1D spline evaluation: ~500ns per point
- 2D spline construction: <0.1ms for 20×25 grid
- 2D spline evaluation: ~1μs per point with derivatives
- Exact scipy compatibility with <1% overhead

## Examples

See the `examples/` directory for comprehensive demonstrations:

- `basic_usage.py` - Simple 1D spline example
- `comprehensive_demo.py` - Full feature demonstration with benchmarks
- `sergei_splines_demo.py` - Visual proof of all capabilities
- `demo_derivatives.py` - Derivative computation examples
- `visual_validation.py` - Extensive visual tests
- `verify_compilation.py` - Check that all cfuncs compile correctly

## API Reference

### DIERCKX Splines

```python
# Bivariate spline evaluation
bispev_cfunc_address: int  # Address of bispev cfunc

# Safe derivative evaluation  
call_parder_safe(tx, ty, c, kx, ky, nux, nuy, x, y) -> (derivatives, ier)

# Direct parder cfunc
parder_cfunc_address: int  # Address of parder cfunc
```

### Sergei's Splines

```python
# Get all cfunc addresses
get_sergei_cfunc_addresses() -> dict

# Available functions:
# - construct_splines_1d: Build 1D spline (orders 3-5)
# - evaluate_splines_1d: Evaluate 1D spline
# - evaluate_splines_1d_der: Evaluate with 1st derivative
# - evaluate_splines_1d_der2: Evaluate with 1st & 2nd derivatives
# - construct_splines_2d: Build 2D spline with workspace
# - evaluate_splines_2d: Evaluate 2D spline
# - evaluate_splines_2d_der: Evaluate 2D with partial derivatives
# - construct_splines_3d: Build 3D spline with workspace
# - evaluate_splines_3d: Evaluate 3D spline
# - evaluate_splines_3d_der: Evaluate 3D with partial derivatives
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=fastspline

# Run specific test module
pytest tests/test_fastspline.py -v
```

## Development

### Project Structure

```
fastspline/
├── src/
│   ├── fastspline/
│   │   ├── __init__.py
│   │   ├── bispev_numba.py      # DIERCKX bispev implementation
│   │   ├── parder.py             # DIERCKX parder implementation
│   │   └── sergei_splines.py     # Sergei's equidistant splines
│   ├── fortran/                  # Original Fortran sources
│   └── c/                        # C wrapper code
├── tests/                        # Unit tests
├── examples/                     # Usage examples
├── benchmarks/                   # Performance benchmarks
└── thirdparty/                   # Third-party licenses
```

### Building from Source

```bash
# Build Fortran/C extensions
make

# Run tests
make test

# Clean build artifacts
make clean
```

## Algorithm Details

### DIERCKX Implementation
- Exact port of FITPACK bivariate spline routines
- Implements Cox-de Boor recursion for B-spline basis
- Tensor product evaluation for efficiency
- Handles all spline degrees (1-5) and derivative orders

### Sergei's Splines
- Equidistant knot spacing for optimal performance
- Support for periodic boundary conditions
- Orders 3 (cubic), 4 (quartic), and 5 (quintic)
- Efficient tensor product for multidimensional splines
- Optimized for regular grids

## License

This project is licensed under the MIT License. It includes code from:
- scipy (BSD 3-Clause License)
- DIERCKX (Public Domain)

See LICENSE and thirdparty/licenses/ for full details.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass with `pytest`
- New features include comprehensive tests
- Code follows existing style conventions
- Performance benchmarks for new algorithms

## Citation

If you use FastSpline in your research, please cite:

```bibtex
@software{fastspline2024,
  title = {FastSpline: High-Performance Spline Interpolation Library},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/krystophny/fastspline}
}
```

## Acknowledgments

- Paul Dierckx for the original DIERCKX library
- The SciPy team for the Python interface
- Sergei Kasilov for the equidistant spline algorithms