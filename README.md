# FastSpline

High-performance B-spline interpolation library with C-compatible interface, optimized using Numba.

## Features

- **Performance at Scale**: Up to 1.78x faster than SciPy for large grid evaluations (1024×1024)
- **C-Compatible**: Direct C function interface via Numba cfunc for multi-language interoperability
- **Automatic Meshgrid**: Smart detection - same length arrays → pointwise, different → meshgrid
- **Machine Precision**: Maintains numerical accuracy comparable to SciPy
- **Memory Efficient**: Pre-allocated arrays with zero overhead

## Installation

```bash
pip install -e .
```

## Quick Start

### 1D Spline Interpolation

```python
import numpy as np
from fastspline import Spline1D

# Create data
x = np.linspace(0, 10, 11)
y = np.sin(x)

# Create spline (cubic by default)
spline = Spline1D(x, y)

# Evaluate at new points
x_new = np.linspace(0, 10, 100)
y_new = spline(x_new)
```

### 2D Spline Surface Fitting (bisplrep/bisplev)

```python
from fastspline import bisplrep, bisplev_scalar

# Scattered data points
x = np.random.uniform(-1, 1, 100)
y = np.random.uniform(-1, 1, 100)
z = np.exp(-(x**2 + y**2))

# Fit B-spline surface
tck = bisplrep(x, y, z, kx=3, ky=3)

# Evaluate at a point
z_interp = bisplev_scalar(0.5, 0.5, *tck)
```

### Array Evaluation with Automatic Meshgrid

```python
from fastspline import bisplev

# Fit surface (as above)
tck = bisplrep(x, y, z)

# Evaluate on arrays
x_eval = np.linspace(-1, 1, 50)
y_eval = np.linspace(-1, 1, 30)

# Automatically creates meshgrid when lengths differ
result = np.zeros((50, 30))
bisplev(x_eval, y_eval, *tck, result)
```

### C-Compatible Function Usage

```python
from fastspline import bisplev_scalar
import ctypes

# Get C function pointer
bisplev_addr = bisplev_scalar.address

# Can be called from C/C++ code directly
# double result = bisplev(x, y, tx, ty, c, kx, ky);
```

## Performance

FastSpline performance compared to SciPy:

- **Large Grids**: Up to 1.78x faster for 1024×1024 meshgrid evaluation
- **Meshgrid Advantage**: 55x speedup when using automatic meshgrid vs scattered points
- **Scaling**: Better O(N²) scaling characteristics - 3x more efficient growth than SciPy
- **C Interoperability**: Direct function addresses for zero-overhead calls from C/C++/Fortran

### Benchmark Results (k=3 cubic splines)
| Grid Size | SciPy (ms) | FastSpline (ms) | Speedup |
|-----------|------------|-----------------|---------|
| 512×512   | 2.32       | 2.16            | 1.07x   |
| 768×768   | 4.60       | 3.31            | 1.39x   |
| 1024×1024 | 7.73       | 4.33            | 1.78x   |

Run benchmarks:
```bash
python benchmarks/benchmark_comprehensive.py
python benchmarks/scaling_analysis.py  # Log-log scaling plots
```

## API Reference

### Core Functions

- `Spline1D(x, y, order=3)`: 1D spline interpolation
- `bisplrep(x, y, z, kx=3, ky=3, s=0)`: Fit 2D B-spline surface
- `bisplev(x, y, tx, ty, c, kx, ky, result)`: Evaluate B-spline (array interface)
- `bisplev_scalar(x, y, tx, ty, c, kx, ky)`: Evaluate B-spline (scalar)

### Key Advantages

1. **Automatic Meshgrid Detection**: When x and y have different lengths, automatically evaluates on meshgrid
2. **C-Compatible Interface**: Direct function addresses for integration with C/C++/Fortran/Julia
3. **Pre-allocated Memory**: Zero allocation overhead during evaluation
4. **Thread-Safe**: Pure C functions without Python GIL restrictions

### Compatibility Note

FastSpline uses the same algorithms as SciPy but with different implementation details. While it maintains numerical accuracy, the fitted coefficients may differ slightly due to numerical precision in the fitting process.

## Project Structure

```
fastspline/
├── src/fastspline/     # Core implementation
├── tests/              # Unit tests (no plots)
├── benchmarks/         # Performance benchmarks
├── examples/           # Usage examples and visual demos
└── tools/              # Development utilities
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.

## Development

Run tests:
```bash
pytest tests/
```

Profile performance:
```bash
python tools/profile_bottlenecks.py
```

## License

MIT License - see [LICENSE](LICENSE) file.