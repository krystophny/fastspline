# FastSpline

Fast spline interpolation library with Numba acceleration for 1D and 2D B-splines.

## Features

- **Fast 1D spline interpolation** with periodic boundary conditions
- **2D tensor product B-splines** for regular grids
- **SciPy-compatible bisplrep/bisplev** for scattered data interpolation
- **Numba cfunc implementation** for C-level performance
- **Machine precision accuracy** compared to reference implementations

## Installation

```bash
pip install -e .
```

## Usage

### 1D Spline Interpolation

```python
import numpy as np
from fastspline import Spline1D

# Create data
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)

# Create spline (cubic by default)
spline = Spline1D(x, y)

# Evaluate at new points
x_new = np.linspace(0, 2*np.pi, 100)
y_new = spline(x_new)
```

### 2D Regular Grid Interpolation

```python
from fastspline import Spline2D

# Create 2D grid data
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
xx, yy = np.meshgrid(x, y)
z = np.exp(-(xx**2 + yy**2))

# Create 2D spline
spline2d = Spline2D(x, y, z)

# Evaluate at new points
x_new = np.linspace(-1, 1, 100)
y_new = np.linspace(-1, 1, 100)
z_new = spline2d(x_new, y_new)
```

### 2D Scattered Data Interpolation (SciPy-compatible bisplrep/bisplev)

```python
import numpy as np
from fastspline import bisplrep, bisplev

# Generate scattered data points
np.random.seed(42)
x = np.random.uniform(-1, 1, 100)
y = np.random.uniform(-1, 1, 100)
z = np.exp(-(x**2 + y**2)) * np.cos(np.pi * x)

# Fit B-spline surface using bisplrep (uses SciPy internally)
tck = bisplrep(x, y, z, kx=3, ky=3, s=0)

# Evaluate at a single point
x_eval, y_eval = 0.5, 0.5
z_value = bisplev([x_eval], [y_eval], tck, grid=False)[0]

# Grid evaluation (default behavior, creates meshgrid)
x_grid = np.linspace(-0.9, 0.9, 50)
y_grid = np.linspace(-0.9, 0.9, 50)
z_grid = bisplev(x_grid, y_grid, tck)  # Returns 50x50 array

# Scattered points evaluation (much faster than SciPy)
x_scatter = np.random.uniform(-0.9, 0.9, 1000)
y_scatter = np.random.uniform(-0.9, 0.9, 1000)
z_scatter = bisplev(x_scatter, y_scatter, tck, grid=False)  # Returns 1000-element array
```

### bisplev Modes and Performance

FastSpline's `bisplev` supports different evaluation modes:

1. **Grid mode (default)**: `bisplev(x, y, tck, grid=True)`
   - Creates meshgrid from x and y arrays (like SciPy)
   - Returns 2D array of shape (len(x), len(y))
   - Use for regular grid evaluation

2. **Scattered points mode**: `bisplev(x, y, tck, grid=False)`
   - Evaluates at (x[i], y[i]) pairs
   - Returns 1D array of length min(len(x), len(y))
   - **~25-40x faster than SciPy** for scattered points

3. **Direct cfunc access** (for maximum performance):
   ```python
   from fastspline import bisplev_cfunc
   
   # Single point evaluation (fastest for tight loops)
   tx, ty, c, kx, ky = tck
   z_value = bisplev_cfunc(x_point, y_point, tx, ty, c, kx, ky)
   ```

## Performance

### bisplev Performance (vs SciPy)
- **Scattered points (grid=False)**: ~25-40x faster than SciPy
- **Grid evaluation (grid=True)**: SciPy is faster (optimized Fortran)
- **Single point evaluation**: Use bisplev_cfunc for best performance
- **Machine precision accuracy**: Differences typically < 1e-15

### Key Optimizations
- Cox-de Boor recursion with optimized memory access
- Numba JIT compilation with cfunc for C-level performance
- Cache-friendly data layout for coefficients
- Specialized implementations for linear vs cubic splines

## Advanced Usage

### Direct cfunc Interface

For maximum performance in Numba-compiled code:

```python
from numba import njit
from fastspline import bisplev_cfunc, bisplev_points_cfunc

@njit
def evaluate_many_points(x_points, y_points, tx, ty, c, kx, ky):
    """Evaluate spline at many points with minimal overhead."""
    n = len(x_points)
    results = np.zeros(n)
    
    for i in range(n):
        results[i] = bisplev_cfunc(x_points[i], y_points[i], tx, ty, c, kx, ky)
    
    return results

# Or use the pre-compiled cfunc for scattered points
result_buffer = np.zeros(len(x_points))
bisplev_points_cfunc(x_points, y_points, tx, ty, c, kx, ky, result_buffer)
```

## Implementation Details

- **bisplev**: Ultra-optimized B-spline evaluation using the Cox-de Boor algorithm
- **bisplrep**: Currently uses SciPy for fitting (our cfunc version is experimental)
- **Boundary handling**: Proper knot multiplicity for accurate boundary evaluation
- **Numerical stability**: Careful handling of edge cases and division by zero

## Requirements

- Python >= 3.8
- NumPy
- Numba
- SciPy (for bisplrep and comparison)