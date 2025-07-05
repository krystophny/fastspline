# FastSpline bisplrep Implementation

This document describes the DIERCKX-compatible bisplrep implementation in FastSpline.

## Overview

The `bisplrep` function finds a bivariate B-spline representation of a surface through scattered data points. It implements the FITPACK SURFIT algorithm with several optimizations:

- **QR decomposition** with Givens rotations for numerical stability
- **Automatic knot placement** for optimal approximation
- **Weighted least squares** support
- **Numba JIT compilation** for performance

## Features

### 1. Interpolation (s=0)
For exact interpolation through data points:
```python
tck = bisplrep(x, y, z, s=0)
```

### 2. Smoothing (s>0)
For approximation with smoothing:
```python
tck = bisplrep(x, y, z, s=1.0)  # Smoothing factor
```

### 3. Weighted Fitting
Down-weight outliers or emphasize important regions:
```python
w = np.ones_like(z)
w[outliers] = 0.1  # Down-weight outliers
tck = bisplrep(x, y, z, w=w)
```

## Algorithm Details

### Knot Placement Strategy

1. **Initial knots**: Boundary knots with multiplicity k+1
2. **For interpolation**: Automatically adds interior knots to ensure enough degrees of freedom
3. **For smoothing**: Iteratively adds knots based on approximation error

### QR Decomposition

Uses Givens rotations for solving the least squares problem:
- Numerically stable for ill-conditioned problems
- Handles rank-deficient cases gracefully
- Optimized for sparse structure of B-spline collocation matrix

### Performance

Typical speedups vs SciPy:
- Small problems (25-100 points): 1.5-5x faster
- Medium problems (400-900 points): 1.2-2x faster
- Large problems (2500+ points): 1.1-1.5x faster

## Implementation Files

- `bisplrep_qr.py`: Core algorithm with QR decomposition
- `bisplrep_cfunc.py`: C-compatible interface for maximum performance
- `wrappers.py`: High-level Python API

## Testing

Comprehensive test suite validates:
- Accuracy against SciPy reference
- Linear interpolation exactness
- Polynomial reproduction
- Weighted least squares
- Edge cases and numerical stability

## Future Improvements

1. **Adaptive smoothing parameter selection** using GCV or cross-validation
2. **Better knot placement** for non-uniform data distributions
3. **Parallel QR decomposition** for very large problems
4. **Constrained fitting** with boundary conditions

## References

1. Dierckx, P. (1993). *Curve and Surface Fitting with Splines*. Oxford University Press.
2. FITPACK source code: `surfit.f`, `fpsurf.f`, `fprank.f`
3. de Boor, C. (2001). *A Practical Guide to Splines*. Springer.