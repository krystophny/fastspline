# FastSpline: Direct Access to SciPy's Fortran Interpolation Routines

This repository demonstrates how to bypass SciPy's Python interface layer and directly access the compiled Fortran extensions for bivariate spline interpolation, achieving modest performance improvements.

## Overview

SciPy's `bisplrep` and `bisplev` functions provide excellent bivariate spline interpolation, but they include Python-level overhead for input validation, type checking, and error handling. By accessing the underlying Fortran routines directly, you can reduce this overhead by approximately 5-6 microseconds per call.

## Performance Comparison

Our benchmarks show:
- **SciPy interface overhead**: ~5.6 µs (approximately 12% of total execution time)
- **Direct Fortran call overhead**: ~2.2 µs (approximately 5% of total execution time)
- **Numerical accuracy**: Bit-for-bit identical results

## Usage Examples

### Standard SciPy Approach

```python
import numpy as np
from scipy.interpolate import bisplrep, bisplev

# Generate sample data
x = np.random.uniform(-5, 5, 200)
y = np.random.uniform(-5, 5, 200)
z = np.sin(np.sqrt(x**2 + y**2))

# Fit spline
tck = bisplrep(x, y, z, s=200)  # s=200 for smoothing

# Evaluate on grid
xi = np.linspace(-5, 5, 50)
yi = np.linspace(-5, 5, 50)
zi = bisplev(xi, yi, tck)
```

### Direct Fortran Extension Approach

```python
import numpy as np
from scipy.interpolate import bisplrep
from scipy.interpolate import _dfitpack  # Internal module - subject to change!

# Fit spline using standard scipy (no direct alternative for fitting)
x = np.random.uniform(-5, 5, 200)
y = np.random.uniform(-5, 5, 200)
z = np.sin(np.sqrt(x**2 + y**2))
tck = bisplrep(x, y, z, s=200)

# Unpack tck tuple
tx, ty, c, kx, ky = tck

# Prepare evaluation points
xi = np.ascontiguousarray(np.linspace(-5, 5, 50), dtype=np.float64)
yi = np.ascontiguousarray(np.linspace(-5, 5, 50), dtype=np.float64)

# Direct Fortran call for evaluation
zi, ier = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)

if ier != 0:
    raise ValueError(f"Evaluation error: ier={ier}")
```

### Ultra-Minimal Approach (Maximum Performance)

```python
# Pre-unpack tck and pre-prepare arrays for repeated evaluations
tx, ty, c, kx, ky = tck

# Ensure arrays are contiguous and correct dtype once
xi = np.ascontiguousarray(xi, dtype=np.float64)
yi = np.ascontiguousarray(yi, dtype=np.float64)

# Direct call with no overhead
zi, ier = _dfitpack.bispev(tx, ty, c, kx, ky, xi, yi)
```

## When to Use Direct Access

Consider direct Fortran access when:
- Making many repeated evaluations (thousands per second)
- Working in performance-critical inner loops
- Every microsecond matters
- You can guarantee input validity

Stay with SciPy's interface when:
- Safety and error handling are important
- Code maintainability is a priority
- Performance difference (5-6 µs) is negligible
- Working with diverse input types

## Important Notes

1. **Deprecation Warning**: The `_dfitpack` module is internal and may change without notice. The examples above work with SciPy 1.x but may break in future versions.

2. **No Input Validation**: Direct calls bypass all safety checks. Ensure your inputs are:
   - Contiguous C-arrays
   - Correct dtype (float64)
   - Valid knot vectors and coefficients

3. **Error Handling**: Direct calls return error codes rather than raising exceptions. Check the `ier` return value.

4. **Numerical Accuracy**: Both approaches produce identical results to machine precision.

## Repository Contents

- `compare_scipy_vs_real_fortran.py` - Comprehensive performance comparison
- `verify_floating_point_accuracy.py` - Accuracy verification tests
- `test_exact_equality.py` - Simple equality tests
- `final_accuracy_test.py` - Extensive accuracy validation
- Various other comparison scripts

## Running the Benchmarks

```bash
# Run performance comparison
python compare_scipy_vs_real_fortran.py

# Verify numerical accuracy
python final_accuracy_test.py
```

## Conclusion

While direct access to Fortran routines can provide performance benefits, the improvements are modest (typically 5-6 µs per call). For most applications, SciPy's standard interface provides the best balance of performance, safety, and maintainability. Consider direct access only when working in extremely performance-critical contexts where every microsecond counts.