# C-Function (cfunc) Usage Guide

FastSpline provides a high-performance C-compatible function `bisplev_cfunc_vectorized` that can be called directly from C, Fortran, Julia, and other languages.

## Function Signature

```c
void bisplev_cfunc_vectorized(
    double* x_arr,      // Input: X coordinates to evaluate
    double* y_arr,      // Input: Y coordinates to evaluate  
    double* tx,         // Input: X knot vector
    double* ty,         // Input: Y knot vector
    double* c,          // Input: Spline coefficients (flattened)
    int64_t kx,         // Input: X spline degree
    int64_t ky,         // Input: Y spline degree
    double* result      // Output: Evaluation results
);
```

## Memory Requirements

- All arrays must be **contiguous C-order** (row-major)
- `x_arr`, `y_arr`, and `result` must have the same length
- No bounds checking - caller must ensure valid array sizes
- Arrays must remain valid for the duration of the function call

## Language Integration Examples

### Python (using Numba's cfunc directly)
```python
from fastspline import bisplev_cfunc_vectorized, bisplrep
import numpy as np

# Fit spline
tck = bisplrep(x_data, y_data, z_data, kx=3, ky=3, s=0)
tx, ty, c, kx, ky = tck

# Evaluate at points
x_eval = np.array([0.1, 0.2, 0.3])
y_eval = np.array([0.4, 0.5, 0.6])
result = np.zeros_like(x_eval)

bisplev_cfunc_vectorized(x_eval, y_eval, tx, ty, c, kx, ky, result)
```

### C Integration
```c
#include <stdio.h>

// Declare the Numba-compiled function
extern void bisplev_cfunc_vectorized(
    double* x_arr, double* y_arr,
    double* tx, double* ty, double* c,
    int64_t kx, int64_t ky,
    double* result
);

void evaluate_spline_batch(double* x_points, double* y_points, int n_points,
                          double* tx, int nx, double* ty, int ny,
                          double* coeffs, int kx, int ky,
                          double* results) {
    // Direct call to optimized FastSpline cfunc
    bisplev_cfunc_vectorized(x_points, y_points, tx, ty, coeffs, kx, ky, results);
}
```

### Fortran Integration
```fortran
interface
    subroutine bisplev_cfunc_vectorized(x_arr, y_arr, tx, ty, c, kx, ky, result) &
        bind(C, name='bisplev_cfunc_vectorized')
        use iso_c_binding
        real(c_double), intent(in) :: x_arr(*), y_arr(*)
        real(c_double), intent(in) :: tx(*), ty(*), c(*)
        integer(c_int64_t), intent(in), value :: kx, ky
        real(c_double), intent(out) :: result(*)
    end subroutine
end interface

! Usage
call bisplev_cfunc_vectorized(x_points, y_points, tx, ty, coeffs, kx, ky, results)
```

### Julia Integration
```julia
# Load the compiled library
const libfastspline = "path/to/fastspline.so"

function evaluate_spline!(x_arr::Vector{Float64}, y_arr::Vector{Float64},
                         tx::Vector{Float64}, ty::Vector{Float64}, 
                         c::Vector{Float64}, kx::Int64, ky::Int64,
                         result::Vector{Float64})
    ccall((:bisplev_cfunc_vectorized, libfastspline), Cvoid,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, 
           Ptr{Float64}, Int64, Int64, Ptr{Float64}),
          x_arr, y_arr, tx, ty, c, kx, ky, result)
end
```

## Performance Benefits

- **Vectorized evaluation**: Process multiple points in a single call
- **No Python overhead**: Direct C-compatible execution  
- **Optimized algorithms**: Uses our step-by-step optimized B-spline evaluation
- **Memory efficient**: No temporary allocations within the function

## Performance Characteristics

- **Linear splines (k=1)**: ~1.4x faster than SciPy
- **Cubic splines (k=3)**: ~2x slower than SciPy (but machine precision accurate)
- **Vectorization**: Minimal overhead for batch evaluation
- **Memory**: O(1) stack usage, no heap allocations during evaluation

## Error Handling

The cfunc provides no error checking for performance reasons. Ensure:

1. Array lengths match: `len(x_arr) == len(y_arr) == len(result)`
2. Valid knot vectors: `len(tx) >= kx+1`, `len(ty) >= ky+1`
3. Valid coefficients: `len(c) == (len(tx)-kx-1) * (len(ty)-ky-1)`
4. Valid degrees: `kx, ky >= 1` (typically 1 or 3)
5. Evaluation points within knot span bounds (or extrapolation is acceptable)

## Building and Linking

The cfunc is automatically compiled when you import FastSpline. To access it from other languages:

1. Install FastSpline: `pip install fastspline` 
2. Find the compiled library location in your Python environment
3. Link against the Numba-generated shared library
4. Use the function signature above for your language's FFI

## Comparison with SciPy

| Feature | FastSpline cfunc | SciPy bisplev |
|---------|------------------|---------------|
| Language support | C/Fortran/Julia/Python | Python only |
| Vectorization | Native batch processing | Loop in Python |
| Performance (k=1) | 1.4x faster | Baseline |
| Performance (k=3) | 2x slower | Baseline |
| Accuracy | Machine precision | Machine precision |
| Memory overhead | Minimal | Python objects |