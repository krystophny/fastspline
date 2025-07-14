# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastSpline is a high-performance spline interpolation project focused on optimizing DIERCKX Fortran library functions for Python. The project explores various optimization strategies for bivariate spline interpolation (`bisplrep`/`bisplev`) including direct f2py wrapper calls, Numba cfunc implementations, and minimal overhead interfaces.

## Current State

The repository contains a complete fastspline implementation with:
- Working Numba cfunc implementations for bivariate spline interpolation
- Comprehensive test suite with 15 passing tests
- Derivative support through scipy.interpolate.dfitpack.parder
- Performance comparison and benchmarking scripts

## Key Commands

### Running Performance Comparisons
```bash
# Compare scipy interface overhead
python compare_scipy_direct_wrapper.py

# Test direct f2py wrapper comparison (may not work with newer scipy)
python compare_scipy_f2py_direct.py

# Alternative performance comparison scripts
python compare_scipy_f2py_performance_final.py
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_derivative_accuracy.py -v
python -m pytest tests/test_fastspline.py -v
python -m pytest tests/test_ci.py -v
```

## Architecture and Implementation Notes

### Performance Optimization Approaches

1. **Direct f2py Wrappers**: Bypassing scipy's Python interface to call Fortran routines directly
2. **Numba cfunc**: JIT-compiled implementations matching DIERCKX algorithms exactly
3. **Pre-allocated Arrays**: Reducing memory allocation overhead in hot paths
4. **Minimal Wrappers**: Stripping validation and conversion overhead

### Key Technical Details

- The project targets exact floating-point accuracy matching DIERCKX/scipy
- Performance comparisons focus on scattered data interpolation use cases
- Smoothing parameter `s` significantly impacts performance (s=0 for exact interpolation is slowest)
- The scipy interface overhead is typically minimal (<1%) for most use cases

### Current File Structure

The implementation includes:
- `fastspline/numba_implementation/`: Numba cfunc implementations including working parder_correct.py
- `src/fortran/`: Fortran source files with compiled object files
- `src/c/`: C wrapper implementations
- `tests/`: Comprehensive test suite with derivative accuracy validation
- Performance comparison scripts in root directory

## Development Considerations

When working with spline interpolation performance:
- Use automatic smoothing (`s=len(data)`) rather than exact interpolation (`s=0`) for large datasets
- scipy's bisplrep can produce warnings about insufficient knots - these indicate the need for smoothing
- Direct f2py access patterns have changed across scipy versions
- Performance gains from optimization are most noticeable for repeated evaluations on small grids