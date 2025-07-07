# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastSpline is a high-performance spline interpolation project focused on optimizing DIERCKX Fortran library functions for Python. The project explores various optimization strategies for bivariate spline interpolation (`bisplrep`/`bisplev`) including direct f2py wrapper calls, Numba cfunc implementations, and minimal overhead interfaces.

## Current State

The repository has undergone significant cleanup, with most implementation files removed (visible in git history). Currently contains only performance comparison and benchmarking scripts that demonstrate scipy interface overhead analysis.

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

### Historical Commands (from deleted files)
```bash
# Build DIERCKX f2py wrapper (when build script existed)
./build_dierckx_f2py.sh

# Run validation tests
python tests/test_dierckx_cfunc.py
python tests/test_automatic_validation.py

# Run benchmarks
python examples/benchmark_performance.py
python examples/benchmark_bisplrep_bisplev.py
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

### Historical File Structure

The complete implementation included:
- `dierckx_cfunc.py`: Core Numba implementations of DIERCKX routines
- `thirdparty/dierckx/`: Original Fortran source files
- `tests/`: Comprehensive validation against scipy
- `examples/`: Usage and benchmark examples

## Development Considerations

When working with spline interpolation performance:
- Use automatic smoothing (`s=len(data)`) rather than exact interpolation (`s=0`) for large datasets
- scipy's bisplrep can produce warnings about insufficient knots - these indicate the need for smoothing
- Direct f2py access patterns have changed across scipy versions
- Performance gains from optimization are most noticeable for repeated evaluations on small grids