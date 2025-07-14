# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastSpline is a high-performance spline interpolation project focused on optimizing DIERCKX Fortran library functions for Python. The project provides a single, clean Numba cfunc implementation of bivariate spline interpolation with derivative support that exactly matches scipy's behavior.

## Current State

**CRITICAL REQUIREMENT: ALL TESTS MUST PASS**

The repository contains a complete fastspline implementation with:
- Single inline Numba cfunc implementation for bivariate spline derivatives (`parder.py`)
- Comprehensive test suite with 14 passing tests, 1 skipped (15 total)
- Full derivative support matching scipy.interpolate.dfitpack.parder exactly
- Performance comparison and benchmarking scripts

**IMPLEMENTATION RULE: When implementing new versions, old versions MUST be removed completely**

## Key Commands

### Running Tests (MUST ALWAYS PASS)
```bash
# Run all tests with pytest - ALL MUST PASS
python -m pytest tests/ -v

# Run specific derivative accuracy tests
python -m pytest tests/test_derivative_accuracy.py -v

# Run CI tests
python -m pytest tests/test_ci.py -v

# Run fastspline tests
python -m pytest tests/test_fastspline.py -v
```

### Running Performance Comparisons
```bash
# Compare scipy interface overhead
python compare_scipy_direct_wrapper.py

# Test direct f2py wrapper comparison (may not work with newer scipy)
python compare_scipy_f2py_direct.py

# Alternative performance comparison scripts
python compare_scipy_f2py_performance_final.py
```

### Testing Individual Components
```bash
# Test parder implementation directly
python fastspline/numba_implementation/parder.py
```

## Architecture and Implementation Notes

### Implementation Requirements

**CRITICAL: ALL implementations must:**
- Use only cfunc decorators (no njit functions)
- Inline all operations within a single cfunc
- Validate against scipy exactly (exact floating-point matches)
- Pass ALL tests in the test suite
- Remove old implementations when creating new ones

### Performance Optimization Approaches

1. **Single cfunc Implementation**: All operations inlined into one cfunc without external function calls
2. **Exact scipy Matching**: Bit-exact compatibility with scipy.interpolate.dfitpack.parder
3. **Pre-allocated Arrays**: Reducing memory allocation overhead in hot paths
4. **Minimal Overhead**: Direct cfunc calls without Python wrapper overhead

### Key Technical Details

- The project targets exact floating-point accuracy matching scipy
- All derivative computations must validate against scipy.interpolate.dfitpack.parder
- Implementation is in `fastspline/numba_implementation/parder.py` as a single cfunc
- Tests verify exact matches for all derivative orders: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1)

### Current File Structure

The implementation includes:
- `fastspline/numba_implementation/parder.py`: Single cfunc implementation with derivative support
- `tests/`: Comprehensive test suite that MUST all pass
- Performance comparison scripts in root directory

## Development Considerations

### Testing Requirements
- ALL tests must pass: `python -m pytest tests/ -v`
- No skipped tests except for missing dependencies
- Derivative accuracy must be exact (< 1e-14 difference from scipy)

### Implementation Standards
- Use only cfunc decorators, no njit functions
- Inline all operations within a single function
- Validate against scipy/DIERCKX exactly
- Remove old implementations when creating new ones
- Test with pytest, not custom test runners

### Performance Optimization
- Use automatic smoothing (`s=len(data)`) rather than exact interpolation (`s=0`) for large datasets
- scipy's bisplrep can produce warnings about insufficient knots - these indicate the need for smoothing
- Performance gains from optimization are most noticeable for repeated evaluations on small grids

**REMEMBER: TESTS MUST PASS - NO EXCEPTIONS**