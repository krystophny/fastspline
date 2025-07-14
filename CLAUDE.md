# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastSpline is a high-performance bivariate spline interpolation library implementing optimized DIERCKX algorithms. The project provides complete Python/Numba cfunc implementations of bivariate spline evaluation and derivatives that exactly match scipy's behavior while delivering superior performance.

## Current State

**CRITICAL REQUIREMENT: ALL TESTS MUST PASS**

The repository contains a complete FastSpline implementation featuring:
- **Complete spline evaluation**: Function values and all derivative orders
- **Pure cfunc implementations**: `bispev_numba.py` for evaluation, `parder.py` for derivatives  
- **Single inlined functions**: All operations inlined within bispev_cfunc and parder_cfunc
- **No numpy arrays in cfunc**: Uses only workspace arrays for proper memory management
- **Safe wrapper functions**: `call_parder_safe` for memory-safe derivative evaluation
- **Comprehensive test suite**: 15/15 tests pass with exact floating-point validation
- **Bit-exact accuracy**: Matches scipy to machine precision (1e-14 relative tolerance)
- **Performance optimization**: Minimal overhead over scipy with native code compilation

**IMPLEMENTATION STATUS**: 
- ✅ Function evaluation (bispev) - Complete cfunc implementation
- ✅ Derivative evaluation (parder) - Complete cfunc implementation with proper workspace usage
- ✅ All tests passing (15/15)
- ✅ No numpy usage in cfuncs
- ✅ All operations inlined

**IMPLEMENTATION RULE: When implementing new versions, old versions MUST be removed completely**

## Key Commands

### Running Tests (MUST ALWAYS PASS)
```bash
# Run all tests with pytest - ALL 15 MUST PASS
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_derivative_accuracy.py -v  # Derivative validation
python -m pytest tests/test_ci.py -v                   # CI integration tests  
python -m pytest tests/test_fastspline.py -v           # Core functionality

# Expected output: 15 passed, 0 failed
```

### Testing Individual Components
```bash
# Test parder implementation with comprehensive validation
python fastspline/numba_implementation/parder.py

# Test bispev implementation
python fastspline/numba_implementation/test_bispev.py
```

### Using cfunc Implementations
```python
# For function evaluation
from fastspline.numba_implementation.bispev_numba import bispev_cfunc_address
import ctypes

# For derivative evaluation with safe wrapper
from fastspline.numba_implementation.parder import call_parder_safe

# Example: Compute derivatives
z_deriv, ier = call_parder_safe(tx, ty, c, kx, ky, nux, nuy, xi, yi)
```

### Running Performance Comparisons
```bash
# Performance benchmarking
python fastspline/numba_implementation/benchmarks.py

# Legacy performance scripts (in benchmarks/scripts/legacy/)
python benchmarks/scripts/legacy/compare_scipy_f2py_performance_final.py
```

## Architecture and Implementation

### Implementation Requirements

**CRITICAL: ALL implementations must:**
- Use only cfunc decorators (NO njit functions allowed)
- Inline ALL operations within single cfunc implementations
- Validate against scipy exactly (bit-exact floating-point matches)
- Pass ALL 15 tests in the test suite without exception
- Remove old/obsolete implementations when creating new ones
- Follow DIERCKX algorithm structure exactly

### Core Implementations

**1. Bivariate Spline Evaluation (`bispev_numba.py`)**
- Single cfunc implementation with inlined fpbisp/fpbspl algorithms
- Cox-de Boor B-spline basis computation
- Optimized tensor product evaluation
- Supports all spline degrees (kx, ky = 1 to 5)

**2. Derivative Evaluation (`parder.py`)**  
- Complete DIERCKX parder algorithm in single cfunc
- Inlined fpbspl with derivative computation via recursive differentiation
- Handles all derivative orders: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1)
- Exact coefficient indexing matching original Fortran structure

### Performance Characteristics

- **Function evaluation**: < 1% overhead vs scipy.interpolate.bisplev
- **Derivative computation**: Bit-exact match with scipy.interpolate.dfitpack.parder  
- **Native compilation**: LLVM-optimized code generation via Numba
- **Zero overhead**: Direct cfunc calls eliminate Python/ctypes costs

### File Structure

**Core Implementation:**
- `fastspline/numba_implementation/parder.py` - Derivative evaluation (single cfunc)
- `fastspline/numba_implementation/bispev_numba.py` - Function evaluation (single cfunc)

**Supporting Modules:**
- `fastspline/numba_implementation/fpbisp_numba.py` - Standalone fpbisp implementation
- `fastspline/numba_implementation/fpbspl_numba.py` - Standalone fpbspl implementation  
- `fastspline/numba_implementation/fpbspl_derivative.py` - B-spline derivative utilities

**Testing:**
- `tests/` - Main test suite (15 tests, all must pass)
- `fastspline/numba_implementation/test_*.py` - Individual component tests

**Performance:**
- `fastspline/numba_implementation/benchmarks.py` - Performance testing
- `benchmarks/` - Comprehensive performance analysis

## Development Standards

### Testing Requirements
- **ALL 15 tests must pass**: `python -m pytest tests/ -v` 
- **Derivative accuracy**: Exact floating-point match with scipy (< 1e-14 difference)
- **Function evaluation**: Bit-exact compatibility with scipy.interpolate.bisplev
- **No skipped tests**: Except for documented missing dependencies

### Implementation Standards
- **Only cfunc decorators**: No njit functions allowed
- **Complete inlining**: All operations within single cfunc implementations
- **Exact validation**: Bit-exact match with scipy/DIERCKX algorithms
- **Clean implementations**: Remove old versions when creating new ones
- **pytest only**: Use pytest for all testing, no custom test runners

### Code Quality
- **Follow DIERCKX structure**: Maintain original algorithm organization
- **Proper indexing**: Careful Fortran 1-based to Python 0-based translation
- **Memory management**: Static workspace allocation, no dynamic allocation
- **Error handling**: Complete input validation and boundary checking

## Performance Optimization Guidelines

### Spline Fitting
- Use smoothing parameter `s > 0` for large datasets (recommended: `s=len(data)`)
- Exact interpolation `s=0` is slowest and may cause numerical issues
- scipy's bisplrep warnings about insufficient knots indicate need for smoothing

### Evaluation Performance  
- Performance gains most noticeable for repeated evaluations on small grids
- Direct cfunc calls provide maximum performance (eliminate Python overhead)
- Numba compilation time is amortized over multiple calls

### Memory Optimization
- Pre-allocate workspace arrays to reduce allocation overhead
- Use contiguous arrays for optimal memory access patterns
- Minimize temporary array creation in hot paths

## Critical Development Rules

1. **TESTS MUST PASS**: All 15 tests must pass before any commit
2. **NO FAKE IMPLEMENTATIONS**: All algorithms must be real, working implementations  
3. **EXACT VALIDATION**: Results must match scipy to machine precision
4. **CFUNC ONLY**: No njit functions, only cfunc decorators allowed
5. **INLINE EVERYTHING**: All operations within single cfunc implementations
6. **CLEAN UP**: Remove old implementations when creating new ones
7. **USE PYTEST**: All testing via pytest, no custom test frameworks

**REMEMBER: IMPLEMENTATION CORRECTNESS AND TEST PASSING ARE NON-NEGOTIABLE**