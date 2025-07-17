# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastSpline is a high-performance spline interpolation library providing:
- Exact scipy compatibility for DIERCKX bivariate splines
- Pure Numba cfunc implementations for zero-overhead function calls
- Sergei's equidistant splines (1D/2D/3D, orders 3-5)
- Memory-efficient evaluation with pre-allocated arrays

## Build Commands

```bash
# Build Fortran/C extensions
make

# Clean build artifacts
make clean

# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_fastspline.py -v

# Run with coverage
python -m pytest tests/ --cov=fastspline
```

## Architecture

### Core Components

1. **DIERCKX Implementation** (`src/fastspline/bispev_numba.py`, `parder.py`)
   - Exact port of FITPACK routines
   - Implements Cox-de Boor recursion
   - Tensor product evaluation for efficiency

2. **Sergei Splines** (`src/fastspline/sergei_splines.py`)
   - Equidistant knot spacing optimization
   - Support for periodic boundaries
   - Orders 3 (cubic), 4 (quartic), 5 (quintic)
   - Efficient tensor product for multidimensional cases

3. **C/Fortran Integration**
   - Original Fortran sources in `src/fortran/`
   - C wrapper in `src/c/bispev_wrapper.c`
   - Shared library built to `lib/libbispev.so`

### Key Design Patterns

- **cfunc Architecture**: All performance-critical functions compiled as Numba cfuncs
- **Memory Pre-allocation**: Evaluation functions work with pre-allocated arrays
- **Tensor Product**: 2D/3D splines use efficient tensor product decomposition
- **Zero-copy Operations**: Direct memory access via ctypes pointers

## Current Task Context

The repository is working on validating memory alignment between Fortran and Python implementations:
- Validation suite in `validation/sergei_splines/`
- Blocked by NumPy 2.3 / Numba compatibility issue
- Need to either downgrade NumPy or update Numba

## Testing Strategy

1. **Unit Tests**: Core functionality tests in `tests/`
2. **Integration Tests**: scipy compatibility verification
3. **Validation Suite**: Fortran vs Python comparison
4. **Visual Tests**: Examples with plots in `examples/`

## Important Implementation Details

- Array indexing: Fortran uses 1-based, Python uses 0-based
- Memory layout: Fortran is column-major, Python/NumPy is row-major by default
- Spline coefficients stored in flattened arrays with specific ordering
- Workspace arrays required for 2D/3D construction to avoid allocations

## Development Workflow

1. Always check existing patterns before implementing new features
2. Maintain exact scipy compatibility where applicable
3. Use pre-allocated arrays for performance-critical paths
4. Test both accuracy and performance for new implementations
5. Document cfunc signatures clearly as they're called from ctypes
6. **IMPORTANT**: Always modify existing scripts in-place. Never create multiple versions or collections of scripts. Replace and update existing files directly.