# FastSpline TODO

## Current Status

### Completed
- [x] Sergei splines cfunc implementation (1D, 2D, 3D)
- [x] Support for cubic, quartic, and quintic splines
- [x] Periodic and non-periodic boundary conditions
- [x] Derivatives support (1D: first and second, 2D/3D: first)
- [x] Created validation framework for memory alignment issues

### In Progress
- [ ] Fix memory alignment issue in cfunc construction
  - ✅ Created comprehensive validation suite in `validation/sergei_splines/`
  - ✅ Fortran validation program works correctly
  - ✅ Python validation script updated to use correct cfunc API
  - ✅ Identified critical memory alignment issue: only first row of coefficients populated
  - ❌ **CRITICAL ISSUE**: Maximum coefficient difference of 81.2 indicates serious memory layout problem
  - Next: Investigate and fix coefficient array memory layout in `src/fastspline/sergei_splines.py`

## Goal

Validate that the Python cfunc implementation of Sergei splines produces identical results to the Fortran implementation, particularly focusing on:
- Memory alignment issues
- Array indexing consistency
- Numerical accuracy
- Performance characteristics

## Next Steps

1. Resolve NumPy/Numba compatibility issue
2. Run full validation suite to compare:
   - Spline coefficients
   - Evaluation results
   - Derivative calculations
3. Fix any memory alignment issues discovered
4. Optimize performance if needed

## Validation Framework

The validation suite (`validation/sergei_splines/`) includes:
- Fortran test program using original `interpolate.f90` and `spl_three_to_five.f90`
- Python test program using cfunc implementation
- Automated comparison tools
- Makefile for building and running tests
- Comprehensive test cases for 1D and 2D splines