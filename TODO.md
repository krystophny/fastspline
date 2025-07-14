# TODO: Implement Derivatives for Bivariate Spline Interpolation

## Objective
Implement complete derivative support for bivariate spline interpolation (`bisplev`/`parder`) in all variants (Fortran, Numba cfunc, and Python wrappers). The DIERCKX library's `parder` function computes partial derivatives of bivariate splines, which is currently missing from the FastSpline implementation.

## Critical Requirements
- **Exact floating-point compatibility** with scipy's `bisplev(dx=..., dy=...)` and `dfitpack.parder`
- **Support for all derivative orders** (dx=0,1,2,... and dy=0,1,2,...)
- **Bit-exact validation** against existing DIERCKX Fortran implementation
- **Pure Numba cfunc implementation** with nopython=True and fastmath=True
- **Comprehensive testing** for all derivative combinations
- **Performance optimization** to match or exceed scipy performance

## Implementation Plan

### Phase 1: Research and Analysis
- [ ] Study DIERCKX `parder` Fortran source code from scipy/dfitpack
- [ ] Analyze the relationship between `bisplev` and `parder` algorithms
- [ ] Document the mathematical theory behind spline derivative computation
- [ ] Understand B-spline basis function derivatives (fpbspl derivatives)
- [ ] Map the call hierarchy: parder → fpbisp_derivative → fpbspl_derivative
- [ ] Create test data generator for comprehensive derivative validation

### Phase 2: Fortran Implementation
- [ ] Extract `parder.f` from scipy's dfitpack sources
- [ ] Add `parder.f` to `src/fortran/` directory
- [ ] Extract required supporting functions (likely fpbisp derivative variant)
- [ ] Add C wrapper in `src/c/bispev_wrapper.c` for parder
- [ ] Update `include/bispev_wrapper.h` with parder function signatures
- [ ] Test Fortran parder implementation against scipy.dfitpack.parder

### Phase 3: Numba fpbspl Derivative Implementation
- [ ] Implement `fpbspl_derivative` function in `fpbspl_numba.py`
- [ ] Handle derivative computation for B-spline basis functions
- [ ] Support arbitrary derivative orders (not just 0, 1, 2)
- [ ] Optimize for common derivative orders (dx=1, dy=1)
- [ ] Unit tests against Fortran fpbspl derivatives:
  - [ ] Test dx=0, dy=0 (should match regular fpbspl)
  - [ ] Test dx=1, dy=0 (first derivative in x)
  - [ ] Test dx=0, dy=1 (first derivative in y)
  - [ ] Test dx=1, dy=1 (mixed second derivative)
  - [ ] Test higher order derivatives (dx=2, dy=2, etc.)
  - [ ] Test edge cases (derivatives at knot positions)

### Phase 4: Numba fpbisp Derivative Implementation
- [ ] Implement `fpbisp_derivative` function in `fpbisp_numba.py`
- [ ] Handle tensor product evaluation with derivatives
- [ ] Support mixed derivatives (dx>0 and dy>0 simultaneously)
- [ ] Optimize coefficient access patterns for derivative computation
- [ ] Unit tests against Fortran fpbisp derivatives:
  - [ ] Test constant surfaces (derivatives should be 0)
  - [ ] Test linear surfaces (second derivatives should be 0)
  - [ ] Test polynomial surfaces with known analytical derivatives
  - [ ] Test large evaluation grids with derivatives

### Phase 5: Numba parder cfunc Implementation
- [ ] Implement `parder_cfunc` in new file `parder_numba.py`
- [ ] Create cfunc signature matching DIERCKX parder interface
- [ ] Handle input validation for derivative orders
- [ ] Integrate with fpbisp_derivative for tensor product evaluation
- [ ] Support both single point and grid evaluation modes
- [ ] Error handling for invalid derivative orders or inputs
- [ ] Full validation against Fortran parder:
  - [ ] Test various spline surfaces and derivative combinations
  - [ ] Test edge cases (evaluation outside domain)
  - [ ] Test performance against scipy implementation

### Phase 6: Python Wrapper Integration
- [ ] Add derivative support to `bispev_numba.py`
- [ ] Extend signature to include `dx` and `dy` parameters
- [ ] Route calls appropriately between evaluation and derivative functions
- [ ] Maintain backward compatibility with existing bisplev interface
- [ ] Add convenience functions for common derivative operations
- [ ] Integration tests with scipy.interpolate.bisplev

### Phase 7: ctypes Wrapper Extension
- [ ] Update `bispev_ctypes.py` to support derivative evaluation
- [ ] Add ctypes interface for parder function
- [ ] Handle memory management for derivative computations
- [ ] Test ctypes derivative implementation against Fortran

### Phase 8: Performance Optimization
- [ ] Profile derivative computation performance
- [ ] Optimize memory access patterns for derivative calculations
- [ ] Implement specialized versions for common derivative orders
- [ ] Add parallel evaluation support using prange
- [ ] Compare performance against scipy.interpolate.bisplev with derivatives
- [ ] Optimize for repeated derivative evaluations

### Phase 9: Comprehensive Testing
- [ ] Create comprehensive test suite in `test_parder.py`
- [ ] Property-based testing for derivative computation
- [ ] Cross-validation between all implementation variants
- [ ] Stress tests with large grids and high derivative orders
- [ ] Numerical accuracy tests (ensure derivatives are mathematically correct)
- [ ] Test boundary conditions and edge cases

### Phase 10: Documentation and Examples
- [ ] Document derivative API in all implementation variants
- [ ] Create usage examples for derivative computation
- [ ] Add performance comparison plots (derivatives vs evaluation)
- [ ] Document mathematical background of spline derivatives
- [ ] Add inline comments explaining derivative algorithms

## Technical Considerations

### Derivative Algorithm Overview
The DIERCKX `parder` function computes partial derivatives of bivariate splines by:
1. Computing derivatives of B-spline basis functions (fpbspl_derivative)
2. Evaluating tensor products with derivative basis functions
3. Applying chain rule for mixed derivatives

### B-spline Derivative Computation
- B-spline basis function derivatives follow recursive formulas
- First derivative: `B'_i,k(x) = k * (B_{i,k-1}(x) / (t_{i+k} - t_i) - B_{i+1,k-1}(x) / (t_{i+k+1} - t_{i+1}))`
- Higher derivatives computed recursively
- Special handling required for repeated knots

### Memory Layout Considerations
- Derivative computation may require additional workspace
- Coefficient access patterns differ for derivative evaluation
- Cache efficiency important for large grid evaluations

### Validation Strategy
1. **Mathematical validation**: Verify derivatives using finite differences
2. **Cross-implementation validation**: Compare Fortran, Numba, and scipy
3. **Exact equality testing**: Ensure bit-exact compatibility
4. **Performance validation**: Ensure derivatives don't significantly slow evaluation

## File Structure
```
fastspline/
├── src/
│   ├── fortran/
│   │   ├── bispev.f
│   │   ├── fpbisp.f
│   │   ├── fpbspl.f
│   │   └── parder.f           # New: DIERCKX parder implementation
│   └── c/
│       └── bispev_wrapper.c   # Extended: Add parder C wrapper
├── fastspline/
│   ├── numba_implementation/
│   │   ├── fpbspl_numba.py    # Extended: Add derivative support
│   │   ├── fpbisp_numba.py    # Extended: Add derivative support
│   │   ├── bispev_numba.py    # Extended: Add dx/dy parameters
│   │   ├── parder_numba.py    # New: Pure Numba parder implementation
│   │   ├── test_fpbspl.py     # Extended: Add derivative tests
│   │   ├── test_fpbisp.py     # Extended: Add derivative tests
│   │   ├── test_bispev.py     # Extended: Add derivative tests
│   │   └── test_parder.py     # New: Comprehensive derivative tests
│   └── ctypes_wrapper/
│       └── bispev_ctypes.py   # Extended: Add derivative support
├── tests/
│   └── test_derivative_validation.py  # New: Cross-validation tests
└── benchmarks/
    └── benchmark_derivatives.py       # New: Derivative performance tests
```

## Success Criteria
1. **Exact compatibility**: All derivative computations match scipy to machine precision
2. **Comprehensive coverage**: Support all derivative orders up to spline degree
3. **Performance**: Derivative computation within 2x of scipy performance
4. **Code quality**: All implementations pass nopython=True compilation
5. **Testing**: >95% test coverage for derivative functionality
6. **Documentation**: Complete API documentation with examples

## Priority Order
1. **Fortran parder implementation** (reference implementation)
2. **Numba fpbspl derivative support** (foundation for higher-level functions)
3. **Numba fpbisp derivative support** (tensor product evaluation)
4. **Pure Numba parder cfunc** (high-performance implementation)
5. **Python wrapper integration** (user-friendly interface)
6. **Performance optimization** (production-ready performance)
7. **Comprehensive testing** (reliability and correctness)

This implementation will provide complete derivative support for bivariate spline interpolation, enabling applications requiring gradient computation, optimization, and numerical analysis of spline surfaces.