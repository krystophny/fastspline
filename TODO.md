# TODO: Port DIERCKX bispev to Pure Numba cfunc Implementation

## Objective
Create a pure Python/Numba implementation of the DIERCKX bispev routine and its dependencies (fpbisp, fpbspl) using only Numba cfuncs with nopython mode and fastmath. This will enable full JIT compilation without any external dependencies.

## Critical Requirements
- **ALL functions must be Numba cfuncs with nopython=True**
- **Use fastmath=True for performance**
- **Maintain bit-exact compatibility with Fortran implementation**
- **Validate each function individually against Fortran**
- **No Python objects or external calls in the implementation**

## Implementation Plan

### Phase 1: Analysis and Setup
- [ ] Study the call hierarchy: bispev → fpbisp → fpbspl
- [ ] Document all array indexing patterns (Fortran 1-based vs Python 0-based)
- [ ] Create test data generator for all intermediate validation steps
- [ ] Set up validation framework to compare Fortran vs Numba at each level

### Phase 2: Implement fpbspl (Lowest Level)
- [ ] Translate fpbspl.f to Python/Numba line by line
- [ ] Handle Fortran 1-based array indexing carefully
- [ ] Create cfunc signature: `fpbspl_cfunc(t, n, k, x, l, h, wrk)`
- [ ] Unit test against Fortran fpbspl with various inputs:
  - [ ] Test case 1: Simple knot vector, single evaluation point
  - [ ] Test case 2: Multiple evaluation points
  - [ ] Test case 3: Edge cases (x at knot positions)
  - [ ] Test case 4: Different spline degrees (k=1,2,3,5)
- [ ] Validate all intermediate array values match Fortran exactly
- [ ] Performance benchmark vs Fortran implementation

### Phase 3: Implement fpbisp (Middle Level)
- [ ] Translate fpbisp.f to Python/Numba line by line
- [ ] Create cfunc signature: `fpbisp_cfunc(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, wx, wy, lx, ly)`
- [ ] Handle the nested loops carefully (Fortran DO loops)
- [ ] Validate intermediate steps:
  - [ ] Test X-direction B-spline evaluation alone
  - [ ] Test Y-direction B-spline evaluation alone
  - [ ] Test combined tensor product evaluation
- [ ] Unit tests against Fortran fpbisp:
  - [ ] Test case 1: Constant spline surface
  - [ ] Test case 2: Linear surface
  - [ ] Test case 3: General polynomial surface
  - [ ] Test case 4: Large evaluation grids
- [ ] Check memory access patterns for cache efficiency

### Phase 4: Implement bispev (Top Level)
- [ ] Translate bispev.f to Python/Numba line by line
- [ ] Create cfunc signature: `bispev_cfunc(tx, nx, ty, ny, c, kx, ky, x, mx, y, my, z, wrk, lwrk, iwrk, kwrk, ier)`
- [ ] Implement input validation logic
- [ ] Handle error codes properly
- [ ] Full validation against Fortran bispev:
  - [ ] Test case 1: Valid inputs with various grid sizes
  - [ ] Test case 2: Invalid inputs (error handling)
  - [ ] Test case 3: Edge cases (single point evaluation)
  - [ ] Test case 4: Large scale tests

### Phase 5: Integration Testing
- [ ] Create comprehensive test suite comparing:
  - Fortran bispev vs Numba bispev
  - scipy.interpolate.bisplev vs Numba bispev
- [ ] Test with scipy's test suite data
- [ ] Property-based testing with random splines
- [ ] Stress tests with large data

### Phase 6: Optimization
- [ ] Profile the Numba implementation
- [ ] Optimize array access patterns
- [ ] Consider loop unrolling for small degrees
- [ ] Implement specialized versions for common cases (k=3)
- [ ] Add parallel evaluation option using prange

### Phase 7: Performance Validation
- [ ] Benchmark vs Fortran implementation
- [ ] Benchmark vs scipy.interpolate.bisplev
- [ ] Test compilation time vs runtime tradeoff
- [ ] Memory usage comparison
- [ ] Create performance scaling plots

### Phase 8: Documentation and Examples
- [ ] Document the implementation approach
- [ ] Create usage examples
- [ ] Add inline comments explaining Fortran→Python translations
- [ ] Document performance characteristics

## Technical Considerations

### Array Indexing Translation
- Fortran uses 1-based indexing, Python uses 0-based
- Fortran column-major vs NumPy row-major storage
- Need careful translation of all array accesses

### Example index translation pattern:
```python
# Fortran: h(i) where i goes from 1 to k+1
# Python:  h[i-1] where i goes from 1 to k+1
# Or better: h[i] where i goes from 0 to k
```

### Fortran Constructs to Handle
1. **DO loops with shared termination labels**
   ```fortran
   do 20 i=1,k1
   do 20 j=1,k1
   20 h(i,j) = 0.
   ```
   Translate to nested loops in Python

2. **GOTO statements**
   - Replace with proper control flow
   - Use early returns or continue statements

3. **Implicit type conversions**
   - Be explicit about all type conversions

### Validation Strategy
1. **Instrumentation approach**:
   - Add debug prints in both Fortran and Numba
   - Compare intermediate values at each step
   - Use small test cases for detailed comparison

2. **Binary compatibility**:
   - Results must match to machine precision
   - Test with various compiler optimization levels
   - Consider floating-point rounding modes

### Performance Targets
- Numba implementation should be within 2x of Fortran performance
- Compilation time should be < 1 second for typical use
- Memory usage should be comparable to Fortran

## File Structure
```
numba_implementation/
├── fpbspl_numba.py      # Lowest level B-spline basis
├── fpbisp_numba.py      # Tensor product evaluation  
├── bispev_numba.py      # Top level with validation
├── test_fpbspl.py       # Unit tests for fpbspl
├── test_fpbisp.py       # Unit tests for fpbisp
├── test_bispev.py       # Integration tests
├── validation_utils.py   # Tools for comparing with Fortran
└── benchmarks.py        # Performance comparisons
```

## Success Criteria
1. All unit tests pass with numerical tolerance < 1e-14
2. Performance within 2x of Fortran implementation  
3. Successfully compiles with nopython=True, fastmath=True
4. No Python object allocations in hot path
5. Passes scipy's interpolation test suite