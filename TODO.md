# FastSpline TODO

## Current Plan: Achieve Perfect Double Precision Fortran Matching

### CRITICAL: Double Precision Validation Status

**Requirement**: All spline implementations must achieve double precision (1e-15 level) matching with Fortran reference implementations.

**Current Status**:
- ✅ **Cubic Regular**: 6.70e-17 error (PERFECT double precision)
- ⚠️ **Cubic Periodic**: 4.87e-12 error (needs improvement to 1e-15)
- ⚠️ **Quintic Regular**: 1.17e-04 error (needs improvement to 1e-15)
- ❌ **Quintic Periodic**: Not implemented/tested

### Phase 1: Fix Cubic Periodic Double Precision
- [ ] Debug cubic periodic construction to achieve 1e-15 precision
- [ ] Test both construction coefficients and evaluation results
- [ ] Add pytest test requiring error < 1e-14
- [ ] Validate against Fortran periodic cubic reference

### Phase 2: Fix Quintic Regular Double Precision  
- [ ] Debug quintic regular construction to achieve 1e-15 precision
- [ ] Test both construction coefficients and evaluation results
- [ ] Add pytest test requiring error < 1e-14
- [ ] Validate against Fortran quintic reference

### Phase 3: Implement Quintic Periodic
- [ ] Implement quintic periodic spline construction
- [ ] Create Fortran reference for quintic periodic
- [ ] Achieve 1e-15 precision matching
- [ ] Add comprehensive pytest tests

### Phase 4: Comprehensive Testing Criteria
For each spline type (cubic/quintic × regular/periodic):
- [ ] **Construction precision**: Coefficients match Fortran at 1e-15 level
- [ ] **Evaluation precision**: Function values match Fortran at 1e-15 level
- [ ] **Boundary conditions**: Perfect continuity for periodic cases
- [ ] **Multiple test points**: Precision maintained across domain
- [ ] **pytest integration**: All tests automated with strict thresholds

### Success Criteria
**BEFORE completion, ALL must be true**:
- [ ] Cubic regular: construction + evaluation < 1e-14 error
- [ ] Cubic periodic: construction + evaluation < 1e-14 error  
- [ ] Quintic regular: construction + evaluation < 1e-14 error
- [ ] Quintic periodic: construction + evaluation < 1e-14 error
- [ ] All 4 cases have pytest tests with strict error thresholds
- [ ] All tests pass consistently

## Completed ✅

### Quartic Splines Management
- ✅ Quartic (4th order) splines disabled due to mathematical property issues
- ✅ Clear error message directs users to cubic or quintic alternatives

### SciPy Competition Validation
- ✅ Cubic splines perfectly competitive with SciPy (1.00x performance ratio)
- ✅ Identical results on smooth functions and noisy data
- ✅ 12 comprehensive pytest tests all passing

### Performance Optimizations
- ✅ All cfuncs use nopython=True, nogil=True, cache=True, fastmath=True
- ✅ Verified cfunc usage EVERYWHERE in sergei splines
- ✅ All function addresses properly exposed via get_cfunc_addresses()

### 2D Splines  
- ✅ Fixed meshgrid indexing issue (use indexing='ij')
- ✅ Validated against Fortran reference
- ✅ Clean validation plots created
- ✅ All redundant scripts removed

## Notes

- **Double precision**: Requires error < 1e-14 to account for numerical tolerance
- **Test both**: Construction (coefficients) AND evaluation (function values)
- **Fortran reference**: Must have exact comparison with validated Fortran code
- Always use `np.meshgrid(..., indexing='ij')` for consistency with Fortran
- Coefficient storage: (order+1) × n1 × n2 × ... for each dimension
- Power basis representation, not B-splines