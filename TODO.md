# FastSpline TODO

## Current Plan: 3D and Periodic Sergei Splines Validation

### Phase 1: 3D Spline Validation ✓
- [ ] Create Fortran validation program for 3D splines
  - [ ] Test function: f(x,y,z) = sin(πx) * cos(πy) * exp(-z/2)
  - [ ] Grid sizes: 8×8×8, 10×10×10, 20×20×20
  - [ ] Orders: 3, 4, 5 (cubic, quartic, quintic)
  - [ ] Output reference values at specific test points
- [ ] Create Python 3D validation script
  - [ ] Use correct meshgrid with indexing='ij' for all 3 dimensions
  - [ ] Compare construction coefficients with Fortran
  - [ ] Compare evaluation results at test points
  - [ ] Generate 3D visualization slices
- [ ] Fix any discrepancies found
  - [ ] Check tensor product ordering for 3D
  - [ ] Verify coefficient storage layout
  - [ ] Ensure proper evaluation algorithm

### Phase 2: Periodic Spline Validation ✓
- [ ] Create Fortran validation for periodic splines
  - [ ] 1D periodic: f(x) = sin(2πx) on [0,1]
  - [ ] 2D periodic: f(x,y) = sin(2πx) * cos(2πy) on [0,1]×[0,1]
  - [ ] 3D periodic: f(x,y,z) = sin(2πx) * cos(2πy) * sin(2πz)
  - [ ] Test continuity at boundaries
- [ ] Create Python periodic validation
  - [ ] Test all dimensions (1D, 2D, 3D)
  - [ ] Verify periodic boundary conditions
  - [ ] Check derivatives at boundaries
  - [ ] Compare with Fortran reference
- [ ] Fix any periodic-specific issues
  - [ ] Ensure proper wrap-around
  - [ ] Verify coefficient calculation for periodic case

### Phase 3: Mixed Boundary Conditions ✓
- [ ] Test mixed periodic/non-periodic in 2D/3D
  - [ ] 2D: periodic in x, non-periodic in y
  - [ ] 3D: various combinations
- [ ] Create test cases for each combination
- [ ] Validate against Fortran reference

### Phase 4: Performance Validation ✓
- [ ] Benchmark 3D spline performance
  - [ ] Construction time for various grid sizes
  - [ ] Evaluation throughput (points/second)
  - [ ] Memory usage analysis
- [ ] Compare with theoretical expectations
- [ ] Document performance characteristics

### Phase 5: Documentation and Release ✓
- [ ] Update README with 3D examples
- [ ] Document periodic boundary conditions
- [ ] Create comprehensive API documentation
- [ ] Add integration tests for all features
- [ ] Update VALIDATION_SUMMARY.md with all results

## Completed ✅

### 1D Splines
- ✅ All orders (3, 4, 5) validated against Fortran
- ✅ Derivatives (1st and 2nd) working correctly
- ✅ Performance optimized with cfunc flags

### 2D Splines  
- ✅ Fixed meshgrid indexing issue (use indexing='ij')
- ✅ Validated against Fortran reference
- ✅ Clean validation plots created
- ✅ All redundant scripts removed

### Performance Optimizations
- ✅ All cfuncs use nopython=True, nogil=True, cache=True, fastmath=True
- ✅ Achieved ~1.7ns per point for 1D evaluation
- ✅ Achieved ~10.3ns per point for 2D evaluation

## Notes

- Always use `np.meshgrid(..., indexing='ij')` for consistency with Fortran
- Fortran uses column-major order, Python uses row-major
- Coefficient storage: (order+1) × n1 × n2 × ... for each dimension
- Power basis representation, not B-splines
- Special quintic constants: ρ+ = 23.247, ρ- = 2.753