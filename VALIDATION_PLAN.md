# BISPL Validation Plan

## Objective
Achieve floating-point accuracy agreement between fastspline's `bisplev_cfunc` and scipy's `interpolate.bisplev` for unstructured 2D B-spline evaluation.

## Current Status
- **Implementation**: Placeholder in `src/fastspline/spline2d.py:363-391`
- **Gap**: Missing proper DIERCKX B-spline evaluation algorithm
- **Required**: Full implementation of tensor product B-spline evaluation

## Validation Criteria

### 1. Absolute Accuracy
- **Polynomial Reproduction**: < 1e-14 (machine precision)
- **Smooth Functions**: < 1e-12 
- **General Cases**: < 1e-10

### 2. Relative Accuracy
- **Target**: < 1e-12 relative error where |f(x,y)| > 1e-10
- **Edge Cases**: Handle near-zero values gracefully

### 3. Numerical Stability
- **Scale Invariance**: Maintain accuracy for data scales 1e-6 to 1e6
- **Condition Numbers**: Stable for ill-conditioned knot sequences

## Test Categories

### 1. Exact Tests (Machine Precision)
- Polynomial reproduction up to degree (kx, ky)
- Linear interpolation (kx=1, ky=1)
- Constant functions

### 2. Approximation Tests
- Smooth functions (Gaussian, trigonometric)
- Various smoothing parameters (s=0, 0.01, 0.1, 1.0)
- Different spline degrees (k=1,2,3,5)

### 3. Boundary Tests
- Evaluation at knot positions
- Extrapolation beyond data range
- Minimal grids (2x2, 3x3)

### 4. Derivative Tests
- First derivatives (dx=1, dy=1)
- Second derivatives (dx=2, dy=2)
- Mixed derivatives (dx=1, dy=1)

### 5. Special Cases
- Periodic boundary conditions
- Non-uniform knot spacing
- Degenerate knot sequences

## Implementation Requirements

### Core Algorithm Components
1. **Knot Interval Finding**
   - Binary search for x in tx, y in ty
   - Handle multiplicity correctly

2. **B-spline Basis Functions**
   - De Boor recursion for basis evaluation
   - Stable computation for high degrees

3. **Tensor Product Evaluation**
   - Efficient summation over active basis functions
   - Proper coefficient indexing

4. **Derivative Computation**
   - Recursive differentiation formulas
   - Maintain numerical stability

## Validation Process

### Phase 1: Unit Tests
1. Run `test_bispl_validation.py`
2. Identify failing tests
3. Implement missing functionality
4. Iterate until all tests pass

### Phase 2: Cross-Validation
1. Generate random test cases
2. Compare against scipy on 1000+ points
3. Statistical analysis of errors

### Phase 3: Performance Testing
1. Benchmark against scipy
2. Memory usage comparison
3. Compilation time with numba

### Phase 4: Integration Testing
1. Test with existing Spline2D class
2. Verify C-interoperability
3. Test in real applications

## Acceptance Criteria

### Mandatory
- [ ] All validation tests pass
- [ ] Maximum absolute error < 1e-10
- [ ] Maximum relative error < 1e-10
- [ ] No regression in existing tests

### Performance
- [ ] Within 2x scipy performance
- [ ] Memory usage comparable to scipy
- [ ] Successful numba compilation

### Compatibility
- [ ] Drop-in replacement for scipy.interpolate.bisplev
- [ ] C-callable via cfunc interface
- [ ] Thread-safe implementation

## Next Steps

1. Implement B-spline basis functions
2. Implement knot interval search
3. Complete bisplev_cfunc implementation
4. Run validation suite
5. Iterate until all criteria met

## References
- DIERCKX, P. (1993). Curve and Surface Fitting with Splines
- de Boor, C. (2001). A Practical Guide to Splines
- scipy.interpolate.bisplev source code