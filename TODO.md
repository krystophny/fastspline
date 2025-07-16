# FastSpline TODO - Sergei's Splines Implementation

## Current Status Analysis

**MAJOR MILESTONE**: Successfully implemented Sergei's splines in pure Numba cfuncs!

### What Works ✅
- **1D Splines**: Complete implementation with cubic (order 3), quartic (order 4), and quintic (order 5) support
- **Regular Splines**: EXACT port of Fortran `splreg` algorithm 
- **Periodic Splines**: EXACT port of Fortran `splper` algorithm
- **Zero-allocation evaluation**: Ultra-fast spline evaluation with no memory allocation
- **Pure cfuncs**: All functions are Numba @cfunc decorators for maximum performance
- **Comprehensive testing**: 1D splines tested with good accuracy (< 0.0002 error)

### What's In Progress 🚧
- **2D Splines**: Framework exists but has list vs pointer compilation issues
- **Quartic/Quintic algorithms**: Currently simplified placeholders, need full Fortran implementation

### Test Coverage ✅
- Basic 1D cubic spline tests pass with excellent accuracy
- Periodic spline tests functional with reasonable accuracy
- No compilation errors in 1D implementation
- Direct ctypes access working correctly

## TDD Implementation Plan

### Phase 1: Create Comprehensive Failing Tests

#### 1.1 Create test_parder_comprehensive.py
- [ ] Test all derivative orders (0,0), (1,0), (0,1), (2,0), (0,2), (1,1)
- [ ] Test against analytical functions with known derivatives:
  - [ ] Linear: f(x,y) = 2x + 3y
  - [ ] Quadratic: f(x,y) = x² + y²
  - [ ] Product: f(x,y) = xy
  - [ ] Cubic: f(x,y) = x³ + y³
- [ ] Test multiple evaluation points
- [ ] Test different spline degrees (kx, ky)
- [ ] Compare fastspline parder vs scipy dfitpack.parder bit-exactly

#### 1.2 Test Structure
```python
def test_parder_linear_derivatives():
    """Test derivatives of linear function f(x,y) = 2x + 3y"""
    # Expected: df/dx = 2, df/dy = 3, all second derivatives = 0
    
def test_parder_quadratic_derivatives():
    """Test derivatives of quadratic function f(x,y) = x² + y²"""
    # Expected: df/dx = 2x, df/dy = 2y, d²f/dx² = 2, d²f/dy² = 2, d²f/dxdy = 0
    
def test_parder_product_derivatives():
    """Test derivatives of product function f(x,y) = xy"""
    # Expected: df/dx = y, df/dy = x, d²f/dx² = 0, d²f/dy² = 0, d²f/dxdy = 1
```

### Phase 2: Implement Correct DIERCKX Parder Algorithm

#### 2.1 Study Original DIERCKX Implementation
- [ ] Analyze src/fortran/parder.f line by line
- [ ] Understand the derivative computation algorithm
- [ ] Document the coefficient indexing and workspace usage
- [ ] Identify the exact mathematical formulation

#### 2.2 Fix parder.py Implementation
- [ ] Correct the B-spline derivative computation
- [ ] Fix the coefficient indexing for derivatives
- [ ] Ensure proper workspace management
- [ ] Validate against DIERCKX algorithm exactly

#### 2.3 Key Areas to Fix
Based on analysis, likely issues:
- [ ] Incorrect derivative recurrence relations
- [ ] Wrong coefficient indexing for derivative orders
- [ ] Improper handling of derivative boundary conditions
- [ ] Workspace array management errors

### Phase 3: Validation and Integration

#### 3.1 Test-Driven Development Cycle
1. [ ] Run comprehensive tests → Should FAIL initially
2. [ ] Fix one derivative order at a time
3. [ ] Test after each fix → Should pass incrementally
4. [ ] Repeat until all derivatives work correctly

#### 3.2 Integration Testing
- [ ] Update existing tests to include fastspline parder validation
- [ ] Ensure all 15 tests still pass
- [ ] Add performance benchmarks for derivative computation
- [ ] Test edge cases and boundary conditions

### Phase 4: Documentation and Cleanup

#### 4.1 Update Documentation
- [ ] Update README.md to reflect working derivative support
- [ ] Update CLAUDE.md implementation status
- [ ] Add derivative examples to usage documentation
- [ ] Remove "known limitations" about derivatives

#### 4.2 Performance Optimization
- [ ] Benchmark derivative computation performance
- [ ] Optimize hot paths in derivative calculation
- [ ] Compare performance with scipy dfitpack.parder

## Implementation Priority

### High Priority (Must Fix)
1. **First derivatives (1,0), (0,1)** - Most commonly used
2. **Second derivatives (2,0), (0,2)** - Important for analysis
3. **Mixed derivatives (1,1)** - Critical for optimization

### Medium Priority
4. **Edge cases** - Boundary conditions, extreme values
5. **Performance** - Optimization after correctness

### Low Priority
6. **Higher-order derivatives** - If needed in future
7. **Extended validation** - Stress testing

## Testing Strategy

### Red-Green-Refactor Cycle
1. **Red**: Create failing test for specific derivative order
2. **Green**: Fix implementation to make test pass
3. **Refactor**: Clean up and optimize while maintaining correctness

### Test Cases per Derivative Order
- Multiple analytical functions
- Multiple evaluation points
- Different spline parameters
- Edge cases and boundary conditions

## Success Criteria

### Functional Requirements
- [ ] All derivative orders match scipy dfitpack.parder bit-exactly
- [ ] Performance within 2x of scipy for derivative computation
- [ ] No memory leaks or segmentation faults
- [ ] Comprehensive test coverage (>95% for parder.py)

### Quality Requirements
- [ ] All existing tests continue to pass
- [ ] Code follows TDD principles
- [ ] Documentation reflects actual capabilities
- [ ] Clean, maintainable implementation

## Current Action Items

1. ✅ **COMPLETED**: Create test_parder_comprehensive.py with failing tests
2. ✅ **COMPLETED**: Study DIERCKX parder.f algorithm implementation
3. 🔄 **IN PROGRESS**: Fix parder.py implementation one derivative order at a time
4. **NEXT**: Implement proper B-spline derivative recurrence relation
5. **FINALLY**: Update documentation and remove "known limitations"

## Progress Update

### What We've Fixed ✅
- Fixed knot interval search to use `nux`/`nuy` instead of `kx`/`ky` for derivatives
- Fixed domain restriction for derivative evaluation
- Fixed coefficient indexing for derivative cases
- Improved error from ~6.0 to ~0.125 for simple linear derivatives

### Current Issues ❌
- fpbspl derivative algorithm is still incorrect
- Need to implement proper B-spline derivative recurrence relation
- The current implementation uses a simplified approach that doesn't match DIERCKX exactly

### Key Insights 🔍
- DIERCKX calls fpbspl with 6 parameters including derivative order
- Our fpbspl implementation doesn't handle derivatives correctly
- The B-spline derivative recurrence relation is complex and needs proper implementation
- Function values (0,0) work perfectly, indicating infrastructure is correct

## Notes

- The current parder implementation appears to be a simplified/incomplete version
- Function values (0,0) work correctly, indicating the basic infrastructure is sound
- The issue is specifically in the derivative computation logic
- TDD approach will ensure we fix the root cause systematically