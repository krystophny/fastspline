# Derivative Mismatch Issue Report

## Problem

Your colleague reported that **cfunc and scipy derivatives don't match**. Investigation confirms this is a **critical issue**.

## Root Cause

The cfunc implementations in `fastspline/numba_implementation/` have multiple serious problems:

### 1. Compilation Errors
- `parder_fixed.py`: Numba compilation fails with "Untyped global name 'fpbspl_inline'"
- `parder_numba.py`: Similar compilation issues with inline functions
- **Result**: cfunc implementations are not functional

### 2. Algorithm Bugs
- Derivative computation logic has implementation errors
- B-spline derivative recurrence relation incorrectly implemented
- Workspace management and indexing issues

### 3. Testing Gap
- No automated tests comparing cfunc vs scipy derivatives
- Existing tests only check scipy internal consistency
- **Result**: Bugs went undetected

## Verification

Testing shows scipy.interpolate.dfitpack.parder works correctly:

```python
# f(x,y) = x² + y²
# Expected: ∂f/∂x(0.5, 0.5) = 1.0, ∂f/∂y(0.5, 0.5) = 1.0

dfitpack.parder results:
  ∂f/∂x(0.5, 0.5): 1.000000 ✓
  ∂f/∂y(0.5, 0.5): 1.000000 ✓
  ∂²f/∂x²(0.5, 0.5): 2.000000 ✓
  ∂²f/∂y²(0.5, 0.5): 2.000000 ✓
```

But cfunc implementations **cannot even compile**.

## Impact

- **HIGH SEVERITY**: Derivative functionality is completely broken
- **Production Risk**: Any code using cfunc derivatives will fail
- **Data Integrity**: Incorrect derivatives could lead to wrong scientific results

## Required Actions

### Immediate (Critical)
1. **Fix cfunc implementations** or **remove them entirely**
2. **Add derivative comparison tests** to prevent regression
3. **Update documentation** to clarify which implementations work

### Medium-term
1. **Implement proper cfunc derivatives** with exact scipy matching
2. **Add comprehensive derivative test suite**
3. **Set up CI to catch compilation failures**

## Recommendations

### Option 1: Remove Broken cfunc (Recommended)
- Remove `parder_fixed.py` and `parder_numba.py` 
- Update documentation to use only scipy.interpolate.dfitpack.parder
- Add tests to ensure scipy derivatives work correctly

### Option 2: Fix cfunc Implementation
- Rewrite cfunc implementations to compile successfully
- Implement exact line-by-line port from parder.f
- Add comprehensive testing against scipy
- **Time estimate**: 2-3 weeks of careful development

## Test Results

The scipy implementation works perfectly:
- ✅ Exact derivatives for polynomial functions
- ✅ Consistent across all derivative orders
- ✅ Matches theoretical expectations
- ✅ Stable across different smoothing parameters

## Conclusion

**The derivative mismatch is due to broken cfunc implementations that don't even compile.** 

The scipy.interpolate.dfitpack.parder implementation works correctly and should be used until proper cfunc implementations are developed and thoroughly tested.