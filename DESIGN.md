# FastSpline Performance Design Document

## B-spline Evaluation Optimization Journey

### Performance Timeline

| Implementation | Time (10k evals) | Speedup | Algorithm Complexity |
|---|---|---|---|
| Original naive | ~28,200ms | 1x | O(n²) basis evaluation |
| Optimized v1 | ~113ms | 250x | O(k² log n) with functions |
| Ultra-fast inline | ~47.5ms | 600x | O(k² log n) fully inlined |
| SciPy reference | ~25.0ms | 1128x | Fortran DIERCKX |

### Key Optimizations Implemented

#### 1. Algorithmic Complexity Reduction (250x speedup)
**Problem**: Original implementation used O(n²) naive basis function evaluation
```python
# BEFORE: O(n²) - evaluated ALL basis functions
for i in range(nx):
    for j in range(ny):
        basis_x = compute_basis(x, tx, i, kx)  # Computed for all points
        basis_y = compute_basis(y, ty, j, ky)
        result += basis_x * basis_y * c[i, j]
```

**Solution**: Implemented proper B-spline evaluation with O(k² log n) complexity
```python
# AFTER: O(k² log n) - only non-zero basis functions
span_x = find_knot_span_binary_search(x, tx, kx)  # O(log n)
span_y = find_knot_span_binary_search(y, ty, ky)  # O(log n)
# Only evaluate (k+1)² non-zero basis functions - O(k²)
```

#### 2. Inline Optimization (2.4x additional speedup)
**Problem**: Function call overhead and stack allocation in hot loops
- `find_knot_span()` called for every evaluation
- `basis_functions()` created temporary arrays
- Cox-de Boor recursion had function call overhead

**Solution**: Fully inlined implementation
```python
@cfunc(nopython=True, fastmath=True, boundscheck=False)
def bisplev_cfunc(x, y, tx, ty, c, kx, ky, nx, ny):
    # === INLINE KNOT SPAN FINDING ===
    if x >= tx[mx]:
        span_x = mx - 1
    elif x <= tx[kx]:
        span_x = kx
    else:
        # Inline binary search - no function calls
        low, high = kx, mx
        while x < tx[mid] or x >= tx[mid + 1]:
            # Binary search logic inlined
    
    # === INLINE COX-DE BOOR BASIS FUNCTIONS ===
    # No temporary arrays, no function calls
    Nx0, Nx1, Nx2, Nx3 = 1.0, 0.0, 0.0, 0.0
    # Degree 1, 2, 3 computations inlined...
    
    # === INLINE TENSOR PRODUCT ===
    # Unrolled loops for 4x4 = 16 terms
    result += Nx0 * Ny0 * c[idx_x * my + idx_y]
    # ... 15 more terms unrolled
```

#### 3. Memory Access Optimization
- **Stack allocation**: No heap allocations in evaluation
- **Cache-friendly indexing**: Row-major coefficient access
- **Minimal bounds checking**: Fast path for common cases, fallback for edge cases

### Current Performance Analysis

#### Comparison with SciPy DIERCKX

| Spline Type | SciPy Time | FastSpline Time | Speedup | Winner |
|---|---|---|---|---|
| **Linear (k=1)** | 24.2ms | 22.4ms | **1.08x** | **FastSpline** |
| **Cubic (k=3)** | 24.5ms | 47.6ms | 0.51x | SciPy |

**Why FastSpline wins for linear splines (k=1):**
1. **Simpler computation**: 2×2 tensor product vs 4×4 for cubics
2. **No recursion**: Linear basis functions are direct interpolation
3. **Better cache efficiency**: Fewer memory accesses per evaluation
4. **Numba optimization**: JIT excels at simple arithmetic patterns

**Why SciPy wins for cubic splines (k=3):**
1. **Native Fortran**: Direct machine code, no JIT overhead
2. **DIERCKX library**: 40+ years of Cox-de Boor optimization
3. **Complex recursion**: Fortran handles branching more efficiently
4. **Memory layout**: Column-major access patterns optimized

### Accuracy Validation
- **Machine precision agreement**: Max difference 2.13e-14 for cubic splines
- **Numerical stability**: Identical to SciPy across all test cases
- **Edge case handling**: Robust bounds checking for small coefficient arrays

## Further Optimization Plan

### Phase 1: Low-hanging fruit (Target: 1.5x faster)

#### 1A. SIMD Vectorization
**Current bottleneck**: Scalar arithmetic in basis function computation
```python
# Target: Vectorize basis function computation
# Use numpy operations where possible within cfunc constraints
```

#### 1B. Coefficient Access Pattern Optimization
**Issue**: Row-major vs column-major memory access
```python
# Current: c[idx_x * my + idx_y] - potential cache misses
# Target: Investigate memory layout optimization
```

#### 1C. Branch Reduction
**Issue**: Conditional logic in basis function computation
```python
# Current: if/else chains for basis function selection
# Target: Lookup tables or mathematical formulations
```

### Phase 2: Advanced optimizations (Target: 1.3x faster)

#### 2A. Coefficient Prefetching
```python
# Pre-load relevant coefficients into registers/cache
# Exploit spatial locality in coefficient access
```

#### 2B. Specialized Kernels
```python
# Separate optimized kernels for:
# - Linear splines (k=1): 2x2 tensor product
# - Cubic splines (k=3): 4x4 tensor product
# - Common knot configurations
```

#### 2C. Assembly Integration
```python
# Critical path in assembly for:
# - Binary search (branch prediction)
# - Tensor product computation (SIMD)
# - Coefficient indexing
```

### Phase 3: Architectural improvements (Target: 1.2x faster)

#### 3A. Batch Evaluation
```python
@cfunc
def bisplev_batch_cfunc(x_array, y_array, tx, ty, c, kx, ky, nx, ny):
    # Vectorized evaluation of multiple points
    # Amortize knot span finding overhead
    # SIMD tensor product computation
```

#### 3B. Adaptive Knot Span Caching
```python
# Cache knot spans for nearby evaluation points
# Exploit spatial coherence in evaluation patterns
```

#### 3C. Parallel Evaluation
```python
# Thread-level parallelism for large evaluation batches
# NUMA-aware memory allocation
```

### Implementation Priority

1. **Phase 1A (SIMD)**: Highest impact, moderate effort
2. **Phase 1B (Memory)**: High impact, low effort  
3. **Phase 2B (Kernels)**: Medium impact, low effort
4. **Phase 1C (Branches)**: Medium impact, medium effort
5. **Phase 2A (Prefetch)**: Low impact, high effort

### Target Performance Goals

| Phase | Target Time | Speedup vs Current | Speedup vs SciPy |
|---|---|---|---|
| Current | 47.5ms | 1.0x | 0.53x |
| Phase 1 | 32ms | 1.5x | 0.78x |  
| Phase 2 | 25ms | 1.9x | 1.0x |
| Phase 3 | 20ms | 2.4x | 1.25x |

### Success Metrics

1. **Performance**: Match or exceed SciPy performance
2. **Accuracy**: Maintain machine precision agreement
3. **Compatibility**: Preserve C-compatible cfunc interface
4. **Maintainability**: Keep code readable and testable

### Risk Assessment

**High Risk**:
- Assembly integration may break cross-platform compatibility
- SIMD operations may not be available in all Numba versions

**Medium Risk**:
- Memory layout changes may affect accuracy
- Batch evaluation requires API changes

**Low Risk**:
- Specialized kernels are additive optimizations
- Branch reduction preserves current logic

### Next Steps

1. **Immediate**: Implement Phase 1A SIMD vectorization
2. **Short-term**: Profile memory access patterns (Phase 1B)
3. **Medium-term**: Develop specialized linear/cubic kernels (Phase 2B)
4. **Long-term**: Investigate batch evaluation API (Phase 3A)