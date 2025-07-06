# DIERCKX Surfit Port to Numba

## Goal
Create a full port of DIERCKX surfit routines to Numba using cfunc everywhere with nopython mode. Match scipy's bisplrep and bisplev interface and validate against scipy versions.

## Phase 1: F2PY Wrapper Setup
- [ ] Create f2py wrapper for DIERCKX surfit.f
- [ ] Create f2py wrapper for DIERCKX surfit dependencies (fpback, fpdisc, fpgivs, fprank, fprati, fprota)
- [ ] Test f2py wrapper compilation and basic functionality
- [ ] Create validation tests comparing f2py output with scipy.interpolate

## Phase 2: Numba cfunc Port - Core Functions
- [ ] Port fpback.f to Numba cfunc
- [ ] Port fpdisc.f to Numba cfunc  
- [ ] Port fpgivs.f to Numba cfunc
- [ ] Port fprank.f to Numba cfunc
- [ ] Port fprati.f to Numba cfunc
- [ ] Port fprota.f to Numba cfunc
- [ ] Create unit tests for each ported function against f2py versions

## Phase 3: Numba cfunc Port - Main Surfit
- [ ] Port surfit.f main routine to Numba cfunc
- [ ] Ensure all array operations use Numba-compatible syntax
- [ ] Handle all GOTO statements with proper control flow
- [ ] Validate intermediate results against f2py version

## Phase 4: High-Level Interface
- [ ] Create bisplrep equivalent using Numba surfit
- [ ] Create bisplev equivalent for evaluation
- [ ] Match scipy's bisplrep/bisplev API exactly
- [ ] Add proper error handling and input validation

## Phase 5: Validation & Testing
- [ ] Create comprehensive test suite comparing with scipy.interpolate.bisplrep
- [ ] Test edge cases (boundaries, knot placement, smoothing factors)
- [ ] Performance benchmarks vs scipy
- [ ] Test with real-world data examples

## Phase 6: Documentation & Optimization
- [ ] Document all functions with proper docstrings
- [ ] Profile and optimize hotspots
- [ ] Add examples and usage guide
- [ ] Final validation against scipy test suite

## Implementation Notes
- Use cfunc with nopython=True for all functions
- Match DIERCKX array indexing (1-based) in intermediate steps
- Convert to 0-based for Python interface
- Preserve numerical stability of original FORTRAN code