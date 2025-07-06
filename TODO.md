# DIERCKX Surfit Port to Numba

## Goal
Create a full port of DIERCKX surfit routines to Numba using cfunc everywhere with nopython mode. Match scipy's bisplrep and bisplev interface and validate against scipy versions. Achieve perfect numerical match with DIERCKX fortran implementation.

## Phase 1: F2PY Wrapper Setup
- [x] Create f2py wrapper for DIERCKX surfit.f
- [x] Create f2py wrapper for DIERCKX surfit dependencies (fpback, fpdisc, fpgivs, fprank, fprati, fprota)
- [x] Test f2py wrapper compilation and basic functionality
- [ ] Fix f2py wrapper to correctly call DIERCKX surfit
- [ ] Create validation tests comparing f2py output with scipy.interpolate

## Phase 2: Numba cfunc Port - Core Functions
- [ ] Port fpback.f to Numba cfunc (nopython mode)
- [ ] Port fpdisc.f to Numba cfunc (nopython mode)
- [ ] Port fpgivs.f to Numba cfunc (nopython mode)
- [ ] Port fprank.f to Numba cfunc (nopython mode)
- [ ] Port fprati.f to Numba cfunc (nopython mode)
- [ ] Port fprota.f to Numba cfunc (nopython mode)
- [ ] Port fporde.f to Numba cfunc (nopython mode)
- [ ] Port fpbspl.f to Numba cfunc (nopython mode)
- [ ] Create unit tests ensuring perfect numerical match with f2py versions

## Phase 3: Numba cfunc Port - Main Routines
- [ ] Port fpsurf.f to Numba cfunc (nopython mode)
- [ ] Port surfit.f main routine to Numba cfunc (nopython mode)
- [ ] Ensure all array operations use Numba-compatible syntax
- [ ] Handle all GOTO statements with proper control flow
- [ ] Validate perfect numerical match against f2py version

## Phase 4: High-Level Interface
- [ ] Create bisplrep equivalent using Numba surfit
- [ ] Create bisplev equivalent for evaluation (port bispev.f)
- [ ] Match scipy's bisplrep/bisplev API exactly
- [ ] Add proper error handling and input validation

## Phase 5: Validation & Testing
- [ ] Create comprehensive test suite with perfect match validation
- [ ] Test all error conditions and edge cases
- [ ] Performance benchmarks: Numba cfunc vs DIERCKX fortran
- [ ] Test with scipy's test data for bisplrep/bisplev
- [ ] Verify bit-for-bit reproducibility with DIERCKX

## Phase 6: Documentation & Optimization
- [ ] Document all functions with proper docstrings
- [ ] Profile and optimize while maintaining numerical accuracy
- [ ] Add examples and usage guide
- [ ] Final validation against scipy test suite

## Implementation Requirements
- ALL functions must use @cfunc with nopython=True
- Perfect numerical match with DIERCKX fortran output
- Match DIERCKX array indexing (1-based) internally
- Convert to 0-based for Python interface only
- Preserve exact numerical behavior of original FORTRAN code
- Include performance metrics in all tests