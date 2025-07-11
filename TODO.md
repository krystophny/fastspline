# TODO: C Wrapper for DIERCKX bispev.f

## Objective
Create a C wrapper around the DIERCKX `bispev.f` Fortran routine that can be used as a cfunc (callable from Numba/ctypes) while maintaining exact validation against scipy's implementation.

## Implementation Plan

### Phase 1: Analysis and Setup
- [ ] Download and analyze the original `bispev.f` from scipy repository
- [ ] Study the Fortran subroutine signature and understand all parameters
- [ ] Identify any dependencies (other DIERCKX routines called by bispev)
- [ ] Create project structure for C wrapper development

### Phase 2: Fortran to C Interface
- [ ] Create C header file defining the bispev function signature
- [ ] Write C wrapper function that matches Fortran calling conventions
- [ ] Handle Fortran array layout (column-major) to C (row-major) if needed
- [ ] Ensure proper type mapping (REAL*8 → double, INTEGER → int)

### Phase 3: Build System
- [ ] Create Makefile or CMake configuration for building the wrapper
- [ ] Link against DIERCKX Fortran library (either from scipy or standalone)
- [ ] Generate shared library (.so) that can be loaded via ctypes
- [ ] Test basic loading and symbol resolution

### Phase 4: Python Interface
- [ ] Create ctypes wrapper for the C function
- [ ] Define proper argtypes and restype for the ctypes function
- [ ] Create Python convenience wrapper with numpy array handling
- [ ] Ensure memory layout compatibility

### Phase 5: Validation Framework
- [ ] Create test harness comparing C wrapper output to scipy.interpolate
- [ ] Test with various input sizes and parameter combinations
- [ ] Verify bit-exact floating point results
- [ ] Test edge cases (extrapolation, boundary conditions)

### Phase 6: Performance Testing
- [ ] Create benchmarks comparing scipy vs C wrapper performance
- [ ] Test overhead of ctypes calls vs scipy's f2py interface
- [ ] Profile memory usage and allocation patterns
- [ ] Document performance characteristics

### Phase 7: Numba Integration
- [ ] Create Numba cfunc signature for the C wrapper
- [ ] Test calling from Numba-jitted functions
- [ ] Verify no Python interpreter involvement in hot path
- [ ] Create example of using in Numba-accelerated code

## Technical Considerations

### Key Parameters of bispev
- `tx, ty`: Knot positions in x and y dimensions
- `nx, ny`: Number of knots
- `c`: B-spline coefficients
- `kx, ky`: Degrees of the spline
- `x, y`: Points to evaluate (can be arrays)
- `z`: Output array for results
- `m`: Number of points to evaluate
- `wrk`: Working space array
- `lwrk`: Size of working space
- `ier`: Error flag

### Validation Strategy
1. Start with simple test cases (regular grids, known functions)
2. Progress to scipy's test suite examples
3. Use property-based testing for comprehensive coverage
4. Ensure exact binary compatibility of results

### Potential Challenges
- Fortran uses 1-based indexing, C uses 0-based
- Column-major vs row-major array layout
- Fortran implicit interfaces vs explicit C declarations
- Error handling and return value conventions
- Thread safety of the Fortran code