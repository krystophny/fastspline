This file provides **MANDATORY** guidance to Claude Code (claude.ai/code) when working with code in this repository.

**⚠️ CRITICAL: YOU MUST ADHERE TO ALL PRINCIPLES BELOW ⚠️**
These are not suggestions - they are strict requirements that MUST be followed in every code change. In particular:

1. Test-Driven Development
2. SOLID, KISS and DRY
3. Single responsibility principle

**⚠️ COMMUNICATION REQUIREMENTS ⚠️**
- Keep responses minimal and direct
- No flattery, congratulations, or celebration language
- Be brutally honest about issues and mistakes
- Answer questions with facts only
- Skip explanations unless specifically requested

**⚠️ WORK ETHICS ⚠️**
- Always prioritize correctness and clarity
- Never be lazy or take shortcuts

**⚠️ DEVELOPMENT WORKFLOW ⚠️**
1. First collect the problem and solution strategy in a github issue
2. Tackle one issue at a time
3. Write tests first, then implement code to pass tests
4. Once tests pass, clean up code while keeping tests green
5. Commit, push, close the issue

**⚠️ PROJECT SPECIFIC REQUIREMENTS ⚠️**

## FastSpline Development Guidelines

### Architecture
- Core implementation in `src/fastspline/` with three main modules:
  - `spline1d.py`: 1D spline interpolation
  - `spline2d.py`: 2D B-spline evaluation (bisplev)
  - `bisplrep.py`: B-spline surface fitting using QR decomposition

### Performance Requirements
- All evaluation functions MUST be implemented as Numba cfuncs
- Optimize for scalar evaluation first, then arrays
- Target 5-10x speedup over SciPy for common operations
- Maintain machine precision accuracy (< 1e-15 relative error)

### Code Style
- Use type hints for all function signatures
- Numba cfunc decorators must specify exact types
- No unnecessary comments - code should be self-documenting
- Aggressive inlining and manual optimizations are acceptable in hot paths

### Testing
- Tests go in `tests/` directory - NO plotting code
- Visual tests and examples go in `examples/` directory
- All tests must pass before committing
- Benchmark against SciPy for both speed and accuracy

### Key Optimizations
- Inline knot span finding (binary search with bit shifts)
- Manual register allocation in hot loops
- Specialized implementations for linear (k=1) and cubic (k=3) cases
- Cache-friendly memory access patterns

### Current Implementation Status
- ✅ Ultra-optimized bisplev with automatic meshgrid handling
- ✅ QR-based bisplrep for numerical stability
- ✅ C-compatible interface via cfunc
- ✅ Full SciPy compatibility for bisplrep/bisplev

### Commit Messages
All commits must end with:
```
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```