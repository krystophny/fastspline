# FastSpline Project Structure

## Directory Layout

```
fastspline/
├── src/                      # Source code
│   └── fastspline/          # Main package
│       ├── __init__.py      # Package exports
│       ├── spline1d.py      # 1D spline implementation
│       ├── spline2d.py      # 2D spline implementation (bisplev)
│       └── bisplrep.py      # B-spline surface fitting (bisplrep)
│
├── tests/                    # Unit tests (no plots)
│   ├── test_spline1d.py     # Tests for 1D splines
│   ├── test_spline2d.py     # Tests for 2D splines
│   ├── test_bisplrep.py     # Tests for bisplrep
│   ├── test_bisplrep_bisplev_integration.py  # Integration tests
│   ├── test_scipy_compatibility.py    # SciPy compatibility tests
│   ├── test_bispl_validation.py       # Validation tests
│   └── test_*.py            # Other automated tests
│
├── benchmarks/               # Performance benchmarks
│   ├── benchmark_comprehensive.py    # Overall performance tests
│   ├── benchmark_bisplrep_comprehensive.py  # bisplrep benchmarks
│   ├── benchmark_scipy_comparison.py # SciPy comparison
│   └── benchmark_*.py       # Other benchmark scripts
│
├── examples/                 # Example usage and visual demos
│   ├── demo_scipy_comparison.py     # Comparison with SciPy
│   ├── demo_cfunc.py               # C-compatible function demo
│   ├── visual_test.py              # Visual validation test
│   ├── visual_test_2d.py           # 2D visual validation
│   ├── test_missing_data_validation.py  # Missing data visualization
│   ├── example_missing_data_visualization.py  # Missing data example
│   └── validate_*.py               # Validation scripts
│
├── tools/                    # Development tools
│   ├── profile_bottlenecks.py      # Performance profiling
│   ├── profile_detailed_bottlenecks.py  # Detailed profiling
│   └── debug_accuracy.py           # Accuracy debugging
│
├── thirdparty/              # Third-party reference code
│   └── dierckx/            # Original FITPACK Fortran code
│
├── pyproject.toml           # Project configuration
├── README.md                # Main documentation
├── CLAUDE.md               # Claude AI instructions
├── DESIGN.md               # Design documentation
├── CFUNC_USAGE.md          # C-function usage guide
├── VALIDATION_PLAN.md      # Validation planning
├── PROJECT_STRUCTURE.md    # This file
└── LICENSE                 # License information
```

## Key Components

### Core Implementation (`src/fastspline/`)

- **spline1d.py**: Fast 1D spline interpolation with periodic boundary support
- **spline2d.py**: Optimized 2D B-spline evaluation (bisplev) with automatic meshgrid handling
- **bisplrep.py**: QR-based B-spline surface fitting for numerical stability

### Testing (`tests/`)

Automated unit and integration tests that run without generating plots:
- Unit tests for individual functions
- Integration tests for bisplrep/bisplev workflow
- Accuracy validation against SciPy
- Performance regression tests
- C-function interface tests

Run with: `pytest tests/`

### Benchmarks (`benchmarks/`)

Performance comparison scripts measuring:
- Construction time (bisplrep)
- Evaluation speed (bisplev)
- Comparison with SciPy implementation
- Scaling with data size

### Examples (`examples/`)

Demonstration scripts and visual tests that may produce plots:
- Basic usage patterns
- SciPy compatibility demonstrations
- C-function integration examples
- Visual validation of interpolation results
- Missing data handling visualization

### Tools (`tools/`)

Development and debugging utilities:
- Performance profiling scripts
- Accuracy debugging tools
- Bottleneck analysis

## Usage

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Run tests (no plots):
   ```bash
   pytest tests/
   ```

3. Run benchmarks:
   ```bash
   python benchmarks/benchmark_comprehensive.py
   ```

4. See visual examples:
   ```bash
   python examples/visual_test_2d.py
   ```

5. Profile performance:
   ```bash
   python tools/profile_bottlenecks.py
   ```

## File Organization Principles

- **tests/**: Contains only automated tests that can run in CI/CD without display
- **examples/**: Contains demonstration code, visual tests, and anything that produces plots
- **benchmarks/**: Performance measurement scripts (may produce comparison plots)
- **tools/**: Development utilities for debugging and profiling