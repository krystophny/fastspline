# FastSpline - Ultra-Optimized DIERCKX Implementation

High-performance Numba implementation of DIERCKX spline library with cfunc optimization.

## Core Files

### Implementations
- `dierckx_numba_optimized.py` - Optimized Numba implementation with performance flags
- `dierckx_numba_ultra.py` - **Ultra-optimized cfunc implementation** (maximum performance)

### Validation & Benchmarking
- `comprehensive_validation.py` - Complete validation against DIERCKX f2py reference
- `ultra_validation_simple.py` - Ultra-optimized validation tests
- `dierckx_vs_numba_benchmark.py` - Performance comparison vs DIERCKX
- `ultra_benchmark.py` - **Ultra performance analysis and scaling plots**

### DIERCKX f2py Interface
- `build_dierckx_f2py.sh` - Builds corrected DIERCKX f2py wrapper
- `dierckx_f2py_corrected.pyf` - Fixed f2py interface with cf2py directives
- `dierckx_f2py_fixed.cpython-311-x86_64-linux-gnu.so` - Working f2py library

## Quick Start

```bash
# Build DIERCKX f2py reference
./build_dierckx_f2py.sh

# Run validation tests
python comprehensive_validation.py
python ultra_validation_simple.py

# Run performance benchmarks
python ultra_benchmark.py
python dierckx_vs_numba_benchmark.py
```

## Performance Results

- **fpback**: Up to 1.33× speedup with ultra cfunc optimization
- **Validation**: All functions validated to floating point accuracy
- **Scaling**: Performance advantage increases with problem size
- **Generated plots**: `examples/ultra_dierckx_performance.png`

## Key Features

- ✅ Ultra-optimized cfunc implementations with maximum Numba flags
- ✅ Comprehensive validation against DIERCKX reference
- ✅ Performance scaling analysis and visualization
- ✅ Static memory allocation and manual loop unrolling
- ✅ Floating point accuracy validation