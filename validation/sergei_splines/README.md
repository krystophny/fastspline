# Sergei Splines Validation Suite

This validation suite compares the Fortran and Python (cfunc) implementations of Sergei splines to identify any memory alignment or numerical issues.

## Structure

```
validation/sergei_splines/
├── src/                      # Fortran source files
│   ├── interpolate.f90       # Main spline module
│   ├── spl_three_to_five.f90 # Spline construction routines
│   └── validate_splines.f90  # Fortran validation program
├── bin/                      # Compiled executables
├── data/                     # Output data files
├── results/                  # Log files
├── validate_python.py        # Python validation program
├── compare_results.py        # Comparison utility
├── run_validation.sh         # Main test runner
└── Makefile                  # Build configuration
```

## Usage

### Quick Run

```bash
./run_validation.sh
```

This will:
1. Build the Fortran validation program
2. Run Fortran validation
3. Run Python validation
4. Compare results and report differences

### Manual Steps

```bash
# Build Fortran program
make

# Run Fortran validation only
make run-fortran

# Run Python validation only
make run-python

# Compare results
make compare

# Clean build artifacts
make clean

# Clean everything
make distclean
```

## Output Files

The validation programs generate several output files in the `data/` directory:

### Fortran Output
- `input_data.txt` - Input test data
- `spline_coeffs_1d.txt` - 1D spline coefficients
- `evaluation_results.txt` - 1D spline evaluation results
- `input_data_2d.txt` - 2D input test data
- `evaluation_results_2d.txt` - 2D spline evaluation results

### Python Output
- `input_data_python.txt` - Input test data (should match Fortran)
- `spline_coeffs_1d_python.txt` - 1D spline coefficients
- `evaluation_results_python.txt` - 1D spline evaluation results
- `derivatives_1d_python.txt` - 1D spline derivatives
- `input_data_2d_python.txt` - 2D input test data
- `evaluation_results_2d_python.txt` - 2D spline evaluation results

## Debugging Memory Alignment Issues

If differences are detected:

1. Check `results/comparison.log` for detailed comparison
2. Look for patterns in the differences (periodic, systematic)
3. Run with debug flags: `make debug`
4. Check memory alignment info printed by Python program

## Test Cases

The validation suite tests:

1. **1D Splines**
   - Order 5 (quintic) splines
   - Non-periodic boundary conditions
   - sin(2πx) test function
   - 10 data points, 21 evaluation points

2. **2D Splines**
   - Order (5,5) tensor product splines
   - Non-periodic boundary conditions
   - sin(2πx)cos(2πy) test function
   - 8×8 data grid, 11×11 evaluation grid

## Expected Results

All differences should be within machine precision (< 1e-10). Larger differences may indicate:
- Memory alignment issues
- Array indexing problems
- Boundary condition handling differences