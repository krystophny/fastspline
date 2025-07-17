# Makefile for Sergei splines validation
# Compiles Fortran program and runs validation tests

FC = gfortran
FFLAGS = -O2 -Wall -fcheck=all -g
LDFLAGS = 

# Source files
FORTRAN_SOURCES = src/spl_three_to_five.f90 src/interpolate.f90 src/validate_splines.f90
FORTRAN_OBJECTS = $(FORTRAN_SOURCES:.f90=.o)
FORTRAN_MODULES = spl_three_to_five_sub.mod interpolate.mod

# Executable
FORTRAN_EXEC = bin/validate_splines

# Python files
PYTHON_VALIDATE = validate_python.py
PYTHON_COMPARE = compare_results.py

# Default target
all: $(FORTRAN_EXEC)

# Create directories
dirs:
	@mkdir -p bin data results

# Build Fortran executable
$(FORTRAN_EXEC): dirs $(FORTRAN_OBJECTS)
	$(FC) $(FFLAGS) -o $@ $(FORTRAN_OBJECTS) $(LDFLAGS)

# Compile Fortran sources
src/%.o: src/%.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Module dependencies
src/interpolate.o: src/spl_three_to_five.o
src/validate_splines.o: src/interpolate.o src/spl_three_to_five.o

# Run Fortran validation
run-fortran: $(FORTRAN_EXEC)
	@echo "Running Fortran validation..."
	@cd validation/sergei_splines && ../../$(FORTRAN_EXEC)

# Run Python validation
run-python:
	@echo "Running Python validation..."
	@cd validation/sergei_splines && python3 $(PYTHON_VALIDATE)

# Compare results
compare: run-fortran run-python
	@echo "Comparing results..."
	@cd validation/sergei_splines && python3 $(PYTHON_COMPARE)

# Run all validations
validate: compare

# Clean build artifacts
clean:
	rm -f src/*.o *.mod $(FORTRAN_MODULES)
	rm -f $(FORTRAN_EXEC)
	rm -rf data/*.txt

# Clean everything including results
distclean: clean
	rm -rf bin data results

# Debug build
debug: FFLAGS += -g -fbacktrace -ffpe-trap=invalid,zero,overflow
debug: clean $(FORTRAN_EXEC)

# Show configuration
info:
	@echo "Fortran compiler: $(FC)"
	@echo "Fortran flags: $(FFLAGS)"
	@echo "Sources: $(FORTRAN_SOURCES)"
	@echo "Executable: $(FORTRAN_EXEC)"

.PHONY: all dirs run-fortran run-python compare validate clean distclean debug info