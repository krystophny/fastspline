#!/bin/bash
# Run complete validation suite for Sergei splines

set -e  # Exit on error

echo "=========================================="
echo "Sergei Splines Validation Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}[PASS]${NC} $2"
    else
        echo -e "${RED}[FAIL]${NC} $2"
    fi
}

# Change to validation directory
cd "$(dirname "$0")"

# Clean previous results
echo "Cleaning previous results..."
make clean > /dev/null 2>&1
rm -rf data/*.txt results/*.txt

# Build Fortran program
echo ""
echo "Building Fortran validation program..."
make > build.log 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "Fortran build successful"
else
    print_status 1 "Fortran build failed"
    echo "Check build.log for details"
    exit 1
fi

# Create data directory
mkdir -p data results

# Run Fortran validation
echo ""
echo "Running Fortran validation..."
./bin/validate_splines > results/fortran_output.log 2>&1
print_status $? "Fortran validation"

# Check if Fortran produced output files
if [ ! -f "data/evaluation_results.txt" ]; then
    echo -e "${RED}ERROR:${NC} Fortran program did not produce expected output files"
    exit 1
fi

# Run Python validation
echo ""
echo "Running Python validation..."
python3 validate_python.py > results/python_output.log 2>&1
print_status $? "Python validation"

# Check if Python produced output files
if [ ! -f "data/evaluation_results_python.txt" ]; then
    echo -e "${RED}ERROR:${NC} Python program did not produce expected output files"
    exit 1
fi

# Compare results
echo ""
echo "Comparing results..."
python3 compare_results.py > results/comparison.log 2>&1
COMPARE_STATUS=$?
print_status $COMPARE_STATUS "Result comparison"

# Show comparison summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
cat results/comparison.log | tail -20

# Additional memory diagnostics if there are issues
if [ $COMPARE_STATUS -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}Running additional memory diagnostics...${NC}"
    
    # Check system info
    echo ""
    echo "System information:"
    uname -a
    echo "Compiler version:"
    gfortran --version | head -1
    echo "Python version:"
    python3 --version
    
    # Run with debug flags
    echo ""
    echo "Running Fortran with debug flags..."
    make debug > /dev/null 2>&1
    ./bin/validate_splines > results/fortran_debug.log 2>&1
    
    echo ""
    echo -e "${YELLOW}Check results/ directory for detailed logs${NC}"
fi

echo ""
echo "=========================================="
if [ $COMPARE_STATUS -eq 0 ]; then
    echo -e "${GREEN}All validation tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Validation tests failed - check for memory alignment issues${NC}"
    echo "See results/comparison.log for details"
    exit 1
fi