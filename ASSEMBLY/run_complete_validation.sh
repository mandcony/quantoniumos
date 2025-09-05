#!/bin/bash
# QuantoniumOS Assembly - Complete Validation Automation Script
# ============================================================
# Builds, tests, and validates the entire assembly implementation

set -e  # Exit on any error

echo "=================================================================="
echo "QUANTONIUMOS ASSEMBLY - AUTOMATED VALIDATION PIPELINE"
echo "=================================================================="
echo "?? Starting complete validation at $(date)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "simd_rft_core.asm" ]; then
    print_error "Must be run from ASSEMBLY/optimized directory"
    exit 1
fi

ASSEMBLY_DIR=$(pwd)
BASE_DIR=$(dirname $(dirname $ASSEMBLY_DIR))

print_status "Base directory: $BASE_DIR"
print_status "Assembly directory: $ASSEMBLY_DIR"

# Step 1: Environment setup
print_status "Setting up Python environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
print_status "Python version: $PYTHON_VERSION"

# Install required packages
print_status "Installing Python dependencies..."
pip3 install --user numpy matplotlib pandas psutil cpuinfo > /dev/null 2>&1

# Try to install optional packages
pip3 install --user seaborn > /dev/null 2>&1 || print_warning "seaborn not installed (optional)"

print_success "Python environment ready"

# Step 2: Build assembly implementation
print_status "Building optimized assembly implementation..."

if [ -f "build_optimized.sh" ]; then
    chmod +x build_optimized.sh
    ./build_optimized.sh
    if [ $? -eq 0 ]; then
        print_success "Assembly build completed"
    else
        print_warning "Assembly build failed, continuing with validation"
    fi
else
    print_warning "build_optimized.sh not found, skipping assembly build"
fi

# Step 3: Create validation output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
VALIDATION_DIR="$BASE_DIR/validation_results_$TIMESTAMP"
mkdir -p "$VALIDATION_DIR"

print_status "Validation results will be saved to: $VALIDATION_DIR"

# Step 4: Run comprehensive validation
print_status "Starting comprehensive validation suite..."

cd "$ASSEMBLY_DIR/.."  # Go to ASSEMBLY directory

# Run the master validation script
VALIDATION_OUTPUT="$VALIDATION_DIR/validation_log.txt"

{
    echo "QuantoniumOS Assembly Validation Log"
    echo "Started: $(date)"
    echo "======================================="
    echo ""
} > "$VALIDATION_OUTPUT"

python3 run_validation.py --output-dir "$VALIDATION_DIR" 2>&1 | tee -a "$VALIDATION_OUTPUT"

VALIDATION_STATUS=${PIPESTATUS[0]}

# Step 5: Generate summary report
print_status "Generating validation summary..."

SUMMARY_FILE="$VALIDATION_DIR/VALIDATION_SUMMARY.md"

{
    echo "# QuantoniumOS Assembly Validation Summary"
    echo ""
    echo "**Validation Date**: $(date)"
    echo "**Validation ID**: $TIMESTAMP"
    echo "**System**: $(uname -a)"
    echo "**Python Version**: $(python3 --version)"
    echo ""
    
    if [ $VALIDATION_STATUS -eq 0 ]; then
        echo "## ? VALIDATION STATUS: SUCCESS"
        echo ""
        echo "All validation tests completed successfully."
    else
        echo "## ? VALIDATION STATUS: ISSUES DETECTED"
        echo ""
        echo "Some validation tests encountered issues. Review detailed logs for specifics."
    fi
    
    echo ""
    echo "## Generated Evidence"
    echo ""
    echo "The following evidence has been generated:"
    echo ""
    
    # List all generated files
    find "$VALIDATION_DIR" -type f -name "*.json" -o -name "*.csv" -o -name "*.png" -o -name "*.md" | while read file; do
        relative_file=${file#$VALIDATION_DIR/}
        echo "- \`$relative_file\`"
    done
    
    echo ""
    echo "## Quick Access Files"
    echo ""
    echo "- **Master Results**: \`MASTER_VALIDATION_RESULTS.json\`"
    echo "- **Executive Summary**: \`executive_summary.md\`"
    echo "- **Technical Report**: \`technical_validation_report.md\`"
    echo "- **Performance Data**: \`performance_summary.csv\`"
    echo ""
    echo "## Next Steps"
    echo ""
    if [ $VALIDATION_STATUS -eq 0 ]; then
        echo "1. ? Review performance benchmarks"
        echo "2. ? Verify quantum computing metrics"
        echo "3. ? Proceed with deployment planning"
    else
        echo "1. ?? Review detailed error logs"
        echo "2. ?? Address identified issues"
        echo "3. ?? Re-run validation after fixes"
    fi
    
} > "$SUMMARY_FILE"

# Step 6: Create evidence package
print_status "Creating evidence package..."

EVIDENCE_PACKAGE="$BASE_DIR/QuantoniumOS_Validation_Evidence_$TIMESTAMP.tar.gz"

cd "$BASE_DIR"
tar -czf "$EVIDENCE_PACKAGE" "validation_results_$TIMESTAMP/"

print_success "Evidence package created: $EVIDENCE_PACKAGE"

# Step 7: Generate quick stats
print_status "Generating quick statistics..."

STATS_FILE="$VALIDATION_DIR/QUICK_STATS.txt"

{
    echo "QuantoniumOS Assembly Validation - Quick Statistics"
    echo "=================================================="
    echo ""
    echo "Files Generated: $(find "$VALIDATION_DIR" -type f | wc -l)"
    echo "JSON Data Files: $(find "$VALIDATION_DIR" -name "*.json" | wc -l)"
    echo "CSV Data Files: $(find "$VALIDATION_DIR" -name "*.csv" | wc -l)"
    echo "Plot Files: $(find "$VALIDATION_DIR" -name "*.png" | wc -l)"
    echo "Report Files: $(find "$VALIDATION_DIR" -name "*.md" | wc -l)"
    echo ""
    echo "Total Evidence Size: $(du -sh "$VALIDATION_DIR" | cut -f1)"
    echo "Evidence Package Size: $(du -sh "$EVIDENCE_PACKAGE" | cut -f1)"
    echo ""
    echo "Validation Duration: $(date)"
    
} > "$STATS_FILE"

# Step 8: Final output and recommendations
echo ""
echo "=================================================================="
echo "VALIDATION PIPELINE COMPLETE"
echo "=================================================================="

if [ $VALIDATION_STATUS -eq 0 ]; then
    print_success "? VALIDATION SUCCESSFUL"
    echo ""
    print_status "?? Evidence Package: $EVIDENCE_PACKAGE"
    print_status "?? Detailed Results: $VALIDATION_DIR"
    print_status "?? Quick Summary: $SUMMARY_FILE"
    echo ""
    print_success "?? READY FOR PRODUCTION DEPLOYMENT"
    echo ""
    echo "Key files to review:"
    echo "  • $SUMMARY_FILE"
    echo "  • $VALIDATION_DIR/executive_summary.md"
    echo "  • $VALIDATION_DIR/technical_validation_report.md"
    
else
    print_error "? VALIDATION ENCOUNTERED ISSUES"
    echo ""
    print_status "?? Review Details: $VALIDATION_OUTPUT"
    print_status "?? Evidence Directory: $VALIDATION_DIR"
    echo ""
    print_warning "?? REVIEW REQUIRED BEFORE DEPLOYMENT"
fi

echo ""
print_status "Validation completed at $(date)"

# Optional: Open summary if on macOS/Linux with GUI
if [ "$1" = "--open" ]; then
    if command -v open &> /dev/null; then
        open "$VALIDATION_DIR"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$VALIDATION_DIR"
    fi
fi

exit $VALIDATION_STATUS