#!/bin/bash
# Clean up root directory - remove temporary organizational files

cd /workspaces/quantoniumos

echo "🧹 Cleaning up root directory..."
echo ""

# Remove temporary markdown files
echo "📄 Removing temporary documentation files..."
rm -f PATH_ALIGNMENT_COMPLETE.md
rm -f PYQT_APPS_AUDIT.md
rm -f PYQT_FIXES_COMPLETE.md
rm -f RESTORATION_COMPLETE.md

# Remove temporary check/test scripts
echo "🔧 Removing temporary scripts..."
rm -f check_all_imports.py
rm -f git_push.sh
rm -f setup_quantoniumos.sh

# Remove paper compilation artifacts
echo "📝 Removing LaTeX compilation artifacts..."
rm -f paper.aux
rm -f paper.bbl
rm -f paper.blg
rm -f paper.log
rm -f paper.out

# Remove test result logs
echo "🧪 Removing temporary test logs..."
rm -f test_results.log
rm -f assembly_vs_classical_results.txt

# Remove Python cache
echo "🐍 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove pytest cache
echo "🧪 Cleaning pytest cache..."
rm -rf .pytest_cache

echo ""
echo "✅ Root directory cleaned!"
echo ""
echo "📋 Remaining important files:"
ls -1 *.md *.txt *.py 2>/dev/null | head -20
