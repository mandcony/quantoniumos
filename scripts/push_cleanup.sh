#!/bin/bash
cd /workspaces/quantoniumos
git add -A
git commit -m "Clean up root directory - remove temporary files

- Removed temporary documentation (PATH_ALIGNMENT_COMPLETE.md, PYQT_*.md, RESTORATION_COMPLETE.md)
- Removed temporary scripts (check_all_imports.py, git_push.sh, setup_quantoniumos.sh)
- Removed LaTeX compilation artifacts (*.aux, *.bbl, *.blg, *.log, *.out)
- Removed test logs (test_results.log, assembly_vs_classical_results.txt)
- Cleaned Python cache (__pycache__, *.pyc, .pytest_cache)
- Cleaned up cleanup script itself"

git push origin main
echo "✅ Changes pushed to GitHub"
