#!/bin/bash
set -e

echo "Installing LaTeX..."
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-latex-extra texlive-science

echo "Installing Python dependencies..."
pip install pandas matplotlib scipy numpy

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Generating Data..."
python3 scripts/verify_rate_distortion.py --export figures/latex_data/rate_distortion.csv
python3 tests/benchmarks/rft_wave_computer_demo.py --export figures/latex_data/wave_computer.csv
python3 scripts/irrevocable_truths.py --export figures/latex_data/unitarity_errors.csv

echo "Generating Figures..."
python3 scripts/generate_pdf_figures_for_latex.py

echo "Compiling Paper..."
cd docs/research
pdflatex -interaction=nonstopmode THE_PHI_RFT_FRAMEWORK_PAPER.tex
pdflatex -interaction=nonstopmode THE_PHI_RFT_FRAMEWORK_PAPER.tex

echo "Done! Paper is at docs/research/THE_PHI_RFT_FRAMEWORK_PAPER.pdf"
