# QuantoniumOS Textbook

This directory contains the source code for the **QuantoniumOS Comprehensive Textbook** (Version 2.0).

## Contents

The textbook covers:

| Part | Chapters | Topics |
|------|----------|--------|
| **I: Foundations** | 1-2 | Introduction, Golden Ratio Mathematics |
| **II: RFT** | 3-6 | Transform Definition, Unitarity, 14 Variants, Wave Computing |
| **III: Quantum** | 7-8 | Qubit Basics, Grover's Algorithm |
| **IV: Applications** | 9-11 | H3 Compression, Cryptography, RFTPU Hardware |
| **Appendix** | A-B | Quick Reference, Glossary |

## Features

- **Study Checklists**: Each chapter starts with a todo list for self-assessment
- **Key Concept Boxes**: Highlighted important ideas
- **Code Examples**: Directly from the QuantoniumOS repository
- **TikZ Diagrams**: Architecture and algorithm visualizations
- **Theorem/Proof Structure**: Rigorous mathematical presentation

## How to Build

You need a LaTeX distribution installed (like TeX Live or MiKTeX).

### Required Packages

```
amsmath, amssymb, amsthm, graphicx, hyperref, listings, 
xcolor, fancyhdr, titlesec, tcolorbox, booktabs, tikz, pgfplots
```

### Using CLI (pdflatex)

```bash
cd docs/textbook
pdflatex QuantoniumOS_Textbook.tex
pdflatex QuantoniumOS_Textbook.tex  # Run twice for TOC/references
```

### Using latexmk (recommended)

```bash
latexmk -pdf QuantoniumOS_Textbook.tex
```

### Output
The command will generate `QuantoniumOS_Textbook.pdf` (~50 pages).
