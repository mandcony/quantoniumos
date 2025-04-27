#!/usr/bin/env python3
"""
Simple PDF generator for the QuantoniumOS Technical Paper
"""

import os
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Set font for the header
        self.set_font('Arial', 'B', 10)
        # Title
        self.cell(0, 10, 'QuantoniumOS Technical Paper', 0, 1, 'R')
        # Line break
        self.ln(10)
    
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Set font for the footer
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        # Set font for chapter title
        self.set_font('Arial', 'B', 14)
        # Title
        self.cell(0, 10, title, 0, 1, 'L')
        # Line break
        self.ln(5)

    def chapter_body(self, text):
        # Set font for chapter body
        self.set_font('Arial', '', 11)
        # Print justified text
        self.multi_cell(0, 6, text)
        # Line break
        self.ln(5)
    
    def section_title(self, title):
        # Set font for section title
        self.set_font('Arial', 'B', 12)
        # Title
        self.cell(0, 8, title, 0, 1, 'L')
        # Line break
        self.ln(3)

def create_pdf_from_markdown(md_file, pdf_file):
    # Read markdown file
    with open(md_file, 'r') as f:
        content = f.read()
    
    # Split into sections based on markdown headers
    sections = content.split('## ')
    
    # Initialize PDF
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title and author
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'QuantoniumOS: A Hybrid Computational Framework', 0, 1, 'C')
    pdf.cell(0, 10, 'for Quantum-Inspired Resonance Simulation', 0, 1, 'C')
    pdf.ln(5)
    
    pdf.set_font('Arial', 'I', 12)
    pdf.cell(0, 10, 'Luis Minier', 0, 1, 'C')
    pdf.cell(0, 5, 'USPTO Application No. 19/169,399', 0, 1, 'C')
    pdf.cell(0, 5, 'DOI: 10.5281/zenodo.15072877', 0, 1, 'C')
    pdf.ln(10)
    
    # Process the abstract (which doesn't have a ## prefix)
    abstract_parts = sections[0].split('# Abstract')
    if len(abstract_parts) > 1:
        pdf.chapter_title('Abstract')
        pdf.chapter_body(abstract_parts[1].strip())
    
    # Process sections
    for section in sections[1:]:
        # Split section into title and content
        parts = section.split('\n', 1)
        if len(parts) < 2:
            continue
        
        title, content = parts
        
        # Check if we need a page break
        if pdf.get_y() > 240:
            pdf.add_page()
        
        pdf.chapter_title(title)
        
        # Process subsections
        subsections = content.split('### ')
        
        # Process the main section content (before any subsections)
        main_content = subsections[0].strip()
        if main_content:
            pdf.chapter_body(main_content)
        
        # Process subsections
        for subsection in subsections[1:]:
            # Split subsection into title and content
            subparts = subsection.split('\n', 1)
            if len(subparts) < 2:
                continue
            
            subtitle, subcontent = subparts
            
            pdf.section_title(subtitle)
            pdf.chapter_body(subcontent.strip())
    
    # Save PDF
    pdf.output(pdf_file)

if __name__ == '__main__':
    create_pdf_from_markdown('QuantoniumOS_Technical_Paper.md', 'QuantoniumOS_Technical_Paper.pdf')
    print(f"PDF created successfully: {os.path.abspath('QuantoniumOS_Technical_Paper.pdf')}")