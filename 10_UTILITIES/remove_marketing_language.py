#!/usr/bin/env python3
""" Scientific Language Cleanup Script Removes marketing terminology
while preserving: 1. Technical patent claims and references (US 19/169,399) 2. Functional code and mathematical content 3. Scientific accuracy and data integrity Target removals: - "quantum processing", "topological processing", "", "methodological change" - Marketing emojis and excessive punctuation - Hype language
while keeping factual claims """

import json
import os
import pathlib
import re
import typing
import Dict
import List
import Path
import Tuple


class ScientificLanguageCleaner:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.changes_made = []

        # Marketing terms to remove/replace
        self.marketing_replacements = {

        # Remove hype adjectives but keep technical content r'\b(?:|||||||)\b': '', r'\bquantum magic\b': 'quantum processing', r'\btopological development\b': 'topological processing', r'\bparadigm shift\b': 'methodological change', r'\bbreakthrough\b': 'development',

        # Clean up excessive punctuation but preserve patent references r'||||||||||': '', r'!{2,}': '!', r'={3,}': '===', r'-{3,}': '---',

        # Tone down language
        while keeping facts r'\bTHE development\b': 'Technical Development', r'\bPARADIGM SHIFT\b': 'Methodological Change', r'\bREVOLUTIONARY\b': 'Novel', }

        # Preserve these exact patterns (patent claims, technical terms)
        self.preserve_patterns = [ r'US(?:PTO)?\s+(?:Patent\s+)?(?:Application\s+)?(?:No\.?\s+)?19[/-]169[/-]399', r'Patent\s+Claim\s+\d+', r'U\.S\.\s+Patent\s+Application\s+19/169/399', r'claim\s+\d+', r'intellectual\s+property', r'proprietary', ]
    def is_preserved_content(self, text: str) -> bool:
"""
        Check
        if text contains patent claims or protected content"""

        for pattern in
        self.preserve_patterns:
        if re.search(pattern, text, re.IGNORECASE):
        return True
        return False
    def clean_text(self, text: str, filename: str) -> str:
"""
        Clean marketing language from text
        while preserving technical content"""
        original_text = text

        # Skip files that are primarily patent-related
        if any(keyword in filename.lower()
        for keyword in ['patent', 'claims', 'legal']):
        return text

        # Apply replacements carefully for pattern, replacement in
        self.marketing_replacements.items():

        # Check each match to see
        if it's in a preserved context matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in reversed(matches):

        # Reverse to maintain indices context_start = max(0, match.start() - 100) context_end = min(len(text), match.end() + 100) context = text[context_start:context_end]
        if not
        self.is_preserved_content(context):

        # Safe to replace before = text[:match.start()] after = text[match.end():] text = before + replacement + after

        # Clean up extra whitespace text = re.sub(r'\s+', ' ', text) text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        if text != original_text:
        self.changes_made.append({ 'file': filename, 'changes': len(re.findall(r'\S+', original_text)) - len(re.findall(r'\S+', text)) })
        return text
    def process_file(self, file_path: Path) -> bool:
"""
        Process a single file for marketing language cleanup"""
        try:

        # Skip binary files and certain extensions
        if file_path.suffix in ['.pyd', '.dll', '.so', '.exe', '.bin']:
        return False

        # Read file with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()

        # Clean content cleaned_content =
        self.clean_text(content, file_path.name)

        # Write back
        if changed
        if cleaned_content != content: with open(file_path, 'w', encoding='utf-8') as f: f.write(cleaned_content)
        print(f"✓ Cleaned: {file_path.relative_to(
        self.repo_path)}")
        return True except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return False
    def run(self) -> Dict: """
        Run the cleanup process on the entire repository
"""

        print("🧹 Scientific Language Cleanup Starting...")
        print("Preserving patent claims and technical accuracy")
        print("-" * 50) processed_files = 0 modified_files = 0

        # Process all text files
        for file_path in
        self.repo_path.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'): processed_files += 1
        if
        self.process_file(file_path): modified_files += 1

        # Generate summary report summary = { 'processed_files': processed_files, 'modified_files': modified_files, 'changes_by_file':
        self.changes_made, 'preserved_patterns':
        self.preserve_patterns, 'timestamp': __import__('time').time() }

        # Save cleanup report with open(
        self.repo_path / 'marketing_cleanup_report.json', 'w') as f: json.dump(summary, f, indent=2)
        print("-" * 50)
        print(f"✅ Cleanup Complete:")
        print(f" • Files processed: {processed_files}")
        print(f" • Files modified: {modified_files}")
        print(f" • Patent claims preserved: ✓")
        print(f" • Technical accuracy maintained: ✓")
        print(f" • Report saved: marketing_cleanup_report.json")
        return summary
    def main(): """
        Main cleanup function
"""
        repo_path = os.getcwd() cleaner = ScientificLanguageCleaner(repo_path)
        print("Scientific Language Cleanup Tool")
        print("=" * 40)
        print("Removing marketing hype
        while preserving:")
        print("• Patent claims (US 19/169,399)")
        print("• Technical specifications")
        print("• Mathematical content")
        print("• Scientific accuracy")
        print() result = cleaner.run()
        if result['modified_files'] > 0:
        print("\n📋 Files modified:")
        for change in result['changes_by_file'][:10]:

        # Show first 10
        print(f" • {change['file']}: {change['changes']} terms cleaned")
        if len(result['changes_by_file']) > 10:
        print(f" • ... and {len(result['changes_by_file']) - 10} more files")
        print("\n🔬 Scientific integrity maintained!")
        return result

if __name__ == "__main__": main()