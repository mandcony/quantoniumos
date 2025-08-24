"""
Advanced code similarity analyzer for the QuantoniumOS project.
This script finds files that might contain similar code despite having different names.
"""

import ast
import difflib
import hashlib
import io
import os
import re
import tokenize
from collections import defaultdict
from pathlib import Path

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    ".venv",
    ".venv-1",
    "__pycache__",
    "node_modules",
    "third_party",
    "09_LEGACY_BACKUPS",
}

# Extensions to analyze
CODE_EXTENSIONS = {".py", ".cpp", ".h", ".hpp"}


def normalize_python_code(code):
    """
    Normalize Python code by removing comments, whitespace, and renaming variables.
    This helps identify code that's functionally similar even if variable names differ.
    """
    try:
        # Try to parse as Python code
        tree = ast.parse(code)

        # Collect variable and function names for normalization
        variable_map = {}
        function_map = {}

        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in variable_map:
                        variable_map[node.id] = f"var_{len(variable_map)}"
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                if node.name not in function_map:
                    function_map[node.name] = f"func_{len(function_map)}"
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                if node.name not in function_map:
                    function_map[node.name] = f"class_{len(function_map)}"
                self.generic_visit(node)

        NameVisitor().visit(tree)

        # Normalize the code using tokenize
        result = []
        tokens = list(tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline))
        for token in tokens:
            if token.type == tokenize.NAME:
                if token.string in variable_map:
                    result.append(variable_map[token.string])
                elif token.string in function_map:
                    result.append(function_map[token.string])
                else:
                    result.append(token.string)
            elif token.type not in (
                tokenize.COMMENT,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
            ):
                if (
                    token.type != tokenize.STRING
                ):  # Keep string literals but remove formatting
                    result.append(token.string)

        return "".join(result)
    except (SyntaxError, UnicodeDecodeError, tokenize.TokenError):
        # If we can't parse it as Python, return a simplified version
        # Remove comments, whitespace, and blank lines
        lines = []
        for line in code.split("\n"):
            line = re.sub(r"#.*$", "", line)  # Remove comments
            line = line.strip()
            if line:
                lines.append(line)
        return " ".join(lines)


def normalize_cpp_code(code):
    """
    Normalize C++ code by removing comments, whitespace, and standardizing some patterns.
    """
    # Remove C-style comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Remove C++-style comments
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

    # Remove preprocessor directives (simplification)
    code = re.sub(r"#\s*include.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#\s*define.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"#\s*if.*?#\s*endif", "", code, flags=re.DOTALL)

    # Normalize whitespace
    code = re.sub(r"\s+", " ", code)

    # Remove string literals (simplification)
    code = re.sub(r'"[^"]*"', '""', code)
    code = re.sub(r"'[^']*'", "''", code)

    return code.strip()


def calculate_code_similarity(code1, code2):
    """
    Calculate similarity between two code snippets.
    Returns a value between 0 and 1, where 1 means identical.
    """
    if not code1 or not code2:
        return 0

    return difflib.SequenceMatcher(None, code1, code2).ratio()


def get_file_content(file_path):
    """Read file content safely."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
            return hashlib.sha256(file_data).hexdigest()
    except (IOError, OSError):
        return None


def should_ignore(file_path):
    """Check if a file should be ignored."""
    parts = Path(file_path).parts
    return any(ignore_dir in parts for ignore_dir in IGNORE_DIRS)


def get_file_extension(file_path):
    """Get the file extension."""
    return os.path.splitext(file_path)[1].lower()


def analyze_code_similarity():
    """
    Analyze code similarity across the project.
    Returns groups of files with similar code.
    """
    print("Scanning for files to analyze...")
    file_paths = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for filename in files:
            file_path = os.path.join(root, filename)

            # Skip if we should ignore this file
            if should_ignore(file_path):
                continue

            # Check file extension
            ext = get_file_extension(file_path)
            if ext in CODE_EXTENSIONS:
                file_paths.append(file_path)

    print(f"Found {len(file_paths)} code files to analyze.")

    # Group by file size first (files with vastly different sizes are unlikely to be similar)
    size_groups = defaultdict(list)
    for file_path in file_paths:
        try:
            file_size = os.path.getsize(file_path)
            # Group files in 5KB size ranges
            size_group = file_size // 5000
            size_groups[size_group].append(file_path)
        except (IOError, OSError):
            continue

    # Now analyze files within each size group
    similarity_groups = []
    processed = set()

    for size_group, group_files in size_groups.items():
        for i, file1 in enumerate(group_files):
            if file1 in processed:
                continue

            # Get file content and normalize it
            content1 = get_file_content(file1)
            if not content1:
                continue

            ext1 = get_file_extension(file1)
            if ext1 == ".py":
                normalized1 = normalize_python_code(content1)
            elif ext1 in (".cpp", ".c", ".h", ".hpp"):
                normalized1 = normalize_cpp_code(content1)
            else:
                continue  # Skip unsupported file types

            similar_files = [(file1, 1.0)]  # File is 100% similar to itself

            # Compare with other files in the same size group
            for file2 in group_files[i + 1 :]:
                if file2 in processed:
                    continue

                # Get file content and normalize it
                content2 = get_file_content(file2)
                if not content2:
                    continue

                ext2 = get_file_extension(file2)
                if ext2 == ".py":
                    normalized2 = normalize_python_code(content2)
                elif ext2 in (".cpp", ".c", ".h", ".hpp"):
                    normalized2 = normalize_cpp_code(content2)
                else:
                    continue  # Skip unsupported file types

                # Check if contents are exactly the same
                if normalized1 == normalized2:
                    similar_files.append((file2, 1.0))
                    processed.add(file2)
                else:
                    # Calculate similarity
                    similarity = calculate_code_similarity(normalized1, normalized2)
                    if similarity > 0.8:  # High similarity threshold
                        similar_files.append((file2, similarity))
                        processed.add(file2)

            # If we found similar files, add them to a group
            if len(similar_files) > 1:
                similarity_groups.append(similar_files)

            processed.add(file1)

            # Print progress periodically
            if len(processed) % 50 == 0:
                print(
                    f"Analyzed {len(processed)}/{len(file_paths)} files. Found {len(similarity_groups)} similarity groups."
                )

    return similarity_groups


def generate_report(similarity_groups):
    """Generate a report of files with similar code."""
    report = "# Code Similarity Analysis Report\n\n"

    if not similarity_groups:
        report += "No similar code files found.\n"
        return report

    # Sort groups by size (number of files)
    similarity_groups.sort(key=lambda x: len(x), reverse=True)

    # Calculate total potential duplicates
    total_files = sum(len(group) for group in similarity_groups)
    total_groups = len(similarity_groups)

    report += f"## Summary\n\n"
    report += f"- Total similar file groups: {total_groups}\n"
    report += f"- Total files involved: {total_files}\n"
    report += f"- These files may contain duplicated logic or highly similar code.\n\n"

    # Add details for each group
    for i, group in enumerate(similarity_groups, 1):
        # Sort by similarity
        group.sort(key=lambda x: x[1], reverse=True)

        report += f"## Similar Code Group {i} ({len(group)} files)\n\n"

        # Primary file is the first in the group
        primary_file = os.path.relpath(group[0][0], PROJECT_ROOT)
        report += f"Primary file: `{primary_file}`\n\n"

        report += "Similar files:\n\n"
        for file_path, similarity in group[1:]:
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            report += f"- `{rel_path}` (Similarity: {similarity:.2%})\n"

        report += "\n"

    report += "## Recommendations\n\n"
    report += "1. Review each group of similar files to determine if they can be consolidated.\n"
    report += "2. Consider creating shared libraries or utility functions for common code patterns.\n"
    report += "3. For files with 100% similarity but different names, choose one canonical version.\n"
    report += (
        "4. Check if similar files serve different purposes despite code similarity.\n"
    )

    return report


def main():
    print("Starting code similarity analysis for QuantoniumOS...")

    similarity_groups = analyze_code_similarity()

    print(
        f"\nAnalysis complete. Found {len(similarity_groups)} groups of similar files."
    )

    report = generate_report(similarity_groups)
    report_path = os.path.join(PROJECT_ROOT, "CODE_SIMILARITY_REPORT.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
