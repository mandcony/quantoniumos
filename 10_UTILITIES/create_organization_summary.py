"""
Create a final organization summary of the QuantoniumOS project.
This script analyzes the current state of the project and generates a
comprehensive report after all the organization work that has been done.
"""

import datetime
import os
import re
from collections import Counter

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def count_files_by_extension():
    """Count the number of files by extension in the project."""
    extension_counts = Counter()
    total_files = 0

    for root, _, files in os.walk(PROJECT_ROOT):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue

        for file in files:
            total_files += 1
            _, ext = os.path.splitext(file)
            extension_counts[ext.lower() if ext else "(no extension)"] += 1

    return extension_counts, total_files


def get_directory_structure():
    """Get the structure of top-level directories with file counts."""
    dir_structure = {}

    for item in os.listdir(PROJECT_ROOT):
        item_path = os.path.join(PROJECT_ROOT, item)

        if os.path.isdir(item_path):
            # Count files in this directory recursively
            file_count = 0
            for root, _, files in os.walk(item_path):
                file_count += len(files)

            dir_structure[item] = file_count

    return dir_structure


def get_recent_files(days=7, limit=20):
    """Get recently modified files."""
    recent_files = []
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=days)

    for root, _, files in os.walk(PROJECT_ROOT):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            try:
                mtime = os.path.getmtime(file_path)
                mod_time = datetime.datetime.fromtimestamp(mtime)

                if mod_time > cutoff:
                    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                    recent_files.append((rel_path, mod_time))
            except:
                pass

    # Sort by modification time (newest first)
    recent_files.sort(key=lambda x: x[1], reverse=True)

    return recent_files[:limit]


def find_important_files():
    """Find important files in the project."""
    important_patterns = [
        r"app\.py$",
        r"launch.*\.py$",
        r".*main.*\.py$",
        r".*config.*\.py$",
        r"setup\.py$",
        r"README\.md$",
        r"requirements\.txt$",
        r"pyproject\.toml$",
        r"Dockerfile$",
        r".*\.cpp$",
        r".*\.h$",
        r"quantonium.*\.py$",
    ]

    important_files = []

    for root, _, files in os.walk(PROJECT_ROOT):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue

        for file in files:
            file_path = os.path.relpath(os.path.join(root, file), PROJECT_ROOT)

            for pattern in important_patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    important_files.append(file_path)
                    break

    return important_files


def analyze_consolidation_results():
    """Analyze the results of the code consolidation."""
    consolidation_report = os.path.join(PROJECT_ROOT, "CODE_CONSOLIDATION_REPORT.md")
    cleanup_summary = os.path.join(PROJECT_ROOT, "CLEANUP_SUMMARY.md")

    consolidated_files = []
    consolidated_groups = 0
    total_groups = 0

    # Read the consolidation report
    if os.path.exists(consolidation_report):
        with open(consolidation_report, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the number of successful consolidations
        success_match = re.search(
            r"Successfully consolidated (\d+)/(\d+) groups", content
        )
        if success_match:
            consolidated_groups = int(success_match.group(1))
            total_groups = int(success_match.group(2))

        # Extract consolidated files
        group_matches = re.finditer(
            r"### Group \d+:.+? \((.*?)\)\s+Source files:\s+(.+?)Target file: `(.+?)`",
            content,
            re.DOTALL,
        )

        for match in group_matches:
            status = match.group(1)
            target = match.group(3)

            if "✅" in status:
                consolidated_files.append(target)

    # Read the cleanup summary
    cleanup_stats = {}
    if os.path.exists(cleanup_summary):
        with open(cleanup_summary, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract statistics
        deleted_match = re.search(r"Deleted (\d+) files", content)
        if deleted_match:
            cleanup_stats["deleted"] = int(deleted_match.group(1))

        moved_match = re.search(r"Moved (\d+) files", content)
        if moved_match:
            cleanup_stats["moved"] = int(moved_match.group(1))

    return {
        "consolidated_groups": consolidated_groups,
        "total_groups": total_groups,
        "consolidated_files": consolidated_files,
        "cleanup_stats": cleanup_stats,
    }


def generate_report():
    """Generate a comprehensive report of the project organization."""
    ext_counts, total_files = count_files_by_extension()
    dir_structure = get_directory_structure()
    recent_files = get_recent_files()
    important_files = find_important_files()
    consolidation_results = analyze_consolidation_results()

    # Sort extensions by count (descending)
    sorted_extensions = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)

    # Sort directories by file count (descending)
    sorted_dirs = sorted(dir_structure.items(), key=lambda x: x[1], reverse=True)

    # Generate the report
    report = "# QuantoniumOS Project Organization Summary\n\n"
    report += (
        f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    )

    # Add project statistics
    report += "## Project Statistics\n\n"
    report += f"- **Total Files**: {total_files}\n"

    # Add cleanup statistics if available
    cleanup_stats = consolidation_results["cleanup_stats"]
    if cleanup_stats:
        if "deleted" in cleanup_stats:
            report += (
                f"- **Files Deleted During Cleanup**: {cleanup_stats['deleted']}\n"
            )
        if "moved" in cleanup_stats:
            report += (
                f"- **Files Moved During Organization**: {cleanup_stats['moved']}\n"
            )

    # Add consolidation statistics
    report += f"- **Code Groups Consolidated**: {consolidation_results['consolidated_groups']}/{consolidation_results['total_groups']}\n"
    report += "\n"

    # Add file types
    report += "## File Types\n\n"
    report += "| Extension | Count | Percentage |\n"
    report += "|-----------|-------|------------|\n"

    for ext, count in sorted_extensions[:15]:  # Show top 15
        percentage = (count / total_files) * 100
        report += f"| {ext} | {count} | {percentage:.1f}% |\n"

    if len(sorted_extensions) > 15:
        other_count = sum(count for _, count in sorted_extensions[15:])
        other_percentage = (other_count / total_files) * 100
        report += f"| Other | {other_count} | {other_percentage:.1f}% |\n"

    report += "\n"

    # Add directory structure
    report += "## Directory Structure\n\n"
    report += "| Directory | File Count |\n"
    report += "|-----------|------------|\n"

    for dir_name, count in sorted_dirs[:20]:  # Show top 20
        report += f"| {dir_name} | {count} |\n"

    report += "\n"

    # Add recent files
    report += "## Recently Modified Files\n\n"
    report += "| File | Last Modified |\n"
    report += "|------|---------------|\n"

    for file_path, mod_time in recent_files:
        report += f"| {file_path} | {mod_time.strftime('%Y-%m-%d %H:%M:%S')} |\n"

    report += "\n"

    # Add important files
    report += "## Key Project Files\n\n"

    # Group important files by category
    core_files = []
    app_files = []
    build_files = []
    config_files = []
    doc_files = []

    for file_path in important_files:
        if "app.py" in file_path or "launch" in file_path or "main" in file_path:
            app_files.append(file_path)
        elif ".cpp" in file_path or ".h" in file_path:
            build_files.append(file_path)
        elif (
            "config" in file_path
            or "setup" in file_path
            or "requirements" in file_path
            or "pyproject" in file_path
        ):
            config_files.append(file_path)
        elif "README" in file_path or ".md" in file_path:
            doc_files.append(file_path)
        else:
            core_files.append(file_path)

    # Add each category
    if app_files:
        report += "### Application Files\n\n"
        for file_path in sorted(app_files):
            report += f"- `{file_path}`\n"
        report += "\n"

    if core_files:
        report += "### Core System Files\n\n"
        for file_path in sorted(core_files):
            report += f"- `{file_path}`\n"
        report += "\n"

    if build_files:
        report += "### Build & Native Code Files\n\n"
        for file_path in sorted(build_files):
            report += f"- `{file_path}`\n"
        report += "\n"

    if config_files:
        report += "### Configuration Files\n\n"
        for file_path in sorted(config_files):
            report += f"- `{file_path}`\n"
        report += "\n"

    if doc_files:
        report += "### Documentation Files\n\n"
        for file_path in sorted(doc_files):
            report += f"- `{file_path}`\n"
        report += "\n"

    # Add consolidated files
    if consolidation_results["consolidated_files"]:
        report += "## Consolidated Files\n\n"
        for file_path in sorted(consolidation_results["consolidated_files"]):
            report += f"- `{file_path}`\n"
        report += "\n"

    # Add recommendations
    report += "## Recommendations\n\n"
    report += "1. **Standardize Project Structure**: Continue organizing files into the appropriate directories.\n"
    report += "2. **Improve Documentation**: Ensure each module has proper documentation and docstrings.\n"
    report += "3. **Consolidate Similar Code**: Look for opportunities to further reduce code duplication.\n"
    report += (
        "4. **Test Coverage**: Increase test coverage to ensure system reliability.\n"
    )
    report += "5. **Build Process**: Streamline the build process for native code components.\n"
    report += (
        "6. **Dependency Management**: Review and optimize external dependencies.\n"
    )

    return report


def main():
    print("Analyzing QuantoniumOS project organization...")
    report = generate_report()

    # Write the report to a file
    report_path = os.path.join(PROJECT_ROOT, "ORGANIZATION_SUMMARY.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Organization summary report generated: {report_path}")


if __name__ == "__main__":
    main()
