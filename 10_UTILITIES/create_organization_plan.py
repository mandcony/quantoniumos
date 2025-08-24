"""
This script analyzes and organizes code with similar logic but different filenames.
It identifies functional duplication across the codebase and suggests consolidation points.
"""

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Files/folders to ignore during organization
IGNORE_PATHS = [
    # Virtual environments
    ".venv",
    ".venv-1",
    # Empty __init__.py files (expected to be similar)
    "__init__.py",
    # Build artifacts (compiled outputs)
    "17_BUILD_ARTIFACTS",
]

# Primary locations for different types of files
PRIMARY_LOCATIONS = {
    # Core logic files
    "core": "core",
    # RFT algorithm files
    "rft": "04_RFT_ALGORITHMS",
    # Cryptography files
    "crypto": "06_CRYPTOGRAPHY",
    # Quantum engine files
    "quantum": "05_QUANTUM_ENGINES",
    # Test files
    "test": "07_TESTS_BENCHMARKS",
    # Application files
    "app": "apps",
    # Utility files
    "util": "10_UTILITIES",
    # Web/UI files
    "ui": "11_QUANTONIUMOS/gui",
    # Debug tools
    "debug": "18_DEBUG_TOOLS",
}


def should_ignore(path):
    """Determine if a path should be ignored."""
    for ignore_path in IGNORE_PATHS:
        if ignore_path in path:
            return True
    return False


def categorize_file(file_path):
    """Categorize a file based on its content and name."""
    filename = os.path.basename(file_path)
    rel_path = os.path.relpath(file_path, PROJECT_ROOT)

    # Initialize with default category
    category = "misc"

    # Check filename patterns
    if any(kw in filename.lower() for kw in ["test", "benchmark"]):
        category = "test"
    elif any(kw in filename.lower() for kw in ["rft", "resonance", "fourier"]):
        category = "rft"
    elif any(
        kw in filename.lower() for kw in ["crypto", "encrypt", "decrypt", "cipher"]
    ):
        category = "crypto"
    elif any(kw in filename.lower() for kw in ["quantum", "qubit", "wave"]):
        category = "quantum"
    elif any(kw in filename.lower() for kw in ["app", "launch", "start"]):
        category = "app"
    elif any(kw in filename.lower() for kw in ["util", "helper", "common"]):
        category = "util"
    elif any(
        kw in filename.lower() for kw in ["ui", "gui", "web", "html", "css", "js"]
    ):
        category = "ui"
    elif any(kw in filename.lower() for kw in ["debug", "fix", "patch"]):
        category = "debug"
    elif "core" in rel_path:
        category = "core"

    return category


def parse_code_similarity_report():
    """Parse the code similarity report to extract groups of similar files."""
    report_path = os.path.join(PROJECT_ROOT, "CODE_SIMILARITY_REPORT.md")
    if not os.path.exists(report_path):
        print(f"Error: Code similarity report not found at {report_path}")
        return []

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract all similarity groups
    similarity_groups = []
    group_sections = re.findall(
        r"## Similar Code Group \d+ \(\d+ files\)(.*?)(?=##|\Z)", content, re.DOTALL
    )

    for section in group_sections:
        # Extract primary file
        primary_match = re.search(r"Primary file: `(.+?)`", section)
        if not primary_match:
            continue

        primary_file = primary_match.group(1)

        # Extract similar files
        similar_files = []
        similar_section = re.search(
            r"Similar files:(.*?)(?=\n\n|\Z)", section, re.DOTALL
        )
        if similar_section:
            file_matches = re.findall(
                r"- `(.+?)` \(Similarity: (.+?)%\)", similar_section.group(1)
            )
            for file_path, similarity in file_matches:
                similar_files.append((file_path, float(similarity)))

        # Only include groups where files are not in ignore paths
        if not should_ignore(primary_file) and not all(
            should_ignore(f[0]) for f in similar_files
        ):
            # Create absolute paths
            primary_abs = os.path.join(PROJECT_ROOT, primary_file)
            similar_abs = [
                (os.path.join(PROJECT_ROOT, f[0]), f[1])
                for f in similar_files
                if not should_ignore(f[0])
            ]

            if similar_abs:  # Only add if there are non-ignored similar files
                similarity_groups.append((primary_abs, similar_abs))

    return similarity_groups


def get_best_location(files):
    """Determine the best location for a set of similar files."""
    # Count file categories
    category_counts = defaultdict(int)
    for file_path in files:
        category = categorize_file(file_path)
        category_counts[category] += 1

    # Get the most common category
    if category_counts:
        primary_category = max(category_counts.items(), key=lambda x: x[1])[0]
        return PRIMARY_LOCATIONS.get(primary_category, "misc")

    return "misc"


def suggest_organization_plan(similarity_groups):
    """Create a plan for organizing similar files."""
    organization_plan = []

    for primary_file, similar_files in similarity_groups:
        # Get all files in this group
        all_files = [primary_file] + [f[0] for f in similar_files]

        # Determine best location
        best_location = get_best_location(all_files)

        # Extract filenames without path
        file_basenames = [os.path.basename(f) for f in all_files]

        # Suggest a common name (use the shortest non-empty name)
        valid_names = [name for name in file_basenames if name.strip()]
        if valid_names:
            common_name = min(valid_names, key=len)
        else:
            common_name = "consolidated_file.py"  # Default name

        # Build target path
        target_dir = os.path.join(PROJECT_ROOT, best_location)
        target_path = os.path.join(target_dir, common_name)

        # Add to organization plan
        organization_plan.append(
            {"files": all_files, "target_path": target_path, "category": best_location}
        )

    return organization_plan


def generate_organization_report(organization_plan):
    """Generate a report detailing the organization plan."""
    report = "# Code Organization Plan\n\n"

    report += "## Overview\n\n"
    report += f"This report outlines a plan to consolidate {len(organization_plan)} groups of similar files.\n\n"

    # Categorize by target directory
    by_category = defaultdict(list)
    for plan in organization_plan:
        by_category[plan["category"]].append(plan)

    report += "## Summary by Category\n\n"
    report += "| Category | Groups to Consolidate | Total Files |\n"
    report += "| -------- | --------------------- | ----------- |\n"

    for category, plans in sorted(by_category.items()):
        total_files = sum(len(plan["files"]) for plan in plans)
        report += f"| {category} | {len(plans)} | {total_files} |\n"

    report += "\n## Detailed Consolidation Plan\n\n"

    for i, plan in enumerate(organization_plan, 1):
        target_rel = os.path.relpath(plan["target_path"], PROJECT_ROOT)
        report += f"### Group {i}: Consolidate to `{target_rel}`\n\n"

        report += "Files to consolidate:\n\n"
        for file_path in plan["files"]:
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            report += f"- `{rel_path}`\n"

        report += "\n"

    report += "## Implementation Strategy\n\n"
    report += "For each group:\n\n"
    report += "1. **Compare files** to understand functional differences (if any)\n"
    report += "2. **Merge code** preserving unique functionality from each file\n"
    report += "3. **Create consolidated file** in the target location\n"
    report += "4. **Add imports/references** to ensure backward compatibility\n"
    report += "5. **Test thoroughly** before removing original files\n\n"

    report += "## Notes\n\n"
    report += "- Empty `__init__.py` files were excluded from consolidation\n"
    report += "- Build artifacts and virtual environment files were excluded\n"
    report += "- Files with >80% similarity were considered for consolidation\n"

    return report


def main():
    print("Analyzing code similarity report and creating organization plan...")

    # Parse the similarity report
    similarity_groups = parse_code_similarity_report()
    print(
        f"Found {len(similarity_groups)} groups of similar files (excluding ignored paths)"
    )

    # Create organization plan
    organization_plan = suggest_organization_plan(similarity_groups)
    print(
        f"Created organization plan with {len(organization_plan)} consolidation targets"
    )

    # Generate report
    report = generate_organization_report(organization_plan)
    report_path = os.path.join(PROJECT_ROOT, "CODE_ORGANIZATION_PLAN.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Organization plan saved to {report_path}")


if __name__ == "__main__":
    main()
