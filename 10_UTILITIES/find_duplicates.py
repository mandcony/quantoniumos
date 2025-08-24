import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path


def get_file_hash(file_path):
    """Computes the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, OSError):
        return None


def get_file_size(file_path):
    """Get the size of a file in bytes."""
    try:
        return os.path.getsize(file_path)
    except (IOError, OSError):
        return 0


def find_duplicate_files(root_folder):
    """Finds duplicate files in a directory and its subdirectories."""
    file_hashes = defaultdict(list)
    total_files = 0

    print("Scanning for duplicates...")
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            total_files += 1
            file_path = os.path.join(dirpath, filename)
            file_hash = get_file_hash(file_path)
            if file_hash:
                file_hashes[file_hash].append(file_path)

            # Progress indicator
            if total_files % 100 == 0:
                sys.stdout.write(f"\rScanned {total_files} files...")
                sys.stdout.flush()

    print(f"\rCompleted scan of {total_files} files.")
    return {
        hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1
    }


def categorize_duplicates(duplicates):
    """Categorize duplicates into project files, virtual env files, etc."""
    categories = {
        "project_code": defaultdict(list),  # Python, C++, etc.
        "documentation": defaultdict(list),  # Markdown, docs
        "virtual_env": defaultdict(list),  # Virtual environment files
        "build_artifacts": defaultdict(list),  # Build outputs
        "legacy_backups": defaultdict(list),  # Backup/archive files
        "misc": defaultdict(list),  # Other files
    }

    # File extensions for categorization
    code_extensions = {
        ".py",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".pyd",
    }
    doc_extensions = {".md", ".txt", ".rst", ".pdf", ".docx"}
    build_artifacts_dirs = {"build", "dist", "17_BUILD_ARTIFACTS"}
    legacy_dirs = {"09_LEGACY_BACKUPS", "archive", "_attic"}
    venv_dirs = {".venv", ".venv-1", "env", "venv"}

    for hash_val, files in duplicates.items():
        for file_path in files:
            path = Path(file_path)
            ext = path.suffix.lower()
            path_parts = path.parts

            # Check for virtual environment files
            if any(venv_dir in path_parts for venv_dir in venv_dirs):
                categories["virtual_env"][hash_val].append(file_path)
            # Check for legacy/backup files
            elif any(legacy_dir in str(path) for legacy_dir in legacy_dirs):
                categories["legacy_backups"][hash_val].append(file_path)
            # Check for build artifacts
            elif any(build_dir in str(path) for build_dir in build_artifacts_dirs):
                categories["build_artifacts"][hash_val].append(file_path)
            # Check for documentation
            elif ext in doc_extensions:
                categories["documentation"][hash_val].append(file_path)
            # Check for code files
            elif ext in code_extensions:
                categories["project_code"][hash_val].append(file_path)
            # Everything else
            else:
                categories["misc"][hash_val].append(file_path)

    return categories


def analyze_wasted_space(duplicates):
    """Calculate wasted space due to duplicates."""
    total_wasted_bytes = 0
    duplicate_sizes = {}

    for hash_val, files in duplicates.items():
        if files:
            # Get size of the first file (all duplicates have the same size)
            file_size = get_file_size(files[0])
            # Wasted space is the size multiplied by the number of redundant copies
            wasted_bytes = file_size * (len(files) - 1)
            total_wasted_bytes += wasted_bytes
            duplicate_sizes[hash_val] = file_size

    return total_wasted_bytes, duplicate_sizes


def generate_report(duplicate_files, report_file):
    """Generates a markdown report of duplicate files with categorization and analysis."""
    categories = categorize_duplicates(duplicate_files)
    total_wasted_bytes, duplicate_sizes = analyze_wasted_space(duplicate_files)

    with open(report_file, "w") as f:
        f.write("# Enhanced Duplicate File Report\n\n")

        # Summary
        f.write("## Summary\n\n")
        total_duplicate_sets = len(duplicate_files)
        total_duplicate_files = sum(len(files) for files in duplicate_files.values())
        f.write(f"- **Total duplicate sets**: {total_duplicate_sets}\n")
        f.write(f"- **Total duplicate files**: {total_duplicate_files}\n")
        f.write(f"- **Wasted disk space**: {format_bytes(total_wasted_bytes)}\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(
            "1. **Project Code Duplicates**: Review and consolidate duplicated Python and C++ files.\n"
        )
        f.write(
            "2. **Documentation Duplicates**: Consider maintaining documentation in a single location with references.\n"
        )
        f.write(
            "3. **Build Artifacts**: Clean up duplicated build artifacts that can be regenerated.\n"
        )
        f.write("4. **Legacy Backups**: Archive or remove redundant backup files.\n")
        f.write(
            "5. **Virtual Environment**: Consider rebuilding virtual environments rather than duplicating them.\n\n"
        )

        # Category breakdown
        f.write("## Duplicate Categories\n\n")
        f.write("| Category | Duplicate Sets | Duplicate Files | Wasted Space |\n")
        f.write("| -------- | -------------- | --------------- | ------------ |\n")

        for category_name, category_hashes in categories.items():
            if category_hashes:
                category_files = sum(len(files) for files in category_hashes.values())
                category_wasted = sum(
                    duplicate_sizes.get(hash_val, 0) * (len(files) - 1)
                    for hash_val, files in category_hashes.items()
                )
                f.write(
                    f"| {category_name.replace('_', ' ').title()} | {len(category_hashes)} | {category_files} | {format_bytes(category_wasted)} |\n"
                )

        f.write("\n")

        # Detailed breakdown by category
        for category_name, category_hashes in categories.items():
            if category_hashes:
                f.write(f"## {category_name.replace('_', ' ').title()} Duplicates\n\n")

                # Sort by wasted space (largest first)
                sorted_hashes = sorted(
                    category_hashes.items(),
                    key=lambda x: duplicate_sizes.get(x[0], 0) * len(x[1]),
                    reverse=True,
                )

                for i, (hash_val, files) in enumerate(sorted_hashes, 1):
                    file_size = duplicate_sizes.get(hash_val, 0)
                    wasted_space = file_size * (len(files) - 1)

                    f.write(f"### Set {i} - Hash: `{hash_val}`\n")
                    f.write(f"- File size: {format_bytes(file_size)}\n")
                    f.write(f"- Duplicates: {len(files)}\n")
                    f.write(f"- Wasted space: {format_bytes(wasted_space)}\n\n")

                    for file_path in files:
                        f.write(f"- `{file_path}`\n")
                    f.write("\n")

        # Original full listing
        f.write("## Complete Duplicate Listing\n\n")
        for i, (hash_val, files) in enumerate(duplicate_files.items(), 1):
            file_size = duplicate_sizes.get(hash_val, 0)
            f.write(
                f"### Duplicate Set {i} (Hash: `{hash_val}`, Size: {format_bytes(file_size)})\n"
            )
            for file_path in files:
                f.write(f"- `{file_path}`\n")
            f.write("\n")


def format_bytes(bytes_value):
    """Format bytes into a human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    duplicates = find_duplicate_files(project_root)
    report_filename = "enhanced_duplicate_report.md"
    generate_report(duplicates, report_filename)
    print(f"Enhanced duplicate file report generated: {report_filename}")
