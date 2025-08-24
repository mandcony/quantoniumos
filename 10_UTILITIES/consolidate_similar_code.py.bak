"""
Direct code consolidation script for QuantoniumOS.
This script identifies key code duplication patterns and consolidates them.
"""

import hashlib
import os
import re
import shutil
from pathlib import Path

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Files to consolidate - based on CODE_SIMILARITY_REPORT.md findings
# Format: [(source_files_list, target_file, description)]
CONSOLIDATION_TARGETS = [
    # App launchers - similar code patterns
    (
        [
            "apps/launch_quantum_simulator.py",
            "apps/launch_q_mail.py",
            "apps/launch_q_notes.py",
            "apps/launch_q_vault.py",
            "apps/launch_rft_visualizer.py",
        ],
        "apps/launcher_base.py",
        "App launcher code with similar patterns",
    ),
    # Build utilities - similar build logic
    (
        [
            "build_crypto_engine.py",
            "10_UTILITIES/build_vertex_engine.py",
            "10_UTILITIES/build_resonance_engine.py",
        ],
        "10_UTILITIES/build_engine_base.py",
        "Build engine utilities with similar patterns",
    ),
    # Core files duplicated between main and 17_BUILD_ARTIFACTS
    (
        ["17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/entropy_qrng.py"],
        "core/encryption/entropy_qrng.py",
        "QRNG implementation",
    ),
    # Geometric container implementations
    (
        [
            "17_BUILD_ARTIFACTS/compiled/build/lib/core/python/utilities/geometric_container.py"
        ],
        "core/python/utilities/geometric_container.py",
        "Geometric container implementation",
    ),
    # Security implementations
    (
        ["17_BUILD_ARTIFACTS/compiled/build/lib/core/security/formal_derivations.py"],
        "core/security/formal_derivations.py",
        "Formal derivations implementation",
    ),
    # Quantum proofs
    (
        ["17_BUILD_ARTIFACTS/compiled/build/lib/core/security/quantum_proofs.py"],
        "core/security/quantum_proofs.py",
        "Quantum proofs implementation",
    ),
    # App implementations
    (["03_RUNNING_SYSTEMS/app.py"], "app.py", "Main application implementation"),
    # Crypto setup utilities
    (
        ["10_UTILITIES/setup_fixed_crypto.py"],
        "10_UTILITIES/setup_crypto.py",
        "Crypto setup utilities",
    ),
    # Launchers
    (
        [
            "15_DEPLOYMENT/installers/launch_pyqt5.py",
            "15_DEPLOYMENT/launchers/start_quantoniumos.py",
        ],
        "15_DEPLOYMENT/launchers/launch_quantoniumos.py",
        "QuantoniumOS launcher",
    ),
    # Encryption test implementations
    (
        [
            "17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/detailed_avalanche_test.py"
        ],
        "core/encryption/detailed_avalanche_test.py",
        "Avalanche test implementation",
    ),
    # Other encryption implementations
    (
        [
            "17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/minimal_resonance_encrypt.py"
        ],
        "core/encryption/minimal_resonance_encrypt.py",
        "Minimal resonance encryption",
    ),
    (
        [
            "17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/optimized_resonance_encrypt.py"
        ],
        "core/encryption/optimized_resonance_encrypt.py",
        "Optimized resonance encryption",
    ),
    (
        [
            "17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/simple_resonance_encrypt.py"
        ],
        "core/encryption/simple_resonance_encrypt.py",
        "Simple resonance encryption",
    ),
    (
        ["17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/test_diffusion.py"],
        "core/encryption/test_diffusion.py",
        "Diffusion test",
    ),
    (
        ["17_BUILD_ARTIFACTS/compiled/build/lib/core/encryption/wave_primitives.py"],
        "core/encryption/wave_primitives.py",
        "Wave primitives",
    ),
]


def ensure_directory(file_path):
    """Ensure the directory for a file exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
            return hashlib.sha256(file_data).hexdigest()
    except (IOError, OSError):
        return None


def get_file_content(file_path):
    """Read file content safely."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def merge_app_launchers(source_files, target_file):
    """Create a base launcher module that can be used by multiple app launchers."""
    # Ensure the target directory exists
    ensure_directory(target_file)

    # Analyze the common patterns in launcher files
    common_imports = set()
    common_functions = []
    common_patterns = []

    # Extract common elements from all files
    for file_path in source_files:
        abs_path = os.path.join(PROJECT_ROOT, file_path)
        content = get_file_content(abs_path)
        if not content:
            continue

        # Extract imports
        import_lines = re.findall(
            r"^import .*$|^from .* import .*$", content, re.MULTILINE
        )
        for imp in import_lines:
            common_imports.add(imp)

        # Find function definitions
        function_matches = re.findall(
            r"def ([a-zA-Z0-9_]+)\(.*?\):", content, re.DOTALL
        )
        for func in function_matches:
            if func not in common_functions:
                common_functions.append(func)

    # Create the base launcher module
    with open(os.path.join(PROJECT_ROOT, target_file), "w", encoding="utf-8") as f:
        f.write('"""\nBase launcher module for QuantoniumOS applications.\n')
        f.write(
            'This module provides common functionality used by various application launchers.\n"""\n\n'
        )

        # Write imports
        for imp in sorted(common_imports):
            f.write(f"{imp}\n")

        f.write("\n\n")

        # Write base launcher class
        f.write("class AppLauncherBase:\n")
        f.write('    """\n    Base class for QuantoniumOS application launchers.\n')
        f.write(
            "    Provides common functionality for launching and managing applications.\n"
        )
        f.write('    """\n\n')

        f.write('    def __init__(self, app_name, app_version="1.0.0"):\n')
        f.write('        """\n        Initialize the application launcher.\n\n')
        f.write("        Args:\n")
        f.write("            app_name (str): Name of the application\n")
        f.write("            app_version (str): Version of the application\n")
        f.write('        """\n')
        f.write("        self.app_name = app_name\n")
        f.write("        self.app_version = app_version\n")
        f.write("        self.config = {}\n")
        f.write("        self.initialized = False\n\n")

        f.write("    def initialize(self):\n")
        f.write(
            '        """\n        Initialize the application environment.\n        """\n'
        )
        f.write('        print(f"Initializing {self.app_name} v{self.app_version}")\n')
        f.write("        self.initialized = True\n")
        f.write("        return True\n\n")

        f.write("    def launch(self):\n")
        f.write('        """\n        Launch the application.\n        """\n')
        f.write("        if not self.initialized:\n")
        f.write("            success = self.initialize()\n")
        f.write("            if not success:\n")
        f.write('                print(f"Failed to initialize {self.app_name}")\n')
        f.write("                return False\n\n")
        f.write('        print(f"Launching {self.app_name}")\n')
        f.write("        return True\n\n")

        f.write("    def cleanup(self):\n")
        f.write(
            '        """\n        Clean up resources when closing the application.\n        """\n'
        )
        f.write('        print(f"Cleaning up {self.app_name} resources")\n')
        f.write("        return True\n\n")

    print(f"Created base launcher module: {target_file}")

    # Now update each individual launcher to use the base class
    for file_path in source_files:
        abs_path = os.path.join(PROJECT_ROOT, file_path)
        content = get_file_content(abs_path)
        if not content:
            continue

        # Extract the app name from the filename
        app_name = (
            os.path.splitext(os.path.basename(file_path))[0]
            .replace("launch_", "")
            .replace("_", " ")
            .title()
        )

        # Create a new version of the launcher using the base class
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write('"""\n')
            f.write(f"Launcher for the {app_name} application in QuantoniumOS.\n")
            f.write('"""\n\n')

            # Import the base launcher
            rel_path = os.path.relpath(
                os.path.dirname(target_file), os.path.dirname(abs_path)
            )
            module_path = (
                rel_path.replace("\\", "/")
                + "/"
                + os.path.splitext(os.path.basename(target_file))[0]
            )
            if module_path.startswith("./"):
                module_path = module_path[2:]
            elif module_path == ".":
                module_path = os.path.splitext(os.path.basename(target_file))[0]

            f.write(f"from {module_path} import AppLauncherBase\n")

            # Add other specific imports that might be needed
            if "quantum" in file_path or "simulator" in file_path:
                f.write("import numpy as np\n")
            if "crypto" in file_path:
                f.write("from core.encryption import simple_resonance_encrypt\n")
            if "visualizer" in file_path:
                f.write("import matplotlib.pyplot as plt\n")

            f.write("\n\n")

            # Create the specific launcher class
            class_name = "".join(word.capitalize() for word in app_name.split())
            f.write(f"class {class_name}Launcher(AppLauncherBase):\n")
            f.write('    """\n')
            f.write(f"    Launcher for the {app_name} application.\n")
            f.write('    """\n\n')

            f.write("    def __init__(self):\n")
            f.write(f'        super().__init__("{app_name}", "1.0.0")\n\n')

            # Add application-specific methods
            f.write("    def initialize(self):\n")
            f.write('        """\n')
            f.write(f"        Initialize the {app_name} application.\n")
            f.write('        """\n')
            f.write("        super().initialize()\n")
            f.write(f'        print("Setting up {app_name}-specific environment")\n')
            f.write("        return True\n\n")

            f.write("    def launch(self):\n")
            f.write('        """\n')
            f.write(f"        Launch the {app_name} application.\n")
            f.write('        """\n')
            f.write("        if not super().launch():\n")
            f.write("            return False\n")
            f.write(f'        print("Running {app_name} application logic")\n')
            f.write("        return True\n\n")

            # Main execution
            f.write("\n\ndef main():\n")
            f.write(f"    launcher = {class_name}Launcher()\n")
            f.write("    launcher.launch()\n")
            f.write("    launcher.cleanup()\n\n")

            f.write('if __name__ == "__main__":\n')
            f.write("    main()\n")

        print(f"Updated launcher: {file_path}")


def merge_build_utilities(source_files, target_file):
    """Create a base build utility that can be used by different engine builders."""
    # Ensure the target directory exists
    ensure_directory(target_file)

    # Create the base build engine module
    with open(os.path.join(PROJECT_ROOT, target_file), "w", encoding="utf-8") as f:
        f.write('"""\nBase build engine module for QuantoniumOS.\n')
        f.write(
            'This module provides common functionality for building various engines.\n"""\n\n'
        )

        # Common imports
        f.write("import os\n")
        f.write("import sys\n")
        f.write("import subprocess\n")
        f.write("import platform\n")
        f.write("from pathlib import Path\n\n")

        # Base builder class
        f.write("class EngineBuilder:\n")
        f.write('    """\n    Base class for building QuantoniumOS engines.\n')
        f.write(
            "    Provides common functionality for compiling and linking C++ engine code.\n"
        )
        f.write('    """\n\n')

        f.write(
            "    def __init__(self, engine_name, source_dir=None, include_dirs=None, libraries=None):\n"
        )
        f.write('        """\n        Initialize the engine builder.\n\n')
        f.write("        Args:\n")
        f.write("            engine_name (str): Name of the engine\n")
        f.write("            source_dir (str): Directory containing the source files\n")
        f.write("            include_dirs (list): List of include directories\n")
        f.write("            libraries (list): List of libraries to link against\n")
        f.write('        """\n')
        f.write("        self.engine_name = engine_name\n")
        f.write('        self.source_dir = source_dir or "core/cpp"\n')
        f.write('        self.include_dirs = include_dirs or ["core/include"]\n')
        f.write("        self.libraries = libraries or []\n")
        f.write('        self.is_windows = platform.system() == "Windows"\n')
        f.write('        self.is_debug = "--debug" in sys.argv\n\n')

        f.write("    def get_compiler(self):\n")
        f.write(
            '        """\n        Get the appropriate compiler for the current platform.\n        """\n'
        )
        f.write("        if self.is_windows:\n")
        f.write('            return "cl" if self._check_command("cl") else "g++"\n')
        f.write("        else:\n")
        f.write('            return "g++"\n\n')

        f.write("    def _check_command(self, cmd):\n")
        f.write('        """\n        Check if a command is available.\n        """\n')
        f.write("        try:\n")
        f.write("            subprocess.check_call(\n")
        f.write(
            '                [cmd, "--version"] if not self.is_windows else ["where", cmd],\n'
        )
        f.write("                stdout=subprocess.DEVNULL,\n")
        f.write("                stderr=subprocess.DEVNULL\n")
        f.write("            )\n")
        f.write("            return True\n")
        f.write("        except (subprocess.SubprocessError, FileNotFoundError):\n")
        f.write("            return False\n\n")

        f.write("    def build(self, source_files):\n")
        f.write('        """\n        Build the engine from source files.\n\n')
        f.write("        Args:\n")
        f.write("            source_files (list): List of source files to compile\n")
        f.write('        """\n')
        f.write("        compiler = self.get_compiler()\n")
        f.write('        print(f"Building {self.engine_name} with {compiler}...")\n\n')

        f.write("        # Prepare include flags\n")
        f.write("        include_flags = []\n")
        f.write("        for include_dir in self.include_dirs:\n")
        f.write('            if compiler == "cl":\n')
        f.write('                include_flags.append(f"/I{include_dir}")\n')
        f.write("            else:\n")
        f.write('                include_flags.append(f"-I{include_dir}")\n\n')

        f.write("        # Prepare library flags\n")
        f.write("        library_flags = []\n")
        f.write("        for lib in self.libraries:\n")
        f.write('            if compiler == "cl":\n')
        f.write('                library_flags.append(f"{lib}.lib")\n')
        f.write("            else:\n")
        f.write('                library_flags.append(f"-l{lib}")\n\n')

        f.write("        # Build command\n")
        f.write('        output_name = f"{self.engine_name}_engine"\n')
        f.write('        if compiler == "cl":\n')
        f.write('            output_flag = f"/Fe:{output_name}.dll"\n')
        f.write(
            '            debug_flags = ["/Z7", "/DEBUG"] if self.is_debug else []\n'
        )
        f.write('            shared_flags = ["/LD"]\n')
        f.write("        else:\n")
        f.write('            output_flag = f"-o {output_name}.so"\n')
        f.write('            debug_flags = ["-g"] if self.is_debug else []\n')
        f.write('            shared_flags = ["-shared", "-fPIC"]\n\n')

        f.write("        # Combine all flags\n")
        f.write("        compile_command = [\n")
        f.write("            compiler,\n")
        f.write("            *source_files,\n")
        f.write("            *include_flags,\n")
        f.write("            *library_flags,\n")
        f.write("            *debug_flags,\n")
        f.write("            *shared_flags,\n")
        f.write("            output_flag\n")
        f.write("        ]\n\n")

        f.write("        # Execute compilation\n")
        f.write("        try:\n")
        f.write('            print(" ".join(compile_command))\n')
        f.write("            subprocess.check_call(compile_command)\n")
        f.write('            print(f"Successfully built {self.engine_name}")\n')
        f.write("            return True\n")
        f.write("        except subprocess.SubprocessError as e:\n")
        f.write('            print(f"Error building {self.engine_name}: {e}")\n')
        f.write("            return False\n\n")

    print(f"Created base build engine module: {target_file}")

    # Update each build utility to use the base class
    for file_path in source_files:
        abs_path = os.path.join(PROJECT_ROOT, file_path)
        if not os.path.exists(abs_path):
            continue

        # Extract engine name from the filename
        engine_name = (
            os.path.splitext(os.path.basename(file_path))[0]
            .replace("build_", "")
            .replace("_engine", "")
        )

        # Create a new version of the build utility using the base class
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write('"""\n')
            f.write(f"Build script for the {engine_name} engine in QuantoniumOS.\n")
            f.write('"""\n\n')

            # Import the base builder
            rel_path = os.path.relpath(
                os.path.dirname(target_file), os.path.dirname(abs_path)
            )
            module_path = (
                rel_path.replace("\\", "/")
                + "/"
                + os.path.splitext(os.path.basename(target_file))[0]
            )
            if module_path.startswith("./"):
                module_path = module_path[2:]
            elif module_path == ".":
                module_path = os.path.splitext(os.path.basename(target_file))[0]

            f.write(f"from {module_path} import EngineBuilder\n")
            f.write("import os\n")
            f.write("import sys\n\n")

            # Create the specific builder class
            class_name = "".join(word.capitalize() for word in engine_name.split("_"))
            f.write(f"class {class_name}EngineBuilder(EngineBuilder):\n")
            f.write('    """\n')
            f.write(f"    Builder for the {engine_name} engine.\n")
            f.write('    """\n\n')

            f.write("    def __init__(self):\n")
            if "crypto" in engine_name:
                f.write("        super().__init__(\n")
                f.write(f'            engine_name="{engine_name}",\n')
                f.write('            source_dir="core/cpp/cryptography",\n')
                f.write(
                    '            include_dirs=["core/include", "third_party/include"],\n'
                )
                f.write('            libraries=["crypto"]\n')
                f.write("        )\n\n")
            elif "rft" in engine_name or "resonance" in engine_name:
                f.write("        super().__init__(\n")
                f.write(f'            engine_name="{engine_name}",\n')
                f.write('            source_dir="core/cpp/engines",\n')
                f.write(
                    '            include_dirs=["core/include", "third_party/eigen"],\n'
                )
                f.write("            libraries=[]\n")
                f.write("        )\n\n")
            elif "vertex" in engine_name:
                f.write("        super().__init__(\n")
                f.write(f'            engine_name="{engine_name}",\n')
                f.write('            source_dir="core/cpp/engines",\n')
                f.write(
                    '            include_dirs=["core/include", "third_party/eigen"],\n'
                )
                f.write("            libraries=[]\n")
                f.write("        )\n\n")
            else:
                f.write(f'        super().__init__("{engine_name}")\n\n')

            # Add engine-specific methods if needed
            f.write("    def get_source_files(self):\n")
            f.write('        """\n')
            f.write(f"        Get the source files for the {engine_name} engine.\n")
            f.write('        """\n')

            if "crypto" in engine_name:
                f.write("        return [\n")
                f.write(
                    '            os.path.join(self.source_dir, "rft_crypto.cpp"),\n'
                )
                f.write(
                    '            os.path.join("core/cpp/bindings", "rft_crypto_bindings.cpp")\n'
                )
                f.write("        ]\n\n")
            elif "rft" in engine_name or "resonance" in engine_name:
                f.write("        return [\n")
                f.write(
                    '            os.path.join(self.source_dir, "true_rft_engine.cpp"),\n'
                )
                f.write(
                    '            os.path.join("core/cpp/bindings", "resonance_engine_bindings.cpp")\n'
                )
                f.write("        ]\n\n")
            elif "vertex" in engine_name:
                f.write("        return [\n")
                f.write(
                    '            os.path.join(self.source_dir, "vertex_engine.cpp"),\n'
                )
                f.write(
                    '            os.path.join("core/cpp/bindings", "vertex_engine_bindings.cpp")\n'
                )
                f.write("        ]\n\n")
            else:
                f.write("        return [\n")
                f.write(
                    f'            os.path.join(self.source_dir, "{engine_name}.cpp"),\n'
                )
                f.write(
                    f'            os.path.join("core/cpp/bindings", "{engine_name}_bindings.cpp")\n'
                )
                f.write("        ]\n\n")

            # Main execution
            f.write("\ndef main():\n")
            f.write(f"    builder = {class_name}EngineBuilder()\n")
            f.write("    source_files = builder.get_source_files()\n")
            f.write("    success = builder.build(source_files)\n")
            f.write("    sys.exit(0 if success else 1)\n\n")

            f.write('if __name__ == "__main__":\n')
            f.write("    main()\n")

        print(f"Updated build utility: {file_path}")


def copy_file(source, target):
    """Copy a file, creating directories as needed."""
    source_path = os.path.join(PROJECT_ROOT, source)
    target_path = os.path.join(PROJECT_ROOT, target)

    # Check if source exists
    if not os.path.exists(source_path):
        print(f"Source file does not exist: {source}")
        return False

    # Skip if source and target are the same file
    if os.path.normpath(source_path) == os.path.normpath(target_path):
        print(f"Source and target are the same file, skipping: {source}")
        return True

    # Ensure target directory exists
    ensure_directory(target_path)

    # Copy the file
    try:
        # Read the content first, then write to target to avoid file lock issues
        with open(source_path, "rb") as src_file:
            content = src_file.read()

        with open(target_path, "wb") as tgt_file:
            tgt_file.write(content)

        # Copy metadata where possible
        try:
            shutil.copystat(source_path, target_path)
        except:
            pass

        print(f"Copied: {source} -> {target}")
        return True
    except Exception as e:
        print(f"Error copying {source} to {target}: {e}")
        return False


def consolidate_files(source_files, target_file, description):
    """Consolidate a group of similar files."""
    # Normalize paths
    source_files = [os.path.normpath(f) for f in source_files]
    target_file = os.path.normpath(target_file)

    # Check if this is a special case requiring custom handling
    if "launcher" in target_file and all("launch_" in f for f in source_files):
        return merge_app_launchers(source_files, target_file)
    elif "build_engine_base" in target_file and all(
        "build_" in f for f in source_files
    ):
        return merge_build_utilities(source_files, target_file)

    # For regular files, find the most recent or most complete version and use it as the target
    most_recent_file = None
    most_recent_time = 0
    most_lines = 0

    for source in source_files:
        source_path = os.path.join(PROJECT_ROOT, source)
        if not os.path.exists(source_path):
            continue

        # Check modification time
        mtime = os.path.getmtime(source_path)

        # Check number of lines
        try:
            with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
                line_count = sum(1 for _ in f)
        except:
            line_count = 0

        # Update most recent/complete file
        if line_count > most_lines or (
            line_count == most_lines and mtime > most_recent_time
        ):
            most_recent_file = source
            most_recent_time = mtime
            most_lines = line_count

    # If we found a suitable source file, copy it to the target
    if most_recent_file:
        return copy_file(most_recent_file, target_file)
    else:
        print(f"No suitable source file found for {description}")
        return False


def generate_report(results):
    """Generate a report of the consolidation results."""
    report = "# Code Consolidation Report\n\n"

    report += "## Summary\n\n"
    total_groups = len(results)
    successful_groups = sum(1 for r in results if r["success"])
    total_files = sum(len(r["source_files"]) for r in results)

    report += f"- Total consolidation groups: {total_groups}\n"
    report += f"- Successfully consolidated groups: {successful_groups}\n"
    report += f"- Total files involved: {total_files}\n\n"

    report += "## Consolidation Details\n\n"

    for i, result in enumerate(results, 1):
        status = "✅ Success" if result["success"] else "❌ Failed"
        report += f"### Group {i}: {result['description']} ({status})\n\n"

        report += "Source files:\n\n"
        for source in result["source_files"]:
            report += f"- `{source}`\n"

        report += f"\nTarget file: `{result['target_file']}`\n\n"

        if not result["success"] and result["error"]:
            report += f"Error: {result['error']}\n\n"

    report += "## Next Steps\n\n"
    report += "1. Review the consolidated files to ensure they function correctly\n"
    report += "2. Test the system to verify that the consolidated files work properly\n"
    report += "3. Update import statements in other files if necessary\n"
    report += (
        "4. Consider removing the original source files once everything is verified\n"
    )

    return report


def main():
    print("Starting code consolidation...")
    results = []

    for source_files, target_file, description in CONSOLIDATION_TARGETS:
        print(f"\nConsolidating: {description}")
        try:
            success = consolidate_files(source_files, target_file, description)
            results.append(
                {
                    "source_files": source_files,
                    "target_file": target_file,
                    "description": description,
                    "success": success,
                    "error": None,
                }
            )
        except Exception as e:
            print(f"Error: {e}")
            results.append(
                {
                    "source_files": source_files,
                    "target_file": target_file,
                    "description": description,
                    "success": False,
                    "error": str(e),
                }
            )

    # Generate report
    report = generate_report(results)
    report_path = os.path.join(PROJECT_ROOT, "CODE_CONSOLIDATION_REPORT.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nConsolidation complete. Report saved to {report_path}")

    # Summary
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Successfully consolidated {successful}/{total} groups")


if __name__ == "__main__":
    main()
