"""
Base build engine module for QuantoniumOS.
This module provides common functionality for building various engines.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


class EngineBuilder:
    """
    Base class for building QuantoniumOS engines.
    Provides common functionality for compiling and linking C++ engine code.
    """

    def __init__(self, engine_name, source_dir=None, include_dirs=None, libraries=None):
        """
        Initialize the engine builder.

        Args:
            engine_name (str): Name of the engine
            source_dir (str): Directory containing the source files
            include_dirs (list): List of include directories
            libraries (list): List of libraries to link against
        """
        self.engine_name = engine_name
        self.source_dir = source_dir or "core/cpp"
        self.include_dirs = include_dirs or ["core/include"]
        self.libraries = libraries or []
        self.is_windows = platform.system() == "Windows"
        self.is_debug = "--debug" in sys.argv

    def get_compiler(self):
        """
        Get the appropriate compiler for the current platform.
        """
        if self.is_windows:
            return "cl" if self._check_command("cl") else "g++"
        else:
            return "g++"

    def _check_command(self, cmd):
        """
        Check if a command is available.
        """
        try:
            subprocess.check_call(
                [cmd, "--version"] if not self.is_windows else ["where", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def build(self, source_files):
        """
        Build the engine from source files.

        Args:
            source_files (list): List of source files to compile
        """
        compiler = self.get_compiler()
        print(f"Building {self.engine_name} with {compiler}...")

        # Prepare include flags
        include_flags = []
        for include_dir in self.include_dirs:
            if compiler == "cl":
                include_flags.append(f"/I{include_dir}")
            else:
                include_flags.append(f"-I{include_dir}")

        # Prepare library flags
        library_flags = []
        for lib in self.libraries:
            if compiler == "cl":
                library_flags.append(f"{lib}.lib")
            else:
                library_flags.append(f"-l{lib}")

        # Build command
        output_name = f"{self.engine_name}_engine"
        if compiler == "cl":
            output_flag = f"/Fe:{output_name}.dll"
            debug_flags = ["/Z7", "/DEBUG"] if self.is_debug else []
            shared_flags = ["/LD"]
        else:
            output_flag = f"-o {output_name}.so"
            debug_flags = ["-g"] if self.is_debug else []
            shared_flags = ["-shared", "-fPIC"]

        # Combine all flags
        compile_command = [
            compiler,
            *source_files,
            *include_flags,
            *library_flags,
            *debug_flags,
            *shared_flags,
            output_flag,
        ]

        # Execute compilation
        try:
            print(" ".join(compile_command))
            subprocess.check_call(compile_command)
            print(f"Successfully built {self.engine_name}")
            return True
        except subprocess.SubprocessError as e:
            print(f"Error building {self.engine_name}: {e}")
            return False
