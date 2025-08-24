#!/usr/bin/env python3
"""
QuantoniumOS Virtual File System

Quantum-aware filesystem that manages:
- Quantum state files (.qstate)
- RFT transformation data (.rft)
- Cryptographic keys (.qkey)
- Quantum circuit definitions (.qcircuit)
- System configuration files
- Application data and user files
"""

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QuantumFile:
    """Quantum file metadata"""

    name: str
    path: str
    size: int
    created: float
    modified: float
    file_type: str  # 'qstate', 'rft', 'qkey', 'qcircuit', 'data', 'config'
    quantum_checksum: str
    permissions: str
    owner: str
    metadata: Dict[str, Any]


class QuantumFileSystem:
    """
    QuantoniumOS Virtual File System

    Provides quantum-aware file operations with:
    - Quantum state preservation
    - Cryptographic integrity
    - RFT transformation storage
    - Virtual directory structure
    """

    def __init__(self, root_path: str = None):
        self.root_path = (
            Path(root_path) if root_path else Path(__file__).parent / "qfs_root"
        )
        self.metadata_file = self.root_path / ".qfs_metadata.json"
        self.file_registry: Dict[str, QuantumFile] = {}

        self._initialize_filesystem()

    def _initialize_filesystem(self):
        """Initialize the quantum filesystem"""
        print("📁 Initializing QuantoniumOS File System...")

        # Create root directory structure
        self.root_path.mkdir(exist_ok=True)

        # Standard QuantoniumOS directories
        directories = [
            "home",
            "system",
            "quantum",
            "quantum/states",
            "quantum/circuits",
            "quantum/rft_data",
            "crypto",
            "crypto/keys",
            "crypto/certificates",
            "apps",
            "config",
            "tmp",
            "dev",
            "proc",
        ]

        for directory in directories:
            (self.root_path / directory).mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

        # Create default system files
        self._create_system_files()

        print(f"✅ QuantoniumOS File System initialized at {self.root_path}")
        print(f"📊 Files in registry: {len(self.file_registry)}")

    def _load_metadata(self):
        """Load file system metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for file_data in data.get("files", []):
                        qfile = QuantumFile(**file_data)
                        self.file_registry[qfile.path] = qfile
                print(f"📋 Loaded {len(self.file_registry)} files from metadata")
            except Exception as e:
                print(f"⚠️ Error loading metadata: {e}")

    def _save_metadata(self):
        """Save file system metadata"""
        try:
            metadata = {
                "filesystem_version": "1.0",
                "created": time.time(),
                "files": [asdict(qfile) for qfile in self.file_registry.values()],
            }

            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")

    def _create_system_files(self):
        """Create default system files"""
        system_files = [
            # System configuration
            (
                "system/quantonium.conf",
                "config",
                {
                    "version": "1.0",
                    "quantum_vertices": 1000,
                    "rft_enabled": True,
                    "crypto_enabled": True,
                    "debug_mode": False,
                },
            ),
            # Quantum system info
            (
                "quantum/system_info.qstate",
                "qstate",
                {
                    "total_vertices": 1000,
                    "grid_topology": "32x32",
                    "coherence_threshold": 0.95,
                    "evolution_rate": 0.01,
                },
            ),
            # RFT configuration
            (
                "quantum/rft_data/config.rft",
                "rft",
                {
                    "engine": "production_canonical_rft",
                    "N": 32,
                    "sigma": 2.0,
                    "phi": 1.618033988749895,
                },
            ),
            # Crypto configuration
            (
                "crypto/config.qkey",
                "qkey",
                {
                    "algorithm": "quantum_safe_rsa",
                    "key_size": 4096,
                    "quantum_resistant": True,
                },
            ),
            # Home directory welcome
            (
                "home/welcome.txt",
                "data",
                "Welcome to QuantoniumOS!\n\nThis is the world's first vertex-based quantum operating system.\n\nYour home directory is ready for quantum computing!",
            ),
        ]

        for file_path, file_type, content in system_files:
            if not self.exists(file_path):
                self.write_file(file_path, content, file_type=file_type, owner="system")

    def _compute_quantum_checksum(self, content: Any) -> str:
        """Compute quantum-aware checksum"""
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        elif isinstance(content, str):
            content_str = content
        else:
            content_str = str(content)

        # Use SHA-256 with quantum-safe salt
        quantum_salt = "QuantoniumOS_2025_QFS"
        return hashlib.sha256((content_str + quantum_salt).encode()).hexdigest()

    def write_file(
        self,
        path: str,
        content: Any,
        file_type: str = "data",
        owner: str = "user",
        permissions: str = "rw-r--r--",
    ) -> bool:
        """Write file to quantum filesystem"""
        try:
            full_path = self.root_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize content based on type
            if file_type in ["qstate", "rft", "qkey", "config"] and isinstance(
                content, dict
            ):
                content_to_write = json.dumps(content, indent=2)
            elif isinstance(content, str):
                content_to_write = content
            else:
                content_to_write = pickle.dumps(content)

            # Write to actual file
            mode = "w" if isinstance(content_to_write, str) else "wb"
            with open(full_path, mode) as f:
                f.write(content_to_write)

            # Create quantum file metadata
            file_size = full_path.stat().st_size
            current_time = time.time()

            qfile = QuantumFile(
                name=full_path.name,
                path=path,
                size=file_size,
                created=current_time,
                modified=current_time,
                file_type=file_type,
                quantum_checksum=self._compute_quantum_checksum(content),
                permissions=permissions,
                owner=owner,
                metadata={
                    "encoding": "utf-8"
                    if isinstance(content_to_write, str)
                    else "binary"
                },
            )

            self.file_registry[path] = qfile
            self._save_metadata()

            return True

        except Exception as e:
            print(f"❌ Error writing file {path}: {e}")
            return False

    def read_file(self, path: str) -> Optional[Any]:
        """Read file from quantum filesystem"""
        try:
            if path not in self.file_registry:
                return None

            qfile = self.file_registry[path]
            full_path = self.root_path / path

            if not full_path.exists():
                return None

            # Read based on file type and encoding
            if qfile.metadata.get("encoding") == "binary":
                with open(full_path, "rb") as f:
                    # Security fix: Use json for safer serialization instead of pickle
                    import json

                    content = json.loads(f.read().decode("utf-8"))
            else:
                with open(full_path, "r") as f:
                    content = f.read()

                # Parse JSON for structured file types
                if qfile.file_type in ["qstate", "rft", "qkey", "config"]:
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass  # Return as string if not valid JSON

            return content

        except Exception as e:
            print(f"❌ Error reading file {path}: {e}")
            return None

    def exists(self, path: str) -> bool:
        """Check if file exists"""
        return path in self.file_registry and (self.root_path / path).exists()

    def delete_file(self, path: str) -> bool:
        """Delete file from quantum filesystem"""
        try:
            if path not in self.file_registry:
                return False

            full_path = self.root_path / path
            if full_path.exists():
                full_path.unlink()

            del self.file_registry[path]
            self._save_metadata()

            return True

        except Exception as e:
            print(f"❌ Error deleting file {path}: {e}")
            return False

    def list_directory(self, path: str = "") -> List[QuantumFile]:
        """List files in directory"""
        dir_path = path.rstrip("/")
        files = []

        for file_path, qfile in self.file_registry.items():
            file_dir = str(Path(file_path).parent)
            if file_dir == dir_path or (dir_path == "" and "/" not in file_path):
                files.append(qfile)

        return sorted(files, key=lambda f: f.name)

    def get_file_info(self, path: str) -> Optional[QuantumFile]:
        """Get file information"""
        return self.file_registry.get(path)

    def verify_integrity(self, path: str) -> bool:
        """Verify quantum file integrity"""
        try:
            qfile = self.file_registry.get(path)
            if not qfile:
                return False

            content = self.read_file(path)
            if content is None:
                return False

            current_checksum = self._compute_quantum_checksum(content)
            return current_checksum == qfile.quantum_checksum

        except Exception as e:
            print(f"❌ Error verifying integrity for {path}: {e}")
            return False

    def get_filesystem_stats(self) -> Dict[str, Any]:
        """Get filesystem statistics"""
        total_files = len(self.file_registry)
        total_size = sum(qfile.size for qfile in self.file_registry.values())

        file_types = {}
        for qfile in self.file_registry.values():
            file_types[qfile.file_type] = file_types.get(qfile.file_type, 0) + 1

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "file_types": file_types,
            "root_path": str(self.root_path),
            "directories_created": len(
                [d for d in self.root_path.rglob("*") if d.is_dir()]
            ),
        }


def demonstrate_quantum_filesystem():
    """Demonstrate QuantoniumOS filesystem"""
    print("=" * 60)
    print("📁 QUANTONIUMOS FILESYSTEM DEMONSTRATION")
    print("🎯 Quantum-Aware File System")
    print("=" * 60)
    print()

    # Initialize filesystem
    qfs = QuantumFileSystem()
    print()

    # Create some quantum files
    print("📝 CREATING QUANTUM FILES:")

    # Quantum state file
    quantum_state = {
        "vertex_0": {"alpha": 0.707, "beta": 0.707},
        "vertex_1": {"alpha": 1.0, "beta": 0.0},
        "entanglement": {"vertices": [0, 1], "correlation": 0.95},
    }
    qfs.write_file(
        "quantum/states/entangled_pair.qstate", quantum_state, file_type="qstate"
    )
    print("   ✅ Created quantum state file")

    # RFT data file
    rft_data = {
        "transform_matrix": "32x32_production_rft",
        "frequency_response": [1.618, 2.618, 4.236],
        "distinctness": 0.803,
    }
    qfs.write_file("quantum/rft_data/experiment_001.rft", rft_data, file_type="rft")
    print("   ✅ Created RFT data file")

    # Quantum circuit
    circuit_def = {
        "name": "Bell State Generator",
        "qubits": 2,
        "gates": [
            {"type": "H", "target": 0},
            {"type": "CNOT", "control": 0, "target": 1},
        ],
    }
    qfs.write_file(
        "quantum/circuits/bell_state.qcircuit", circuit_def, file_type="qcircuit"
    )
    print("   ✅ Created quantum circuit file")

    # User data
    user_note = "My first quantum computation on QuantoniumOS!\nResults: Successfully generated Bell state with 99.7% fidelity."
    qfs.write_file("home/my_notes.txt", user_note)
    print("   ✅ Created user data file")
    print()

    # List directory contents
    print("📂 DIRECTORY LISTINGS:")
    directories = ["", "quantum", "quantum/states", "home", "system"]

    for directory in directories:
        files = qfs.list_directory(directory)
        print(f"\n   📁 /{directory}:")
        for qfile in files:
            print(f"      {qfile.name} ({qfile.file_type}) - {qfile.size} bytes")
    print()

    # Read and verify files
    print("🔍 FILE OPERATIONS:")

    # Read quantum state
    state_data = qfs.read_file("quantum/states/entangled_pair.qstate")
    print(f"   📖 Read quantum state: {len(state_data)} properties")

    # Verify integrity
    integrity_ok = qfs.verify_integrity("quantum/states/entangled_pair.qstate")
    print(f"   🛡️ Integrity check: {'✅ PASS' if integrity_ok else '❌ FAIL'}")

    # Get file info
    file_info = qfs.get_file_info("quantum/rft_data/experiment_001.rft")
    if file_info:
        print(
            f"   📋 File info: {file_info.name} created {time.ctime(file_info.created)}"
        )
    print()

    # Filesystem statistics
    print("📊 FILESYSTEM STATISTICS:")
    stats = qfs.get_filesystem_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    print()

    print("✅ QUANTONIUMOS FILESYSTEM DEMONSTRATION COMPLETE!")
    print("🎯 Quantum-aware file system operational!")

    return qfs


if __name__ == "__main__":
    qfs = demonstrate_quantum_filesystem()
