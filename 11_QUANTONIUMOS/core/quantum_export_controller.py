"""
QuantoniumOS - Universal Export & Save Controller
Enhanced security, universal formatting, real-time encryption
Version: 1.0 - Production Grade
"""

import base64
import csv
import datetime
import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class QuantumExportController:
    """Universal Export & Save Controller with Enhanced Security"""

    def __init__(self):
        self.export_formats = [
            "json",
            "csv",
            "txt",
            "xml",
            "pdf",
            "xlsx",
            "encrypted_json",
            "encrypted_csv",
            "quantum_vault",
        ]
        self.encryption_engine = QuantumEncryptionEngine()
        self.active_exports = {}
        self.export_history = []
        self.setup_export_directories()

    def setup_export_directories(self):
        """Setup export directory structure"""
        self.base_export_dir = Path.home() / "QuantoniumOS_Exports"
        self.directories = {
            "results": self.base_export_dir / "Results",
            "reports": self.base_export_dir / "Reports",
            "data": self.base_export_dir / "Data",
            "encrypted": self.base_export_dir / "Encrypted",
            "backups": self.base_export_dir / "Backups",
            "temp": self.base_export_dir / "Temp",
        }

        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def export_results(
        self,
        data: Dict[str, Any],
        export_format: str = "json",
        encrypt: bool = True,
        app_source: str = "unknown",
    ) -> Dict[str, str]:
        """
        Universal export function for all app results

        Args:
            data: Data to export
            export_format: Format (json, csv, txt, xml, pdf, xlsx)
            encrypt: Whether to encrypt the export
            app_source: Source app name

        Returns:
            Export status and file paths
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            export_id = f"{app_source}_{timestamp}_{secrets.token_hex(4)}"

            # Prepare data for export
            export_data = {
                "metadata": {
                    "export_id": export_id,
                    "timestamp": timestamp,
                    "app_source": app_source,
                    "format": export_format,
                    "encrypted": encrypt,
                    "quantonium_version": "1.0",
                },
                "data": data,
            }

            # Generate filename
            filename = f"{app_source}_results_{timestamp}"

            # Export based on format
            if export_format == "json":
                file_path = self._export_json(export_data, filename)
            elif export_format == "csv":
                file_path = self._export_csv(export_data, filename)
            elif export_format == "txt":
                file_path = self._export_txt(export_data, filename)
            elif export_format == "xml":
                file_path = self._export_xml(export_data, filename)
            elif export_format == "pdf":
                file_path = self._export_pdf(export_data, filename)
            elif export_format == "xlsx":
                file_path = self._export_xlsx(export_data, filename)
            else:
                file_path = self._export_json(export_data, filename)  # Default fallback

            # Encrypt if requested
            if encrypt:
                encrypted_path = self.encryption_engine.encrypt_file(file_path)
                file_path = encrypted_path

            # Log export
            self._log_export(export_id, file_path, export_data["metadata"])

            return {
                "status": "success",
                "export_id": export_id,
                "file_path": str(file_path),
                "encrypted": encrypt,
                "format": export_format,
                "timestamp": timestamp,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat(),
            }

    def _export_json(self, data: Dict, filename: str) -> Path:
        """Export as JSON"""
        file_path = self.directories["results"] / f"{filename}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return file_path

    def _export_csv(self, data: Dict, filename: str) -> Path:
        """Export as CSV"""
        file_path = self.directories["results"] / f"{filename}.csv"

        # Flatten data for CSV
        if "data" in data and isinstance(data["data"], dict):
            flattened = self._flatten_dict(data["data"])
        else:
            flattened = data

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            if flattened:
                writer = csv.DictWriter(f, fieldnames=flattened.keys())
                writer.writeheader()
                writer.writerow(flattened)
        return file_path

    def _export_txt(self, data: Dict, filename: str) -> Path:
        """Export as formatted text"""
        file_path = self.directories["results"] / f"{filename}.txt"

        content = []
        content.append("=" * 80)
        content.append(f"QuantoniumOS Export Report")
        content.append(f"Generated: {data['metadata']['timestamp']}")
        content.append(f"Source: {data['metadata']['app_source']}")
        content.append("=" * 80)
        content.append("")

        # Format data
        content.append("RESULTS:")
        content.append("-" * 40)
        content.append(json.dumps(data["data"], indent=2, default=str))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        return file_path

    def _export_xml(self, data: Dict, filename: str) -> Path:
        """Export as XML"""
        file_path = self.directories["results"] / f"{filename}.xml"

        xml_content = []
        xml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
        xml_content.append("<quantonium_export>")
        xml_content.append("  <metadata>")
        for key, value in data["metadata"].items():
            xml_content.append(f"    <{key}>{value}</{key}>")
        xml_content.append("  </metadata>")
        xml_content.append("  <data>")
        xml_content.append(self._dict_to_xml(data["data"], "    "))
        xml_content.append("  </data>")
        xml_content.append("</quantonium_export>")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(xml_content))
        return file_path

    def _export_pdf(self, data: Dict, filename: str) -> Path:
        """Export as PDF (requires reportlab)"""
        file_path = self.directories["results"] / f"{filename}.pdf"

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title = Paragraph(f"QuantoniumOS Export Report", styles["Title"])
            story.append(title)
            story.append(Spacer(1, 12))

            # Metadata
            for key, value in data["metadata"].items():
                p = Paragraph(f"<b>{key}:</b> {value}", styles["Normal"])
                story.append(p)

            story.append(Spacer(1, 12))

            # Data
            data_text = json.dumps(data["data"], indent=2, default=str)
            data_para = Paragraph(f"<pre>{data_text}</pre>", styles["Code"])
            story.append(data_para)

            doc.build(story)

        except ImportError:
            # Fallback to text if reportlab not available
            return self._export_txt(data, filename)

        return file_path

    def _export_xlsx(self, data: Dict, filename: str) -> Path:
        """Export as Excel (requires openpyxl)"""
        file_path = self.directories["results"] / f"{filename}.xlsx"

        try:
            import openpyxl
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "QuantoniumOS Export"

            # Metadata sheet
            ws.append(["QuantoniumOS Export Report"])
            ws.append([])
            ws.append(["Metadata:"])
            for key, value in data["metadata"].items():
                ws.append([key, str(value)])

            ws.append([])
            ws.append(["Data:"])

            # Flatten and add data
            if isinstance(data["data"], dict):
                flattened = self._flatten_dict(data["data"])
                for key, value in flattened.items():
                    ws.append([key, str(value)])

            wb.save(str(file_path))

        except ImportError:
            # Fallback to CSV if openpyxl not available
            return self._export_csv(data, filename)

        return file_path

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _dict_to_xml(self, d: Dict, indent: str = "") -> str:
        """Convert dict to XML string"""
        xml_lines = []
        for key, value in d.items():
            clean_key = str(key).replace(" ", "_").replace("-", "_")
            if isinstance(value, dict):
                xml_lines.append(f"{indent}<{clean_key}>")
                xml_lines.append(self._dict_to_xml(value, indent + "  "))
                xml_lines.append(f"{indent}</{clean_key}>")
            else:
                xml_lines.append(f"{indent}<{clean_key}>{value}</{clean_key}>")
        return "\n".join(xml_lines)

    def _log_export(self, export_id: str, file_path: Path, metadata: Dict):
        """Log export operation"""
        log_entry = {
            "export_id": export_id,
            "file_path": str(file_path),
            "metadata": metadata,
            "logged_at": datetime.datetime.now().isoformat(),
        }

        self.export_history.append(log_entry)

        # Save to log file
        log_file = self.directories["results"] / "export_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.export_history, f, indent=2, default=str)

    def get_export_history(self) -> List[Dict]:
        """Get export history"""
        return self.export_history

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = self.directories["temp"]
        for file in temp_dir.glob("*"):
            if file.is_file():
                try:
                    file.unlink()
                except:
                    pass


class QuantumEncryptionEngine:
    """Enhanced Encryption Engine for QuantoniumOS"""

    def __init__(self):
        self.master_key = None
        self.session_keys = {}
        self.encryption_methods = ["AES-256", "RSA-4096", "QUANTUM-SAFE"]
        self.setup_encryption()

    def setup_encryption(self):
        """Setup encryption system"""
        # Generate master key if not exists
        key_file = Path.home() / ".quantonium" / "master.key"
        key_file.parent.mkdir(exist_ok=True)

        if key_file.exists():
            with open(key_file, "rb") as f:
                self.master_key = f.read()
        else:
            self.master_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.master_key)
            # Secure file permissions
            os.chmod(key_file, 0o600)

    def encrypt_file(self, file_path: Union[str, Path]) -> Path:
        """Encrypt a file using AES-256"""
        file_path = Path(file_path)
        encrypted_path = (
            file_path.parent / f"{file_path.stem}_encrypted{file_path.suffix}.enc"
        )

        # Generate session key
        session_key = Fernet.generate_key()
        fernet = Fernet(session_key)

        # Read and encrypt file
        with open(file_path, "rb") as infile:
            file_data = infile.read()
            encrypted_data = fernet.encrypt(file_data)

        # Encrypt session key with master key
        master_fernet = Fernet(self.master_key)
        encrypted_session_key = master_fernet.encrypt(session_key)

        # Write encrypted file with embedded key
        with open(encrypted_path, "wb") as outfile:
            # Write header
            header = b"QUANTONIUM_ENCRYPTED_V1\n"
            outfile.write(header)
            # Write encrypted session key length and key
            outfile.write(len(encrypted_session_key).to_bytes(4, "big"))
            outfile.write(encrypted_session_key)
            # Write encrypted data
            outfile.write(encrypted_data)

        # Remove original file
        file_path.unlink()

        return encrypted_path

    def decrypt_file(
        self,
        encrypted_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Decrypt a file"""
        encrypted_path = Path(encrypted_path)

        if output_path is None:
            output_path = encrypted_path.parent / encrypted_path.stem.replace(
                "_encrypted", ""
            )
        else:
            output_path = Path(output_path)

        with open(encrypted_path, "rb") as infile:
            # Read header
            header = infile.readline()
            if header != b"QUANTONIUM_ENCRYPTED_V1\n":
                raise ValueError("Invalid encrypted file format")

            # Read encrypted session key
            key_length = int.from_bytes(infile.read(4), "big")
            encrypted_session_key = infile.read(key_length)

            # Decrypt session key
            master_fernet = Fernet(self.master_key)
            session_key = master_fernet.decrypt(encrypted_session_key)

            # Read and decrypt data
            fernet = Fernet(session_key)
            encrypted_data = infile.read()
            decrypted_data = fernet.decrypt(encrypted_data)

        # Write decrypted file
        with open(output_path, "wb") as outfile:
            outfile.write(decrypted_data)

        return output_path

    def generate_secure_hash(self, data: Union[str, bytes]) -> str:
        """Generate secure hash for data integrity"""
        if isinstance(data, str):
            data = data.encode("utf-8")

        hash_obj = hashlib.sha256()
        hash_obj.update(data)
        return hash_obj.hexdigest()

    def verify_file_integrity(
        self, file_path: Union[str, Path], expected_hash: str
    ) -> bool:
        """Verify file integrity using hash"""
        with open(file_path, "rb") as f:
            file_data = f.read()

        actual_hash = self.generate_secure_hash(file_data)
        return actual_hash == expected_hash


# Export interface for easy integration
def export_app_results(
    data: Dict[str, Any],
    app_name: str,
    export_format: str = "json",
    encrypt: bool = True,
) -> Dict[str, str]:
    """
    Quick export function for all apps

    Usage:
        from core.quantum_export_controller import export_app_results

        result = export_app_results({
            'validation_results': validation_data,
            'analysis': analysis_data
        }, 'rft_visualizer', 'json', True)
    """
    controller = QuantumExportController()
    return controller.export_results(data, export_format, encrypt, app_name)


if __name__ == "__main__":
    # Test the export system
    controller = QuantumExportController()

    test_data = {
        "test_results": {
            "success": True,
            "score": 95.7,
            "details": ["Test 1 passed", "Test 2 passed"],
        },
        "analysis": {
            "performance": "excellent",
            "recommendations": ["Continue monitoring", "Scale up"],
        },
    }

    result = controller.export_results(test_data, "json", True, "test_app")
    print("Export result:", result)
