"""
Build script for the crypto engine in QuantoniumOS.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _10_UTILITIES.build_engine_base import EngineBuilder


class CryptoEngineBuilder(EngineBuilder):
    """
    Builder for the crypto engine.
    """

    def __init__(self):
        super().__init__(
            engine_name="crypto",
            source_dir="core/cpp/cryptography",
            include_dirs=["core/include", "third_party/include"],
            libraries=["crypto"],
        )

    def get_source_files(self):
        """
        Get the source files for the crypto engine.
        """
        return [
            os.path.join(self.source_dir, "rft_crypto.cpp"),
            os.path.join("core/cpp/bindings", "rft_crypto_bindings.cpp"),
        ]


def main():
    builder = CryptoEngineBuilder()
    source_files = builder.get_source_files()
    success = builder.build(source_files)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
