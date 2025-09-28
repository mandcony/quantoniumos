#!/usr/bin/env python3
"""Compatibility wrapper for the GPT-OSS-120B streaming compressor.

This file is kept for backwards compatibility with existing scripts that
import :class:`RealGPTOSS120BCompressor` directly from
``add_gpt_oss_120b_to_brain``. The real implementation now lives in
``gpt_oss_120b_streaming.py`` which provides a streaming-aware download and
compression pipeline.
"""

from gpt_oss_120b_streaming import (
    RealGPTOSS120BCompressor,
    build_arg_parser,
    main as _streaming_main,
)

__all__ = ["RealGPTOSS120BCompressor", "build_arg_parser"]


def main() -> None:  # pragma: no cover - thin CLI
    _streaming_main()


if __name__ == "__main__":  # pragma: no cover
    main()