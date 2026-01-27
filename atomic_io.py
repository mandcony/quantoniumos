#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Atomic file utilities for crash-safe artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fp:
            fp.write(text)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, target)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def atomic_write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    payload = json.dumps(obj, indent=indent, ensure_ascii=False)
    atomic_write_text(path, payload)


class AtomicJsonlWriter:
    """Thread-safe JSONL writer with fsync for crash safety."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fp = open(self._path, "a", encoding="utf-8")

    def write(self, obj: Any) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            self._fp.write(line)
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def close(self) -> None:
        with self._lock:
            self._fp.close()

    def __enter__(self) -> "AtomicJsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
