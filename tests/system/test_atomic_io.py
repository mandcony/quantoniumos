# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from atomic_io import atomic_write_json


def test_atomic_write_json(tmp_path):
    target = tmp_path / "report.json"
    payload = {"ok": True, "items": [1, 2, 3]}
    atomic_write_json(target, payload, indent=2)

    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload
