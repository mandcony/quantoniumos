# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import threading

from atomic_io import AtomicJsonlWriter


def test_atomic_jsonl_writer_thread_safety(tmp_path):
    log_path = tmp_path / "events.jsonl"
    total_threads = 8
    per_thread = 250
    total_expected = total_threads * per_thread

    writer = AtomicJsonlWriter(log_path)

    def worker(tid: int):
        for i in range(per_thread):
            writer.write({"thread": tid, "seq": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(total_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    writer.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == total_expected

    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == total_expected
