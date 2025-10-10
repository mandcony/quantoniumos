# Repository Restructure Plan (Dry-Run)

Goal: make the project easy to navigate with clear, minimal, and scalable structure. This plan is non-destructive until approved.

## Target top-level layout

```
quantoniumos/
├── apps/                  # (was src/apps) end-user apps & launchers
├── core/                  # math, transforms, codecs, crypto (src/core merged)
├── kernels/               # (was src/assembly) native code & bindings
├── engine/                # orchestration/experimental engines (src/engine)
├── ui/                    # frontend/desktop shells (src/frontend, ui)
├── tools/                 # CLIs and scripts (dev/tools merged)
├── data/                  # small config/data (non-large)
├── models/                # ai/models + encoded_models + decoded_models (gate via LFS)
├── tests/                 # test suites (unchanged)
├── docs/                  # docs & reports (unchanged)
├── results/               # generated outputs (ignored by git)
├── scripts/               # short helpers (optional)
├── README.md
└── boot.py                # (was quantonium_boot.py)
```

## Proposed renames/moves (high level)

- src/apps/ → apps/
- src/core/ → core/
- src/assembly/ → kernels/
- src/engine/ → engine/
- src/frontend/ → ui/
- ai/models/, encoded_models/, decoded_models/ → models/
- quantonium_boot.py → boot.py
- dev/tools/, tools/ → tools/ (merge; dedupe name collisions)

## File-level normalizations

- Use snake_case file names for scripts/CLIs.
- Consolidate launcher files under apps/ with `launch_*.py` prefix.
- Co-locate tests by domain (tests/apps, tests/core, tests/crypto, etc.) already exists—keep and prune duplicates.
- Move stray artifacts (e.g., root PDFs) into docs/assets/.

## Git hygiene

- Ignore caches: `__pycache__/`, `.pytest_cache/`, `results/`, `logs/`.
- Gate large assets via Git LFS or Releases: `models/`.
- Keep a small whitelist of verified assets (e.g., tiny-gpt2) in-repo if needed.

## Mapping examples (partial)

- `src/apps/compressed_model_router.py` → `apps/compressed_model_router.py`
- `src/core/rft_vertex_codec.py` → `core/rft_vertex_codec.py`
- `src/core/rft_hybrid_codec.py` → `core/rft_hybrid_codec.py`
- `src/assembly/kernel/rft_kernel.c` → `kernels/kernel/rft_kernel.c`
- `src/frontend/quantonium_desktop.py` → `ui/quantonium_desktop.py`
- `dev/tools/print_rft_invariants.py` → `tools/print_rft_invariants.py`
- `encoded_models/tiny_gpt2_lossless/` → `models/encoded/tiny_gpt2_lossless/`
- `decoded_models/tiny_gpt2_lossless/` → `models/decoded/tiny_gpt2_lossless/`
- `ai/models/quantum/` → `models/quantum/`

## Open questions

- Keep legacy paths for backward-compat in import shims? e.g., `src/__init__.py` importing from new modules for one release.
- Which model assets must remain in-repo vs. moved to Releases/LFS?
- Do we split `tools/` into `tools/` (end-user CLIs) and `dev/` (developer utilities)?

## Next steps

1. Confirm the target layout and mapping list.
2. Run a dry-run script to list planned moves and potential conflicts.
3. Apply the changes in a feature branch and fix imports/tests.
4. Add import shims for one release cycle.
