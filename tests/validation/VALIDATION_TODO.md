# QuantoniumOS Validation TODO

## Build & Benchmark
- [x] Compile native UnitaryRFT kernel (`cd src/assembly && make all`) and rerun `validation/run_all_validations.py` to confirm Python fallback is gone.
- [ ] Capture latency/throughput metrics for the native kernel (per transform size) and archive them under `validation/results/`.

## Scale & Stress Tests
- [ ] Extend `test_rft_vertex_codec_roundtrip.py` with ≥1 MB tensors to verify lossless round-trips at scale.
- [ ] Add large-sample tests for `encode_tensor_hybrid` / `decode_tensor_hybrid` showing error bounds hold on megabyte inputs.
- [ ] Introduce streaming/CTR/CBC mode tests for `EnhancedRFTCryptoV2` using ≥1 MB random payloads.

## Statistical Certification
- [ ] Run NIST SP 800-22 on ciphertext and hash streams; store command lines, seeds, and reports in `validation/results/`.
- [ ] Run Dieharder on the same streams and capture outputs in `validation/results/`.

## Documentation & Traceability
- [ ] Update `REQUIRED_BENCHMARKS.md` and patent alignment docs with explicit test-to-claim mappings (report paths + hashes).
- [ ] Publish an updated `MASTER_VALIDATION_REPORT` with links to all generated metrics, native kernel benchmarks, and randomness logs.

## Full Replication Workflow
- [ ] Create `scripts/bootstrap_and_validate.sh` (or CI job) that builds dependencies from scratch, compiles kernels, runs the entire validation suite, and exports the final report.
