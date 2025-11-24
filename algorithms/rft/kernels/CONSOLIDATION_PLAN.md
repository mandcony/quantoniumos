## Kernel Engine Consolidation Notes

- `engines/crypto/`
	- `include/feistel_round48.h`, `include/sha256_portable.h`
	- `src/feistel_round48.c`, `src/sha256_portable.c`
	- `asm/feistel_round48.asm`
	- `bindings/feistel_pybind.cpp`
- `engines/orchestrator/`
	- `asm/rft_transform.asm`

Build integration updated via `unified_build/CMakeLists.txt` to point at the flattened include/src layout.
