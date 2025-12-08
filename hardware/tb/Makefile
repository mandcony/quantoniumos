# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# RFTPU Testbench Makefile
# ========================
# Build and run testbenches using Verilator
#

# Directories
TOP_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
BUILD_DIR := $(TOP_DIR)/build
TB_DIR := $(TOP_DIR)/tb
RESULTS_DIR := $(TB_DIR)/results

# Tools
VERILATOR := verilator
PYTHON := python3
SANDPIPER := sandpiper-saas

# Source files
TLV_SRC := $(TOP_DIR)/rftpu_architecture.tlv
SV_GEN := $(BUILD_DIR)/rftpu_architecture.sv
PSEUDO_RAND := $(BUILD_DIR)/pseudo_rand.sv

# Testbenches
TB_RFT := $(TB_DIR)/tb_rft_core.sv
TB_RFTPU := $(TB_DIR)/tb_rftpu_accelerator.sv
TB_FORMAL := $(TB_DIR)/rftpu_formal_props.sv

# Test vectors
TEST_VEC_PY := $(TB_DIR)/rftpu_test_vectors.py
TEST_VEC_SVH := $(TB_DIR)/rft_test_vectors.svh
TEST_VEC_JSON := $(TB_DIR)/test_vectors_metadata.json

# Verilator flags
VERILATOR_FLAGS := --binary -j 0 --trace
VERILATOR_FLAGS += -Wall --Wno-IMPLICIT --Wno-WIDTHEXPAND --Wno-UNUSEDSIGNAL
VERILATOR_FLAGS += -CFLAGS "-g -O2"
VERILATOR_FLAGS += -I$(BUILD_DIR) -I$(TB_DIR)
VERILATOR_FLAGS += -DTRACE

# Default target
.PHONY: all
all: test-vectors lint

# Generate test vectors from Python
.PHONY: test-vectors
test-vectors: $(TEST_VEC_SVH)

$(TEST_VEC_SVH): $(TEST_VEC_PY)
	@echo "=== Generating test vectors ==="
	cd $(TB_DIR) && $(PYTHON) rftpu_test_vectors.py
	@echo "Generated: $(TEST_VEC_SVH)"
	@echo "Generated: $(TEST_VEC_JSON)"

# Compile TLV to SV (requires SandPiper)
.PHONY: compile-tlv
compile-tlv: $(SV_GEN)

$(SV_GEN): $(TLV_SRC)
	@echo "=== Compiling TLV to SystemVerilog ==="
	cd $(TOP_DIR) && $(SANDPIPER) -i $< -o $(BUILD_DIR)/rftpu_architecture.sv \
		--sv --bestsv --inlineGen --iArgs
	@echo "Generated: $@"

# Lint with Verilator (no simulation)
.PHONY: lint
lint: $(SV_GEN) $(PSEUDO_RAND) $(TEST_VEC_SVH)
	@echo "=== Verilator Lint Check ==="
	$(VERILATOR) --lint-only -Wall \
		--Wno-IMPLICIT --Wno-WIDTHEXPAND --Wno-UNUSEDSIGNAL \
		-I$(BUILD_DIR) -I$(TB_DIR) \
		$(SV_GEN) $(PSEUDO_RAND)
	@echo "Lint passed!"

# Build RFT core testbench
.PHONY: build-tb-rft
build-tb-rft: $(TEST_VEC_SVH) $(SV_GEN)
	@echo "=== Building RFT Core Testbench ==="
	mkdir -p $(RESULTS_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module tb_rft_core \
		$(TB_RFT) $(SV_GEN) $(PSEUDO_RAND) \
		-Mdir $(RESULTS_DIR)/obj_rft

# Build full accelerator testbench
.PHONY: build-tb-rftpu
build-tb-rftpu: $(TEST_VEC_SVH) $(SV_GEN)
	@echo "=== Building RFTPU Accelerator Testbench ==="
	mkdir -p $(RESULTS_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) \
		--top-module tb_rftpu_accelerator \
		$(TB_RFTPU) $(SV_GEN) $(PSEUDO_RAND) \
		-Mdir $(RESULTS_DIR)/obj_rftpu

# Run RFT core tests
.PHONY: run-tb-rft
run-tb-rft: build-tb-rft
	@echo "=== Running RFT Core Tests ==="
	$(RESULTS_DIR)/obj_rft/Vtb_rft_core
	@echo "Waveform: $(RESULTS_DIR)/obj_rft/tb_rft_core.vcd"

# Run full accelerator tests
.PHONY: run-tb-rftpu
run-tb-rftpu: build-tb-rftpu
	@echo "=== Running RFTPU Accelerator Tests ==="
	$(RESULTS_DIR)/obj_rftpu/Vtb_rftpu_accelerator
	@echo "Waveform: $(RESULTS_DIR)/obj_rftpu/tb_rftpu_accelerator.vcd"

# Run all tests
.PHONY: test
test: run-tb-rft run-tb-rftpu
	@echo ""
	@echo "=== All Tests Complete ==="

# Cross-validation testbench
TB_CROSSVAL := $(TB_DIR)/tb_canonical_rft_crossval.sv
CROSSVAL_PY := $(TB_DIR)/verify_rtl_vs_python.py

.PHONY: build-crossval
build-crossval: $(TEST_VEC_SVH)
	@echo "=== Building Cross-Validation Testbench ==="
	mkdir -p $(RESULTS_DIR)
	$(VERILATOR) --binary -j 0 \
		--Wno-IMPLICIT --Wno-WIDTHEXPAND --Wno-UNUSEDSIGNAL \
		--Wno-DECLFILENAME --Wno-UNDRIVEN --Wno-BLKSEQ \
		-I$(TB_DIR) \
		--top-module tb_canonical_rft_crossval \
		$(TB_CROSSVAL) \
		-Mdir $(RESULTS_DIR)/obj_crossval
	@echo "Built: $(RESULTS_DIR)/obj_crossval/Vtb_canonical_rft_crossval"

.PHONY: run-crossval
run-crossval: build-crossval
	@echo "=== Running RTL Cross-Validation ==="
	cd $(TB_DIR) && $(RESULTS_DIR)/obj_crossval/Vtb_canonical_rft_crossval
	@echo ""
	@echo "=== Verifying against Python reference ==="
	cd $(TB_DIR) && $(PYTHON) verify_rtl_vs_python.py

# Formal verification (requires SymbiYosys)
.PHONY: formal
formal:
	@echo "=== Formal Verification ==="
	@echo "TODO: Generate SymbiYosys script and run"
	@echo "Properties defined in: $(TB_FORMAL)"

# Quick check - lint only, no simulation
.PHONY: check
check: test-vectors lint
	@echo ""
	@echo "=== Quick Check Complete ==="

# Clean generated files
.PHONY: clean
clean:
	rm -rf $(RESULTS_DIR)/obj_*
	rm -f $(TB_DIR)/*.vcd
	rm -f $(TB_DIR)/test_vectors_metadata.json
	rm -f $(TB_DIR)/rft_test_vectors.svh
	rm -f $(TB_DIR)/rft_test_vectors.hex

# Clean everything including SV build
.PHONY: distclean
distclean: clean
	rm -rf $(BUILD_DIR)/*

# View waveforms (requires GTKWave)
.PHONY: wave-rft
wave-rft:
	gtkwave $(RESULTS_DIR)/obj_rft/tb_rft_core.vcd &

.PHONY: wave-rftpu
wave-rftpu:
	gtkwave $(RESULTS_DIR)/obj_rftpu/tb_rftpu_accelerator.vcd &

# Help
.PHONY: help
help:
	@echo "RFTPU Testbench Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Generate test vectors and lint (default)"
	@echo "  test-vectors   - Generate test vectors from Python"
	@echo "  compile-tlv    - Compile TLV to SystemVerilog"
	@echo "  lint           - Verilator lint check"
	@echo "  build-tb-rft   - Build RFT core testbench"
	@echo "  build-tb-rftpu - Build full accelerator testbench"
	@echo "  run-tb-rft     - Run RFT core tests"
	@echo "  run-tb-rftpu   - Run accelerator tests"
	@echo "  test           - Run all tests"
	@echo "  formal         - Run formal verification (WIP)"
	@echo "  check          - Quick lint check"
	@echo "  clean          - Remove generated files"
	@echo "  help           - Show this help"
