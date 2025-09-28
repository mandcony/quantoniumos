# Makefile for QuantoniumOS RFT Kernel
# Builds the Resonance Field Theory kernel implementation

CC = gcc
AS = nasm
AR = ar
CFLAGS = -Wall -Wextra -O3 -march=native -fPIC -std=c99
ASMFLAGS = -f elf64
LDFLAGS = -shared -lm

# Directories
SRCDIR = kernel
INCDIR = include
BUILDDIR = build
COMPILEDDIR = compiled

# Source files
C_SOURCES = $(SRCDIR)/rft_kernel_fixed.c $(SRCDIR)/rft_kernel_ui.c $(SRCDIR)/quantum_symbolic_compression.c
ASM_SOURCES = $(SRCDIR)/rft_kernel_asm.asm
HEADERS = $(INCDIR)/rft_kernel.h $(INCDIR)/rft_kernel_ui.h $(SRCDIR)/quantum_symbolic_compression.h

# Object files
C_OBJECTS = $(C_SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
ASM_OBJECTS = $(ASM_SOURCES:$(SRCDIR)/%.asm=$(BUILDDIR)/%.o)
OBJECTS = $(C_OBJECTS) $(ASM_OBJECTS)

# Target library
TARGET = $(COMPILEDDIR)/libquantum_symbolic.so
TARGET_STATIC = $(COMPILEDDIR)/libquantum_symbolic.a

# Windows targets
ifeq ($(OS),Windows_NT)
    TARGET = $(COMPILEDDIR)/libquantum_symbolic.dll
    ASMFLAGS = -f win64
    LDFLAGS = -shared -lm -Wl,--out-implib,$(COMPILEDDIR)/libquantum_symbolic.lib
endif

.PHONY: all clean install test

all: $(TARGET) $(TARGET_STATIC)

# Create directories
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(COMPILEDDIR):
	mkdir -p $(COMPILEDDIR)

# Compile C sources
$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(HEADERS) | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Assemble ASM sources
$(BUILDDIR)/%.o: $(SRCDIR)/%.asm | $(BUILDDIR)
	$(AS) $(ASMFLAGS) $< -o $@

# Link shared library
$(TARGET): $(OBJECTS) | $(COMPILEDDIR)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

# Create static library
$(TARGET_STATIC): $(OBJECTS) | $(COMPILEDDIR)
	$(AR) rcs $@ $(OBJECTS)

# Test the library
test: $(TARGET)
	cd python_bindings && python3 quantum_symbolic_engine.py

# Install headers and libraries
install: $(TARGET) $(TARGET_STATIC)
	sudo cp $(HEADERS) /usr/local/include/
	sudo cp $(TARGET) $(TARGET_STATIC) /usr/local/lib/
	sudo ldconfig

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR) $(COMPILEDDIR)

# Development targets
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

profile: CFLAGS += -pg
profile: $(TARGET)

# AddressSanitizer build
asan: CFLAGS += -O1 -g -fsanitize=address -fno-omit-frame-pointer
asan: LDFLAGS += -fsanitize=address
asan: $(TARGET) $(TARGET_STATIC)

# Documentation
docs:
	doxygen Doxyfile

# Show build info
info:
	@echo "QuantoniumOS RFT Kernel Build System"
	@echo "===================================="
	@echo "CC: $(CC)"
	@echo "AS: $(AS)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "ASMFLAGS: $(ASMFLAGS)"
	@echo "TARGET: $(TARGET)"
	@echo "SOURCES: $(C_SOURCES) $(ASM_SOURCES)"
