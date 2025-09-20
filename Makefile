# QuantoniumOS Assembly Build System
CXX = g++
CXXFLAGS = -std=c++17 -O3 -fPIC -march=native
INCLUDES = -Isystem/assembly/assembly/include -Isystem/assembly/assembly/engines
LIBDIR = system/assembly/assembly/lib

# Python configuration
PYTHON_VERSION = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDES = $(shell python3 -m pybind11 --includes)
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags)

# Source files
ENGINE_SOURCES = $(wildcard system/assembly/assembly/engines/*.cpp)
UNIFIED_SOURCES = $(wildcard system/assembly/assembly/unified/*.cpp)
BINDING_SOURCES = $(wildcard system/assembly/assembly/python_bindings/*.cpp)

# Object files
ENGINE_OBJECTS = $(ENGINE_SOURCES:.cpp=.o)
UNIFIED_OBJECTS = $(UNIFIED_SOURCES:.cpp=.o)
BINDING_OBJECTS = $(BINDING_SOURCES:.cpp=.o)

# Targets
all: lib/libquantum_symbolic.dll python_modules/quantum_assembly_bindings.pyd

# Create directories
directories:
	@mkdir -p lib python_modules

# Assembly library
lib/libquantum_symbolic.dll: directories $(ENGINE_OBJECTS) $(UNIFIED_OBJECTS)
	$(CXX) -shared -o $@ $(ENGINE_OBJECTS) $(UNIFIED_OBJECTS) $(PYTHON_LDFLAGS)

# Python bindings
python_modules/quantum_assembly_bindings.pyd: directories $(BINDING_OBJECTS) lib/libquantum_symbolic.dll
	$(CXX) -shared -o $@ $(BINDING_OBJECTS) -Llib -lquantum_symbolic $(PYTHON_LDFLAGS)

# Object file compilation
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYTHON_INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(ENGINE_OBJECTS) $(UNIFIED_OBJECTS) $(BINDING_OBJECTS)
	rm -f lib/libquantum_symbolic.dll python_modules/quantum_assembly_bindings.pyd

# Install
install: all
	cp lib/libquantum_symbolic.dll system/assembly/assembly/lib/
	cp python_modules/quantum_assembly_bindings.pyd system/assembly/assembly/python_bindings/

# Test
test: all
	python3 test_assembly_compilation.py

.PHONY: all clean install test directories
