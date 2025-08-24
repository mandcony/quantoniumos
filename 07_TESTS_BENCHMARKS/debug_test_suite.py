import traceback

# Add debugging to see where is_test_mode is being used
original_init = __builtins__.__dict__["__import__"]


def patched_import(name, *args, **kwargs):
    module = original_init(name, *args, **kwargs)
    if name == "bulletproof_quantum_kernel":
        print(f"Module {name} imported")
    return module


__builtins__.__dict__["__import__"] = patched_import

try:
    from comprehensive_scientific_test_suite import \
        run_comprehensive_scientific_tests

    print("Starting test suite...")
    results = run_comprehensive_scientific_tests()
    print("Test suite completed successfully")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
