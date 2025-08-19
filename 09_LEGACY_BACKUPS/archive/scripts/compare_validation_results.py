"""
Compare validation results between C++ and Rust implementations
"""

import sys
import json
from typing import Dict, Any

def load_results(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

def compare_results(cpp_results: Dict[str, Any], rust_results: Dict[str, Any]) -> bool:
    if cpp_results['total_vectors'] != rust_results['total_vectors']:
        print(f"❌ Vector count mismatch: C++={cpp_results['total_vectors']}, Rust={rust_results['total_vectors']}")
        return False

    if cpp_results['passed'] != rust_results['passed']:
        print(f"❌ Pass count mismatch: C++={cpp_results['passed']}, Rust={rust_results['passed']}")
        return False

    cpp_failures = {f['vector_index']: f for f in cpp_results['failed']}
    rust_failures = {f['vector_index']: f for f in rust_results['failed']}

    if cpp_failures.keys() != rust_failures.keys():
        print("❌ Failure pattern mismatch between implementations")
        print(f"C++ failed vectors: {sorted(cpp_failures.keys())}")
        print(f"Rust failed vectors: {sorted(rust_failures.keys())}")
        return False

    print("\n✅ All validation results match!")
    print(f"Total vectors: {cpp_results['total_vectors']}")
    print(f"Passed: {cpp_results['passed']}")
    print(f"Failed: {len(cpp_results['failed'])}")

    return True

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cpp_results.json> <rust_results.json>")
        sys.exit(1)

    cpp_results = load_results(sys.argv[1])
    rust_results = load_results(sys.argv[2])

    if not compare_results(cpp_results, rust_results):
        sys.exit(1)

if __name__ == '__main__':
    main()
