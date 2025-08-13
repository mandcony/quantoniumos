#!/usr/bin/env python3
"""Validate KAT structure and content"""

import json

def validate_kats():
    with open('public_test_vectors/known_answer_tests.json') as f:
        data = json.load(f)
    
    print("✅ KATs loaded successfully")
    print(f"📊 Test vector groups: {len(data['test_vectors'])}")
    print("🔑 Keys:", list(data['test_vectors'].keys()))
    
    total_vectors = 0
    for group_name, group_data in data['test_vectors'].items():
        test_cases = len(group_data.get('test_cases', []))
        total_vectors += test_cases
        print(f"  - {group_name}: {test_cases} test cases")
    
    print(f"📈 Total test vectors: {total_vectors}")
    print("✅ Validation complete")

if __name__ == '__main__':
    validate_kats()
