#!/usr/bin/env python3
"""Script to organize test files by category."""

import os
import shutil
import glob

# Define file patterns for each category
categories = {
    'cryptography': [
        'test_AES.py', 'test_ARC*.py', 'test_BLAKE2.py', 'test_Blowfish.py', 'test_CAST.py',
        'test_CBC.py', 'test_CCM.py', 'test_CFB.py', 'test_CMAC.py', 'test_CTR.py',
        'test_ChaCha*.py', 'test_Counter.py', 'test_DES*.py', 'test_DSA.py', 'test_EAX.py',
        'test_ECC_*.py', 'test_ElGamal.py', 'test_GCM.py', 'test_HMAC.py', 'test_HPKE.py',
        'test_KDF.py', 'test_KMAC.py', 'test_KW.py', 'test_KangarooTwelve.py', 'test_MD*.py',
        'test_OCB.py', 'test_OFB.py', 'test_OpenPGP.py', 'test_PBES.py', 'test_PKCS*.py',
        'test_Padding.py', 'test_Poly1305.py', 'test_Primality.py', 'test_RIPEMD160.py',
        'test_RSA.py', 'test_SHA*.py', 'test_SHAKE.py', 'test_SIV.py', 'test_Salsa20.py',
        'test_SecretSharing.py', 'test_TupleHash.py', 'test_TurboSHAKE.py', 'test_cSHAKE.py',
        'test_keccak.py', 'test_Numbers.py', 'test_import_*.py', 'test_pkcs1_*.py',
        'test_pss.py', 'test_ecdh.py', 'test_eddsa.py', 'test_rfc1751.py', 'test_sign_verify.py',
        'crypto_test.py', 'test_crypto.py', 'corrected_rft_crypto_test.py', 'deterministic_crypto_test.py',
        'high_performance_rft_crypto_test.py', 'quick_cipher_test.py', 'simplified_rft_crypto_test.py',
        'comprehensive_rft_crypto_statistical_test.py', 'test_avalanche.py', 'detailed_avalanche_test.py',
        'direct_avalanche_test.py', 'test_key_avalanche.py', 'test_xor_avalanche.py',
        'test_collision_resistance.py', 'test_constant_time.py', 'test_encrypt*.py',
        'test_encryption*.py', 'test_fixed_crypto.py', 'test_fixed_encryption.py',
        'test_formal_security.py', 'test_hash_sigma_tightening.py', 'test_key_management.py',
        'test_mixer_validation.py', 'test_non_randomness.py', 'test_working_encryption_engines.py',
        'enhanced_hash_test.py'
    ],
    'quantum': [
        'test_bulletproof_quantum_kernel.py', 'test_bell_state.py', 'test_quantum_*.py',
        'test_qubit_*.py', 'enhanced_quantum_vertex_validation.py', 'rigorous_quantum_vertex_validation.py',
        'test_50_qubit_vertices.py', 'quantum_*.py', 'test_randomized_benchmarking.py',
        'test_hamiltonian_recovery.py', 'test_unitarity.py', 'test_time_evolution.py',
        'test_trotter_error.py', 'test_resonance.py', 'test_choi_channel.py'
    ],
    'rft': [
        'test_rft_*.py', 'test_current_rft.py', 'test_true_rft_*.py', 'canonical_*.py',
        'test_mathematical_rft_validation.py', 'test_minimal_rft_demo.py', 'test_dynamic_rft_routing.py',
        'test_enhanced_cpp_rft.py', 'test_rft_geometric_waveform.py', 'test_geometric_waveform.py',
        'test_waveform_hash.py', 'test_geometric_hash_functions.py', 'test_geometric_vault.py',
        'test_patent_math.py', 'test_surgical_fix.py', 'test_comprehensive_fix.py',
        'test_rft_basis_fix.py', 'test_channel_capacity_fix.py'
    ],
    'performance': [
        'benchmark_*.py', 'advanced_rft_compression_benchmark.py', 'performance_*.py',
        'test_performance.py', 'test_final_performance.py', 'impulse_scaling_test.py',
        'test_state_evolution_benchmarks.py', 'test_vertex_scaling.py', 'rft_quantum_performance_test.py',
        'comprehensive_scientific_test_suite.py', 'test_superiority_test.py'
    ],
    'system': [
        'test_system.py', 'quick_system_test.py', 'core_ecosystem_validation.py',
        'comprehensive_*.py', 'full_patent_test*.py', 'final_paper_compliance_test.py',
        'test_all_claims.py', 'test_claim*_direct.py', 'run_*.py', 'better_run_tests.py',
        'test_auth.py', 'test_security.py', 'test_process*.py'
    ],
    'scientific': [
        'comprehensive_statistical_testing.py', 'portable_statistical_tests.py',
        'nist_statistical_tests.py', 'test_energy_conservation.py', 'test_epsilon_reproducibility.py',
        'ultra_low_variance_test.py', 'test_sensitivity_analysis.py', 'test_verification_suite.py'
    ],
    'utilities': [
        'test_utils.py', 'testutils.py', 'benchmark_utils.py', 'common_tests.py',
        'fix_test_imports.py', 'generate_test_vectors.py', 'hierarchy_test_data.py',
        'patch_test_suite.py', 'debug_test_suite.py', 'conftest.py', 'testing.py',
        'testTools.py', 'test_testutils.py', '_*.py'
    ]
}

def move_files_by_patterns(category, patterns):
    """Move files matching patterns to category directory."""
    moved_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file) and not file.startswith('./'):
                try:
                    dest = os.path.join(category, os.path.basename(file))
                    shutil.move(file, dest)
                    moved_files.append(file)
                    print(f"Moved {file} to {dest}")
                except Exception as e:
                    print(f"Error moving {file}: {e}")
    return moved_files

def main():
    # Change to the tests directory
    os.chdir('/workspaces/quantoniumos/07_TESTS_BENCHMARKS')
    
    total_moved = 0
    
    for category, patterns in categories.items():
        print(f"\n=== Processing {category} category ===")
        moved = move_files_by_patterns(category, patterns)
        total_moved += len(moved)
        print(f"Moved {len(moved)} files to {category}/")
    
    print(f"\nTotal files moved: {total_moved}")
    
    # Move remaining third-party/legacy files
    print("\n=== Moving legacy/third-party files ===")
    legacy_patterns = [
        'test_*.py'  # Any remaining test files
    ]
    
    remaining_files = []
    for pattern in legacy_patterns:
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file):
                remaining_files.append(file)
    
    # Filter out files that don't look like scipy/numpy/matplotlib tests
    quantonium_keywords = ['rft', 'quantum', 'crypto', 'avalanche', 'patent', 'claim', 'encryption']
    
    for file in remaining_files:
        filename = os.path.basename(file).lower()
        is_quantonium = any(keyword in filename for keyword in quantonium_keywords)
        
        if not is_quantonium:
            try:
                dest = os.path.join('legacy', os.path.basename(file))
                shutil.move(file, dest)
                print(f"Moved {file} to legacy/")
            except Exception as e:
                print(f"Error moving {file}: {e}")

if __name__ == '__main__':
    main()
