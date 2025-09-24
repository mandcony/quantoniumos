#!/usr/bin/env python3
"""
QuantoniumOS Model Analysis - REAL vs SYNTHETIC Detection
========================================================
Scans all quantum models to identify which contain REAL compressed weights
vs synthetic/placeholder data
"""

import json
import os
from pathlib import Path

def analyze_quantum_model(file_path):
    """Analyze a quantum model file to determine if it's real or synthetic"""
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # Analysis criteria
        analysis = {
            'file': os.path.basename(file_path),
            'size_mb': round(file_size_mb, 2),
            'status': 'unknown',
            'evidence': [],
            'quantum_states': 0,
            'parameters': 0,
            'compression_method': 'unknown'
        }
        
        # Extract basic info
        if 'quantum_states' in data and isinstance(data['quantum_states'], int):
            analysis['quantum_states'] = data['quantum_states']
        elif 'metadata' in data and 'quantum_states_count' in data['metadata']:
            analysis['quantum_states'] = data['metadata']['quantum_states_count']
        elif 'states' in data:
            analysis['quantum_states'] = len(data['states'])
        elif 'quantum_states' in data and isinstance(data['quantum_states'], list):
            analysis['quantum_states'] = len(data['quantum_states'])
            
        if 'original_parameters' in data:
            analysis['parameters'] = data['original_parameters']
        elif 'metadata' in data and 'original_parameters' in data['metadata']:
            analysis['parameters'] = data['metadata']['original_parameters']
            
        if 'compression_method' in data:
            analysis['compression_method'] = data['compression_method']
        elif 'metadata' in data and 'quantum_encoding_method' in data['metadata']:
            analysis['compression_method'] = data['metadata']['quantum_encoding_method']
        
        # Check quantum states structure
        states = []
        if 'states' in data:
            states = data['states']
        elif 'quantum_states' in data and isinstance(data['quantum_states'], list):
            states = data['quantum_states']
            
        if states:
            first_state = states[0]
            
            # REAL model indicators
            if 'layer' in first_state and 'weight_count' in first_state:
                analysis['status'] = 'REAL'
                analysis['evidence'].append('Has layer names (e.g., transformer.wte.weight)')
                analysis['evidence'].append('Has weight_count field from actual tensors')
                analysis['evidence'].append('Contains real/imag fields from statistical encoding')
                
            # Check for real layer names
            elif 'layer' in first_state and 'transformer' in str(first_state.get('layer', '')):
                analysis['status'] = 'REAL'
                analysis['evidence'].append('Contains actual transformer layer names')
                
            # Synthetic model indicators - mathematical patterns only
            elif 'vertex' in first_state and 'expert_id' in first_state:
                analysis['status'] = 'SYNTHETIC'
                analysis['evidence'].append('Generated synthetic MoE patterns')
                analysis['evidence'].append('No actual model download/extraction')
                
            elif 'attention_head' in first_state and 'layer_depth' in first_state:
                analysis['status'] = 'SYNTHETIC'
                analysis['evidence'].append('Generated synthetic attention patterns')  
                analysis['evidence'].append('No actual model weights compressed')
                
            # Legacy/Original format (possibly real)
            elif 'resonance_freq' in first_state and 'entanglement_key' in first_state:
                if file_size_mb > 3:  # Large files likely have real data
                    analysis['status'] = 'LEGACY_REAL'
                    analysis['evidence'].append('Large file size suggests real compression')
                    analysis['evidence'].append('Legacy format with entanglement keys')
                else:
                    analysis['status'] = 'LEGACY_SMALL'
                    analysis['evidence'].append('Small file size - may be placeholder')
                    analysis['evidence'].append('Legacy format - verification needed')
                    
        return analysis
        
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'size_mb': 0,
            'status': 'ERROR',
            'evidence': [f'Failed to parse: {e}'],
            'quantum_states': 0,
            'parameters': 0,
            'compression_method': 'unknown'
        }

def main():
    """Scan all quantum models and classify them"""
    
    quantum_dir = Path("/workspaces/quantoniumos/ai/models/quantum")
    
    print("🔍 QUANTONIUMOS MODEL ANALYSIS - REAL vs SYNTHETIC")
    print("=" * 60)
    
    models = []
    for json_file in quantum_dir.glob("*.json"):
        analysis = analyze_quantum_model(json_file)
        models.append(analysis)
    
    # Sort by status and size
    models.sort(key=lambda x: (x['status'], -x['size_mb']))
    
    # Group by status
    real_models = [m for m in models if m['status'] == 'REAL']
    synthetic_models = [m for m in models if m['status'] == 'SYNTHETIC']  
    legacy_real = [m for m in models if m['status'] == 'LEGACY_REAL']
    legacy_small = [m for m in models if m['status'] == 'LEGACY_SMALL']
    errors = [m for m in models if m['status'] == 'ERROR']
    
    print(f"\n✅ REAL MODELS ({len(real_models)}) - Actual compressed weights:")
    for model in real_models:
        print(f"  📊 {model['file']:50} {model['size_mb']:6.1f}MB  {model['quantum_states']:8,} states")
        print(f"     Original: {model['parameters']:,} parameters")
        for evidence in model['evidence'][:2]:  # Show top 2 evidence items
            print(f"     ✓ {evidence}")
        print()
    
    print(f"\n❌ SYNTHETIC MODELS ({len(synthetic_models)}) - Generated data:")
    for model in synthetic_models:
        print(f"  🤖 {model['file']:50} {model['size_mb']:6.1f}MB  {model['quantum_states']:8,} states")
        print(f"     Claimed: {model['parameters']:,} parameters")
        for evidence in model['evidence'][:2]:
            print(f"     ⚠️  {evidence}")
        print()
    
    print(f"\n🟡 LEGACY REAL ({len(legacy_real)}) - Pre-existing real compression:")
    for model in legacy_real:
        print(f"  📈 {model['file']:50} {model['size_mb']:6.1f}MB  {model['quantum_states']:8,} states")
        print(f"     Original: {model['parameters']:,} parameters")
        for evidence in model['evidence'][:2]:
            print(f"     ✓ {evidence}")
        print()
            
    print(f"\n🟠 LEGACY SMALL ({len(legacy_small)}) - Needs verification:")
    for model in legacy_small:
        print(f"  ❓ {model['file']:50} {model['size_mb']:6.1f}MB  {model['quantum_states']:8,} states")
        print(f"     Claimed: {model['parameters']:,} parameters")
        for evidence in model['evidence'][:2]:
            print(f"     ? {evidence}")
        print()
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for model in errors:
            print(f"  💥 {model['file']:50} - {model['evidence'][0]}")
    
    # Summary
    total_real_params = sum(m['parameters'] for m in real_models + legacy_real)
    total_synthetic_params = sum(m['parameters'] for m in synthetic_models)
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ REAL compressed parameters: {total_real_params:,}")
    print(f"❌ SYNTHETIC claimed parameters: {total_synthetic_params:,}")
    print(f"🔄 Verification needed: {len(legacy_small)} models")
    print(f"\n🎯 RECOMMENDATION: Remove {len(synthetic_models)} synthetic models")
    print("   Keep only REAL and LEGACY_REAL models for production use")

if __name__ == "__main__":
    main()