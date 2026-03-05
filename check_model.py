#!/usr/bin/env python3
"""
Check ML-PES Model Contents
============================
Diagnostic tool to see what's actually in your model pickle file.

Usage:
    python3 check_model.py <model.pkl>
"""

import sys
import pickle
import numpy as np
from pathlib import Path


def check_model(filepath):
    """Check what's in the model file."""
    print("\n" + "="*70)
    print(f"CHECKING: {filepath}")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded successfully")
    print(f"  Type: {type(data).__name__}")
    
    # If it's a dict, show keys
    if isinstance(data, dict):
        print(f"\n📦 Dictionary keys:")
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key:20s}: ndarray {value.shape}")
            elif hasattr(value, '__class__'):
                print(f"  {key:20s}: {type(value).__name__}")
            else:
                print(f"  {key:20s}: {type(value).__name__}")
        
        # Check critical components
        print(f"\n🔍 Critical Components:")
        
        # Model
        if 'model' in data:
            print(f"  ✓ model: {type(data['model']).__name__}")
        elif 'energy_model' in data:
            print(f"  ✓ energy_model: {type(data['energy_model']).__name__}")
        else:
            print(f"  ❌ No 'model' or 'energy_model' key!")
        
        # Scalers
        if 'scaler_X' in data:
            print(f"  ✓ scaler_X: {type(data['scaler_X']).__name__}")
        else:
            print(f"  ❌ No scaler_X!")
        
        if 'scaler_y' in data:
            print(f"  ✓ scaler_y: {type(data['scaler_y']).__name__}")
        else:
            print(f"  ❌ No scaler_y!")
        
        # Descriptor
        if 'descriptor' in data:
            if data['descriptor'] is not None:
                print(f"  ✓ descriptor: {type(data['descriptor']).__name__}")
            else:
                print(f"  ⚠️  descriptor: None (will create on-the-fly)")
        else:
            print(f"  ⚠️  No descriptor key (will create on-the-fly)")
        
        # Symbols
        if 'symbols' in data:
            print(f"  ✓ symbols: {data['symbols']}")
        elif 'metadata' in data and isinstance(data['metadata'], dict):
            if 'symbols' in data['metadata']:
                print(f"  ✓ symbols (in metadata): {data['metadata']['symbols']}")
            else:
                print(f"  ⚠️  No symbols in metadata")
        else:
            print(f"  ⚠️  No symbols found!")
        
        # Metadata
        if 'metadata' in data:
            metadata = data['metadata']
            if isinstance(metadata, dict):
                print(f"\n📋 Metadata:")
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool, list)):
                        if isinstance(value, list) and len(value) > 10:
                            print(f"  {key:20s}: list[{len(value)}]")
                        else:
                            print(f"  {key:20s}: {value}")
                    else:
                        print(f"  {key:20s}: {type(value).__name__}")
            else:
                print(f"\n📋 Metadata: {type(metadata).__name__}")
    
    else:
        # It's an object
        print(f"\n📦 Object attributes:")
        for attr in dir(data):
            if not attr.startswith('_'):
                value = getattr(data, attr)
                if not callable(value):
                    print(f"  {attr:20s}: {type(value).__name__}")
    
    # Summary
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if isinstance(data, dict):
        has_model = 'model' in data or 'energy_model' in data
        has_scalers = 'scaler_X' in data and 'scaler_y' in data
        has_descriptor = data.get('descriptor') is not None
        has_symbols = 'symbols' in data or ('metadata' in data and isinstance(data['metadata'], dict) and 'symbols' in data['metadata'])
        
        print(f"\n{'Component':<20s} {'Status':<10s}")
        print("-"*70)
        print(f"{'Model':<20s} {'✓' if has_model else '❌':<10s}")
        print(f"{'Scalers (X, y)':<20s} {'✓' if has_scalers else '❌':<10s}")
        print(f"{'Descriptor':<20s} {'✓' if has_descriptor else '⚠️ (can create)':<10s}")
        print(f"{'Symbols':<20s} {'✓' if has_symbols else '❌':<10s}")
        
        if has_model and has_scalers:
            print(f"\n✅ Model can be used with wrapper!")
            if not has_descriptor:
                print(f"   Descriptor will be created automatically")
        else:
            print(f"\n❌ Model is incomplete!")
    else:
        print(f"\n✓ Object-based model (should work directly)")
    
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python3 check_model.py <model.pkl>")
        print("\nExample:")
        print("  python3 check_model.py outputs/refined_*/mlpes_model_energy_forces.pkl")
        sys.exit(1)
    
    model_file = sys.argv[1]
    
    if not Path(model_file).exists():
        print(f"\n❌ File not found: {model_file}")
        sys.exit(1)
    
    check_model(model_file)
