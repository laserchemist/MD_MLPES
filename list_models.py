#!/usr/bin/env python3
"""
List Available ML-PES Models and Training Data

Shows all models and training data with metadata to help you choose which to use.

Usage:
    python3 list_models.py
    python3 list_models.py --detailed
    python3 list_models.py --molecule CH2OO
"""

import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from modules.data_formats import load_trajectory
    FRAMEWORK = True
except ImportError:
    FRAMEWORK = False
    print("⚠️  Framework not available - limited info")

def format_date(date_str):
    """Format date from filename or metadata."""
    try:
        dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return date_str

def get_model_info(model_path):
    """Extract metadata from ML-PES model."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        theory = metadata.get('theory', {})
        model_meta = metadata.get('model', {})
        
        return {
            'path': str(model_path),
            'symbols': ' '.join(data.get('symbols', [])),
            'method': theory.get('method', 'unknown'),
            'basis': theory.get('basis', 'unknown'),
            'rmse': model_meta.get('test_rmse_kcal', None),
            'r2': model_meta.get('r2_score', None),
            'n_configs': model_meta.get('training_configs', None),
            'version': data.get('version', 'unknown'),
            'refined': 'refined' in str(model_path).lower() or metadata.get('model', {}).get('refined', False)
        }
    except Exception as e:
        return {
            'path': str(model_path),
            'error': str(e)
        }

def get_training_data_info(data_path):
    """Extract metadata from training data."""
    if not FRAMEWORK:
        return {'path': str(data_path), 'error': 'Framework not available'}
    
    try:
        data = load_trajectory(str(data_path))
        
        metadata = data.metadata if hasattr(data, 'metadata') and data.metadata else {}
        theory = metadata.get('theory', {})
        
        # Check if refined
        refinements = metadata.get('refinement', [])
        is_refined = len(refinements) > 0
        
        # Energy statistics
        e_kcal = data.energies * 627.509
        
        return {
            'path': str(data_path),
            'symbols': ' '.join(data.symbols),
            'method': theory.get('method', 'unknown'),
            'basis': theory.get('basis', 'unknown'),
            'n_frames': data.n_frames,
            'n_atoms': len(data.symbols),
            'has_forces': data.forces is not None,
            'energy_range': e_kcal.max() - e_kcal.min(),
            'energy_std': e_kcal.std(),
            'refined': is_refined,
            'n_refinements': len(refinements),
            'refinement_info': refinements[-1] if refinements else None
        }
    except Exception as e:
        return {
            'path': str(data_path),
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='List available ML-PES models and training data')
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    parser.add_argument('--molecule', help='Filter by molecule formula (e.g., CH2OO)')
    parser.add_argument('--outputs-dir', default='outputs', help='Outputs directory to scan')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("  AVAILABLE ML-PES MODELS AND TRAINING DATA")
    print("=" * 100)
    
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"\n❌ Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    # Find all models
    print("\n" + "=" * 100)
    print("  ML-PES MODELS (.pkl)")
    print("=" * 100)
    
    model_files = list(outputs_dir.rglob("*mlpes*.pkl")) + list(outputs_dir.rglob("*model*.pkl"))
    
    # Remove duplicates
    model_files = list(set(model_files))
    
    if not model_files:
        print("\n⚠️  No ML-PES models found in outputs/")
    else:
        models_info = []
        for model_path in model_files:
            info = get_model_info(model_path)
            if args.molecule:
                # Filter by molecule
                formula = ''.join(sorted(info.get('symbols', '').replace(' ', '')))
                filter_formula = ''.join(sorted(args.molecule.replace(' ', '')))
                if formula.lower() != filter_formula.lower():
                    continue
            models_info.append(info)
        
        # Sort by date (newest first)
        models_info.sort(key=lambda x: x['path'], reverse=True)
        
        for i, info in enumerate(models_info, 1):
            if 'error' in info:
                print(f"\n[{i}] ❌ {Path(info['path']).name}")
                print(f"    Error: {info['error']}")
                continue
            
            # Extract directory and date
            path = Path(info['path'])
            parent_dir = path.parent.name
            
            # Status
            status = "🔄 REFINED" if info['refined'] else "📊 INITIAL"
            
            print(f"\n[{i}] {status} {path.name}")
            print(f"    📁 {parent_dir}/")
            print(f"    🧪 {info['symbols']} | {info['method']}/{info['basis']}")
            
            if info['rmse'] is not None:
                rmse_str = f"{info['rmse']:.3f}" if isinstance(info['rmse'], float) else str(info['rmse'])
                quality = "✅" if info['rmse'] and info['rmse'] < 1.0 else "⚠️" if info['rmse'] and info['rmse'] < 10.0 else "❌"
                print(f"    {quality} RMSE: {rmse_str} kcal/mol", end="")
                if info['r2']:
                    print(f" | R²: {info['r2']:.4f}", end="")
                print()
            
            if info['n_configs']:
                print(f"    📊 Training configs: {info['n_configs']}")
            
            if args.detailed:
                print(f"    📄 Full path: {info['path']}")
    
    # Find all training data
    print("\n" + "=" * 100)
    print("  TRAINING DATA (.npz)")
    print("=" * 100)
    
    data_files = list(outputs_dir.rglob("*training*.npz")) + \
                 list(outputs_dir.rglob("*augmented*.npz")) + \
                 list(outputs_dir.rglob("mlpes_v2*.npz"))
    
    # Remove duplicates
    data_files = list(set(data_files))
    
    if not data_files:
        print("\n⚠️  No training data found in outputs/")
    else:
        data_info = []
        for data_path in data_files:
            info = get_training_data_info(data_path)
            if args.molecule:
                # Filter by molecule
                formula = ''.join(sorted(info.get('symbols', '').replace(' ', '')))
                filter_formula = ''.join(sorted(args.molecule.replace(' ', '')))
                if formula.lower() != filter_formula.lower():
                    continue
            data_info.append(info)
        
        # Sort by date (newest first)
        data_info.sort(key=lambda x: x['path'], reverse=True)
        
        for i, info in enumerate(data_info, 1):
            if 'error' in info:
                print(f"\n[{i}] ❌ {Path(info['path']).name}")
                print(f"    Error: {info['error']}")
                continue
            
            path = Path(info['path'])
            parent_dir = path.parent.name
            
            # Status
            if info['refined']:
                status = f"🔄 REFINED ({info['n_refinements']}x)"
            else:
                status = "📊 ORIGINAL"
            
            print(f"\n[{i}] {status} {path.name}")
            print(f"    📁 {parent_dir}/")
            print(f"    🧪 {info['symbols']} | {info['method']}/{info['basis']}")
            print(f"    📊 {info['n_frames']} configs | {info['n_atoms']} atoms", end="")
            print(f" | Forces: {'Yes ✅' if info['has_forces'] else 'No ❌'}")
            print(f"    ⚡ Energy range: {info['energy_range']:.2f} kcal/mol | Std: {info['energy_std']:.2f} kcal/mol")
            
            if info['refined'] and info['refinement_info']:
                ref = info['refinement_info']
                print(f"    🔧 Last refinement: +{ref.get('n_added', 0)} points ({ref.get('threshold', 'N/A')})")
            
            if args.detailed:
                print(f"    📄 Full path: {info['path']}")
    
    # Recommendations
    print("\n" + "=" * 100)
    print("  RECOMMENDATIONS")
    print("=" * 100)
    
    if models_info and data_info:
        # Find best model (lowest RMSE, refined)
        valid_models = [m for m in models_info if 'error' not in m and m['rmse'] is not None]
        if valid_models:
            # Prefer refined models
            refined_models = [m for m in valid_models if m['refined']]
            if refined_models:
                best_model = min(refined_models, key=lambda x: x['rmse'])
            else:
                best_model = min(valid_models, key=lambda x: x['rmse'])
            
            # Find matching training data
            model_symbols = best_model['symbols']
            matching_data = [d for d in data_info if 'error' not in d and d['symbols'] == model_symbols]
            
            if matching_data:
                # Prefer refined data
                refined_data = [d for d in matching_data if d['refined']]
                if refined_data:
                    best_data = max(refined_data, key=lambda x: x['n_frames'])
                else:
                    best_data = max(matching_data, key=lambda x: x['n_frames'])
                
                print(f"\n💡 Recommended for production:")
                print(f"\n   Model: {Path(best_model['path']).name}")
                print(f"   RMSE: {best_model['rmse']:.3f} kcal/mol | Status: {'Refined ✅' if best_model['refined'] else 'Initial'}")
                print(f"\n   Training data: {Path(best_data['path']).name}")
                print(f"   Configs: {best_data['n_frames']} | Status: {'Refined ✅' if best_data['refined'] else 'Original'}")
                
                print(f"\n   🚀 Use with:")
                print(f"   python3 simple_production_md.py \\")
                print(f"       --model {best_model['path']} \\")
                print(f"       --training-data {best_data['path']} \\")
                print(f"       --temp 300 --steps 10000")
    
    print("\n" + "=" * 100)
    print(f"  Found {len(models_info)} models and {len(data_info)} training datasets")
    print("=" * 100)
    print()

if __name__ == '__main__':
    main()
