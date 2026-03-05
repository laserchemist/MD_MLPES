#!/usr/bin/env python3
"""
Metadata Migration Tool for Old Computations

This script updates old training data files to include proper metadata.
It can:
1. Scan for old files without metadata
2. Interactively fill in missing information
3. Update files with complete metadata
4. Register files in computation database

Usage:
    python3 migrate_metadata.py [--auto] [--scan-only]
"""

import sys
import argparse
import glob
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Try to import framework
try:
    from modules.data_formats import TrajectoryData, save_trajectory, load_trajectory
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    print("⚠️  Framework not available - limited functionality")

class MetadataMigrator:
    """Migrate old training data to include metadata."""
    
    def __init__(self):
        self.found_files = []
        self.files_with_metadata = []
        self.files_without_metadata = []
        self.files_with_errors = []
    
    def scan_directories(self):
        """Scan for training data files."""
        
        print("\n" + "=" * 80)
        print("  SCANNING FOR TRAINING DATA")
        print("=" * 80)
        
        patterns = [
            'outputs/*/training_data/*.npz',
            'outputs/*/*.npz',
            'outputs/*/training_data.npz'
        ]
        
        all_files = set()
        for pattern in patterns:
            all_files.update(glob.glob(pattern))
        
        all_files = sorted(all_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)
        
        print(f"\n📂 Found {len(all_files)} file(s)")
        
        for filepath in all_files:
            self.check_file(filepath)
        
        print(f"\n📊 Summary:")
        print(f"   ✅ With metadata: {len(self.files_with_metadata)}")
        print(f"   ⚠️  Without metadata: {len(self.files_without_metadata)}")
        print(f"   ❌ Errors: {len(self.files_with_errors)}")
        
        return self.files_without_metadata
    
    def check_file(self, filepath):
        """Check if file has proper metadata."""
        
        try:
            # Load file
            data = np.load(filepath, allow_pickle=True)
            
            # Check for metadata
            has_metadata = False
            metadata_valid = False
            
            if 'metadata' in data:
                metadata = data['metadata']
                
                # Check if it's a proper dict
                if metadata is not None:
                    if isinstance(metadata, np.ndarray):
                        metadata = metadata.item()
                    
                    if isinstance(metadata, dict):
                        # Check if it has key fields
                        if 'theory' in metadata and 'molecule' in metadata:
                            has_metadata = True
                            metadata_valid = True
            
            # Categorize
            if metadata_valid:
                self.files_with_metadata.append(filepath)
                print(f"   ✅ {filepath}")
            else:
                self.files_without_metadata.append(filepath)
                print(f"   ⚠️  {filepath} (no valid metadata)")
        
        except Exception as e:
            self.files_with_errors.append((filepath, str(e)))
            print(f"   ❌ {filepath} (error: {e})")
    
    def update_file_interactive(self, filepath):
        """Interactively update file metadata."""
        
        print("\n" + "=" * 80)
        print(f"  UPDATING: {filepath}")
        print("=" * 80)
        
        try:
            # Load existing data
            if FRAMEWORK_AVAILABLE:
                traj = load_trajectory(filepath)
            else:
                data = np.load(filepath, allow_pickle=True)
                
                class SimpleTraj:
                    def __init__(self, data):
                        self.symbols = data['symbols'].tolist() if 'symbols' in data else []
                        self.coordinates = data['coordinates'] if 'coordinates' in data else None
                        self.energies = data['energies'] if 'energies' in data else None
                        self.forces = data['forces'] if 'forces' in data else None
                        self.n_frames = len(self.energies) if self.energies is not None else 0
                
                traj = SimpleTraj(data)
            
            # Show current info
            print(f"\n📊 Current data:")
            print(f"   Configurations: {traj.n_frames}")
            print(f"   Atoms: {len(traj.symbols)}")
            print(f"   Symbols: {' '.join(traj.symbols)}")
            if traj.energies is not None:
                e_range = (traj.energies.max() - traj.energies.min()) * 627.509
                print(f"   Energy range: {e_range:.2f} kcal/mol")
            print(f"   Has forces: {traj.forces is not None}")
            
            # Collect metadata
            print(f"\n📝 Enter metadata (press Enter to skip):\n")
            
            # Theory level
            print("Theory Level:")
            method = input("   Method (e.g., B3LYP, HF, MP2) [B3LYP]: ").strip() or "B3LYP"
            basis = input("   Basis set (e.g., 6-31G*, cc-pVDZ) [6-31G*]: ").strip() or "6-31G*"
            
            # Molecule
            print("\nMolecule:")
            mol_name = input("   Name (e.g., water, formaldehyde) [unknown]: ").strip() or "unknown"
            
            # Generate formula from symbols
            from collections import Counter
            counts = Counter(traj.symbols)
            formula = ''.join([f"{s}{c if c > 1 else ''}" for s, c in sorted(counts.items())])
            print(f"   Formula (detected): {formula}")
            
            # Training info
            print("\nTraining:")
            n_traj = input("   Number of trajectories [unknown]: ").strip()
            temps = input("   Temperatures (comma-separated) []: ").strip()
            
            # Create metadata
            metadata = {
                'theory': {
                    'method': method,
                    'basis': basis,
                    'reference': 'RHF'
                },
                'molecule': {
                    'name': mol_name,
                    'formula': formula,
                    'n_atoms': len(traj.symbols),
                    'symbols': traj.symbols
                },
                'training': {
                    'n_configs': traj.n_frames,
                    'n_trajectories': int(n_traj) if n_traj else 0,
                    'temperatures': [int(t.strip()) for t in temps.split(',')] if temps else [],
                    'training_type': 'energy_forces' if traj.forces is not None else 'energy_only',
                    'date_created': datetime.fromtimestamp(Path(filepath).stat().st_mtime).isoformat()
                },
                'files': {
                    'training_data': filepath
                },
                'version': '2.2_migrated'
            }
            
            # Show summary
            print(f"\n📋 Metadata summary:")
            print(json.dumps(metadata, indent=2))
            
            confirm = input("\n   Save this metadata? [y/n]: ").strip().lower()
            
            if confirm == 'y':
                # Save updated file
                if FRAMEWORK_AVAILABLE:
                    traj_with_meta = TrajectoryData(
                        symbols=traj.symbols,
                        coordinates=traj.coordinates,
                        energies=traj.energies,
                        forces=traj.forces,
                        metadata=metadata
                    )
                    
                    # Backup original
                    backup_path = filepath + '.backup'
                    import shutil
                    shutil.copy(filepath, backup_path)
                    print(f"   ✅ Backup: {backup_path}")
                    
                    # Save with metadata
                    save_trajectory(traj_with_meta, filepath)
                    print(f"   ✅ Updated: {filepath}")
                else:
                    print("   ⚠️  Framework not available - cannot save")
                    print("   Metadata would be:")
                    print(json.dumps(metadata, indent=2))
                
                return metadata
            else:
                print("   ❌ Skipped")
                return None
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def update_file_auto(self, filepath, default_method='B3LYP', default_basis='6-31G*'):
        """Automatically update file with default metadata."""
        
        try:
            if not FRAMEWORK_AVAILABLE:
                print(f"   ⚠️  Framework required for auto-update")
                return None
            
            # Load
            traj = load_trajectory(filepath)
            
            # Generate formula
            from collections import Counter
            counts = Counter(traj.symbols)
            formula = ''.join([f"{s}{c if c > 1 else ''}" for s, c in sorted(counts.items())])
            
            # Create metadata
            metadata = {
                'theory': {
                    'method': default_method,
                    'basis': default_basis,
                    'reference': 'RHF'
                },
                'molecule': {
                    'name': 'unknown',
                    'formula': formula,
                    'n_atoms': len(traj.symbols),
                    'symbols': traj.symbols
                },
                'training': {
                    'n_configs': traj.n_frames,
                    'training_type': 'energy_forces' if traj.forces is not None else 'energy_only',
                    'date_created': datetime.fromtimestamp(Path(filepath).stat().st_mtime).isoformat()
                },
                'files': {
                    'training_data': filepath
                },
                'version': '2.2_auto_migrated'
            }
            
            # Save
            traj_with_meta = TrajectoryData(
                symbols=traj.symbols,
                coordinates=traj.coordinates,
                energies=traj.energies,
                forces=traj.forces,
                metadata=metadata
            )
            
            # Backup
            backup_path = filepath + '.backup'
            import shutil
            shutil.copy(filepath, backup_path)
            
            # Save
            save_trajectory(traj_with_meta, filepath)
            
            print(f"   ✅ Auto-updated: {filepath}")
            return metadata
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

def main():
    """Main migration workflow."""
    
    parser = argparse.ArgumentParser(description='Migrate old training data to include metadata')
    parser.add_argument('--auto', action='store_true', help='Automatically update with defaults')
    parser.add_argument('--scan-only', action='store_true', help='Only scan, don\'t update')
    parser.add_argument('--method', default='B3LYP', help='Default method for auto-update')
    parser.add_argument('--basis', default='6-31G*', help='Default basis for auto-update')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  METADATA MIGRATION TOOL")
    print("=" * 80)
    
    migrator = MetadataMigrator()
    
    # Scan
    files_without_metadata = migrator.scan_directories()
    
    if args.scan_only:
        print("\n✅ Scan complete (no updates)")
        return
    
    if not files_without_metadata:
        print("\n✅ All files have metadata!")
        return
    
    print(f"\n📝 Found {len(files_without_metadata)} file(s) without metadata")
    
    if args.auto:
        # Auto-update all
        print("\n🤖 Auto-updating with defaults...")
        print(f"   Method: {args.method}")
        print(f"   Basis: {args.basis}")
        
        for filepath in files_without_metadata:
            migrator.update_file_auto(filepath, args.method, args.basis)
        
        print("\n✅ Auto-update complete!")
    
    else:
        # Interactive update
        print("\n🔧 Interactive update mode")
        
        for i, filepath in enumerate(files_without_metadata, 1):
            print(f"\n[{i}/{len(files_without_metadata)}]")
            
            choice = input(f"Update {filepath}? [y/n/q]: ").strip().lower()
            
            if choice == 'q':
                print("Quitting...")
                break
            elif choice == 'y':
                migrator.update_file_interactive(filepath)
            else:
                print("Skipped")
        
        print("\n✅ Interactive update complete!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
