#!/usr/bin/env python3
"""
Computation Database for ML-PES Framework

This module provides a simple database system to track computations
instead of scanning directories every time.

Features:
- Register new computations
- Query existing computations
- Update computation metadata
- Fast lookup without directory scanning

Database format: JSON file (computations_db.json)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class ComputationDatabase:
    """Simple JSON-based database for tracking computations."""
    
    def __init__(self, db_path: str = 'computations_db.json'):
        self.db_path = Path(db_path)
        self.computations = {}
        
        # Load existing database
        if self.db_path.exists():
            self.load()
        else:
            print(f"📂 Creating new database: {self.db_path}")
    
    def load(self):
        """Load database from JSON file."""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.computations = data.get('computations', {})
            print(f"✅ Loaded {len(self.computations)} computation(s) from database")
        except Exception as e:
            print(f"⚠️  Error loading database: {e}")
            self.computations = {}
    
    def save(self):
        """Save database to JSON file."""
        try:
            data = {
                'version': '2.2',
                'last_updated': datetime.now().isoformat(),
                'computations': self.computations
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Saved database: {self.db_path}")
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def register(self, 
                 filepath: str,
                 method: str,
                 basis: str,
                 molecule_name: str,
                 molecule_formula: str,
                 n_configs: int,
                 n_atoms: int,
                 symbols: List[str],
                 has_forces: bool = False,
                 training_type: str = 'energy_only',
                 n_trajectories: int = 0,
                 temperatures: List[int] = None,
                 energy_range_kcal: float = 0.0,
                 notes: str = '') -> str:
        """
        Register a new computation in the database.
        
        Returns: computation_id
        """
        
        # Generate unique ID
        comp_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create entry
        entry = {
            'id': comp_id,
            'filepath': str(filepath),
            'theory': {
                'method': method,
                'basis': basis
            },
            'molecule': {
                'name': molecule_name,
                'formula': molecule_formula,
                'n_atoms': n_atoms,
                'symbols': symbols
            },
            'training': {
                'n_configs': n_configs,
                'n_trajectories': n_trajectories,
                'temperatures': temperatures or [],
                'training_type': training_type,
                'has_forces': has_forces,
                'energy_range_kcal': energy_range_kcal
            },
            'metadata': {
                'date_created': datetime.now().isoformat(),
                'date_modified': datetime.now().isoformat(),
                'notes': notes,
                'valid': True
            }
        }
        
        # Add to database
        self.computations[comp_id] = entry
        
        # Save
        self.save()
        
        print(f"✅ Registered: {comp_id}")
        
        return comp_id
    
    def get(self, comp_id: str) -> Optional[Dict]:
        """Get computation by ID."""
        return self.computations.get(comp_id)
    
    def get_by_filepath(self, filepath: str) -> Optional[Dict]:
        """Get computation by filepath."""
        for comp_id, entry in self.computations.items():
            if entry['filepath'] == str(filepath):
                return entry
        return None
    
    def query(self,
              method: Optional[str] = None,
              basis: Optional[str] = None,
              molecule: Optional[str] = None,
              min_configs: int = 0,
              has_forces: Optional[bool] = None,
              valid_only: bool = True) -> List[Dict]:
        """
        Query computations with filters.
        
        Returns: List of matching computations
        """
        
        results = []
        
        for comp_id, entry in self.computations.items():
            # Skip invalid entries
            if valid_only and not entry.get('metadata', {}).get('valid', True):
                continue
            
            # Apply filters
            if method and entry['theory']['method'].lower() != method.lower():
                continue
            
            if basis and entry['theory']['basis'].lower() != basis.lower():
                continue
            
            if molecule and entry['molecule']['name'].lower() != molecule.lower():
                continue
            
            if entry['training']['n_configs'] < min_configs:
                continue
            
            if has_forces is not None and entry['training']['has_forces'] != has_forces:
                continue
            
            results.append(entry)
        
        # Sort by date (newest first)
        results.sort(key=lambda x: x['metadata']['date_modified'], reverse=True)
        
        return results
    
    def update(self, comp_id: str, updates: Dict):
        """Update computation entry."""
        if comp_id in self.computations:
            # Update fields
            for key, value in updates.items():
                if key in self.computations[comp_id]:
                    self.computations[comp_id][key].update(value)
            
            # Update timestamp
            self.computations[comp_id]['metadata']['date_modified'] = datetime.now().isoformat()
            
            # Save
            self.save()
            
            print(f"✅ Updated: {comp_id}")
        else:
            print(f"❌ Computation not found: {comp_id}")
    
    def mark_invalid(self, comp_id: str, reason: str = ''):
        """Mark computation as invalid (e.g., file deleted)."""
        if comp_id in self.computations:
            self.computations[comp_id]['metadata']['valid'] = False
            self.computations[comp_id]['metadata']['invalid_reason'] = reason
            self.save()
            print(f"⚠️  Marked invalid: {comp_id} ({reason})")
    
    def verify_files(self):
        """Verify that all registered files still exist."""
        print("\n🔍 Verifying files...")
        
        invalid_count = 0
        
        for comp_id, entry in self.computations.items():
            filepath = Path(entry['filepath'])
            
            if not filepath.exists():
                print(f"   ❌ Missing: {filepath}")
                self.mark_invalid(comp_id, 'File not found')
                invalid_count += 1
        
        if invalid_count == 0:
            print("   ✅ All files verified")
        else:
            print(f"   ⚠️  {invalid_count} file(s) marked invalid")
    
    def list_all(self, valid_only: bool = True):
        """List all computations."""
        
        results = self.query(valid_only=valid_only)
        
        if not results:
            print("\n📂 No computations in database")
            return
        
        print(f"\n📂 Found {len(results)} computation(s):\n")
        
        for i, entry in enumerate(results, 1):
            comp_id = entry['id']
            method = entry['theory']['method']
            basis = entry['theory']['basis']
            mol_name = entry['molecule']['name']
            mol_formula = entry['molecule']['formula']
            n_configs = entry['training']['n_configs']
            e_range = entry['training'].get('energy_range_kcal', 0)
            date_mod = entry['metadata']['date_modified'][:10]
            
            print(f"  [{i}] {comp_id}")
            print(f"      {mol_name} ({mol_formula}) | {method}/{basis}")
            print(f"      Configs: {n_configs}, Energy range: {e_range:.2f} kcal/mol")
            print(f"      Modified: {date_mod}")
            print(f"      Path: {entry['filepath']}")
            print()
    
    def export_summary(self, output_file: str = 'computations_summary.txt'):
        """Export database summary to text file."""
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("  COMPUTATION DATABASE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total computations: {len(self.computations)}\n")
            f.write(f"Last updated: {datetime.now().isoformat()}\n\n")
            
            for comp_id, entry in self.computations.items():
                f.write(f"\n{comp_id}\n")
                f.write("-" * 40 + "\n")
                f.write(json.dumps(entry, indent=2))
                f.write("\n")
        
        print(f"✅ Exported summary: {output_file}")


def scan_and_register(db: ComputationDatabase, scan_dirs: List[str] = None, verbose: bool = True):
    """
    Scan directories and register any unregistered computations.
    Now handles files with missing/incomplete metadata.
    """
    
    if scan_dirs is None:
        scan_dirs = ['outputs']
    
    print("\n🔍 Scanning for unregistered computations...")
    
    import glob
    import numpy as np
    from collections import Counter
    
    patterns = [
        'outputs/*/training_data/*.npz',
        'outputs/*/*.npz',
        'outputs/*/training_data.npz'
    ]
    
    all_files = set()
    for pattern in patterns:
        all_files.update(glob.glob(pattern))
    
    print(f"   Found {len(all_files)} file(s)\n")
    
    new_count = 0
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for filepath in sorted(all_files):
        try:
            # Check if already registered
            existing = db.get_by_filepath(filepath)
            if existing and not verbose:
                continue
            
            if verbose:
                print(f"📂 Processing: {filepath}")
            
            # Try to load
            data = np.load(filepath, allow_pickle=True)
            
            # Extract basic info
            symbols = data['symbols'].tolist() if 'symbols' in data else []
            energies = data['energies'] if 'energies' in data else None
            forces = data['forces'] if 'forces' in data else None
            coords = data['coordinates'] if 'coordinates' in data else None
            
            if energies is None or len(symbols) == 0:
                if verbose:
                    print(f"   ⚠️  Incomplete data (no energies or symbols)\n")
                skipped_count += 1
                continue
            
            # Get metadata if available
            metadata = {}
            if 'metadata' in data and data['metadata'] is not None:
                try:
                    meta = data['metadata']
                    if isinstance(meta, np.ndarray):
                        meta = meta.item()
                    if isinstance(meta, dict):
                        metadata = meta
                        if verbose:
                            print(f"   ✅ Found metadata in file")
                    else:
                        if verbose:
                            print(f"   ⚠️  Metadata exists but wrong type: {type(meta)}")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Error reading metadata: {e}")
            else:
                if verbose:
                    print(f"   ⚠️  No metadata in file")
            
            # Extract or infer details
            method = 'unknown'
            basis = 'unknown'
            mol_name = 'unknown'
            mol_formula = '?'
            n_trajectories = 0
            temperatures = []
            
            # Try to get from metadata
            if metadata:
                if 'theory' in metadata:
                    method = metadata['theory'].get('method', 'unknown')
                    basis = metadata['theory'].get('basis', 'unknown')
                    if verbose and method != 'unknown':
                        print(f"   📋 Theory from metadata: {method}/{basis}")
                
                if 'molecule' in metadata:
                    mol_name = metadata['molecule'].get('name', 'unknown')
                    mol_formula = metadata['molecule'].get('formula', '?')
                    if verbose and mol_name != 'unknown':
                        print(f"   📋 Molecule from metadata: {mol_name}")
                
                if 'training' in metadata:
                    n_trajectories = metadata['training'].get('n_trajectories', 0)
                    temperatures = metadata['training'].get('temperatures', [])
            
            # If formula still unknown, compute from symbols
            if mol_formula == '?':
                counts = Counter(symbols)
                mol_formula = ''.join([f"{s}{c if c > 1 else ''}" for s, c in sorted(counts.items())])
            
            # Try to infer molecule name from formula
            if mol_name == 'unknown':
                formula_to_name = {
                    'H2O': 'water',
                    'CH4': 'methane',
                    'NH3': 'ammonia',
                    'CH2O': 'formaldehyde',
                    'C2H4': 'ethylene',
                    'H2O2': 'hydrogen_peroxide',
                    'CH4O': 'methanol',
                    'CH2O2': 'formaldehyde_oxide',
                    'CH2OO': 'formaldehyde_oxide',
                    'C6H6': 'benzene'
                }
                mol_name = formula_to_name.get(mol_formula, 'unknown')
                if mol_name != 'unknown' and verbose:
                    print(f"   💡 Inferred molecule name: {mol_name} from formula {mol_formula}")
            
            # Calculate stats
            n_configs = len(energies)
            n_atoms = len(symbols)
            has_forces = forces is not None
            e_range = (energies.max() - energies.min()) * 627.509
            e_mean = energies.mean() * 627.509
            e_std = energies.std() * 627.509
            
            if verbose:
                print(f"   📊 Details:")
                print(f"      Theory: {method}/{basis}")
                print(f"      Molecule: {mol_name} ({mol_formula})")
                print(f"      Configs: {n_configs}, Atoms: {n_atoms}")
                print(f"      Energy range: {e_range:.2f} kcal/mol")
                print(f"      Has forces: {has_forces}")
            
            # Register or update
            if existing:
                # Update existing entry
                updates = {
                    'theory': {'method': method, 'basis': basis},
                    'molecule': {
                        'name': mol_name,
                        'formula': mol_formula,
                        'n_atoms': n_atoms,
                        'symbols': symbols
                    },
                    'training': {
                        'n_configs': n_configs,
                        'n_trajectories': n_trajectories,
                        'temperatures': temperatures,
                        'has_forces': has_forces,
                        'energy_range_kcal': float(e_range),
                        'training_type': 'energy_forces' if has_forces else 'energy_only'
                    }
                }
                db.update(existing['id'], updates)
                updated_count += 1
                if verbose:
                    print(f"   ✅ Updated existing entry\n")
            else:
                # Register new
                db.register(
                    filepath=filepath,
                    method=method,
                    basis=basis,
                    molecule_name=mol_name,
                    molecule_formula=mol_formula,
                    n_configs=n_configs,
                    n_atoms=n_atoms,
                    symbols=symbols,
                    has_forces=has_forces,
                    energy_range_kcal=float(e_range),
                    training_type='energy_forces' if has_forces else 'energy_only',
                    n_trajectories=n_trajectories,
                    temperatures=temperatures
                )
                new_count += 1
                if verbose:
                    print(f"   ✅ Registered new entry\n")
        
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"   ❌ Error: {e}\n")
            else:
                print(f"   ❌ Error: {filepath} ({e})")
    
    print(f"\n{'='*60}")
    print(f"📊 SCAN SUMMARY:")
    print(f"   ✅ Newly registered: {new_count}")
    print(f"   🔄 Updated: {updated_count}")
    print(f"   ⏭️  Skipped (incomplete): {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📊 Total in database: {len(db.computations)}")
    print(f"{'='*60}\n")


def main():
    """CLI for database management."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Manage computation database',
        epilog="""
Examples:
  python3 computation_database.py scan          # Scan and register all files
  python3 computation_database.py scan -v       # Verbose scan with details
  python3 computation_database.py list          # List all registered computations
  python3 computation_database.py verify        # Check if files still exist
  python3 computation_database.py export        # Export summary to text file
        """
    )
    parser.add_argument('command', choices=['list', 'scan', 'verify', 'export', 'stats'],
                       help='Command to execute')
    parser.add_argument('--db', default='computations_db.json',
                       help='Database file path')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output (for scan command)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  COMPUTATION DATABASE")
    print("=" * 80)
    
    db = ComputationDatabase(args.db)
    
    if args.command == 'list':
        db.list_all()
    
    elif args.command == 'scan':
        scan_and_register(db, verbose=args.verbose)
    
    elif args.command == 'verify':
        db.verify_files()
    
    elif args.command == 'export':
        db.export_summary()
    
    elif args.command == 'stats':
        # Show statistics
        print(f"\n📊 Database Statistics:\n")
        
        valid_comps = [c for c in db.computations.values() if c.get('metadata', {}).get('valid', True)]
        
        print(f"   Total entries: {len(db.computations)}")
        print(f"   Valid entries: {len(valid_comps)}")
        print(f"   Invalid entries: {len(db.computations) - len(valid_comps)}")
        
        if valid_comps:
            # Count by molecule
            molecules = {}
            for comp in valid_comps:
                mol = comp['molecule']['name']
                molecules[mol] = molecules.get(mol, 0) + 1
            
            print(f"\n   By molecule:")
            for mol, count in sorted(molecules.items(), key=lambda x: x[1], reverse=True):
                print(f"      {mol}: {count}")
            
            # Count by theory
            theories = {}
            for comp in valid_comps:
                theory = f"{comp['theory']['method']}/{comp['theory']['basis']}"
                theories[theory] = theories.get(theory, 0) + 1
            
            print(f"\n   By theory level:")
            for theory, count in sorted(theories.items(), key=lambda x: x[1], reverse=True):
                print(f"      {theory}: {count}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
